import os
from args import get_args_parser
args = get_args_parser().parse_args()
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

import time
import dataset
import numpy as np
import wandb
import random
from datetime import datetime
from log import save_checkpoint, printSave_one_epoch, printSave_start_condition, printSave_end_state
from utils import accuracy, AverageMeter
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from optimizer import get_optimizer_and_scheduler
from sklearn import metrics
from model import get_model

import warnings
warnings.filterwarnings("ignore")

ROOT_DIR = "/home/ljj0512/private/workspace/CV-project/Computer-Vision-Project/train"
result_folder_name = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
best_acc = -1
best_f1 = -1


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
seed_everything(0) # Seed 고정



def train_one_epoch(train_loader, model, criterion, optimizer, scheduler, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()
    total = 0
    correct = 0
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        end = time.time()
        # load datat from cpu to gpu
        input, target = input.cuda(), target.cuda()
        # compute output
        embed_vec = model(input)
        
        loss = criterion(output, target)
        # measure accuracy and record loss
        _, predicted = torch.max(output.data, dim=1)
        total += target.size(0)
        correct += predicted.eq(target.data).cpu().sum()
        losses.update(loss.item(), input.size(0))
        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if args.verbose:
            acc = 100*correct/total
            print('Epoch: [{0}/{1}]\t'
                    'LR: {LR:.6f}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Accuracy {acc:.4f}({cor}/{total})\t'
                    .format(epoch, args.epoch, 
                    LR=scheduler.get_lr()[0], batch_time=batch_time, data_time=data_time,
                    loss=losses, acc=acc, cor=correct, total=total))
    acc = 100*correct/total
    print('Epoch: [{0}/{1}]\t'
            'LR: {LR:.6f}\t'
            'Time {batch_time.val:.3f} ({epoch_time.avg:.3f})\t'
            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            'Accuracy {acc:.4f}({cor}/{total})\t'
            .format(epoch, args.epoch, 
            LR=scheduler.get_lr()[0], epoch_time=batch_time, data_time=data_time,
            loss=losses, acc=acc, cor=correct, total=total))
    return losses.avg




def validate(val_loader, model, criterion, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()
    pred_labels = []
    true_labels = []
    total = 0
    correct = 0
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        data_time.update(time.time() - end)
        true_labels += target.tolist()
        input, target = input.cuda(), target.cuda()
        output = model(input)
        loss = criterion(output, target)
        # measure accuracy and record loss
        # err1, err5 = accuracy(output.data, target, topk=(1, 5))
        _, predicted = torch.max(output.data, dim=1)
        total += target.size(0)
        correct += predicted.eq(target.data).cpu().sum()
        pred_labels += predicted.cpu().tolist()
        losses.update(loss.item(), input.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    f1_score = metrics.f1_score(y_true=true_labels, y_pred=pred_labels, average='macro')
    acc = 100*correct/total
    print('Test (on val set): [{0}/{1}]\t'
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            'Accuracy {acc:.4f}({cor}/{total})\t'
            'F1 {F1:.4f}\t'
            .format(epoch, args.epoch, batch_time=batch_time, loss=losses,
            data_time=data_time, acc=acc, cor=correct, total=total, F1=f1_score))

    return f1_score, acc, losses.avg


# def inference(model, test_loader):
#     model.cuda()
#     model.eval()
#     preds = []
#     submit = pd.read_csv("./1001_sample_submission.csv")
#     with torch.no_grad():
#         for img in (test_loader):
#             img = img.cuda()
#             output = model(img)
#             _, predicted = torch.max(output.data, dim=1)
#             preds += predicted.cpu().tolist()
    
#     submit['result'] = preds
#     submit.to_csv('./1001_submission.csv', index=False)


def main():
    start = time.time()
    global args, best_f1, best_acc, ROOT_DIR

    # create model
    train_loader, val_loader, test_loader, numberofclass = dataset.create_dataloader(args)
    model = get_model(ROOT_DIR)
    printSave_start_condition(args)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smooth).cuda()
    optimizer, scheduler = get_optimizer_and_scheduler(model, args, len(train_loader))

    for epoch in range(0, args.epoch):
        # train for one epoch
        train_loss = train_one_epoch(train_loader, model, criterion, optimizer, scheduler, epoch, args)

        # evaluate on validation set
        f1, acc, val_loss = validate(val_loader, model, criterion, epoch, args)

        # save checkpoint
        if best_f1 < f1:
            best_acc = acc
            best_f1 = f1
            save_checkpoint({
                'epoch': epoch,
                'best_f1': best_f1,
                'best_acc': best_acc,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, args)
            
        if args.wandb == True:
            wandb.log({'valid f1 score':f1, 'acc':acc, 'train loss':train_loss, 'validation loss':val_loss})

        print(f'Current best score =>\tf1: {best_f1} , acc: {best_acc}')
        print("-"*100)

    total_time = time.time()-start
    total_time = time.strftime('%H:%M:%S', time.localtime(total_time))
    print(f"finish training (total time: {total_time}")
    # inference(model, test_loader)



if __name__ == '__main__':
    if args.wandb == True:
        wandb.init(project='CP_urban-datathon', name=args.expname, entity='jaejungscene')
    main()
