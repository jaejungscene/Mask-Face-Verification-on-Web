import os
from args import get_args_parser
# args = get_args_parser().parse_args()
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

import time
import numpy as np
import wandb
import random
from datetime import datetime
from log import save_checkpoint, printSave_start_condition
from utils import accuracy, AverageMeter
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from marginloss import CombinedMarginLoss
from fclayer import FCSoftmax
from optimizer import get_optimizer_and_scheduler
from dataset import get_dataloader, cutout_mask
from model import get_model
import warnings
warnings.filterwarnings("ignore")
ROOT_DIR = "/home/ljj0512/private/workspace/CV-project/Computer-Vision-Project/train-and-experiment"
result_folder_name = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
seed_everything(0)




def train_one_epoch(train_loader, model, fc_softmax, criterion, optimizer, scheduler, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()
    fc_softmax.train()
    total = 0
    correct = 0
    end = time.time()
    for i, (inputs, targets) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        end = time.time()
        # load datat from cpu to gpu
        inputs, targets = inputs.to(args.device), targets.cuda(args.device)
        if args.cutout_p > 0.0:
            inputs = cutout_mask(inputs, args.cutout_p)
        # compute output
        embed_vec = model(inputs)
        logits = fc_softmax(embed_vec, targets)
        loss = criterion(logits, targets)
        # measure accuracy and record loss
        _, predicted = torch.max(logits.data, dim=1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        losses.update(loss.item(), inputs.size(0))
        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if args.verbose and i%args.verbose_freq==0:
            acc = 100*correct/total
            print('Epoch[{0}({1})/{2}]\t'
                    'LR: {LR:.6f}\t'
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data: {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Accuracy: {acc:.4f}({cor}/{total})\t'
                    .format(epoch, i+1, args.epoch, 
                    LR=scheduler.get_lr()[0], batch_time=batch_time, data_time=data_time,
                    loss=losses, acc=acc, cor=correct, total=total))
    acc = 100*correct/total
    print('Epoch[{0}/{1}]\t'
            'LR: {LR:.6f}\t'
            'Time: {epoch_time.val:.3f} ({epoch_time.avg:.3f})\t'
            'Data: {data_time.val:.3f} ({data_time.avg:.3f})\t'
            'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
            'Accuracy: {acc:.4f}({cor}/{total})\t'
            .format(epoch, args.epoch, 
            LR=scheduler.get_lr()[0], epoch_time=batch_time, data_time=data_time,
            loss=losses, acc=acc, cor=correct, total=total))
    return acc, losses.avg




def validate(val_loader, model, fc_softmax, criterion, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.eval()
    fc_softmax.eval()
    total = 0
    correct = 0
    end = time.time()
    for i, (inputs, targets) in enumerate(val_loader):
        data_time.update(time.time() - end)
        inputs, targets = inputs.to(args.device), targets.to(args.device)
        embed_vec = model(inputs)
        outputs = fc_softmax(embed_vec, targets)
        loss = criterion(outputs, targets)
        # measure accuracy and record loss
        _, predicted = torch.max(outputs.data, dim=1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        losses.update(loss.item(), inputs.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    acc = 100*correct/total
    print('Test (on val set): [{0}/{1}]\t'
            'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Data: {data_time.val:.3f} ({data_time.avg:.3f})\t'
            'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
            'Accuracy: {acc:.4f}({cor}/{total})\t'
            .format(epoch, args.epoch, batch_time=batch_time, loss=losses,
            data_time=data_time, acc=acc, cor=correct, total=total))

    return acc, losses.avg


def main(args):
    global ROOT_DIR
    best_acc = -1
    start = time.time()

    train_loader, train_class_num, val_loader= get_dataloader(args)
    # iresnet model
    model = get_model(ROOT_DIR).to(args.device)
    # margin loss(arcface, cosface)
    margin_loss = CombinedMarginLoss(64, args.m1, args.m2, args.m3).to(args.device)
    # fclayer and margin softmax
    fc_softmax = FCSoftmax(margin_loss, 512, train_class_num).to(args.device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smooth).to(args.device)
    optimizer, scheduler = get_optimizer_and_scheduler(model, fc_softmax, args, len(train_loader))

    if args.resume:
        checkpoint = torch.load(os.path.join(ROOT_DIR,"weights/baseline-arcface.pth"))
        model.load_state_dict(checkpoint)
        checkpoint = torch.load(os.path.join(ROOT_DIR,"weights/baseline-finetune-fc_softmax.pth"))
        fc_softmax.load_state_dict(checkpoint)


    printSave_start_condition(args)
    for epoch in range(1, args.epoch+1):
        train_acc, train_loss = train_one_epoch(train_loader, model, fc_softmax, criterion, optimizer, scheduler, epoch, args)
        val_acc, val_loss = validate(val_loader, model, fc_softmax, criterion, epoch, args)

        # if best_acc < val_acc: <---------
        if best_acc < train_acc:
            # best_acc = val_acc <---------
            best_acc = train_acc
            save_checkpoint({
                'epoch': epoch,
                'best_acc': best_acc,
                'state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                'fc_state_dict': fc_softmax.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, args)
            print(f'Current best score =>\t: {best_acc}  |  Changed!!!!')
        else:
            print(f'Current best score =>\t: {best_acc}')
        print("-"*100)
            
        if args.wandb == True:
            wandb.log({'valid accuracy':val_acc, 'valid loss':val_loss, 'train loss':train_loss, 'train accuracy':train_acc})
            # wandb.log({'train loss':train_loss, 'train accuracy':train_acc})
    total_time = time.time()-start
    total_time = time.strftime('%H:%M:%S', time.localtime(total_time))
    print(f"finish training (total time): {total_time}")



if __name__ == '__main__':
    args = get_args_parser().parse_args()
    if args.wandb == True:
        wandb.init(project='mask_face_verification', name=args.expname, entity='jaejungscene')
    main(args)
