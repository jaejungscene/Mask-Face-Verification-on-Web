import sys, os
from MS1Mset import ms1m_align_112_dataset as msm1Dataset, ImageTransform \
    as Transform
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader
from torch import nn
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from args import get_args_parser
# args = get_args_parser().parse_args()
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
import time
from dataset import get_dataloader
import numpy as np
import wandb
import random
from datetime import datetime
from log import save_checkpoint, printSave_start_condition
from utils import accuracy, AverageMeter
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from marginloss import CombinedMarginLoss
from fclayer import FCSoftmax
from optimizer import get_optimizer_and_scheduler
from sklearn import metrics
from model import get_model
import warnings
warnings.filterwarnings("ignore")

path_parser = '\\'
ROOT_DIR = "C:/Users/gmk_0/source/repos/pythonProject/IT2/Computer-Vision-Project"
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


#sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

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
    for i, (inputs, targets) in enumerate(val_loader):
        data_time.update(time.time() - end)
        true_labels += targets.tolist()
        inputs, targets = inputs.cuda(), targets.cuda()
        output = model(inputs)
        loss = criterion(output, targets)
        # measure accuracy and record loss
        # err1, err5 = accuracy(output.data, targets, topk=(1, 5))
        _, predicted = torch.max(output.data, dim=1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        pred_labels += predicted.cpu().tolist()
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

    # dataloader & num of class
    root = os.path.join(os.getcwd(), '..')
    root = os.path.join(ROOT_DIR, \
        'dataset'+path_parser+'ms1m_align_112'+path_parser+'imgs')
    label_list = os.listdir(root)

    file_list = []
    file_paths = []

    # MS1M 데이터셋 경로 얻기    
    for label in label_list:
        file_list = os.listdir(os.path.join(root,label))
        if len(file_list) == 0:
            print("file doesn't exist!")
            break
        for file in file_list:
            file_paths.append(os.path.join(label,file))
            
    print(len(file_paths))   

    # Train / Test Split
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    labels = [int(x.split(path_parser)[0]) for x in file_paths]

    labels = np.array(labels)
    print(labels)
    file_paths = np.array(file_paths)

    for train_idx, test_idx in sss.split(file_paths, labels):
        x_train, x_test = file_paths[train_idx], file_paths[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]


    # Train / Valid Split
    for train_idx, valid_idx in sss.split(x_train, y_train):
        x_train, x_valid = file_paths[train_idx], file_paths[valid_idx]
        y_train, y_valid = labels[train_idx], labels[valid_idx]


    print("first item: ", x_train[0], y_train[0])

    # Dataset 정의
    train_dataset = msm1Dataset(root=root, filelist=x_train, transform=Transform(resize=112, mean=0.5, std=0.2), phase='train')
    valid_dataset = msm1Dataset(root=root, filelist=x_valid, transform=Transform(resize=112, mean=0.5, std=0.2), phase='val')
    test_dataset = msm1Dataset(root=root, filelist=x_test, transform=Transform(resize=112, mean=0.5, std=0.2), phase='val')
    
    # Dataloader 정의
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
    
    train_class_num, val_class_num = 85742, 85742
    
    # iresnet model
    model = get_model(ROOT_DIR).to(args.device)
    # margin loss(arcface, cosface)
    margin_loss = CombinedMarginLoss(64, args.m1, args.m2, args.m3).to(args.device)
    # fclayer and margin softmax
    fc_softmax = FCSoftmax(margin_loss, 512, train_class_num).to(args.device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smooth).to(args.device)
    optimizer, scheduler = get_optimizer_and_scheduler(model, fc_softmax, args, len(train_loader))

    # if args.resume:
    #     checkpoint = torch.load(os.path.join(ROOT_DIR,"weights/baseline-arcface.pth"), map_location=torch.device("cpu"))
    #     model.load_state_dict(checkpoint)


    printSave_start_condition(args)
    for epoch in range(1, args.epoch+1):
        train_acc, train_loss = train_one_epoch(train_loader, model, fc_softmax, criterion, optimizer, scheduler, epoch, args)
        # val_acc, val_loss = validate(val_loader, model, criterion, epoch, args)

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
            # wandb.log({'validation accuracy':val_acc, 'validation loss':val_loss, 'train loss':train_loss, 'train accuracy':train_acc})
            wandb.log({'train loss':train_loss, 'train accuracy':train_acc})

    total_time = time.time()-start
    total_time = time.strftime('%H:%M:%S', time.localtime(total_time))
    print(f"finish training (total time: {total_time}")


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    if args.wandb == True:
        wandb.init(project='mask_face_verification', name=args.expname, entity='gmk0904')
    main(args)