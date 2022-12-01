import os
import time
import torch
import argparse
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import cv2
import wandb
from PIL import Image
from torch import nn
from torchvision import transforms
from model import get_model
from util import setup_seed, printSave_end_state, printSave_one_epoch, printSave_start_condition, save_checkpoint
import dataset
import warnings
warnings.filterwarnings("ignore")

base_dir = "/home/ljj0512/private/workspace/CV-project/Computer-Vision-Project/"
result_folder_name = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def get_args_parser():
    parser = argparse.ArgumentParser(description='training CIFAR-10, CIFAR-100 for self-directed research')
    parser.add_argument("--fold", default = 5, type = int)
    parser.add_argument('--model', default='resnet', type=str, help='networktype: resnet')
    parser.add_argument('--batch_size', default=128, type=int, metavar='N', help='batch size (default: 256)')
    parser.add_argument('--img_size', default=112, type=int, metavar='N', help='input image size')
    parser.add_argument('--lr', default=0.1, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--epoch', default=100, type=int, metavar='N', help='number of total epochs to run')                    
    parser.add_argument('--cuda', type=str, default='0', help='select used GPU')
    parser.add_argument('--wandb', type=int, default=1, help='choose activating wandb')
    parser.add_argument('--seed', type=str, default=41, help='set seed')
    parser.add_argument('--expname', default=result_folder_name, type=str, help='name of experiment')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='W', help='number of data loading workers (default: 4)')
    parser.add_argument('--device', default="cuda:0", type=str)
    parser.add_argument('--cude', default="0", type=str)

    optimizer = parser.add_argument_group('optimizer')
    optimizer.add_argument('--optimizer', type=str, default='adamw', help='optimizer name')
    optimizer.add_argument('--momentum', type=float, default=0.9, help='optimizer momentum')
    optimizer.add_argument('--weight-decay', type=float, default=1e-3, help='optimizer weight decay')
    optimizer.add_argument('--nesterov', action='store_true', default=False, help='use nesterov momentum')
    optimizer.add_argument('--betas', type=float, nargs=2, default=[0.9, 0.999], help='adam optimizer beta parameter')
    optimizer.add_argument('--eps', type=float, default=1e-6, help='optimizer eps')
    optimizer.add_argument('--decay-rate', type=float, default=0.1, help='lr decay rate')

    scheduler = parser.add_argument_group('scheduler')
    scheduler.add_argument('--cosine-freq', type=int, default=5, help='cosine scheduler frequency')
    scheduler.add_argument('--restart-epoch', type=int, default=20, help='warmup restart epoch period')
    scheduler.add_argument('--scheduler', type=str, default='cosine', help='lr scheduler')
    scheduler.add_argument('--three-phase', action='store_true', help='one cycle lr three phase')
    scheduler.add_argument('--step-size', type=int, default=2, help='lr decay step size')
    scheduler.add_argument('--min-lr', type=float, default=1e-6, help='lowest lr used for cosine scheduler')
    scheduler.add_argument('--milestones', type=int, nargs='+', default=[150, 225], help='multistep lr decay step')
    scheduler.add_argument('--warmup-epoch', type=int, default=5, help='warmup epoch')
    scheduler.add_argument('--warmup-scheduler', type=str, default='linear', help='warmup lr scheduler type')
    scheduler.add_argument('--warmup-lr', type=float, default=1e-4, help='warmup start lr')

    return parser


def main(args):
    
    start = time.time()
    best_err1, best_err5 = 0, 0

    train_loader, val_loader, numberofclass = dataset.create_dataloader(args)
    model = get_model(base_dir)
    printSave_start_condition(args, sum([p.data.nelement() for p in model.parameters()]))
    model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = create.create_criterion(args, numberofclass)
    optimizer, scheduler = get_optimizer_and_scheduler(model, args, len(train_loader))

    for epoch in range(0, args.epochs):
        train_loss = train_one_epoch(train_loader, model, criterion, optimizer, scheduler, epoch, args)
        err1, err5, val_loss = validate(val_loader, model, criterion, epoch, args)

        # remember best prec@1 and save checkpoint
        is_best = err1 <= best_err1 # if err1 <= best_err1, is_best is True
        if is_best:
            best_err1 = err1
            best_err5 = err5

        if args.wandb == True:
            wandb.log({'top-1 err': err1, 'top-5 err':err5, 'train loss':train_loss, 'validation loss':val_loss})
        print('Current best accuracy (top-1 and 5 error):\t', best_err1, 'and', best_err5)
        save_checkpoint({
            'epoch': epoch,
            'arch': args.net_type,
            'best_err1': best_err1,
            'best_err5': best_err5,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best, args)

    total_time = time.time()-start
    total_time = time.strftime('%H:%M:%S', time.localtime(total_time))
    printSave_end_state(args, best_err1, best_err5, total_time)





def train_one_epoch(train_loader, model, criterion, optimizer, scheduler, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    current_LR = get_learning_rate(optimizer)[0]
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        end = time.time()

        input = input.cuda()
        target = target.cuda()

        # compute output
        output = model(input)
        if args.distil > 0:
            loss = criterion(input, output, target)
        else:
            loss = criterion(output, target)

        # measure accuracy and record loss
        if args.distil > 0:
            err1, err5 = accuracy(output[0].data, target, topk=(1, 5))
        else:
            err1, err5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.verbose == True:
            print('Epoch: [{0}/{1}][{2}/{3}]\t'
                  'LR: {LR:.6f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-err {top1.val:.4f} ({top1.avg:.4f})\t'
                  'Top 5-err {top5.val:.4f} ({top5.avg:.4f})'.format(
                epoch, args.epochs, i, len(train_loader), LR=current_LR, batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
                
    printSave_one_epoch(epoch, args, batch_time, data_time, top1, top5, losses)
    # print('* Epoch[{0}/{1}]\t Top 1-err {top1.avg:.3f}\t  Top 5-err {top5.avg:.3f}\t Train Loss {loss.avg:.3f}'.format(
    #     epoch, args.epochs, top1=top1, top5=top5, loss=losses))

    return losses.avg




def validate(val_loader, model, criterion, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        data_time.update(time.time() - end)
        target = target.cuda()

        output = model(input)
        if args.distil > 0:
            loss = criterion(input, output, target, val=True)
            # if args.wandb == True:
                # wandb.log({"roc": wandb.plot.roc_curve(target, output[0])})
                # wandb.log({"pr": wandb.plots.precision_recall(target, output[0])})
        else:
            loss = criterion(output, target)
            # if args.wandb == True:
                # wandb.log({"roc": wandb.plot.roc_curve(target, output)})
                # wandb.log({"pr": wandb.plots.precision_recall(target, output)})

        # measure accuracy and record loss
        err1, err5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))

        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.verbose == True:
            print('Test (on val set): [{0}/{1}][{2}/{3}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-err {top1.val:.4f} ({top1.avg:.4f})\t'
                  'Top 5-err {top5.val:.4f} ({top5.avg:.4f})'.format(
                epoch, args.epochs, i, len(val_loader), batch_time=batch_time, loss=losses,
                data_time=data_time, top1=top1, top5=top5))

    printSave_one_epoch(epoch, args, batch_time, data_time, top1, top5, losses, False)
    return top1.avg, top5.avg, losses.avg



if __name__ == "__main__":
    args = get_args_parser().parse_args()
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    setup_seed(0)
    if args.wandb:
        wandb.init(project='comparsion', name=args.expname, entity='jaejungscene')
    main(args)




# vector_list = []
# for i in range(4):
#     print("-"*50)
#     np_img = cv2.imread(os.path.join(base_dir,f"images/{i}.jpg"))
#     np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
#     # np_img = np.array(img)
#     if np_img.shape[-1] > 3:
#         print("alpha channel remove")
#         np_img = np_img[:,:,0:3]

#     print(np_img.shape)
#     plt.imshow(np_img)
#     plt.show()

#     max_size = max(np_img.shape[0:-1])
#     min_size = min(np_img.shape[0:-1])
#     image_transforms = transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.CenterCrop((min_size,min_size)),
#             transforms.Resize(size=(112,112)), 
#             transforms.ToTensor(),
#             transforms.Normalize(
#                 [0.485, 0.456, 0.406], 
#                 [0.229, 0.224, 0.225]
#             )
#         ])

#     torch_img = image_transforms(np_img)
#     C, H, W = torch_img.size()
#     torch_img = torch_img.view(1,C,H,W)
#     print(torch_img.shape)
#     plt.imshow(torch_img.view(C,H,W).permute(1,2,0))
#     plt.show()
#     print("inputs shape:", torch_img.shape)
    
#     with torch.no_grad():
#         model.eval()
#         outputs = model(torch_img)
#     print("outputs shape:", outputs.shape)
#     vector_list.append(outputs)