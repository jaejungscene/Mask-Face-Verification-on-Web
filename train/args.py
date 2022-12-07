import argparse
from datetime import datetime
result_folder_name = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def get_args_parser():
    parser = argparse.ArgumentParser(description='competition')
    parser.add_argument("--fold", default = 5, type = int)
    parser.add_argument('--model', default='resnet', type=str, help='networktype: resnet')
    parser.add_argument('--batch_size', default=128, type=int, metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', default=0.1, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--epoch', default=100, type=int, metavar='N', help='number of total epochs to run')                    
    parser.add_argument('--wandb', type=bool, default=False, help='choose activating wandb')
    parser.add_argument('--seed', type=str, default=0, help='set seed')
    parser.add_argument('--expname', default=result_folder_name, type=str, help='name of experiment')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='W', help='number of data loading workers (default: 4)')
    parser.add_argument('--device', default="cuda:0", type=str)
    parser.add_argument('--cuda', type=str, default='0', help='select used GPU')
    parser.add_argument('--local_rank', type=int, default=0, help='select used GPU')
    parser.add_argument('--verbose', type=bool, default=False)
    parser.add_argument('--verbose_freq', type=int, default=1000)
    parser.add_argument('--cutout_p', type=float, default=0.0)
    parser.add_argument('--resume', type=bool, default=False)

    loss = parser.add_argument_group('loss')
    loss.add_argument('--label_smooth', type=float, default=0.0, help='optimizer name')
    loss.add_argument('--m1', type=float, default=1.0, help='optimizer name')
    loss.add_argument('--m2', type=float, default=0.5, help='optimizer name')
    loss.add_argument('--m3', type=float, default=0.0, help='optimizer name')

    optimizer = parser.add_argument_group('optimizer')
    optimizer.add_argument('--optimizer', type=str, default='adamw', help='optimizer name')
    optimizer.add_argument('--momentum', type=float, default=0.9, help='optimizer momentum')
    optimizer.add_argument('--weight_decay', type=float, default=1e-3, help='optimizer weight decay')
    optimizer.add_argument('--nesterov', action='store_true', default=True, help='use nesterov momentum')
    optimizer.add_argument('--betas', type=float, nargs=2, default=[0.9, 0.999], help='adam optimizer beta parameter')
    optimizer.add_argument('--eps', type=float, default=1e-8, help='optimizer eps')
    optimizer.add_argument('--decay-rate', type=float, default=0.1, help='lr decay rate')

    scheduler = parser.add_argument_group('scheduler')
    scheduler.add_argument('--scheduler', type=str, default='cosinerestarts', help='lr scheduler')
    scheduler.add_argument('--cosine_freq', type=int, default=1, help='cosine scheduler frequency')
    scheduler.add_argument('--eta_max', type=float, default=1e-2, help='cosinerestarts max lr')
    scheduler.add_argument('--warmup_epoch', type=int, default=1, help='warmup epoch')

    scheduler.add_argument('--warmup_lr', type=float, default=1e-4, help='warmup start lr')
    scheduler.add_argument('--restart_epoch', type=int, default=20, help='warmup restart epoch period')
    scheduler.add_argument('--three_phase', action='store_true', help='one cycle lr three phase')
    scheduler.add_argument('--step_size', type=int, default=2, help='lr decay step size')
    scheduler.add_argument('--min_lr', type=float, default=1e-6, help='lowest lr used for cosine scheduler')
    scheduler.add_argument('--milestones', type=int, nargs='+', default=[150, 225], help='multistep lr decay step')
    scheduler.add_argument('--warmup_scheduler', type=str, default='linear', help='warmup lr scheduler type')


    return parser