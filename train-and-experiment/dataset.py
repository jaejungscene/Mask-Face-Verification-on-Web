import os
import glob
import numbers
import mxnet as mx
import numpy as np
import queue as Queue
import threading
import torch
import random
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings("ignore")

class Transforms:
    # normalize = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
    normalize = [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]
    train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*normalize),
    ])
    test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*normalize)
    ])



def cutout_mask(inputs, p=0.5):
    if p < random.random():
        return inputs
    size = inputs.size(-1)
    y1 = 60
    y2 = size
    inputs[:,:,y1:y2,:] = 0.
    return inputs


class MXFaceDataset(Dataset):
    def __init__(self, path_imgrec, path_imgidx, local_rank, transform):
        super(MXFaceDataset, self).__init__()
        self.transform = transform
        self.local_rank = local_rank
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy()
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

    def __len__(self):
        return len(self.imgidx)



class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, local_rank, max_prefetch=6):
        super(BackgroundGenerator, self).__init__()
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.start()

    def run(self):
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self



class CustomDataLoader(DataLoader):

    def __init__(self, local_rank, **kwargs):
        super(CustomDataLoader, self).__init__(**kwargs)
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super(CustomDataLoader, self).__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                self.batch[k] = self.batch[k].to(device=self.local_rank, non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch



def get_dataloader(
        args,
        root_dir="/home/ljj0512/private/workspace/CV-project/Computer-Vision-Project/train/data/",
    ):
    """
        local_rank: GPU number
    """
    train_class_num = 85742
    train_dir = "ms1mv2"
    rec = os.path.join(root_dir, train_dir, "train.rec")
    idx = os.path.join(root_dir, train_dir, "train.idx")    
    if os.path.exists(rec) and os.path.exists(idx):
        train_dataset = MXFaceDataset(
            path_imgrec=rec, 
            path_imgidx=idx,
            local_rank=args.local_rank,
            transform=Transforms.train
        )
    else:
        raise Exception(f"dosen exist {rec} file and {idx} file")
    # train_loader = CustomDataLoader(
    #     local_rank=args.local_rank,
    #     dataset=train_dataset,
    #     batch_size=args.batch_size,
    #     num_workers=args.workers,
    #     shuffle=True
    # )

    skf = KFold(n_splits=200)
    for train_idx, val_idx in skf.split(train_dataset):
        train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        valid_sampler = torch.utils.data.SubsetRandomSampler(val_idx)
        train_loader = CustomDataLoader(
            sampler=train_sampler,
            local_rank=args.local_rank,
            dataset=train_dataset,
            batch_size=args.batch_size,
            num_workers=args.workers,
        )
        valid_loader = CustomDataLoader(
            sampler=valid_sampler,
            local_rank=args.local_rank,
            dataset=train_dataset,
            batch_size=args.batch_size,
            num_workers=args.workers,
        )
        break
    # valid_class_num = 5749
    # valid_dir = os.path.join(root_dir,"lfw")
    # valid_dataset = ImageFolder(valid_dir, transform=Transforms.test)
    # valid_loader = DataLoader(
    #     dataset=valid_dataset,
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     num_workers=args.workers
    # )
    print(f'train dataset len: {train_idx.shape[0]:,}')
    print(f'train dataloader len: {len(train_loader):,}')
    print(f'valid dataset len: {val_idx.shape[0]:,}')
    print(f'valid dataloader len: {len(valid_loader):,}')
    # print(f"train dataset length:  {len(train_dataset):,}")
    # print(f"valid dataset length:  {len(valid_dataset):,}")
    return train_loader, train_class_num, valid_loader

# def get_test__dataloader(
#         batch_size,
#         num_workers,
#         root_dir="/home/ljj0512/private/workspace/CV-project/Computer-Vision-Project/train/data/",
#     ):