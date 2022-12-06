import os
import glob
import numbers
import mxnet as mx
import numpy as np
import queue as Queue
import threading
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
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
            local_rank=0, 
            transform=Transforms.train
        )
    else:
        raise Exception(f"dosen exist {rec} file and {idx} file")
    train_loader = CustomDataLoader(
        local_rank=args.local_rank,
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=True
    )

    valid_class_num = 5749
    valid_dir = os.path.join(root_dir,"lfw")
    valid_dataset = ImageFolder(valid_dir, transform=Transforms.test)
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers
    )
    print(f"train dataset length:  {len(train_dataset):,}")
    print(f"valid dataset length:  {len(valid_dataset):,}")
    return train_loader, train_class_num, valid_loader, valid_class_num




# class CustomDataset(Dataset):
#     def __init__(self, df, transform=None):
#         self.df = df
#         self.transform = transform
    
#     def __len__(self):
#         return self.df.shape[0]

#     def __getitem__(self, index):
#         img_path = self.df["image_path"].iloc[index]
#         if self.transform:
#             image = self.transform(Image.open(img_path))
#         else:
#             image = io.imread(img_path) # numpy array로 읽어옴
#         label = self.df["label"].iloc[index]
#         return [image, label]


# class TestDataset(Dataset):
#     def __init__(self, transform=None):
#         self.transform = transform
#         self.dir = "/home/ljj0512/private/workspace/CP_urban-datathon_CT/test"
    
#     def __len__(self):
#         return len(os.listdir(self.dir))

#     def __getitem__(self, index):
#         img_path = os.listdir(self.dir)[index]
#         image = self.transform(Image.open(os.path.join(self.dir,img_path)))
#         return image



# def create_dataloader(args):
#     train_df = pd.read_csv("train_df.csv")
#     val_df = pd.read_csv("val_df.csv")
#     train_dataset = CustomDataset(train_df, transform=get_transform.train)
#     val_set = CustomDataset(val_df, transform=get_transform.val)
#     test_set = TestDataset(transform=get_transform.test)
#     # train_dataset = ImageFolder(root = DATA_DIR+"/Train",
#     #                     transform = get_transform("train"))

#     # val_set = ImageFolder(root = DATA_DIR+"/Validation",
#     #                         transform = get_transform("valid"))
#     train_loader = DataLoader(dataset=train_dataset,
#                                 batch_size=args.batch_size,
#                                 shuffle=True,
#                                 num_workers=4)
#     val_loader = DataLoader(dataset=val_set,
#                             batch_size=args.batch_size,
#                             shuffle=False,
#                             num_workers=4)
#     test_loader = DataLoader(dataset=test_set,
#                             batch_size=args.batch_size,
#                             shuffle=False,
#                             num_workers=4)
    
#     return train_loader, val_loader,  test_loader, 5


# def get_transform(param):
# class get_transform:
#     train = transforms.Compose([
#                 transforms.RandomHorizontalFlip(),
#                 transforms.RandomVerticalFlip(),
#                 transforms.RandomAdjustSharpness(sharpness_factor=2),
#                 transforms.ToTensor(),
#             ])
#     val = transforms.Compose([
#                 transforms.ToTensor(),
#             ])
#     test = transforms.Compose([
#                 transforms.ToTensor(),
#             ])
    
