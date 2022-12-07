from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
import os
import numpy as np
from PIL import Image

path_parser = '/'  # macOS
path_parser = '\\' # Window


class ms1m_align_112_dataset(Dataset):
    def __init__(self, root, filelist, transform=None, phase='train'):
        self.root = root
        self.filelist = filelist
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        img_path = self.filelist[idx]
        print("img path " ,img_path)
        label = img_path.split(path_parser)[-2]
        img_path = os.path.join(self.root, img_path)
        img = Image.open(fp=img_path, mode='r')
        img = self.transform(img, self.phase)

        return img, label

class ImageTransform() :
    def __init__(self, resize, mean=0.5, std=0.2) :
        self.data_transform = {
            'train' : transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(p=0.5),
                #transforms.RandomRotation((-15, 15)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),

            ]),
            'val' : transforms.Compose([
                transforms.Resize(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
                
            ]),
            'show' : transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
            ])
        }
        
    def __call__(self, img, phase) :
        return self.data_transform[phase](img)
    

if __name__ == "__main__":
    root = os.path.join(os.getcwd(), \
        'dataset'+path_parser+'ms1m_align_112'+path_parser+'imgs')
    label_list = os.listdir(root)

    print("num of class: ", len(label_list))

    file_list = []
    file_paths = []
        
    for label in label_list:
        file_list = os.listdir(os.path.join(root,label))
        if len(file_list) == 0:
            print("file doesn't exist!")
            break
        for file in file_list:
            file_paths.append(os.path.join(label,file))
            
    print(len(file_paths))   

    dataset = ms1m_align_112_dataset(root=root, filelist=file_paths, transform=ImageTransform(resize=112), phase='show')
    print(dataset[0])
    
            
