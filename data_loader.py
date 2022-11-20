from torch.utils.data import Dataset
import os
from zipfile import ZipFile
from PIL import Image
import numpy as np
from torchvision import transforms



class masked_face_dataset(Dataset):
    def __init__(self, dir_path, transform=None, phase='train'):
        self.dir_path = dir_path
        self.transform = transform
        self.with_mask = []
        self.without_mask = []
        self.phase = phase

        if not os.path.isfile(self.dir_path+'.zip'):
            print("no dataset found")
        else:
            with ZipFile(self.dir_path + '.zip') as myZip:
                self.file_list = myZip.namelist()
                for path in self.file_list:
                    option, id = path.split('/')
                    if option == "with_mask":
                        self.with_mask.append(id)
                    else:
                        self.without_mask.append(id)

    def __getitem__(self, idx):
        x_label = self.with_mask[idx]
        y_label = self.without_mask[idx]

        with ZipFile(self.dir_path + '.zip') as myZip:
            with myZip.open('with_mask/'+x_label) as x_file, myZip.open('without_mask/'+y_label) as y_file:
                x_img = Image.open(x_file)
                x_img = np.array(x_img, dtype=np.float)
                y_img = Image.open(y_file)
                y_img = np.array(y_img, dtype=np.float)

                if self.transform is not None:
                    x_img = self.transform(x_img, self.phase)
                    y_img = self.transform(y_img, self.phase)

        return x_label, x_img, y_label, y_img

    def __len__(self):
        return len(self.file_list)

class ImageTransform():
    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomResizedCrop(resize, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize(mean, std)
            ]),
            'test': transforms.Compose([
                transforms.ToTensor(),
                transforms.CenterCrop(resize),
                transforms.Normalize(mean, std)
            ]),
            'show': transforms.Compose([
                transforms.ToTensor(),
                transforms.CenterCrop(resize),
            ])
        }

    def __call__(self, img, phase):
        return self.data_transform[phase](img)

