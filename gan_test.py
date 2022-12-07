from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from zipfile import ZipFile
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.ml.image import ImageSchema
from pyspark.sql.functions import lit
from functools import reduce

config = {
    "base_dir": "./dataset/",
    "img_dir": "ms1m_align_112"
}

PATH = config["base_dir"]
IMG_PATH = config["img_dir"]
img_path = os.path.join(PATH, IMG_PATH)


class ms1mZip_dataset(Dataset):
    def __init__(self, dir_path, transform=None, phase='show'):
        self.dir_path = dir_path
        self.transform = transform
        self.phase = phase
        self.file_list = []
        self.labels = []

        if not os.path.isfile(self.dir_path+'.zip'):
            print("no dataset found")
        else:
            with ZipFile(self.dir_path + '.zip') as myZip:
                for path in myZip.namelist():
                    if path[-4:] != '.jpg':
                        continue

                    try:
                        if '/' in path:
                            parsed = path.split('/')
                        elif '\\' in path:
                            parsed = path.split('\\')
                        else:
                            continue

                        if len(parsed) == 3 and parsed[-1][-4:] == '.jpg':

                            # print("label: ", face_id, "name: ", name)
                            self.file_list.append(path)
                            if parsed[-2] not in self.labels:
                                self.labels.append(parsed[-2])

                    except ValueError:
                        print("not an image : ", path)
                        continue

    def __getitem__(self, idx):
        print("get item is accessed")
        _img_path = self.file_list[idx]
        with ZipFile(self.dir_path + '.zip') as archive:
            with archive.open(_img_path) as path:
                img = np.array(Image.open(path))
                img = self.transform(img, self.phase)
                if '/' in _img_path:
                    label = _img_path.split('/')[1]
                elif '\\' in _img_path:
                    label = _img_path.split('\\')[1]

                """if self.transform is not None:
                    x_img = self.transform(x_img, self.phase)
                    y_img = self.transform(y_img, self.phase)
                    print('transformed!')"""

        return img, label

    def __len__(self):
        return len(self.file_list)

    def list_file(self):
        return self.file_list

    def get_labels(self):
        return self.labels, len(self.labels)


class ImageTransform:
    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(resize, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'test': transforms.Compose([
                transforms.ToPILImage(),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'show': transforms.Compose([
                transforms.ToPILImage(),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
            ])
        }

    def __call__(self, img, phase):
        return self.data_transform[phase](img)


"""
def get_norm_param(dataset):
    # To normalize the dataset, calculate the mean and std
    meanRGB = [np.mean(x.numpy(), axis=(0, 1)) for x, _ in dataset]
    stdRGB = [np.std(x.numpy(), axis=(0, 1)) for x, _ in dataset]

    meanR = np.mean([m[0] for m in meanRGB])
    meanG = np.mean([m[1] for m in meanRGB])
    meanB = np.mean([m[2] for m in meanRGB])
    stdR = np.mean([s[0] for s in stdRGB])
    stdG = np.mean([s[1] for s in stdRGB])
    stdB = np.mean([s[2] for s in stdRGB])

    print("CIFAR10_MEAN:", meanR, meanG, meanB)  # Cifar10_stats -> mean
    print("CIFAR10_STD:", stdR, stdG, stdB)  # Cifar10_stats-> std

    return (meanR, meanG, meanB), (stdR, stdG, stdB)
"""
if __name__ == "__main__":
    ms1mDataset = ms1mZip_dataset(dir_path=img_path, transform=ImageTransform(112, 0.5, 0.3), phase="show")

    full_dl = DataLoader(ms1mDataset, batch_size=4, shuffle=False)
    print(ms1mDataset.list_file())
    labels, n_labels = ms1mDataset.labels()
    print(ms1mDataset.labels(), n_labels)
    #print(len(ms1mDataset)) # 5822653
    #print(ms1mDataset[0][0].shape) # (112, 112, 3)

    batch_iterator = iter(full_dl)
    inputs, labels = next(batch_iterator)

    print(inputs.size())
    print(labels)

    print('===== print images =====')
    figure = plt.figure(figsize=(16, 64))
    cols, rows= 2, 2
    for i, img, label in enumerate(zip(inputs, labels)):
        np_img = img.permute(1, 2, 0)
        figure.add_subplot(rows, cols, i)
        plt.title(label)
        plt.axis("off")
        plt.imshow(np_img.squeeze(), cmap='gray')
    plt.show()
    print(np_img, np_img.shape)
