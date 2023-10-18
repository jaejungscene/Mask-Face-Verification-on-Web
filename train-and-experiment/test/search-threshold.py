import glob
import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from model import get_model
from dataset import cutout_mask
import random
import wandb



def cos_sim(x:torch.tensor, y:torch.tensor)->torch.tensor:
    return x.view(-1).dot(y.view(-1)) / (torch.norm(x)*torch.norm(y))


DATA_DIR = "/home/ljj0512/private/workspace/CV-project/Computer-Vision-Project/train-and-experiment/data/lfw"


file_list = glob.glob(os.path.join(DATA_DIR,"*"))
sorted_list = sorted(file_list, key=lambda x: x.split("/")[-1])
print("the total number of IDs: ",len(file_list))

total_img = 0
more_than_two_img = 0
total_more_than_two_img = 0
data_path_list = []
for file in file_list:
    L = glob.glob(os.path.join(file, "*"))
    if len(L) > 1:
        data_path_list.append(L)
        total_img += len(L)
        total_more_than_two_img += len(L)
        more_than_two_img += 1
        # print(len(L))
        # print(L)
    else:
        total_img += 1

print("the number of images: ",total_img)
print("-"*50)
print("the number of IDs with more than one image: ",more_than_two_img)
print("the number of imagess with more than one image: ",total_more_than_two_img)



class Transforms:
    # normalize = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
    normalize = [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]
    test_lfw = transforms.Compose([
        transforms.CenterCrop(112),
        # transforms.Resize((112,112)),
        transforms.ToTensor(),
        transforms.Normalize(*normalize)
    ])

class LFWMatchDataset(Dataset):
    def __init__(self, data_path_list, transform=None):
        self.data_paths = data_path_list
        self.transform = transform
    
    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, index):
        images = []
        for path in self.data_paths[index]:
            if self.transform:
                images.append(self.transform(Image.open(path)))
            else:
                images.append(Image.open(path))
        return {"images":images, "label":torch.tensor(index)}

test_set =  LFWMatchDataset(data_path_list,Transforms.test_lfw)
data_dir = "/home/ljj0512/private/workspace/CV-project/Computer-Vision-Project/train-and-experiment/"
model = get_model(base_dir=data_dir)

ids = len(test_set)
device = "cuda:1"
model = model.to(device)
model.eval()
print(ids)

cutout = True

TP = 0
FP = 0
TN = 0
FN = 0

wandb.init(project='mask_face_verification', entity='jaejungscene')

sim_threshold = 0.33
max_threshold = 0.42
increase = 0.01
with torch.no_grad():
    while sim_threshold <= max_threshold:
        print(f"===========================<< {sim_threshold:0.2} >>=========================")
        print(f"idx\t| TP\t TN\t FN\t FP\t | TAR%   @ FAR%")
        print("--------------------------------------------------------")
        TP, FP, TN, FN = 0,0,0,0
        for i in range(ids):
            images = test_set[i]["images"]
            label = test_set[i]["label"]

            # print()
            # make same_vec_list
            same_vec_list = []
            for img in images:
                img, label = img.to(device), label.to(device)
                if cutout:
                    img = cutout_mask(img.unsqueeze(0))
                else:
                    img = img.unsqueeze(0)
                embed_vec = model(img)
                same_vec_list.append(embed_vec)

            # make diff_vec_list
            diff_vec_list = []
            for _ in range(5):
                randidx = random.randint(0,ids-1)
                while randidx == i:
                    randidx = random.randint(0,ids-1)
                n = random.randint(0,len(test_set[randidx]["images"])-1)
                if cutout:
                    img = cutout_mask(test_set[randidx]["images"][n].unsqueeze(0)).to(device)
                else:
                    img = test_set[randidx]["images"][n].unsqueeze(0).to(device)
                embed_vec = model(img)
                diff_vec_list.append(embed_vec)
            
            # same_vec_list의 0번쨰 vector를 등록된 vector로 가정한다.
            for j in range(1,len(same_vec_list)):
                sim_score = cos_sim(same_vec_list[0], same_vec_list[j])
                # print(sim_score)
                if sim_score > sim_threshold:
                    TP += 1
                else:
                    FN += 1     # positive이지만 negative라고 질못 판단함
            # print()
            for j in range(len(diff_vec_list)):
                sim_score = cos_sim(same_vec_list[0], diff_vec_list[j])
                # print(sim_score)
                if sim_score > sim_threshold:
                    FP += 1     # negative이지만 positive라고 잘못 판단함
                else:
                    TN += 1
            FAR = (FP/(FP+TN))*100
            TAR = (TP/(TP+FN))*100
            if i%100 == 0:
                print(f"{i}\t| {TP}\t {TN}\t {FN}\t {FP}\t | {TAR:0.4}% @ {FAR:0.4}%")
        print(f"Last\t| {TP}\t {TN}\t {FN}\t {FP}\t | {TAR:0.4}% @ {FAR:0.4}%")
        with open("/home/ljj0512/private/workspace/CV-project/Computer-Vision-Project/train-and-experiment/search-threshold.txt", "a") as f:
            f.write(f"threshold: {sim_threshold}, {TAR:0.4}% @ {FAR:0.4}%\n")
        print()
        sim_threshold += increase
            # break
