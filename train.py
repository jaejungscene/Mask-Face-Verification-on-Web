import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from data_loader import masked_face_dataset as FaceDataset, ImageTransform as Transform
from model import Net
import torch.optim as optim

PATH = './dataset/mask_dataset'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trans_config = {"mean": (0.5, 0.5, 0.5), "std": (0.2, 0.2, 0.2), "size": 1024}

#myDataset = FaceDataset(dir_path=PATH, phase='train',
#                        transform=Transform(trans_config["size"], trans_config["mean"], trans_config["mean"]))
myDataset = FaceDataset(dir_path=PATH, phase='train')
print(len(myDataset))


def show_img(x_label, x_img, y_label, y_img):
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(8, 6))
    fig.add_subplot(2, 2, 1)
    plt.title(x_label)
    plt.imshow(x_img.squeeze())
    fig.add_subplot(2, 2, 2)
    plt.title(y_label)
    plt.imshow(y_img.squeeze())
    plt.show()

show_img(myDataset[0][0], myDataset[0][1], myDataset[0][2], myDataset[0][3])
# --------데이터 출력----------
# plt.style.use('dark_background')
# fig = plt.figure()
#
# for i in range(len(train_dataset)):
#     x, y = train_dataset[i]
#
#
#     plt.subplot(2, 1, 1)
#     plt.title(str(y_train[i]))
#     print(x_train[i].shape)
#     plt.imshow(x_train[i].reshape((26, 34)), cmap='gray')
#     print(x_train[i].shape)
#     print(x_train[i].reshape((26, 34)).shape)
#
#     plt.show()


def accuracy(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc


train_dataloader = DataLoader(myDataset, batch_size=32, shuffle=True, num_workers=0)
"""
model = Net()
model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

epochs = 50

for epoch in range(epochs):
    running_loss = 0.0
    running_acc = 0.0

    model.train()

    for i, data in enumerate(train_dataloader, 0):
        input_1, labels = data[0].to(device), data[1].to(device)

        input = input_1.transpose(1, 3).transpose(2, 3)
        # print("input: ", input.shape)
        # print("labels ",labels.shape)


        optimizer.zero_grad()

        outputs = model(input)
        # print("outputs:  ", outputs.shape)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_acc += accuracy(outputs, labels)
        print("I = ", i)
        if i % 80 == 79:
            print('epoch: [%d/%d] train_loss: %.5f train_acc: %.5f' % (
                epoch + 1, epochs, running_loss / 80, running_acc / 80))
            running_loss = 0.0

print("learning finish")
torch.save(model.state_dict(), PATH)
"""