import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0)
        self.conv1_1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.conv3_3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0)
        self.conv4_4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        self.fc1 = nn.Linear(6400, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x), 2)
        x = F.max_pool2d(F.relu(self.conv1_1(x)), 2)
        x = F.relu(self.conv2(x), 2)
        x = F.max_pool2d(F.relu(self.conv2_2(x)), 2)
        x = F.relu(self.conv3(x), 2)
        x = F.max_pool2d(F.relu(self.conv3_3(x)), 2)
        x = F.relu(self.conv4(x), 2)
        x = F.max_pool2d(F.relu(self.conv4_4(x)), 2)

        x = x.reshape(-1, 6400)
        # 추가
        #x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


model = Net()
summary(model, (1, 150, 150))
