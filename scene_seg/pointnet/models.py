from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class STN3d(nn.Module):
    def __init__(self, num_points = 2048):
        super(STN3d, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.mp1 = nn.MaxPool1d(num_points)

    def forward(self, x):
        batch_size = x.size()[0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.mp1(x)
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        I = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1, 9).repeat(batch_size, 1)
        if x.is_cuda:
            I = I.cuda()
        x = x + I
        x = x.view(-1, 3, 3)
        return x


class PointNetFeature(nn.Module):
    def __init__(self, num_points = 2048):
        super(PointNetFeature, self).__init__()
        self.num_points = num_points
        self.stn = STN3d(num_points)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.mp1 = nn.MaxPool1d(num_points)

    def forward(self, x):
        batch_size = x.size()[0]
        T = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, T)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        point_features = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = self.mp1(x)
        x = x.view(-1, 1024)
        x = x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
        return torch.cat([x, point_features], 1), T


class PointDenseClassifier(nn.Module):
    def __init__(self, num_points = 2048, k = 2):
        super(PointDenseClassifier, self).__init__()
        self.num_points = num_points
        self.k = k
        self.feat = PointNetFeature(num_points)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batch_size = x.size()[0]
        x, T = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        x = F.log_softmax(x.view(-1,self.k), dim=-1)
        x = x.view(batch_size, self.num_points, self.k)
        return x, T


if __name__ == '__main__':
    data = Variable(torch.rand(32, 3, 4096))
    model = PointDenseClassifier(num_points=4096, k=3)
    output, _ = model(data)
    print(output.size())
