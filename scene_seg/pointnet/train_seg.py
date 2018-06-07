from __future__ import print_function
import sys
import os
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as data
import visdom
from models import PointDenseClassifier
from datasets import SceneNNDataset


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--seed', type=int, default=40)
parser.add_argument('--log-interval', type=int, default=10)
args = parser.parse_args()
args.cuda = torch.cuda.is_available()
print('CUDA:', args.cuda)

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
dataset = SceneNNDataset(root='scenenn')
loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

cudnn.benchmark = True

ignore_classes = torch.LongTensor([0,])
num_classes = 41
weights = torch.ones(num_classes)
weights[ignore_classes] = 0.0
model = PointDenseClassifier(num_points = 4096, k = num_classes)
if args.cuda:
    model.cuda()
    weights = weights.cuda()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

num_batches = len(loader.dataset) / args.batch_size
for epoch in range(args.epochs):
    train_loss = 0.0
    for i, (points, labels) in enumerate(loader):
        points = Variable(points)
        labels = Variable(labels).long()
        if args.cuda:
            points = points.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()
        pred, _ = model(points.transpose(2, 1))
        pred = pred.view(-1, num_classes)
        labels = labels.view(-1, 1)[:,0]
        loss = F.nll_loss(pred, labels, weights)
        loss.backward()
        optimizer.step()

        if i % args.log_interval == 0:
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(labels.data).cpu().sum()
            accuracy = float(correct.data) / float(args.batch_size * 4096)
            print('[{}: {}/{}]\tloss: {:.6f}\taccuracy: {:.4f}'.format(
                epoch, i, num_batches, loss.data, accuracy))

    torch.save(model.state_dict(), 'pointnet_scenenn_epoch{0:03}.pth'.format(epoch))
