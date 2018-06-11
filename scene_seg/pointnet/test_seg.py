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
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--steps', type=int, default=20000)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--seed', type=int, default=41)
parser.add_argument('--log-interval', type=int, default=10)
args = parser.parse_args()
args.cuda = torch.cuda.is_available()
print('CUDA:', args.cuda)

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
dataset = SceneNNDataset(root='scenenn', training=False)
loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

cudnn.benchmark = True

num_classes = 41
model = PointDenseClassifier(num_points = 4096, k = num_classes)
model.load_state_dict(torch.load('pointnet_scenenn_epoch070.pth'))
if args.cuda:
    model.cuda()
model.eval()


per_class_correct = np.zeros(41, dtype=long)
per_class_labels = np.zeros(41, dtype=long)
num_batches = len(loader.dataset) / args.batch_size
for i, (points, labels) in enumerate(loader):
    points = Variable(points)
    labels = Variable(labels).long()
    if args.cuda:
        points = points.cuda()
        labels = labels.cuda()

    pred, _ = model(points.transpose(2, 1))
    pred = pred.view(-1, num_classes)
    labels = labels.view(-1, 1)[:,0]

    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(labels.data).cpu()
    correct_labels = pred_choice[correct]

    per_class_correct += np.bincount(correct_labels.data, minlength=num_classes)
    per_class_labels += np.bincount(labels.data, minlength=num_classes)

accuracy = per_class_correct.astype(np.float64) / per_class_labels.astype(np.float64)

for i in accuracy:
    print('{0:0.3f}'.format(i))
