import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import os, argparse
from datetime import datetime
from utils.data import get_loader
from utils.func import label_edge_prediction, AvgMeter
from model.MAIRN_SwinB import MAIRN

parser = argparse.ArgumentParser()
argument = parser.add_argument('--epoch', type=int, default=60, help='epoch number')
parser.add_argument('--lr', type=float, default=2e-3, help='learning rate')
parser.add_argument('--batchsize', type=int, default=10, help='batch size')
parser.add_argument('--trainsize', type=int, default=224, help='input size')
parser.add_argument('--trainset', type=str, default='DUTS-TR', help='training  dataset')

opt = parser.parse_args()
data_path = 'E:/NSI_dataset/'
image_root = data_path + opt.trainset + '/train-images/'
gt_root = data_path + opt.trainset + '/train-labels/'

train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
total_step = len(train_loader)

# build models
model = MAIRN()
model.cuda()
params = model.parameters()
optimizer = torch.optim.SGD(params, opt.lr, momentum=0.9, weight_decay=5e-4)
scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

CE = torch.nn.BCEWithLogitsLoss()
def DiceLoss(inputs, targets):
    eps = 1e-5
    dice_loss = []
    for i in range(inputs.size(0)):
        inter = (inputs[i] * targets[i]).sum()
        dice = (2 * inter + eps) / (inputs[i].sum() + targets[i].sum() + eps)
        dice_loss.append(1 - dice)
    return torch.mean(torch.tensor(dice_loss))

# training
for epoch in range(0, opt.epoch):
    model.train()
    loss_record1, loss_record2 = AvgMeter(), AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()

        images, gts = pack
        images = Variable(images).cuda()
        gts = Variable(gts).cuda()
        # edge prediction
        gt_edges = label_edge_prediction(gts)

        # forward
        sal, sal_sig, pred_edge = model(images)

        loss1 = CE(sal, gts) + DiceLoss(sal_sig, gts)
        loss2 = CE(pred_edge, gt_edges)
        loss = loss1 + loss2
        loss.backward()

        optimizer.step()

        loss_record1.update(loss1.data, opt.batchsize)
        loss_record2.update(loss2.data, opt.batchsize)

        if i % 100 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}]'.format(datetime.now(), epoch, opt.epoch, i, total_step))
            print('loss1：', loss_record1.show())
            print('loss2：', loss_record2.show())

    scheduler.step()
    if (epoch + 1) > 41:
        save_path = 'models/MAIRN_SwinB/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + 'MAIRN_SwinB' + '_' + str(epoch + 1) + '.pth')
