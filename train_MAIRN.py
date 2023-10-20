import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.autograd import Variable
import os, argparse
from datetime import datetime
from utils.data import get_loader
from utils.func import label_edge_prediction, AvgMeter
from model.MAIRN import MAIRN
# from model.MAIRN_Res2Net50 import MAIRN

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=60, help='epoch number')
parser.add_argument('--lr', type=float, default=2e-3, help='learning rate')
parser.add_argument('--batchsize', type=int, default=10, help='batch size')
parser.add_argument('--trainsize', type=int, default=352, help='input size')
parser.add_argument('--trainset', type=str, default='DUTS-TR', help='training  dataset')

opt = parser.parse_args()
data_path = 'E:/NSI_dataset/'   # D:/DUTS/
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
    inter = (inputs * targets).sum()
    eps = 1e-5
    dice = (2 * inter + eps) / (inputs.sum() + targets.sum() + eps)

    return 1 - dice

size_rates = [0.75, 1, 1.25]  # multi-scale training

# training
for epoch in range(0, opt.epoch):
    model.train()
    loss_record1, loss_record2 = AvgMeter(), AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()

            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            # edge prediction
            gt_edges = label_edge_prediction(gts)

            trainsize = int(round(opt.trainsize*rate/32)*32)
            if rate != 1:
                images = F.interpolate(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.interpolate(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gt_edges = F.interpolate(gt_edges, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            # forward
            sal, sal_sig, pred_edge = model(images)

            loss1 = CE(sal, gts) + DiceLoss(sal_sig, gts)
            loss2 = CE(pred_edge, gt_edges)
            loss = loss1 + loss2
            loss.backward()

            optimizer.step()

            if rate == 1:
                loss_record1.update(loss1.data, opt.batchsize)
                loss_record2.update(loss2.data, opt.batchsize)

        if i % 100 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}]'.format(datetime.now(), epoch, opt.epoch, i, total_step))
            print('loss1：', loss_record1.show())
            print('loss2：', loss_record2.show())

    scheduler.step()
    if (epoch + 1) > 41:
        save_path = 'models/MAIRN/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + 'MAIRN_DUTS-TR' + '_' + str(epoch + 1) + '.pth')
