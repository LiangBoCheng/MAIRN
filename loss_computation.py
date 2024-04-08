import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import os
from scipy import misc
import imageio
from utils.data import test_dataset
from model.MAIRN_Res2Net50 import MAIRN

model = MAIRN()
model.load_state_dict(torch.load('E:\my_model\MAIRN\models\MAIRN_Res2Net50.pth')) # Loading model weights
model.cuda()
model.eval()

dataset_path = 'E:/NSI_dataset/'  # Dataset Path
test_datasets = ['DUTS-TR']  # valset = ['DUTS-TR', 'DUTS-TE']

for dataset in test_datasets:
    save_path = './results/' + 'MAIRN-Res2Net50-' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset + '/train-images/'
    print(dataset)
    gt_root = dataset_path + dataset + '/train-labels/'
    test_loader = test_dataset(image_root, gt_root, testsize=352)
    BCE = nn.BCEWithLogitsLoss()
    loss_sum = 0
    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        gt_tensor = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0).cuda()

        res, res_sig, edge = model(image)

        res = F.interpolate(res, size=gt.shape[:2], mode='bilinear', align_corners=False)
        res_sig = F.interpolate(res_sig, size=gt.shape[:2], mode='bilinear', align_corners=False)

        loss = BCE(res_sig, gt_tensor)
        loss_sum += loss.item()

        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        imageio.imsave(save_path+name, res)

    average_loss = loss_sum / test_loader.size
    print('Average cross-entropy loss on {} dataset: {:.5f}'.format(dataset, average_loss))

