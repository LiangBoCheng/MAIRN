import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os
# from scipy import misc
import imageio
import time
from utils.data import test_dataset
from model.MAIRN import MAIRN

model = MAIRN()
model.load_state_dict(torch.load('./models/MAIRN/MAIRN_DUTS-TR.pth'))
model.cuda()
model.eval()

dataset_path = 'E:/NSI_dataset/'
test_datasets = ['ECSSD']  # valset = ['HKU-IS', 'PASCALS', 'DUT-O', 'DUTS-TE','ECSSD']

for dataset in test_datasets:
    save_path = './results/' + 'MAIRN-' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset + '/test-images/'
    print(dataset)
    gt_root = dataset_path + dataset + '/test-labels/'
    test_loader = test_dataset(image_root, gt_root, testsize=352)
    time_sum = 0
    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        time_start = time.time()
        res, res_sig, edge = model(image)
        time_end = time.time()
        time_sum = time_sum+(time_end-time_start)
        res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        imageio.imsave(save_path+name, res)
        if i == test_loader.size-1:
            print('Running time {:.5f}'.format(time_sum/test_loader.size))
            print('Average speed: {:.4f} fps'.format(test_loader.size/time_sum))