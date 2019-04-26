## find the broken image in the training set
from __future__ import absolute_import
# though cupy is not used but without this line, it raise errors...
import sys
# sys.path.insert(0, "/pylon5/ir5fp5p/xzheng4/conda_install/envs/py3/lib/python3.6/site-packages/")

import cupy as cp
import os

import numpy as np
import ipdb
import matplotlib
from tqdm import tqdm

from utils.config import opt
from data.dataset import Dataset, TestDataset, inverse_normalize
from model import FasterRCNNVGG16
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from utils import array_tool as at
# from utils.vis_tool import visdom_bbox
from utils.vis_tool_new import vis_pts, vis_bbox
# from utils.eval_tool import eval_detection_voc
from utils.eval_tool_new import eval_detection

## tensorboard recording
from logger import Logger

# fix for ulimit
# https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

matplotlib.use('agg')

def debug(**kwargs):
    opt._parse(kwargs)

    dataset = Dataset(opt)  # temp setting
    print('load data')
    logger = Logger('./logs')

    dataloader = data_.DataLoader(dataset, \
                                  batch_size=1, \
                                  shuffle=False, \
                                  num_workers=opt.num_workers)
    
    faster_rcnn = FasterRCNNVGG16(n_fg_class=1)
    print('model construct completed')
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    best_map = 0
    lr_ = opt.lr
    (img, points_, labels_, scale) = dataset.__getitem__(7433)
    scale = at.scalar(scale)
    img, points, labels = img.cuda().float(), points_.cuda(), labels_.cuda()
    trainer.train_step(img, points, labels, scale)



def train(**kwargs):
    opt._parse(kwargs)

    dataset = Dataset(opt)  # temp setting
    print('load data')
    logger = Logger('./logs')

    dataloader = data_.DataLoader(dataset, \
                                  batch_size=1, \
                                  shuffle=False, \
                                  num_workers=opt.num_workers)
    
    faster_rcnn = FasterRCNNVGG16(n_fg_class=1)
    print('model construct completed')
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    best_map = 0
    lr_ = opt.lr
    for epoch in range(opt.epoch):
        trainer.reset_meters()
        for ii, (img, points_, labels_, scale) in tqdm(enumerate(dataloader)):
            scale = at.scalar(scale)
            img, points, labels = img.cuda().float(), points_.cuda(), labels_.cuda()
            ## skip abnormal images and zero points
            if len(img.shape) < 4 or len(points.shape) < 3 or points.shape[2] < 1 or img.shape[3] < 600:
                continue
            
            ## --- just debug ----
            if ii == 7433:
                ipdb.set_trace()
                trainer.train_step(img, points, labels, scale)
            else:
                trainer.train_step(img, points, labels, scale)


if __name__ == '__main__':
    import fire

    fire.Fire()
