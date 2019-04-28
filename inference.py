from __future__ import absolute_import
# though cupy is not used but without this line, it raise errors...
import sys, os, cv2, ipdb

import numpy as np
import matplotlib
from tqdm import tqdm

from utils.config import opt
from data.dataset import InferDataset, inverse_normalize
from model import FasterRCNNVGG16
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from utils import array_tool as at
from utils.vis_tool_new import vis_pts, vis_bbox

## tensorboard recording
from logger import Logger

# fix for ulimit
# https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

matplotlib.use('agg')

def inference(**kwargs):
    opt._parse(kwargs)
    print('load inference data')
    dataset = InferDataset(opt)
    dataloader = data_.DataLoader(dataset, \
                                  batch_size=1, \
                                  shuffle=False, \
                                  num_workers=opt.num_workers)

    print('load model')
    faster_rcnn = FasterRCNNVGG16(n_fg_class=1)
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    trainer.load(os.path.join(opt.model_dir, opt.model_name))
    for ii, (img, scale, img_name) in tqdm(enumerate(dataloader)):
        ipdb.set_trace()
        img = img.cuda().float()
        ori_img_ = inverse_normalize(at.tonumpy(img[0]))
        _bboxes, _labels, _scores = trainer.faster_rcnn.predict(
            [ori_img_], visualize=True)
        ori_img_ = vis_bbox(ori_img_, 
                    at.tonumpy(_bboxes[0]), 
                    at.tonumpy(_labels[0]).reshape(-1),
                    at.tonumpy(_scores[0]))

        cv2.imwrite(os.path.join(opt.inference_out_dir, img_name[0]), ori_img_.transpose((1,2,0)))


if __name__ == '__main__':
    import fire

    fire.Fire()
