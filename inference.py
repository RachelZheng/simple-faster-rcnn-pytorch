from __future__ import absolute_import
# though cupy is not used but without this line, it raise errors...
import sys, os, cv2

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.config import opt
from data.dataset import InferDataset, inverse_normalize
from model import FasterRCNNVGG16
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from utils import array_tool as at
import train

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
        img = img.cuda().float()
        ori_img_ = inverse_normalize(at.tonumpy(img[0]))
        _bboxes, _labels, _scores = trainer.faster_rcnn.predict(
            [ori_img_], visualize=True)

        if len(_labels[0]) > 0:
            ori_img_ = _vis_bbox(ori_img_, 
                        at.tonumpy(_bboxes[0]), 
                        at.tonumpy(_labels[0]).reshape(-1),
                        at.tonumpy(_scores[0]))

        ori_img_ = ori_img_.transpose((1,2,0))
        cv2.imwrite(os.path.join(opt.inference_out_dir, img_name[0]), ori_img_)


def _vis_bbox(img, bbox, labels, scores, clr=(0,255,0)):
    """ visualize bboxes in the image
    Args:
        img: 3 x n x n numpy 
        bbox: l x 4 numpy
        labels: l
        scores: l
    Return: img
    """
    # transpose (C, H, W) -> (H, W, C)
    img_ = np.copy(img).transpose((1, 2, 0))
    bbox_ = np.round(np.copy(bbox)).astype(int)
    for bb in bbox_:
        img_ = cv2.rectangle(img_, (
            bb[1], bb[0]), (bb[3], bb[2]), clr, 3)
    return img_


def eval_testset(**kwargs):
    """ evaluate the performance on the whole test set
    1. compute the precision and recall according to the model
    """
    opt._parse(kwargs)
    logger = Logger('./logs')

    faster_rcnn = FasterRCNNVGG16(n_fg_class=1)
    print('model construct completed')
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    trainer.load(os.path.join(opt.model_dir, opt.model_name))

    for eval_split in ['val_all', 'test_all']:
        valset = Dataset(opt, split=eval_split)
        val_dataloader = data_.DataLoader(valset,
                                           batch_size=1,
                                           num_workers=opt.test_num_workers,
                                           shuffle=False, \
                                           pin_memory=True)

        val_result = eval(val_dataloader, trainer.faster_rcnn, test_num=len(valset))
        # save the images in output dir
        plt.figure()
        plt.axes().set_aspect('equal')
        plt.plot(val_result['prec'][0], val_result['rec'][0], '+b', label='P-R of BBOX')  # A small noise in the source
        plt.plot(val_result['prec'][0], val_result['rec_pts'][0], 'xr', label='P-R of PTS')
        plt.savefig(os.path.join(opt.inference_out_dir, eval_split + '.pdf'))
        plt.close()

        del valset, val_dataloader, val_result


if __name__ == '__main__':
    import fire

    fire.Fire()
