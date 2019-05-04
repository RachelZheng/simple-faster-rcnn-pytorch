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
    eval_split = 'inference'
    dataset = DatasetGeneral(opt, split=eval_split)

    dataloader = data_.DataLoader(dataset, \
                                  batch_size=1, \
                                  shuffle=False, \
                                  pin_memory=False)

    print('load model')
    faster_rcnn = FasterRCNNVGG16(n_fg_class=1)
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    trainer.load(os.path.join(opt.model_dir, opt.model_name))
    f_pts = open(os.path.join(opt.eval_dir, '%s_layer%d_%s_pts.txt'%(
        eval_split, opt.n_layer_fix, opt.model_name)), 'w')
    f_bbox = open(os.path.join(opt.eval_dir, '%s_layer%d_%s_bbox.txt'%(
        eval_split, opt.n_layer_fix, opt.model_name)), 'w')

    ## put the inference files onto the text file
    for ii, (img, points_, labels_, scale, img_name) in tqdm(enumerate(dataloader)):
        img = at.tonumpy(img[0])
        ori_img_ = inverse_normalize(img)
        pred_bboxes_, pred_labels_, pred_scores_ = trainer.faster_rcnn.predict([ori_img_], visualize=True)
        pred_bboxes_, pred_labels_ = at.tonumpy(pred_bboxes_[0]), at.tonumpy(pred_labels_[0])
        points_, labels_, pred_scores_ = at.tonumpy(points_[0]), at.tonumpy(labels_[0]), at.tonumpy(pred_scores_[0])
        if (not len(pred_bboxes_) and not len(points_)):
            continue

        scale, img_name = at.scalar(scale), img_name[0]
        bbox_catch_scores_ = np.zeros((len(pred_bboxes_), ))

        if len(points_):
            pts_catch_scores_ = np.zeros((len(points_), ))
            if len(pred_bboxes_):
                match_score = bbox_event(pred_bboxes_, pred_scores_, points_)
                pts_catch_scores_ = np.max(match_score, axis=0)
                bbox_catch_scores_ = np.max(match_score, axis=1)
            for point, pts_catch_score in six.moves.zip(points_, pts_catch_scores_):
                f_pts.write('{} {:.03f} {:.03f} {:.03f}\n'.format(
                    img_name, pts_catch_score, point[0], point[1]))

        if len(pred_bboxes_):
            for pred_bbox, bbox_catch_score, pred_score in six.moves.zip(
                pred_bboxes_, bbox_catch_scores_, pred_scores_):
                f_bbox.write('{} {:.03f} {:.03f} {:.03f} {:.03f} {:.03f} {:.03f}\n'.format(
                    img_name, bbox_catch_score, pred_score, pred_bbox[0], 
                    pred_bbox[1], pred_bbox[2], pred_bbox[3]))

    f_pts.close()
    f_bbox.close()




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
        valset = DatasetGeneral(opt, split=eval_split)
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
