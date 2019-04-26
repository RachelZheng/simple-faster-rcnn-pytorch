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


def eval(dataloader, faster_rcnn, test_num=100):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_pts, gt_labels = list(), list()
    for ii, (img, points_, labels_, scale) in tqdm(enumerate(dataloader)):
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(img, [img.shape[2:]])
        gt_pts += list(points_.numpy())
        gt_labels += list(labels_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num: break

    result = eval_detection(
        pred_bboxes, pred_labels, pred_scores,
        gt_pts, gt_labels)
    return result


def train(**kwargs):
    opt._parse(kwargs)

    dataset = Dataset(opt)  # temp setting
    print('load data')
    logger = Logger('./logs')

    dataloader = data_.DataLoader(dataset, \
                                  batch_size=1, \
                                  shuffle=False, \
                                  # pin_memory=True,
                                  num_workers=opt.num_workers)
    testset = TestDataset(opt)
    test_dataloader = data_.DataLoader(testset,
                                       batch_size=1,
                                       num_workers=opt.test_num_workers,
                                       shuffle=False, \
                                       pin_memory=True
                                       )
    faster_rcnn = FasterRCNNVGG16(n_fg_class=1)
    print('model construct completed')
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    if opt.load_path:
        trainer.load(opt.load_path)
        print('load pretrained model from %s' % opt.load_path)
    # trainer.vis.text(dataset.db.label_names, win='labels')
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
            """
            if (ii + 1) % opt.plot_every == 0:
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()

                # plot loss
                trainer.vis.plot_many(trainer.get_meter_data())

                # plot groud truth bboxes
                ori_img_ = inverse_normalize(at.tonumpy(img[0]))
                gt_img = visdom_bbox(ori_img_,
                                     at.tonumpy(bbox_[0]),
                                     at.tonumpy(label_[0]))
                trainer.vis.img('gt_img', gt_img)

                # plot predicti bboxes
                _bboxes, _labels, _scores = trainer.faster_rcnn.predict([ori_img_], visualize=True)
                pred_img = visdom_bbox(ori_img_,
                                       at.tonumpy(_bboxes[0]),
                                       at.tonumpy(_labels[0]).reshape(-1),
                                       at.tonumpy(_scores[0]))
                trainer.vis.img('pred_img', pred_img)

                # rpn confusion matrix(meter)
                trainer.vis.text(str(trainer.rpn_cm.value().tolist()), win='rpn_cm')
                # roi confusion matrix
                trainer.vis.img('roi_cm', at.totensor(trainer.roi_cm.conf, False).float())
            """
            if (ii + 1) % opt.plot_every == 0:
                info = trainer.get_meter_data()
                for tag, value in info.items():
                    logger.scalar_summary(tag + str(epoch), value, ii+1)
                
                ori_img_ = inverse_normalize(at.tonumpy(img[0]))

                # plot image with points and bboxes
                pred_img_ = vis_pts(ori_img_, at.tonumpy(points_[0]))
                _bboxes, _labels, _scores = trainer.faster_rcnn.predict([ori_img_], visualize=True)
                pred_img_ = vis_bbox(pred_img_, 
                    at.tonumpy(_bboxes[0]), 
                    at.tonumpy(_labels[0]).reshape(-1),
                    at.tonumpy(_scores[0]))

                key_img = 'img' + str(ii+1)
                info = { key_img: pred_img_}
                for tag, images in info.items():
                    logger.image_summary(tag, np.expand_dims(images.transpose((1,2,0)), axis=0) , ii+1)

            # evaluation on every batch
            eval_result = eval(test_dataloader, trainer.faster_rcnn, test_num=opt.test_num)
            ipdb.set_trace()
            lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']
            print('epoch {}, lr:{}, loss:{}, precision:{}, recall:{}\n'.format(
                str(epoch), str(lr_), str(trainer.get_meter_data()), 
                str(eval_result['prec'][2]), str(eval_result['rec'][2])))
        
        # Log scalar values (scalar summary)
        # logger.scalar_summary('accuracy', eval_result['map'], epoch+1)
        for tag, value in trainer.get_meter_data().items():
            logger.scalar_summary(tag, value, epoch+1)

        """
        if eval_result['map'] > best_map:
            best_map = eval_result['map']
            best_path = trainer.save(best_map=best_map)
        if epoch == 9:
            trainer.load(best_path)
            trainer.faster_rcnn.scale_lr(opt.lr_decay)
            lr_ = lr_ * opt.lr_decay
        """
        # save the model for every epoch
        path = trainer.save(n_epoch=epoch,
            prec=np.round(eval_result['prec'][2], 2),
            rec=np.round(eval_result['rec'][2], 2))


        if epoch == 13: 
            break


if __name__ == '__main__':
    import fire

    fire.Fire()
