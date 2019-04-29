## test files for computing PR curve and AP score on all the severe weather events 
import pickle, os, six
import numpy as np

from model.utils.creator_tool_pts import AnchorPointTargetCreator, ProposalPointTargetCreator

import torch as t
from torch import nn
from torchvision.models import vgg16
from model.region_proposal_network import RegionProposalNetwork
from model.faster_rcnn import FasterRCNN
from model.faster_rcnn_vgg16 import decom_vgg16
from model.roi_module import RoIPooling2D
from utils import array_tool as at
from utils.config import opt
from torch.utils import data as data_
from collections import namedtuple
from model import FasterRCNNVGG16

import ipdb

from data.dataset import DatasetGeneral, inverse_normalize

from trainer import FasterRCNNTrainer
from logger import Logger

LossTuple = namedtuple('LossTuple',
                       ['rpn_cls_loss',
                        'roi_cls_loss',
                        'total_loss'
                        ])

logger = Logger('./logs')

faster_rcnn = FasterRCNNVGG16(n_fg_class=1)
print('model construct completed')
trainer = FasterRCNNTrainer(faster_rcnn).cuda()
trainer.load(os.path.join(opt.model_dir, opt.model_name))

eval_split = 'test_all'
valset = DatasetGeneral(opt, split=eval_split)

val_dataloader = data_.DataLoader(valset,
                                   batch_size=1,
                                   num_workers=opt.test_num_workers,
                                   shuffle=False, \
                                   pin_memory=True)

## record the data into one text file
f_pts = open('/pylon5/ir5fp5p/xzheng4/temp/pts.txt', 'w')
f_bbox = open('/pylon5/ir5fp5p/xzheng4/temp/bbox.txt', 'w')

for ii, (img, points_, labels_, scale, img_name) in tqdm(enumerate(dataloader)):
	ipdb.set_trace()
	if img is None:
		continue
	pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(img, [img.shape[2:]])

	if (not len(pred_bboxes_) and not len(points_)):
		continue

	bbox_catch_scores_ = np.zeros((len(pred_bboxes_), ))

	if len(points_):
		points_ /= scale
		match_score = bbox_event(pred_bboxes_, pred_scores_, points_)
		pts_catch_scores_ = np.max(match_score, axis=0)
		bbox_catch_scores_ = np.max(match_score, axis=1)
		for point, pts_catch_score in six.moves.zip(points_, pts_catch_scores_):
			f_pts.write('{} {} {} {}\n'.format(img_name, np.round(pts_catch_score,3),
				point[0], point[1]))

	if len(pred_bboxes_):
		pred_bboxes_ /= scale
		for pred_bbox, bbox_catch_score, pred_score in six.moves.zip(
			pred_bboxes_, bbox_catch_scores_, pred_scores_):
			f_bbox.write('{} {} {} {} {} {} {}\n'.format(img_name, np.round(bbox_catch_score,3), 
				np.round(pred_score, 3), pred_bbox[0], pred_bbox[1], pred_bbox[2], pred_bbox[3]))

f_pts.close()
f_bbox.close()