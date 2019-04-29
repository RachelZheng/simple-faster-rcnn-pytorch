## test files for computing PR curve and AP score on all the severe weather events 
import pickle, os, six, cv2
import numpy as np
from tqdm import tqdm

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

def _vis_pts(img, pts, clr=(0,0,255)):
	""" visualize points in the image
	Args:
		img: 3 x n x n numpy 
		pts: l x 2 numpy 
	Return: img
	"""
	# transpose (C, H, W) -> (H, W, C)
	img_ = np.copy(img).transpose((1, 2, 0))
	pts_ = np.round(np.copy(pts)).astype(int)
	for pt in pts_:
		img_ = cv2.circle(img_, (pt[1], pt[0]), 3, clr, 3)

	# transpose (H, W, C) -> (C, H, W)	
	img_ = img_.transpose((2, 0, 1))
	return img_


def _vis_bbox(img, bbox, labels, scores, clr=(0,255,0)):
	""" visualize bboxes in the image
	Args:
		img: 3 x n x n numpy 
		bbox: l x 4 numpy
		labels:	l
		scores: l
	Return: img
	"""
	# transpose (C, H, W) -> (H, W, C)
	img_ = np.copy(img).transpose((1, 2, 0))
	bbox_ = np.round(np.copy(bbox)).astype(int)
	for bb in bbox_:
		img_ = cv2.rectangle(img_, (bb[1], bb[0]), (bb[3], bb[2]), clr, 3)

	# transpose (H, W, C) -> (C, H, W)
	img_ = img_.transpose((2, 0, 1))
	return img_



if __name__ == '__main__':
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
		shuffle=False,
		pin_memory=False)

	## record the data into one text file
	folder = os.path.join('/pylon5/ir5fp5p/xzheng4/temp/', opt.model_name)
	os.system('mkdir %s'%(folder))
	f_pts = open(os.path.join(folder, 'pts.txt'), 'w')
	f_bbox = open(os.path.join(folder, 'bbox.txt'), 'w')

	dataloader_iterator = iter(val_dataloader)
	for i in range(len(valset)):
		try:
			(img, points_, labels_, scale, img_name) = next(dataloader_iterator)
		except:
			print('error at num ' + str(i))

"""
	for ii, (img, points_, labels_, scale, img_name) in tqdm(enumerate(val_dataloader)):
		pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(img, [img.shape[2:]])
		pred_bboxes_, pred_labels_, pred_scores_ = pred_bboxes_[0], pred_labels_[0], pred_scores_[0]
		points_, labels_ = at.tonumpy(points_[0]), at.tonumpy(labels_[0])
		if (not len(pred_bboxes_) and not len(points_)):
			continue

		img, scale, img_name = at.tonumpy(img[0]), at.scalar(scale), img_name[0]
		bbox_catch_scores_ = np.zeros((len(pred_bboxes_), ))

		## plot 
		if (ii + 1) % opt.plot_every == 0 and len(pred_bboxes_) and len(pred_bboxes_):
			ori_img_ = inverse_normalize(img)
			# plot image with points and bboxes
			pred_img_ = _vis_pts(ori_img_, points_)
			pred_img_ = _vis_bbox(pred_img_, pred_bboxes_, pred_labels_.reshape(-1), pred_scores_)
			ipdb.set_trace()
			cv2.imwrite(os.path.join(folder, img_name), pred_img_.transpose((2, 0, 1)))

		if len(points_):
			points_ /= scale
			match_score = bbox_event(pred_bboxes_, pred_scores_, points_)
			pts_catch_scores_ = np.max(match_score, axis=0)
			bbox_catch_scores_ = np.max(match_score, axis=1)
			for point, pts_catch_score in six.moves.zip(points_, pts_catch_scores_):
				f_pts.write('{} {:.03f} {:.03f} {:.03f}\n'.format(
					img_name, pts_catch_score, point[0], point[1]))

		if len(pred_bboxes_):
			pred_bboxes_ /= scale
			for pred_bbox, bbox_catch_score, pred_score in six.moves.zip(
				pred_bboxes_, bbox_catch_scores_, pred_scores_):
				f_bbox.write('{} {:.03f} {:.03f} {:.03f} {:.03f} {:.03f} {:.03f}\n'.format(
					img_name, bbox_catch_score,	pred_score, pred_bbox[0], pred_bbox[1], pred_bbox[2], pred_bbox[3]))

	f_pts.close()
	f_bbox.close()
"""