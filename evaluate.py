## test files for computing PR curve and AP score on all the severe weather events 
import pickle, os, six, cv2, glob, sys
import numpy as np
from tqdm import tqdm

from model.utils.creator_tool_pts import AnchorPointTargetCreator, ProposalPointTargetCreator
from model.utils.bbox_pts_tools import bbox_event

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
	if isinstance(img_, np.ndarray):
		img_ = img_.transpose((2, 0, 1))
	else:
		img_ = img_.get().transpose((2, 0, 1))
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
	if isinstance(img_, np.ndarray):
		img_ = img_.transpose((2, 0, 1))
	else:
		img_ = img_.get().transpose((2, 0, 1))
	return img_



if __name__ == '__main__':
	logger = Logger('./logs')

	faster_rcnn = FasterRCNNVGG16(n_fg_class=1)
	print('model construct completed')
	trainer = FasterRCNNTrainer(faster_rcnn).cuda()

	eval_split = 'val_all'
	valset = DatasetGeneral(opt, split=eval_split)

	val_dataloader = data_.DataLoader(valset, 
		batch_size=1,
		num_workers=opt.test_num_workers, 
		shuffle=False,
		pin_memory=False)

	## record the data into one text file
	os.chdir(opt.model_dir)
	n_epoch = int(sys.argv[1])
	for name_model in glob.glob('fasterrcnn_*_%d_*'%(n_epoch)):
		f_pts = open(os.path.join(opt.eval_dir, '%s_layer%d_%s_pts.txt'%(
			eval_split, opt.n_layer_fix, name_model)), 'w')
		f_bbox = open(os.path.join(opt.eval_dir, '%s_layer%d_%s_bbox.txt'%(
			eval_split, opt.n_layer_fix, name_model)), 'w')
		trainer.load(os.path.join(opt.model_dir, name_model))

		for ii, (img, points_, labels_, scale, img_name) in tqdm(enumerate(val_dataloader)):
			img = at.tonumpy(img[0])
			ori_img_ = inverse_normalize(img)
			pred_bboxes_, pred_labels_, pred_scores_ = trainer.faster_rcnn.predict([ori_img_], visualize=True)
			pred_bboxes_, pred_labels_ = at.tonumpy(pred_bboxes_[0]), at.tonumpy(pred_labels_[0])
			points_, labels_, pred_scores_ = at.tonumpy(points_[0]), at.tonumpy(labels_[0]), at.tonumpy(pred_scores_[0])
			if (not len(pred_bboxes_) and not len(points_)):
				continue

			scale, img_name = at.scalar(scale), img_name[0]
			bbox_catch_scores_ = np.zeros((len(pred_bboxes_), ))

			"""
			## plot 
			if (ii + 1) % opt.plot_every == 0:
				if len(points_):
					ori_img_ = _vis_pts(ori_img_, points_)
				if len(pred_bboxes_):
					ori_img_ = _vis_bbox(ori_img_, pred_bboxes_, pred_labels_.reshape(-1), pred_scores_)
				ori_img_ = ori_img_.transpose((1, 2, 0)).astype('uint8')
				cv2.imwrite(os.path.join(folder, img_name), ori_img_)
			"""
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
						img_name, bbox_catch_score,	pred_score, pred_bbox[0], 
						pred_bbox[1], pred_bbox[2], pred_bbox[3]))

		f_pts.close()
		f_bbox.close()
