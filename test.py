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
from data.dataset import DatasetGeneral, inverse_normalize, Dataset
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
dataset = Dataset(opt, split='val')
dataloader = data_.DataLoader(dataset, \
                              batch_size=1, \
                              shuffle=True, \
                              num_workers=opt.num_workers)

img, points_, labels_, scale = dataset.__getitem__(21315)


dataloader_iterator = iter(dataloader)
img, points_, labels_, scale = next(dataloader_iterator)
