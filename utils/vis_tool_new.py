## new visualization tool
import cv2, matplotlib
import numpy as np
import torch as t
import ipdb


def vis_pts(img, pts, clr=(0,0,255)):
	""" visualize points in the image
	Args:
		img: 3 x n x n numpy 
		pts: l x 2 numpy 
	Return: img
	"""
	# transpose (C, H, W) -> (H, W, C)
	img_ = np.copy(img).transpose((1,2,0))
	pts_ = np.round(np.copy(pts)).astype(int)
	# ipdb.set_trace()
	for pt in pts_:
		img_ = cv2.circle(img_, (pt[1], pt[0]), 3, clr, 3)

	# transpose (H, W, C) -> (C, H, W)	
	img_ = img_.get().transpose((2, 0, 1))
	return img_


def vis_bbox(img, bbox, labels, scores, clr=(0,255,0)):
	""" visualize bboxes in the image
	Args:
		img: 3 x n x n numpy 
		bbox: l x 4 numpy
		labels:	l
		scores: l
	Return: img
	"""
	# transpose (C, H, W) -> (H, W, C)
	img_ = np.copy(img).transpose((1,2,0))
	bbox_ = np.round(np.copy(bbox)).astype(int)
	for bb in bbox_:
		img_ = cv2.rectangle(img_, (bb[1],bb[0]),(bb[3],bb[2]), clr, 3)

	# transpose (H, W, C) -> (C, H, W)	
	img_ = img_.get().transpose((2, 0, 1))
	return img_