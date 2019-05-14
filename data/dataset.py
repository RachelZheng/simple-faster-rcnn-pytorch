from __future__ import absolute_import
from __future__ import division
import torch as t
from skimage import transform as sktsf
from torchvision import transforms as tvtsf
from data import util
import numpy as np
from utils.config import opt
import ipdb

# add personal dataset
from data.storm_dataset import StormDataset, ModelDataset

# ------ change for storm dataset ------
MEAN_IMG = [3.2187, 3.2187, 3.2187]
STD_IMG = [13.6640, 13.6640, 13.6640]


def inverse_normalize(img):
    if opt.caffe_pretrain:
        img = img + np.array(MEAN_IMG).reshape(3, 1, 1)
        return img[::-1, :, :]
    # approximate un-normalize for visualize
    return (img * np.mean(STD_IMG) + np.mean(MEAN_IMG)).clip(min=0, max=255)


def pytorch_normalze(img):
    """
    https://github.com/pytorch/vision/issues/223
    return appr -1~1 RGB
    """
    normalize = tvtsf.Normalize(mean=MEAN_IMG, std=STD_IMG)
    img = normalize(t.from_numpy(img).float())
    return img.numpy()


def caffe_normalize(img):
    """
    return appr -125-125 BGR
    """
    img = img[[2, 1, 0], :, :]  # RGB-BGR
    # img = img * 255
    mean = np.array(MEAN_IMG).reshape(3, 1, 1)
    img = (img - mean).astype(np.float32, copy=True)
    return img


def preprocess(img, min_size=600, max_size=1000):
    """Preprocess an image for feature extraction.

    The length of the shorter edge is scaled to :obj:`self.min_size`.
    After the scaling, if the length of the longer edge is longer than
    :param min_size:
    :obj:`self.max_size`, the image is scaled to fit the longer edge
    to :obj:`self.max_size`.

    After resizing the image, the image is subtracted by a mean image value
    :obj:`self.mean`.

    Args:
        img (~numpy.ndarray): An image. This is in CHW and RGB format.
            The range of its value is :math:`[0, 255]`.

    Returns:
        ~numpy.ndarray: A preprocessed image.

    """
    C, H, W = img.shape
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)
    # img = img / 255.
    img = sktsf.resize(img, (C, H * scale, W * scale), mode='reflect',anti_aliasing=False)
    # both the longer and shorter should be less than
    # max_size and min_size
    if opt.caffe_pretrain:
        normalize = caffe_normalize
    else:
        normalize = pytorch_normalze
    return normalize(img)


class Transform(object):

    def __init__(self, min_size=600, max_size=1000, bool_img_only=True):
        self.min_size = min_size
        self.max_size = max_size
        self.bool_img_only = bool_img_only

    def __call__(self, in_data):
        if self.bool_img_only:
            img = in_data
        else:
            img, points, labels = in_data

        _, H, W = img.shape
        img = preprocess(img, self.min_size, self.max_size)
        _, o_H, o_W = img.shape
        scale = o_H / H

        if self.bool_img_only:
            return img, scale
        else:
            points = util.resize_pts(points, (H, W), (o_H, o_W))            
            return img, points, labels, scale


class Dataset:
    def __init__(self, opt, split='train'):
        self.opt = opt
        self.db = StormDataset(opt, split=split)
        self.tsf = Transform(opt.min_size, opt.max_size, bool_img_only=False)

    def __getitem__(self, idx):
        ori_img, points, labels = self.db.get_example(idx)
        if ori_img.shape[0] == 3:
            img, points, labels, scale = self.tsf((ori_img, points, labels))
            return img.copy(), points.copy(), labels.copy(), scale
        else:
            return ori_img.copy(), points.copy(), labels.copy(), 1

    def __len__(self):
        return len(self.db)


class DatasetGeneral:
    def __init__(self, opt, split='val_all'):
        self.opt = opt
        if split == 'inference':
            self.db = ModelDataset(opt.inference_dir, opt.inference_annotation_dir, opt.split_dir, 
                split=split, bool_img_only=False)
        else:
            self.db = ModelDataset(opt.data_dir, opt.annotation_dir, opt.split_dir, 
                 split=split, bool_img_only=False)
        self.tsf = Transform(opt.min_size, opt.max_size, bool_img_only=True)

    def __getitem__(self, idx):
        ori_img, points, labels, img_name = self.db.get_example(idx)
        img, scale = self.tsf((ori_img))

        if len(labels):
            points_new = (points*scale).copy()
            labels_new = labels.copy()
        else:
            points_new = np.zeros((0,2)).astype(np.float32)
            labels_new = np.zeros((0,)).astype(np.int32)
        return img.copy(), points_new, labels_new, scale, img_name

    def __len__(self):
        return len(self.db)
