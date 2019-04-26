from __future__ import absolute_import
from __future__ import division
import torch as t
from skimage import transform as sktsf
from torchvision import transforms as tvtsf
from data import util
import numpy as np
from utils.config import opt

# add personal dataset
from data.storm_dataset import StormDataset

# MEAN_IMG = [122.7717, 115.9465, 102.9801]
# STD_IMG = [58.395, 57.12 , 57.375]

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

    def __init__(self, min_size=600, max_size=1000):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, in_data):
        img, points, labels = in_data
        _, H, W = img.shape
        img = preprocess(img, self.min_size, self.max_size)
        _, o_H, o_W = img.shape
        scale = o_H / H
        points = util.resize_pts(points, (H, W), (o_H, o_W))

        # horizontally flip
        img, params = util.random_flip(img, x_random=True, return_param=True)
        points = util.flip_pts(points, (o_H, o_W), x_flip=params['x_flip'])

        # keep the points within the range
        points[:,0] = np.clip(points[:,0], 1, o_H)
        points[:,1] = np.clip(points[:,1], 1, o_W)
        
        return img, points, labels, scale


class Dataset:
    def __init__(self, opt, split='train'):
        self.opt = opt
        self.db = StormDataset(opt.data_dir, opt.annotation_dir, opt.split_dir, split=split)
        self.tsf = Transform(opt.min_size, opt.max_size)

    def __getitem__(self, idx):
        ori_img, points, labels = self.db.get_example(idx)
        img, points, labels, scale = self.tsf((ori_img, points, labels))

        # TODO: check whose stride is negative to fix this instead copy all
        # some of the strides of a given numpy array are negative.
        return img.copy(), points.copy(), labels.copy(), scale

    def __len__(self):
        return len(self.db)



class TestDataset:
    def __init__(self, opt, split='test'):
        self.opt = opt
        self.db = StormDataset(opt.data_dir, opt.annotation_dir, opt.split_dir, split=split)
        self.tsf = Transform(opt.min_size, opt.max_size)

    def __getitem__(self, idx):
        ori_img, points, labels = self.db.get_example(idx)

        # img = preprocess(ori_img)
        # return img, ori_img.shape[1:], points, labels
        img, points, labels, scale = self.tsf((ori_img, points, labels))
        return img.copy(), points.copy(), labels.copy(), scale

    def __len__(self):
        return len(self.db)
