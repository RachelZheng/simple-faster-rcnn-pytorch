## creator tools for 
import numpy as np
import cupy as cp

# from model.utils.bbox_tools import bbox2loc, bbox_iou, loc2bbox
from model.utils.bbox_pts_tools import bbox_score_intense_event
from model.utils.nms import non_maximum_suppression

class ProposalPointTargetCreator(object):
    """ Assign ground truth points to given RoIs.
    """
    def __init__(self,
                 n_sample=128,
                 pos_ratio=0.25, 
                 pos_score_thresh=1,
                 neg_score_thresh_hi=3,
                 neg_score_thresh_lo=0.5
                 ):
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio

        ## assign pre-defined threshold
        self.pos_score_thresh = pos_score_thresh
        self.neg_score_thresh_hi = neg_score_thresh_hi
        self.neg_score_thresh_lo = neg_score_thresh_lo
       

    def __call__(self, img, roi, points, label,
                 loc_normalize_mean=(0., 0., 0., 0.),
                 loc_normalize_std=(0.1, 0.1, 0.2, 0.2),
                 method='intensity_event_per_pixel'):
        """ assign the ground-truth to the sampled proposals

        The function samples total of n_samples RoIs from total RoIs.
        for each RoI we compute a score that weights the relationships between 
        storm events, pixels, and reflection intensity.

        Losses includes:
        `intensity_event_per_pixel`: n_event * Relu(avg_intensity) / n_pixel

        TBD: find the best loss function

        Args:
            img (array): intensity after normalization. with approx. 
                range [-1,1] for pytorch.
            roi (array): Region of Interests (RoIs) from which we sample.
                Its shape is :math:`(R, 4)`
            bbox (array): The coordinates of ground truth bounding boxes.
                Its shape is :math:`(R', 4)`.
            label (array): Ground truth bounding box labels. Its shape
                is :math:`(R',)`. Its range is :math:`[0, L - 1]`, where
                :math:`L` is the number of foreground classes.
            loc_normalize_mean (tuple of four floats): Mean values to normalize
                coordinates of bouding boxes.
            loc_normalize_std (tupler of four floats): Standard deviation of
                the coordinates of bounding boxes.
        """
        pos_roi_per_image = np.round(self.n_sample * self.pos_ratio)

        ## manually create some bboxes
        _, H, W = img.shape
        # bbox = self.create_bbox(points, img_size=(H, W))
        # roi = np.concatenate((roi, bbox), axis=0)


        if method == 'intensity_event_per_pixel':
            score_perpix, intensity_perpix, cnt_exten = bbox_score_intense_event(
                img, roi, points)
            ## other methods TBD

        # select positive objects
        pos_index = np.where(score_perpix >= self.pos_score_thresh)[0]
        pos_roi_per_this_image = int(min(pos_roi_per_image, pos_index.size))
        if pos_index.size > 0:
            pos_index = np.random.choice(
                pos_index, size=pos_roi_per_this_image, replace=False)

        # Select background RoIs as whose intensity within
        # [neg_iou_thresh_lo, neg_iou_thresh_hi) and score == 0
        neg_index = np.where((cnt_exten == 0) & (intensity_perpix > self.neg_score_thresh_lo) &
            (intensity_perpix < self.neg_score_thresh_hi))[0]
        neg_roi_per_this_image = self.n_sample - pos_roi_per_this_image
        neg_roi_per_this_image = int(min(neg_roi_per_this_image, neg_index.size))

        if neg_index.size > 0:
            neg_index = np.random.choice(
                neg_index, size=neg_roi_per_this_image, replace=False)

        # The indices that we're selecting (both positive and negative).
        keep_index = np.append(pos_index, neg_index)

        # for 2 classes cases, just make labels to be 1 and 0
        gt_roi_label = np.ones((pos_roi_per_this_image + neg_roi_per_this_image,)).astype(int)
        gt_roi_label[pos_roi_per_this_image:] = 0  # negative labels --> 0
        sample_roi = roi[keep_index]

        return sample_roi, gt_roi_label


    def create_bbox(self, points, img_size, sizes=(100,200,250,280,300)):
        """ manually create bboxes from ground_truth 
        """
        pass 

class AnchorPointTargetCreator(object):
    """Assign the ground truth bounding boxes to anchors.

    Assigns the ground truth bounding boxes to anchors for training Region
    Proposal Networks introduced in Faster R-CNN [#]_.

    Offsets and scales to match anchors to the ground truth are
    calculated using the encoding scheme of
    :func:`model.utils.bbox_tools.bbox2loc`.

    .. [#] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.

    Args:
        n_sample (int): The number of regions to produce.
        pos_iou_thresh (float): Anchors with IoU above this
            threshold will be assigned as positive.
        neg_iou_thresh (float): Anchors with IoU below this
            threshold will be assigned as negative.
        pos_ratio (float): Ratio of positive regions in the
            sampled regions.

    """

    def __init__(self,
                 n_sample=256,
                 pos_score_thresh=1, 
                 neg_score_thresh_hi=3,
                 neg_score_thresh_lo=0.5,
                 pos_ratio=0.5):
        self.n_sample = n_sample
        self.pos_score_thresh = pos_score_thresh
        self.neg_score_thresh_hi = neg_score_thresh_hi
        self.neg_score_thresh_lo = neg_score_thresh_lo
        self.pos_ratio = pos_ratio

    def __call__(self, img, points, anchor, img_size):
        """Assign ground truth supervision to sampled subset of anchors.

        Types of input arrays and output arrays are same.

        Here are notations.

        * :math:`S` is the number of anchors.
        * :math:`R` is the number of bounding boxes.

        Args:
            bbox (array): Coordinates of bounding boxes. Its shape is
                :math:`(R, 4)`.
            anchor (array): Coordinates of anchors. Its shape is
                :math:`(S, 4)`.
            img_size (tuple of ints): A tuple :obj:`H, W`, which
                is a tuple of height and width of an image.

        Returns:
            (array, array):

            #NOTE: it's scale not only  offset
            * **loc**: Offsets and scales to match the anchors to \
                the ground truth bounding boxes. Its shape is :math:`(S, 4)`.
            * **label**: Labels of anchors with values \
                :obj:`(1=positive, 0=negative, -1=ignore)`. Its shape \
                is :math:`(S,)`.

        """

        img_H, img_W = img_size

        n_anchor = len(anchor)
        inside_index = _get_inside_index(anchor, img_H, img_W)
        anchor = anchor[inside_index]
        label = self._create_label(img, inside_index, anchor, points)

        # map up to original set of anchors
        label = _unmap(label, n_anchor, inside_index, fill=-1)

        return label


    def _create_label(self, img, inside_index, anchor, points):
        # label: 1 is positive, 0 is negative, -1 is dont care
        label = np.empty((len(inside_index),), dtype=np.int32)
        label.fill(-1)

        score_perpix, intensity_perpix, cnt_exten = bbox_score_intense_event(
            img, anchor, points)

        # assign positive examples
        label[score_perpix > self.pos_score_thresh] = 1

        # assign negative examples
        idx_neg = np.where(
            (cnt_exten == 0) & 
            (intensity_perpix > self.neg_score_thresh_lo) &
            (intensity_perpix < self.neg_score_thresh_hi)
            )[0]
        label[idx_neg] = 0

        # subsample positive labels if we have too many
        n_pos = int(self.pos_ratio * self.n_sample)
        pos_index = np.where(label == 1)[0]
        if len(pos_index) > n_pos:
            disable_index = np.random.choice(
                pos_index, size=(len(pos_index) - n_pos), replace=False)
            label[disable_index] = -1

        # subsample negative labels if we have too many
        n_neg = self.n_sample - np.sum(label == 1)
        neg_index = np.where(label == 0)[0]
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(
                neg_index, size=(len(neg_index) - n_neg), replace=False)
            label[disable_index] = -1

        return label


def _unmap(data, count, index, fill=0):
    # Unmap a subset of item (data) back to the original set of items (of
    # size count)

    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=data.dtype)
        ret.fill(fill)
        ret[index] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=data.dtype)
        ret.fill(fill)
        ret[index, :] = data
    return ret


def _get_inside_index(anchor, H, W):
    # Calc indicies of anchors which are located completely inside of the image
    # whose size is speficied.
    index_inside = np.where(
        (anchor[:, 0] >= 0) &
        (anchor[:, 1] >= 0) &
        (anchor[:, 2] < H) &
        (anchor[:, 3] < W)
    )[0]
    return index_inside