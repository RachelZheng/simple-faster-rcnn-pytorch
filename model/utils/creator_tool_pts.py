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
        if method == 'intensity_event_per_pixel':
            score, intensity = bbox_score_intense_event(img, roi, points)
            ## other methods TBD

        # select positive objects
        pos_index = np.where(score >= pos_score_thresh)[0]
        pos_roi_per_this_image = int(min(pos_roi_per_image, pos_index.size))
        if pos_index.size > 0:
            pos_index = np.random.choice(
                pos_index, size=pos_roi_per_this_image, replace=False)

        # Select background RoIs as whose intensity within
        # [neg_iou_thresh_lo, neg_iou_thresh_hi) and score == 0
        neg_index = np.where((score == 0) & (intensity >= neg_score_thresh_lo) &
            (intensity <= neg_score_thresh_hi))[0]
        neg_roi_per_this_image = self.n_sample - pos_roi_per_this_image
            neg_roi_per_this_image = int(min(neg_roi_per_this_image, neg_index.size))

        if neg_index.size > 0:
            neg_index = np.random.choice(
                neg_index, size=neg_roi_per_this_image, replace=False)

        # The indices that we're selecting (both positive and negative).
        keep_index = np.append(pos_index, neg_index)
        gt_roi_label = gt_roi_label[keep_index]
        gt_roi_label[pos_roi_per_this_image:] = 0  # negative labels --> 0
        sample_roi = roi[keep_index]

        return sample_roi, gt_roi_label

    