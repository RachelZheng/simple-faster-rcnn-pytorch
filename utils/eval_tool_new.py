## new evaluation tool for 
from collections import defaultdict

from model.utils.bbox_pts_tools import bbox_event
import itertools
import six
import numpy as np

from sklearn.metrics import average_precision_score, precision_recall_curve

def eval_detection(pred_bboxes, pred_labels, pred_scores, gt_pts, gt_labels):
    pred_bboxes = iter(pred_bboxes)
    pred_labels = iter(pred_labels)
    pred_scores = iter(pred_scores)
    gt_pts = iter(gt_pts)
    gt_labels = iter(gt_labels)

    pts_catch_score = defaultdict(list)
    # for every point record the maximum score of bbox
    bbox_catch_score = defaultdict(list)
    # for every bbox record the scores for catched and not catched bboxes
    bbox_total_score = defaultdict(list)
    # record all the bbox scores in descending order
    prec = defaultdict(list)
    rec = defaultdict(list)
    rec_pts = defaultdict(list)
    ap = defaultdict(float)

    for pred_bbox, pred_label, pred_score, gt_pt, gt_label in six.moves.zip(
        pred_bboxes, pred_labels, pred_scores, gt_pts, gt_labels):
        ## only consider single class for now
        for l in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):
            pred_mask_l = pred_label == l
            pred_bbox_l = pred_bbox[pred_mask_l]
            pred_score_l = pred_score[pred_mask_l]
            # sort by score in descending order
            order = pred_score_l.argsort()[::-1]
            pred_bbox_l = pred_bbox_l[order]
            pred_score_l = pred_score_l[order]

            gt_mask_l = gt_label == l
            gt_pt_l = gt_pt[gt_mask_l]

            if gt_pt_l.shape[0] == 0 and pred_bbox_l.shape[0] == 0:
                continue
            elif gt_pt_l.shape[0] == 0:
                bbox_catch_score[l] += [0] * pred_bbox_l.shape[0]
            elif pred_bbox_l.shape[0] == 0:
                pts_catch_score[l] += [0] * gt_pt_l.shape[0]
            else:
                match_score = bbox_event(pred_bbox_l, pred_score_l, gt_pt_l)
                pts_catch_score[l] += np.max(match_score, axis=0).tolist()
                bbox_catch_score[l] += np.max(match_score, axis=1).tolist()
            bbox_total_score[l] += pred_score_l.tolist()

    del pred_bboxes, pred_labels, pred_scores, gt_pts, gt_labels

    ## set all the scores below threshold as 0, compute prec and rec
    for l in pts_catch_score:
        pts_catch_score_l = np.array(pts_catch_score[l])
        bbox_catch_score_l = np.array(bbox_catch_score[l])
        bbox_total_score_l = np.array(bbox_total_score[l])
        n_pts = max(len(pts_catch_score_l),1)
        bbox_catch_label = (bbox_catch_score_l > 0).astype(int)

        prec[l], rec[l], thres = precision_recall_curve(bbox_catch_label, bbox_total_score_l)
        ap[l] = average_precision_score(bbox_catch_label, bbox_total_score_l)
        for t in thres:
            rec_pts[l].append(len(np.where(bbox_catch_score_l >= t)[0]) / n_pts)

        rec_pts[l].append(0)
        rec_pts[l] = np.array(rec_pts[l])

    del pts_catch_score, bbox_catch_score, bbox_total_score

    return {'prec': prec, 'rec': rec, 'ap': ap, 'rec_pts': rec_pts}
