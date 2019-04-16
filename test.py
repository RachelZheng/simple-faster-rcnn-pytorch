## test files for testing roi generation

from model.utils.creator_tool import AnchorTargetCreator, ProposalTargetCreator
import torch as t
from torch import nn
from torchvision.models import vgg16
from model.region_proposal_network import RegionProposalNetwork
from model.faster_rcnn import FasterRCNN
from model.faster_rcnn_vgg16 import decom_vgg16
from model.roi_module import RoIPooling2D
from utils import array_tool as at
from utils.config import opt

proposal_target_creator = ProposalTargetCreator()
extractor, classifier = decom_vgg16()

# get an example of the dataset
dataset = Dataset(opt)
logger = Logger('./logs')
dataloader = data_.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=opt.num_workers)
dataloader_iterator = iter(dataloader)
test = next(dataloader_iterator)
img = test[0]
bbox = test[1]
scale = test[-1].numpy()[0]

features = extractor(img)
_, _, H, W = img.shape
img_size = (H, W)

rpn = RegionProposalNetwork(512, 512, ratios=[0.5, 1, 2],anchor_scales=[8, 16, 32],feat_stride=16)
rpn_locs, rpn_scores, rois, roi_indices, anchor = rpn(features, img_size, scale)



sample_roi, gt_roi_loc, gt_roi_label = proposal_target_creator(
            roi, at.tonumpy(bbox),
            at.tonumpy(label),
            loc_normalize_mean,
            loc_normalize_std)


