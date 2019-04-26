## test files for testing roi generation
import pickle
import numpy as np

# from model.utils.creator_tool import AnchorTargetCreator, ProposalTargetCreator
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

from data.dataset import Dataset, TestDataset, inverse_normalize

from logger import Logger

LossTuple = namedtuple('LossTuple',
                       ['rpn_cls_loss',
                        'roi_cls_loss',
                        'total_loss'
                        ])

proposal_target_creator = ProposalPointTargetCreator()
extractor, classifier = decom_vgg16()

# get an example of the dataset
dataset = Dataset(opt, split='inference')
logger = Logger('./logs')
dataloader = data_.DataLoader(dataset, batch_size=1, 
	shuffle=False, num_workers=opt.num_workers)

dataloader_iterator = iter(dataloader)
test = next(dataloader_iterator)
img = test[0]

trainer = FasterRCNNTrainer(faster_rcnn).cuda()
trainer.load(os.path.join(opt.model_dir, opt.model_name))
pred_bboxes_, pred_labels_, pred_scores_ = trainer.faster_rcnn.predict(
            img, [img.shape[2:]])
