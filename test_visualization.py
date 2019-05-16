## test the visualization of the pretrained model
import pickle, os, six, cv2
import numpy as np
from tqdm import tqdm
import torch as t
from torch import nn
from torchvision.models import vgg16
from torch.utils import data as data_
from collections import namedtuple

from model import FasterRCNNVGG16
from model.region_proposal_network import RegionProposalNetwork
from model.faster_rcnn import FasterRCNN
from model.faster_rcnn_vgg16 import decom_vgg16
from model.roi_module import RoIPooling2D
from model.utils.creator_tool_pts import AnchorPointTargetCreator, ProposalPointTargetCreator

from my_misc_functions import get_example_params
from gradcam import CamExtractor, GradCam

target_example = 0  # Snake
(original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
    get_example_params(target_example)
# Grad cam
grad_cam = GradCam(pretrained_model, target_layer=11)
# Generate cam mask
cam = grad_cam.generate_cam(prep_img, target_class)
# Save mask
save_class_activation_images(original_image, cam, file_name_to_export)
print('Grad cam completed')



logger = Logger('./logs')
faster_rcnn = FasterRCNNVGG16(n_fg_class=1)
print('model construct completed')
trainer = FasterRCNNTrainer(faster_rcnn).cuda()
trainer.load(os.path.join(opt.model_dir, opt.model_name))
## trainer.load(os.path.join(opt.model_dir, opt.model_name))
dataset = Dataset(opt, split='val')
dataloader = data_.DataLoader(dataset, \
                              batch_size=1, \
                              shuffle=True, \
                              num_workers=opt.num_workers)

img, points_, labels_, scale = dataset.__getitem__(21315)


dataloader_iterator = iter(dataloader)
img, points_, labels_, scale = next(dataloader_iterator)