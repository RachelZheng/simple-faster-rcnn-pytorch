## test the visualization of the pretrained model
import pickle, os, six, cv2
import numpy as np
from tqdm import tqdm
import torch as t
from torch import nn
from torchvision.models import vgg16
from torch.utils import data as data_
from collections import namedtuple

from misc_functions import save_class_activation_images
from my_misc_functions import get_example_params
from my_gradcam import CamExtractor, GradCam


target_example = 0  # Snake
(original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
    get_example_params(target_example)
# Grad cam
grad_cam = GradCam(pretrained_model, target_layer=30)
# Generate cam mask
cam = grad_cam.generate_cam(prep_img, target_class)
# Save mask
save_class_activation_images(original_image, cam, file_name_to_export)
print('Grad cam completed')
