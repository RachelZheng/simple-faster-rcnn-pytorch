## test the visualization of the pretrained model
import pickle, os, six, cv2
import numpy as np
from tqdm import tqdm
import torch as t
from PIL import Image

from torch import nn
from torchvision.models import vgg16
from torch.utils import data as data_
from collections import namedtuple

from trainer import FasterRCNNTrainer
from model import FasterRCNNVGG16

from misc_functions import save_class_activation_images
from my_misc_functions import preprocess_image
from my_gradcam import CamExtractor, GradCam

# ---- load the model ----
# Grad cam
pretrained_model = FasterRCNNTrainer(FasterRCNNVGG16(n_fg_class=1))
pretrained_model.load(
        '/pylon5/ir5fp5p/xzheng4/test_pytorch/simple-faster-rcnn-pytorch/first_round_results/checkpoints/layer10/fasterrcnn_04302305_4_0.67_1.00')
grad_cam = GradCam(pretrained_model, target_layer=30)

# --- load the image ----
original_image = Image.open('input_img/n0r_200801081245.png').convert('RGB')
file_name_to_export = 'input_img/out.png'
prep_img = preprocess_image(np.array(original_image)).unsqueeze_(0)

# Generate cam mask
cam = grad_cam.generate_cam(prep_img, '0')
# Save mask
save_class_activation_images(original_image, cam, file_name_to_export)
print('Grad cam completed')


