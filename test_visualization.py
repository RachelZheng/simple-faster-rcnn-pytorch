## test the visualization of the pretrained model
import pickle, os, six, cv2
import numpy as np
from tqdm import tqdm
import torch as t
from PIL import Image
import xml.etree.ElementTree as ET

from torch import nn
from torchvision.models import vgg16
from torch.utils import data as data_
from collections import namedtuple
from torch.autograd import Variable, Function

from trainer import FasterRCNNTrainer
from model import FasterRCNNVGG16
from data.storm_dataset import imgname2idx

"""
from misc_functions import save_class_activation_images
from my_gradcam import CamExtractor, GradCam, preprocess_image
"""
from visualization.my_gradcam import preprocess_image
from visualization.grad-cam import GradCam

# ---- load the model ----
# Grad cam
pretrained_model = FasterRCNNTrainer(FasterRCNNVGG16(n_fg_class=1))
pretrained_model.load(
        '/pylon5/ir5fp5p/xzheng4/test_pytorch/simple-faster-rcnn-pytorch/first_round_results/checkpoints/layer10/fasterrcnn_04302305_4_0.67_1.00')
grad_cam = GradCam(pretrained_model, target_layer=30)

# ------- load the image ----
name_img = 'n0r_200801081245.png'
original_image = Image.open('input_img/' + name_img).convert('RGB')
file_name_to_export = 'input_img/out.png'
prep_img = preprocess_image(np.array(original_image)).unsqueeze_(0)
input = Variable(prep_img, requires_grad = True)

"""
# ------- load the label points ---- 
name_xml = 'input_img/%07d.xml'%(imgname2idx(name_img))
anno = ET.parse(name_xml).getroot()
points = list()

## add all the points into the dataset 
for obj in anno.findall('point'):
	points.append([int(obj.find(tag).text) for tag in ('y', 'x')])

points = np.stack(points).astype(np.float32)


# Generate cam mask
cam = grad_cam.generate_cam(prep_img, '0')
# Save mask
save_class_activation_images(original_image, cam, file_name_to_export)
print('Grad cam completed')

x = prep_img
for module_pos, module in enumerate(pretrained_model.faster_rcnn._modules['extractor']):
    x = module(x)  # Forward
    if int(module_pos) == 29:
        x.register_hook(extract)
        conv_output = x  # Save the convolution output on that layer
"""
model2 = vgg16(pretrained=True)

grad_cam = GradCam(model = models.vgg19(pretrained=True), \
					target_layer_names = ["35"], use_cuda=args.use_cuda)


