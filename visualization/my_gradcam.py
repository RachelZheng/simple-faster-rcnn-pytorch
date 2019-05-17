## custom the gradcam

import os
import copy
import numpy as np
from PIL import Image

from skimage import transform as sktsf
import torch as t
from torchvision import transforms as tvtsf

from trainer import FasterRCNNTrainer
from model import FasterRCNNVGG16
from model.faster_rcnn import FasterRCNN
from utils import array_tool as at


MEAN_IMG = [3.2187, 3.2187, 3.2187]
STD_IMG = [13.6640, 13.6640, 13.6640]

def get_example_params(example_index):
    """
        Gets used variables for almost all visualizations, like the image, model etc.

    Args:
        example_index (int): Image id to use from examples

    returns:
        original_image (numpy arr): Original image read from the file
        prep_img (numpy_arr): Processed image
        target_class (int): Target class for the image
        file_name_to_export (string): File name to export the visualizations
        pretrained_model(Pytorch model): Model to use for the operations
    """
    # Pick one of the examples
    example_list = (('input_img/n0r_200801081245.png', 0),)
    img_path = example_list[example_index][0]
    target_class = example_list[example_index][1]
    file_name_to_export = img_path[img_path.rfind('/')+1:img_path.rfind('.')]
    # Read image
    original_image = np.array(Image.open(img_path).convert('RGB'))
    # Process image
    prep_img = preprocess_image(original_image)
    # Define model
    # pretrained_model = models.alexnet(pretrained=True)
    pretrained_model = FasterRCNNTrainer(FasterRCNNVGG16(n_fg_class=1)).cuda()
    pretrained_model.load(
        '/pylon5/ir5fp5p/xzheng4/test_pytorch/simple-faster-rcnn-pytorch/first_round_results/checkpoints/layer10/fasterrcnn_04302305_4_0.67_1.00')
    return (at.totensor(original_image),
            prep_img,
            target_class,
            file_name_to_export,
            pretrained_model)


def preprocess_image(img):
    img2 = img.transpose(2, 0, 1)
    img_resize = sktsf.resize(img2, (3, 512, 512))
    normalize = tvtsf.Normalize(mean=MEAN_IMG, std=STD_IMG)
    img_resize = normalize(t.from_numpy(img_resize).float())
    return img_resize



class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        for module_pos, module in enumerate(self.model.faster_rcnn._modules['extractor']):
            x = module(x)  # Forward
            if int(module_pos) == (self.target_layer - 1):
                x.register_hook(self.save_gradient)
                conv_output = x  # Save the convolution output on that layer

        # get the bbox of the model
        rpn_locs, rpn_scores, rois, roi_indices, anchor = \
            self.model.faster_rcnn.rpn(x, [600, 600], 6/26)
        roi_cls_locs, roi_scores = self.model.faster_rcnn.head(
            x, rois, roi_indices)
        return conv_output, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, features = self.forward_pass_on_convolutions(x)
        """
        features = features.view(features.size(0), -1)  # Flatten
        # Forward pass on the classifier
        x = self.model.faster_rcnn.head.classifier(features)
        locs = self.faster_rcnn.head.cls_loc(x)
        score = self.faster_rcnn.head.score(x)
        """
        return conv_output, features


class GradCam():
    """
        Produces class activation map
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, input_image, points):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the output of the locations
        conv_output, locs = self.extractor.forward_pass(input_image)
        # Get hooked gradients
        self.model.faster_rcnn.extractor.zero_grad()
        self.model.faster_rcnn.rpn.zero_grad()
        self.model.faster_rcnn.head.zero_grad()

        guided_gradients = self.extractor.gradients.data.numpy()[0]
        # Get convolution outputs
        target = conv_output.data.numpy()[0]
        # Get weights from gradients
        weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                       input_image.shape[3]), Image.ANTIALIAS))/255
        # ^ I am extremely unhappy with this line. Originally resizing was done in cv2 which
        # supports resizing numpy matrices with antialiasing, however,
        # when I moved the repository to PIL, this option was out of the window.
        # So, in order to use resizing with ANTIALIAS feature of PIL,
        # I briefly convert matrix to PIL image and then back.
        # If there is a more beautiful way, do not hesitate to send a PR.
        return cam
