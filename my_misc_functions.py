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
    example_list = (('input_img/n0r_200801081245.png', 0),
        )
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
    return (original_image,
            prep_img,
            target_class,
            file_name_to_export,
            pretrained_model)


def preprocess_image(img):
    img2 = img.transpose(2, 0, 1)
    img_resize = sktsf.resize(img2
        , (3, 512, 512))
    normalize = tvtsf.Normalize(mean=MEAN_IMG, std=STD_IMG)
    img_resize = normalize(t.from_numpy(img_resize).float())
    return img_resize.numpy()


if __name__ == '__main__':
    # from my_misc_functions import get_example_params, preprocess_image
    img = Image.open('/pylon5/ir5fp5p/xzheng4/test_pytorch/simple-faster-rcnn-pytorch/input_img/n0r_200801081245.png'
        ).convert('RGB')
    img_resized = preprocess_image(np.float32(img))