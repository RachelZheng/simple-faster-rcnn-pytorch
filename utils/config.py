from pprint import pprint
import os

# Default Configs for training
# NOTE that, config items could be overwriten by passing argument through command line.
# e.g. --voc-data-dir='./data/'

class Config:
    # data
    prefixes = ('/pylon5/ir5fp5p/xzheng4/data_meteo/', 
        '/oasis/projects/nsf/pen150/xinye/data_meteo/')
    prefixes_pkg = ('/pylon5/ir5fp5p/xzheng4/test_pytorch/simple-faster-rcnn-pytorch/',
        '/oasis/projects/nsf/pen150/xinye/meteology/simple-faster-rcnn-pytorch/')

    idx = 0
    prefix = prefixes[idx]
    prefix_pkg = prefixes_pkg[idx]
    data_dir = os.path.join(prefix, 'ref_grayscale/')   # radar observation dir

    bool_train_one_hour = False  ## train the model for one hour images
    if bool_train_one_hour:
        annotation_dir = os.path.join(prefix, 'ref_dataset/Annotations_tracking/')
    else:
        annotation_dir = os.path.join(prefix, 'ref_dataset/Annotations/')

    split_dir = os.path.join(prefix, 'ref_dataset/Data_split/')

    ## model inference data
    inference_annotation_dir = os.path.join(
        prefix, 'ref_dataset/Annotations_model_pts/')
    inference_dir = os.path.join(prefix, 'model_ref_new/')
    # inference_out_dir = os.path.join(prefix, 'model_inference_result/')
    eval_dir = os.path.join(prefix, 'ref_dataset/eval/')
    n_layer_fix = 10 ## number of fixed layer in cnn
    model_dir = os.path.join(prefix_pkg, 
        'checkpoints/layer{}/1hr_{:d}/'.format(n_layer_fix, bool_train_one_hour))
    logger_dir = os.path.join(prefix_pkg, 
        'logs/layer{}/1hr_{:d}/'.format(n_layer_fix, bool_train_one_hour))
    model_name = 'fasterrcnn_04302305_4_0.67_1.00'

    min_size = 600  # image resize
    max_size = 1000 # image resize
    num_workers = 8
    test_num_workers = 128
    plot_every = 100
    
    # sigma for l1_smooth_loss
    rpn_sigma = 3.
    roi_sigma = 1.

    # param for optimizer
    # 0.0005 in origin paper but 0.0001 in tf-faster-rcnn
    weight_decay = 0.0005
    lr_decay = 0.1  # 1e-3 -> 1e-4
    lr = 1e-3

    # preset
    pretrained_model = 'vgg16'

    # training
    epoch = 12


    use_adam = False # Use Adam optimizer
    use_chainer = False # try match everything as chainer
    use_drop = False # use dropout in RoIHead

    test_num = 1000
    # model
    load_path = None
    bool_load_model = False

    caffe_pretrain = False # use caffe pretrained model instead of torchvision

    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}


opt = Config()
