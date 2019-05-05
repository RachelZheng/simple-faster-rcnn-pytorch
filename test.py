import numpy as np
import pickle, sys, os, glob
import tensorflow as tf
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

tags = ['rpn_cls_loss', 'roi_cls_loss', 'total_loss']

layer = 10
folder = '/pylon5/ir5fp5p/xzheng4/test_pytorch/simple-faster-rcnn-pytorch/logs/layer%d/previous_round/'%(layer)
os.chdir(folder)
files = glob.glob('events.out.tfevents.*')
files.sort(key=lambda x: os.path.getmtime(x))

dict_val = defaultdict(list)

for f in files:
	s = tf.train.summary_iterator(f)
	try:
		for e in s:
			for v in e.summary.value:
				tag_char = ''.join(i for i in v.tag if not i.isdigit())
				if 'img' not in v.tag and tag_char in tags:
					dict_val[v.tag].append(v.simple_value)
					pickle.dump(dict_val, open(folder + 'dict.p', 'wb'))
	except:
		dict_val = pickle.load(open(folder + 'dict.p', 'rb'))


## plot the loss
dict_val = pickle.load(open(
	'/pylon5/ir5fp5p/xzheng4/test_pytorch/simple-faster-rcnn-pytorch/logs/layer10/previous_round/all_dict.p', 'rb'))


n_epoch_iter = 1178
rpn_loss_, roi_loss_, total_loss_ = [], [], []
for n_epoch in range(0,13):
	rpn_loss_ += dict_val['rpn_cls_loss' + str(n_epoch)]
	roi_loss_ += dict_val['roi_cls_loss' + str(n_epoch)]
	total_loss_ += dict_val['total_loss' + str(n_epoch)]

# plot losses
