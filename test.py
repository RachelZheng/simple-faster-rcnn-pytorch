import numpy as np
import pickle, sys, os, glob
import tensorflow as tf
from collections import defaultdict

tags = ['rpn_cls_loss', 'roi_cls_loss', 'total_loss']

layer = 10 
folder = '/pylon5/ir5fp5p/xzheng4/test_pytorch/simple-faster-rcnn-pytorch/logs/layer%d/'%(layer)
os.chdir(folder)
files = glob.glob('events.out.tfevents.*')
files.sort(key=lambda x: os.path.getmtime(x))

dict_val = defaultdict(list)

for f in files:
	s = tf.train.summary_iterator(f)
	for e in s:
		for v in e.summary.value:
			tag_char = ''.join(i for i in v.tag if not i.isdigit())
			if 'img' not in v.tag and tag_char in tags:
				dict_val[v.tag].append(v.simple_value)

pickle.dump(dict_val, open(folder + 'all_dict.p', 'wb'))