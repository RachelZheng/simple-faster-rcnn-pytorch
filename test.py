import numpy as np
import pickle, sys, os
import tensorflow as tf
from collections import defaultdict


folder = '/pylon5/ir5fp5p/xzheng4/test_pytorch/simple-faster-rcnn-pytorch/logs/layer10/'
s = tf.train.summary_iterator(folder + 
	"events.out.tfevents.1556765537.gpu050.pvt.bridges.psc.edu")

dict_val = defaultdict(list)
for e in s:
	for v in e.summary.value:
		if 'img' not in v.tag:
			dict_val[v.tag].append(v.simple_value)

## save the datastructure into the folder
pickle.dump(open(), )