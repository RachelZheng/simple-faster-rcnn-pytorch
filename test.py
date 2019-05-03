import numpy as np
import ipdb
import tensorflow as tf

folder = '/pylon5/ir5fp5p/xzheng4/test_pytorch/simple-faster-rcnn-pytorch/logs/layer10/'
s = tf.train.summary_iterator(folder + 
	"events.out.tfevents.1556765537.gpu050.pvt.bridges.psc.edu")

for item in s:
	ipdb.set_trace()
	print(item)
