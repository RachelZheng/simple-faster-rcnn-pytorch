import numpy as np
import pickle, sys, os, glob
import tensorflow as tf
from collections import defaultdict
import pandas as pd
import seaborn as sns

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


# n_epoch_iter = 1168
n, n1, n2 = 0, 3, 5
rpn_loss_, roi_loss_, total_loss_, x_epoch = [], [], [], [0]
rpn_cls_loss, roi_cls_loss, total_loss = [],[],[]
for n_epoch in np.concatenate([np.arange(n1),np.arange(n2, 13)]):
	rpn_loss_ += dict_val['rpn_cls_loss' + str(n_epoch)]
	roi_loss_ += dict_val['roi_cls_loss' + str(n_epoch)]
	total_loss_ += dict_val['total_loss' + str(n_epoch)]
	x_epoch.append(len(rpn_loss_))
	rpn_cls_loss.append(dict_val['rpn_cls_loss' + str(n_epoch)][0])
	roi_cls_loss.append(dict_val['roi_cls_loss' + str(n_epoch)][0])
	total_loss.append(dict_val['total_loss' + str(n_epoch)][0])

x = np.arange(len(rpn_loss_))
rpn_cls_loss.append(rpn_loss_[-1])
roi_cls_loss.append(roi_loss_[-1])
total_loss.append(total_loss_[-1])

"""
## plot with time
df = pd.DataFrame(columns=['time', 'loss_type', 'loss_val'])

for i in range(len(rpn_loss_)):
	df = df.append({'time': x[i], 'loss_type':'RPN_loss', 'loss_val':rpn_loss_[i]}, ignore_index=True)
	df = df.append({'time': x[i], 'loss_type':'ROI_loss', 'loss_val':roi_loss_[i]}, ignore_index=True)
	df = df.append({'time': x[i], 'loss_type':'total_loss', 'loss_val':total_loss_[i]}, ignore_index=True)

ax = sns.lineplot(x="time", y="loss_val", hue="loss_type", data=df)
fig = ax.get_figure()
fig.savefig("/pylon5/ir5fp5p/xzheng4/test_pytorch/simple-faster-rcnn-pytorch/logs/layer10/loss.png")


# plot losses
rpn_cls_loss = dict_val['rpn_cls_loss'][:n1] + dict_val['rpn_cls_loss'][n2:]
roi_cls_loss = dict_val['roi_cls_loss'][:n1] + dict_val['roi_cls_loss'][n2:]
total_loss = dict_val['total_loss'][:n1] + dict_val['total_loss'][n2:]
"""

plt.figure()
fig, ax1 = plt.subplots()
ax1.plot(x, rpn_loss_, color='r', label='RPN loss')
ax1.plot(x, roi_loss_, color='g', label='ROI loss')
ax1.plot(x, total_loss_, color='b', label='total loss')
ax1.plot(x_epoch, rpn_cls_loss, color='lightsalmon')
ax1.plot(x_epoch, roi_cls_loss, color='greenyellow')
ax1.plot(x_epoch, total_loss, color='skyblue')
ax1.legend(loc='upper right')
ax1.set_xlabel('Time')
ax1.set_ylabel('Loss')
plt.savefig('/pylon5/ir5fp5p/xzheng4/test_pytorch/simple-faster-rcnn-pytorch/logs/layer10/loss_10.png')
