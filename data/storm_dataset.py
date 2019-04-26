import os, warnings
import xml.etree.ElementTree as ET
from torchvision import transforms as T

import numpy as np

from .util import read_image


class StormDataset:
	def  __init__(self, data_dir, annotation_dir, split_dir,
		sub_dataset='all', split='trainval'):
		if sub_dataset == 'all' or split == 'inference':
			id_list_file = os.path.join(
				split_dir, '{}.txt'.format(split))
		else:
			id_list_file = os.path.join(
				split_dir, '{}_{}.txt'.format(sub_dataset, split))

		self.ids = [int(id_.split('.')[0]) for id_ in open(id_list_file)]
		self.data_dir = data_dir
		self.annotation_dir = annotation_dir
		self.split_dir = split_dir
		self.sub_dataset = sub_dataset
		self.label_names = STORM_LABEL_NAMES
		self.bool_inference = split == 'inference'

	def __len__(self):
		return len(self.ids)

	def inference_idx2imgname(self, idx):
		""" convert the image index to the image name
		"""
		c_mon, c_day = 1488, 48 # 48 simulations per day
		mon_ = idx // c_mon + 1
		day_ = (idx % c_mon) // c_day + 1
		sim_ = idx % c_day + 1
		img_name = 'diags_d02_2017%02d%02d00_mem_10_f0%02d.png'%(mon_, day_, sim_)
		return img_name

	def get_example(self, i):
		""" get the i-th example
		"""
		id_img = self.ids[i]
		points = list()
		labels = list()

		if self.bool_inference:
			name_img = os.path.join(self.data_dir, self.inference_idx2imgname(id_img))
			img = read_image(name_img, color=True)

		else:		
			name_xml = os.path.join(self.annotation_dir, '{:07d}.xml'.format(id_img))
			anno = ET.parse(name_xml).getroot()
			img = read_image(os.path.join(self.data_dir, anno.find("filename").text), color=True)
			
			## add all the points into the dataset 
			for obj in anno.findall('point'):
				"""
				label_type = obj.find('event_type').text
				if (self.sub_dataset != 'all') and (self.sub_dataset != label_type):
					continue
				"""
				points.append([int(obj.find(tag).text) for tag in ('y', 'x')])
				# labels.append(STORM_LABEL_NAMES.index(label_type))
				## in this case labels are all 0
				labels.append(0)

		points = np.stack(points).astype(np.float32)
		labels = np.stack(labels).astype(np.int32)

		return img, points, labels


	__getitem__ = get_example



STORM_LABEL_NAMES = ('all')