import os, warnings
import xml.etree.ElementTree as ET
from torchvision import transforms as T

import numpy as np

from .util import read_image


class StormDataset:
	def  __init__(self, data_dir, annotation_dir, split_dir,
		sub_dataset='torn', split='trainval'):
		if sub_dataset == 'all':
			id_list_file = os.path.join(
				split_dir, '{}.txt'.format(name_split))
		else:
			id_list_file = os.path.join(
				split_dir, '{}_{}.txt'.format(sub_dataset, split))

		self.ids = [int(id_.split('.')[0]) for id_ in open(id_list_file)]
		self.data_dir = data_dir
		self.annotation_dir = annotation_dir
		self.split_dir = split_dir
		self.sub_dataset = sub_dataset
		self.label_names = STORM_LABEL_NAMES

	def __len__(self):
		return len(self.ids)


	def get_example(self, i):
		""" get the i-th example
		"""
		id_img = self.ids[i]
		name_xml = os.path.join(self.annotation_dir, '{:07d}.xml'.format(id_img))
		anno = ET.parse(name_xml).getroot()
		img = read_image(os.path.join(self.data_dir, anno.find("filename").text), color=True)
		points = list()
		labels = list()
		
		## add all the points into the dataset 
		for obj in anno.findall('point'):
			label_type = obj.find('event_type').text
			if (self.sub_dataset != 'all') and (self.sub_dataset != label_type):
				continue
			points.append([int(obj.find(tag).text) for tag in ('y', 'x')])
			labels.append(STORM_LABEL_NAMES.index(label_type))

		points = np.stack(points).astype(np.float32)
		labels = np.stack(label).astype(np.int32)

		return img, points, labels


	__getitem__ = get_example



STORM_LABEL_NAMES = ('hail','wind','torn')