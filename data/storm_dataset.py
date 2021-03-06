import os, warnings
import xml.etree.ElementTree as ET
from torchvision import transforms as T
import numpy as np

from .util import read_image, read_3_imgs, read_exact_three_imgs

def idx2imgname(idx, yr_range=[2008, 2017]):
	""" convert the image index to the image name
	"""
	c_yr, c_mon, c_day, c_hr = 107136, 8928, 288, 12
	yr_, mon_, day_ = idx//c_yr + yr_range[0], (idx % c_yr)//c_mon + 1, (idx % c_mon)//c_day + 1
	hr_, min_ = (idx % c_day)//c_hr, idx % c_hr * 5
	img_name = 'n0r_%04d%02d%02d%02d%02d.png'%(yr_, mon_, day_, hr_, min_)
	return img_name

def imgname2idx(img_name, yr_range=[2008, 2017]):
	""" convert the image name to the index in the dataset
	"""
	c_yr, c_mon, c_day, c_hr = 107136, 8928, 288, 12
	yr_, mon_, day_ = int(img_name[4:8]), int(img_name[8:10]), int(img_name[10:12])
	hr_, min_ = int(img_name[12:14]),int(img_name[14:16])
	num = int((yr_ - yr_range[0]) * c_yr + (mon_ - 1) * c_mon + (day_ - 1) * c_day + (hr_) * c_hr + min_/5)
	return num

def inference_idx2imgname(idx):
	""" convert the image index to the image name
	"""
	c_mon, c_day = 1488, 48 # 48 simulations per day
	mon_ = idx // c_mon + 1
	day_ = (idx % c_mon) // c_day + 1
	sim_ = idx % c_day + 1
	img_name = 'diags_d02_2017%02d%02d00_mem_10_f0%02d.png'%(mon_, day_, sim_)
	return img_name


class StormDataset:
	def  __init__(self, opt, sub_dataset='all', split='train'):
		if opt.bool_train_one_hour:
			id_name = '{}_1hr.txt'.format(split)
		else:
			id_name = '{}.txt'.format(split)

		id_list_file = os.path.join(opt.split_dir, id_name)

		self.ids = [int(id_.split('.')[0]) for id_ in open(id_list_file)]
		self.data_dir = opt.data_dir
		self.annotation_dir = opt.annotation_dir
		self.sub_dataset = sub_dataset
		self.bool_train_one_hour = opt.bool_train_one_hour
		self.label_names = STORM_LABEL_NAMES

	def __len__(self):
		return len(self.ids)

	def get_example(self, i):
		""" get the i-th example
		"""
		id_img = self.ids[i]

		name_xml = os.path.join(self.annotation_dir, '{:07d}.xml'.format(id_img))
		anno = ET.parse(name_xml).getroot()
		# img = read_image(os.path.join(self.data_dir, anno.find("filename").text), color=True)
		try:
			if self.bool_train_one_hour:
				img = read_3_imgs(self.data_dir, idx2imgname(id_img))
			else:
				img = read_exact_three_imgs(self.data_dir, idx2imgname(id_img))
		except:
			img = np.zeros((0,0,3))
		points = list()
		labels = list()	
		
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



class ModelDataset:
	def  __init__(self, data_dir, annotation_dir, split_dir, 
		split='inference', bool_img_only=True):
		id_list_file = os.path.join(split_dir, '{}.txt'.format(split))
		self.ids = [int(id_.split('.')[0]) for id_ in open(id_list_file)]
		self.data_dir = data_dir
		self.annotation_dir = annotation_dir 
		self.bool_img_only = bool_img_only
		self.split = split

	def __len__(self):
		return len(self.ids)
	
	def get_example(self, i):
		id_img = self.ids[i]
		if self.split == 'inference':
			img_name = inference_idx2imgname(id_img)
		else:
			img_name = idx2imgname(id_img)

		img = read_image(os.path.join(self.data_dir, img_name), color=True)

		if self.bool_img_only:
			return img, img_name
		else:
			xml_name = os.path.join(self.annotation_dir, '{:07d}.xml'.format(id_img))
			if os.path.isfile(xml_name):
				anno = ET.parse(xml_name).getroot()
				points = list()
				labels = list()	
				
				## add all the points into the dataset 
				for obj in anno.findall('point'):
					points.append([int(obj.find(tag).text) for tag in ('y', 'x')])
					labels.append(0)

				points = np.stack(points).astype(np.float32)
				labels = np.stack(labels).astype(np.int32)
			else:
				points = np.zeros((0,2)).astype(np.float32)
				labels = np.zeros((0,)).astype(np.int32)
			return img, points, labels, img_name

	__getitem__ = get_example


STORM_LABEL_NAMES = ('all')