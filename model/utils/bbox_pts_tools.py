import numpy as np


def bbox_score_intense_event(img, roi, points, 
	neg_dist=80, max_len_bbox=400):
	""" get the intensity per pixel
	args:
		img: c x n1 x n2. np.array
		roi: n x 4. np.array; in the sequence of ymin, xmin, ymax, xmax
		points: n' x 2. np.array; in the sequence of y, x
	return:
		score_perpix
	"""

	img_ = np.copy(img[0,:,:])
	H, W = img_.shape
	roi_ = np.round(np.copy(roi)).astype(int)
	roi_[:,2] = np.minimum(roi_[:,2], H - 1)
	roi_[:,3] = np.minimum(roi_[:,3], W - 1)
	n_roi = roi.shape[0]
	scores = np.zeros((n_roi,))
	img_points = np.zeros_like(img_)

	## compute points in the region
	for [n_y, n_x] in points:
		n_x = np.floor(n_x).astype(int)
		n_y = np.floor(n_y).astype(int)
		img_points[n_y:,n_x:] += 1

	# compute the intensity in the region
	img_ = np.maximum(img_, 0)
	img_ = np.cumsum(img_, axis=0)
	img_ = np.cumsum(img_, axis=1)

	## compute the average score of roi
	inten_roi = (img_[(roi_[:,2].flatten(), roi_[:,3].flatten())] 
		- img_[(roi_[:,0].flatten(), roi_[:,3].flatten())] 
		- img_[(roi_[:,2].flatten(), roi_[:,1].flatten())]
		+ img_[(roi_[:,0].flatten(), roi_[:,1].flatten())])
	
	pts_roi = (img_points[(roi_[:,2].flatten(), roi_[:,3].flatten())] 
		- img_points[(roi_[:,0].flatten(), roi_[:,3].flatten())] 
		- img_points[(roi_[:,2].flatten(), roi_[:,1].flatten())]
		+ img_points[(roi_[:,0].flatten(), roi_[:,1].flatten())])

	area = (roi_[:,3] - roi_[:,1]) * (roi_[:,2] - roi_[:,0])

	# make sure negative bboxes should be away from event at least neg_dist
	roi_[:,:2] = np.maximum(roi_[:,:2] - neg_dist,0)
	roi_[:,2] = np.minimum(roi_[:,2] + neg_dist, H-1)
	roi_[:,3] = np.minimum(roi_[:,3] + neg_dist, W-1)
	cnt_exten = (img_points[(roi_[:,2].flatten(), roi_[:,3].flatten())] 
		- img_points[(roi_[:,0].flatten(), roi_[:,3].flatten())] 
		- img_points[(roi_[:,2].flatten(), roi_[:,1].flatten())]
		+ img_points[(roi_[:,0].flatten(), roi_[:,1].flatten())])

	score_perpix = pts_roi * inten_roi / area
	intensity_perpix = inten_roi / area
	
	# filter the bbox whose max length >= max_len_bbox
	idx_bbox_largesize = np.where(
		(roi_[:,2] - roi_[:,0] > max_len_bbox) & 
		(roi_[:,3] - roi_[:,1] > max_len_bbox))[0]
	score_perpix[idx_bbox_largesize] = 0
	cnt_exten[idx_bbox_largesize] = 0

	return score_perpix, intensity_perpix, cnt_exten



def bbox_event(roi, roi_score, points):
	""" check the points are within the bbox or not
	args:
		roi: m x 4. np.array; in the sequence of ymin, xmin, ymax, xmax
		roi_score: m, np.array
		points: n x 2. np.array; in the sequence of y, x
	"""
	m, n = roi.shape[0], points.shape[0]
	res = np.zeros((m, n))
	for i, pt in enumerate(points):
		idx = np.where((roi[:,0] <= pt[0]) &
			(roi[:,1] <= pt[1]) & 
			(roi[:,2] >= pt[0]) &
			(roi[:,3] >= pt[1]))[0]
		res[idx, i] = roi_score[idx]
	return res