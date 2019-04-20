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
	inten_roi = (img_[(roi[:,2].flatten(), roi[:,3].flatten())] 
		- img_[(roi[:,0].flatten(), roi[:,3].flatten())] 
		- img_[(roi[:,2].flatten(), roi[:,1].flatten())]
		+ img_[(roi[:,0].flatten(), roi[:,1].flatten())])
	
	pts_roi = (img_points[(roi[:,2].flatten(), roi[:,3].flatten())] 
		- img_points[(roi[:,0].flatten(), roi[:,3].flatten())] 
		- img_points[(roi[:,2].flatten(), roi[:,1].flatten())]
		+ img_points[(roi[:,0].flatten(), roi[:,1].flatten())])

	area = (roi[:,3] - roi[:,1]) * (roi[:,2] - roi[:,0])

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

