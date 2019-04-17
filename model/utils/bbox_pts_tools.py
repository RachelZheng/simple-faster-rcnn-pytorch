import numpy as np


def bbox_score_intense_event(img, roi, points):
	""" get the intensity per pixel
	args:
		img: n1 x n2. np.array
		roi: n x 4. np.array; in the sequence of ymin, xmin, ymax, xmax
		points: n' x 2. np.array; in the sequence of y, x
	return:
		score_perpix
	"""

	img_ = img[0,:,:] 
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
	score_perpix = pts_roi * inten_roi / area
	intensity_perpix = inten_roi / area
	
	return score_perpix, intensity_perpix