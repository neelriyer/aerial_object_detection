import os
import numpy as np
import PIL

def _convert_cv2_to_pillow(cv2_image):
	return PIL.Image.fromarray(cv2_image[:, :, ::-1])

def _convert_pillow_to_cv2(pil_image):
	open_cv_image = np.array(pil_image) 
	open_cv_image = open_cv_image[:, :, ::-1].copy()  # Convert RGB to BGR 
	return open_cv_image

def _clean_up(file):
	if os.path.exists(file): os.remove(file)

def _clean_up_files(files):
	for file in files: _clean_up(file)


