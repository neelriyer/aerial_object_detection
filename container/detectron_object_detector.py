import cv2
import json
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.modeling import build_model
import torch
import utils
import numpy as np
from PIL import Image

class detector():

	def __init__(self):

		# set model and test set
		self.model = 'retinanet_R_50_FPN_1x.yaml'

		# obtain detectron2's default config
		self.cfg = get_cfg() 

		# load values from a file
		self.cfg.merge_from_file("config.yaml")
		# self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/"+self.model)) 

		# set device to cpu
		self.cfg.MODEL.DEVICE = "cpu"

		# get weights 
		# self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/"+self.model) 
		self.cfg.MODEL.WEIGHTS = "models/model_final.pth"

		# set the testing threshold for this model
		# self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7 
		self.cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.7   

		# build model from weights
		# self.cfg.MODEL.WEIGHTS = self.convert_model_for_inference()

	# build model and convert for inference
	def convert_model_for_inference(self):

		# build model
		model = build_model(self.cfg)

		# save as checkpoint
		torch.save(model.state_dict(), 'checkpoint.pth')

		# return path to inference model
		return 'checkpoint.pth'

	# detectron model
	# adapted from detectron2 colab notebook: https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5	
	def inference(self, file, write = False, json = False):

		predictor = DefaultPredictor(self.cfg)
		im = cv2.imread(file)
		outputs = predictor(im)

		if json:
			with open(self.curr_dir+'/data.txt', 'w') as fp:
				json.dump(outputs['instances'], fp)
				json.dump(cfg.dump(), fp)

		# get metadata
		metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])

		im = cv2.imread(file)
		outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
		v = Visualizer(im[:, :, ::-1],
		              metadata=metadata, 
		              scale=1)
		out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
		# im = out.get_image()[:, :, ::-1]

		# get image 
		img = PIL.Image.fromarray(np.uint8(out.get_image()))

		# write to jpg
		if write: 
			cv2.imwrite('img.jpg',out.get_image())

		return img



