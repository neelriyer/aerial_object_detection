from fastai import *
from fastai.vision import *
from fastai.callbacks.hooks import *
from fastai.callbacks import *
# import matplotlib.pyplot as plt
# import matplotlib.image as immg

class segmenter():

	def __init__(self, model_path = '/model/', model_name = 'dron_mask.pkl'):
		self.learn = load_learner(model_path, model_name)

	def inference(self, f):
		img = open_image(f).resize((3,200,300))
		mask = self.learn.predict(img)[0]

		# img.show(y=mask, title='masked')
		# plt.savefig('masked.png', bbox_inches = 'tight', pad_inches = 0)
		# plt.close()

		# mask.show(title='mask only', alpha=1.)
		# plt.savefig('mask_only.png', bbox_inches = 'tight', pad_inches = 0)
		# plt.close()

		# img.save('original.png')

		return mask

	   


