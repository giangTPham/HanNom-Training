import os
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from .utils import check_exist_and_download_fonts, check_font_extension

class FontStorage:
	def __init__(self):
		self.current_dir = os.path.dirname(os.path.realpath(__file__))
		self.font_path = os.path.join(self.current_dir, 'fonts')
		check_exist_and_download_fonts(self.font_path)
		self.font_list = []
		self.font_names = list(filter(check_font_extension, os.listdir(self.font_path)))
		self.cached_img_size=-1
		self.n_fonts = len(self.font_names)

	def load_font(self, font_size=45):
		self.font_list = []
		for font_name in self.font_names:
			font = ImageFont.truetype(os.path.join(self.font_path, font_name), font_size)
			self.font_list.append(font)

	def check_fontsize_change(self, img_size):
		change = self.cached_img_size != img_size
		if change:
			new_font_size = int(45/64. * img_size)
			self.cached_img_size = img_size
			self.load_font(new_font_size)
			
	def __len__(self):
		return self.n_fonts

	def gen_char_img(self, text, font_index, img_size=64, include_font_name=False):
		self.check_fontsize_change(img_size)
		im = Image.new("RGB", (img_size, img_size), (255, 255, 255))
		dr = ImageDraw.Draw(im)
		dr.text((2, 1), text, font=self.font_list[font_index%self.n_fonts], fill="#000000")
		if include_font_name:
			dr.text((0, 0), self.font_names[font_index%self.n_fonts], fill="#000000")
		img_np = np.array(im, dtype=np.float32)
		im.close()
		return img_np
	
