from torch.utils.data import Dataset
from .imGen import FontStorage
from .ChineseDictionary import allCharacters
from .dataAugment import basic_transforms
import random

class BaseDataset(Dataset):
	'''
	Base class for creating dataset.
	Support generating synthetic images online.
	'''
	def __init__(self, cfg, transform=None):
		self.fonts = FontStorage()
		self.n_fonts = len(self.fonts)
		self.n_chars = len(allCharacters)
		self.len = self.n_fonts * self.n_chars
		self.transform = transform
		self.img_size = cfg.data.input_shape
		
		if self.transform is None:
			self.transform = basic_transforms(cfg)
			
	def __len__(self):
		return self.len
		
	def __getitem__(self, i):
		raise NotImplementedError

class SimSiamDataset(BaseDataset):
	'''
	Dataset used in Simsiam experiment.
	Return two different "views" of the same characters.
	Two views are essentially generated from different fonts, with augmentation.
	'''
	def __getitem__(self, i):
		char_index = i % self.n_fonts
		font_index = i // self.n_fonts
		char = allCharacters[char_index]
		
		x1 = self.fonts.gen_char_img(char, font_index, self.img_size)
		x2 = self.fonts.gen_char_img(char, random.randint(0, self.n_fonts), self.img_size)
		
		return self.transform(x1), self.transform(x2)
		
class TripletDataset(BaseDataset):
	'''
	Dataset used in Triplet experiment.
	Return augmented images and their corresponding labels.
	'''	
	def __getitem__(self, i):
		char_index = i % self.n_fonts
		font_index = i // self.n_fonts
		char = allCharacters[char_index]
		
		x = self.fonts.gen_char_img(char, font_index, self.img_size)
		
		return self.transform(x), char_index
		