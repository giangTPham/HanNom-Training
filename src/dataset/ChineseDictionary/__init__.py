import os

import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

dir_path = os.path.dirname(os.path.realpath(__file__))
chinese_dict_path = os.path.join(dir_path, 'cleaned-chinese-word-list.txt')

allCharacters = []
with open(chinese_dict_path, 'r', encoding="utf-8") as f:
	allCharacters = f.readlines()[0]
	
__all__ = ['allCharacters']