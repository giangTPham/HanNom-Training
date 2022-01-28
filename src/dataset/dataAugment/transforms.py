from typing import Tuple, Dict

import kornia
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image


MEAN = (100.,)*3
STD = (70,)*3



def augment_transforms(cfg) -> nn.Sequential:
    augs = nn.Sequential(
        # kornia.augmentation.ColorJitter(1, 1, 1, 5, p=0.5),
		kornia.augmentation.RandomBoxBlur(p=0.3),
        kornia.augmentation.RandomGaussianNoise(std=50),
		kornia.augmentation.RandomPerspective(.3, p=.5),
		kornia.augmentation.RandomAffine(25, 0.1, scale=(0.95,1.1)),
        kornia.augmentation.RandomErasing(scale=(0.01, cfg.data.augmentation.random_erase), value=1, p=0.3),
        kornia.augmentation.RandomGrayscale(p=0.2),
        # kornia.augmentation.RandomResizedCrop(
            # size=[cfg.data.input_shape]*2,
            # scale=(cfg.data.augmentation.resize_scale, 1.0),
            # ratio=(0.25, 1.33),
            # p=0.2
        # ),
        kornia.augmentation.Normalize(
            mean=torch.tensor(MEAN),
            std=torch.tensor(STD)
        )
    )
    # augs = augs.to(cfg.device)
    return augs


def basic_transforms(cfg) -> T.Compose:
    return T.Compose([
		ToTensor(),
        T.Resize(size=[cfg.data.input_shape]*2),
        T.RandomApply([T.GaussianBlur(kernel_size=11, sigma=(0.1, 2.0))])
		# T.Normalize(mean=MEAN, std=STD)
    ])


def test_transforms(cfg) -> T.Compose:
    return T.Compose([
        T.Resize(size=cfg.data.input_shape),
        T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))]),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])

class ToTensor(nn.Module):
	def __init__(self):
		super().__init__()
		
	def forward(self, x):
		x = torch.Tensor(x)
		size = x.shape
		assert len(size) >= 3
		x = x.transpose(-1,-2).transpose(-2, -3)
		return x