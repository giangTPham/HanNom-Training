import os

import torch
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
import os

from models import SimSiamModel
from loss import negative_cosine_similarity
from dataset.dataAugment import augment_transforms, test_transforms
from dataset import SimSiamDataset
from utils import parse_args, AverageMeter, calculate_std_l2_norm
# from src.utils import eval_self_sup_model, calculate_std_l2_norm, AverageMeter, parse_aug


def main(cfg) -> None:

	model = SimSiamModel(
		backbone=cfg.model.backbone,
		latent_dim=cfg.model.latent_dim,
		proj_hidden_dim=cfg.model.proj_hidden_dim,
		pred_hidden_dim=cfg.model.pred_hidden_dim,
		load_pretrained=cfg.model.pretrained,
	)
	
	model = model.to(cfg.device)
	model.train()


	opt = torch.optim.AdamW(
		params=model.parameters(),
		lr=cfg.train.lr,
		betas=(0.9, 0.999),
		weight_decay=cfg.train.weight_decay
	)

	train_dataset = SimSiamDataset(cfg)

	train_dataloader = torch.utils.data.DataLoader(
					dataset=train_dataset,
					batch_size=cfg.train.batch_size,
					shuffle=True,
					drop_last=False,
					pin_memory=True,
					num_workers=torch.multiprocessing.cpu_count()
	)


	train_aug = augment_transforms(cfg=cfg).to(cfg.device)

	writer = SummaryWriter()

	n_iter = 0
	std_tracker = AverageMeter('std_stacker')
	for epoch in range(cfg.train.epochs):
		std_tracker.reset()
		pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), position=0, leave=False)
		for batch, (x1, x2) in pbar:
			opt.zero_grad()
			x1, x2 = x1.to(cfg.device), x2.to(cfg.device)
			
			x1, x2 = train_aug(x1), train_aug(x2)
			
			# project
			z1, z2 = model(x1), model(x2)

			# predict
			p1, p2 = model.predict(z1), model.predict(z2)

			# compute loss
			loss1 = negative_cosine_similarity(p1, z2)
			loss2 = negative_cosine_similarity(p2, z1)
			loss = (loss1 + loss2)/2
			loss.backward()
			opt.step()
			with torch.no_grad():
				z1_std = calculate_std_l2_norm(z1)
				z2_std = calculate_std_l2_norm(z2)
				std_tracker.update(z1_std + z2_std)

			pbar.set_description("Epoch {}, Loss: {:.4f}, Std: {:.6f}".format(epoch, float(loss), std_tracker.avg))

			if n_iter % cfg.train.log_interval == 0:
				writer.add_scalar(tag="loss/train", scalar_value=float(loss), global_step=n_iter)
				writer.add_scalar(tag='loss/std', scalar_value=std_tracker.avg, global_step=n_iter)

			n_iter += 1
			

	# save model
	dir_path = os.path.dirname(os.path.realpath(__file__))
	weight_path = os.path.join(dir_path, 'weights')
	if not (os.path.exists(weight_path)):
		os.makedirs(weight_path)
	torch.save(model.state_dict(), os.path.join(weight_path, cfg.model.name + "_final.pt"))


if __name__ == "__main__":
	import argparse
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--cfg_path', type=str, 
						default='experiment_configs/train_simsiam.yaml',
                        help='Config path')
	args = parser.parse_args()
	
	cfg = parse_args(args.cfg_path)
	cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
	main(cfg)
