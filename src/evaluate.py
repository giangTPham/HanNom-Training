from dataset import TripletDataset
from dataset.dataAugment import *
from utils import *
import torch
import numpy as np
import faiss

def _topk(sample_labels, test_labels, I):
    correct = 0
    for test_label, i in zip(I, test_labels):
        if test_label in i:
            correct += 1
    
    acc = correct/float(len(test_labels))
    return acc

def _k_neighbors(cfg, sample_dataset, test_dataset, k, embedding_dim):
    sample_embedding, sample_labels = get_embedding(cfg, model, sample_dataset)
    test_embedding, test_labels = get_embedding(cfg, model, test_dataset)

    faiss_index = faiss.IndexFlatL2(embedding_dim)
    faiss_index.add(sample_embedding)
    
    D, I = faiss_index.search(test_embedding, k)
    
    return sample_labels, test_labels, I

def evaluate(cfg, k: int, model, save_to='visualize.png'):
    sample_dataset = TripletDataset(cfg, transform=test_transforms(cfg), one_font_only=True)
    test_dataset = TripletDataset(cfg)
    
    knn = _k_neighbors(cfg, sample_dataset, test_dataset, k, model.embedding_dim)
    acc = _topk(*knn)
    print('Top {} accuracy: {:.2f}%'.format(k, acc*100))
    _, _, I = knn
    visualize(I, test_dataset, sample_dataset, save_to)
    
def visualize(I, test_dataset, sample_dataset, save_to, n=5):
    k = len(I[0])
    imshape = test_dataset[0][0].shape
    imgs = []
    for i in np.random.randint(0, len(test_dataset), size=n):
        img, _ = test_dataset[i]
        imgs.append(img.squeeze().unsqueeze(0).numpy())
        for j in I[i]:
            imgs.append(sample_dataset[j][0].squeeze().unsqueeze(0).numpy())
            
    imgs = np.concatenate(imgs, axis=0)
    imgs = torch.from_numpy(imgs)
    
    from torchvision.utils import save_image
    
    save_image(imgs, save_to, ncol=5,
            normalize=True, range=(0, 255))
        
def init_simsiam_model(cfg):
    model = SimSiamModel(
        backbone=cfg.model.backbone,
        latent_dim=cfg.model.latent_dim,
        proj_hidden_dim=cfg.model.proj_hidden_dim,
        pred_hidden_dim=cfg.model.pred_hidden_dim,
        load_pretrained=cfg.model.pretrained,
    )
    return model
    
def init_triplet_model(cfg): 
    model = TripletModel(
		backbone=cfg.model.backbone,
		embedding_dim=cfg.model.embedding_dim,
		pretrained=cfg.model.pretrained,
		freeze=cfg.model.freeze
	)
    return model
    
if __name__ == '__main__':
    import argparse
    from utils import parse_args
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--pipeline', type=str, 
                        default='simsiam',
                        help='Name of the training process')
    parser.add_argument('--cfg_path', type=str, 
                        default='experiment_configs/train_simsiam.yaml',
                        help='Config path')
                        
    parser.add_argument('--model_path', type=str, 
                        default='weights/pretrained_final.pt',)
                        
    args = parser.parse_args()
    
    cfg = parse_args(args.cfg_path)
    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    from models import *
    if 'simsiam' in args.pipeline:
        model = init_simsiam_model(cfg)
    elif 'triplet' in args.pipeline:
        model = init_triplet_model(cfg)
    model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    print('Done loading weights')
    evaluate(cfg, 5, model)
    