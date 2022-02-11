import torch
import numpy as np

def get_embedding(cfg, model, dataset):
    dataloader =  torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=torch.multiprocessing.cpu_count()
    )
    model.eval()
    embeddings = np.ones((len(dataset), model.embedding_dim), np.float32)
    labels = np.ones((len(dataset), 1), np.int)
    with torch.no_grad():
        img_iter = 0
        for batch, (img, label) in enumerate(dataloader):
            batch_size = img.shape[0]
            img_iter += batch_size
            img = img.to(cfg.device)
            img_embedding = model(img).cpu().numpy()
            embeddings[img_iter - batch_size: img_iter] = img_embedding
            labels[img_iter - batch_size: img_iter] = label.cpu().numpy()
            
        # np.save(cfg.embedding_info.embedding_file_path, embeddings)
    return embeddings, labels