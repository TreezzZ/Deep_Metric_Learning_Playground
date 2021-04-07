import os
import random
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from loguru import logger
from torch.cuda.amp import GradScaler, autocast
from torchvision.models import resnet50
from tqdm import tqdm

from criterion import MultiSimilarityLoss
from dataset import get_dataloader
from evaluation import calc_recall


def main(args):
    # Set seed for reproducibility
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    # Get logger
    wandb.init(config=args, project="Multi-Similarity", dir=args.work_dir)
    params = []
    for k, v in vars(args).items():
            params.append(f"{k}: {v}")
    logger.info("\n" + "\n".join(params)) 

    # Get data loader
    train_dataloader, test_dataloader = get_dataloader(
        args.data_dir, 
        args.img_per_class,
        args.batch_size, 
        args.num_workers,
        args,
    )

    # Get model
    model = resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, args.embed_dim)
    if args.is_frozen:
        for module in filter(lambda m: type(m) == nn.BatchNorm2d, model.modules()):
            module.eval()
            module.train = lambda _: None
    model.train()
    model.to(args.device)
    wandb.watch(model)

    # Get optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=args.lr,
        weight_decay=4e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        args.max_epochs,
    )

    # Get criterion
    criterion = MultiSimilarityLoss(
        args.alpha,
        args.beta,
        args.lamda,
        args.epsilon,
    )

    # FP16
    scaler = GradScaler()

    best_recall = [0.] * 6
    # Start loop
    running_loss = 0.
    cur_iter = 0
    for epoch in range(1, args.max_epochs+1):
        # Training
        model.train()
        for data in tqdm(train_dataloader, desc="Training"):
            imgs = data["data"]
            labels = data["label"]
            cur_iter += 1
            imgs, labels = imgs.to(args.device), labels.to(args.device)

            with autocast():
                embed = model(imgs)
                if args.is_normalize:
                    embed = F.normalize(embed)
                loss = criterion(embed, labels)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

            if cur_iter % args.log_step == 0:
                wandb.log({'train_loss': running_loss / args.log_step})
                running_loss = 0.
        scheduler.step()
    
        # Evaluation
        if epoch % args.eval_step == 0:
            model.eval()
            K = 32

            with torch.no_grad():
                # Get embedding
                embed = []
                gt = []
                for data in tqdm(test_dataloader, "Evaluation-embedding"):
                    imgs = data["data"]
                    labels = data["label"]
                    imgs = imgs.to(args.device)

                    embed.append(F.normalize(model(imgs)))
                    gt.append(labels)
                embed = torch.vstack(embed)
                gt = torch.hstack(gt).squeeze().float()

                # Calculate prediction
                S = embed @ embed.t()
                pred = gt[S.topk(1 + K)[1][:, 1:]].float().cpu()

                # Calculate R@K
                recall = []
                logger.info(f"Epoch: {epoch}")
                for k in [1, 2, 4, 8, 16, 32]:
                    recall_at_k = calc_recall(pred, gt, k)
                    recall.append(recall_at_k)
                    logger.info("R@{} : {:.3f}".format(k, 100 * recall_at_k))
                    wandb.log({f'R@{k}': 100*recall_at_k})
                if best_recall[0] < recall[0]:
                    best_recall = recall
    return best_recall


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--work_dir', type=str, default='work_dir')
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--eval_step', type=int, default=1)
    parser.add_argument('--log_step', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--is_frozen', action="store_true")
    parser.add_argument('--is_normalize', action="store_true")
    parser.add_argument('--seed', type=int, default=24)

    parser.add_argument('--arch', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--img_per_class', type=int, default=2)
    parser.add_argument('--max_epochs', type=int, default=90)
    parser.add_argument('--embed_dim', type=int, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--alpha', type=float, required=True)
    parser.add_argument('--beta', type=float, required=True)
    parser.add_argument('--lamda', type=float, required=True)
    parser.add_argument('--epsilon', type=float, required=True)

    args = parser.parse_args()
    args.device = torch.device("cuda", args.gpu)
    os.makedirs(args.work_dir, exist_ok=True)
    recall = main(args)
    wandb.run.summary["best_recall_1"] = recall[0]
