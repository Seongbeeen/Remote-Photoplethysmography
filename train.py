import argparse

import numpy as np
from datasets.COHFACE import COHFACE, COHFACE_vreader
from datasets.MAHNOB import MAHNOB, MAHNOB_vreader
from datasets.PURE import PURE, PURE_vreader
from datasets.UBFC_PHYS import UBFC_PHYS, UBFC_PHYS_vreader
from datasets.UBFC_rPPG import UBFC_rPPG, UBFC_rPPG_vreader
from datasets.V4V import V4V, V4V_vreader
from archs.SAttST_CNN import SAttST_CNN
from archs.ST_CNN import ST_CNN

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np


def train_loop(trainloader, model, criterion, optimizer, batch_size, overlap=0, monitor=10):
    model.train()
    size = len(trainloader.dataset)
    train_preds = np.empty(0)
    train_targets = np.empty(0)

    for batch, (inputs, targets) in enumerate(trainloader):
        inputs = inputs.to(device)
        targets = targets.to(device).squeeze()
        pred = model(inputs).squeeze()

        for cnt, pr in enumerate(pred):
            if (batch_size & batch + cnt) % int(1/(1-overlap)) == 0:
                p = pr.cpu().detach().numpy().squeeze().flatten()
                t = targets[cnt].cpu().detach().numpy().squeeze().flatten()
                train_preds = np.concatenate((train_preds, p))
                train_targets = np.concatenate((train_targets, t))
        loss = criterion()(pred, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % monitor == monitor-1:
            loss, current = loss.item(), batch * len(targets)
            print(f"loss: {loss:>7f}  [{batch:>5d}/{size:>5d}]")

    return train_preds, train_targets


def main():
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "name", type=str, help="The name of the dataset to train")
    parser.add_argument(
        "--video_path", type=str, help="List of video/image path")
    parser.add_argument(
        "--label_path", type=str, default=None, help="List of label path")
    parser.add_argument(
        "--depth", type=int, default=512, help="Time depth")
    parser.add_argument(
        "--vidStore", type=bool, default=False, help="Choose vreader or vread")
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--model", type=str, default="SAttST_CNN")
    parser.add_argument(
        "--lr", type=float, default=1e-4)
    parser.add_argument(
        "--epochs", type=int, default=120)
    parser.add_argument(
        "--ckpt_dir", type=str, defalut="0000")
    args = parser.parse_args()

    # Process dataset
    print(f"Training {args.name} dataset...")
    if args.vidStore:
        if args.name == "COHFACE":
            trainset = COHFACE(args)
        elif args.name == "MAHNOB":
            trainset = MAHNOB(args)
        elif args.name == "PURE":
            trainset = PURE(args)
        elif args.name == "UBFC_PHYS":
            trainset = UBFC_PHYS(args)
        elif args.name == "UBFC_rPPG":
            trainset = UBFC_rPPG(args)
        elif args.name == "V4V":
            trainset = V4V(args)
        else:
            print(f"ERROR: Unknown dataset {args.name}")
            exit(1)
    else:
        if args.name == "COHFACE":
            trainset = COHFACE_vreader(args)
        elif args.name == "MAHNOB":
            trainset = MAHNOB_vreader(args)
        elif args.name == "PURE":
            trainset = PURE_vreader(args)
        elif args.name == "UBFC_PHYS":
            trainset = UBFC_PHYS_vreader(args)
        elif args.name == "UBFC_rPPG":
            trainset = UBFC_rPPG_vreader(args)
        elif args.name == "V4V":
            trainset = V4V_vreader(args)
        else:
            print(f"ERROR: Unknown dataset {args.name}")
            exit(1)

    if args.model == "SAtt_ST_CNN":
        model = SAttST_CNN()
    elif args.model == "ST_CNN":
        model = ST_CNN()
    else:
        print(f"Error: Unknown model {args.model}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cpu":
        model - model.cuda()
    trainloader = DataLoader(trainset, batch_size=args.batch_size)

    criterion = nn.MSELoss()
    learning_rate = float(args.lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    train_losses = []
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1} \n-------------------------------")
        train_preds, train_targets = train_loop(
            trainloader, model, criterion, optimizer)

        torch.save(model.state_dict(), 'checkpoints/' + args.model +
                   '/' + args.ckpt_dir + f"/model_ep{epoch+1}.pth")
        train_loss = criterion(torch.Tensor(
            train_preds, torch.Tensor(train_targets)))

        print(f"Train loss: {train_loss:>8f}\n")

        train_losses.append(train_loss)
    print("Finished!")


if __name__ == "__main__":
    main()
