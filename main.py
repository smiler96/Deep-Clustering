import os
import sys
import argparse
from loguru import logger
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import *
from importlib import import_module
import matplotlib.pyplot as plt
from utils import add_noise, acc, visualize
import cv2
from sklearn.cluster import KMeans


def pretrain(**kwargs):
    model = kwargs['model']
    dataloader = kwargs['dataloader']
    epochs = kwargs['epochs']
    pth_file = kwargs['pth']
    png_file = kwargs['png']
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()

    best_loss = 1e10
    model.train()
    for epoch in range(1, epochs+1):
        train_loss = 0.0
        for x, _ in dataloader:
            _, c, h, w = x.shape
            x = x.view((x.shape[0], -1))
            noisy_x = add_noise(x)
            noisy_x = noisy_x.cuda()
            x = x.cuda()
            # ===================forward=====================
            output = model(noisy_x)['rec']
            # output = output.squeeze(1)
            # output = output.view(output.size(0), 28 * 28)
            loss = criterion(output, x)
            train_loss += loss.item()
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ===================visualize====================
            x = x[0].view(c, h, w)
            noisy_x = noisy_x[0].view(c, h, w)
            output = output[0].view(c, h, w)
            final = torch.cat([x, noisy_x, output], dim=1).detach().cpu().numpy()
            final = np.transpose(final, (2, 1, 0))
            final = np.clip(final*255.0, 0, 255).astype(np.uint8)
            cv2.imwrite(png_file, final)

        # ===================log========================
        train_loss /= len(dataloader)
        logger.info('epoch [{}/{}], MSE_loss:{:.4f}'.format(epoch, epochs, train_loss))
        if best_loss > train_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), pth_file)

def train(**kwargs):
    model = kwargs['model']
    dataloader = kwargs['dataloader']
    epochs = kwargs['epochs']
    pth_file = kwargs['pth']
    root = kwargs['root']

    optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-5)
    # optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3, weight_decay=1e-5)
    mse_criterion = nn.MSELoss()
    kl_criterion = nn.KLDivLoss(reduction='mean')

    # ============K-means=======================================
    features = []
    y_true = []
    for x, y in dataloader:
        y_true.append(y.detach().cpu().numpy())
        x = x.view(x.shape[0], -1)
        x = x.cuda()
        f = model(x)['feature']
        features.append(f.detach().cpu().numpy())
    features = np.concatenate(features, axis=0)
    kmeans = KMeans(n_clusters=model.n_clusters, random_state=0).fit(features)
    cluster_centers = kmeans.cluster_centers_
    cluster_centers = torch.tensor(cluster_centers, dtype=torch.float).cuda()
    model.ClusteringLayer.centers = torch.nn.Parameter(cluster_centers)
    # =========================================================
    y_pred = kmeans.predict(features)
    y_true = np.concatenate(y_true, axis=0)
    accuracy = acc(y_true, y_pred)
    logger.info('Initial Accuracy: {}'.format(accuracy))

    best_acc = 0.0
    model.train()
    for epoch in range(1, epochs+1):
        train_mse_loss = 0.0
        train_kl_loss = 0.0
        accuracy = 0.0
        for cnt, (x, y) in enumerate(dataloader):
            # noisy_x = add_noise(x)
            # noisy_x = noisy_x.cuda()
            _, c, h, w = x.shape
            x = x.view((x.shape[0], -1))
            x = x.cuda()
            # ===================forward=====================
            output = model(x)
            x_hat = output['rec']
            x_hat = x_hat.squeeze(1)
            x_hat = x_hat.view(x_hat.size(0), 28 * 28)
            rec_loss = mse_criterion(x_hat, x)
            train_mse_loss += rec_loss.item()

            source_ = output['source']
            # if target not detach, the model collapse
            target_ = model.target_distribute(source_).detach()
            kl_loss = kl_criterion(source_.log(), target_)
            train_kl_loss += kl_loss.item()

            y_pred = source_.argmax(1)
            accuracy += acc(y.cpu().numpy(), y_pred.cpu().numpy())
            if epoch % 10 == 0 and cnt == 0:
                visualize(epoch, output['feature'].detach().cpu().numpy(), y.detach().cpu().numpy(), root)
                x = x[0].view(c, h, w)
                x_hat = x_hat[0].view(c, h, w)
                final = torch.cat([x, x_hat], dim=1).detach().cpu().numpy()
                final = np.transpose(final, (2, 1, 0))
                final = np.clip(final * 255.0, 0, 255).astype(np.uint8)
                cv2.imwrite(f"{root}/clustering.png", final)
            # ===================backward====================
            optimizer.zero_grad()
            total_loss = rec_loss + kl_loss
            total_loss.backward()
            optimizer.step()

        # ===================log========================
        train_mse_loss /= len(dataloader)
        train_kl_loss /= len(dataloader)
        accuracy /= len(dataloader)
        logger.info('epoch [{}/{}], MSE_loss:{:.4f}, KL_loss:{:.4f}, Accuracv:{:.4f}'.format(epoch, epochs, train_mse_loss, train_kl_loss, accuracy))

        if best_acc < accuracy:
            torch.save(model.state_dict(), pth_file)



def parse_args():
    parser = argparse.ArgumentParser('clustering')

    parser.add_argument("--model", type=str, default='DEC')
    parser.add_argument("--dataset", type=str, default='MNIST')
    parser.add_argument("--pretrain", action="store_true")

    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=300)

    parser.add_argument("--n_clusters", type=int, default=10)
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--save_root", type=str, default='saves')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    args.save_root = os.path.join(args.save_root, f"{args.model}_{args.dataset}")
    os.makedirs(args.save_root, exist_ok=True)
    pretrain_pth = os.path.join(args.save_root, "pretrain.pth")
    model_pth = os.path.join(args.save_root, "model.pth")
    if args.dataset == "MNIST":
        args.input_dim = 1*28*28
        args.n_clusters = 10
        args.latent_dim = 10
    elif args.dataset == "STL10":
        args.input_dim = 3*96*96
        args.n_clusters = 10
        args.latent_dim = 10
    else:
        raise NotImplementedError("chose dataset aqain!")

    model = import_module(f"models.{args.model.lower()}").wrapper(input_dims=args.input_dim,
                                                         latent_dim=args.latent_dim,
                                                         n_clusters=args.n_clusters)
    model = model.cuda()
    dataset = eval(f"{args.dataset}Dataset")()
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    # args.pretrain = True
    if args.pretrain:
        pretrain(model=model, dataloader=dataloader, epochs=args.epochs, pth=pretrain_pth,
                 png=os.path.join(args.save_root, "pic.png"))
    else:
        ckpt = torch.load(pretrain_pth)
        model.load_state_dict(ckpt)

        train(model=model, dataloader=dataloader, epochs=args.epochs, pth=model_pth, root=args.save_root)

if __name__ == "__main__":
    # sys.path.append('.')
    main()