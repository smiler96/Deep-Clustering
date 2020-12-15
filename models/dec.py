# Unsupervised Deep Embedding for Clustering Analysis
import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, input_dims=784, latent_dim=10):
        super(AutoEncoder, self).__init__()
        params = [input_dims, 500, 500, 2000, latent_dim]

        self.encoder = []
        for i in range(len(params) - 1):
            self.encoder.append(nn.Linear(params[i], params[i+1]))
            if i != len(params)-2:
                self.encoder.append(nn.ReLU(True))
        self.encoder = nn.Sequential(*self.encoder)

        self.decoder = []
        for i in range(len(params)-1, 0, -1):
            self.decoder.append(nn.Linear(params[i], params[i-1]))
            if i != 1:
                self.decoder.append(nn.ReLU(True))
        self.decoder = nn.Sequential(*self.decoder)

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        latents = self.encoder(x)
        x = self.decoder(latents)
        return x, latents


class ClusteringLayer(nn.Module):
    def __init__(self, n_clusters=10, latent_dim=10, alpha=1.0, init_centers=None):
        super(ClusteringLayer, self).__init__()
        self.alpha = alpha
        if init_centers is None:
            init_centers = torch.zeros([n_clusters, latent_dim], dtype=torch.float32).cuda()
            nn.init.xavier_normal_(init_centers)

        self.centers = nn.Parameter(init_centers)

    def forward(self, x):
        x = x.unsqueeze(1)
        votes = torch.sum((x - self.centers) ** 2, dim=2)
        confidences = 1.0 / (1.0 + (votes / self.alpha))
        power = (self.alpha + 1.0) / 2
        confidences = confidences ** power

        # t_distributation = confidences / torch.sum(confidences, dim=1)
        # https://pytorch.org/docs/stable/generated/torch.div.html
        t_distributation = (confidences.t() / torch.sum(confidences, dim=1)).t()
        return t_distributation

class DEC(nn.Module):
    def __init__(self, input_dims=784, latent_dim=10, n_clusters=10):
        super(DEC, self).__init__()
        self.n_clusters = n_clusters
        self.AE = AutoEncoder(input_dims=input_dims, latent_dim=latent_dim)
        self.ClusteringLayer = ClusteringLayer(n_clusters=n_clusters, latent_dim=latent_dim)

    def target_distribute(self, q):
        p = (q ** 2) / torch.sum(q, 0)
        p = (p.t() / torch.sum(p, dim=1)).t()
        return p

    def forward(self, x):
        x, features = self.AE(x)
        source = self.ClusteringLayer(features)
        return {"rec":  x,
                "feature": features,
                "source": source}

def wrapper(**kwargs):
    return DEC(input_dims=kwargs["input_dims"], latent_dim=kwargs["latent_dim"], n_clusters=kwargs["n_clusters"])

if __name__ == "__main__":
    model = DEC(784, 10).cuda()
    import numpy as np
    x = np.random.uniform(0, 1, [2, 784]).astype(np.float32)
    x = torch.tensor(x).cuda()

    # loss = nn.L1Loss()
    # Adam = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.99, 0.999))
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        y, source_ = model(x)
        target_ = model.target_distribute(source_)
    print(prof)
    print(y.shape)
    print(target_.shape)