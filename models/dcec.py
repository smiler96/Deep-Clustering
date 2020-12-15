# Unsupervised Deep Embedding for Clustering Analysis
import torch
import torch.nn as nn


class AutoEncoderCNN(nn.Module):
    def __init__(self, input_dims=(1, 28, 28), latent_dim=10):
        super(AutoEncoderCNN, self).__init__()
        channels = [input_dims[0], 32, 64, 128]
        kernel_size = [5, 5, 3]
        stride = [2, 2, 2]

        self.convs = []
        padding = [kernel_size[i]//stride[i] for i in range(len(kernel_size))]
        if input_dims[1] % 8 != 0:
            padding[-1] = 0
        for i in range(len(kernel_size)):
            self.convs.append(
                nn.Conv2d(channels[i], channels[i+1], kernel_size=kernel_size[i], stride=stride[i], padding=padding[i])
            )
            self.convs.append(nn.ReLU(True))
        self.convs = nn.Sequential(*self.convs)

        self.fcs = []
        fc_dims = channels[-1] * (input_dims[1]//8) * (input_dims[2]//8)

        self.fcs.append(nn.Linear(fc_dims, latent_dim))
        self.fcs = nn.Sequential(*self.fcs)

        self.dfcs = []
        self.dfcs.append(nn.Linear(latent_dim, fc_dims))
        self.dfcs.append(nn.ReLU(True))
        self.dfcs = nn.Sequential(*self.dfcs)

        self.dconvs = []
        if input_dims[1] % 8 != 0:
            kernel_size[-1] -= 1
        for i in range(len(kernel_size)-1, -1, -1):
            self.dconvs.append(
                nn.ConvTranspose2d(channels[i + 1], channels[i], kernel_size=kernel_size[i]+1, stride=stride[i],
                          padding=padding[i])
            )
            if i != 0:
                self.dconvs.append(nn.ReLU(True))
        self.dconvs = nn.Sequential(*self.dconvs)

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        latent = self.convs(x)
        b, c, w, h = latent.shape
        latent = latent.view((latent.shape[0], -1))
        latent = self.fcs(latent)

        x = self.dfcs(latent)
        x = x.view((b, c, w, h))
        x = self.dconvs(x)
        return x, latent


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

class DCEC(nn.Module):
    def __init__(self, input_dims=(1, 28, 28), latent_dim=10, n_clusters=10):
        super(DCEC, self).__init__()
        self.n_clusters = n_clusters
        self.AE = AutoEncoderCNN(input_dims=input_dims, latent_dim=latent_dim)
        self.ClusteringLayer = ClusteringLayer(n_clusters=n_clusters, latent_dim=latent_dim)

    def target_distribute(self, q):
        p = (q ** 2) / torch.sum(q, 0)
        p = (p.t() / torch.sum(p, dim=1)).t()
        return p

    def forward(self, x):
        x, feature = self.AE(x)
        source = self.ClusteringLayer(feature)
        return {"rec":  x,
                "feature": feature,
                "source": source}

def wrapper(**kwargs):
    return DCEC(input_dims=kwargs["input_dims"], latent_dim=kwargs["latent_dim"], n_clusters=kwargs["n_clusters"])

if __name__ == "__main__":
    input_dims = (1, 96, 96)
    model = DCEC(input_dims, 10).cuda()
    import numpy as np
    x = np.random.uniform(0, 1, [2, input_dims[0], input_dims[1], input_dims[2]]).astype(np.float32)
    x = torch.tensor(x).cuda()
    import torchsummary
    torchsummary.summary(model, input_dims)

    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        y, source_ = model(x)
        target_ = model.target_distribute(source_)
    print(prof)
    print(y.shape)
    print(target_.shape)