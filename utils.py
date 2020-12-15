import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
# from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment as linear_assignment

nmi = normalized_mutual_info_score
ari = adjusted_rand_score


def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    ind = list(map(lambda x, y:[x, y], ind[0], ind[1]))
    acc_ = sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
    return acc_


import torch


def add_noise(img):
    noise = torch.randn(img.size()) * 0.2
    noisy_img = img + noise
    return noisy_img


import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize(epoch, x, y, root):
    fig = plt.figure()
    ax = plt.subplot(111)
    # x = x.detach().cpu().numpy()
    # y = y.detach().cpu().numpy()
    x_embedded = TSNE(n_components=2).fit_transform(x)
    plt.scatter(x_embedded[:,0], x_embedded[:,1], c=y)
    fig.savefig(f'{root}/{epoch}.png')
    plt.close(fig)