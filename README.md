## Unsupervised Deep Clustering

In the repository, the following UDC methods were implemented with pytorch
+ **DEC**: [Unsupervised Deep Embedding for Clustering Analysis - ICML2015](http://arxiv.org/abs/1511.06335)
+ **DCEC**: [Deep Clustering with Convolutional Autoencoders - ICONIP2017](https://xifengguo.github.io/papers/ICONIP17-DCEC.pdf)

### Reproduced Accuracy compared with original papers

| Method | MNIST           ||
| :----: | :----: | :----: | 
| | reproduced | paper | 
| DEC | 88.27 | 84.08 |
| DCEC | 87.41 | 88.97 |

### Using Details
+ **DEC**
```python
python main.py --model DEC --dataset MNIST --n_clusters 10 --alpha 0.1 --batch_size 1024 --epochs 200 --pretrain --denoising
python main.py --model DEC --dataset MNIST --n_clusters 10 --alpha 0.1 --batch_size 1024 --epochs 500 --denoising
```

+ **DCEC**
```python
python main_conv.py --model DCEC --dataset MNIST --n_clusters 10 --alpha 0.1 --batch_size 1024 --epochs 200 --pretrain --denoising
python main_conv.py --model DCEC --dataset MNIST --n_clusters 10 --alpha 0.1 --batch_size 1024 --epochs 500 --denoising
```

### Reference
+ DCE-Pytorch: [https://github.com/Deepayan137/DeepClustering](https://github.com/Deepayan137/DeepClustering)
+ DCEC-Keras: [https://github.com/XifengGuo/DCEC](https://github.com/XifengGuo/DCEC)