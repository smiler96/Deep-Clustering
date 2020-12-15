## Unsupervised Deep Clustering

In the repository, the following UDC methods were implemented with pytorch
+ **DEC**: [Unsupervised Deep Embedding for Clustering Analysis](http://arxiv.org/abs/1511.06335)
+ **DCEC**: [Deep Clustering with Convolutional Autoencoders](https://xifengguo.github.io/papers/ICONIP17-DCEC.pdf)

Reproduced Accuracy compared with original papers
| Method | MNIST |
| ---- | :----: | :----: | 
| | reproduced | paper | 
| DEC | | |
| DCEC | | |

Using Details
+ **DEC**
```python
python main.py --model DEC --dataset MNIST --n_clusters 10 --alpha 0.1 --batch_size 1024 --epochs 200 --pretrain --denoising
python main.py --model DEC --dataset MNIST --n_clusters 10 --alpha 0.1 --batch_size 1024 --epochs 500
```

+ **DCEC**
```python
python main_conv.py --model DCEC --dataset MNIST --n_clusters 10 --alpha 0.1 --batch_size 1024 --epochs 200 --pretrain --denoising
python main_conv.py --model DCEC --dataset MNIST --n_clusters 10 --alpha 0.1 --batch_size 1024 --epochs 500
```
