## Unsupervised Deep Clustering

In the repository, the following UDC methods were implemented with pytorch
+ **DEC**: [Unsupervised Deep Embedding for Clustering Analysis](http://arxiv.org/abs/1511.06335)
+ **DCEC**: [Deep Clustering with Convolutional Autoencoders](https://xifengguo.github.io/papers/ICONIP17-DCEC.pdf)

Reproduced Accuracy compared with original papers

<table>
   <tr>
      <td>Method</td>
      <td>       MNIST</td>
      <td></td>
   </tr>
   <tr>
      <td>           reproduced paper</td>
   </tr>
   <tr>
      <td>DEC</td>
      <td>88.27</td>
      <td>84.08</td>
   </tr>
   <tr>
      <td>DCEC</td>
      <td>87.41</td>
      <td>88.97</td>
   </tr>
</table>

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
