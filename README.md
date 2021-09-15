# GHCN
This is the PyTorch implementation of our paper: "Tackling Over-Smoothing: Graph Hollow Convolution Network with Topological Layer Fusion"

## Requirement
- pytorch >= 1.8.0

- torch-geometric

- torchmetrics

Note that the versions of PyTorch and PyTorch Geometric should be compatible.

## Datsetset
The `data` folder contains three benchmark datasets (Cora, Citeseer, Pubmed). 

Cora, CiteSeer, and PubMed are pre-downloaded from [`torch_geometric.datasets`](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets.Planetoid).

We use the same full-supervised setting as [FastGCN](https://github.com/matenure/FastGCN).

## Example
For Cora dataset, please run:
```
python main.py --dataset=Cora
```
For CiteSeer dataset, please run:
```
python main.py --dataset=CiteSeer --weight_decay=5e-3 --k=20
```
For PubMed dataset, please run:
```
python main.py --dataset=CiteSeer --weight_decay=5e-5 --k=9 --dropout=0.5
```


## Baselines
The `baseline` folder contains 6 SOTA GNN methods as follows：

- [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) [[code]](https://github.com/tkipf/gcn)
```
python baseline/gcn.py
```
- [Graph Attention Networks](https://arxiv.org/abs/1710.10903) [[code]](https://github.com/PetarV-/GAT)
```
python baseline/gat.py
```
- [Simplifying Graph Convolutional Networks](https://arxiv.org/abs/1902.07153) [[code]](https://github.com/Tiiiger/SGC)
```
python baseline/sgc.py
```
- [SIGN: Scalable Inception Graph Neural Networks](https://arxiv.org/abs/2004.11198) [[code]](https://github.com/twitter-research/sign)
```
python baseline/sign.py
```
- [Towards Deeper Graph Neural Networks](https://arxiv.org/abs/2007.09296) [[code]](https://github.com/mengliu1998/DeeperGNN)
```
python baseline/dagnn.py
```
- [Simple and Deep Graph Convolutional Networks](https://arxiv.org/abs/2007.02133) [[code]](https://github.com/chennnM/GCNII)
```
python baseline/gcnii.py
```