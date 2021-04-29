import torch
from torch_geometric.datasets import Planetoid
from utils import normalize_adj, sparse_diag


class Data():
    def __init__(self, path, dataset, split, k):
        """Load dataset
           Preprocess feature, label, normalized adjacency matrix and train/val/test index

        Args:
            path (str): file path
            dataset (str): dataset name
            split (str): type of dataset split
            k (int) k-hop aggregation
        """
        data = Planetoid(root=path, name=dataset, split=split)
        self.feature = data[0].x
        self.edge = data[0].edge_index
        self.label = data[0].y
        self.idx_train = torch.where(data[0].train_mask)[0]
        self.idx_val = torch.where(data[0].val_mask)[0]
        self.idx_test = torch.where(data[0].test_mask)[0]
        self.n_node = data[0].num_nodes
        self.n_edge = data[0].num_edges
        self.n_class = data.num_classes
        self.n_feature = data.num_features

        self.adj = torch.sparse_coo_tensor(self.edge, torch.ones(self.n_edge), [self.n_node, self.n_node])
        self.adj = self.adj + sparse_diag(torch.ones(self.n_node))
        self.adjs = [self.adj]
        for i in range(k - 1):
            adj = torch.sparse.mm(self.adjs[i], self.adj)
            self.adjs.append(torch.sparse_coo_tensor(adj._indices(), torch.ones_like(adj._values()), adj.size()))

        self.adjs = NodeTrim(self.adjs, k)
        self.k = len(self.adjs)
        self.node_rank = NodeRank(self.adjs, self.k)
        self.norm_adjs = [normalize_adj(i, symmetric=True) for i in self.adjs]
        self.feature_diffused = []
        for i in range(self.k):
            feature = torch.sparse.mm(self.norm_adjs[i], self.feature)
            feature = torch.sparse.mm(sparse_diag(self.node_rank[i]), feature)
            self.feature_diffused.append(feature)


def NodeRank(adjs, k):
    adjs = [normalize_adj(i, symmetric=False) for i in adjs]
    degree = torch.sparse.sum(adjs[0], dim=1).unsqueeze(dim=0)
    node_rank = [torch.sparse.mm(degree, adjs[0])]
    for i in range(1, k):
        node_rank.append(torch.sparse.mm(node_rank[i - 1], adjs[i]))
    node_rank = torch.softmax(torch.cat(node_rank, dim=0).to_dense(), dim=0)

    return node_rank


def NodeTrim(adjs, k):
    filters = [adjs[0]]
    for i in range(1, k - 1):
        filters.append(filters[i - 1] + adjs[i])
    adj_trimmed = [adjs[0]]
    for i in range(1, k):
        adj = eliminate_negative(adjs[i] - filters[i - 1])
        if adj._nnz() == 0:
            print('{}-hop NodeTrim adj matrix is empty!'.format(i + 1))
            break
        adj_trimmed.append(adj)

    return adj_trimmed


def eliminate_negative(adj):
    mask = adj._values() > 0
    return torch.sparse_coo_tensor(adj._indices()[:, mask], adj._values()[mask], adj.size())
