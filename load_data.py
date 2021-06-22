import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, Reddit
from utils import normalize_adj, sparse_diag, eliminate_negative


class Data():
    def __init__(self, path, dataset, split, k, nodetrim, fusion):
        """Load dataset
           Preprocess feature, label, normalized adjacency matrix and train/val/test index

        Args:
            path (str): file path
            dataset (str): dataset name
            split (str): type of dataset split
            k (int) k-hop aggregation
            nodetrim (bool) activate NodeTrim or not
            fusion (str) type of fusion
        """
        if dataset == 'Reddit':
            data = Reddit(root=path + dataset)
        else:
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
        self.norm_adj = normalize_adj(self.adj, symmetric=True)
        self.norm_adjs = [self.norm_adj]
        for i in range(k-1):
            adj = torch.sparse.mm(self.norm_adj, self.norm_adjs[i])
            if nodetrim:
                adj = NodeTrim(adj, self.norm_adjs[i])
            self.norm_adjs.append(adj)


        self.feature_diffused = [self.feature]
        for i in range(k):
            feature = torch.sparse.mm(self.norm_adjs[i], self.feature)
            self.feature_diffused.append(feature)


def NodeTrim(current, previous):
    mask = torch.sparse_coo_tensor(previous._indices(), torch.ones(previous._nnz()), previous.size())
    return torch.mul(current, mask)

# def NodeTrim(current, previous):
#     mask = (previous._indices() == current._indices().T.unsqueeze(-1)).all(1).any(1)
#     adj = torch.sparse_coo_tensor(current._indices()[:, mask], current._values()[mask], current.size())
#     return adj
