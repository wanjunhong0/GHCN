import torch
from torch_geometric.datasets import Planetoid
from utils import normalize_adj


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
        self.norm_adj = normalize_adj(self.adj, symmetric=True)
        self.feature_diffused = [self.feature]
        for i in range(k):
            self.feature_diffused.append(torch.sparse.mm(self.norm_adj, self.feature_diffused[i]))
        