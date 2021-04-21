import torch


def normalize_adj(adj, symmetric):
    """Convert adjacency matrix into normalized laplacian matrix

    Args:
        adj (torch sparse tensor): adjacency matrix
        symmetric (boolean) True: D^{-1/2}AD^{-1/2}; False: D^{-1}A

    Returns:
        (torch sparse tensor): Normalized laplacian matrix
    """
    degree = torch.sparse.sum(adj, dim=1).to_dense()
    if symmetric:
        degree_ = sparse_diag(degree.pow(-0.5))
        norm_adj = torch.sparse.mm(torch.sparse.mm(degree_, adj), degree_)
    else:
        degree_ = sparse_diag(degree.pow(-1))
        norm_adj = torch.sparse.mm(degree_, adj)

    return norm_adj

def sparse_diag(value):
    """Convert vector into diagonal matrix

    Args:
        value (torch tensor): vector

    Returns:
        (torch sparse tensor): sparse matrix with only diagonal values
    """
    n = len(value)
    index = torch.stack([torch.arange(n), torch.arange(n)])

    return torch.sparse_coo_tensor(index, value, [n ,n])
