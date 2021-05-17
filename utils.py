import torch


def normalize_adj(adj, symmetric):
    """Convert adjacency matrix into normalized laplacian matrix

    Args:
        adj (torch sparse tensor): adjacency matrix
        symmetric (boolean) True: D^{-1/2}AD^{-1/2}; False: D^{-1}A

    Returns:
        (torch sparse tensor): Normalized laplacian matrix
    """
    degree = torch.sparse.sum(adj, dim=1)
    if symmetric:
        degree_ = sparse_diag(degree.pow(-0.5))
        norm_adj = torch.sparse.mm(torch.sparse.mm(degree_, adj), degree_)
    else:
        degree_ = sparse_diag(degree.pow(-1))
        norm_adj = torch.sparse.mm(degree_, adj)

    return norm_adj

def sparse_diag(vector):
    """Convert vector into diagonal matrix

    Args:
        vector (torch tensor): diagonal values of the matrix

    Returns:
        (torch sparse tensor): sparse matrix with only diagonal values
    """
    if not vector.is_sparse:
        vector = vector.to_sparse()
    n = len(vector)
    index = torch.stack([vector._indices()[0], vector._indices()[0]])

    return torch.sparse_coo_tensor(index, vector._values(), [n ,n])

def eliminate_negative(adj):
    """Eliminate negative values in torch sparse tensor

    Args:
        adj (torch sparse tensor): original sparse matrix

    Returns:
        (torch sparse tensor): sparse matrix without negative values
    """
    mask = adj._values() > 0
    return torch.sparse_coo_tensor(adj._indices()[:, mask], adj._values()[mask], adj.size())


class EarlyStopping:
    """Early stops the training if validation metrics doesn't improve after a given patience."""
    def __init__(self, patience=10, mode='min', delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation improved.
            mode (str): Max or min is prefered improvement
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path for the checkpoint to be saved to.
        """
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_min = float('inf')
        self.delta = delta
        self.path = path

    def __call__(self, val, model):
        if self.mode == 'min':
            score = -val
        if self.mode == 'max':
            score = val

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), self.path)
        self.val_min = val_loss
