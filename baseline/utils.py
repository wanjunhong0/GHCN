import torch


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
