import torch
import torch.nn.functional as F


class GNNplus(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_class, k, dropout):
        """
        Args:
            n_feature (int): the dimension of feature
            n_hidden (int): the dimension of hidden layer
            n_class (int): the number of classification label
            k (int): k-hop aggregation
            dropout (float): dropout rate
        """
        super(GNNplus, self).__init__()

        self.k = k
        self.dropout = dropout
        self.lins = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        for _ in range(self.k + 1):
            self.lins.append(torch.nn.Linear(n_feature, n_hidden))
            self.bns.append(torch.nn.BatchNorm1d(n_hidden))
        self.fc = torch.nn.Linear((k + 1) * n_hidden, n_class)

    def forward(self, feature):
        """
        Args:
            feature (torch Tensor): feature input

        Returns:
            (torch Tensor): log probability for each class in label
        """
        xs = []
        for i in range(self.k + 1):
            x = self.lins[i](feature[i])
            x = F.relu(self.bns[i](x))
            x = F.dropout(x, self.dropout, training=self.training)
            xs.append(x)
        xs = torch.cat(xs, dim=1)
        out = self.fc(xs)

        return F.log_softmax(out, dim=1)
