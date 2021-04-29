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
        self.linears = torch.nn.ModuleList()
        for _ in range(self.k):
            self.linears.append(torch.nn.Linear(n_feature, n_hidden))
        self.fc = torch.nn.Linear(n_hidden, n_class)

    def forward(self, feature):
        """
        Args:
            feature (torch Tensor): feature input

        Returns:
            (torch Tensor): log probability for each class in label
        """
        xs = []
        for i in range(self.k):
            x = self.linears[i](feature[i])
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
            xs.append(x)
        out = torch.sum(torch.stack(xs), dim=0)
        out = self.fc(out)

        return F.log_softmax(out, dim=1)
