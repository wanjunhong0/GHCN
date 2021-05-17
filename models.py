import torch
import torch.nn.functional as F
from attention import KhopAttention


class GNNplus(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_class, k, dropout, fusion):
        """
        Args:
            n_feature (int): the dimension of feature
            n_hidden (int): the dimension of hidden layer
            n_class (int): the number of classification label
            k (int): k-hop aggregation
            dropout (float): dropout rate
            fusion (str) type of fusion
        """
        super(GNNplus, self).__init__()

        self.k = k
        self.dropout = dropout
        self.fusion = fusion

        self.Ws = torch.nn.ModuleList()
        for _ in range(self.k):
            self.Ws.append(torch.nn.Linear(n_feature, n_hidden))
        if self.fusion == 'attention':
            self.attention = KhopAttention(n_hidden, k, share_weight=False)
        if self.fusion == 'concat':
            self.fc = torch.nn.Linear(k * n_hidden, n_class)
        else:
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
            x = self.Ws[i](feature[i])
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
            xs.append(x)

        if self.fusion == 'noderank':
            out = torch.sum(torch.stack(xs), dim=0)
        elif self.fusion == 'attention':
            out = self.attention(xs)
        elif self.fusion == 'concat':
            out = torch.cat(xs, dim=1)
        out = self.fc(out)

        return F.log_softmax(out, dim=1)
