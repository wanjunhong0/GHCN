import torch
import torch.nn.functional as F
from utils import sparse_diag


class KhopAttention(torch.nn.Module):
    def __init__(self, n_hidden, k, dropout):

        super(KhopAttention, self).__init__()
        self.dropout= dropout
        self.W = torch.nn.Linear(n_hidden, 1)

    def forward(self, xs):
        attentions = []
        for x in xs:
            attentions.append(F.leaky_relu(self.W(x)))

        attentions = torch.softmax(torch.cat(attentions, dim=1), dim=1)
        attentions = F.dropout(attentions, self.dropout, training=self.training)
        for i, x in enumerate(xs):
            xs[i] = torch.sparse.mm(sparse_diag(attentions[:, i]), x)

        return torch.sum(torch.stack(xs), dim=0)
