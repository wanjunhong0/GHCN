import torch
import torch.nn.functional as F
from utils import sparse_diag


class KhopAttention(torch.nn.Module):
    def __init__(self, n_hidden, k, share_weight=False):

        super(KhopAttention, self).__init__()
        self.share_weight = share_weight
        if self.share_weight:
            self.W = torch.nn.Linear(n_hidden, 1)
        else:
            self.Ws = torch.nn.ModuleList()
            for _ in range(k):
                self.Ws.append(torch.nn.Linear(n_hidden, 1))

    def forward(self, xs):
        attentions = []
        if self.share_weight:
            for x in xs:
                attentions.append(F.leaky_relu(self.W(x)))
        else:
            for i, x in enumerate(xs):
                attentions.append(F.leaky_relu(self.Ws[i](x)))

        attentions = torch.softmax(torch.cat(attentions, dim=1), dim=1)
        for i, x in enumerate(xs):
            xs[i] = torch.sparse.mm(sparse_diag(attentions[:, i]), x)

        return torch.sum(torch.stack(xs), dim=0)
