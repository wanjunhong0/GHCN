import argparse
import time
import torch
import torch.nn.functional as F
import torchmetrics
from torch_geometric.datasets import Planetoid, Reddit
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops
import torch_geometric.transforms as T

from utils import EarlyStopping


"""
===========================================================================
Configuation
===========================================================================
"""
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='./data/', help='Input data path')
parser.add_argument('--model_path', type=str, default='checkpoint.pt', help='Saved model path.')
parser.add_argument('--dataset', type=str, default='Cora', help='Choose a dataset from {Cora, CiteSeer, PubMed}')
parser.add_argument('--split', type=str, default='full', help='The type of dataset split {public, full, random}')
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--epoch', type=int, default=5000, help='Number of epochs to train')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate')
parser.add_argument('--weight_decay', type=float, default=0.005, help='Weight decay (L2 norm on parameters)')
parser.add_argument('--k', type=int, default=10, help='k-hop aggregation')
parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
parser.add_argument('--patience', type=int, default=100, help='How long to wait after last time validation improved')

args = parser.parse_args()
for arg in vars(args):
    print('{0} = {1}'.format(arg, getattr(args, arg)))
torch.manual_seed(args.seed)
# training on the first GPU if not available on CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Training on device = {}'.format(device))

early_stop = EarlyStopping(patience=args.patience, mode='max', path=args.model_path)

"""
===========================================================================
Loading data
===========================================================================
"""
if args.dataset == 'Reddit':
    dataset = Reddit(root=args.data_path + args.dataset)
else:
    dataset = Planetoid(root=args.data_path, name=args.dataset, split=args.split, transform=T.NormalizeFeatures())
data = dataset[0]

"""
===========================================================================
Model
===========================================================================
"""
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):

    fill_value = 2. if improved else 1.
    num_nodes = int(edge_index.max()) + 1 if num_nodes is None else num_nodes
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                 device=edge_index.device)

    if add_self_loops:
        edge_index, tmp_edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)
        assert tmp_edge_weight is not None
        edge_weight = tmp_edge_weight

    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


class Prop(MessagePassing):
    def __init__(self, num_classes, k):
        super(Prop, self).__init__(aggr='add')
        self.k = k
        self.proj = torch.nn.Linear(num_classes, 1)

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, norm = gcn_norm(edge_index, edge_weight, x.size(0), dtype=x.dtype)

        preds = []
        preds.append(x)
        for _ in range(self.k):
            x = self.propagate(edge_index, x=x, norm=norm)
            preds.append(x)

        pps = torch.stack(preds, dim=1)
        retain_score = self.proj(pps)
        retain_score = retain_score.squeeze()
        retain_score = torch.sigmoid(retain_score)
        retain_score = retain_score.unsqueeze(1)
        out = torch.matmul(retain_score, pps).squeeze()
        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={})'.format(self.__class__.__name__, self.K)

    def reset_parameters(self):
        self.proj.reset_parameters()


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lin1 = torch.nn.Linear(dataset.num_features, args.hidden)
        self.lin2 = torch.nn.Linear(args.hidden, dataset.num_classes)
        self.prop = Prop(dataset.num_classes, args.k)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.prop.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, args.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, args.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop(x, edge_index)
        return F.log_softmax(x, dim=1)

"""
===========================================================================
Training
===========================================================================
"""
# Model and optimizer
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
metric = torchmetrics.Accuracy().to(device)

for epoch in range(1, args.epoch+1):
    t = time.time()
    # Training
    model.train()
    optimizer.zero_grad()
    output = model(data.x, data.edge_index)[data.train_mask]
    loss_train = F.nll_loss(output, data.y[data.train_mask])
    acc_train = metric(output.max(1)[1], data.y[data.train_mask])
    loss_train.backward()
    optimizer.step()

    # Validation
    model.eval()
    output = model(data.x, data.edge_index)[data.test_mask]
    loss_val = F.nll_loss(output, data.y[data.test_mask])
    acc_val = metric(output.max(1)[1], data.y[data.test_mask])

    print('Epoch {0:04d} | Time: {1:.2f}s | Loss = [train: {2:.4f}, val: {3:.4f}] | ACC = [train: {4:.4f}, val: {5:.4f}]'
          .format(epoch, time.time() - t, loss_train, loss_val, acc_train, acc_val))

    # Early stop
    early_stop(acc_val, model)
    if early_stop.early_stop:
        print('Early stop triggered at epoch {0}!'.format(epoch - args.patience))
        model.load_state_dict(torch.load(args.model_path))
        print('{:.4f}'.format(early_stop.best_score))
        break
