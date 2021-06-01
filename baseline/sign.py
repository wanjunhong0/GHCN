import argparse
import time
import torch
import torch.nn.functional as F
import torchmetrics
from torch_geometric.datasets import Planetoid, Reddit
from torch_geometric.nn import GCNConv
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
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 norm on parameters)')
parser.add_argument('--k', type=int, default=3, help='k-hop aggregation')
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
transform = T.Compose([T.NormalizeFeatures(), T.SIGN(args.k)])
if args.dataset == 'Reddit':
    dataset = Reddit(root=args.data_path + args.dataset, transform=transform)
else:
    dataset = Planetoid(root=args.data_path, name=args.dataset, split=args.split, transform=transform)
data = dataset[0]

"""
===========================================================================
Model
===========================================================================
"""
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.lins = torch.nn.ModuleList()
        for _ in range(args.k + 1):
            self.lins.append(torch.nn.Linear(dataset.num_node_features, args.hidden))
        self.lin = torch.nn.Linear((args.k + 1) * args.hidden, dataset.num_classes)

    def forward(self, xs):
        hs = []
        for x, lin in zip(xs, self.lins):
            h = lin(x).relu()
            h = F.dropout(h, args.dropout, training=self.training)
            hs.append(h)
        h = torch.cat(hs, dim=-1)
        h = self.lin(h)

        return h.log_softmax(dim=-1)

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
    xs = [data.x.to(device)]
    xs += [data[f'x{i}'].to(device) for i in range(1, args.k + 1)]
    output = model(xs)[data.train_mask]
    loss_train = F.nll_loss(output, data.y[data.train_mask])
    acc_train = metric(output.max(1)[1], data.y[data.train_mask])
    loss_train.backward()
    optimizer.step()

    # Validation
    model.eval()
    xs = [data.x.to(device)]
    xs += [data[f'x{i}'].to(device) for i in range(1, args.k + 1)]
    output = model(xs)[data.test_mask]
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
