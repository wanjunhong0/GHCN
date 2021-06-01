import argparse
import time
import torch
import torch.nn.functional as F
import torchmetrics
from torch_geometric.datasets import Planetoid, Reddit
from torch_geometric.nn import GCNConv

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
parser.add_argument('--layer', type=int, default=2, help='The layers of graph convolution')
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
    dataset = Planetoid(root=args.data_path, name=args.dataset, split=args.split)
data = dataset[0]

"""
===========================================================================
Model
===========================================================================
"""
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.convs = torch.nn.ModuleList()
        for i in range(args.layer):
            dim_in = dataset.num_features if i == 0 else args.hidden
            dim_out = dataset.num_classes if i == args.layer - 1 else args.hidden
            self.convs.append(GCNConv(dim_in, dim_out))

    def forward(self, x, edge_index):
        for i in range(args.layer):
            x = self.convs[i](x, edge_index)
            if i < args.layer - 1:
                x = F.relu(x)
                x = F.dropout(x, args.dropout, training=self.training)

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
