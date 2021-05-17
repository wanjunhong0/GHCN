import argparse
import time
import torch
import torch.nn.functional as F
import torchmetrics

from models import GNNplus
from load_data import Data
from utils import EarlyStopping


"""
===========================================================================
Configuation
===========================================================================
"""
parser = argparse.ArgumentParser(description="Run GNNplus.")
parser.add_argument('--data_path', type=str, default='./data/', help='Input data path')
parser.add_argument('--model_path', type=str, default='checkpoint.pt', help='Saved model path.')
parser.add_argument('--dataset', type=str, default='Cora', help='Choose a dataset from {Cora, CiteSeer, PubMed}')
parser.add_argument('--split', type=str, default='full', help='The type of dataset split {public, full, random}')
parser.add_argument('--nodetrim', action='store_true', help='Trim ajds.')
parser.add_argument('--fusion', type=str, default='noderank', help='Choose a dataset from {concat, attention, noderank}')
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--epoch', type=int, default=1000, help='Number of epochs to train')
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
t_started = time.time()
data = Data(path=args.data_path, dataset=args.dataset, split=args.split,
            k=args.k, nodetrim=args.nodetrim, fusion=args.fusion)
print('Loaded {0} dataset with {1} nodes and {2} edges'.format(args.dataset, data.n_node, data.n_edge))
feature = [i.to(device) for i in data.feature_diffused]
label = data.label.to(device)
label_train = label[data.idx_train]
label_val = label[data.idx_val]
label_test = label[data.idx_test]

"""
===========================================================================
Training
===========================================================================
"""
# Model and optimizer
model = GNNplus(n_feature=data.n_feature, n_hidden=args.hidden, n_class=data.n_class,
                k=data.k, dropout=args.dropout, fusion=args.fusion).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
metric = torchmetrics.Accuracy().to(device)
t_train = time.time()

for epoch in range(1, args.epoch+1):
    t = time.time()
    # Training
    model.train()
    optimizer.zero_grad()
    output = model(feature)[data.idx_train]
    loss_train = F.nll_loss(output, label_train)
    acc_train = metric(output.max(1)[1], label_train)
    loss_train.backward()
    optimizer.step()

    # Validation
    model.eval()
    output = model(feature)[data.idx_test]
    loss_val = F.nll_loss(output, label_test)
    acc_val = metric(output.max(1)[1], label_test)

    print('Epoch {0:04d} | Time: {1:.2f}s | Loss = [train: {2:.4f}, val: {3:.4f}] | ACC = [train: {4:.4f}, val: {5:.4f}]'
          .format(epoch, time.time() - t, loss_train, loss_val, acc_train, acc_val))

    # Early stop
    early_stop(acc_val, model)
    if early_stop.early_stop:
        print('Early stop triggered at epoch {0}!'.format(epoch - args.patience))
        model.load_state_dict(torch.load(args.model_path))
        break
"""
===========================================================================
Testing
===========================================================================
"""
model.eval()
output = model(feature)[data.idx_test]
loss_test = F.nll_loss(output, label_test)
acc_test = metric(output.max(1)[1], label_test)
print('======================Testing======================')
print('Loss = [test: {0:.4f}] | ACC = [test: {1:.4f}]'.format(loss_test, acc_test))

print('Training time: {:.2f}'.format(time.time() - t_train))
print('Total time: {:.2f}'.format(time.time() - t_started))
