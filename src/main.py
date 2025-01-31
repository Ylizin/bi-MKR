import argparse
import numpy as np
from data_loader import RSDataset, KGDataset
from train import train

np.random.seed(42)

parser = argparse.ArgumentParser()

# movie
# parser.add_argument('--dataset', type=str, default='movie', help='which dataset to use')
parser.add_argument("--n_epochs", type=int, default=6, help="the number of epochs")
parser.add_argument(
    "--dim", type=int, default=8, help="dimension of user and entity embeddings"
)
parser.add_argument("--L", type=int, default=1, help="number of low layers")
parser.add_argument("--H", type=int, default=1, help="number of high layers")
parser.add_argument("--batch_size", type=int, default=10,help="batch size")
parser.add_argument(
    "--l2_weight", type=float, default=1e-6, help="weight of l2 regularization"
)
parser.add_argument(
    "--lr_rs", type=float, default=0.02, help="learning rate of RS task"
)
parser.add_argument(
    "--lr_kge", type=float, default=0.01, help="learning rate of KGE task"
)
parser.add_argument(
    "--kge_interval", type=int, default=5, help="training interval of KGE task"
)
parser.add_argument(
    "--use_inner_product",
    type=bool,
    default=True,
    help="decide user inner product or mlp",
)
parser.add_argument('-uh',
    "--user_enhanced",
    type=int,
    default=1,
    help="using the user kg enhanced of the item kg, if 0 item enhanced, 1 user enhanced, 2 user and item enhanced.",
)

"""
# book
parser.add_argument('--dataset', type=str, default='book', help='which dataset to use')
parser.add_argument('--n_epochs', type=int, default=10, help='the number of epochs')
parser.add_argument('--dim', type=int, default=8, help='dimension of user and entity embeddings')
parser.add_argument('--L', type=int, default=1, help='number of low layers')
parser.add_argument('--H', type=int, default=1, help='number of high layers')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--l2_weight', type=float, default=1e-6, help='weight of l2 regularization')
parser.add_argument('--lr_rs', type=float, default=2e-4, help='learning rate of RS task')
parser.add_argument('--lr_kge', type=float, default=2e-5, help='learning rate of KGE task')
parser.add_argument('--kge_interval', type=int, default=2, help='training interval of KGE task')
"""

"""
# music
parser.add_argument('--dataset', type=str, default='music', help='which dataset to use')
parser.add_argument('--n_epochs', type=int, default=10, help='the number of epochs')
parser.add_argument('--dim', type=int, default=4, help='dimension of user and entity embeddings')
parser.add_argument('--L', type=int, default=2, help='number of low layers')
parser.add_argument('--H', type=int, default=1, help='number of high layers')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--l2_weight', type=float, default=1e-6, help='weight of l2 regularization')
parser.add_argument('--lr_rs', type=float, default=1e-3, help='learning rate of RS task')
parser.add_argument('--lr_kge', type=float, default=2e-4, help='learning rate of KGE task')
parser.add_argument('--kge_interval', type=int, default=2, help='training interval of KGE task')
"""

parser.add_argument(
    "--workers", type=int, default=3, help="number of data loading workers"
)
parser.add_argument("-sl", "--show_loss", action="store_true", help="show loss or not")
parser.add_argument("-st", "--show_topk", action="store_true", help="show topK or not")
parser.add_argument(
    "-sum",
    "--summary_path",
    type=str,
    default="./summary",
    help="path to store training summary",
)
args = parser.parse_args()

if __name__ == "__main__":
    rs_dataset = RSDataset(args)
    kg_dataset = KGDataset(args)
    train(args, rs_dataset, kg_dataset)

