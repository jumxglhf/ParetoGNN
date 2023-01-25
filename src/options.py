import argparse
import os
from pathlib import Path

from numpy import require


class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialize_parser()

    def add_optim_options(self):
        self.parser.add_argument('--warmup_steps', type=int, default=1000)
        self.parser.add_argument('--total_steps', type=int, default=100000)
        self.parser.add_argument('--scheduler_steps', type=int, default=None, 
                        help='total number of step for the scheduler, if None then scheduler_total_step = total_step')
        self.parser.add_argument('--accumulation_steps', type=int, default=1)
        self.parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
        self.parser.add_argument('--clip', type=float, default=1., help='gradient clipping')
        self.parser.add_argument('--optim', type=str, default='adam')
        self.parser.add_argument('--scheduler', type=str, default='fixed')
        self.parser.add_argument('--weight_decay', type=float, default=1e-5)
        self.parser.add_argument('--fixed_lr', action='store_true')

    def add_ParetoGNN_options(self):
        self.parser.add_argument('--hid_dim', type=int, nargs='+', default=[256, 128])
        self.parser.add_argument('--predictor_dim', type=int, default=512)
        self.parser.add_argument('--inter_dim', type=int, default=0)
        self.parser.add_argument('--tasks', type=str, nargs='+', default=['p_gm', 'p_link', 'p_ming', 'p_minsg'])
        self.parser.add_argument('--n_layer', type=int, default=100)
        self.parser.add_argument('--n_group', type=int, default=2)
        self.parser.add_argument('--dropout', type=float, default=0., help='dropout rate')
        self.parser.add_argument('--temperature_gm', type=float, default=0.2, help='gm: tau for infoNCE')
        self.parser.add_argument('--temperature_minsg', type=float, default=0.1, help='minsg: tau for infoNCE')

        self.parser.add_argument('--sub_size', type=int, default=100, help='size for subgraphs used in gm')
        self.parser.add_argument('--decor_size', type=int, default=100, help='size for subgraphs used in decor')
        self.parser.add_argument('--decor_lamb', type=float, default=1e-3, help='lambda for decor')
        self.parser.add_argument('--decor_der', type=float, default=0.2, help='der for decor')
        self.parser.add_argument('--decor_dfr', type=float, default=0.2, help='dfr for decor')
        self.parser.add_argument('--minsg_der', type=float, default=0.3, help='der for minsg')
        self.parser.add_argument('--minsg_dfr', type=float, default=0.3, help='dfr for minsg')
        self.parser.add_argument('--use_prelu', action='store_true')
        self.parser.add_argument('--pretrain_label_dir', required=True, type=str, default='path to pretrain labels')
        self.parser.add_argument('--dataset', type=str, required=True, choices=['wiki_cs', 'co_cs', 'co_phy', 'co_photo', 'co_computer', 'actor', 'chameleon', 'squirrel', 'pubmed', 'cora', 'citeseer', 'arxiv', 'products'], help='dataset')
        self.parser.add_argument('--grad_norm', type=str, default='l2', choices=['l2', 'loss', 'loss+', 'none'])
        self.parser.add_argument('--split', type=str, default='random', choices=['public', 'random'], help='dataset')
        self.parser.add_argument('--mask_edge', action='store_true', help='mask edge for link prediction')
        self.parser.add_argument('--tvt_addr', type=str, default='none', help='tvt file for the dataset')
        self.parser.add_argument('--hetero_graph_path', type=str, default='none', help='tvt file for the dataset')
        self.parser.add_argument('--dual_encoder', action='store_true')
        self.parser.add_argument('--use_saint', action='store_true')
        self.parser.add_argument('--lp_neg_ratio', type=int, default=1, help='negative ratio for link prediction pretrain')
        self.parser.add_argument('--no_self_loop', action='store_true')
        self.parser.add_argument('--norm', type=str, default='batch')
        self.parser.add_argument('--not_use_pareto', action='store_true')
        
    def initialize_parser(self):
        # basic parameters
        self.parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment')
        self.parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint/', help='models are saved here')
        self.parser.add_argument('--model_path', type=str, default='none', help='path for retraining')
        self.parser.add_argument('--is_distributed', action='store_true')
        self.parser.add_argument("--per_gpu_batch_size", default=1, type=int, 
                        help="Batch size per GPU/CPU for training.")
        self.parser.add_argument("--batch_size_multiplier_minsg", default=10, type=int, 
                        help="Batch size multiplier for minsg.")
        self.parser.add_argument("--batch_size_multiplier_ming", default=10, type=int, 
                        help="Batch size multiplier for ming.")
        self.parser.add_argument('--khop_ming', type=int, default=3, help='order for ming sampling')
        self.parser.add_argument('--khop_minsg', type=int, default=3, help='order for minsg sampling')
        self.parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
        self.parser.add_argument("--world_size", type=int, default=1,
                        help="For distributed training: world_size")
        self.parser.add_argument('--seed', type=int, default=0, help="random seed for initialization")
        self.parser.add_argument('--save_freq', type=int, default=1000, help="random seed for initialization")
        self.parser.add_argument("--worker", type=int, default=3,
                        help="number of workers for dataloader")
        self.parser.add_argument('--wandb', action='store_true')
        self.parser.add_argument('--debug', action='store_true')

    def parse(self):
        opt = self.parser.parse_args()
        return opt