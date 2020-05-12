import argparse
import functools
import platform
import warnings

import torch
import torch.nn.functional as F


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.normal_(0.0, 0.01)


def entropy_loss(logits):
    p_softmax = F.softmax(logits, dim=1)
    mask = p_softmax.ge(0.000001)  # greater or equal to
    mask_out = torch.masked_select(p_softmax, mask)
    entropy = -(torch.sum(mask_out * torch.log(mask_out)))
    return entropy / float(p_softmax.size(0))


class OptimizerManager:
    def __init__(self, optims):
        self.optims = optims  # if isinstance(optims, Iterable) else [optims]

    def __enter__(self):
        for op in self.optims:
            op.zero_grad()

    def __exit__(self, exceptionType, exception, exceptionTraceback):
        for op in self.optims:
            op.step()
        self.optims = None
        if exceptionTraceback:
            print(exceptionTraceback)
            return False
        return True


class EvaluationManager:
    def __init__(self, nets):
        self.nets = nets

    def __enter__(self):
        self.prev = torch.is_grad_enabled()
        torch._C.set_grad_enabled(False)
        for net in self.nets:
            net.eval()

    def __exit__(self, *args):
        torch.set_grad_enabled(self.prev)
        for net in self.nets:
            net.train()
        return False

    def __call__(self, func):
        @functools.wraps(func)
        def decorate_no_grad(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return decorate_no_grad


class IteratorWrapper:
    def __init__(self, loader):
        self.loader = loader
        self.iterator = iter(loader)

    def __iter__(self):
        self.iterator = iter(self.loader)

    def get_next(self):
        try:
            items = self.iterator.next()
        except:
            self.__iter__()
            items = self.iterator.next()
        return items


def add_base_args(parser: argparse.ArgumentParser):
    # Dataset arguments
    parser.add_argument('--target', default='ROD', choices=['ROD', 'valHB'])
    parser.add_argument('--source', default='synROD', choices=['synROD', 'synHB'])
    parser.add_argument("--data_root_source", default=None)
    parser.add_argument("--data_root_target", default=None)
    parser.add_argument("--train_file_source", default=None)
    parser.add_argument("--test_file_source", default=None)
    parser.add_argument("--train_file_target", default=None)
    parser.add_argument("--test_file_target", default=None)
    parser.add_argument("--class_num", default=51)

    parser.add_argument("--task", default="rgbd-rr")
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--snapshot", default="snapshot/")
    parser.add_argument("--tensorboard", default="tensorboard")
    parser.add_argument('--gpu', default=0)
    parser.add_argument('--suffix', default="")

    # hyper-params
    parser.add_argument("--net", default="resnet18")
    parser.add_argument("--epoch", default=40, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--lr_mult", default=1.0, type=float)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--weight_decay", default=0.05, type=float)
    parser.add_argument("--dropout_p", default=0.5)

    parser.add_argument("--weight_rot", default=1.0, type=float)
    parser.add_argument('--weight_ent', default=0.1, type=float)


def default_paths(args):
    print("{} -> {}".format(args.source, args.target))
    data_root_source, data_root_target, split_source_train, split_source_test, split_target = make_paths(
        source=args.source, target=args.target)
    args.data_root_source = args.data_root_source or data_root_source
    args.data_root_target = args.data_root_target or data_root_target
    args.train_file_source = args.train_file_source or split_source_train
    args.test_file_source = args.test_file_source or split_source_test
    args.train_file_target = args.train_file_target or split_target
    args.test_file_target = args.test_file_target or split_target


def make_paths(source='synROD', target='ROD'):
    node = platform.node()

    data_root_source, data_root_target, split_source_train, split_source_test, split_target = None, None, None, None, \
                                                                                              None

    # Machine-dependent default paths
    """
    if node == 'machine-name':
        print("Setting default paths for {}".format(node))
        if source == 'synROD':
            data_root_source = ''
            split_source_train = ''
            split_source_test = ''
        elif source == 'synHB':
            data_root_source = ''
            split_source_train = ''
            split_source_test = ''

        if target == 'ROD':
            data_root_target = ''
            split_target = ''
        elif target == 'valHB':
            data_root_target = ''
            split_target = ''
    """

    if None in [data_root_source, data_root_target, split_source_train, split_source_test, split_target]:
        if source == 'synROD':
            data_root_source = './datasets/synROD/synARID_crops_square'
            split_source_train = './datasets/synROD/synARID_crops_square/synARID_50k-split_sync_train1.txt'
            split_source_test = './datasets/synROD/synARID_crops_square/synARID_50k-split_sync_test1.txt'
        elif source == 'synHB':
            data_root_source = './datasets/HB/HB_Syn_crops_square'
            split_source_train = './datasets/HB/HB_Syn_crops_square/HB_Syn_crops_25k-split_sync_train1.txt'
            split_source_test = './datasets/HB/HB_Syn_crops_square/HB_Syn_crops_25k-split_sync_test1.txt'

        if target == 'ROD':
            data_root_target = './datasets/ROD'
            split_target = './datasets/ROD/wrgbd_40k-split_sync.txt'
        elif target == 'valHB':
            data_root_target = './datasets/HB/HB_val_crops_square'
            split_target = './datasets/HB/HB_val_crops_square/HB_val_crops_25k-split_sync.txt'

    return data_root_source, data_root_target, split_source_train, split_source_test, split_target


def map_to_device(device, t):
    return tuple(map(lambda x: x.to(device), t))
