import torch
import numpy as np
from torch_geometric.utils import degree
from torch_geometric.loader import DataLoader
import pandas as pd
from torch_geometric.data import Data, Batch
from ogb.graphproppred import PygGraphPropPredDataset
from pathlib import Path


def get_data_loaders(data_dir, dataset_name, batch_size, splits, random_state, mutag_x=False):
    multi_label = False
    assert dataset_name in ['sp2020', 'bpi2020pl', 'bpi2017w', 'bpi2018al']

    if dataset_name == 'sp2020':
        dataset = sp2020.get_dataset(dataset_dir=data_dir, dataset_name='sp2020', task=None)
        dataloader, (train_set, valid_set, test_set) = sp2020.get_dataloader(dataset, batch_size=batch_size,
                                                                                 degree_bias=False, seed=random_state)
        print('[INFO] Using default splits!')
        loaders = {'train': dataloader['train'], 'valid': dataloader['eval'], 'test': dataloader['test']}
        test_set = dataset  # used for visualization
    elif dataset_name == 'bpi2020pl':
        dataset = bpi2020pl.get_dataset(dataset_dir=data_dir, dataset_name='bpi2020pl', task=None)
        dataloader, (train_set, valid_set, test_set) = bpi2020pl.get_dataloader(dataset, batch_size=batch_size,
                                                                             degree_bias=False, seed=random_state)
        print('[INFO] Using default splits!')
        loaders = {'train': dataloader['train'], 'valid': dataloader['eval'], 'test': dataloader['test']}
        test_set = dataset  # used for visualization
    elif dataset_name == 'bpi2017w':
        dataset = bpi2017w.get_dataset(dataset_dir=data_dir, dataset_name='bpi2017w', task=None)
        dataloader, (train_set, valid_set, test_set) = bpi2017w.get_dataloader(dataset, batch_size=batch_size,
                                                                             degree_bias=False, seed=random_state)
        print('[INFO] Using default splits!')
        loaders = {'train': dataloader['train'], 'valid': dataloader['eval'], 'test': dataloader['test']}
        test_set = dataset  # used for visualization

    elif dataset_name == 'bpi2018al':
        dataset = bpi2018al.get_dataset(dataset_dir=data_dir, dataset_name='bpi2018al', task=None)
        dataloader, (train_set, valid_set, test_set) = bpi2018al.get_dataloader(dataset, batch_size=batch_size,
                                                                             degree_bias=False, seed=random_state)
        print('[INFO] Using default splits!')
        loaders = {'train': dataloader['train'], 'valid': dataloader['eval'], 'test': dataloader['test']}
        test_set = dataset  # used for visualization

    x_dim = test_set[0].x.shape[1]
    edge_attr_dim = 0 if test_set[0].edge_attr is None else test_set[0].edge_attr.shape[1]
    if isinstance(test_set, list):
        num_class = Batch.from_data_list(test_set).y.unique().shape[0]
    elif test_set.data.y.shape[-1] == 1 or len(test_set.data.y.shape) == 1:
        num_class = test_set.data.y.unique().shape[0]
    else:
        num_class = test_set.data.y.shape[-1]
        multi_label = True

    print('[INFO] Calculating degree...')

    batched_train_set = Batch.from_data_list(train_set)
    d = degree(batched_train_set.edge_index[1], num_nodes=batched_train_set.num_nodes, dtype=torch.long)
    deg = torch.bincount(d, minlength=10)

    aux_info = {'deg': deg, 'multi_label': multi_label}
    print('[DOWN]')
    return loaders, test_set, x_dim, edge_attr_dim, num_class, aux_info


def get_random_split_idx(dataset, splits, random_state=None, x=False):

    if random_state is not None:
        np.random.seed(random_state)

    print('[INFO] Randomly split dataset!')
    idx = np.arange(len(dataset))
    np.random.shuffle(idx)

    if not x:
        n_train, n_valid = int(splits['train'] * len(idx)), int(splits['valid'] * len(idx))
        train_idx = idx[:n_train]
        valid_idx = idx[n_train:n_train+n_valid]
        test_idx = idx[n_train+n_valid:]
    else:
        print('[INFO] mutag_x is True!')
        n_train = int(splits['train'] * len(idx))
        train_idx, valid_idx = idx[:n_train], idx[n_train:]
        test_idx = [i for i in range(len(dataset)) if (dataset[i].y.squeeze() == 0 and dataset[i].edge_label.sum() > 0)]
    return {'train': train_idx, 'valid': valid_idx, 'test': test_idx}


def get_loaders_and_test_set(batch_size, dataset=None, split_idx=None, dataset_splits=None):
    if split_idx is not None:

        assert dataset is not None
        train_loader = DataLoader(dataset[split_idx["train"]], batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(dataset[split_idx["test"]], batch_size=batch_size, shuffle=False)
        test_set = dataset.copy(split_idx["test"])  # For visualization
    else:
        assert dataset_splits is not None
        train_loader = DataLoader(dataset_splits['train'], batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(dataset_splits['valid'], batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(dataset_splits['test'], batch_size=batch_size, shuffle=False)
        test_set = dataset_splits['test']  # For visualization
    return {'train': train_loader, 'valid': valid_loader, 'test': test_loader}, test_set

