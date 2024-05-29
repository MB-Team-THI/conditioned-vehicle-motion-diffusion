import scipy
import torchvision
import os
import pandas as pd
from torch.utils.data import DataLoader


def load_data(
    *, data_dir, batch_size, class_cond=False, deterministic=False
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    
    def mat_loader(path):
        _, filename = os.path.split(path)
        # highD
        keys = ['predicted_ax', 'predicted_dpsi', 'predicted_x', 'predicted_y', 'psi_0', 'v0']
        data = [scipy.io.loadmat(path).get(key) for key in keys]
        ddict = {keys[0]: data[0],
                keys[1]: data[1],
                keys[2]: data[2],
                keys[3]: data[3],
                keys[4]: data[4],
                'vx0': data[5],
                'file': path,
                'scenario_id': filename.replace('.mat','')}
        return ddict
        
    def get_dataset(data_dir):
        dataset = torchvision.datasets.DatasetFolder(data_dir, mat_loader, extensions='mat')
        return dataset
    
    if not data_dir:
        raise ValueError("unspecified data directory")
 

    if class_cond:
        dataset = get_dataset(data_dir) 
    else:
        dataset = get_dataset(data_dir)
        dataset.classes = None

    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def load_data_once(
    *, data_dir, batch_size, class_cond=False, deterministic=False
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    
    def mat_loader(path):
        _, filename = os.path.split(path)
        # highD
        keys = ['predicted_ax', 'predicted_dpsi', 'predicted_x', 'predicted_y', 'psi_0', 'v0']
        data = [scipy.io.loadmat(path).get(key) for key in keys]
        ddict = {keys[0]: data[0],
                keys[1]: data[1],
                keys[2]: data[2],
                keys[3]: data[3],
                keys[4]: data[4],
                'vx0': data[5],
                'file': path,
                'scenario_id': filename.replace('.mat','')}
        return ddict
        
    def get_dataset(data_dir):
        dataset = torchvision.datasets.DatasetFolder(data_dir, mat_loader, extensions='mat')
        return dataset
    
    if not data_dir:
        raise ValueError("unspecified data directory")
 

    if class_cond:
        dataset = get_dataset(data_dir) 
    else:
        dataset = get_dataset(data_dir)
        dataset.classes = None

    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    yield from loader