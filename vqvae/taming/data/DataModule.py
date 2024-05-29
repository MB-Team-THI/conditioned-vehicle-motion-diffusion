import pytorch_lightning as pl
import os
from torch.utils.data import DataLoader, Dataset
from taming.data.motion_dataset import *


# class WrappedDataset(Dataset):
#     """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""
#     def __init__(self, dataset):
#         self.data = dataset

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         return self.data[idx]

class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None,
                  wrap=False, num_workers=None):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size*2
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
            self.train_data_dir = os.path.join(os.getcwd(), '..', 'data', train['dataset'], 'train/')
        if validation is not None:
            self.dataset_configs["validation"] = test
            self.val_dataloader = self._val_dataloader
            self.val_data_dir = os.path.join(os.getcwd(), '..', 'data', train['dataset'], 'val/')
        else:
            self.dataset_configs["validation"] = train
            self.val_dataloader = self._train_dataloader
            self.val_data_dir = os.path.join(os.getcwd(), '..', 'data', train['dataset'], 'train/')
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = self._test_dataloader
            self.test_data_dir = os.path.join(os.getcwd(), '..', 'data', test['dataset'], 'test/')
        self.wrap = wrap

    
    def setup(self, stage: str):
        self.dataset_train = MotionDataset(self.train_data_dir)
        self.dataset_val = MotionDataset(self.train_data_dir, p=0.15)
        self.dataset_test = MotionDataset(self.test_data_dir)
        
    def _train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True,)

    def _val_dataloader(self):
        return DataLoader(self.dataset_val,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False, )

    def _test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False, )
