import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class ADNI(Dataset):
    def __init__(self, data, targets=None):
        self.x_bar = torch.tensor(np.mean(data, axis=0)).float()
        self.data = torch.from_numpy(data)
        self.targets = targets

    def __getitem__(self, item):
        return (
        *filter(lambda x: x is not None, (self.data[item], self.targets[item] if self.targets is not None else None)),)

    def __len__(self):
        if self.targets is not None:
            assert len(self.data) == len(self.targets)
        return len(self.data)

    def get_x_bar(self):
        try:
            return self.x_bar
        except AttributeError:
            x_bar = 0
            for sample in self.data:
                x_bar += sample[0]
            self.x_bar = x_bar / self.num_data
            return self.x_bar

class ADNIDataloader:
    def __init__(self, data1, data2, batch_size):

        self.data1 = data1
        self.data2 = data2

        self.dataloader1 = DataLoader(self.data1, batch_size=batch_size, shuffle=True)
        self.dataloader2 = DataLoader(self.data2, batch_size=batch_size, shuffle=True)
    def __iter__(self):
        return zip(enumerate(self.dataloader1), enumerate(self.dataloader2))

    def __len__(self):
        # assert len(self.dataloader1) == len(self.dataloader2)
        return len(self.dataloader1)

def adni_data():
    mri_dir = '../compfs/datasets/adni_data/mri_label_CV_2_Train.csv'
    mri = pd.read_csv(mri_dir)
    labels = mri.iloc[:, 0]
    mri = mri.iloc[:, 1:]
    mri = mri.values.astype(np.float32)
    labels = torch.from_numpy(labels.values.astype(np.float32))
    train_mri = ADNI(mri, labels)
    mri_dir = '../compfs/datasets/adni_data/mri_label_CV_2_Test.csv'
    mri = pd.read_csv(mri_dir)
    labels = mri.iloc[:, 0]
    mri = mri.iloc[:, 1:]
    mri = mri.values.astype(np.float32)
    labels = labels.values.astype(np.float32)
    test_mri = ADNI(mri, labels)
    snps_dir = '../compfs/datasets/adni_data/snps_label_CV_2_Train.csv'
    snps = pd.read_csv(snps_dir)
    snps = snps.iloc[:, 1:]
    snps = snps.values.astype(np.float32)
    train_snps = ADNI(snps)
    snps_dir = '../compfs/datasets/adni_data/snps_label_CV_2_Test.csv'
    snps = pd.read_csv(snps_dir)
    snps = snps.iloc[:, 1:]
    snps = snps.values.astype(np.float32)
    test_snps = ADNI(snps)
    return [train_snps, train_mri], [test_snps, test_mri]
