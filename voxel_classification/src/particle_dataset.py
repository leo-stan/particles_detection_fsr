#!/usr/bin/env python

import numpy as np
from torch.utils.data import Dataset
import os.path as osp
import yaml
from sklearn.externals import joblib

class ParticleDataset(Dataset):
    """Lidar dataset"""

    def __init__(self, dataset_dir, scaler = None):

        self.data = []
        self.labels = []
        self.dataset_dir = dataset_dir

        with open(osp.join(dataset_dir, 'config.yaml'), 'r') as f:
            self.params = yaml.load(f,Loader=yaml.SafeLoader)
        # If scaler provided then use it (prediction), otherwise use dataset scaler (training)
        if scaler:
            self.scaler = scaler
        else:
            self.scaler = joblib.load(osp.join(dataset_dir, 'scaler.pkl'))
        self.features = self.params['features']


    def __len__(self):
        return self.params['nb_scans']

    def __getitem__(self, idx):

        data =  np.load(osp.join(self.dataset_dir, 'scan_voxels', 'voxels_' + str(idx) + '.npy'))
        data = self.scaler.transform(data.reshape(-1,data.shape[2])).reshape(-1,data.shape[1],data.shape[2])
        labels = np.load(osp.join(self.dataset_dir, 'scan_voxels', 'labels_' + str(idx) + '.npy'))
        coords = np.load(osp.join(self.dataset_dir, 'scan_voxels', 'coords_' + str(idx) + '.npy'))

        sample = {'inputs': data,'coords': coords, 'labels': labels}

        return sample



