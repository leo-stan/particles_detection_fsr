#!/usr/bin/env python

import sys
sys.path.insert(0, '../model')
sys.path.insert(0, '../utils')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Model
from particle_dataset import ParticleDataset
from config import cfg
import os
from datetime import datetime
import os.path as osp
from trainer import Trainer
import yaml
from sklearn.externals import joblib
import numpy as np


def detection_collate(batch):
    voxel_features = []
    voxel_coords = []
    labels = []

    for i, sample in enumerate(batch):
        voxel_features.append(sample['inputs'])

        voxel_coords.append(
            np.pad(sample['coords'], ((0, 0), (1, 0)),
                mode='constant', constant_values=i))

        labels.append(sample['labels'])

    return {
        'inputs': np.concatenate(voxel_features),
        'coords': np.concatenate(voxel_coords),
        'labels': np.concatenate(labels)}

if __name__ == '__main__':

    root_dir = cfg.ROOT_DIR
    datasets_dir = cfg.DATASETS_DIR

    output_model = 'test_model'

    train_data = 'use_case'
    val_data = 'val'

    now = datetime.now()
    ensemble = []
    criterion = nn.CrossEntropyLoss()

    with open(osp.join(datasets_dir, train_data, 'config.yaml'), 'r') as f:
        dataset_params = yaml.load(f,Loader=yaml.SafeLoader)
    # Check what hardware is available
    assert torch.cuda.is_available()
    device = torch.device("cuda:0")
    print('Device used: ' + device.type)

    # Parameters
    parameters = {
        "resume_model": '',
        "train_data": train_data,
        "val_data": val_data,
        "max_iterations": 10000,
        "interval_validate": 500,
        "batch_size": 1,
        "features": dataset_params['features'],
        "features_size": dataset_params['features_size'],
        "lr": 0.001,
        "betas": (0.9, 0.999),
        "eps": 1e-08,
        "weight_decay": 0,
    }

    train_data = osp.join(datasets_dir, train_data)
    val_data = osp.join(datasets_dir, val_data)

    output_path = osp.join(cfg.LOG_DIR, now.strftime('%Y%m%d_%H%M%S')+'_'+output_model)

    # Save training parameters
    os.makedirs(output_path)

    # Datasets
    train_loader = DataLoader(ParticleDataset(dataset_dir=train_data), batch_size=parameters['batch_size'], shuffle=False, num_workers=4, collate_fn=detection_collate)

    val_loader = DataLoader(ParticleDataset(dataset_dir=val_data, scaler=train_loader.dataset.scaler), batch_size=parameters['batch_size'], shuffle=False, num_workers=4, collate_fn=detection_collate)

    # Models
    model = Model(features_in=parameters["features_size"]).to(device)

    if parameters['resume_model'] != '':
        checkpoint = torch.load(osp.join(cfg.LOG_DIR,parameters['resume_model'],'checkpoint.pth.tar'))
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        start_iteration = checkpoint['iteration']
    else:
        start_epoch = 0
        start_iteration = 0

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=parameters['lr'], betas=parameters['betas'], eps=parameters['eps'], weight_decay=parameters['weight_decay'], amsgrad=False)
    if parameters['resume_model'] != '':
        optimizer.load_state_dict(checkpoint['optim_state_dict'])

    # Save model parameters
    with open(osp.join(output_path, 'config.yaml'), 'w') as f:
        yaml.safe_dump(parameters, f, default_flow_style=False)
    joblib.dump(train_loader.dataset.scaler,osp.join(output_path,'scaler.pkl'))

    # Start training
    trainer = Trainer(
        device=device,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        out=output_path,
        max_iter=parameters['max_iterations'],
        interval_validate=parameters['interval_validate'],
    )
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()
