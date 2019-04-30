#!/usr/bin/env python

import sys
sys.path.insert(0, '../utils')

import os
import os.path as osp
import numpy as np
import torch
import tqdm
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report
import yaml
from model import Model
from particle_dataset import ParticleDataset
from segment_scan import segment_scan
from config import cfg
from datetime import datetime
from sklearn.externals import joblib
from torch.utils.data import DataLoader
from torch.autograd import Variable

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

# Models
input_models = [
    '20190429_152306_test_model'
]

# Evaluation parameters
save_pcl = True
eval_title = 'val'
test_data = 'val'

now = datetime.now()

for m in input_models:
    # Load model parameters
    with open(osp.join(cfg.LOG_DIR, m, 'config.yaml'), 'r') as f:
        parameters = yaml.load(f,Loader=yaml.SafeLoader)

    eval_dir = osp.join(cfg.LOG_DIR, m, 'evaluations/eval_' + now.strftime('%Y%m%d_%H%M%S') + '_' + eval_title)
    os.makedirs(eval_dir)
    # os.makedirs(osp.join(eval_dir,'scans'))

    criterion = nn.CrossEntropyLoss()
    # Choose device for computation
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    model = Model(features_in=parameters["features_size"])
    # Load the best model saved
    model_state = osp.join(cfg.LOG_DIR, m, 'best.pth.tar')
    model_state = torch.load(model_state)
    model.load_state_dict(model_state['model_state_dict'])
    model.eval()
    model.to(device)

    # ParticleDataset
    scaler = joblib.load(osp.join(cfg.LOG_DIR,m,'scaler.pkl'))

    # Load dataset parameters
    with open(osp.join(cfg.DATASETS_DIR, test_data, 'config.yaml'), 'r') as f:
        ds_parameters = yaml.load(f,Loader=yaml.SafeLoader)
    nb_scans = ds_parameters['nb_scans']




    y_target = []
    y_pred = []
    val_loss = 0

    eval_pcl = []

    test_loader = DataLoader(ParticleDataset(dataset_dir=osp.join(cfg.DATASETS_DIR, test_data), scaler=scaler), batch_size=1, shuffle=False, num_workers=4, collate_fn=detection_collate)

    for s, sample in tqdm.tqdm(enumerate(test_loader), total=len(test_loader),
                               desc='Evaluating scan', ncols=80,
                               leave=False):

        pcl = np.load(osp.join(cfg.DATASETS_DIR, test_data, 'scan_pcls', str(s) + '.npy'))

        inputs = Variable(torch.FloatTensor(sample['inputs']))
        coords = Variable(torch.LongTensor(sample['coords']))
        labels = Variable(torch.LongTensor(sample['labels']))

        inputs, coords, labels = inputs.to(device), coords.to(device), labels.to(device)

        # forward
        with torch.no_grad():
            pred = model(inputs, coords)

        [proba, index] = pred.max(dim=1)

        pred = index.cpu().data.numpy()

        # convert predicted voxels back into pointcloud
        raw_pred_points = segment_scan(pcl, pred)
        # np.save(osp.join(eval_dir, 'scans', str(s)), raw_pred_points[:, :7])

        # Removes any scan with ground truth = 1 for whole scan
        if np.sum((raw_pred_points[:, 5] == 1).astype(np.int8)) / float(raw_pred_points.shape[0]) < 0.5:
            if save_pcl:
                eval_pcl.append(raw_pred_points[:, :7])
            y_pred.append(raw_pred_points[:, 6])
            y_target.append(raw_pred_points[:, 5])

    if save_pcl:
        np.save(osp.join(eval_dir,'scans'),eval_pcl)
    y_pred = np.concatenate(y_pred)
    y_target = np.concatenate(y_target)

    with open(osp.join(eval_dir, 'eval_results.txt'), 'w') as f:

        f.write('Evaluation parameters:\n')
        f.write('ParticleDataset: %s\n' % test_data)
        f.write('nb_scans: %s\n' % ds_parameters['nb_scans'])
        f.write('dataset_size: %s\n' % ds_parameters['dataset_size'])
        f.write('\n\nEvaluation results:\n')

        # Compute performance scores
        print('\n')
        print("Evaluation results for model: %s" % m)

        print('Confusion Matrix')
        f.write("Confusion Matrix\n")

        cnf_matrix = confusion_matrix(y_target, y_pred).astype(np.float32)  # Compute confusion matrix
        cnf_matrix /= cnf_matrix.sum(1, keepdims=True)  # put into ratio
        print(cnf_matrix)
        f.write(str(cnf_matrix))
        f.write('\n')

        # Can only use this if both classes are at least predicted once
        if len(np.unique(y_pred)) > 1:
            print('Classification Report')
            f.write('Classification Report\n')

            cr = classification_report(y_target, y_pred)
            print(cr)
            f.write(cr)
            f.write('\n')


