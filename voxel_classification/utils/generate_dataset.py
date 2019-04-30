import sys
sys.path.insert(0, '../src')

import numpy as np
import os.path as osp
from config import cfg
from sklearn.preprocessing import StandardScaler
import os
import yaml
from sklearn.externals import joblib
from extract_scan import extract_scan
from joblib import Parallel, delayed
import multiprocessing
import tqdm

# Structure: [file_name,start_scan,end_scan] -1 value will use default values (start=0, end=last)

# Training
# datasets = [
#     ['11-smoke',-1,-1],
#     ['12-smoke',-1,-1],
#     ['13-smoke',-1,-1],
#     ['17-smoke',-1,-1],
#     ['2-dust',-1,-1],
#     ['8-dust',-1,-1],
#     ['4-dust',-1,-1],
#     ['7-dust',-1,-1],
#     ['9-smoke',-1,-1],
# ]

# Validation
datasets = [
    ['19-smoke',-1,300],
    ['3-dust',-1,300],
]

# Testing
# datasets = [
#     ['10-smoke',-1,-1],
#     ['13-smoke',-1,-1],
#     ['15-smoke',-1,-1],
#     ['16-smoke',-1,-1],
#     ['18-smoke',-1,-1],
#     ['1-dust',-1,-1],
#     ['5-dust',-1,-1],
#     ['6-dust',-1,-1],
# ]

# Use cases

# datasets = [
#     ['10_pred',-1,-1],
#     ['13_pred',-1,-1],
#     ['15_pred',-1,-1],
#     ['16_pred',-1,-1],
#     ['18_pred',-1,-1],
#     ['1_pred',-1,-1],
#     ['5_pred',-1,-1],
#     ['6_pred',-1,-1],
# ]

parameters = {
    "shuffle": False,  # Shuffle data for training only
    "datasets": datasets,
    "separate_fog_dust": False,  # consider fog and dust as two different labels (fog=1, dust=2)
    "features": [
        'intensity',
        'echo_one_hot',
        'rel_pos',
    ],
    "max_scans": 1000
}

dataset_name = 'val' # dataset name

dataset_dir = osp.join(cfg.DATASETS_DIR, dataset_name)
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)
    os.makedirs(osp.join(dataset_dir, 'scan_voxels'))
    os.makedirs(osp.join(dataset_dir, 'scan_pcls'))

# datasets = [osp.join(data_dir,d) for d in datasets]

parameters['dataset_size'] = 0
split_id = 0
scaler = StandardScaler()

scan_id = 0
lookup_table = np.array([],dtype=int).reshape((0,2))

for idx, [d, start_id, end_id] in enumerate(datasets):

    print("Generating voxels for dataset: %s..." % d)

    pcls = np.load(osp.join(cfg.RAW_DATA_DIR, d + '_converted.npy'))

    if start_id < 0:
        start_id = 0

    if end_id < 0 or (end_id - start_id) > parameters['max_scans']:
        if start_id + parameters['max_scans'] < pcls.shape[0]:
            end_id = start_id + parameters['max_scans']
        else:
            end_id = pcls.shape[0]

    pcls = pcls[start_id:end_id]

    voxels = []
    coords = []
    labels = []
    raw_pcl = []

    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores)(delayed(extract_scan)(pcl, parameters['features']) for pcl in tqdm.tqdm(
        pcls, total=pcls.shape[0], desc='Generating Voxel Data:', ncols=80, leave=False))
    for buffer in results:
        voxels.append(buffer[0])
        coords.append(buffer[1])
        labels.append(buffer[2])
        raw_pcl.append(buffer[3])
    del results
    del pcls

    print("Post Processing dataset: %s" % d)

    if parameters['shuffle']:
        voxels = np.asarray(voxels)
        labels = np.asarray(labels)
        coords = np.asarray(coords)
        raw_pcl = np.asarray(raw_pcl)
        p = np.random.permutation(voxels.shape[0])
        voxels = voxels[p]
        coords = coords[p]
        labels = labels[p]
        raw_pcl = raw_pcl[p]

    parameters['features_size'] = voxels[0].shape[2]

    # Process each scan
    for v, c, l, pcl in zip(voxels, coords, labels, raw_pcl):

        if not parameters['separate_fog_dust']:
            # Check if fog or dust file
            particle_label = np.max(l)
            # Check that there is at least one particle label in scan
            if particle_label > 0:
                # If so, overwrites the particle label to 1
                output_label = 1
                l[l == particle_label] = output_label
                pcl[pcl[:, 5] == particle_label,5] = output_label

        # Update scaler for dataset
        scaler.partial_fit(v.reshape(-1,v.shape[2]))

        # Save point cloud scan only when full point cloud is computed for voxels
        np.save(osp.join(dataset_dir, 'scan_pcls', str(scan_id)), pcl)
        np.save(osp.join(dataset_dir, 'scan_voxels', 'voxels_' + str(scan_id)), v)
        np.save(osp.join(dataset_dir, 'scan_voxels', 'labels_' + str(scan_id)), l)
        np.save(osp.join(dataset_dir, 'scan_voxels', 'coords_' + str(scan_id)), c)

        parameters["dataset_size"] += v.shape[0]
        scan_id += 1

parameters["nb_scans"] = scan_id

# Save parameters in yaml file
with open(osp.join(dataset_dir, 'config.yaml'), 'w') as f:
    yaml.safe_dump(parameters, f, default_flow_style=False)

# Save scaler
joblib.dump(scaler, osp.join(dataset_dir, 'scaler.pkl'))
