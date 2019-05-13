# Voxel Classification of Airborne Particles

## Data preparation
1. Download dataset from https://cloudstor.aarnet.edu.au/plus/s/oQwj9AkaLlqNU1a
2. Convert dataset using convert_dataset.py
3. Fill in paths in src/config.py
4. Generate voxel datasets from files using generate_dataset.py
5. Datasets can be visualised in Rviz (ROS) using visualise_dataset_pcl.py and visualise_dataset_voxel.py

## Model Training

- Run train_model.py providing a training and validation dataset created above

## Model Evaluation

- Evaluate any model using evaluate_model.py by providing a test dataset
- The results and predicted data is saved in the model folder under 'evaluations'
- The evaluation can be visualised in Rviz (ROS) using visualise_predicted_pcl.py

