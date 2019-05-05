# TensorFlow and tf.keras
import tensorflow as tf
import tensorflow.contrib as tfcontrib
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import models
from tensorflow.python.keras import backend as K
from tensorflow.python.client import session as sess
import matplotlib.pyplot as plt
import time

# Helper libraries
import numpy as np
import os
import glob
import zipfile
import functools
import h5py
from scipy.misc import imresize


load_weights_file = "/home/juli/Desktop/models/weights_big_dataset_single_point_cloud_geometry.hdf5"
load_file_test = "/media/juli/98F29C83F29C67721/SemesterProject/data/2_validation/kitti_2011_09_26_drive_0001_synced_with_rings_labeled_img_3125"
load_file_test = "/home/juli/Desktop/1-dust_labeled_spaces_img"
#img_shape = (32, 2144, 4)
#number_channels = 2
batch_size = 2
#width_pixel = 2144

images_test = np.load(load_file_test + ".npy")
#images_test = images_test[:,::-1]
#images_test = images_test[:,:32]
#images_test = images_test[:,0::2]
print images_test.shape
# Find out appropriate width of image
# To be as general as possible
width_pixel = len(images_test[0,0,:])
width_pixel = width_pixel - width_pixel % 16 # Because the network only supports multiple widths of 16 (pooling)

if "dual" in load_weights_file:
    features = images_test[:,:,:,[0, 4, 6, 10]]
    if "geometry" in load_weights_file:
        features = images_test[:, :, :, [0, 6]]
    elif "intensities" in load_weights_file:
        features = images_test[:, :, :, [4, 10]]
else:
    features = images_test[:, :, :, [0, 4]]
    if "geometry" in load_weights_file:
        features = images_test[:, :, :, [0]]
    elif "intensities" in load_weights_file:
        features = images_test[:, :, :, [4]]
labels = images_test[:, :, :, len(images_test[0,0,0,:])-3:]
features = features[:,:,:width_pixel,:]
labels = labels[:,:,:width_pixel,:]

# Resizing approach BAD RESULTS SO FAR
#new_features = np.zeros([features.shape[0],features.shape[1],width_pixel,features.shape[3]])
#for i, feature in enumerate(features):
    #new_features[i,:,:,0] = imresize(feature[:,:,0], [features.shape[1],width_pixel])
#features = new_features
#del new_features

print features.shape

img_shape = (features.shape[1:4])

# Building the model
def conv_block(input_tensor, num_filters):
    encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Activation('relu')(encoder)
    encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(encoder)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Activation('relu')(encoder)
    return encoder


def encoder_block(input_tensor, num_filters):
    encoder = conv_block(input_tensor, num_filters)
    encoder_pool = layers.MaxPooling2D((1, 2), strides=(1, 2))(encoder)

    return encoder_pool, encoder


def decoder_block(input_tensor, concat_tensor, num_filters):
    decoder = layers.Conv2DTranspose(num_filters, (1, 2), strides=(1, 2), padding='same')(input_tensor)
    decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation('relu')(decoder)
    decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation('relu')(decoder)
    decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation('relu')(decoder)
    return decoder

inputs = layers.Input(shape=img_shape)
# 2144
encoder0_pool, encoder0 = encoder_block(inputs, 16)
# 1072
encoder1_pool, encoder1 = encoder_block(encoder0_pool, 32)
# 536
encoder2_pool, encoder2 = encoder_block(encoder1_pool, 64)
# 268
encoder3_pool, encoder3 = encoder_block(encoder2_pool, 128)
# 134
center = conv_block(encoder3_pool, 256)
# center
decoder3 = decoder_block(center, encoder3, 128)
# 268
decoder2 = decoder_block(decoder3, encoder2, 64)
# 536
decoder1 = decoder_block(decoder2, encoder1, 32)
# 1072
decoder0 = decoder_block(decoder1, encoder0, 16)
# 2144
outputs = layers.Conv2D(3, (1, 1), activation='softmax')(decoder0)

model = models.Model(inputs=[inputs], outputs=[outputs])

def dice_coeff(y_true, y_pred):
    smooth = 1.
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

def bce_dice_loss(y_true, y_pred):
    loss = losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss

model.compile(optimizer='adam', loss=bce_dice_loss, metrics=[dice_loss]) # Build new model

model.load_weights(load_weights_file) # And only load weights

def generator(features, labels):
    while True:
        for i in range(int(np.floor(len(features) / float(batch_size)))): # -1 just for the one case
            feature_new = np.zeros((batch_size, features[batch_size*i].shape[0], width_pixel,
                                    features[batch_size*i].shape[2]))
            label_new = np.zeros((batch_size, labels[batch_size*i].shape[0], width_pixel,
                                    labels[batch_size*i].shape[2]))
            for c in range(batch_size):
                #if batch_size*i+c >= len(features):
                    #break
                feature_new[c] = features[batch_size*i+c]
                label_new[c] = labels[batch_size*i+c]
            yield feature_new, label_new

# Image representation
output = np.zeros([batch_size*int(np.floor(len(features) / float(batch_size))),images_test.shape[1], images_test.shape[2],
                   images_test.shape[3]])
output[:,:,:,0:output.shape[3]-3] = images_test[:batch_size*int(np.floor(len(features) / float(batch_size))),:,:,0:output.shape[3]-3]
del images_test

#for i in range(len(features)):
    #batch_of_imgs, label = tf.keras.backend.get_session().run(next_element)
    #predicted_label = model.predict(batch_of_imgs, steps=1)
    #output[i,:,:2144,12] = predicted_label[0,:,:,0]
    #print(i)
predicted_label = model.predict_generator(generator(features, labels),
                              steps=int(np.floor(len(features) / float(batch_size))))
print predicted_label.shape
output[:,:,:width_pixel,output.shape[3]-3:] = predicted_label[:,:,:,:]
output[:,:,width_pixel:,output.shape[3]-3:] = np.asarray([1,0,0]) # For all non labeled pixels assume non smoke and non dust

np.save(load_file_test + "_predicted.npy", output)

