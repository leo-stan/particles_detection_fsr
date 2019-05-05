# TensorFlow and tf.keras
import tensorflow as tf
import tensorflow.contrib as tfcontrib
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import models
from tensorflow.python.keras import backend as K
from tensorflow.python.client import session as sess

def return_model(img_shape, load_weights_file):
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

    model.compile(optimizer='adam', loss=bce_dice_loss, metrics=[dice_loss])  # Build new model

    model.load_weights(load_weights_file)  # And only load weights
    #model.summary()
    return model