import tensorflow as tf
import numpy as np
import time

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

class RESNET:

    def __init__(self, input_shape, n_dim):
        
        self.input_shape = input_shape
        self.n_dim = n_dim
        self.build_model()


    def build_model(self):

        n_feature_maps = 64

        input_layer = tf.keras.layers.Input(self.input_shape)

        reshape = tf.keras.layers.Reshape(target_shape=(self.input_shape[0],1),name='fff')(input_layer)

        # BLOCK 1

        conv_x = tf.keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(reshape)
        conv_x = tf.keras.layers.BatchNormalization()(conv_x)
        conv_x = tf.keras.layers.Activation('relu')(conv_x)

        conv_y = tf.keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
        conv_y = tf.keras.layers.BatchNormalization()(conv_y)
        conv_y = tf.keras.layers.Activation('relu')(conv_y)

        conv_z = tf.keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
        conv_z = tf.keras.layers.BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = tf.keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(reshape)
        shortcut_y = tf.keras.layers.BatchNormalization()(shortcut_y)

        output_block_1 = tf.keras.layers.add([shortcut_y, conv_z])
        output_block_1 = tf.keras.layers.Activation('relu')(output_block_1)

        # BLOCK 2

        conv_x = tf.keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
        conv_x = tf.keras.layers.BatchNormalization()(conv_x)
        conv_x = tf.keras.layers.Activation('relu')(conv_x)

        conv_y = tf.keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = tf.keras.layers.BatchNormalization()(conv_y)
        conv_y = tf.keras.layers.Activation('relu')(conv_y)

        conv_z = tf.keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = tf.keras.layers.BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = tf.keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
        shortcut_y = tf.keras.layers.BatchNormalization()(shortcut_y)

        output_block_2 = tf.keras.layers.add([shortcut_y, conv_z])
        output_block_2 = tf.keras.layers.Activation('relu')(output_block_2)

        # BLOCK 3

        conv_x = tf.keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2)
        conv_x = tf.keras.layers.BatchNormalization()(conv_x)
        conv_x = tf.keras.layers.Activation('relu')(conv_x)

        conv_y = tf.keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = tf.keras.layers.BatchNormalization()(conv_y)
        conv_y = tf.keras.layers.Activation('relu')(conv_y)

        conv_z = tf.keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = tf.keras.layers.BatchNormalization()(conv_z)

        # no need to expand channels because they are equal
        shortcut_y = tf.keras.layers.BatchNormalization()(output_block_2)

        output_block_3 = tf.keras.layers.add([shortcut_y, conv_z])
        output_block_3 = tf.keras.layers.Activation('relu')(output_block_3)

        # FINAL

        gap_layer = tf.keras.layers.GlobalAveragePooling1D()(output_block_3)

        output_layer = tf.keras.layers.Reshape(target_shape=(self.n_dim,1))(gap_layer)

        self.model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)