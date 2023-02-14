import tensorflow as tf


gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

class FCN:

    def __init__(self,input_shape,n_dim):

        self.input_shape = input_shape
        self.n_dim = n_dim
        self.build_model()
    
    def build_model(self):

        self.input_layer = tf.keras.layers.Input(self.input_shape)

        self.reshape = tf.keras.layers.Reshape(target_shape=(self.input_shape[0],1),name='fff')(self.input_layer)

        self.conv1 = tf.keras.layers.Conv1D(filters=128,kernel_size=8,padding='same')(self.reshape)
        self.bn1 = tf.keras.layers.BatchNormalization()(self.conv1)
        self.relu1 = tf.keras.layers.Activation(activation='relu')(self.bn1)

        self.conv2 = tf.keras.layers.Conv1D(filters=256,kernel_size=5,padding='same')(self.relu1)
        self.bn2 = tf.keras.layers.BatchNormalization()(self.conv2)
        self.relu2 = tf.keras.layers.Activation(activation='relu')(self.bn2)

        self.conv3 = tf.keras.layers.Conv1D(filters=128,kernel_size=3,padding='same')(self.relu2)
        self.bn3 = tf.keras.layers.BatchNormalization()(self.conv3)
        self.relu3 = tf.keras.layers.Activation(activation='relu')(self.bn3)

        self.gap = tf.keras.layers.GlobalAveragePooling1D()(self.relu3)

        self.output_layer = tf.keras.layers.Reshape(target_shape=(self.n_dim,1))(self.gap)

        self.model = tf.keras.models.Model(inputs=self.input_layer,outputs=self.output_layer)