import tensorflow as tf
import numpy as np
from encoders.fcn import FCN
from encoders.inception import INCEPTION
from encoders.resnet import RESNET
from triplet_loss import triplet_loss_function
import matplotlib.pyplot as plt
from utils.augmentation import triplet_generation


class MODEL:

    def __init__(self,length_TS,n_dim,encoder_name,output_directory,alpha=1e-2,epochs=1000,batch_size=32):

        self.length_TS = length_TS
        self.n_dim = n_dim
        self.alpha = alpha
        self.encoder_name = encoder_name

        self.output_directory = output_directory
        
        self.epochs = epochs
        self.batch_size = batch_size

        self.build_model()

    def build_model(self):

        input_shape = (self.length_TS,)

        refference_input = tf.keras.layers.Input(input_shape)
        positive_input = tf.keras.layers.Input(input_shape)
        negative_input = tf.keras.layers.Input(input_shape)

        if self.encoder_name == 'inception':
            encoder = INCEPTION(input_shape=input_shape,n_dim=self.n_dim)
        
        elif self.encoder_name == 'fcn':
            encoder = FCN(input_shape=input_shape,n_dim=self.n_dim)
        
        elif self.encoder_name == 'resnet':
            encoder = RESNET(input_shape=input_shape,n_dim=self.n_dim)

        else:
            raise ValueError("No module named "+self.encoder_name)

        model_to_use_as_layers = encoder.model

        refference_output = model_to_use_as_layers(refference_input)
        positive_output = model_to_use_as_layers(positive_input)
        negative_output = model_to_use_as_layers(negative_input)


        all_layers_combined = tf.keras.layers.concatenate([refference_output,positive_output,negative_output],axis=2)

        self.model = tf.keras.models.Model(inputs=[refference_input,positive_input,negative_input],outputs=all_layers_combined)

        my_learning_rate = 0.001
        my_optimizer = tf.keras.optimizers.Adam(learning_rate=my_learning_rate)

        my_loss = triplet_loss_function(alpha=self.alpha)

        self.model.compile(loss=my_loss,optimizer=my_optimizer)

    def fit(self,xtrain,xval):

        min_loss = 1e9

        loss = []
        val_loss = []

        for _epoch in range(self.epochs):

            ref_train, pos_train, neg_train = triplet_generation(xtrain)
            # ref_val, pos_val, neg_val = triplet_generation(xval)

            hist = self.model.fit([ref_train,pos_train,neg_train],np.zeros(shape=ref_train.shape),
                                   epochs=1,batch_size=self.batch_size,verbose=False)
            
            loss.append(hist.history['loss'][0])
            # val_loss.append(hist.history['val_loss'][0])

        self.model.save(self.output_directory+'last_model.hdf5')
            
        plt.figure(figsize=(20,10))

        plt.plot(loss,lw=3,color='blue',label='train loss')
        # plt.plot(val_loss,lw=3,color='red',label='val loss')

        plt.legend()
        plt.savefig(self.output_directory+'loss.png')

    def predict(self,x):

        return self.model.predict([x,x,x])