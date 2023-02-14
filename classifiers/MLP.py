import tensorflow as tf
import numpy as np
from sklearn.preprocessing import OneHotEncoder as OHE


class MLP_Classifier:

    def __init__(self,xtrain,ytrain,lr_0=0.01,epochs=1000):

        self.lr_0 = lr_0

        self.epochs = 1000
        self.xtrain = xtrain
        self.ytrain = ytrain

        self.batch_size = 16

        self.mini_batch_size = int(min(self.xtrain.shape[0]/10, self.batch_size))

        self.build_model()

    def build_model(self):
    
        self.input_layer = tf.keras.layers.Input(self.xtrain.shape[1:])
        self.output_layer = tf.keras.layers.Dense(units=len(np.unique(self.ytrain)),activation='softmax')(self.input_layer)

        self.model = tf.keras.models.Model(inputs=self.input_layer,outputs=self.output_layer)

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, 
			min_lr=0.0001)

        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='temp_best_model.hdf5', monitor='loss', 
        save_best_only=True)

        self.callbacks = [reduce_lr,model_checkpoint]

        self.model.compile(loss='categorical_crossentropy',optimizer=tf.keras.optimizers.SGD(learning_rate=self.lr_0))

    def fit(self):

        ohe = OHE(sparse=False)

        self.ytrain = np.expand_dims(self.ytrain,axis=1)

        self.ytrain = ohe.fit_transform(self.ytrain)

        self.model.fit(self.xtrain,self.ytrain,epochs=self.epochs,batch_size=self.mini_batch_size,callbacks=self.callbacks,
                        verbose=True)
    
    def predict(self,xtest):

        self.model = tf.keras.models.load_model('temp_best_model.hdf5')

        ypred = self.model.predict(xtest)

        return np.argmax(ypred,axis=1)