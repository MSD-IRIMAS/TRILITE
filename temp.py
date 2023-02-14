import tensorflow as tf
from utils.utils import load_data, znormalisation, encode_labels, split_ypred
import numpy as np


gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

xtrain, ytrain, xtest, ytest = load_data(file_name='UWaveGestureLibraryZ')

xtrain = znormalisation(xtrain)
xtest = znormalisation(xtest)

ytrain = encode_labels(ytrain)
ytest = encode_labels(ytest)

old_path = 'results/fcn/run_4/UWaveGestureLibraryZ/'

model = tf.keras.models.load_model(old_path+'last_model.hdf5',compile=False)
ypred_train = model.predict([xtrain,xtrain,xtrain])
ypred_test = model.predict([xtest,xtest,xtest])

new_xtrain, new_xtest = split_ypred(ypred_train=ypred_train,ypred_test=ypred_test)

np.save(arr=new_xtest,file=old_path+'v_test.npy')