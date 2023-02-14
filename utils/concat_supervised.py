import tensorflow as tf
import numpy as np

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def split_to_batches(x,batch_size=16):

    n = int(x.shape[0])
    batches = []
    for i in range(0,n-batch_size+1,batch_size):
        batches.append(x[i:i+batch_size])
    if n % batch_size > 0:
        batches.append(x[i+batch_size:n])
    return batches

def rejoin_batches(x,n):

    m = len(x)
    x_rejoin = np.zeros(shape=(n,len(x[0][0][0])))
    filled = 0

    for i in range(m):
        _stop = len(x[i][0])
        x_rejoin[filled:filled+_stop,:] = x[i][0]
        filled += len(x[i][0])
    
    return x_rejoin

def concatenate_supervised_unsupervised(path_model_supervised,xtrain,xtest,vtrain,vtest):

    n = int(xtrain.shape[0])
    l = int(vtrain.shape[1])

    new_xtrain = np.zeros(shape=(n,l*2))
    n = int(xtest.shape[0])
    new_xtest = np.zeros(shape=(n,l*2))

    supervised_model = tf.keras.models.load_model(path_model_supervised)

    input_supervised = supervised_model.input
    outputs_supervised = [layer.output for layer in supervised_model.layers[:-1]]
    functors_supervised = [tf.keras.backend.function([input_supervised], [out]) for out in outputs_supervised]

    batches_train = split_to_batches(xtrain)
    batches_test = split_to_batches(xtest)

    # batches_train = np.asarray(batches_train,dtype=np.float64)

    layers_outputs_supervised_train = []
    layers_outputs_supervised_test = []

    for batch in batches_train:
        # print(len(batch))
        layers_outputs_supervised_train.append(functors_supervised[-1](batch))
    
    for batch in batches_test:
        layers_outputs_supervised_test.append(functors_supervised[-1](batch))

    layers_outputs_supervised_train = rejoin_batches(x=layers_outputs_supervised_train,n=int(xtrain.shape[0]))
    layers_outputs_supervised_test = rejoin_batches(x=layers_outputs_supervised_test,n=int(xtest.shape[0]))

    new_xtrain[:,0:l] = layers_outputs_supervised_train
    new_xtest[:,0:l] = layers_outputs_supervised_test

    new_xtrain[:,l:] = vtrain
    new_xtest[:,l:] = vtest

    return new_xtrain, new_xtest


