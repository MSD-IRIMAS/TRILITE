import sys
import os
import numpy as np
import pandas as pd
from utils.utils import load_data, znormalisation, create_directory, encode_labels, split_ypred,draw,draw_before
from model import MODEL
from apply_classifier import apply_classifier

def extract_args():

    if len(sys.argv) != 5:
        raise ValueError("No options were specified")
    
    else:

        return sys.argv[2], sys.argv[4]
    
    # example : python3 main.py -e inception -d Coffee


if __name__ == "__main__":


    runs = 5

    n_dim = 128

    encoder_name, file_name = extract_args()

    output_directory_parent = 'results/'
    create_directory(output_directory_parent)

    output_directory_parent = output_directory_parent + encoder_name + '/'
    create_directory(output_directory_parent)

    xtrain, ytrain, xtest, ytest = load_data(file_name=file_name)
    
    xtrain = znormalisation(xtrain)
    xtest = znormalisation(xtest)

    ytrain = encode_labels(ytrain)
    ytest = encode_labels(ytest)

    l = int(xtrain.shape[1])
    

    for _run in range(runs):
    
        output_directory = output_directory_parent + 'run_'+str(_run) + '/'
        create_directory(output_directory)
        output_directory = output_directory + file_name + '/'
        create_directory(output_directory)

        model = MODEL(length_TS=l,n_dim=n_dim,encoder_name=encoder_name,output_directory=output_directory)

        model.fit(xtrain=xtrain,xval=xtest)

        ypred_train = model.predict(xtrain)
        ypred_test = model.predict(xtest)

        # draw(ypred_test=ypred_test,labels_test=ytest,output_directory=output_directory)
        # draw_before(xtest=xtest,ytest=ytest,output_directory=output_directory)

        new_xtrain, new_xtest = split_ypred(ypred_train=ypred_train,ypred_test=ypred_test)

        new_xtrain = np.asarray(new_xtrain)
        new_xtest = np.asarray(new_xtest)

        np.save(arr=new_xtrain,file=output_directory+'v_train.npy')
        np.save(arr=new_xtest,file=output_directory+'v_test.npy')
