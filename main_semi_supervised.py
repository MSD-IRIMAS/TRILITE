import sys
import os
import numpy as np
from utils.utils import load_data, semi_supervised_indices, znormalisation, create_directory, encode_labels, split_ypred,draw,draw_before
from model import MODEL

def extract_args():

    if len(sys.argv) != 7:
        raise ValueError("No options were specified")
    
    else:

        return sys.argv[2], sys.argv[4], int(sys.argv[6])
    
    # example : python3 main.py -e inception -d Coffee -p 0.2


if __name__ == "__main__":

    runs = 5

    semi_experiments = 5

    n_dim = 128

    encoder_name, file_name, perc = extract_args()

    output_directory_parent = 'results_semi_'+str(perc)+'/'
    create_directory(output_directory_parent)

    output_directory_parent = output_directory_parent + encoder_name + '/'
    create_directory(output_directory_parent)

    xtrain, ytrain, xtest, ytest = load_data(file_name=file_name)

    print(file_name,xtrain.shape)

    xtrain = znormalisation(xtrain)
    xtest = znormalisation(xtest)

    l = int(xtrain.shape[1])
    
    classification_score = []

    for _run in range(runs):

        output_directory = output_directory_parent + 'run_'+str(_run) + '/'
        create_directory(output_directory)
        output_directory = output_directory + file_name + '/'
        create_directory(output_directory)

        for _exp in range(semi_experiments):
        
            exp_output_directory = output_directory + 'exp_' + str(_exp) + '/'
            if os.path.exists(exp_output_directory+'v_train.npy'):
                continue
            create_directory(exp_output_directory)

            semi_indices = semi_supervised_indices(xtrain=xtrain,ytrain=ytrain,perc=perc/100)

            np.save(file=exp_output_directory+'/train_indices.npy',arr=semi_indices)
            semi_xtrain = xtrain[semi_indices,:]

            model = MODEL(length_TS=l,n_dim=n_dim,encoder_name=encoder_name,output_directory=exp_output_directory)

            model.fit(xtrain=semi_xtrain,xval=xtest)

            ypred_train = model.predict(semi_xtrain)
            ypred_test = model.predict(xtest)

            # draw(ypred_test=ypred_test,labels_test=ytest,output_directory=exp_output_directory)
            # draw_before(xtest=xtest,ytest=ytest,output_directory=exp_output_directory)

            new_xtrain, new_xtest = split_ypred(ypred_train=ypred_train,ypred_test=ypred_test)

            new_xtrain = np.asarray(new_xtrain)
            new_xtest = np.asarray(new_xtest)

            np.save(arr=new_xtrain,file=exp_output_directory+'v_train.npy')
            np.save(arr=new_xtest,file=exp_output_directory+'v_test.npy')