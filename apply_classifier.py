from matplotlib.pyplot import cla
import numpy as np
import pandas as pd
import sys
import os
import tensorflow as tf
from classifiers.KNN import KNN_Classifier
from classifiers.SVM import SVM_Classifier
from classifiers.MLP import MLP_Classifier
from classifiers.RIDGE import Ridge_Classifier
from utils.utils import load_data, znormalisation, encode_labels
from utils.concat_supervised import concatenate_supervised_unsupervised
from sklearn.metrics import accuracy_score

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# returns score

def extract_args():
    
    if len(sys.argv) != 7:
        raise ValueError("No options were specified")
    
    else:

        return sys.argv[2], sys.argv[4], sys.argv[6]
    
    # example : python3 main.py -e inception -o results/ -d Coffee


def apply_classifier(classifier_name,xtrain,ytrain,xtest,ytest):

    if classifier_name == 'knn':
        clf = KNN_Classifier()
    
    elif classifier_name == 'mlp':
        clf = MLP_Classifier(xtrain=xtrain,ytrain=ytrain,lr_0=0.1,epochs=2000)

    elif classifier_name == 'svm':
        clf = SVM_Classifier()

    elif classifier_name == 'ridge':
        clf = Ridge_Classifier()
    
    else:
        raise ValueError("no such classifier as "+classifier_name)
    
    if classifier_name != 'mlp':
        clf.fit(xtrain=xtrain,ytrain=ytrain)
    elif classifier_name == 'mlp':
        clf.fit()
    
    ypred = clf.predict(xtest=xtest)

    return accuracy_score(y_true=ytest,y_pred=ypred,normalize=True)




if __name__ == "__main__":

    encoder_name, output_directory, file_name = extract_args()

    print(file_name)

    xtrain, ytrain, xtest, ytest = load_data(file_name=file_name)

    ytrain = encode_labels(ytrain)
    ytest = encode_labels(ytest)

    runs = 5

    exps = 5

    if output_directory == 'results/':

        if os.path.exists(output_directory+encoder_name+'/results_ucr.csv'):
            df = pd.read_csv(output_directory+encoder_name+'/results_ucr.csv')
        else:
            df = pd.DataFrame(columns=['encoder','dataset','1-NN','1-NN-std','SVM','SVM-std',
                                        'RIDGE','RIDGE-std','1-LP','1-LP-std',
                                        'concatenate-supervised-1-LP','concatenate-supervised-1-LP-std'])

        Score_knn = []
        Score_svm = []
        Score_ridge = []
        Score_mlp = []
        Score_concat = []

        for _run in range(runs):

            vtrain = np.load(output_directory+encoder_name+'/run_'+str(_run)+'/'+file_name+'/v_train.npy')
            vtest = np.load(output_directory+encoder_name+'/run_'+str(_run)+'/'+file_name+'/v_test.npy')

            Score_knn.append(apply_classifier(classifier_name='knn',xtrain=vtrain,ytrain=ytrain,
                                                                    xtest=vtest,ytest=ytest))
            Score_svm.append(apply_classifier(classifier_name='svm',xtrain=vtrain,ytrain=ytrain,
                                                                    xtest=vtest,ytest=ytest))
            Score_ridge.append(apply_classifier(classifier_name='ridge',xtrain=vtrain,ytrain=ytrain,
                                                                    xtest=vtest,ytest=ytest))
            Score_mlp.append(apply_classifier(classifier_name='mlp',xtrain=vtrain,ytrain=ytrain,
                                                                    xtest=vtest,ytest=ytest))
            
            tf.keras.backend.clear_session()

            path_model_supervised = 'supervised/'+encoder_name+'/run_'+str(_run)+'/'+file_name+'/best_model.hdf5'

            new_xtrain, new_xtest = concatenate_supervised_unsupervised(path_model_supervised=path_model_supervised,
                                                                        xtrain=xtrain,xtest=xtest,
                                                                        vtrain=vtrain,vtest=vtest)
            
            Score_concat.append(apply_classifier(classifier_name='mlp',xtrain=new_xtrain,ytrain=ytrain,xtest=new_xtest,ytest=ytest))

            tf.keras.backend.clear_session()
        
        df = df.append({
            'encoder' : encoder_name,
            'dataset' : file_name,
            '1-NN' : np.mean(Score_knn),
            '1-NN-std' : np.std(Score_knn),
            'SVM' : np.mean(Score_svm),
            'SVM-std' : np.std(Score_svm),
            'RIDGE' : np.mean(Score_ridge),
            'RIDGE-std' : np.std(Score_ridge),
            '1-LP' : np.mean(Score_mlp),
            '1-LP-std' : np.std(Score_mlp),
            'concatenate-supervised-1-LP' : np.mean(Score_concat),
            'concatenate-supervised-1-LP-std' : np.std(Score_concat)
        },ignore_index=True)

        df.to_csv(output_directory+encoder_name+'/results_ucr.csv',index=False)
    
    elif output_directory[:-4] == 'results_semi':

        if os.path.exists(output_directory+encoder_name+'/results_ucr.csv'):
            df = pd.read_csv(output_directory+encoder_name+'/results_ucr.csv')
        else:
            df = pd.DataFrame(columns=['encoder','dataset','RIDGE-semi','RIDGE-std-semi','RIDGE','RIDGE-std'])

        Score_knn = []
        Score_svm = []
        Score_ridge = []
        Score_ridge_semi = []
        Score_mlp = []

        for _run in range(runs):

            v_train = np.load('results/'+encoder_name+'/run_'+str(_run)+'/'+file_name+'/v_train.npy')
            vtest = np.load('results/'+encoder_name+'/run_'+str(_run)+'/'+file_name+'/v_test.npy')

            for _exp in range(exps):
    
                vtrain_semi = np.load(output_directory+encoder_name+'/run_'+str(_run)+'/'+file_name+'/exp_'+str(_exp)+'/v_train.npy')
                vtest_semi = np.load(output_directory+encoder_name+'/run_'+str(_run)+'/'+file_name+'/exp_'+str(_exp)+'/v_test.npy')

                semi_indices = np.load(output_directory+encoder_name+'/run_'+str(_run)+'/'+file_name+'/exp_'+str(_exp)+'/train_indices.npy')

                semi_ytrain = ytrain[semi_indices]

                vtrain = v_train[semi_indices]

                # Score_knn.append(apply_classifier(classifier_name='knn',xtrain=vtrain,ytrain=semi_ytrain,
                                                                        # xtest=vtest,ytest=ytest))
                # Score_svm.append(apply_classifier(classifier_name='svm',xtrain=vtrain,ytrain=semi_ytrain,
                                                                        # xtest=vtest,ytest=ytest))
                Score_ridge_semi.append(apply_classifier(classifier_name='ridge',xtrain=vtrain_semi,ytrain=semi_ytrain,
                                                                        xtest=vtest_semi,ytest=ytest))
                Score_ridge.append(apply_classifier(classifier_name='ridge',xtrain=vtrain,ytrain=semi_ytrain,
                                                                        xtest=vtest,ytest=ytest))
                # Score_mlp.append(apply_classifier(classifier_name='mlp',xtrain=vtrain,ytrain=semi_ytrain,
                                                                    # xtest=vtest,ytest=ytest))
        
        df = df.append({
            'encoder' : encoder_name,
            'dataset' : file_name,
            'RIDGE-semi' : np.mean(Score_ridge_semi),
            'RIDGE-std-semi' : np.std(Score_ridge_semi),
            'RIDGE' : np.mean(Score_ridge),
            'RIDGE-std' : np.std(Score_ridge),
            
        },ignore_index=True)

        df.to_csv(output_directory+encoder_name+'/results_ucr.csv',index=False)