import itertools
import random
from turtle import color
import numpy as np
import os
from math import ceil, floor
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.patches as mp
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

random.seed(datetime.now())

def create_directory(directory_path):
    
    if not os.path.isdir(directory_path):
        os.mkdir(directory_path)

def load_data(file_name):
    
    folder_path = "/home/hadi/datasets/UCRArchive_2018/"
    # folder_path = "/mnt/nfs/ceres/bla/archives/new/UCRArchive_2018/UCRArchive_2018/"
    folder_path += (file_name + "/")

    train_path = folder_path + file_name + "_TRAIN.tsv"
    test_path = folder_path + file_name + "_TEST.tsv"

    if (os.path.exists(test_path) <= 0):
        print("File not found")
        return None, None, None, None

    train = np.loadtxt(train_path, dtype=np.float64)
    test = np.loadtxt(test_path, dtype=np.float64)

    ytrain = train[:, 0]
    ytest = test[:, 0]

    xtrain = np.delete(train, 0, axis=1)
    xtest = np.delete(test, 0, axis=1)

    return xtrain, ytrain, xtest, ytest

def znormalisation(x):

    stds = np.std(x,axis=1,keepdims=True)
    if len(stds[stds == 0.0]) > 0:
        stds[stds == 0.0] = 1.0
        return (x - x.mean(axis=1, keepdims=True)) / stds
    return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True))

def semi_supervised_indices(xtrain,ytrain,perc=0.3):


    n = int(xtrain.shape[0])
    l = int(xtrain.shape[1])
    classes, class_counts = np.unique(ytrain,return_counts=True)
    n_classes = len(classes)

    if (class_counts < ceil(ceil(n*perc)/n_classes)).sum() == 0:

        semi_indices = []
        for i in range(len(classes)):
            semi_indices.append(np.random.choice(a=np.where(ytrain==classes[i])[0],size=ceil(n*perc/n_classes)))
        
        indices_semi = []

        for semi_indices_class in semi_indices:
            for i in semi_indices_class:
                indices_semi.append(i)

        return indices_semi

    classes_uniform = np.where(class_counts >= ceil(n*perc/n_classes))[0]
    classes_non_uniform = np.where(class_counts < ceil(n*perc/n_classes))[0]


    if len(classes_non_uniform) == 0:
        
        semi_indices = []
        for i in range(len(classes)):
            semi_indices.append(np.random.choice(a=np.where(ytrain==classes[i])[0],size=class_counts[i]))
        
        indices_semi = []

        for semi_indices_class in semi_indices:
            for i in semi_indices_class:
                indices_semi.append(i)

        return indices_semi
            
    else:

        semi_indices = []

        for c in classes_uniform:
            semi_indices.append(np.random.choice(a=np.where(ytrain==classes[c])[0],size=floor(n*perc/n_classes)))
        for c in classes_non_uniform:
            semi_indices.append(np.random.choice(a=np.where(ytrain==classes[c])[0],size=class_counts[c]))
        
        indices_semi = []

        for semi_indices_class in semi_indices:
            for i in semi_indices_class:
                indices_semi.append(i)

        indices_all = np.arange(len(ytrain))
        indices_all = np.delete(arr=indices_all,obj=indices_semi)
        
        rest_of_semi = np.random.choice(a=indices_all,size=ceil(n*perc) - len(indices_semi))
        for i in rest_of_semi:
            indices_semi.append(i)

        return indices_semi


def split_ypred(ypred_train, ypred_test):

    size_of_new_vector = int(ypred_train.shape[1])
    
    xtrain = ypred_train.copy()
    xtest = ypred_test.copy()

    xtrain = np.delete(xtrain, obj=[1, 2], axis=2)
    xtest = np.delete(xtest, obj=[1, 2], axis=2)

    xtrain.shape = (-1, size_of_new_vector)
    xtest.shape = (-1, size_of_new_vector)

    return xtrain, xtest


def generate_array_of_colors(n):

    colors_list = []

    r = int(np.random.choice(np.arange(start=32,stop=255),size=1))
    g = int(np.random.choice(np.arange(start=128,stop=255),size=1))
    b = int(np.random.choice(np.arange(start=64,stop=255),size=1))

    alpha = 1.0
    step = 256 / n

    for _ in range(n):

        r += step
        g += step
        b += step

        r = int(r) % 256
        g = int(g) % 256
        b = int(b) % 256

        colors_list.append((r / 255, g / 255, b / 255, alpha))

    # return colors_list

    colors = ["#"+''.join([random.choice('0123456789ABCDEF') for i in range(6)])
       for j in range(n)]

    return colors

def draw(ypred_test,labels_test,output_directory,colors=None):

    classes=np.unique(labels_test)
    classes=np.sort(classes)

    if colors is None:
        colors=generate_array_of_colors(classes.shape[0])
    colors=np.sort(colors)

    fig,sub=plt.subplots()

    num_units = int(ypred_test.shape[1])

    temp = ypred_test.copy()
    
    if len(temp.shape) > 2:
        temp = np.delete(temp, obj=[1, 2], axis=2)
        temp.shape = (labels_test.shape[0], num_units)

    embd=TSNE(n_components=2,random_state=12)
    temp=embd.fit_transform(temp)

    for i in range(labels_test.shape[0]):

        index = int(np.where(classes == labels_test[i])[0])
        sub.scatter(temp[i, 0], temp[i, 1],s=80, color=colors[index], marker="o")

    legends=[]

    for i in range(classes.shape[0]):

      temp_str="Class -"+str(classes[i])+"-"
      legend=mp.Patch(color=colors[i],hatch='o',linewidth=3,label=temp_str)
      legends.append(legend)

    plt.legend(handles=legends,prop={'size': 25})
    plt.title("On latent space.",fontsize=30)

    plt.savefig(output_directory+'2D.pdf')

def draw_before(xtest,ytest,output_directory,colors=None):

    classes = np.unique(ytest)
    classes = np.sort(classes)

    if colors is None:
        colors = generate_array_of_colors(classes.shape[0])
    colors = np.sort(colors)

    fig,sub = plt.subplots()

    embd = TSNE(n_components=2,random_state=12)
    xtest = embd.fit_transform(xtest)

    for i in range(ytest.shape[0]):

        index = int(np.where(classes == ytest[i])[0])
        sub.scatter(xtest[i,0], xtest[i,1],s=80, color=colors[index], marker="o")

    legends=[]

    for i in range(classes.shape[0]):

      temp_str="Class -"+str(classes[i])+"-"
      legend=mp.Patch(color=colors[i],hatch='o',linewidth=3,label=temp_str)
      legends.append(legend)

    plt.legend(handles=legends,prop={'size': 25})
    plt.title("On raw data.",fontsize=30)

    plt.savefig(output_directory+'2D_before.pdf')

def encode_labels(y):

    y = np.expand_dims(y,axis=1)
    labenc = LabelEncoder()
    
    return labenc.fit_transform(y)


if __name__ == "__main__":

    n = 4

    xtrain, ytrain, xtest, ytest = load_data(file_name="WordSynonyms")

    semi_indices = semi_supervised_indices(xtrain=xtrain,ytrain=ytrain)

    print(len(ytrain),len(semi_indices))