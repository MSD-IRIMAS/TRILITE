import numpy as np
import sys
from utils.utils import load_data, znormalisation, draw, draw_before, generate_array_of_colors

def extract_argv():

    if len(sys.argv) != 7:
        raise ValueError("Not enough options")
    
    else:

        return sys.argv[2], sys.argv[4], sys.argv[6]
    

if __name__ == "__main__":

    encoder_name, directory_parent, file_name = extract_argv()

    directory_parent = directory_parent + encoder_name + '/run_'
    
    runs = 5

    for _run in range(runs):

        directory = directory_parent + str(_run) + '/' + file_name + '/'

        xtrain, ytrain, xtest, ytest = load_data(file_name=file_name)
        xtrain = znormalisation(xtrain)
        xtest = znormalisation(xtest)

        vtest = np.load(directory+'v_test.npy')

        # colors = generate_array_of_colors(n=len(np.unique(ytrain)))
        # colors = ['green','orange']

        draw_before(xtest=xtest,ytest=ytest,output_directory=directory)
        draw(ypred_test=vtest,labels_test=ytest,output_directory=directory)