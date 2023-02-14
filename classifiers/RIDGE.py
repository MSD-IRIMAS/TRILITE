import numpy as np
from sklearn.linear_model import RidgeClassifierCV

class Ridge_Classifier:

    def __init__(self):

        self.ridgeClasiifier = RidgeClassifierCV(alphas=np.logspace(-3,3,10),normalize=True)
    
    def fit(self,xtrain,ytrain):
        
        self.ridgeClasiifier.fit(xtrain,ytrain)
    
    def predict(self,xtest):

        return self.ridgeClasiifier.predict(xtest)
