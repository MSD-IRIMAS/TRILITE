from sklearn.neighbors import KNeighborsClassifier

class KNN_Classifier:
    
    def __init__(self):

        self.KNN = KNeighborsClassifier(n_neighbors=1)
    
    def fit(self,xtrain,ytrain):

        self.KNN.fit(xtrain,ytrain)

    def predict(self,xtest):
        return self.KNN.predict(xtest)