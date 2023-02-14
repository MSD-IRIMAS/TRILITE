import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

class SVM_Classifier:
    def __init__(self):
        self.gscv = None

    def fit(self, xtrain, ytrain):
        hyperparameters = [{'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000,
                                  np.inf],
                            'kernel': ['rbf'],
                            'degree': [3],
                            'gamma': ['scale'],
                            'coef0': [0],
                            'shrinking': [True],
                            'probability': [False],
                            'tol': [0.001],
                            'cache_size': [200],
                            'class_weight': [None],
                            'verbose': [False],
                            'max_iter': [10000000],
                            'decision_function_shape': ['ovr'],
                            'random_state': [None]
                            }]

        gscv = GridSearchCV(SVC(), hyperparameters, scoring='accuracy',
                            cv=5, n_jobs=5)
        gscv.fit(xtrain, ytrain)
        self.gscv = gscv

    def predict(self, xtest):
        return self.gscv.predict(xtest)