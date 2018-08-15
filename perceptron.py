#!python3
#   perceptron algorithm 
#   for classification
#   Author: Helber Dussan
#   At:     Solving4x
#
import csv
import operator
import numpy as np
import get_data_set as gds
from random import shuffle
'''
    n_features = Number of Features 
    n_samples  = Number of data taken
'''
class perceptron:
    def __init__(self, number_epochs, number_features):
        self.n_epochs   = number_epochs
        self.n_features = number_features 
  
    def find_pars(self, x, y, mu = 1.0):
        n_epochs = 0
        weights = np.zeros(self.n_features)
        for epoch in range(self.n_epochs):
            count = 0
            n_epochs = epoch            
            for n in range(self.n_samples):
                y_xT_w = y[n] * np.dot(x[n], weights)
                if y_xT_w <= 0.0:
                    weights += mu * y[n] * x[n]
                    count += 1
                 
            if count == 0.0:
                n_epochs = epoch                 
                break
               
        return weights, n_epochs  

    def predict(self, x, pars):
        return np.sign(np.dot(x, pars))

    def score(self, pars, x_test, y_test):
        counter = 0.0
        for xi, yi in zip(x_test, y_test):
            ypredict = self.predict(xi, pars)
            if( ypredict == yi):
                counter += 1
        counter /= y_test.size
        return counter * 100.


    def train_and_test(self, dataset, split = 0.6):
        train_set = []
        test_set  = []
        gds.split_dataset(split, dataset, train_set, test_set)
        self.n_samples  = len(train_set)
        return train_set, test_set

    ''' 
        Perceptron used for classification with 2 classes:
        label0, label1 
        They got to be converted to -1 and 1 to use the algorithm
    '''
    def prepare_xy(self, data_set, label0):
        x = []
        y = []
        mask = np.array([True, True, True, True, False])
        data_set = np.array(data_set)
        X = data_set[:, mask]  
        X = np.array(X)
        x = X.astype(np.float)
        y = data_set[:, -1]   
        y = np.where(y == label0, -1.0, 1.0)      
        return x, y
    
    

