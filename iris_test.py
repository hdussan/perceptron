#!python3
#  Test perceptron algorithm using the Iris data set
#  pathname = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
#
#
import sys
import csv
import operator
import numpy as np
import get_data_set as gds
import perceptron as perc
from random import shuffle

def process_dataset(datasetname, number_features, feature_to_drop = ''):
  dataset = gds.get_dataset(datasetname, number_features)    
  data_set = [d for d in dataset if feature_to_drop not in d]
  #shuffles rows  
  shuffle(data_set)  
  return data_set

def main():
  print("Perceptron main instructions")
  data_set_name = sys.argv[1]
  number_features = int(sys.argv[2])
  number_epochs   = int(sys.argv[3])
  print(" Data Set Name = ", data_set_name)
  print(" Features in Data Set  =  ", number_features)
  print(" Number of Epochs  =  ", number_epochs)
  #drop_this = 'Iris-virginica'
  drop_this = 'Iris-setosa'
  data_set = process_dataset(data_set_name, number_features, drop_this)
  #data_set = process_dataset(data_set_name, number_features)

  iris_p = perc.perceptron(number_epochs, number_features)

  train_set, test_set = iris_p.train_and_test(data_set)
  label0 = 'Iris-versicolor'
  x_train, y_train = iris_p.prepare_xy(train_set, label0)
  x_test, y_test = iris_p.prepare_xy(test_set, label0)

  pars, epochs = iris_p.find_pars(x_train, y_train)
  print (" Epochs completed = ", epochs)

  score = iris_p.score(pars, x_test, y_test)
  print("   SCORE =  ", score)

  xsample = np.array([4.9,3.0,1.4,0.2]) # Iris Setosa
  print( iris_p.predict(xsample,pars))

  xsample = np.array([5.1,2.5,3.0,1.1]) # Iris Versicolour
  print( iris_p.predict(xsample,pars))
  

if __name__ == "__main__":
  main()
    