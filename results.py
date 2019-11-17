import pandas as pd
import numpy as np 
from NeuralNetwork import *

net = NeuralNetwork([40,256,128,64,32,1],['relu']*6)
#rint(['relu']*6)

data = pd.read_csv('final_NN_feat.csv')
data = data.drop('timestamp', axis = 1)
data = data.iloc[:,1:].to_numpy()


labels = pd.read_csv('training_labels.csv')
#print(labels.iloc[:,1:].to_numpy())

net.train(data,labels)