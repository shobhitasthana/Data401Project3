import pandas as pd
import numpy as np 
from NeuralNetwork import *
from sklearn.metrics import mean_squared_error

# net = NeuralNetwork([39,256,128,64,32,1],['relu']*6)
# #rint(['relu']*6)

data = pd.read_csv('final_NN_feat.csv')
data = data.drop('timestamp', axis = 1)
data = data.iloc[:,1:].to_numpy()

labels = pd.read_csv('training_labels.csv')
labels = labels.iloc[:,1:].to_numpy()
# #print(labels.iloc[:,1:].to_numpy())

# net.train(data,labels)

#grid search reads in 2-d array of Nodes and Activations
#reads in array of batch_sizes and epochs
#data and labels
def grid_search(Nodes,Activations, data, labels,batch_sizes = [100], epochs = [6], split = 0.75):
	data_training, data_testing = data[:int(len(data)*split)], data[int(len(data)*split):]
	labels_training, labels_testing = labels[:int(len(data)*split)], labels[int(len(data)*split):]
	results = []
	for i in range(len(Nodes)):
		net = NeuralNetwork(Nodes[i],Activations[i])
		if len(batch_sizes) ==1 and len(epochs) ==1:
			net.train(data_training,labels_training, batch_size = batch_sizes[0], epochs = epochs[0])
			pred = net.predict(data_testing)
			mse = mean_squared_error(labels_testing,pred)
			results.append({'mse': mse,'Nodes': Nodes[i], 'Activations': Activations[i], 'batch_size': batch_sizes[0], 'epochs': epochs[0], 'split': split})
			print(results)
		else:
			for j in batch_sizes:
				for k in epochs:
					net.train(data_training,labels_training,batch_size = j, epochs = k)
					pred = net.predict(data_testing)
					mse = mean_squared_error(labels_testing,pred)
					results.append({'mse': mse,'Nodes': Nodes[i], 'Activations': Activations[i], 'batch_size': j, 'epochs': epochsk, 'split': split})
	return results

Node_list = [[39,40,1],[39,50,50,1],[39,256,128,64,1],[39,512,256,128,64,32,1]]
activations = [['relu'] * len(i) for i in Node_list]
print(grid_search(Node_list,activations,data,labels))