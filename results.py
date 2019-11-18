import pandas as pd
import numpy as np 
from NeuralNetwork import *
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# net = NeuralNetwork([39,256,128,64,32,1],['relu']*6)
# #rint(['relu']*6)

data = pd.read_csv('final_NN_feat (1).csv')
#print(data.sample(frac = 1))
data = data.sample(frac = 1)

labels = data['target'].reset_index()
data = data.drop(['target','timestamp'], axis = 1)
print(data.shape)
data = data.iloc[:,1:].to_numpy()

#data =(data -np.mean(data,axis = 0))/np.std(data,axis =0)
print(data.shape)

#labels = (labels - np.mean(labels,axis =0))/np.std(labels, axis =0)

#labels = pd.read_csv('training_labels.csv')
labels = labels.iloc[:,1:].to_numpy()

print(sum(data == np.nan))
print(labels)

# #print(labels.iloc[:,1:].to_numpy())

# net.train(data,labels)

#grid search reads in 2-d array of Nodes and Activations
#reads in array of batch_sizes and epochs
#data and labels

def grid_search(Nodes,Activations, data, labels,batch_sizes = [300], epochs = [8], split = 0.66):
	data_training, data_testing = data[:int(len(data)*split)], data[int(len(data)*split):]
	labels_training, labels_testing = labels[:int(len(data)*split)], labels[int(len(data)*split):]
	results = []
	for i in range(len(Nodes)):
		net = NeuralNetwork(Nodes[i],Activations[i])
		if len(batch_sizes) ==1 and len(epochs) ==1:
			net.train(data_training,labels_training, batch_size = batch_sizes[0], epochs = epochs[0])
			pred = net.predict(data_testing)
			mse = mean_squared_error(labels_testing,pred)
			r2 = r2_score(labels_testing,pred)
			results.append({'r2': r2,'mse': mse,'Nodes': Nodes[i], 'Activations': Activations[i], 'batch_size': batch_sizes[0], 'epochs': epochs[0], 'split': split})
			print(results)
		else:
			for j in batch_sizes:
				for k in epochs:
					net.train(data_training,labels_training,batch_size = j, epochs = k)
					pred = net.predict(data_testing)
					mse = mean_squared_error(labels_testing,pred)
					r2 = r2_score(labels_testing,pred)
					results.append({'r2': r2,'mse': mse,'Nodes': Nodes[i], 'Activations': Activations[i], 'batch_size': batch_sizes[0], 'epochs': epochs[0], 'split': split})
					print(results)
	return results

Node_list = [[35,40,1],[35,50,50,1],[35,256,128,64,1],[35,512,256,128,64,32,1]]
activations = [['relu'] * len(i) for i in Node_list]
print(grid_search(Node_list,activations,data,labels))
