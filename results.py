import pandas as pd
import numpy as np 
from NeuralNetwork import *
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from keras import layers
from keras import models
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

data = pd.read_csv('final_NN_feat (1).csv')
target = data["target"]
data = data.drop(['timestamp',"target",'Unnamed: 0'], axis = 1)

# training and test split


def grid_search(data, target, Nodes, activation, batch_sizes, rates):
    X_train, X_test, y_train, y_test = train_test_split(data
                                                        , target, test_size=0.33, random_state=42)

    scalerX = preprocessing.StandardScaler().fit(X_train)
    scalery = preprocessing.StandardScaler().fit(np.array(y_train).reshape(-1, 1))
    X_train = scalerX.transform(X_train)
    X_test = scalerX.transform(X_test)
    y_train = scalery.transform(np.array(y_train).reshape(-1, 1))
    y_test = scalery.transform(np.array(y_test).reshape(-1, 1))
    
    results = []
    #nodes = [35, 15, 1]
    
    for n in Nodes:
        print(n)
        activations = [activation] * len(n)
        print(activations)
        for rate in rates:
            net = NeuralNetwork(n,activations, rate)

            for batch in batch_sizes:
                net.train(X_train,y_train,batch_size = batch, epoch_MSE = False)
                pred = net.predict(X_test)
                mse = mean_squared_error(y_test,pred)
                r2 = r2_score(y_test,pred)
                results.append({'r2': r2,'mse': mse, 'Nodes': n,'activation':activation, 'batch_size': batch, 'learning_rate': rate})
                print(results)
    return results

Node_list = [[35,15,1],[35,20,20,10,1],[35,50,50,1],[35,256,128,64,1],[35,512,256,128,64,32,1]]

#activations = [['relu'] * len(i) for i in Node_list]
print(grid_search(data,target, Node_list,'sigmoid',[10,25,100,200],[0.1,0.01,0.001]))


'''
Keras implementation for comparison

model = models.Sequential()
model.add(layers.Dense(35,activation='relu',input_shape=(35,)))
model.add(layers.Dense(20,activation='relu'))
model.add(layers.Dropout(.25))
model.add(layers.Dense(20,activation='relu'))
#model.add(layers.Dropout(.25))
model.add(layers.Dense(10,activation='relu'))
#model.add(layers.Dropout(.25))
#model.add(layers.Dense(8,activation='relu'))
model.add(layers.Dense(1,activation ='relu'))

model.compile(optimizer = 'rmsprop',loss = 'mean_squared_error', metrics = ['mse'])
split = 0.8
data_training, data_testing = data[:int(len(data)*split)], data[int(len(data)*split):]
labels_training, labels_testing = labels[:int(len(data)*split)], labels[int(len(data)*split):]

model.fit(data_training,labels_training, batch_size = 100, epochs = 8)
pred = model.predict(data_testing)
print(r2_score(labels_testing,pred))
print(model.evaluate(data_testing,labels_testing))
'''
