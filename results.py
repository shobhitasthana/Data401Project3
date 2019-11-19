import pandas as pd
import numpy as np 
from NeuralNetwork import *
from keras import layers
from keras import models
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import *

data = pd.read_csv('final_NN_feat (1).csv')
data = data.sample(frac = 1)
target = data["target"]
data = data.drop(['timestamp',"target",'Unnamed: 0'], axis = 1)

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
    return results

Node_list = [[35,15,1],[35,20,20,10,1],[35,50,50,1],[35,256,128,64,1],[35,512,256,128,64,32,1]]

sigmoid_search = grid_search(data,target, Node_list,'sigmoid',[10,25,100,200],[0.01,0.001])
relu_search  = grid_search(data,target, Node_list,'relu',[10,25,100,200],[0.01,0.001])
best_sigmoid = sorted(sigmoid_search, key = lambda model: model['mse'])[0]
best_relu = sorted(relu_search, key = lambda model: model['mse'])[0]

print("Sigmoid Search:",sigmoid_search)
print("Relu Search:",relu_search)

print("Best Sigmoid:", best_sigmoid)
print("Best ReLU:", best_relu)

cv = KFold(n_splits=5)

folds = cv.split(X=data)
lr_MSE_results = []
lr_r2_results = []
nn_sig_MSE_results = []
nn_sig_r2_results = []
nn_relu_MSE_results = []
nn_relu_r2_results = []

for train_idx, test_idx in folds:
    lr = LinearRegression()
    X_train = data.iloc[train_idx,:]
    X_test = data.iloc[test_idx,:]
    y_train = target[train_idx]
    y_test = target[test_idx]
    
    scalerX = preprocessing.StandardScaler().fit(X_train)
    scalery = preprocessing.StandardScaler().fit(np.array(y_train).reshape(-1, 1))
    X_train = scalerX.transform(X_train)
    X_test = scalerX.transform(X_test)
    y_train = scalery.transform(np.array(y_train).reshape(-1, 1))
    y_test = scalery.transform(np.array(y_test).reshape(-1, 1))
    
    lr.fit(X_train,y_train)
    y_pred = lr.predict(X_test)
    lr_MSE_results.append(mean_squared_error(y_test, y_pred))
    lr_r2_results.append(r2_score(y_test,y_pred))
    
    net_sig = NeuralNetwork(best_sigmoid['Nodes'],['sigmoid']*5, rate=best_sigmoid['learning_rate'])
    net_sig.train(X_train, y_train, batch_size=best_sigmoid['batch_size'], epoch_MSE=False)
    y_pred_nn_sig = net_sig.predict(X_test)
    nn_sig_MSE_results.append(mean_squared_error(y_test, y_pred_nn_sig))
    nn_sig_r2_results.append(r2_score(y_test,y_pred_nn_sig))
    
    net_relu = NeuralNetwork(best_relu['Nodes'],['relu']*5, rate=best_relu['learning_rate'])
    net_relu.train(X_train, y_train, batch_size=best_relu['batch_size'], epoch_MSE=False)
    y_pred_nn_relu = net_relu.predict(X_test)
    nn_relu_MSE_results.append(mean_squared_error(y_test, y_pred_nn_relu))
    nn_relu_r2_results.append(r2_score(y_test,y_pred_nn_relu))
                          
print("Regression Results:")
print("MSE:", lr_MSE_results.mean(), "R2:", lr_r2_results.mean())
print("Neural Network (Sigmoid Activation) Results:")
print("MSE:", nn_sig_MSE_results.mean(), "R2:", nn_sig_r2_results.mean())
print("Neural Network (ReLU Activation) Results:")
print("MSE:", nn_relu_MSE_results.mean(), "R2:", nn_relu_r2_results.mean())
