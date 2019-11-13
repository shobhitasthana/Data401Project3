import pandas as pd
import numpy as np 


class NeuralNetwork:
    """
    Layers: an Integer value representing the total number of hidden layers in the network (input and output layers are extra)
    Nodes: an integer array of size [0,..,Layers+1] containing the dimensions of the neural network. 
    Nodes[0] shall represent the input size (typically, 50), Nodes[Layers+1] shall represent the number of output nodes (typically, 1). 
    All other values Nodes[i] represent the number of nodes in hidden layer i.

    NNodes: a possible alternative to the Nodes parameter for situations where you want each hidden layer of the neural network to be of the same size. 
    In this case, the size of the output layer is assumed to be 1, and the size of the input layer can be inferred from the dataset.

    Activations: an array of size [0,..,Layers+1] (for the sake of compatibility) in which Activations[0] and Activations[Layers+1] are not used, while all other Activations[i] values are labels indicating the activation function used in layer i. 
    This allows you to build neural networks with different activation functions in each layer.
    """
    def __init__(self,Nodes,Activations):



        self.Layers = len(Nodes) - 2
        self.Nodes = Nodes
        self.Activations = Activations
        self.parameter_dict ={}
        self.rate = 0.1

    def initialize_net(self):
        '''
        parameter dict format:
        layer: {w,h,z,delta, activation}
        '''
        parameter_dict = {k: {'w':0, 'h':0, 'z':0, 'delta':0, 'activation':0, 'gradient':0} for k in range(len(self.Nodes))}
        for i in range(1,len(self.Nodes)):

            # add the intercept
            h = np.matrix(np.append(1,np.random.randn(self.Nodes[i])))
            w = np.matrix(np.random.randn((self.Nodes[i-1]+1),(self.Nodes[i]+1)))
            z = np.matrix(np.zeros(self.Nodes[i]+1))
            delta = np.matrix(np.random.randn(self.Nodes[i]+1))
            gradient = np.matrix(np.zeros((self.Nodes[i-1]+1,self.Nodes[i]+1)))
            
            activation = self.Activations[i-1]

            parameter_dict[i] = {'w':w, 'h':h, 'z':z, 'delta':delta, 'activation':activation, 'gradient':gradient}
        parameter_dict['y_hat'] = np.random.randint(100000,size =1)[0]
        self.parameter_dict = parameter_dict
        return parameter_dict
    
    def activate(self, z,activation='relu'):
        if activation =='relu':
            if z > 0:
                return z
            else:
                return 0
        if activation == 'sigmoid':
            return (1/(1+np.e** -z))
        
    def activate_prime(self, z, activation='relu'):
        if activation == 'relu':
            if z > 0:
                return 1
            return 0
        
    def forward_propogate(self,data):
        new = {}
        new['z'] = []
        new['h'] = []
        for i in len(self.Nodes-1):
            #new z value calculated by multiplying node weights and adding bias 
            new['z'].append(np.matmul(self.parameter_dict[i][1],self.parameter_dict[i+1][0][1:,1:]) + self.parameter_dict[i][0][0][0])
            new['h'].append(activate(newz,self.Activations[i]))
        return new