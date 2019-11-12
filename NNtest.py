from NeuralNetwork import *


nodes = [50,10,5,3000,2,1]
activation =['relu','relu','relu','relu','relu','relu']


nn = NeuralNetwork(nodes,activation)
print(nn.initialize_net())