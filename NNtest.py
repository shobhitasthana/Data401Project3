from NeuralNetwork import *

nodes = [3,2,2,1]
activation = ['relu','sigmoid','sigmoid','relu']
layers = len(nodes) - 1
net = NeuralNetwork(nodes,activation)
net.initialize_net()
data = np.matrix([1,-2,2])
net.parameter_dict[0]['h'] = data
net.parameter_dict[1]['w'] = np.matrix([[0,0],[3.,1.],[2.,0],[1,-1]])
net.parameter_dict[2]['w'] = np.matrix([[0,0],[-2.,4],[0,-3.]])
net.parameter_dict[3]['w'] = np.matrix([0,5.,2])

print(net.forward_propogate(data))

print(net.parameter_dict)