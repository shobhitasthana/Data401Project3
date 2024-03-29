from NeuralNetwork import *
import unittest
import warnings


class TestNN(unittest.TestCase):
    
    def setUp(self):
        warnings.filterwarnings("ignore")
        self.net = NeuralNetwork([3,2,2,1], ['relu','relu','relu','relu'])
        self.net.initialize_net()
        self.net.parameter_dict[1]['w'] = np.matrix([[3.,1.],[2.,0],[1,-1]])
        self.net.parameter_dict[2]['w'] = np.matrix([[-2.,4],[0,-3.]])
        self.net.parameter_dict[3]['w'] = np.matrix([[5.],[2]])
        self.net.parameter_dict[1]['bias'] = np.matrix([[0,0]])
        self.net.parameter_dict[2]['bias'] = np.matrix([[0,0]])
        self.net.parameter_dict[3]['bias'] = np.matrix([[0]])
  
    def test_forward_prop(self):
        self.net.forward_propogate([1,-2,2])
        d = self.net.parameter_dict
        self.assertTrue((d[1]['z'] == [[1,-1]]).all())
        self.assertTrue((d[1]['h'] == [[1,0]]).all())
        
        self.assertTrue((d[2]['z'] == [[-2,4]]).all())
        self.assertTrue((d[2]['h'] == [[0,4]]).all())
        
        self.assertEqual(d['y_hat'], 8)
        
    def test_delta_calculations(self):
        self.net.forward_propogate([1,-2,2])
        self.net.calculate_deltas2(3)
        d = self.net.parameter_dict
        
        self.assertEqual(d[3]['delta'], 5)
        self.assertTrue((d[2]['delta'] == [[0,10]]).all())
        self.assertTrue((d[1]['delta'] == [[40,0]]).all())
       
    def test_gradient_calc(self):
        self.net.forward_propogate([1,-2,2])
        self.net.calculate_deltas2(3)
        self.net.update_gradient()
        d = self.net.parameter_dict
        
        self.assertTrue((d[3]['gradient'] == [[0],[2]]).all())
        self.assertTrue((d[2]['gradient'] == [[0,1],[0,0]]).all())
        self.assertTrue((d[1]['gradient'] == [[4,0],[-8,0],[8,0]]).all())

    def test_update_weights_self(self):
        self.net.forward_propogate([1,-2,2])
        print(self.net.parameter_dict)
        self.net.calculate_deltas2(3)
        self.net.update_gradient()
        self.net.update_weights()

        print(self.net.parameter_dict)
        return
        
        
if __name__ == "__main__":
    unittest.main()

    #net.calculate_deltas2(2.3)