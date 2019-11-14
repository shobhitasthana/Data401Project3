def calculate_deltas(self,true):
    """
    true: the value of the true y-value
    assumes that all the entries in the dictionary are np.array where applicable
    assumes weights matrix appears as j being constant across rows and i across cols
    e.g. [[w11,w21],[w12,w22]] 
    """
    # calculate sum of dervatives of cost
    error = self.parameter_dict["y_hat"] - true
    # last layer d (indexing is correct?)
    # get Zs from the last layer as well
    # error * g'(z)
    self.parameter_dict[len(self.Nodes) - 1]["delta"] = error * np.array([self.activate_prime(x) for 
                                                           x in self.parameter_dict[len(self.Nodes) - 1]["z"]])
    # last layer is special case, now loop through all the previous layers to calculate sets of deltas
    # from the second to last layer to the first layer – backwards
    for i in range(len(self.Nodes) - 2,-1,-1):
        # delta = weights.T x diag(g'(z)) x delta[i+1]
        # extra Ts are just making things into column vectors
        g_prime_layer = np.array([self.activate_prime(x) for x in self.parameter_dict[i]["z"]]).T
        self.parameter_dict[i]["delta"] = self.parameter_dict[i]["w"].T.dot(
            np.diag(g_prime_layer)).dot(self.parameter_dict[i + 1]["delta"].T)