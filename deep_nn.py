# -*- coding: utf-8 -*-
import numpy as np

class (object):
    def __init__(self):
        pass
        
    def initialize_parameters(self, layer_dims):
        
        """
        Arguments:
        layer_dims -- python array (list) containing the dimensions of each layer in our network
        
        Returns:
        parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                        Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                        bl -- bias vector of shape (layer_dims[l], 1)
        """
        
        parameters = {}
        L = len(layer_dims)            # number of layers in the network

        # loop over every layer starting from the first
        # and initialize weights and biases
        
        for l in range(1, L):
            # create size (nl, nl-1) weight matrix  
            parameters['W' + str(l)] = np.random.randn( layer_dims[l], layer_dims[l-1]) * 0.01

            # create size (nl, 1) bias matrix 
            parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
              
        return parameters

    # forward propagation functions

    def linear_forward(self, A, W, b):
        """
        Description: First half of forward propagation that calculates Z Linear
        
        Arguments:
        A -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)

        Returns:
        Z -- the input of the activation function, also called pre-activation parameter 
        cache -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently
        """

    
        # calculate z linear 
        Z = np.dot(W,A) + b

        cache = (A, W, b)

        return Z, cache

    def linear_activation_forward(self, A_prev, W, b, activation):
        """
        Description: Forward propagation for the LINEAR->ACTIVATION layer

        Arguments:
        A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

        Returns:
        A -- the output of the activation function, also called the post-activation value 
        cache -- a python tuple containing "linear_cache" and "activation_cache";
                 stored for computing the backward pass efficiently
        """
        if activation == "sigmoid":
            # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
            Z, linear_cache = linear_forward(A_prev, W, b)
            A, activation_cache = sigmoid(Z)
        
        elif activation == "relu":
            # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
            Z, linear_cache = linear_forward(A_prev, W, b)
            A, activation_cache = relu(Z)
     
        # store cache for later use in backward propagation pass
        cache = (linear_cache, activation_cache)

        return A, cache

    def L_model_forward(self, X, parameters):

        """
        Description: forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

        Arguments:
        X -- data, numpy array of shape (input size, number of examples)
        parameters -- output of initialize_parameters_deep()

        Returns:
        AL -- last post-activation value
        caches -- list of caches containing:
                every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)
        """

        caches = []
        A = X
        L = len(parameters) // 2 # number of layers in the neural network
        
        # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
        for l in range(1, L):
            A_prev = A 
            ### START CODE HERE ### (≈ 2 lines of code)
            A, cache = self.linear_activation_forward(A_prev, parameters["W"+str(l)], parameters["b"+str(l)], "relu")
            caches.append(cache)
            ### END CODE HERE ###
        
        # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
        AL, cache = self.linear_activation_forward(A, parameters["W"+str(L)], parameters["b"+str(L)], "sigmoid")
        caches.append(cache)

                
        return AL, caches

    def compute_cost(self, AL, Y):
        """
        Description: compute the cost function defined by equation (7).

        Arguments:
        AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
        Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

        Returns:
        cost -- cross-entropy cost
        """
        
        m = Y.shape[1]

        # Compute loss from aL and y.
        cost = -(1.0 / m) * np.sum( np.dot(Y, np.log(AL).T) + np.dot( (1.0 - Y), np.log(1.0 - AL.T)), axis = 1, keepdims = True)

        # Squeeze cost into a dimensionless type
        cost = np.squeeze(cost) 
        
        return cost


        
    def on_gradient_descent(self):
        pass

    # to do: implement backward propagation 
        
