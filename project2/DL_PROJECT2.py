# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 11:07:06 2021

@author: mauro
"""

import torch
import math
class Module ( object ) :
    def forward ( self , * input ) :
        raise NotImplementedError
    def backward ( self , * gradwrtoutput ) :
        raise NotImplementedError
    def param ( self ) :
        return []
    


#Activation funcitons here 
#Required
class Relu ( Module ) :
    
    def __init__(self,dimension):
        self.input = torch.FloatTensor(dim = dimension)
        self.output = torch.FloatTensor(dim = dimension)
    """
        Bro why using FloatTensor?
        
        self.dimension = dimension
    """    
    def forward(self,toRelu):
        "forward pass for Relu activation: output is unchanged if positive and goes to zero otherwise"
        self.output = toRelu
        self.output[self.output < 0] = 0
        return self.output
    def backward(self, der_loss_wrt_activations):
        "backward pass for Relu activation: derivative is one if the output is > 0 and zero otherwise. "
        bkw = self.output
        bkw[bkw > 0] = 1
        return bkw * der_loss_wrt_activations
        "the values <0 are alredy zero therefore no nees for putting them to 0"
        return bkw
    def param ( self ) :
        return []
        
class Tanh ( Module ) :
    def __init__(self,dimension):
        self.input = torch.FloatTensor(dim = dimension)
        self.output = torch.FloatTensor(dim = dimension)

    def forward(self,toTanh):
        "forward pass: applcation of tanh to the input https://pytorch.org/docs/stable/generated/torch.nn.Tanh.html"
        self.output = math.tanh(toTanh)
        return self.output
    def backward(self, der_loss_wrt_activations):
        "bkw pass: d/dx (tanh(x)) = 1-tanh(x)^2"
        bkw = 1-self.output**2
        return bkw*der_loss_wrt_activations
  
    def param ( self ) :
        return []
    
#optionals
class Sigmoid ( Module ) :
    def __init__(self,dimension):
        self.input = torch.FloatTensor(dim = dimension)
        self.output = torch.FloatTensor(dim = dimension)

    def forward(self,toSigmoid):
        "forward pass: applcation of sigmoid to the input https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html"
        self.output = 1/(1+math.exp(-toSigmoid))
        return self.output

    def backward(self, der_loss_wrt_activations):
        " derivatives of the loss w.r.t. the activation"
        bkw = self.output*(1-self.output)
        return bkw*der_loss_wrt_activations
        
    def param ( self ) :
        return []
"""
class Softmax ( Module ) :
    def __init__(self,dimension):
        self.input = torch.FloatTensor(dim = dimension)
        self.output = torch.FloatTensor(dim = dimension)

    def forward(self,toRelu):

    def backward(self):

    def param ( self ) :
        return []
"""
#Fully connected layer here 
#Required
class Linear ( Module ) :
    def __init__(self,input_dimension,output_dimension):
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.input = FloatTensor(input_dimension)
        self.output = FloatTensor(output_dimension)
        self.weights = FloatTensor(input_dimension,output_dimension).uniform_(-1, 1)
        self.bias = FloatTensor(output_dimension).uniform_(-1, 1)
        self.weights_grad = [FloatTensor(input_dimension,output_dimension).zero_()]
        self.bias_grad = [FloatTensor(output_dimension).zero_()]
    def forward(self,input_):
        self.output = self.weights @ input_ + self.bias
        return self.output

    def backward(self,der_loss_wrt_activations):
        "slides 3.6 on backpropagation"
        bkw = self.weights.T @ der_loss_wrt_activations
        bias_grad = der_loss_wrt_activations
        weights_grad = der_loss_wrt_activations.view(-1, 1) @ self.input.view(1, -1)

    def param ( self ) :
        return [(self.weights,self.weights_grad)] + [(self.bias,self.biases_grad)]
    def gradient_update(self, learning_rate):
        self.weights -= learning_rate * self.weights_grad[-1]
        self.bias -= learning_rate * self.bias_grad[-1]
#Combinations 
#Required
class Sequential ( Module ) :
#Losses 
#Required
class LossMSE ( Module ) :
    def loss_value(predictions,actuals):
        return ((predictions-actuals)**2).mean()
    def loss_grad(predictions,actuals):
        return 2*(predictions-actuals)
    "factor 2 is needed?!"
class LossMAE (Module) :
    def loss_value(predictions,actuals):
        return (math.abs(predictions-actuals)).mean()
    def loss_grad(predictions,actuals):
        grad = math.abs(predictions-actuals)
        grad[grad < 0] *= -1
        return  grad
class LossMBE(Module):
    loss_value(predictions,actuals):
        return ((predictions-actuals)).mean()
    def loss_grad(predictions,actuals):
        return (predictions-actuals)
        
    
#optionals
class Sequential ( Module ) :