#Linear fully connected layer class below:
from module import Module
from torch import empty
#Required (Project desctiption)
class Linear ( Module ) :
    
    def __init__(self,input_dimension,output_dimension):
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.input = empty(input_dimension)
        self.output = empty(output_dimension)
        self.weights = empty(output_dimension,input_dimension).uniform_(-1, 1)
        self.bias = empty(output_dimension).uniform_(-1, 1)
        self.weights_grad = empty(output_dimension,input_dimension).zero_()
        self.bias_grad = empty(output_dimension).zero_()
        self.moment_grad_prev = empty(output_dimension,input_dimension).zero_()
        self.moment_grad =  empty(output_dimension,input_dimension).zero_()
        
    def forward(self,input_):
        "Forward pass for a linear layer y = WX + b"
        self.input = input_
        self.output = (self.weights@input_)+self.bias.reshape(-1,1)
        return self.output

    def backward(self,der_loss_wrt_activations, momentum):
        "slides 3.6 on backpropagation: bkw = W.T @ derivative wrt activations"
        bkw = (self.weights.T @ der_loss_wrt_activations)
        self.bias_grad = (der_loss_wrt_activations).mean()
        self.weights_grad = (der_loss_wrt_activations @ self.input.T)
        if momentum is not None: self.moment_grad = self.moment_grad_prev*momentum
        return bkw

    def param ( self ) :
        return [(self.weights,self.weights_grad)] + [(self.bias,self.bias_grad)]
    
    def gradient_update(self, learning_rate):
        "just gradient descent"
        self.weights -= (learning_rate * self.weights_grad + self.moment_grad)
        self.moment_grad_prev = learning_rate * self.weights_grad + self.moment_grad
        self.bias -= learning_rate * self.bias_grad