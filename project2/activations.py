#Activation funcitons classes:
from module import Module
from torch import empty
#Required (Project description)

class Relu ( Module ) :
    
    def __init__(self,dimension):
        self.input = empty(dimension)
        self.output = empty(dimension)
 
    def forward(self,toRelu):
        "forward pass for Relu activation: output is unchanged if positive and goes to zero otherwise"
        self.output = toRelu
        self.output[self.output < 0] = 0
        return self.output
    
    def backward(self, der_loss_wrt_activations, momentum):
        "backward pass for Relu activation: derivative is one if the output is > 0 and zero otherwise. "
        bkw = self.output
        bkw[bkw > 0] = 1
        return bkw * der_loss_wrt_activations
        "the values <0 are alredy zero therefore no nees for putting them to 0"
    
    def param ( self ) :
        "no parameters"
        return []
        
class Tanh ( Module ) :
    def __init__(self,dimension):
        self.input = empty(dimension)
        self.output = empty(dimension)

    def forward(self,toTanh):
        "forward pass: applcation of tanh to the input"
        self.output = toTanh.apply_(math.tanh)
        return self.output
    
    def backward(self, der_loss_wrt_activations, momentum):
        "bkw pass: d/dx (tanh(x)) = 1-tanh(x)^2"
        bkw = 1-self.output**2
        return bkw*der_loss_wrt_activations
  
    def param ( self ) :
        "no parameters"
        return []
    
#additionals: We decided to implement the Sigmoid activation

class Sigmoid ( Module ) :
    def __init__(self,dimension):
        self.input = empty(dimension)
        self.output = empty(dimension)

    def forward(self,toSigmoid):
        "forward pass: applcation of sigmoid to the input https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html"
        self.output = 1/(1+(-toSigmoid).apply_(math.exp))
        return self.output

    def backward(self, der_loss_wrt_activations, momentum):
        " derivatives of the loss w.r.t. the activation"
        bkw = self.output*(1-self.output)
        return bkw*der_loss_wrt_activations
        
    def param ( self ) :
        "no parameters"        
        return []