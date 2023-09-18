#Losses classes:
from module import Module
#Required (Project description)
class LossMSE ( Module ) :
    def loss_value(self,predictions,actuals):
        return ((predictions-actuals)**2).mean()
    def loss_grad(self,predictions,actuals):
        return 2.0*(predictions-actuals)
    def param ( self ) :
        return []

#additionals: We decided to add MeanAbsoluteError and MeanBiasError
class LossMAE (Module) :
    def loss_value(self,predictions,actuals):
        return (abs(predictions-actuals)).mean()
    def loss_grad(self,predictions,actuals):
        grad = empty(predictions.shape)
        for i in range(predictions.shape[0]):
            for j in range(predictions.shape[1]):
                if (predictions[i][j]-actuals[i][j]) >= 0:
                    grad[i][j] = 1
                else:
                    grad[i][j] = -1
        return  grad
    def param ( self ) :
        return []
    
class LossMBE(Module):
    def loss_value(self,predictions,actuals):
        return ((predictions-actuals)).mean()
    def loss_grad(self,predictions,actuals):
        return (predictions-actuals)
    def param ( self ) :
        return []      