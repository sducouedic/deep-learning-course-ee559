
#Sequential Neural Network class:
from module import Module
from losses import *
from activations import *
from linear import *
plot = False
if plot:   
    import matplotlib.pyplot as plt
#Required (Project description)
class Sequential ( Module ) :
    def __init__(self,loss,input_dimension,batch_idx):
        self.layers = []
        self.input_dimension = input_dimension
        self.loss = loss
        self.batch_idx = batch_idx
    def new_layer(self,newlayer):
        "just add a Layer to the list of layers"
        self.layers.append(newlayer)
    def forward(self, model_prev_output):
        """for every layer it takes the output of the previous layer and forward it
        the first layer should have the input wich is the case because when we call 
        the fit function we are passing the imputs as arguments"""
        for layer in self.layers:
            model_prev_output = layer.forward(model_prev_output)
        return model_prev_output
    def backward(self,der_loss_wrt_output, momentum = None):
        "for every layer, in reversed order, it takes the output of the next layer and backward it"
        for layer in reversed(self.layers):
            der_loss_wrt_output = layer.backward(der_loss_wrt_output, momentum)
    def gradient_update(self,learning_rate):
        "Gradient descent on all the layers"
        for layer in self.layers:
            if isinstance(layer,Linear):
                layer.gradient_update(learning_rate)

    def fit(self, traindata, traintarget, testdata, testtarget, epochs, minibatch_size , learning_rate, schedule , momentum ):
        "SGD with epochs and batch_size as parameters. The structure was taken by course exercises"
        losses = []
        for epoch in range(epochs):
                   
            predictions = []
            targets = []
            
            #shuffling
            randmtensor = empty(traindata.shape[0]).uniform_(0, traindata.shape[0])
            for i in range (traindata.shape[0]):                
                 tmp = traindata[int(randmtensor[i])].clone()
                 tmp2 = traintarget[int(randmtensor[i])].clone()
                 traindata[int(randmtensor[i])] = traindata[i].clone()
                 traintarget[int(randmtensor[i])] = traintarget[i].clone()
                 traindata[i] = tmp.clone()
                 traintarget[i] = tmp2.clone()

            for batch in range(0,traindata.size(0),minibatch_size):    
                if (batch+minibatch_size < traindata.size(0)):
                    prediction = self.forward(traindata[batch:batch+minibatch_size].T)
                else:
                    """avoding problems when lenght is not a multiple of the batch_size.
                    We still want to use the gradient on all directions"""
                    batch_temp = traindata.size(0) - minibatch_size -1
                    prediction = self.forward(traindata[batch_temp:batch_temp+minibatch_size].T)
                if (batch+minibatch_size < traindata.size(0)):
                    der_loss_wrt_output = self.loss.loss_grad(prediction, traintarget[batch:batch+minibatch_size].T)
                else:
                    batch_temp = traindata.size(0) - minibatch_size -1
                    der_loss_wrt_output = self.loss.loss_grad(prediction, traintarget[batch_temp:batch_temp+minibatch_size].T)
                self.backward(der_loss_wrt_output, momentum)
                self.gradient_update(learning_rate)
                self.batch_idx += minibatch_size
            """for every batch the predictions and errors are computed thanks to the predict function"""    
            actual_trainpred, actual_trainloss, actual_trainaccuracy,actual_testpred, actual_testloss, actual_testaccuracy = self.predict(traindata, traintarget, testdata,testtarget)
            print('epoch number: ', epoch, 'accuracy reached: ', actual_trainaccuracy,actual_testaccuracy)
            losses.append(actual_testloss)
            
            learning_rate = learning_rate*schedule
        "Plotting the final results below. We want to visulize the eroors and the correct predictions"
        if plot:

            
            x_loss = [i for i in range(epochs)]

            y_loss = losses

            plt.xlabel('Number of epochs')
            plt.ylabel('Loss value')
            plt.plot(x_loss,y_loss)
            plt.show()
            
        
    def predict(self,traindata, traintarget, testdata,testtarget):
        '''make a function named getoutputdim'''
        output_size = self.layers[-1].output_dimension
        
        trainpredictions = empty(traindata.shape[0], output_size)
        "compute prediction by forwarding the traindata"
        trainpredictions = self.forward(traindata.T)
        for i in range(traindata.shape[0]):
            "thresholding on 0.5"
            if trainpredictions.T[i] > 0.5:
                trainpredictions.T[i] = 1
            else:
                trainpredictions.T[i] = 0
        testpredictions = empty(testdata.shape[0], output_size)
        "compute prediction by forwarding the testdata"
        testpredictions = self.forward(testdata.T)
        for i in range(traindata.shape[0]):
            "thresholding on 0.5"
            if testpredictions.T[i] > 0.5:
                testpredictions.T[i] = 1
            else:
                testpredictions.T[i] = 0
        "Computing losses"
        if traintarget is not None:
            trainloss = self.loss.loss_value(trainpredictions.T, traintarget)
        else: trainloss = None
            
        
        if testtarget is not None:
            testloss = self.loss.loss_value(testpredictions.T, testtarget)

        else: testloss = None
        "Accuracy = nb(corrctly identified)/nb(all)"
        trainaccuracy = (trainpredictions.T == traintarget).sum() / (trainpredictions.shape[1]*trainpredictions.shape[0])
        testaccuracy = (testpredictions.T == testtarget).sum() / (testpredictions.shape[1] *testpredictions.shape[0])
        
        return trainpredictions.T, trainloss, trainaccuracy,testpredictions.T, testloss, testaccuracy
    
    def param ( self ) :
        flatten = lambda t: [item for sublist in t for item in sublist]
        return flatten([i.param() for i in self.layers])  