from torch import empty
from sequential import *
import math



def create_toy_dataset(n):
    #different points with 2 coordinate between [0,1]
    points = empty(n, 2).uniform_(0, 1)
    #check if the points are in the circle using euclidian distance from the center=[0.5,0.5]
    labels = ((points - empty(2).fill_(0.5)).norm(p=2, dim=1) < 1 / math.sqrt(2 * math.pi)).float()

    return points, labels
traindata,traintarget = create_toy_dataset(1000)
traintarget = traintarget.view(-1,1)
testdata, testtarget = create_toy_dataset(1000)
testtarget = testtarget.view(-1,1)
newmodel = Sequential(LossMSE(), 2,0)
newmodel.new_layer(Linear(2, 25))
newmodel.new_layer(Relu(25))
newmodel.new_layer(Linear(25, 25))
newmodel.new_layer(Relu(25))
newmodel.new_layer(Linear(25, 25))
newmodel.new_layer(Relu(25))
newmodel.new_layer(Linear(25, 1))
#fit(self, traindata, traintarget, testdata, testtarget, epochs, minibatch_size, learning_rate, scheduling, momentum)
newmodel.fit(traindata, traintarget, testdata, testtarget, 40, 5, 0.001,0.9,0.5)