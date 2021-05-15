from abc import abstractmethod

import torch
from torch import nn


class Model(nn.Module):
    """ Represent a deep network with the added capability to train itself and provide
        statistics about the training and testing performances.

        Attributes
        ----------
        sets_size = 1000 : int
            the size of both the training and the testing sets

        mini_batch_size = 100 : int
            size of a mini-batch

        f_gen_data : function
            a function taking a an int and generating the training and testing data

        Methods
        -------
        train_and_test_round() :
            generates new data, train itself and compute performance statistics.
    """

    sets_size = 1000
    mini_batch_size = 100

    def __init__(self, f_gen_data):
        self.generate_data = f_gen_data

    def train_and_test_round(self):
        train_input, train_target, train_classes, \
            test_input, test_target, test_classes = self.generate_data(self.sets_size)

        self.train(train_input, train_target, train_classes)
        # TODO complete

    @abstractmethod
    def _train(self, train_input, train_target, train_classes):
        pass

    def __compute_train_errors(self, train_input, train_target):
        """ Computes the number of training errors """
        return self.__compute_errors(train_input, train_target)

    def __compute_test_errors(self, test_input, test_target):
        """ Computes the number of testing errors """
        return self.__compute_errors(test_input, test_target)

    def __compute_errors(self, data_input, data_target):
        """ Computes the number of errors.

            Runs the model with data_input as input, and compare the results
            with data_target and count the number of mis-predictions.
        """
        nb_data_errors = 0

        for b in range(0, data_input.size(0), self.mini_batch_size):
            output = self(data_input.narrow(0, b, self.mini_batch_size))
            _, predicted_classes = torch.max(output, 1)
            for k in range(self.mini_batch_size):
                if data_target[b + k] != predicted_classes[k]:
                    nb_data_errors = nb_data_errors + 1

        return nb_data_errors
