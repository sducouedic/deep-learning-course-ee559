from abc import abstractmethod

import torch
from torch import nn
from torch import optim


class Model(nn.Module):
    """ A deep network with the added capability to train itself and provide
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

    def __init__(self, f_gen_data, nb_epochs=25, mini_batch_size=100, learning_rate=1e-3):
        super().__init__()
        self.generate_data = f_gen_data
        self.epochs = nb_epochs
        self.batch_size = mini_batch_size
        self.lr = learning_rate

    def train_and_test_round(self):
        """ This method runs a "round" of generating new data, train with itself with the parameters, test the new
            produced model and generating related statistics.
        """
        train_input, train_target, train_classes, \
            test_input, test_target, test_classes = self.generate_data(self.sets_size)

        losses = self._train(train_input, train_target, train_classes)
        train_errors = self.__compute_train_errors(train_input, train_target)
        test_errors = self.__compute_test_errors(test_input, test_target)

        print('train_error {:.02f}% test_error {:.02f}%'.format(
            train_errors / train_input.size(0) * 100,
            test_errors / test_input.size(0) * 100
        )
        )
        return losses, train_errors, test_errors

    def _train(self, train_input, train_target, train_classes):
        criterion = nn.CrossEntropyLoss()

        losses = []

        for e in range(self.epochs):
            acc_loss = 0
            for b in range(0, train_input.size(0), self.batch_size):
                output = self(train_input.narrow(0, b, self.batch_size))
                loss = criterion(output, train_target.narrow(0, b, self.batch_size))
                acc_loss = acc_loss + loss.item()

                self.zero_grad()
                loss.backward()

                with torch.no_grad():
                    for p in self.parameters():
                        p -= self.lr * p.grad

            losses.append(acc_loss)

        return losses

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

        for b in range(0, data_input.size(0), self.batch_size):
            output = self(data_input.narrow(0, b, self.batch_size))
            _, predicted_classes = torch.max(output, 1)
            for k in range(self.batch_size):
                if data_target[b + k] != predicted_classes[k]:
                    nb_data_errors = nb_data_errors + 1

        return nb_data_errors
