import torch
from torch import nn
from torch import optim

from plot import ModelResult


class Model(nn.Module):
    """ A deep network with the added capability to train itself and provide
        statistics about the training and testing performances.

        Attributes
        ----------
        sets_size : int
            the size of both the training and the testing sets

        name : string
            name of the model

        f_gen_data : function
            a function taking a an int and generating the training and testing data

        batch_size : int
            size of a mini-batch

        Methods
        -------
        train_and_test_round() :
            generates new data, train itself and compute performance statistics.

        train_and_test_rounds(nb_rounds) :
            does nb_rounds iteration of train_and_test_round
    """

    sets_size = 1000

    def __init__(self, f_gen_data, name, nb_epochs=25, mini_batch_size=100, learning_rate=1e-3):
        super().__init__()
        self.generate_data = f_gen_data
        self.name = name
        self.epochs = nb_epochs
        self.batch_size = mini_batch_size
        self.lr = learning_rate

    def train_and_test(self):
        """ The model runs a complete "round" consisting of the following steps :
            1. Generate new train and test data
            2. Trains with the new generated data
            3. Compute the train and test error rates and other performance statistics

            Returns
            -------
            TODO complete doc
        """

        train_input, train_target, train_classes, \
        test_input, test_target, test_classes = self.generate_data(self.sets_size)

        losses = self._train(train_input, train_target, train_classes)
        train_err_rate = self.__compute_errors(train_input, train_target)
        test_err_rate = self.__compute_errors(test_input, test_target)

        return train_err_rate, test_err_rate, losses

    def train_and_test_rounds(self, nb_rounds):
        """ Complete nb_rounds iterations of train_and_test and returns the train and test error
            rates and other performance statistics in an ModelResult object

            Parameter
            ---------
            nb_rounds : int
                the number of round

            Returns
            -------
            a ModelResult object containing the overall rounds performance statistics
        """

        trains_err_rates = []
        tests_err_rates = []
        losses = []

        for i in range(nb_rounds):
            trains_err_rate, tests_err_rate, losses_ = self.train_and_test()
            trains_err_rates.append(trains_err_rate)
            tests_err_rates.append(tests_err_rate)
            losses += losses_

        return ModelResult(self.name, trains_err_rates, tests_err_rates, losses)

    def _train(self, train_input, train_target, train_classes):
        """ Train the model (method can be override by child class) """

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), self.lr)

        losses = []

        for e in range(self.epochs):
            for b in range(0, train_input.size(0), self.batch_size):
                output = self(train_input.narrow(0, b, self.batch_size))
                loss = criterion(output, train_target.narrow(0, b, self.batch_size))
                self.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.data.item())

        return losses

    # --- Private methods --- #

    def __compute_errors(self, data_input, data_target):
        """ Computes the number of errors produced by the model """
        nb_data_errors = 0

        for b in range(0, data_input.size(0), self.batch_size):
            output = self(data_input.narrow(0, b, self.batch_size))
            _, predicted_classes = torch.max(output, 1)
            for k in range(self.batch_size):
                if data_target[b + k] != predicted_classes[k]:
                    nb_data_errors = nb_data_errors + 1

        return 100 * nb_data_errors / data_input.size(0)
