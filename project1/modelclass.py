from abc import abstractmethod

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim


class Model(nn.Module):
    """ A deep network with the added capability to train itself and provide
        statistics about the training and testing performances.

        Attributes
        ----------
        sets_size : int
            the size of both the training and the testing sets

        name : string
            name of the model

        useAuxiliary : Boolean
            determines if the model uses auxiliary loss or not

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

    def __init__(self, f_gen_data, nb_epochs=25, mini_batch_size=100, learning_rate=1e-3):
        super().__init__()
        self.generate_data = f_gen_data
        self.epochs = nb_epochs
        self.batch_size = mini_batch_size
        self.lr = learning_rate
        self.useAuxiliary = False  # Need to resize tensors if True

    @abstractmethod
    def reset(self):
        """ Reinitialize the weights of the model, need to be redefined"""

    def train_and_test(self):
        """ The model runs a complete "round" consisting of the following steps :
            1. Generate new train and test data
            2. Trains with the new generated data
            3. Compute the train and test error rates and other performance statistics

            Returns
            -------
            train_err_rate : double

            test_err_rate :double

            losses : array[double]
                the loss at each iteration
        """

        train_input, train_target, train_classes, \
        test_input, test_target, test_classes = self.generate_data(self.sets_size)

        if self.useAuxiliary:
            train_input = train_input.view(-1, 1, 14, 14)
            train_classes = train_classes.view(-1)
            test_input = test_input.view(-1, 1, 14, 14)
            test_classes = test_classes.view(-1)

        losses = self._train(train_input, train_target, train_classes)
        train_err_rate = self.__compute_errors(train_input, train_target)
        test_err_rate = self.__compute_errors(test_input, test_target)

        return train_err_rate, test_err_rate, losses

    def _train(self, train_input, train_target, train_classes):
        """ Train the model """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), self.lr)

        losses = []

        for e in range(self.epochs):
            for b in range(0, train_input.size(0), self.batch_size):

                if self.useAuxiliary:
                    digit_class, final_class = self(train_input.narrow(0, b, self.batch_size))
                    loss = criterion(
                        final_class,
                        train_target.narrow(0, b // 2, self.batch_size // 2)
                    ) + criterion(
                        digit_class,
                        train_classes.narrow(0, b, self.batch_size)
                    )

                else:
                    final_class = self(train_input.narrow(0, b, self.batch_size))
                    loss = criterion(final_class, train_target.narrow(0, b, self.batch_size))

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

            if self.useAuxiliary:
                output = output[1]

            _, predicted_classes = torch.max(output, 1)

            target_b = b if not self.useAuxiliary else b // 2
            target_batch_size = self.batch_size if not self.useAuxiliary else self.batch_size // 2

            for k in range(target_batch_size):
                if data_target[target_b + k] != predicted_classes[k]:
                    nb_data_errors = nb_data_errors + 1

        return 100 * nb_data_errors / data_input.size(0)
