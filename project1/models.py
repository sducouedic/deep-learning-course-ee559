from abc import ABC

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

from modelclass import Model


class Baseline(Model):
    """ The Baseline model is composed only of fully-connected layers. """

    def __init__(self, f_gen_data, nb_epochs=25, mini_batch_size=100, learning_rate=1e-3):
        super().__init__(f_gen_data, nb_epochs, mini_batch_size, learning_rate)

        self.fc1 = nn.Linear(392, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 10)
        self.fc4 = nn.Linear(10, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x.view(x.size()[0], -1)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x

    def reset(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()
        self.fc4.reset_parameters()


# TODO make this work and compare
class Baseline2(Model):
    """ The Baseline model is composed only of fully-connected layers. """

    def __init__(self, f_gen_data, nb_epochs=25, mini_batch_size=100, learning_rate=1e-3):
        super().__init__(f_gen_data, nb_epochs, mini_batch_size, learning_rate)

        self.fc1 = nn.Linear(392, 200)
        self.bn1 = nn.BatchNorm1d(200)
        self.fc2 = nn.Linear(200, 100)
        self.bn2 = nn.BatchNorm1d(100)
        self.fc3 = nn.Linear(100, 10)
        self.fc4 = nn.Linear(10, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x.view(x.size()[0], -1))))
        x = F.max_pool1d(x, kernel_size=2)
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.max_pool1d(x, kernel_size=2)

        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x

    def reset(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()
        self.fc4.reset_parameters()


class Auxiliary(Model):
    """ This model extends the baseline model by adding auxiliary loss """

    def __init__(self, f_gen_data, nb_epochs=25, mini_batch_size=100, learning_rate=1e-3):
        super().__init__(f_gen_data, nb_epochs, mini_batch_size, learning_rate)

        # tell parent class we use auxiliary loss
        self.useAuxiliary = True

        self.fc1 = nn.Linear(196, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 10)
        self.fc4 = nn.Linear(10, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x.view(x.size()[0], -1)))
        x = F.relu(self.fc2(x))
        digit_class = self.fc3(x)

        x = F.relu(digit_class)
        x = F.relu(self.fc4(x))
        final_class = x.view(x.size()[0] // 2, -1)
        return digit_class, final_class

    def reset(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()
        self.fc4.reset_parameters()


# TODO is this really weight sharing?
# TODO add max_pooling
class CNN(Model):
    """ This model implements weight sharing """

    def __init__(self, f_gen_data, nb_epochs=25, mini_batch_size=100, learning_rate=1e-3):
        super().__init__(f_gen_data, nb_epochs, mini_batch_size, learning_rate)

        self.conv1 = nn.Conv2d(2, 32, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 16, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(16)

        self.fc1 = nn.Linear(576, 200)
        self.fc2 = nn.Linear(200, 10)
        self.fc3 = nn.Linear(10, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.fc3(x)
        x = F.relu(x)
        return x


class CNN_Auxiliary(Model):
    """ This model implements weight sharing """

    def __init__(self, f_gen_data, nb_epochs=25, mini_batch_size=100, learning_rate=1e-3):
        super().__init__(f_gen_data, nb_epochs, mini_batch_size, learning_rate)

        # tell parent class we use auxiliary loss
        self.useAuxiliary = True

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 16, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(16)

        self.fc1 = nn.Linear(576, 200)
        self.fc2 = nn.Linear(200, 10)
        self.fc3 = nn.Linear(10, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        digit_class = self.fc2(x)

        x = F.relu(digit_class)
        x = F.relu(self.fc3(x))
        final_class = x.view(x.size()[0] // 2, -1)

        return digit_class, final_class

    def reset(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()

    def _train(self, train_input, train_target, train_classes):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), self.lr)

        losses = []

        for e in range(self.epochs):
            for b in range(0, train_input.size(0), self.batch_size):
                digit_class, final_class = self(train_input.narrow(0, b, self.batch_size))

                regLoss = criterion(final_class,
                                    train_target.narrow(0, b // 2, self.batch_size // 2))

                auxLoss = criterion(digit_class, train_classes.narrow(0, b, self.batch_size))
                loss = regLoss + auxLoss

                self.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.data.item())

        return losses
