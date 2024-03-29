from abc import ABC

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

from modelclass import Model


class Baseline(Model):
    """ The Baseline model is composed only of fully-connected layers. """

    def __init__(self, f_gen_data, nb_epochs=25, mini_batch_size=100, learning_rate=0.00112,
                 l2_rate=0.33333333):
        super().__init__(f_gen_data, nb_epochs, mini_batch_size, learning_rate, l2_rate)

        # layer 1
        self.fc1 = nn.Linear(392, 160)
        self.bn1 = nn.BatchNorm1d(160)
        self.drop1 = nn.Dropout(0.5)

        # layer 2
        self.fc2 = nn.Linear(80, 100)
        self.bn2 = nn.BatchNorm1d(100)
        self.drop2 = nn.Dropout(0.5)

        # layer 3
        self.fc3 = nn.Linear(50, 10)

        # layer 4
        self.fc4 = nn.Linear(10, 2)

    def forward(self, x):
        n = x.size()[0]

        # layer 1
        x = F.relu(self.bn1(self.fc1(x.view(n, -1))))
        x = F.max_pool1d(x.view(n, 1, -1), kernel_size=2)
        x = self.drop1(x)

        # layer 2
        x = F.relu(self.bn2(self.fc2(x.view(n, -1))))
        x = F.max_pool1d(x.view(n, 1, -1), kernel_size=2)
        x = self.drop2(x)

        # layer 3
        x = F.relu(self.fc3(x.view(n, -1)))

        # layer 4
        x = self.fc4(x)
        return x

    def reset(self):
        self.__init__(self.generate_data, self.epochs, self.batch_size, self.lr, self.l2)


class Base_Aux(Model):
    """ This model extends the baseline model by adding auxiliary loss """

    def __init__(self, f_gen_data, nb_epochs=25, mini_batch_size=100, learning_rate=0.0056,
                 l2_rate=0.0):
        super().__init__(f_gen_data, nb_epochs, mini_batch_size, learning_rate, l2_rate)

        # tell parent class we use auxiliary loss
        self.useAuxiliary = True

        # layer 1
        self.fc1 = nn.Linear(196, 160)
        self.bn1 = nn.BatchNorm1d(160)
        self.drop1 = nn.Dropout(0.5)

        # layer 2
        self.fc2 = nn.Linear(80, 100)
        self.bn2 = nn.BatchNorm1d(100)
        self.drop2 = nn.Dropout(0.5)

        # layer 3 : digit classification
        self.fc3 = nn.Linear(50, 10)

        # layer 4 : digit comparison
        self.fc4 = nn.Linear(20, 2)

    def forward(self, x):
        n = x.size()[0]

        # layer 1
        x = F.relu(self.bn1(self.fc1(x.view(n, -1))))
        x = F.max_pool1d(x.view(n, 1, -1), kernel_size=2)
        x = self.drop1(x)

        # layer 2
        x = F.relu(self.bn2(self.fc2(x.view(n, -1))))
        x = F.max_pool1d(x.view(n, 1, -1), kernel_size=2)
        x = self.drop2(x)

        # layer 3 : digit classification
        digit_class = self.fc3(x.view(n, -1))
        x = F.relu(digit_class)

        # layer 4 : digit comparison
        final_class = self.fc4(x.view(x.size()[0] // 2, -1))
        return digit_class, final_class

    def reset(self):
        self.__init__(self.generate_data, self.epochs, self.batch_size, self.lr, self.l2)


class CNN(Model):
    """ This model implements weight sharing """

    def __init__(self, f_gen_data, nb_epochs=25, mini_batch_size=100, learning_rate=1e-3,
                 l2_rate=0.1):
        super().__init__(f_gen_data, nb_epochs, mini_batch_size, learning_rate, l2_rate)

        # 1st layer : convolutional
        self.conv1 = nn.Conv2d(2, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(32)

        # 2nd layer : convolutional
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=3)
        self.bn2 = nn.BatchNorm2d(64)
        # 3rd layer : convolutional
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=3)
        self.bn3 = nn.BatchNorm2d(32)
        # 4th layer : convolutional
        self.conv4 = nn.Conv2d(32, 16, kernel_size=5, padding=3)
        self.bn4 = nn.BatchNorm2d(16)
        # 5th layer : fully-connected
        self.fc1 = nn.Linear(144, 100)
        # 6th layer : fully-connected (digit classification)
        self.fc2 = nn.Linear(100, 10)
        # 7th layer : fully-connected (digit comparison)
        self.fc3 = nn.Linear(10, 2)

    def forward(self, x):
        # layer 1
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), kernel_size=3)
        # layer 2
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), kernel_size=2)
        # layer 3
        x = F.max_pool2d(F.relu(self.bn3(self.conv3(x))), kernel_size=2)
        # layer 4
        x = F.max_pool2d(F.relu(self.bn4(self.conv4(x))), kernel_size=2)
        # layer 5
        x = F.relu(self.fc1(x.view(x.size()[0], -1)))

        # layer 6 : digit classification
        x = F.relu(self.fc2(x))

        # layer 7 : digit comparison
        x = (self.fc3(x))
        return x

    def reset(self):
        self.__init__(self.generate_data, self.epochs, self.batch_size, self.lr, self.l2)


class CNN_Aux(Model):
    """ This model implements weight sharing uses auxiliary loss """

    def __init__(self, f_gen_data, nb_epochs=25, mini_batch_size=100, learning_rate=0.00223,
                 l2_rate=0.11111111):
        super().__init__(f_gen_data, nb_epochs, mini_batch_size, learning_rate, l2_rate)

        # tell parent class we use auxiliary loss
        self.useAuxiliary = True

        # 1st layer : convolutional
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(32)

        # 2nd layer : convolutional
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=3)
        self.bn2 = nn.BatchNorm2d(64)

        # 3rd layer : convolutional
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=3)
        self.bn3 = nn.BatchNorm2d(32)

        # 4th layer : convolutional
        self.conv4 = nn.Conv2d(32, 16, kernel_size=5, padding=3)
        self.bn4 = nn.BatchNorm2d(16)

        # 5th layer : fully-connected
        self.fc1 = nn.Linear(144, 100)

        # 6th layer : fully-connected (digit classification)
        self.fc2 = nn.Linear(100, 10)

        # 7th layer : fully-connected (digit comparison)
        self.fc3 = nn.Linear(20, 2)

    def forward(self, x):
        # layer 1
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), kernel_size=3)

        # layer 2
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), kernel_size=2)

        # layer 3
        x = F.max_pool2d(F.relu(self.bn3(self.conv3(x))), kernel_size=2)

        # layer 4
        x = F.max_pool2d(F.relu(self.bn4(self.conv4(x))), kernel_size=2)

        # layer 5
        x = F.relu(self.fc1(x.view(x.size()[0], -1)))

        # layer 6 : digit classification
        digit_class = F.relu(self.fc2(x))
        x = F.relu(digit_class)

        # layer 7 : digit comparison
        # artificially create two channel
        final_class = self.fc3(x.view(x.size()[0] // 2, -1))
        return digit_class, final_class

    def reset(self):
        self.__init__(self.generate_data, self.epochs, self.batch_size, self.lr, self.l2)

