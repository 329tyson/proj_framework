import torch.nn as nn

from utils.layer_utils import summarize_network
from clfnets.modelwrapper import ModelWrapper


class ExampleNetwork(ModelWrapper):
    def __init__(self, **kwargs):
        super(ExampleNetwork, self).__init__(**kwargs)
        self.layers = []
        self.__build_layers()

        if self.verbose:
            summarize_network(self.cuda(), input_size=(3, 64, 64))

    def __build_layers(self):
        self.layers.append(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1))
        self.layers.append(nn.MaxPool2d(2))
        self.layers.append(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1))
        self.layers.append(nn.MaxPool2d(2))
        self.layers.append(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1))

        self.layers = nn.Sequential(*self.layers)

    def forward(self, *x):
        x = x[0]
        return self.layers(x)
