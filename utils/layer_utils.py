from pathlib import Path
from torchsummary import summary

import torch

import torch.nn as nn
import torchvision.models as models


def summarize_network(model: nn.Module, input_size: tuple = (3, 224, 224)):
    '''
        This function summarize and show network forward_paths
        Input size of network are set to (224, 224) image by default
    '''
    summary(model, input_size=input_size)


def adjust_last_fc(model: nn.Module, num_classes: int):
    '''
        This functions adjusts last fc for fintuning operation
        For example, if using *Resnet18* for finetuning on CUB dataset,
        adjust_last_fc(models.resnet18(), 200) returns resnet18 with last fc adjusted for CUB(512, 200)
    '''
    if isinstance(model, models.ResNet):
        final_layer = model.fc

        in_features = final_layer.in_features
        final_layer = nn.Linear(in_features, num_classes)
        model.fc = final_layer
    else:
        raise NotImplementedError


def load_weights(model: nn.Module, path_to_weights):
    '''
        This functions loads weights from given path.
        Path variable can be either pathlib.path object or string
        It also handles weights trained in multi-gpu format.
    '''
    if isinstance(path_to_weights, Path):
        path = path_to_weights.resolve().as_posix()

    elif isinstance(path_to_weights, str):
        path = path_to_weights

    else:
        raise NotImplementedError

    new_weights = {}
    weights = torch.load(path)

    # in case it loads weights trained in multi-gpu format
    for k, v in weights.items():
        if "module." in k:
            new_weights[k.replace("module.", "")] = v
        else:
            new_weights[k] = v

    model.load_state_dict(new_weights)
    return model


class SobelFilter(nn.Module):
    '''
        This class builds backprop-able soblefilter.
        By default, parameters of this layer are set to non-trainable.
    '''
    def __init__(self):
        super(SobelFilter, self).__init__()
        grayscale = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0)
        grayscale.weight.data.fill_(1.0 / 3.0)
        grayscale.bias.data.zero_()
        sobel_filter = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1)
        sobel_filter.weight.data[0, 0].copy_(
            torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        )
        sobel_filter.weight.data[1, 0].copy_(
            torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        )
        sobel_filter.bias.data.zero_()
        self.sobel = nn.Sequential(grayscale, sobel_filter)

        for p in self.sobel.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.sobel(x)
