# Personal PyTorch Framework for DeepLearning

## Overview

This repository contains basic framework components for training PyTorch networks.

All scripts are separated within folders depending on its use and purposes.

Folder names are follows:

```
- clfnets
- dloaders
- evaluations
- parsers
- utils
```

Each descriptions are supplied for each sections.

## clfnets

This folder contains python scripts for classification networks.

For example, *modelwrapper.py* supplies wrapping class for building custom PyTorch models.

*clfnets/example_network.py* shows how to use this library for building your own networks.

## dloaders

This folder is intentionally left empty.

It's designed to have custom dataloaders, which is not necessarilly filled for framework repository.


## parsers

This folder is desinged to contain argparsers.

For now, it has basic argparser for training classical classification network.

You may build another argparser inheriting existing *parsers/basic_parser.py*.


## utils

This folder contains utility scripts frequently used in training networks.

For example, *layer_utils.py* contains utility functions like *adjust_last_fc* for finetuning,

*vis_utils.py* contains functions related to visualization.

You may add another functions to this file that are not fit into existing categories of this repository.


To be continued ...
