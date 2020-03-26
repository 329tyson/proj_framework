import argparse

from pathlib import Path


def base_parser():
    '''
        This function contains basic parser for training pytorch models
    '''
    parser = argparse.ArgumentParser(description="Base parser for training", add_help=False)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--decay_period", type=int)
    parser.add_argument("--train_batch_size", type=int)
    parser.add_argument("--test_batch_size", type=int)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--num_class", type=int)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--checkpoints", type=str, default=Path("./checkpoints"))
    parser.add_argument("--desc", type=str, default="no description supplied")

    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    parser.set_defaults(vis=False)
    parser.set_defaults(verbose=False)

    return parser
