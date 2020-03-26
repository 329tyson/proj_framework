import tqdm
import wandb
import torch
import argparse

import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from pathlib import Path

from utils.vis_utils import plot_to_local
from utils.vis_utils import plot_images_to_wandb
from utils.layer_utils import summarize_network
from utils.layer_utils import adjust_last_fc
from utils.augmentation import imagenet_normalize
from parsers.basic_parser import base_parser
from evaluations.evaluator import count_match
from evaluations.evaluator import log_best_state
from evaluations.evaluator import evaluate_classification
# from clfnets.example_network import ExampleNetwork


def main(args):
    '''
        This example script contains way to utilize this framework repository on Cifar-10
        - how to use dataloaders
        - how to use modelwrapper
    '''

    if not args.checkpoints.exists():
        print(f"Path {args.checkpoints} doesn't exists, Making one...")
        Path.mkdir(args.checkpoints)
    else:
        pass
        # Checkpoint directory already exists...?

    # Build your dataloader here or from another script
    tsfrm = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        imagenet_normalize(),
    ])

    test_tsfrm = transforms.Compose([
        transforms.ToTensor(),
        imagenet_normalize(),
    ])

    cifar_dataset = datasets.CIFAR10("/data/Cifar10/", train=True, transform=tsfrm, download=True)
    cifar_test = datasets.CIFAR10("/data/Cifar10/", train=False, transform=test_tsfrm, download=True)

    cifar_loader = data.DataLoader(
        dataset=cifar_dataset,
        num_workers=args.num_workers,
        shuffle=True,
        batch_size=args.train_batch_size,
        pin_memory=True,
    )

    cifar_test_loader = data.DataLoader(
        dataset=cifar_test,
        num_workers=args.num_workers,
        batch_size=args.test_batch_size,
        pin_memory=True,
    )

    # If using custom network,
    # model = ExampleNetwork(a=1, b=2, c=3, verbose=True)

    model = models.resnet18(pretrained=True)
    adjust_last_fc(model, args.num_class)
    wandb.watch(model)
    model.cuda()

    summarize_network(model, input_size=(3, 32, 32))

    params_list = []
    params_list.append({"params": model.conv1.parameters()})
    params_list.append({"params": model.layer1.parameters()})
    params_list.append({"params": model.layer2.parameters()})
    params_list.append({"params": model.layer3.parameters()})
    params_list.append({"params": model.layer4.parameters()})
    params_list.append({"params": model.fc.parameters(), "lr": args.lr * 10})

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(params_list, lr=args.lr, momentum=0.9)

    print(f"Cifar10 load complete with len({len(cifar_loader)}/{len(cifar_dataset)})")
    idx = 0
    best_acc = 0.
    for epoch in range(args.epochs):
        model.train()
        clf_loss = 0
        for i, batch in enumerate(tqdm.tqdm(cifar_loader, desc=f"Training[{epoch}]")):
            image, label = batch
            image = image.cuda()
            label = label.cuda()

            pred = model(image)

            # uncomment following lines to plot training images
            if i == 0 and args.vis:
                # If you want to plot images to local
                # plot_to_local(idx, image[:4])

                # If you want to plot images to wandb
                plot_images_to_wandb(image[:4], name="Train_images", step=epoch)
                idx += 1

            loss = criterion(pred, label)
            clf_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        wandb.log({
            "Loss@Clf": clf_loss / len(cifar_loader),
        }, step=epoch)

        print(f"Epoch[{epoch}] loss: {clf_loss / len(cifar_loader)}")

        acc = evaluate_classification(model, cifar_test_loader, epoch=epoch)
        best_acc = log_best_state(model, args.desc, acc, best_acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[base_parser()])
    parser.add_argument("--wandb_proj_name", type=str, required=True)

    args = parser.parse_args()
    wandb.init(project=args.wandb_proj_name, name=args.desc, config=args)
    main(args)
