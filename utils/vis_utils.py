import torch
import wandb

import torchvision.utils as vutils

from pathlib import Path
from torchviz import make_dot
from torch.autograd import Variable


def plot_to_local(index:int, tensors, directory: Path = Path("./results")):
    '''
        Save pytorch tensor images to results folder with its index
    '''
    if isinstance(directory, str):
        directory = Path(directory)

    save_path = directory / Path(f"{index}.jpg")
    vutils.save_image(tensors, save_path, padding=0, normalize=True, scale_each=True)



def plot_images_to_wandb(images: list, name: str):
    # images are should be list of RGB images tensors in shape (C, H, W)
    images = vutils.make_grid(images, normalize=True, range=(-2.11785, 2.64005))

    if images.dim() == 3:
        images = images.permute(1, 2, 0)
    images = images.detach().cpu().numpy()

    images = wandb.Image(images, caption=name)

    wandb.log({name: images})


def plot_network(model, name):
    x = Variable(torch.randn(32, 3, 227, 227))
    y = model(x)
    g = make_dot(y, params=dict(model.named_parameters()))
    g.format = "png"

    save_path = Path(f"model_plots/{name}")
    if not save_path.exists():
        print(f"Model visualised in [{save_path.parent}]")
        Path.mkdir(save_path.parent)

    g.render(f"model_plots/{name}")
