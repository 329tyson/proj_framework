import cv2
import torch
import wandb

import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import torchvision.transforms as transforms

from pathlib import Path
from torchviz import make_dot
from torch.autograd import Variable


def plot_to_local(index: int, tensors, directory: Path = Path("./results")):
    '''
        Save pytorch tensor images to results folder with its index
    '''
    if isinstance(directory, str):
        directory = Path(directory)

    save_path = directory / Path(f"{index}.jpg")
    vutils.save_image(tensors, save_path, padding=0, normalize=True, scale_each=True)


def plot_images_to_wandb(images: list, name: str, step: int):
    # images are should be list of RGB images tensors in shape (C, H, W)
    images = vutils.make_grid(images, normalize=True, scale_each=True, padding=1)

    if images.dim() == 3:
        images = images.permute(1, 2, 0)
    images = images.detach().cpu().numpy()

    images = wandb.Image(images, caption=name)

    wandb.log({name: images}, step=step)


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



class Hook_Grad():
    grad = None
    def __init__(self, m):
        self.hook = m.register_backward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.grad = torch.sum(torch.abs(output[0].detach()), dim=1)

    def remove(self):
        self.hook.remove()


class Hook_Feature():
    def __init__(self, m, name="default"):
        '''
            1. use this function when you need direct access to feature.
            2. use this function when you want to plot CAM.
        '''
        self.count = 0
        self.name = name
        self.hook = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # assume output is just feature itself.
        self.features = output
        if not self.count == 0:
            self.channel_wise_activation = np.mean(output.detach().cpu().numpy(), axis=(0, 2, 3)) + self.channel_wise_activation * self.count
            self.count += 1
            self.channel_wise_activation = self.channel_wise_activation / self.count
        else:
            self.channel_wise_activation = np.mean(output.detach().cpu().numpy(), axis=(0, 2, 3))
            self.count += 1

    def plot_attention_to_wandb(self, epoch):
        '''
            returns its accumulated average attention in python-dictionary format
        '''
        fig, ax = plt.subplots()
        ax.bar(list(range(len(self.channel_wise_activation))), self.channel_wise_activation, label=self.name, align="center")
        ax.set_xlabel("Channels")
        ax.set_ylabel("Attentions")
        ax.legend()

        wandb.log({
            self.name: plt,
        }, step=epoch)

    def clear(self):
        self.count = 0

    def remove(self):
        self.hook.remove()


def compute_gradCAMs(feature, grad):
    b, c, h, w = feature.shape
    weight = torch.unsqueeze(grad, dim=1)

    gradCAM = (feature * weight).sum(dim=1)
    gradCAM = gradCAM.view(b, h, w)

    return gradCAM

def plot_gradcam(overlay, image, image_size=(32, 32)):
    image = image.detach().cpu()
    cams = []
    for ov, im in zip(overlay, image):
        ov -= ov.min()
        ov /= ov.max()
        ov = ov.detach().cpu().numpy()
        ov = cv2.resize(ov, image_size)
        heatmap = cv2.applyColorMap(np.uint8(255 * ov), cv2.COLORMAP_JET)
        heatmap[np.where(heatmap < 0.2)] = 0
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = transforms.ToTensor()(heatmap)
        heatmap = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(heatmap)
        cam = im + heatmap * 0.5
        cams.append(cam)

    cams = torch.stack(cams, dim=0)
    return cams
