import cv2
import torch
import wandb

import numpy as np
import torchvision.transforms as transforms
import torchvision.utils as vutils

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
    features = None

    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output

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
