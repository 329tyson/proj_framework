import tqdm
import torch
import wandb

import torch.nn as nn

from pathlib import Path


def count_match(pred, label):
    '''
        This function returns the number of matching element in tensors
        For numpy operation, you need to implement separate function
        We assume pred tensor is in shape (B, num_class)
        And labels are in shape (B)
    '''
    _, idx = torch.max(pred, dim=1)
    return len(torch.nonzero(idx == label))


def evaluate_classification(model, dataloader, epoch: int):
    total_hit = 0
    test_loss = 0.
    criterion = nn.CrossEntropyLoss().cuda()
    with torch.no_grad():
        model.eval()
        for batch in tqdm.tqdm(dataloader, desc=f"Testing[{epoch}]"):
            image, label = batch
            image = image.cuda()
            label = label.cuda()

            pred = model(image)
            total_hit += count_match(pred, label)
            loss = criterion(pred, label)

            test_loss += loss.item()

    acc = total_hit / len(dataloader.dataset) * 100
    wandb.log({
        "Loss@Val": test_loss / len(dataloader),
        "ACC@1": acc,
    }, step=epoch)
    print(f"Epoch[{epoch}] test loss: {test_loss / len(dataloader)}, acc: {acc:.3f}")

    return acc


def log_best_state(model, description, acc, best_acc):
    if acc > best_acc:
        save_path = Path(f"./checkpoints/{description}/")
        if not save_path.exists():
            Path.mkdir(save_path)
        else:
            file_name = Path(f"Acc_{best_acc:.3f}.ckpt.pt")
            Path.unlink(save_path / file_name)
        file_name = Path(f"Acc_{acc:.3f}.ckpt.pt")
        wandb.run.summary["best_accuracy"] = best_acc
        torch.save(model.state_dict(), (save_path / file_name).resolve().as_posix())
        return acc

    return best_acc
