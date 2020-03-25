import torch


def count_match(pred, label):
    '''
        This function returns the number of matching element in tensors
        For numpy operation, you need to implement separate function
        We assume pred tensor is in shape (B, num_class)
        And labels are in shape (B)
    '''
    _, idx = torch.max(pred, dim=1)
    return len(torch.nonzero(idx == label))
