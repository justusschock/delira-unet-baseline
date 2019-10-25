import torch
import numpy as np


def make_onehot_npy(labels, n_classes):
    """
    Function to convert a batch of class indices to onehot encoding
    Parameters
    ----------
    labels : np.ndarray
        the batch of class indices
    n_classes : int
        the number of classes
    Returns
    -------
    np.ndarray
        the onehot-encoded version of :param:`labels`
    """
    labels = labels.reshape(-1).astype(np.uint8)
    return np.eye(n_classes)[labels]


def make_onehot_torch(labels, n_classes):
    """
    Function to convert a batch of class indices to onehot encoding
    Parameters
    ----------
    labels : torch.Tensor
        the batch of class indices
    n_classes : int
        the number of classes
    Returns
    -------
    torch.Tensor
        the onehot-encoded version of :param:`labels`
    """
    idx = labels.to(dtype=torch.long)

    new_shape = list(labels.unsqueeze(dim=1).shape)
    new_shape[1] = n_classes
    labels_onehot = torch.zeros(*new_shape, device=labels.device,
                                dtype=labels.dtype)
    labels_onehot.scatter_(1, idx.unsqueeze(dim=1), 1)
    return labels_onehot