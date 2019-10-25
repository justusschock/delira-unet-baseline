import torch
from delira_unet.utils import make_onehot_torch


class SoftDiceLossPyTorch(torch.nn.Module):
    def __init__(self, square_nom=False, square_denom=False, weight=None,
                 smooth=1., reduction="elementwise_mean", non_lin=None):
        """
        SoftDice Loss
        Parameters
        ----------
        square_nom : bool
            square nominator
        square_denom : bool
            square denominator
        weight : iterable
            additional weighting of individual classes
        smooth : float
            smoothing for nominator and denominator
        """
        super().__init__()
        self.square_nom = square_nom
        self.square_denom = square_denom

        self.smooth = smooth

        if weight is not None:
            self.register_buffer("weight", torch.tensor(weight))
        else:
            self.weight = None

        self.reduction = reduction
        self.non_lin = non_lin

    def forward(self, inp, target):
        """
        Compute SoftDice Loss
        Parameters
        ----------
        inp : torch.Tensor
            prediction
        target : torch.Tensor
            ground truth tensor
        Returns
        -------
        torch.Tensor
            loss
        """
        # number of classes for onehot
        n_classes = inp.shape[1]
        with torch.no_grad():
            target_onehot = make_onehot_torch(target, n_classes=n_classes)
        # sum over spatial dimensions
        dims = tuple(range(2, inp.dim()))

        # apply nonlinearity
        if self.non_lin is not None:
            inp = self.non_lin(inp)

        # compute nominator
        if self.square_nom:
            nom = torch.sum((inp * target_onehot.float()) ** 2, dim=dims)
        else:
            nom = torch.sum(inp * target_onehot.float(), dim=dims)
        nom = 2 * nom + self.smooth

        # compute denominator
        if self.square_denom:
            i_sum = torch.sum(inp ** 2, dim=dims)
            t_sum = torch.sum(target_onehot ** 2, dim=dims)
        else:
            i_sum = torch.sum(inp, dim=dims)
            t_sum = torch.sum(target_onehot, dim=dims)

        denom = i_sum + t_sum.float() + self.smooth

        # compute loss
        frac = nom / denom

        # apply weight for individual classesproperly
        if self.weight is not None:
            frac = weight * frac

        # average over classes
        frac = - torch.mean(frac, dim=1)

        if self.reduction == 'elementwise_mean':
            return torch.mean(frac)
        if self.reduction == 'none':
            return frac
        if self.reduction == 'sum':
            return torch.sum(frac)
        raise AttributeError('Reduction parameter unknown.')