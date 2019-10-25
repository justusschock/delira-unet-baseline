import logging
from abc import abstractmethod
import torch
from delira.models.backends.torch.abstract_network import AbstractPyTorchNetwork
from delira.models.backends.torch.utils import scale_loss

from delira_unet.model.nd_wrapper import ConvWrapper as ConvNdTorch, \
    PoolingWrapper as PoolingNdTorch, \
    NormWrapper as NormNdTorch
import torch
from torch.nn import functional as F
from functools import partial


class UNetTorch(AbstractPyTorchNetwork):
    """
    The :class:`UNetTorch` is a convolutional encoder-decoder neural
    network.
    Contextual spatial information (from the decoding,
    expansive pathway) about an input tensor is merged with
    information representing the localization of details
    (from the encoding, compressive pathway).
    Depending on the input argument, this implementation can also become a
    LinkNet
    Notes
    -----
    Differences to the original paper:
        * padding is used in 3x3 convolutions to prevent loss of border pixels
        * merging outputs does not require cropping due to (1)
        * residual connections can be used by specifying ``merge_mode='add'``
        * if non-parametric upsampling is used in the decoder pathway (
            specified by upmode='upsample'), then an additional 1x1 2d
            convolution occurs after upsampling to reduce channel
            dimensionality by a factor of 2. This channel halving happens
            with the convolution in the tranpose convolution (specified by
            ``upmode='transpose'``)
    References
    ----------
    https://arxiv.org/abs/1505.04597
    https://arxiv.org/abs/1707.03718
    """

    def __init__(self, num_classes, in_channels=1, depth=5,
                 start_filts=64, n_dim=2, norm_layer="Batch",
                 up_mode='transpose', merge_mode='add',
                 per_class=True):
        """
        Parameters
        ----------
        num_classes : int
            number of output classes
        in_channels : int
            number of channels for the input tensor (default: 1)
        depth : int
            number of MaxPools in the U-Net (default: 5)
        start_filts : int
            number of convolutional filters for the first conv (affects all
            other conv-filter numbers too; default: 64)
        up_mode : str
            type of upconvolution. Must be one of ['transpose', 'upsample']
            if 'transpose':
                Use transpose convolution for upsampling
            if 'upsample':
                Use bilinear Interpolation for upsampling (no additional
                trainable parameters)
            default: 'transpose'
        merge_mode : str
            mode of merging the two paths (with and without pooling). Must
            be one of ['concat', 'add']
            if 'concat':
                Concatenates along the channel dimension (Original UNet)
            if 'add':
                Adds both tensors (Residual behaviour; LinkNet)
            default: 'merge'
        per_class : bool
            whether to use a per-class final activation (Sigmoid) or a more global one (Softmax)

        """
        super().__init__()

        if up_mode in ('transpose', 'upsample'):
            self.up_mode = up_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for "
                             "upsampling. Only \"transpose\" and "
                             "\"upsample\" are allowed.".format(up_mode))

        if merge_mode in ('concat', 'add'):
            self.merge_mode = merge_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for"
                             "merging up and down paths. "
                             "Only \"concat\" and "
                             "\"add\" are allowed.".format(up_mode))

        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        if self.up_mode == 'upsample' and self.merge_mode == 'add':
            raise ValueError("up_mode \"upsample\" is incompatible "
                             "with merge_mode \"add\" at the moment "
                             "because it doesn't make sense to use "
                             "nearest neighbour to reduce "
                             "depth channels (by half).")

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = depth
        self._norm_layer = norm_layer

        self.down_convs = torch.nn.ModuleList()
        self.up_convs = torch.nn.ModuleList()

        self.conv_final = None

        self._build_model(n_dim=n_dim, num_classes=num_classes,                 in_channels=in_channels, depth=depth, start_filts=start_filts,      norm_layer=norm_layer)

        self.reset_params()
        self.per_class = per_class

        if per_class:
            self.final_activation = torch.nn.Sigmoid()
            self.prepare_batch = self.prepare_batch_per_class
        else:
            self.final_activation = torch.nn.Softmax(dim=1)
            self.prepare_batch = self.prepare_batch_multiclass

    @staticmethod
    def weight_init(m):
        """
        Initializes weights with xavier_normal and bias with zeros
        Parameters
        ----------
        m : torch.nn.Module
            module to initialize
        """
        if isinstance(m, ConvNdTorch):
            torch.nn.init.xavier_normal_(m.conv.weight)
            torch.nn.init.constant_(m.conv.bias, 0)

    def reset_params(self):
        """
        Initialize all parameters
        """
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x) -> dict:
        """
        Feed tensor through network
        Parameters
        ----------
        x : torch.Tensor
        Returns
        -------
        torch.Tensor
            Prediction
        """
        encoder_outs = []

        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i + 2)]
            x = module(before_pool, x)

        # No softmax is used during training. This means you need to use
        # torch.nn.CrossEntropyLoss is your training script,
        # as this module includes a softmax already.
        x = self.conv_final(x)

        if not self.training:
            x = self.final_activation(x)
        return {"pred": x}

    def _build_model(self, n_dim, num_classes, in_channels=1, depth=5,
                     start_filts=64, norm_layer='Batch') -> None:
        """
        Builds the actual model
        Parameters
        ----------
        num_classes : int
            number of output classes
        in_channels : int
            number of channels for the input tensor (default: 1)
        depth : int
            number of MaxPools in the U-Net (default: 5)
        start_filts : int
            number of convolutional filters for the first conv (affects all
            other conv-filter numbers too; default: 64)
        Notes
        -----
        The Helper functions and classes are defined within this function
        because ``delira`` offers a possibility to save the source code
        along the weights to completely recover the network without needing
        a manually created network instance and these helper functions have
        to be saved too.
        """

        def conv3x3(n_dim, in_channels, out_channels, stride=1,
                    padding=1, bias=True, groups=1):
            return ConvNdTorch(
                n_dim,
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=padding,
                bias=bias,
                groups=groups)

        def upconv2x2(n_dim, in_channels, out_channels, mode='transpose'):
            if mode == 'transpose':
                return ConvNdTorch(
                    n_dim,
                    in_channels,
                    out_channels,
                    kernel_size=2,
                    stride=2,
                    transposed=True
                )
            else:
                # out_channels is always going to be the same
                # as in_channels
                if n_dim == 2:
                    upsample_mode = "bilinear"
                elif n_dim == 3:
                    upsample_mode = "trilinear"
                else:
                    raise ValueError

                return torch.nn.Sequential(
                    torch.nn.Upsample(mode=upsample_mode, scale_factor=2),
                    conv1x1(n_dim, in_channels, out_channels))

        def conv1x1(n_dim, in_channels, out_channels, groups=1):
            return ConvNdTorch(
                n_dim,
                in_channels,
                out_channels,
                kernel_size=1,
                groups=groups,
                stride=1)

        class DownConv(torch.nn.Module):
            """
            A helper Module that performs 2 convolutions and 1 MaxPool.
            A ReLU activation follows each convolution.
            """

            def __init__(self, n_dim, in_channels, out_channels,
                         pooling=True, norm_layer="Batch"):
                super().__init__()

                self.n_dim = n_dim
                self.in_channels = in_channels
                self.out_channels = out_channels
                self.pooling = pooling

                self.conv1 = conv3x3(self.n_dim, self.in_channels,
                                     self.out_channels)
                self.norm1 = NormNdTorch(norm_layer, n_dim, self.out_channels)
                self.conv2 = conv3x3(self.n_dim, self.out_channels,
                                     self.out_channels)
                self.norm2 = NormNdTorch(norm_layer, n_dim, self.out_channels)

                if self.pooling:
                    self.pool = PoolingNdTorch("Max", n_dim, 2)

            def forward(self, x):
                x = F.relu(self.norm1(self.conv1(x)))
                x = F.relu(self.norm2(self.conv2(x)))
                before_pool = x
                if self.pooling:
                    x = self.pool(x)
                return x, before_pool

        class UpConv(torch.nn.Module):
            """
            A helper Module that performs 2 convolutions and 1 UpConvolution.
            A ReLU activation follows each convolution.
            """

            def __init__(self, n_dim, in_channels, out_channels,
                         merge_mode='concat', up_mode='transpose'):
                super().__init__()

                self.n_dim = n_dim
                self.in_channels = in_channels
                self.out_channels = out_channels
                self.merge_mode = merge_mode
                self.up_mode = up_mode

                self.upconv = upconv2x2(self.n_dim, self.in_channels,
                                        self.out_channels,
                                        mode=self.up_mode)

                if self.merge_mode == 'concat':
                    self.conv1 = conv3x3(
                        self.n_dim,
                        2 * self.out_channels,
                        self.out_channels)
                else:
                    # num of input channels to conv2 is same
                    self.conv1 = conv3x3(self.n_dim,
                                         out_channels,
                                         self.out_channels)
                self.norm1 = NormNdTorch(norm_layer, n_dim, self.out_channels)
                self.conv2 = conv3x3(self.n_dim,
                                     self.out_channels,
                                     self.out_channels)
                self.norm2 = NormNdTorch(norm_layer, n_dim, self.out_channels)

            def forward(self, from_down, from_up):
                from_up = self.upconv(from_up)
                if self.merge_mode == 'concat':
                    x = torch.cat((from_up, from_down), 1)
                else:
                    x = from_up + from_down
                x = F.relu(self.norm1(self.conv1(x)))
                x = F.relu(self.norm1(self.conv2(x)))
                return x

        outs = in_channels
        # create the encoder pathway and add to a list
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = start_filts * (2 ** i)
            pooling = True if i < depth - 1 else False

            down_conv = DownConv(n_dim, ins, outs, pooling=pooling)
            self.down_convs.append(down_conv)

        # create the decoder pathway and add to a list
        # - careful! decoding only requires depth-1 blocks
        for i in range(depth - 1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(n_dim, ins, outs, up_mode=self.up_mode,
                             merge_mode=self.merge_mode)
            self.up_convs.append(up_conv)

        self.conv_final = conv1x1(n_dim, outs, num_classes)

   
    @staticmethod
    def closure(model, data_dict: dict, optimizers: dict, losses={}, fold=0, **kwargs):
        """
        closure method to do a single backpropagation step
        Parameters
        ----------
        model : :class:`ClassificationNetworkBaseTorch`
            trainable model
        data_dict : dict
            dictionary containing the data
        optimizers : dict
            dictionary of optimizers to optimize model's parameters
        losses : dict
            dict holding the criterions to calculate errors
            (gradients from different criterions will be accumulated)
        fold : int
            Current Fold in Crossvalidation (default: 0)
        **kwargs:
            additional keyword arguments
        Returns
        -------
        dict
            Loss values (with same keys as input dict criterions)
        list
            Arbitrary number of predictions as torch.Tensor
        Raises
        ------
        AssertionError
            if optimizers or criterions are empty or the optimizers are not
            specified
        """

        assert (optimizers and losses) or not optimizers, \
            "Criterion dict cannot be emtpy, if optimizers are passed"

        loss_vals = {}
        metric_vals = {}
        total_loss = 0

        # choose suitable context manager:
        if optimizers:
            context_man = torch.enable_grad

        else:
            context_man = torch.no_grad

        with context_man():

            inputs = data_dict["data"]
            preds = model(inputs)

            if data_dict:

                for key, crit_fn in losses.items():
                    _loss_val = crit_fn(preds["pred"], data_dict["label"])
                    loss_vals[key] = _loss_val.detach()
                    total_loss = total_loss + _loss_val

        if optimizers:
            optimizers['default'].zero_grad()
            # perform loss scaling via apex if half precision is enabled
            with scale_loss(total_loss, optimizers["default"]) as scaled_loss:
                scaled_loss.backward()
            optimizers['default'].step()

        # perform final activation before returning the predictions
        with torch.no_grad():
            for k, v in preds.items():
                preds[k] = model.final_activation(v)

        return {k: v.cpu().numpy() for k, v in loss_vals.items()}, \
               {k: v.detach() for k, v in preds.items()}

    @staticmethod
    def prepare_batch_per_class(batch: dict, input_device, output_device):
        """
        Helper Function to prepare Network Inputs and Labels (convert them
        to correct type and shape and push them to correct devices)
        Parameters
        ----------
        batch : dict
            dictionary containing all the data
        input_device : torch.device
            device for network inputs
        output_device : torch.device
            device for network outputs
        Returns
        -------
        dict
            dictionary containing data in correct type and shape and on
            correct device
        """
        data = torch.from_numpy(
            batch.pop("data")).to(input_device).to(torch.float)
        label = torch.from_numpy(
            batch.pop("label")).to(output_device).to(torch.float)
        return {'data': data, 'label': label, **batch}


    @staticmethod
    def prepare_batch_multiclass(batch: dict, input_device, output_device):
        """
        Helper Function to prepare Network Inputs and Labels (convert them
        to correct type and shape and push them to correct devices)
        Parameters
        ----------
        batch : dict
            dictionary containing all the data
        input_device : torch.device
            device for network inputs
        output_device : torch.device
            device for network outputs
        Returns
        -------
        dict
            dictionary containing data in correct type and shape and on
            correct device
        """
        data = torch.from_numpy(
            batch.pop("data")).to(input_device).to(torch.float)
        label = torch.from_numpy(
            batch.pop("label")).to(output_device).to(torch.long).squeeze(1)
        return {'data': data, 'label': label, **batch}
