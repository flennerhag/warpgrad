"""Base Omniglot models. Based on original implementation:
https://github.com/amzn/metalearn-leap
"""
import torch.nn as nn
from wrapper import (WarpGradWrapper, LeapWrapper, MAMLWrapper, NoWrapper,
                      FtWrapper, FOMAMLWrapper, ReptileWrapper)

NUM_CLASSES = 50
ACT_FUNS = {
    'none': None,
    'leakyrelu': nn.LeakyReLU,
    'relu': nn.ReLU,
    'sigmoid': nn.Sigmoid,
    'tanh': nn.Tanh
}


def get_model(args, criterion):
    """Construct model from main args"""
    kwargs = dict(num_classes=args.classes,
                  num_layers=args.num_layers,
                  kernel_size=args.kernel_size,
                  num_filters=args.num_filters,
                  imsize=args.imsize,
                  padding=args.padding,
                  batch_norm=args.batch_norm,
                  multi_head=args.multi_head)
    if "warp" in args.meta_model.lower():
        model = WarpedOmniConv(warp_num_layers=args.warp_num_layers,
                               warp_num_filters=args.warp_num_filters,
                               warp_residual_connection=args.warp_residual,
                               warp_act_fun=args.warp_act_fun,
                               warp_batch_norm=args.warp_batch_norm,
                               warp_final_head=args.warp_final_head,
                               **kwargs)
    else:
        model = OmniConv(**kwargs)

    if args.cuda:
        model = model.cuda()

    if args.log_ival > 0:
        print(model)

    if "warp" in args.meta_model.lower():
        return WarpGradWrapper(
            model,
            args.inner_opt,
            args.outer_opt,
            args.inner_kwargs,
            args.outer_kwargs,
            args.meta_kwargs,
            criterion)

    if args.meta_model.lower() == 'leap':
        return LeapWrapper(
            model,
            args.inner_opt,
            args.outer_opt,
            args.inner_kwargs,
            args.outer_kwargs,
            args.meta_kwargs,
            criterion,
        )

    if args.meta_model.lower() == 'no':
        return NoWrapper(
            model,
            args.inner_opt,
            args.inner_kwargs,
            criterion,
        )

    if args.meta_model.lower() == 'ft':
        return FtWrapper(
            model,
            args.inner_opt,
            args.inner_kwargs,
            criterion,
        )

    if args.meta_model.lower() == 'fomaml':
        return FOMAMLWrapper(
            model,
            args.inner_opt,
            args.outer_opt,
            args.inner_kwargs,
            args.outer_kwargs,
            criterion,
        )

    if args.meta_model.lower() == 'reptile':
        return ReptileWrapper(
            model,
            args.inner_opt,
            args.outer_opt,
            args.inner_kwargs,
            args.outer_kwargs,
            criterion,
        )

    if args.meta_model.lower() == 'maml':
        return MAMLWrapper(
            model,
            args.inner_opt,
            args.outer_opt,
            args.inner_kwargs,
            args.outer_kwargs,
            criterion,
        )
    raise NotImplementedError('Meta-learner {} unknown.'.format(
        args.meta_model.lower()))


###############################################################################


class UnSqueeze(nn.Module):

    """Create channel dim if necessary."""

    def __init__(self):
        super(UnSqueeze, self).__init__()

    def forward(self, input):
        """Creates channel dimension on a 3-d tensor.

        Null-op if input is a 4-d tensor.

        Arguments:
            input (torch.Tensor): tensor to unsqueeze.
        """
        if input.dim() == 4:
            return input
        return input.unsqueeze(1)


class Squeeze(nn.Module):

    """Undo excess dimensions"""

    def __init__(self):  # pylint: disable=useless-super-delegation
        super(Squeeze, self).__init__()

    def forward(self, input):
        """Squeeze singular dimensions of an input tensor.

        Arguments:
            input (torch.Tensor): tensor to unsqueeze.
        """
        if input.size(0) != 0:
            return input.squeeze()
        input = input.squeeze()
        return input.view(1, *input.size())


class Linear(nn.Module):

    """Wrapper around torch.nn.Linear to deal with single/multi-headed mode.

    Arguments:
        multi_head (bool): multi-headed mode.
        num_features_in (int): number of features in input.
        num_features_out (int): number of features in output.
        **kwargs: optional arguments to pass to torch.nn.Linear.
    """

    def __init__(self, multi_head, num_features_in,
                 num_features_out, **kwargs):
        super(Linear, self).__init__()
        self.num_features_in = num_features_in
        self.num_features_out = num_features_out

        self.multi_head = multi_head

        def _linear_factory():
            return nn.Linear(num_features_in, num_features_out, **kwargs)

        if self.multi_head:
            self.linear = nn.ModuleList([_linear_factory()] * NUM_CLASSES)
        else:
            self.linear = _linear_factory()

    def forward(self, x, idx=None):
        if self.multi_head:
            assert idx is not None, "Pass head idx in multi-headed mode."
            return self.linear[idx](x)
        return self.linear(x)

    def reset_parameters(self):
        """Reset parameters if in multi-headed mode."""
        if self.multi_head:
            for lin in self.linear:
                lin.reset_parameters()
        else:
            self.linear.reset_parameters()

###############################################################################


class OmniConv(nn.Module):

    """ConvNet classifier.

    Arguments:
        num_classes (int): number of classes to predict in each alphabet
        num_layers (int): number of convolutional layers (default=4).
        kernel_size (int): kernel size in each convolution (default=3).
        num_filters (int): number of output filters in each convolution
            (default=64)
        imsize (tuple): tuple of image height and width dimension.
        padding (bool, int, tuple): padding argument to convolution layers
            (default=True).
        batch_norm (bool): use batch normalization in each convolution layer
            (default=True).
        multi_head (bool): multi-headed training (default=False).
    """

    def __init__(self, num_classes, num_layers=4, kernel_size=3,
                 num_filters=64, imsize=(28, 28), padding=True,
                 batch_norm=True, multi_head=False):
        super(OmniConv, self).__init__()
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.imsize = imsize
        self.batch_norm = batch_norm
        self.multi_head = multi_head

        def conv_block(nin):
            block = [nn.Conv2d(nin, num_filters, kernel_size, padding=padding),
                     nn.MaxPool2d(2)]
            if batch_norm:
                block.append(nn.BatchNorm2d(num_filters))
            block.append(nn.ReLU())
            return block

        layers = [UnSqueeze()]
        for i in range(num_layers):
            layers.extend(conv_block(1 if i == 0 else num_filters))

        layers.append(Squeeze())

        self.conv = nn.Sequential(*layers)
        self.head = Linear(self.multi_head, num_filters, num_classes)

    def forward(self, input, idx=None):
        input = self.conv(input)
        return self.head(input, idx)

    def init_adaptation(self):
        """Reset stats for new task"""
        # Reset if multi-head, otherwise null-op
        self.head.reset_parameters()

        # Reset BN running stats
        for m in self.modules():
            if hasattr(m, 'reset_running_stats'):
                m.reset_running_stats()

###############################################################################


class WarpLayer(nn.Module):

    """Warp-layer module.

    Allows flexible configuration of convolutional warp-layers.

    Arguments:
        num_features_in (int): number of input filters.
        num_features_out (int): number of output filters.
        kernel_size (int): kernel size in each convolution (default=3).
        padding (bool, int, tuple): padding argument to convolution layer.
        residual_connection (bool): add residual connection.
        batch_norm (bool): use batch normalization in warp-layer.
        act_fun (fun): non-linearity in warp-layer (optional).
    """

    def __init__(self, num_features_in, num_features_out,
                 kernel_size, padding, residual_connection,
                 batch_norm, act_fun):
        super(WarpLayer, self).__init__()
        self.residual_connection = residual_connection
        self.bn_in = None
        self.bn_out = None
        if batch_norm:
            self.bn_in = nn.BatchNorm2d(num_features_in)
            if self.residual_connection:
                self.bn_out = nn.BatchNorm2d(num_features_out)

        self.conv = nn.Conv2d(num_features_in,
                              num_features_out,
                              kernel_size,
                              padding=padding)

        self.act_fun = act_fun if act_fun is None else act_fun()

        if residual_connection and num_features_in != num_features_out:
            self.scale = nn.Conv2d(num_features_in, num_features_out, 1)
        else:
            self.scale = None

    def forward(self, x):
        h = x

        if self.bn_in is not None:
            h = self.bn_in(h)

        h = self.conv(h)

        if self.act_fun is not None:
            h = self.act_fun(h)

        if self.residual_connection:
            if self.scale is not None:
                x = self.scale(x)

            h = x + h

        if self.bn_out is not None:
            h = self.bn_out(h)

        return h


class WarpedOmniConv(nn.Module):

    """ConvNet classifier.

    Same as the OmniConv except for additional warp-layers.

    Arguments:
        num_classes (int): number of classes to predict in each alphabet
        num_layers (int): number of convolutional layers (default=4).
        kernel_size (int): kernel size in each convolution (default=3).
        num_filters (int): number of output filters in each convolution
            (default=64)
        imsize (tuple): tuple of image height and width dimension.
        padding (bool, int, tuple): padding argument to convolution layers
            (default=True).
        batch_norm (bool): use batch normalization in each convolution layer
            (default=True).
        multi_head (bool): multi-headed training (default=False).
        warp_num_layers (int): number of warp-layers per adaptable conv block
            (default=1).
        warp_num_filters number of output filters internally in warp-layers,
            if `warp_num_lavers>1`. Final number of output filters of
            warp-layers are always same as number of input filters
            (default=64).
        warp_residual_connection (bool): use residual connection in
            warp-layers (default=False).
        warp_act_fun (str): activation function in warp-layers (optional).
        warp_batch_norm (bool): activation function in warp-layer.
        warp_final_head (bool): add a warp-layer to final output of model.
    """

    def __init__(self,
                 num_classes,
                 num_layers=4,
                 kernel_size=3,
                 num_filters=64,
                 imsize=(28, 28),
                 padding=True,
                 batch_norm=True,
                 multi_head=False,
                 warp_num_layers=1,
                 warp_num_filters=64,
                 warp_residual_connection=False,
                 warp_act_fun=None,
                 warp_batch_norm=True,
                 warp_final_head=False):
        super(WarpedOmniConv, self).__init__()
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.imsize = imsize
        self.batch_norm = batch_norm
        self.multi_head = multi_head
        self.warp_num_layers = warp_num_layers
        self.warp_num_filters = warp_num_filters
        self.warp_residual_connection = warp_residual_connection
        self.warp_act_fun = ACT_FUNS[warp_act_fun.lower()]
        self.warp_batch_norm = warp_batch_norm
        self.warp_final_head = warp_final_head
        self._conv_counter = 0
        self._warp_counter = 0

        def conv_block(nin):
            # Task adaptable conv block, same as OmniConv
            _block = [nn.Conv2d(nin,
                                num_filters,
                                kernel_size,
                                padding=padding),
                      nn.MaxPool2d(2)]
            if batch_norm:
                _block.append(nn.BatchNorm2d(num_filters))
            _block.append(nn.ReLU())
            return nn.Sequential(*_block)

        def warp_layer(nin, nout):
            # We use same kernel_size and padding as OmniConv for simplicity
            return WarpLayer(nin, nout, kernel_size, padding,
                             self.warp_residual_connection,
                             self.warp_batch_norm,
                             self.warp_act_fun)

        def block(nin):
            # Main block, wraps warp_layers around a conv_block.

            # Task-adaptable layer
            self._conv_counter += 1
            setattr(self, 'conv{}'.format(self._conv_counter), conv_block(nin))

            # Warp-layers
            nin = num_filters
            for _ in range(self.warp_num_layers):
                self._warp_counter = \
                    self._warp_counter % self.warp_num_layers + 1

                if self._warp_counter == self.warp_num_layers:
                    nout = num_filters
                else:
                    nout = self.warp_num_filters

                setattr(self, 'warp{}{}'.format(self._conv_counter,
                                                self._warp_counter),
                        warp_layer(nin, nout))

                nin = nout

        # Build model
        block(1)
        for _ in range(self.num_layers-1):
            block(num_filters)

        if self.warp_final_head:
            self.head = Linear(self.multi_head, num_filters, num_filters)
            self.warp_head = nn.Linear(num_filters, num_classes)
        else:
            self.head = Linear(self.multi_head, num_filters, num_classes)

        self.squeeze = Squeeze()

    def forward(self, x, idx=None):
        """Forward-pass through model."""
        for i in range(1, self._conv_counter+1):
            # Task-adaptable layer
            x = getattr(self, 'conv{}'.format(i))(x)
            # Warp-layer(s)
            for j in range(1, self._warp_counter+1):
                x = getattr(self, 'warp{}{}'.format(i, j))(x)

        x = self.squeeze(x)
        x = self.head(x, idx)

        if self.warp_final_head:
            return self.warp_head(x)
        return x

    def adapt_modules(self):
        """Iterator for task-adaptable modules"""
        for i in range(1, self.num_layers+1):
            conv = getattr(self, 'conv{}'.format(i))
            yield conv
        yield self.head

    def warp_modules(self):
        """Iterator for warp-layer modules"""
        for i in range(1, self.num_layers+1):
            for j in range(1, self.warp_num_layers+1):
                warp = getattr(self, 'warp{}{}'.format(i, j))
                yield warp

        if self.warp_final_head:
            yield self.warp_head

    def init_adaptation(self):
        """Reset stats for new task"""
        # Reset head if multi-headed, otherwise null-op
        self.head.reset_parameters()

        # Reset BN running stats
        for m in self.modules():
            if hasattr(m, 'reset_running_stats'):
                m.reset_running_stats()
