"""Meta-gradient compatible load_state_dict method"""
# pylint: disable=protected-access
# pylint: disable=redefined-builtin
from collections import OrderedDict


def _load_from_par_dict(module, par_dict, prefix):
    """Replace the module's _parameter dict with par_dict."""
    _new_parameters = OrderedDict()
    for name, param in module._parameters.items():
        key = prefix + name
        if key in par_dict:
            input_param = par_dict[key]
        else:
            input_param = param

        if input_param.shape != param.shape:
            # local shape should match the one in checkpoint
            raise ValueError(
                'size mismatch for {}: copying a param of {} from checkpoint, '
                'where the shape is {} in current model.'.format(
                    key, param.shape, input_param.shape))

        _new_parameters[name] = input_param
    module._parameters = _new_parameters


def load_state_dict(module, state_dict):
    r"""Replaces parameters and buffers.

    Replaces parameters and buffers from :attr:`state_dict` into
    the given module and its descendants. In contrast to the module's
    method, this function will *not* do in-place copy of underlying data on
    *parameters*, but instead replace the ``_parameter`` dict in each
    module and its descendants. This allows us to backpropr through previous
    gradient steps using the standard top-level API.

    .. note::
        You must store the original state dict (with keep_vars=True) separately
        and, when ready to update them, use :funct:`load_state_dict` to return
        as the module's parameters.

    Arguments:
        module (torch.nn.Module): a module instance whose state to update.
        state_dict (dict): a dict containing parameters and
            persistent buffers.
    """
    par_names = [n for n, _ in module.named_parameters()]

    par_dict = OrderedDict({k: v for k, v in state_dict.items()
                            if k in par_names})
    no_par_dict = OrderedDict({k: v for k, v in state_dict.items()
                               if k not in par_names})
    excess = [k for k in state_dict.keys()
              if k not in list(no_par_dict.keys()) + list(par_dict.keys())]

    if excess:
        raise ValueError(
            "State variables %r not in the module's state dict %r" % (
                excess, par_names))

    metadata = getattr(state_dict, '_metadata', None)
    if metadata is not None:
        par_dict._metadata = metadata
        no_par_dict._metadata = metadata

    module.load_state_dict(no_par_dict, strict=False)

    def load(module, prefix=''): # pylint: disable=missing-docstring
        _load_from_par_dict(module, par_dict, prefix)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(module)
