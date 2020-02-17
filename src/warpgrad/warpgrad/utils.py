"""Utilities for WarpGrad.

:author: Sebastian Flennerhag
"""
from collections import OrderedDict

import os
import torch

from ._dict import load_state_dict


def grad(x, y, model, params, criterion):
    """Compute new parameters with computation graph intact.

     Arguments:
        x (torch.Tensor): input tensor.
        y (torch.Tensor): target tensor.
        model (Warp): warped model.
        params (list): list of parameters to differentiate
            the loss with respect to.
        criterion (fun): task loss criterion.
    """
    device = next(model.parameters()).device

    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)

    p = model(x, cache_parameters=False)
    loss = criterion(p, y)
    return torch.autograd.grad(loss, params, create_graph=True)


def approx_step(x_outer, y_outer, model, criterion, scorer):
    """Compute approximate meta gradient (no inner step).

    Arguments:
        x_outer (torch.Tensor): input to obtain meta learner gradient.
        y_outer (torch.Tensor): target to obtain meta learner gradient.
        model (Warp): warped model.
        criterion (fun): task loss criterion.
        scorer (fun): scoring function (optional).
    """
    device = next(model.parameters()).device

    x_outer = x_outer.to(device, non_blocking=True)
    y_outer = y_outer.to(device, non_blocking=True)
    pred_outer = model(x_outer, cache_parameters=False)
    loss_outer = criterion(pred_outer, y_outer)

    score_outer = None
    if scorer is not None:
        score_outer = scorer(pred_outer.detach(), y_outer.detach())

    return loss_outer, score_outer


def step(x_inner, y_inner, x_outer, y_outer,
         model, optimizer, criterion, scorer):
    """Compute gradients of meta-parameters using the WarpGrad objective.

    Arguments:
        x_inner (torch.Tensor): input to obtain task learner gradient.
        y_inner (torch.Tensor): target to obtain task learner gradient.
        x_outer (torch.Tensor): input to obtain meta learner gradient.
        y_outer (torch.Tensor): target to obtain meta learner gradient.
        model (Warp): warped model.
        optimizer (warpgrad.optim.SGD, warpgrad.optim.Adam): task optimizer,
            must be differentiable.
        criterion (fun): task loss criterion.
        scorer (fun): scoring function (optional).
    """
    device = next(model.parameters()).device
    original_tparams = OrderedDict(model.named_adapt_parameters())

    x_inner = x_inner.to(device, non_blocking=True)
    y_inner = y_inner.to(device, non_blocking=True)
    x_outer = x_outer.to(device, non_blocking=True)
    y_outer = y_outer.to(device, non_blocking=True)

    # Parameter update
    pred_inner = model(x_inner, cache_parameters=False)
    loss_inner = criterion(pred_inner, y_inner)

    backward(loss_inner, model.adapt_parameters(), create_graph=True)
    _, new_params = optimizer.step(retain_graph=True)

    replace(model.model, new_params, original_tparams)

    # Get forward loss
    pred_outer = model(x_outer, cache_parameters=False)
    loss_outer = criterion(pred_outer, y_outer)

    # Reset model parameters
    load_state_dict(model.model, original_tparams)

    score_inner = None
    score_outer = None
    if scorer is not None:
        score_inner = scorer(pred_inner.detach(), y_inner.detach())
        score_outer = scorer(pred_outer.detach(), y_outer.detach())

    return loss_outer, (loss_inner.detach(), score_inner, score_outer)


def replace(model, new_params, old_params):
    """Helper for updating model dict in a back-prop compatible way."""
    par_names = list(old_params.keys())
    assert len(par_names) == len(new_params)

    new_state = OrderedDict(zip(par_names, new_params))
    load_state_dict(model, new_state)
    for p in old_params.values():
        p.grad = None  # drop current gradients to avoid accumulating into them


def backward(loss, args, create_graph=False):
    """Partial derivatives of loss wrt args."""
    args = list(args)
    grads = torch.autograd.grad(loss, args, create_graph=create_graph)
    for p, g in zip(args, grads):
        p.backward(g)

#############################################################################


def get_data(iterator, n_iterations):
    """Helper for setting up data."""
    out = []
    iterator.dataset.train()
    for i, batch in enumerate(iterator):
        out.append(batch)
        if i+1 == n_iterations:
            break
    return out


def freeze(iterable):
    """Freeze params in module."""
    for p in iterable:
        p.requires_grad = False


def unfreeze(iterable):
    """Freeze params in module."""
    for p in iterable:
        p.requires_grad = True


def get_groups(parameters, opt_params, tensor=True):
    """Return layer-wise optimization hyper-parameters."""
    groups = []
    for p, lr, mom in zip(parameters, opt_params.lr, opt_params.momentum):
        if not tensor:
            lr = lr.item()
            mom = mom.item()
        groups.append({'params': p, 'lr': lr, 'momentum': mom})
    return groups

#############################################################################


def stem(fpath):
    """Returns task-id and iter-id."""
    fname = str(os.path.basename(fpath).split('.')[0])
    return fname.split('_')


def load(path):
    """Load stored data in to task dict."""
    files = sorted(os.listdir(path))  # sorting -> iter order
    mapped_files = {stem(f)[0]: [] for f in files}

    for fname in files:
        fpath = os.path.join(path, fname)

        n, i = stem(fname)
        assert len(mapped_files) == int(i)

        mapped_files[n].append(torch.load(fpath))

    return mapped_files


def clear(path):
    """delete all files in path."""
    files = os.listdir(path)
    for f in files:
        os.unlink(os.path.join(path, f))


def state_dict_to_par_list(state_dict, par_names):
    """Prune a state_dict and return a list of parameters."""
    return [tensor for name, tensor in state_dict.items() if
            name in par_names]


def clone(tensor, device=None):
    """Clone a list of tensors."""
    if not isinstance(tensor, torch.Tensor):
        return [clone(t) for t in tensor]

    cloned = tensor.detach().clone()
    cloned.requires_grad = tensor.requires_grad
    if device is not None:
        cloned = cloned.to(device)
    return cloned


def clone_state(state_dict, *args, **kwargs):
    """Clone a list of tensors."""
    cloned_state = OrderedDict()
    for n, p in state_dict.items():
        cloned_state[n] = clone(p, *args, **kwargs)
    return cloned_state


def copy_opt(param_states):
    """Copy buffers from an optimizer state."""
    cloned_states = []
    for param_state in param_states:
        cloned_state = OrderedDict()
        for k, v in param_state.items():
            cloned_state[k] = v.clone().cpu()
        cloned_states.append(cloned_state)
    return cloned_states


def copy(to_tensors, from_tensors):
    """Copy tensor data from one set of iterables to another."""
    if isinstance(to_tensors, (list, tuple)):
        for p, q in zip(to_tensors, from_tensors):
            p.data.copy_(q.data)

    elif isinstance(to_tensors, (dict, OrderedDict)):
        for (n, p), (m, q) in zip(to_tensors.items(), from_tensors.items()):
            if n != m:
                raise ValueError(
                    'target state variable {}' 
                    'does not match source state variable{}'.format(n, m))
            p.data.copy_(q.data)

    else:
        raise ValueError('Unknown iterables type {}'.format(type(to_tensors)))


def zero_grad(tensor_like):
    """Set tensor gradient to zero. Null op if argument is not a tensor.

    Argument:
        tensor_like: objects to zero grad for.
            If list, will iterate over elements.
    """
    if isinstance(tensor_like, (tuple, list)):
        for p in tensor_like:
            zero_grad(p)

    if not hasattr(tensor_like, 'grad'):
        return

    if tensor_like.grad is None:
        if tensor_like.dim() == 0:
            tensor_like.grad = tensor_like.detach().clone()
        else:
            tensor_like.grad = tensor_like.new(*tensor_like.shape)
    tensor_like.grad.zero_()

#############################################################################


def n_correct(logits, targets):
    """Number correct predictions.

    Args:
        logits (torch.Tensor): tensor of prediction logits.
        targets (torch.Tensor): tensor of class targets.
    """
    _, predictions = logits.max(1)
    correct = (predictions == targets).sum().item()
    return correct


def acc_fn(p, y):
    """Accuracy of discrete predictions."""
    return n_correct(p, y) / y.size(0)


def global_norm(tensors, detach=True, eps=1e-9):
    """Compute a global norm over a list of tensors."""
    norm = 0.
    for tensor in tensors:
        tensor = tensor.view(-1)
        if detach:
            tensor = tensor.detach().data
        norm += torch.dot(tensor, tensor)  # pylint: disable=no-member
    norm = norm.sqrt()
    return norm + eps
