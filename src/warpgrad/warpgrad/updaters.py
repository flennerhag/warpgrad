"""Meta learner objectives for WarpGrad.

Updaters are classes that manage how meta-updates should be performed. The
main `DualUpdater` allows for dual updates to warp parameters and the
initialization.

:author: Sebastian Flennerhag.
"""
# pylint: disable=too-many-arguments
# pylint: disable=invalid-name
# pylint: disable=redefined-builtin
# pylint: disable=protected-access
# pylint: disable=too-many-locals
# pylint: disable=arguments-differ
import numpy as np
import joblib
import torch

from .optim import SGD
from .utils import (step, approx_step, unfreeze, freeze,
                    get_data, acc_fn, global_norm, backward,
                    state_dict_to_par_list)


class DualUpdater:

    """Implements the WarpGrad meta-objective.

    This updater applies the WarpGrad meta-objective to warp-parameters and
    if specified, to the initialization using the given meta-loss for the
    initialization.
    """

    def __init__(self, criterion, warp_objective=1, init_objective=1,
                 epochs=1, bsz=1, norm=True, approx=False):
        """Initialize an updater.

        Arguments:
            criterion (function): task loss criterion.
            warp_objective (int): type of WarpGrad objective.
            init_objective (int): type of objective for initialization
                (optional).
            epochs (int): number of times to iterate over buffer (default=1).
            bsz (int): task parameter batch size between updates (default=1).
            norm (bool): use the norm in the Leap objective (d1)
                (default=True).
            approx (bool): use approximate (Hessian-free) meta-objective.
        """
        self.warp_objective = warp_objective
        self.init_objective = init_objective
        self.criterion = criterion
        self.epochs = epochs
        self.approx = approx
        self.norm = norm
        self.bsz = bsz

    def backward(self, model, step_fn, **opt_kwargs):
        """Compute meta gradients wrt dec and code

        Arguments:
            model (Warp): warped model to backprop through.
            step_fn (function): step function for the meta gradient.
            **opt_kwargs (kwargs): optional arguments to inner optimizer.
        """
        out = model.buffer.dataset
        if len(out) == 2:
            optimizer_buffers = None
            data, params = out
        else:
            data, params, optimizer_buffers = out

        warp_objective = WARP_OBJECTIVES[self.warp_objective]
        warp_objective(model, self.criterion, params, optimizer_buffers, data,
                       step_fn, opt_kwargs, self.epochs, self.bsz, self.approx)

        init_objective= INIT_OBJECTIVES[self.init_objective]
        init_objective(model.named_init_parameters(suffix=None),
                       params, self.norm, self.bsz, step_fn)

def warp_on_same_loss(model, criterion, trj, brj, tds, step_fn,
                      opt_kwargs, epochs, bsz, approx):
    """WarpGrad uses same objective in first and second step."""
    unfreeze(model.meta_parameters(include_init=False))
    unfreeze(model.adapt_parameters())
    model.train()

    def _get(t, i):
        state = trj[t][i]
        buffer = brj[t][i] if brj else None
        # pylint: disable=unbalanced-tuple-unpacking
        (x, y), (x2, y2) = get_data(tds[t], 2)
        return x, y, state, buffer, x2, y2

    def _step(batch):
        loss = 0
        for (x, y, state, buffer, x2, y2) in batch:
            model.set_state(state)

            opt = SGD(model.optimizer_parameter_groups(tensor=True),
                      **opt_kwargs)
            opt.zero_grad()
            if buffer:
                for p, b in zip(
                        model.optimizer_parameter_groups(tensor=True), buffer):
                    opt.state[p] = b

            if approx:
                l1 = a1 = None
                l2, a2 = approx_step(x, y, model, criterion, acc_fn)
            else:
                l2, (l1, a1, a2) = step(x, y, x2, y2, model,
                                        opt, criterion, acc_fn)
            del l1, a1, a2  # unused for now.

            loss = loss + l2
        loss = loss / bsz

        backward(loss, model.meta_parameters(include_init=False))
        step_fn()

    for _ in range(epochs):
        datapoints = [_get(t, i) for t in trj for i in range(len(trj[t]))]
        np.random.shuffle(datapoints)

        if bsz > 0:
            for i in range(0, len(datapoints), bsz):
                _step(datapoints[i:i+bsz])
        else:
            _step(datapoints)

    freeze(model.meta_parameters(include_init=False))


def simplified_leap(named_init, trj, norm, bsz, step_fn):
    """One step of Leap over trajectories, wrt shared init.
        Similar to Leap objective except the loss delta is omitted.
    """
    del bsz  # unused

    # TODO: allow choice of cpu or gpu
    par_names, init = zip(*named_init)
    device = init[0].device
    unfreeze(init)

    with joblib.Parallel(n_jobs=-1, backend='threading') as parallel:
        adds = parallel(
            joblib.delayed(line_seg_len)(
                trj[t][i], trj[t][i + 1], par_names, norm, device)
            for t in trj
            for i in range(0, len(trj[t])-1)
        )

    for i, a in zip(init, zip(*adds)):
        a = torch.stack(a, dim=0)  # pylint: disable=no-member
        i.grad = a.data.sum(dim=0)
        i.grad.div_(len(trj))
    step_fn()
    freeze(init)


def line_seg_len(entry_state, exit_state, par_names, norm, device):
    """Compute partial grad for line segment"""
    entry_params = state_dict_to_par_list(entry_state, par_names)
    exit_params = state_dict_to_par_list(exit_state, par_names)
    add = [e.data.to(device) - x.data.to(device)
           for e, x in zip(entry_params, exit_params)]
    if norm:
        norm = global_norm(add, detach=True, eps=1e-9)
        for l in add:
            l.data.div_(norm)
    return add


def null_func(*args, **kwargs):
    """Vacuous call"""
    del args, kwargs  # unused.
    return


WARP_OBJECTIVES = {
    0: null_func,
    1: warp_on_same_loss,
}

INIT_OBJECTIVES = {
    0: null_func,
    1: simplified_leap,
}
