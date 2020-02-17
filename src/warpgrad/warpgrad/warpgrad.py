"""Warped Gradient Descent.

Model wrapper that implements the WarpGrad logic
on a generic PyTorch model.

:author: Sebastian Flennerhag
"""
# pylint: disable=too-many-arguments
# pylint: disable=invalid-name
# pylint: disable=redefined-builtin
# pylint: disable=too-many-instance-attributes
# pylint: disable=protected-access
# pylint: disable=too-many-locals
# pylint: disable=arguments-differ
from collections import OrderedDict

import os
import uuid
import tempfile
import torch

from .utils import (copy, copy_opt, clone_state, load, clear,
                    unfreeze, freeze, zero_grad, get_groups)


class ReplayBuffer:

    """Cache for parameters during meta-training."""

    def __init__(self, inmem=True, tmpdir=None):
        """Initialize replay buffer.

        Arguments:
            inmem (bool): in-memory buffer (on CPU, default=True).
            tmpdir (str): if not inmem, root of buffer (optional).

        """
        self.inmem = inmem
        self._data_buffer = {}
        self._state_buffer = {}
        self._optimizer_buffer = {}
        self._idx = {}
        if not inmem and tmpdir is None:
            tmpdir = tempfile.mkdtemp('_WGDTMP')
        self.tmpdir = tmpdir

    def clear(self):
        """Clear buffer."""
        self._data_buffer.clear()
        self._idx.clear()
        if self.inmem:
            self._state_buffer.clear()
            self._optimizer_buffer.clear()
        else:
            clear(self.tmpdir)

    def init(self, slot, data):
        """Initialize slot in buffer and attach dataloader."""
        if slot in self._idx:
            raise ValueError('slot {} already in buffer'.format(slot))
        self._idx[slot] = 0
        self._data_buffer[slot] = data

    def update(self, slot, state, buffer=None):
        """Persists a copy of current parameters under given slot.

        Arguments:
            slot (str): a group-level identifier for the parameters
              (i.e. task id).
            state(OrderedDict, torch.Tensor): state_dict to add to buffer.
            buffer (list, dict, None): list of optimiser parameter buffers
              (default=None).
        """
        assert slot in self._idx, 'slot not in buffer. Call init_slot first'
        self._idx[slot] += 1

        if self.inmem:
            if slot not in self._state_buffer:
                assert self._idx[slot] == 1
                self._state_buffer[slot] = []
                if buffer is not None:
                    self._optimizer_buffer[slot] = []

            self._state_buffer[slot].append(clone_state(state, device='cpu'))
            if buffer is not None:
                self._optimizer_buffer[slot].append(copy_opt(buffer))
            return

        if buffer is not None:
            raise NotImplementedError(
                "Putting optimizer parameters on disk not implemented.")

        fname = '{}_{}.{}'.format(slot, self._idx[slot], '.tar')
        fpath = os.path.join(self.tmpdir, fname)
        torch.save(state, fpath)

    @property
    def dataset(self):
        """Current replay buffer."""
        if self.inmem:
            if self._optimizer_buffer:
                return (self._data_buffer, self._state_buffer,
                        self._optimizer_buffer)
            return self._data_buffer, self._state_buffer

        param_cache = load(self.tmpdir)
        return self._data_buffer, param_cache


class OptimizerParameters:

    """Container for Optimizer Parameters."""

    def __init__(self, trainable, default_lr, default_momentum):
        """Initialize optimizer parameter manager.

        Arguments:
            trainable (bool): whether optimizer parameters are trainable.
            default_lr (float): initial learning rate (prior to training).
            default_momentum (float): initial momentum (prior to training).
        """
        self._opt = None
        self._trainable = trainable
        self._param_names = []

        self._lr = []
        self._momentum = []
        self.default_lr = default_lr
        self.default_momentum = default_momentum

    def init(self, named_parameters):
        """Initialize opt parameter groups for parameters."""
        # pylint: disable=not-callable
        self._lr = []
        self._momentum = []
        self._param_names = []

        for n, p in named_parameters:
            self._param_names.append(n)
            pl = torch.tensor(self.default_lr,
                              device=p.device,
                              requires_grad=self.trainable)
            pm = torch.tensor(self.default_momentum,
                              device=p.device,
                              requires_grad=self.trainable)
            self._lr.append(pl)
            self._momentum.append(pm)

    @property
    def lr(self):
        """Learning rates."""
        for l in self._lr:
            if l.item() < 0:
                l.data.fill_(1e-6)
            yield l

    @property
    def momentum(self):
        """Momentum rates."""
        for m in self._momentum:
            if m.item() < 0:
                m.data.fill_(1e-6)
            yield m

    @property
    def trainable(self):
        """Trainable parameters flag."""
        return self._trainable

    @trainable.setter
    def trainable(self, trainable):
        self._trainable = trainable

        if self._trainable and self.default_momentum == 0:
            self.default_momentum = 1e-4
            for m in self._momentum:
                m.data.fill_(self.default_momentum)

        for p in self.parameters():
            p.requires_grad = self._trainable

    def parameters(self):
        """Optimizer parameters."""
        for p in self._lr + self._momentum:
            yield p

    def named_parameters(self):
        """Optimizer parameters."""
        for n, p in zip(self._param_names, self._lr):
            n += '.lr'
            yield n, p

        for n, p in zip(self._param_names, self._momentum):
            n += '.mom'
            yield n, p

    def groups(self, parameters, tensor):
        """Parameter groups."""
        return get_groups(parameters, self, tensor=tensor)


class Parameters:

    """Attributes for parameter partitioning."""

    def __init__(self, model, adapt_modules, warp_modules,
                 optimizer_parameters):
        """Initialize partitioning.

        Arguments:
            model (torch.nn.Module): main model.
            adapt_modules (list, tuple): adaptable modules.
            warp_modules (list, tuple): warp modules.
            optimizer_parameters (OptimizerParameters): optimizer parameters
                manager.
        """
        self.model = model
        self.adapt_modules = adapt_modules
        self.warp_modules = warp_modules

        self._optimizer = None
        self._learn_optimizer = optimizer_parameters.trainable
        self._optimizer_parameters = optimizer_parameters
        self._optimizer_parameters.init(self.named_adapt_parameters())

        self._init_state = clone_state(self.adapt_state())
        self._init_parameters = [(n, p) for n, p in self._init_state.items()
                                 if p.requires_grad]

    def set_parameters(self, new_parameters):
        """Set task parameters to new_params.

        Arguments:
            new_parameters (list, torch.Tensor): list of task parameters.
        """
        copy(self.adapt_parameters(), new_parameters)

    def set_state(self, new_state):
        """Set task parameters to new_params.

        Arguments:
            new_state (OrderedDict): state dictionary over task parameters
              and buffers.
        """
        copy(self.adapt_state(), new_state)

    def init_state(self):
        return self._init_state

    def adapt_state(self):
        """Return state_dict for adapt modules."""
        model_state = self.model.state_dict(keep_vars=True)
        adapt_tensors = [id(t) for m in self.adapt_modules
                         for t in m.state_dict(keep_vars=True).values()]
        return OrderedDict((n, t) for n, t in model_state.items()
                           if id(t) in adapt_tensors)

    def adapt_parameters(self):
        """Adapt parameters."""
        for m in self.adapt_modules:
            for p in m.parameters():
                yield p

    def named_adapt_parameters(self):
        """Named adapt parameters."""
        # We can't use adapt_modules.named_parameters()
        # need to go through main model to get correct names
        adapt_ids = list(map(id, self.adapt_parameters()))
        for n, p in self.model.named_parameters():
            if id(p) in adapt_ids:
                yield n, p

    def parameters(self):
        """All parameters."""
        return self.model.parameters()

    def optimizer_buffer(self):
        """Return stored optimizer buffer, if any."""
        buffer = None
        if self._optimizer is not None:
            # opt.state is not ordered in pytorch v1
            buffer = []
            param_names = [n for n, _ in self.adapt_state()]
            for n in param_names:
                # check since opt.state is a dict factory
                if n in self._optimizer.state:
                    buffer.append(self._optimizer.state[n])
        return buffer

    def optimizer_parameters(self):
        """Optimizer parameters."""
        return self._optimizer_parameters.parameters()

    def named_optimizer_parameters(self):
        """Named optimizer parameters."""
        return self._optimizer_parameters.named_parameters()

    def init_parameters(self):
        """Initialization parameters."""
        for _, p in self.named_init_parameters():
            yield p

    def named_init_parameters(self, suffix='.init'):
        """Named initialization parameters."""
        for n, p in self._init_parameters:
            if suffix is not None:
                n += suffix
            yield n, p

    def warp_parameters(self):
        """Warp parameters."""
        for m in self.warp_modules:
            for p in m.parameters():
                yield p

    def named_warp_parameters(self, suffix=None):
        """Named warp parameters."""
        # We can't use warp_modules.named_parameters()
        # need to go through main model to get correct names
        meta_param_ids = list(map(id, self.warp_parameters()))
        for n, p in self.model.named_parameters():
            if id(p) in meta_param_ids:
                if suffix is not None:
                    n += suffix
                yield n, p

    def meta_parameters(self,
                        include_warp=True,
                        include_init=True,
                        include_optimizer=True):
        """All meta-parameters.

        Arguments:
            include_warp (bool): include warp parameters.
            include_init (bool): include the initialization.
            include_optimizer (bool): include optimizer parameters.
        """
        if self.learn_optimizer and include_optimizer:
            for p in self.optimizer_parameters():
                yield p

        if include_init:
            for p in self.init_parameters():
                yield p

        if include_warp:
            for p in self.warp_parameters():
                yield p

    def named_meta_parameters(self,
                              include_warp=True,
                              include_init=True,
                              include_opt=True):
        """Named meta parameters.

        Arguments:
            include_warp (bool): include warp parameters.
            include_init (bool): include the initialization.
            include_opt (bool): include optimizer parameters
                (if `learn_optimizer=True`).
        """
        if self.learn_optimizer and include_opt:
            for n, p in self.named_optimizer_parameters():
                yield n, p

        if include_init:
            for n, p in self.named_init_parameters():
                yield n, p

        if include_warp:
            for n, p in self.named_warp_parameters():
                yield n, p

    def optimizer_parameter_groups(self, tensor=False):
        """Parameter groups for optimizer.

        Arguments:
            tensor (bool): return parameters as tensors
              (use with warpgrad.optim).
        """
        return self._optimizer_parameters.groups(
            self.adapt_parameters(), tensor)

    def register_optimizer(self, optimizer):
        """Register an optimizer during task training to collect buffers."""
        self._optimizer = optimizer

    def unregister_optimizer(self):
        """Unregister optimizer to stop collecting buffers."""
        self._optimizer = None

    @property
    def learn_optimizer(self):
        """Learn optimizer parameters."""
        return self._optimizer_parameters.trainable

    @learn_optimizer.setter
    def learn_optimizer(self, learn_optimizer):
        self._optimizer_parameters.trainable = learn_optimizer


class Warp(Parameters):

    """Model wrapper for WarpGrad."""

    def __init__(self, model, adapt_modules, warp_modules,
                 updater, buffer, optimizer_parameters):
        """Initialize warp over given model.

        Args:
            model (torch.nn.Module): main model.
            adapt_modules (torch.nn.Module): adapt modules in main model.
            warp_modules (torch.nn.Module): warp modules in main model.
            updater (updater.DualUpdater): the meta parameter update handler.
            buffer (ReplayBuffer): adapt parameter replay buffer.
            optimizer_parameters (OptimizerParameters): handler of optimizer
              parameters.
        """
        super(Warp, self).__init__(model,
                                   adapt_modules,
                                   warp_modules,
                                   optimizer_parameters)
        self.updater = updater
        self._task = None
        self._collect = True
        self.buffer = buffer
        self.zero_meta_grads()
        self.zero_task_grads()

    def __call__(self, *inputs, cache_parameters=None):
        if cache_parameters is None:
            cache_parameters = self._collect
        if cache_parameters:
            self._dump()
        return self.model(*inputs)

    def register_task(self, data):
        """Register a distinct task in buffer.

        Args:
            data: the tasks data generator.
        """
        self._task = uuid.uuid4().hex
        self.buffer.init(self._task, data)

    def init_adaptation(self, reset_adapt_parameters=None):
        """Calls init_adaptation on model.

        Arguments:
            reset_adapt_parameters (bool): whether to reset the initialisation
            of adaptable parameters. If not specified, will be reset if
            the initialization is meta-learned (in the `updater`).
        """
        self.model.init_adaptation()

        if reset_adapt_parameters is None:
            # Will be 0 if no meta-objective for initialization is specified
            reset_adapt_parameters = self.updater.init_objective

        if reset_adapt_parameters:
            copy(self.adapt_state(), self.init_state())

        freeze(self.meta_parameters())
        unfreeze(self.adapt_parameters())

        self.model.train()

    def clear(self):
        """Clears parameter trajectory buffer."""
        self.buffer.clear()

    def collect(self):
        """Switch on task parameter collection in buffer."""
        self._collect = True

    def no_collect(self):
        """Switch off task parameter collection in buffer."""
        self._collect = False

    def train(self):
        """Switch to train mode in task learner."""
        self.model.train()

    def eval(self):
        """Switch to eval mode in task learner."""
        self.model.eval()

    def zero_meta_grads(self):
        """Set meta gradients to zero."""
        zero_grad(list(self.meta_parameters()))

    def zero_task_grads(self):
        """Set task learner gradients to zero."""
        zero_grad(list(self.adapt_parameters()))

    def backward(self, *args, retain_trajectories=False,
                 retain_optimizer=False, **kwargs):
        """Compute gradients of meta-parameters.

        Arguments:
            *args: arguments to pass to the updater.
            retain_trajectories (bool): keep current buffer (default=False).
            retain_optimizer (bool): keep registered optimizer (default=False).
            **kwargs: keyword arguments to pass to the updater.
        """
        collecting = self.collecting
        if collecting:
            self.no_collect()

        self.updater.backward(self, *args, **kwargs)

        if not retain_trajectories:
            self.clear()

        if not retain_optimizer:
            self.unregister_optimizer()

        if collecting:
            self.collect()

    def _dump(self):
        """Persists a copy of current parameters under [task, iter]."""
        self.buffer.update(self._task,
                           self.adapt_state(),
                           self.optimizer_buffer())

    @property
    def collecting(self):
        """Flag for whether we are collect parameters in buffer."""
        return self._collect

    @property
    def dataset(self):
        """Return copy of codes."""
        return self.buffer.dataset
