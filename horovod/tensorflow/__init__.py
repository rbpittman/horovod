# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications copyright (C) 2017 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# pylint: disable=g-short-docstring-punctuation
"""## Communicating Between Processes with MPI

TensorFlow natively provides inter-device communication through send and
receive ops and inter-node communication through Distributed TensorFlow, based
on the same send and receive abstractions. On HPC clusters where Infiniband or
other high-speed node interconnects are available, these can end up being
insufficient for synchronous data-parallel training (without asynchronous
gradient descent). This module implements a variety of MPI ops which can take
advantage of hardware-specific MPI libraries for efficient communication.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from horovod.tensorflow.mpi_ops import size
from horovod.tensorflow.mpi_ops import local_size
from horovod.tensorflow.mpi_ops import rank
from horovod.tensorflow.mpi_ops import global_rank
from horovod.tensorflow.mpi_ops import global_size
from horovod.tensorflow.mpi_ops import local_rank
from horovod.tensorflow.mpi_ops import allgather
from horovod.tensorflow.mpi_ops import gather
from horovod.tensorflow.mpi_ops import broadcast
from horovod.tensorflow.mpi_ops import _allreduce
from horovod.tensorflow.mpi_ops import init


def allreduce(tensor, group, average=True, device_dense='', device_sparse=''):
    """Perform an allreduce on a tf.Tensor or tf.IndexedSlices.

    Arguments:
        tensor: tf.Tensor, tf.Variable, or tf.IndexedSlices to reduce.
        The shape of the input must be identical across all ranks.
        average: If True, computes the average over all ranks.
                 Otherwise, computes the sum over all ranks.
        device_dense: Device to be used for dense tensors. Uses GPU by default
                      if Horovod was build with HOROVOD_GPU_ALLREDUCE.
        device_sparse: Device to be used for sparse tensors. Uses GPU by default
                       if Horovod was build with HOROVOD_GPU_ALLGATHER.

    This function performs a bandwidth-optimal ring allreduce on the input
    tensor. If the input is an tf.IndexedSlices, the function instead does an
    allgather on the values and the indices, effectively doing an allreduce on
    the represented tensor.
    """
    if isinstance(tensor, tf.IndexedSlices):
        with tf.device(device_sparse):
            # For IndexedSlices, do two allgathers intead of an allreduce.
            horovod_size = tf.cast(size(), tensor.values.dtype)
            values = allgather(tensor.values, group)
            indices = allgather(tensor.indices, group)

            # To make this operation into an average, divide all gathered values by
            # the Horovod size.
            new_values = tf.div(values, horovod_size) if average else values
        return tf.IndexedSlices(new_values, indices,
                                dense_shape=tensor.dense_shape)
    else:
        with tf.device(device_dense):
            horovod_size = tf.cast(size(), tensor.dtype)
            summed_tensor = _allreduce(tensor)
            new_tensor = (tf.div(summed_tensor, horovod_size)
                          if average else summed_tensor)
        return new_tensor


def broadcast_global_variables(root_rank, group):
    """Broadcasts all global variables from root rank to all other processes.

    Arguments:
        root_rank: rank of the process from which global variables will be broadcasted
        to all other processes.
    """
    return tf.group(*[tf.assign(var, broadcast(var, root_rank, group))
                      for var in tf.global_variables()])


class BroadcastGlobalVariablesHook(tf.train.SessionRunHook):
    """
    SessionRunHook that will broadcast all global variables from root rank
    to all other processes during initialization.

    This is necessary to ensure consistent initialization of all workers when
    training is started with random weights or restored from a checkpoint.
    """

    def __init__(self, root_rank, group, device=''):
        """Construct a new BroadcastGlobalVariablesHook that will broadcast all
        global variables from root rank to all other processes during initialization.

        Args:
          root_rank:
            Rank that will send data, other ranks will receive data.
          device:
            Device to be used for broadcasting. Uses GPU by default
            if Horovod was build with HOROVOD_GPU_BROADCAST.
        """
        super(BroadcastGlobalVariablesHook, self).__init__()
        self.root_rank = root_rank
        self.bcast_op = None
        self.device = device
        self.group = group
        
    def begin(self):
        if not self.bcast_op:
            with tf.device(self.device):
                self.bcast_op = broadcast_global_variables(self.root_rank, self.group)

    def after_create_session(self, session, coord):
        session.run(self.bcast_op)


class DistributedOptimizer(tf.train.Optimizer):
    """An optimizer that wraps another tf.Optimizer, using an allreduce to
    average gradient values before applying gradients to model weights."""

    def __init__(self, optimizer, group, name=None, use_locking=False, device_dense='',
                 device_sparse=''):
        """Construct a new DistributedOptimizer, which uses another optimizer
        under the hood for computing single-process gradient values and
        applying gradient updates after the gradient values have been averaged
        across all the Horovod ranks.

        Args:
          optimizer:
            Optimizer to use for computing gradients and applying updates.
          name:
            Optional name prefix for the operations created when applying
            gradients. Defaults to "Distributed" followed by the provided
            optimizer type.
          use_locking:
            Whether to use locking when updating variables.
            See Optimizer.__init__ for more info.
          device_dense:
            Device to be used for dense tensors. Uses GPU by default
            if Horovod was build with HOROVOD_GPU_ALLREDUCE.
          device_sparse:
            Device to be used for sparse tensors. Uses GPU by default
            if Horovod was build with HOROVOD_GPU_ALLGATHER.
        """
        if name is None:
            name = "Distributed{}".format(type(optimizer).__name__)

        self._optimizer = optimizer
        self._device_dense = device_dense
        self._device_sparse = device_sparse
        self.group = group
        super(DistributedOptimizer, self).__init__(
            name=name, use_locking=use_locking)

    def compute_gradients(self, *args, **kwargs):
        """Compute gradients of all trainable variables.

        See Optimizer.compute_gradients() for more info.

        In DistributedOptimizer, compute_gradients() is overriden to also
        allreduce the gradients before returning them.
        """
        gradients = (super(DistributedOptimizer, self)
                     .compute_gradients(*args, **kwargs))
        if size() > 1:
            averaged_gradients = []
            with tf.name_scope(self._name + "_Allreduce"):
                for grad, var in gradients:
                    if grad is not None:
                        avg_grad = allreduce(grad, device_dense=self._device_dense,
                                             device_sparse=self._device_sparse)
                        averaged_gradients.append((avg_grad, var))
                    else:
                        averaged_gradients.append((None, var))
            return averaged_gradients
        else:
            return gradients

    def _apply_dense(self, *args, **kwargs):
        """Calls this same method on the underlying optimizer."""
        return self._optimizer._apply_dense(*args, **kwargs)

    def _resource_apply_dense(self, *args, **kwargs):
        """Calls this same method on the underlying optimizer."""
        return self._optimizer._resource_apply_dense(*args, **kwargs)

    def _resource_apply_sparse_duplicate_indices(self, *args, **kwargs):
        """Calls this same method on the underlying optimizer."""
        return self._optimizer._resource_apply_sparse_duplicate_indices(*args, **kwargs)

    def _resource_apply_sparse(self, *args, **kwargs):
        """Calls this same method on the underlying optimizer."""
        return self._optimizer._resource_apply_sparse(*args, **kwargs)

    def _apply_sparse_duplicate_indices(self, *args, **kwargs):
        """Calls this same method on the underlying optimizer."""
        return self._optimizer._apply_sparse_duplicate_indices(*args, **kwargs)

    def _apply_sparse(self, *args, **kwargs):
        """Calls this same method on the underlying optimizer."""
        return self._optimizer._apply_sparse(*args, **kwargs)

    def _prepare(self, *args, **kwargs):
        """Calls this same method on the underlying optimizer."""
        return self._optimizer._prepare(*args, **kwargs)

    def _create_slots(self, *args, **kwargs):
        """Calls this same method on the underlying optimizer."""
        return self._optimizer._create_slots(*args, **kwargs)

    def _valid_dtypes(self, *args, **kwargs):
        """Calls this same method on the underlying optimizer."""
        return self._optimizer._valid_dtypes(*args, **kwargs)

    def _finish(self, *args, **kwargs):
        """Calls this same method on the underlying optimizer."""
        return self._optimizer._finish(*args, **kwargs)
