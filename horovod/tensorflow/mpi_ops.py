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
# =============================================================================
"""Inter-process communication using MPI."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ctypes
import re
import sysconfig
from tensorflow.python.framework import load_library
from tensorflow.python.framework import ops
from tensorflow.python.platform import resource_loader


def _get_ext_suffix():
    """Determine library extension for various versions of Python."""
    ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')
    if ext_suffix:
        return ext_suffix

    ext_suffix = sysconfig.get_config_var('SO')
    if ext_suffix:
        return ext_suffix

    return '.so'


def _load_library(name, op_list=None):
    """Loads a .so file containing the specified operators.

    Args:
      name: The name of the .so file to load.
      op_list: A list of names of operators that the library should have. If None
          then the .so file's contents will not be verified.

    Raises:
      NameError if one of the required ops is missing.
      NotFoundError if were not able to load .so file.
    """
    filename = resource_loader.get_path_to_datafile(name)
    library = load_library.load_op_library(filename)
    for expected_op in (op_list or []):
        for lib_op in library.OP_LIST.op:
            if lib_op.name == expected_op:
                break
        else:
            raise NameError(
                'Could not find operator %s in dynamic library %s' %
                (expected_op, name))
    return library


def _load_ctypes_dll(name):
    filename = resource_loader.get_path_to_datafile(name)
    return ctypes.CDLL(filename, mode=ctypes.RTLD_GLOBAL)


MPI_LIB = _load_library('mpi_lib' + _get_ext_suffix(),
                        ['HorovodAllgather', 'HorovodAllreduce'])


MPI_LIB_CTYPES = _load_ctypes_dll('mpi_lib' + _get_ext_suffix())


def init(group=-1, group_ranks=None):
    """A function which initializes Horovod.
    Defaults to using 1 group. If group is specified, then it is used as the group index,
    and group_ranks should all be specified. 
    
    WARNING: Very limited error checking on multiple init specifications. Don't
             provide incorrect input...
    """
    if group != -1 and group_ranks == None:
        #Invalid parameters
        raise ValueError("Invalid parameters sent to init, must specify group_ranks")
    if group == -1:
        num_group_ranks = 0
    else:
        num_group_ranks = len(group_ranks)
    array_type = ctypes.c_int * num_group_ranks
    return MPI_LIB_CTYPES.horovod_tensorflow_init(ctypes.c_int(group),
                                                  ctypes.c_int(num_group_ranks),
                                                  array_type(*group_ranks))

def size(group=-1):
    """A function which returns the number of Horovod processes.

    Returns:
      An integer scalar containing the number of Horovod processes.
    """
    size = MPI_LIB_CTYPES.horovod_tensorflow_size(ctypes.c_int(group))
    if size == -1:
        raise ValueError(
            'Horovod has not been initialized; use horovod.tensorflow.init().')
    return size

def global_size():
    """A function which returns the number of Horovod processes.

    Returns:
      An integer scalar containing the number of Horovod processes.
    """
    size = MPI_LIB_CTYPES.horovod_tensorflow_global_size()
    if size == -1:
        raise ValueError(
            'Horovod has not been initialized; use horovod.tensorflow.init().')
    return size


def local_size():
    """A function which returns the number of Horovod processes within the
    node the current process is running on.

    Returns:
      An integer scalar containing the number of local Horovod processes.
    """
    local_size = MPI_LIB_CTYPES.horovod_tensorflow_local_size()
    if local_size == -1:
        raise ValueError(
            'Horovod has not been initialized; use horovod.tensorflow.init().')
    return local_size


def rank(group=-1):
    """A function which returns the Horovod rank of the calling process.

    Returns:
      An integer scalar with the Horovod rank of the calling process.
    """
    rank = MPI_LIB_CTYPES.horovod_tensorflow_rank(ctypes.c_int(group))
    if rank == -1:
        raise ValueError(
            'Horovod has not been initialized; use horovod.tensorflow.init().')
    return rank

def global_rank():
    """A function which returns the Horovod rank of the calling process.

    Returns:
      An integer scalar with the Horovod rank of the calling process.
    """
    rank = MPI_LIB_CTYPES.horovod_tensorflow_global_rank()
    if rank == -1:
        raise ValueError(
            'Horovod has not been initialized; use horovod.tensorflow.init().')
    return rank


def local_rank():
    """A function which returns the local Horovod rank of the calling process, within the
    node that it is running on. For example, if there are seven processes running
    on a node, their local ranks will be zero through six, inclusive.

    Returns:
      An integer scalar with the local Horovod rank of the calling process.
    """
    local_rank = MPI_LIB_CTYPES.horovod_tensorflow_local_rank()
    if local_rank == -1:
        raise ValueError(
            'Horovod has not been initialized; use horovod.tensorflow.init().')
    return local_rank


def _normalize_name(name):
    """Normalizes operation name to TensorFlow rules."""
    return re.sub('[^a-zA-Z0-9_]', '_', name)


def _allreduce(tensor, name=None, group=-1):
    """An op which sums an input tensor over all the Horovod processes.

    The reduction operation is keyed by the name of the op. The tensor type and
    shape must be the same on all Horovod processes for a given name. The reduction
    will not start until all processes are ready to send and receive the tensor.

    Returns:
      A tensor of the same shape and type as `tensor`, summed across all
      processes.
    """
    if name is None:
        name = 'HorovodAllreduce_%s' % _normalize_name(tensor.name)
    return MPI_LIB.horovod_allreduce(tensor, name=name, group=group)


ops.NotDifferentiable('HorovodAllreduce')


def allgather(tensor, name=None, group=-1):
    """An op which concatenates the input tensor with the same input tensor on
    all other Horovod processes.

    The concatenation is done on the first dimension, so the input tensors on the
    different processes must have the same rank and shape, except for the first
    dimension, which is allowed to be different.

    Returns:
      A tensor of the same type as `tensor`, concatenated on dimension zero
      across all processes. The shape is identical to the input shape, except for
      the first dimension, which may be greater and is the sum of all first
      dimensions of the tensors in different Horovod processes.
    """
    if name is None:
        name = 'HorovodAllgather_%s' % _normalize_name(tensor.name)
    return MPI_LIB.horovod_allgather(tensor, name=name, group=group)


ops.NotDifferentiable('HorovodAllgather')


def broadcast(tensor, root_rank, name=None, group=-1):
    """An op which broadcasts the input tensor on root rank to the same input tensor
    on all other Horovod processes.

    The broadcast operation is keyed by the name of the op. The tensor type and
    shape must be the same on all Horovod processes for a given name. The broadcast
    will not start until all processes are ready to send and receive the tensor.

    Returns:
      A tensor of the same shape and type as `tensor`, with the value broadcasted
      from root rank.
    """
    if name is None:
        name = 'HorovodBroadcast_%s' % _normalize_name(tensor.name)
    return MPI_LIB.horovod_broadcast(tensor, name=name, root_rank=root_rank, group=group)


ops.NotDifferentiable('HorovodBroadcast')

def gather(tensor, root_rank, name=None, group=-1):
    """An op which concatenates the input tensor with the same input tensor on
    all other Horovod processes, and sends the result to root_rank only. 
    
    The concatenation is done on the first dimension, so the input tensors on the
    different processes must have the same rank and shape, except for the first
    dimension, which is allowed to be different.
    
    Returns:
      A tensor of the same type as `tensor`, concatenated on dimension zero
      across all processes, but only for root_rank. The shape is
      identical to the input shape, except for 
      the first dimension, which may be greater and is the sum of all first
      dimensions of the tensors in different Horovod processes.
    """
    if name is None:
        name = 'HorovodGather_%s' % _normalize_name(tensor.name)
    return MPI_LIB.horovod_gather(tensor, name=name, root_rank=root_rank, group=group)


ops.NotDifferentiable('HorovodGather')
