# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

import tensorflow as tf

from tensorflow.contrib.mpi_collectives.ops import gen_mpi_ops
from tensorflow.contrib.util import loader
from tensorflow.python.framework import ops
from tensorflow.python.platform import resource_loader

_mpi_ops_so = loader.load_op_library(
    resource_loader.get_path_to_datafile("_mpi_ops.so"))

def size(name=None):
  """An op which returns the number of MPI processes.

  This is equivalent to running `MPI_Comm_size(MPI_COMM_WORLD, ...)` to get the
  size of the global communicator.

  Returns:
    An integer scalar containing the number of MPI processes.
  """
  return gen_mpi_ops.mpi_size(name=name)


ops.NotDifferentiable('MPISize')


def rank(name=None):
  """An op which returns the MPI rank of the calling process.

  This is equivalent to running `MPI_Comm_rank(MPI_COMM_WORLD, ...)` to get the
  rank of the current process in the global communicator.

  Returns:
    An integer scalar with the MPI rank of the calling process.
  """
  return gen_mpi_ops.mpi_rank(name=name)


ops.NotDifferentiable('MPIRank')


def init(name=None):
  """An op which initializes MPI on the device on which it is run.

  All future MPI ops must be run on the same device that the `init` op was run
  on.
  """
  return gen_mpi_ops.mpi_init(name=name)


ops.NotDifferentiable('MPIInit')


def local_rank(name=None):
  """An op which returns the local MPI rank of the calling process, within the
  node that it is running on. For example, if there are seven processes running
  on a node, their local ranks will be zero through six, inclusive.

  This is equivalent to running `MPI_Comm_rank(...)` on a new communicator
  which only includes processes on the same node.

  Returns:
    An integer scalar with the local MPI rank of the calling process.
  """
  return gen_mpi_ops.mpi_local_rank(name=name)


ops.NotDifferentiable('MPILocalRank')


def _allreduce(tensor, name=None):
  """An op which sums an input tensor over all the MPI processes.

  The reduction operation is keyed by the name of the op. The tensor type and
  shape must be the same on all MPI processes for a given name. The reduction
  will not start until all processes are ready to send and receive the tensor.

  Returns:
    A tensor of the same shape and type as `tensor`, summed across all
    processes.
  """
  return gen_mpi_ops.mpi_allreduce(tensor, name=name)


ops.NotDifferentiable('MPIAllreduce')


def allgather(tensor, name=None):
  """An op which concatenates the input tensor with the same input tensor on
  all other MPI processes.

  The concatenation is done on the first dimension, so the input tensors on the
  different processes must have the same rank and shape, except for the first
  dimension, which is allowed to be different.

  Returns:
    A tensor of the same type as `tensor`, concatenated on dimension zero
    across all processes. The shape is identical to the input shape, except for
    the first dimension, which may be greater and is the sum of all first
    dimensions of the tensors in different MPI processes.
  """
  # Specify that first allgather is to collect the tensor gather sizes,
  # indicated by passing in a scalar (0-D tensor) of value 0
  sizes_flag = tf.constant(0, dtype=tf.int64, name="size_flag_const")
  my_size = tf.slice(tf.shape(tensor, out_type=tf.int64), [0], [1], name="size_slice")
  if name is None:
    name = "allgather"
  sizing_name = "{}_sizing".format(name)
  sizes = gen_mpi_ops.mpi_allgather(my_size, sizes_flag, name=sizing_name)
  return gen_mpi_ops.mpi_allgather(tensor, sizes, name=name)


ops.NotDifferentiable('MPIAllgather')


