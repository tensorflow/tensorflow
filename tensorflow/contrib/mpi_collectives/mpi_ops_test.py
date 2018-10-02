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

"""Tests for tensorflow.contrib.mpi_collectives.mpi_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import itertools

import tensorflow as tf

import tensorflow.contrib.mpi_collectives as mpi


def mpi_env_rank_and_size():
  """Get MPI rank and size from environment variables and return them as a
  tuple of integers.

  Most MPI implementations have an `mpirun` or `mpiexec` command that will
  run an MPI executable and set up all communication necessary between the
  different processors. As part of that set up, they will set environment
  variables that contain the rank and size of the MPI_COMM_WORLD
  communicator. We can read those environment variables from Python in order
  to ensure that `mpi.rank()` and `mpi.size()` return the expected values.

  Since MPI is just a standard, not an implementation, implementations
  typically choose their own environment variable names. This function tries
  to support several different implementation, but really it only needs to
  support whatever implementation we want to use for the TensorFlow test
  suite.

  If this is not running under MPI, then defaults of rank zero and size one
  are returned. (This is appropriate because when you call MPI_Init in an
  application not started with mpirun, it will create a new independent
  communicator with only one process in it.)
  """
  rank_env = "PMI_RANK OMPI_COMM_WORLD_RANK".split()
  size_env = "PMI_SIZE OMPI_COMM_WORLD_SIZE".split()

  for rank_var, size_var in zip(rank_env, size_env):
    rank = os.environ.get(rank_var)
    size = os.environ.get(size_var)
    if rank is not None and size is not None:
      return int(rank), int(size)

  # Default to rank zero and size one if there are no environment variables
  return 0, 1


class MPITests(tf.test.TestCase):
  """
  Tests for MPI ops in tensorflow.contrib.mpi_collectives.
  """

  def test_mpi_rank(self):
    """Test that the rank returned by mpi.rank() is correct."""
    true_rank, _ = mpi_env_rank_and_size()
    with self.test_session() as session:
      rank = session.run(mpi.rank())
      self.assertEqual(true_rank, rank)

  def test_mpi_size(self):
    """Test that the size returned by mpi.size() is correct."""
    _, true_size = mpi_env_rank_and_size()
    with self.test_session() as session:
      size = session.run(mpi.size())
      self.assertEqual(true_size, size)

  def test_mpi_allreduce_cpu(self):
    """Test on CPU that the allreduce correctly sums 1D, 2D, 3D tensors."""
    with self.test_session() as session:
      size = session.run(mpi.size())

      dtypes = [tf.int32, tf.float32]
      dims = [1, 2, 3]
      for dtype, dim in itertools.product(dtypes, dims):
        tf.set_random_seed(1234)
        tensor = tf.random_uniform([17] * dim, -100, 100, dtype=dtype)
        summed = mpi.allreduce(tensor, average=False)
        multiplied = tensor * size
        max_difference = tf.reduce_max(tf.abs(summed - multiplied))

        # Threshold for floating point equality depends on number of
        # ranks, since we're comparing against precise multiplication.
        if size <= 3:
          threshold = 0
        elif size < 10:
          threshold = 1e-4
        elif size < 15:
          threshold = 5e-4
        else:
          break

        diff = session.run(max_difference)
        self.assertTrue(diff <= threshold,
                        "mpi.allreduce produces incorrect results")

  def test_mpi_allreduce_gpu(self):
    """Test that the allreduce works on GPUs.

    This test will crash badly if used with an MPI implementation that does
    not support GPU memory transfers directly, as it will call MPI_Send on
    a GPU data pointer."""
    # Only do this test if there are GPUs available.
    if not tf.test.is_gpu_available(cuda_only=True):
      return

    no_gpus = tf.GPUOptions(visible_device_list="")
    cpu_config = tf.ConfigProto(gpu_options=no_gpus)
    with self.test_session(config=cpu_config) as session:
      local_rank = session.run(mpi.local_rank())

    one_gpu = tf.GPUOptions(visible_device_list=str(local_rank))
    gpu_config = tf.ConfigProto(gpu_options=one_gpu)
    with self.test_session(config=gpu_config) as session:
      size = session.run(mpi.size())

      dtype = tf.float32
      dim = 3
      with tf.device("/gpu:0"):
        tf.set_random_seed(1234)
        tensor = tf.random_uniform([17] * dim, -100, 100, dtype=dtype)
        summed = mpi.allreduce(tensor, average=False)
        multiplied = tensor * size
        max_difference = tf.reduce_max(tf.abs(summed - multiplied))

      # Threshold for floating point equality depends on number of
      # ranks, since we're comparing against precise multiplication.
      if size <= 3:
        threshold = 0
      elif size < 10:
        threshold = 1e-4
      elif size < 15:
        threshold = 5e-4
      else:
        return

      diff = session.run(max_difference)
      self.assertTrue(diff <= threshold,
                      "mpi.allreduce on GPU produces incorrect results")

  def test_mpi_allreduce_error(self):
    """Test that the allreduce raises an error if different ranks try to
    send tensors of different rank or dimension."""
    with self.test_session() as session:
      rank = session.run(mpi.rank())
      size = session.run(mpi.size())

      # This test does not apply if there is only one worker.
      if size == 1:
        return

      # Same rank, different dimension
      tf.set_random_seed(1234)
      dims = [17 + rank] * 3
      tensor = tf.random_uniform(dims, -1.0, 1.0)
      with self.assertRaises(tf.errors.FailedPreconditionError):
        session.run(mpi.allreduce(tensor))

      # Same number of elements, different rank
      tf.set_random_seed(1234)
      if rank == 0:
        dims = [17, 23 * 57]
      else:
        dims = [17, 23, 57]
      tensor = tf.random_uniform(dims, -1.0, 1.0)
      with self.assertRaises(tf.errors.FailedPreconditionError):
        session.run(mpi.allreduce(tensor))

  def test_mpi_allreduce_type_error(self):
    """Test that the allreduce raises an error if different ranks try to
    send tensors of different type."""
    with self.test_session() as session:
      rank = session.run(mpi.rank())
      size = session.run(mpi.size())

      # This test does not apply if there is only one worker.
      if size == 1:
        return

      # Same rank, different dimension
      dims = [17] * 3
      tensor = tf.ones(dims, dtype=tf.int32 if rank % 2 == 0 else tf.float32)
      with self.assertRaises(tf.errors.FailedPreconditionError):
        session.run(mpi.allreduce(tensor))

  def test_mpi_allgather(self):
    """Test that the allgather correctly gathers 1D, 2D, 3D tensors."""
    with self.test_session() as session:
      size = session.run(mpi.size())
      rank = session.run(mpi.rank())

      dtypes = tf.int32, tf.float32
      dims = 1, 2, 3
      for dtype, dim in itertools.product(dtypes, dims):
        tensor = tf.ones([17] * dim, dtype=dtype) * rank
        gathered = mpi.allgather(tensor)

        gathered_tensor = session.run(gathered)
        self.assertEqual(list(gathered_tensor.shape),
                         [17 * size] + [17] * (dim - 1))

        for i in range(size):
          rank_tensor = tf.slice(gathered_tensor, [i * 17] + [0] * (dim - 1),
                                 [17] + [-1] * (dim - 1))
          self.assertEqual(list(rank_tensor.shape), [17] * dim)
          self.assertTrue(session.run(tf.reduce_all(tf.equal(rank_tensor, i))),
                          "mpi.allgather produces incorrect gathered tensor")

  def test_mpi_allgather_variable_size(self):
    """Test that the allgather correctly gathers 1D, 2D, 3D tensors,
    even if those tensors have different sizes along the first dim."""
    with self.test_session() as session:
      size = session.run(mpi.size())
      rank = session.run(mpi.rank())

      dtypes = tf.int32, tf.float32
      dims = 1, 2, 3
      for dtype, dim in itertools.product(dtypes, dims):
        # Support tests up to MPI Size of 35
        if size > 35:
          break

        tensor_sizes = [17, 32, 81, 12, 15, 23, 22] * 5
        tensor_sizes = tensor_sizes[:size]

        tensor = tf.ones([tensor_sizes[rank]] + [17] * (dim - 1),
                         dtype=dtype) * rank
        gathered = mpi.allgather(tensor)

        gathered_tensor = session.run(gathered)
        expected_size = sum(tensor_sizes)
        self.assertEqual(list(gathered_tensor.shape),
                         [expected_size] + [17] * (dim - 1))

        for i in range(size):
          rank_size = [tensor_sizes[i]] + [17] * (dim - 1)
          rank_tensor = tf.slice(gathered,
                                 [sum(tensor_sizes[:i])] + [0] * (dim - 1),
                                 rank_size)
          self.assertEqual(list(rank_tensor.shape), rank_size)
          self.assertTrue(session.run(tf.reduce_all(tf.equal(rank_tensor, i))),
                          "mpi.allgather produces incorrect gathered tensor")

  def test_mpi_allgather_error(self):
    """Test that the allgather returns an error if any dimension besides
    the first is different among the tensors being gathered."""
    with self.test_session() as session:
      rank = session.run(mpi.rank())
      size = session.run(mpi.size())

      # This test does not apply if there is only one worker.
      if size == 1:
        return

      tensor_size = [17] * 3
      tensor_size[1] = 10 * (rank + 1)
      tensor = tf.ones(tensor_size, dtype=tf.float32) * rank
      with self.assertRaises(tf.errors.FailedPreconditionError):
        session.run(mpi.allgather(tensor))

  def test_mpi_allgather_type_error(self):
    """Test that the allgather returns an error if the types being gathered
    differ among the processes"""
    with self.test_session() as session:
      rank = session.run(mpi.rank())
      size = session.run(mpi.size())

      # This test does not apply if there is only one worker.
      if size == 1:
        return

      tensor_size = [17] * 3
      dtype = tf.int32 if rank % 2 == 0 else tf.float32
      tensor = tf.ones(tensor_size, dtype=dtype) * rank
      with self.assertRaises(tf.errors.FailedPreconditionError):
        session.run(mpi.allgather(tensor))


if __name__ == '__main__':
  tf.test.main()
