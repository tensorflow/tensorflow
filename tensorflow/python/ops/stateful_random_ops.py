# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Operations for generating random numbers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_stateful_random_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.training.tracking import \
tracking
from tensorflow.python.util.tf_export import tf_export

# A seed for random ops (stateful and stateless) will always be 1024
# bits, all of which will be sent to the C++ code. The actual C++
# implementation of some algorithms may only use a lower part of the bits.

MAX_INT64 = 2**63 - 1
MIN_INT64 = -(2**63)
UINT64_SPAN = 2**64
# 'Variable' doesn't support uint32 or uint64 yet (due to reasons explained in
# b/111604096 and cl/171681867), so I use signed int here. I choose int64
# instead of int32 here because `VarHandleOp` doesn't support int32 on GPU.
SEED_TYPE = "int64"
SEED_MIN = MIN_INT64
SEED_MAX = MAX_INT64
SEED_UINT_SPAN = UINT64_SPAN
SEED_TYPE_BITS = 64
SEED_BIT_MASK = 0xFFFFFFFFFFFFFFFF
SEED_SIZE = 16  # in units of SEED_TYPE


STATE_TYPE = SEED_TYPE
ALGORITHM_TYPE = STATE_TYPE
RNG_ALG_PHILOX = 1
RNG_ALG_THREEFRY = 2
DEFAULT_ALGORITHM = RNG_ALG_PHILOX


PHILOX_STATE_SIZE = 3
THREEFRY_STATE_SIZE = 2


def non_deterministic_seed():
  """Makes a non-deterministic seed.

  The implementation will be changed soon from pure Python to an op.

  Returns:
    a 1-D tensor.
  """
  return np.random.randint(
      low=SEED_MIN, high=SEED_MAX + 1, size=SEED_SIZE,
      dtype=SEED_TYPE)


def _uint_to_int(n):
  if n > SEED_MAX:
    n = n - SEED_UINT_SPAN
  return n


def _make_1d_state(state_size, seed):
  """Makes a 1-D RNG state.

  Args:
    state_size: an integer.
    seed: an integer or 1-D tensor.

  Returns:
    a 1-D tensor of shape [state_size] and dtype STATE_TYPE.
  """
  int_types = (int,) if sys.version_info >= (3, 0) else (int, long)
  if isinstance(seed, int_types):
    # chop the Python integer (infinite precision) into chunks of SEED_TYPE
    ls = []
    for _ in range(state_size):
      ls.append(seed & SEED_BIT_MASK)
      seed >>= SEED_TYPE_BITS
    seed = ls
  # to avoid overflow error from np.asarray
  seed = list(map(_uint_to_int, seed))
  seed = np.asarray(seed, dtype=STATE_TYPE)
  if len(seed.shape) != 1:
    raise ValueError(
        "seed should only have one dimension; got shape: %s" % seed.shape)
  seed = seed[0:state_size]
  # Padding with zeros on the right if too short
  seed_size = seed.shape[0]
  if seed_size < state_size:
    seed = np.pad(
        seed, [(0, state_size - seed_size)],
        mode="constant",
        constant_values=0)
  assert seed.shape == (state_size,), "Wrong seed.shape: %s" % seed.shape
  return seed


def _make_philox_state(seed):
  """Makes a RNG state for Philox algorithm.

  Args:
    seed: an integer or 1-D tensor.

  Returns:
    a 1-D tensor.
  """
  return _make_1d_state(PHILOX_STATE_SIZE, seed)


def _make_threefry_state(seed):
  """Makes a RNG state for ThreeFry algorithm.

  Args:
    seed: an integer or 1-D tensor.

  Returns:
    a 1-D tensor.
  """
  return _make_1d_state(THREEFRY_STATE_SIZE, seed)


def _make_state_from_seed(seed, algorithm):
  if algorithm == RNG_ALG_PHILOX:
    return _make_philox_state(seed)
  elif algorithm == RNG_ALG_THREEFRY:
    return _make_threefry_state(seed)
  else:
    raise ValueError("Unsupported algorithm id: %s" % algorithm)


@tf_export("random.create_rng_state")
def create_rng_state(seed, algorithm):
  """Creates a RNG state.

  Args:
    seed: an integer or 1-D tensor.
    algorithm: an integer representing the RNG algorithm.

  Returns:
    a 1-D tensor whose size depends on the algorithm.
  """
  return _make_state_from_seed(seed, algorithm)


def _shape_tensor(shape):
  """Convert to an int32 or int64 tensor, defaulting to int64 if empty."""
  if isinstance(shape, (tuple, list)) and not shape:
    dtype = dtypes.int64
  else:
    dtype = None
  return ops.convert_to_tensor(shape, dtype=dtype, name="shape")


@tf_export("random.experimental.Generator")
class Generator(tracking.AutoTrackable):
  """Random-number generator.

  It uses Variable to manage its internal state.
  """

  def __init__(self, copy_from=None, seed=None, algorithm=None):
    """Creates a generator.

    Args:
      copy_from: (optional) a generator to be copied from.
      seed: (optional) the seed for the RNG. If None, it will be chosen
            nondeterministically
      algorithm: (optional) the RNG algorithm. If None, it will be
                 auto-selected.
    """
    if copy_from is None:
      if seed is None:
        seed = non_deterministic_seed()
      if algorithm is None:
        # TODO(wangpeng): more sophisticated algorithm selection
        algorithm = DEFAULT_ALGORITHM
      state = create_rng_state(seed, algorithm)
      self._state_var = variables.Variable(state, dtype=STATE_TYPE)
      self._alg_var = algorithm
    else:
      assert seed is None
      self._state_var = variables.Variable(copy_from.state, dtype=STATE_TYPE)
      self._alg_var = copy_from.algorithm

  def reset(self, seed):
    """Resets the generator.

    Args:
      seed: the seed to reset the RNG to.
    """
    state = create_rng_state(seed, self.algorithm)
    self._state_var.assign(state)

  @property
  def state(self):
    return self._state_var

  @property
  def algorithm(self):
    return self._alg_var

  # The following functions return a tensor and as a side effect update
  # self._state_var.
  def normal(self, shape, mean=0.0, stddev=1.0, dtype=dtypes.float32,
             name=None):
    with ops.name_scope(name, "stateful_normal", [shape, mean, stddev]) as name:
      shape = _shape_tensor(shape)
      mean = ops.convert_to_tensor(mean, dtype=dtype, name="mean")
      stddev = ops.convert_to_tensor(stddev, dtype=dtype, name="stddev")
      rnd = gen_stateful_random_ops.stateful_standard_normal_v2(
          self.state.handle, self.algorithm, shape, dtype=dtype)
      return math_ops.add(rnd * stddev, mean, name=name)

  def uniform(self, shape, minval=0, maxval=None,
              dtype=dtypes.float32, name=None):
    dtype = dtypes.as_dtype(dtype)
    if maxval is None:
      if dtype.is_integer:
        raise ValueError("Must specify maxval for integer dtype %r" % dtype)
      maxval = 1
    with ops.name_scope(name, "stateful_uniform",
                        [shape, minval, maxval]) as name:
      shape = _shape_tensor(shape)
      minval = ops.convert_to_tensor(minval, dtype=dtype, name="min")
      maxval = ops.convert_to_tensor(maxval, dtype=dtype, name="max")
      if dtype.is_integer:
        return gen_stateful_random_ops.stateful_uniform_int(
            self.state.handle, self.algorithm, shape=shape,
            minval=minval, maxval=maxval, name=name)
      else:
        # TODO(wangpeng): implement uniform for floats
        raise ValueError("uniform for floats not implemented yet")

  def uniform_full_int(self, shape, dtype=dtypes.uint64, name=None):
    """Uniform distribution on an integer type's entire range.

    The other method `uniform` only covers the range [minval, maxval), which
    cannot be `dtype`'s full range because `maxval` is of type `dtype`.

    Args:
      shape: the shape of the output.
      dtype: (optional) the integer type, default to uint64.
      name: (optional) the name of the node.

    Returns:
      A tensor of random numbers of the required shape.
    """
    dtype = dtypes.as_dtype(dtype)
    with ops.name_scope(name, "stateful_uniform_full_int",
                        [shape]) as name:
      shape = _shape_tensor(shape)
      return gen_stateful_random_ops.stateful_uniform_full_int(
          self.state.handle, self.algorithm, shape=shape,
          dtype=dtype, name=name)

  # TODO(wangpeng): implement other distributions (
  #   `truncated_normal`, etc.)
  # TODO(wangpeng): implement `make_seeds`
  # TODO(wangpeng): implement `make_generators`


# It's not safe to create TF ops before `init_google` is called, so this is
# initialized to None and get a value the first time `get_global_generator` is
# called.
global_generator = None


@tf_export("random.experimental.get_global_generator")
def get_global_generator():
  global global_generator
  if global_generator is None:
    global_generator = Generator()
  return global_generator


@tf_export("random.experimental.set_global_generator")
def set_global_generator(generator):
  global global_generator
  global_generator = generator


# This function creates a new Generator object (and the Variable object within),
# which does not work well with tf.function because (1) tf.function puts
# restrictions on Variable creation thus reset_global_generator can't be freely
# used inside tf.function; (2) redirecting a global variable to
# a new object is problematic with tf.function because the old object may be
# captured by a 'tf.function'ed function and still be used by it.
# A 'tf.function'ed function only keeps weak references to variables,
# so deleting a variable and then calling that function again may raise an
# error, as demonstrated by
# random_test.py/RandomTest.testResetGlobalGeneratorBadWithDefun .
# The function 'set_global_generator' below also has this problem.
@tf_export("random.experimental.reset_global_generator")
def reset_global_generator(seed, algorithm=None):
  global global_generator
  if algorithm is None:
    # preserve the old algorithm
    algorithm = int(get_global_generator().algorithm)
  global_generator = Generator(seed=seed, algorithm=algorithm)
