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
from tensorflow.python.training.checkpointable import \
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
RNG_ALG_PHILOX = 1
DEFAULT_ALGORITHM = RNG_ALG_PHILOX


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


PHILOX_STATE_SIZE = 3


def _make_philox_state(seed):
  """Makes a RNG state for Philox algorithm.

  Args:
    seed: an integer or 1-D tensor.

  Returns:
    a 1-D tensor.
  """
  int_types = (int,) if sys.version_info >= (3, 0) else (int, long)
  if isinstance(seed, int_types):
    # chop the Python integer (infinite precision) into chunks of SEED_TYPE
    ls = []
    for _ in range(PHILOX_STATE_SIZE):
      ls.append(seed & SEED_BIT_MASK)
      seed >>= SEED_TYPE_BITS
    seed = ls
  # to avoid overflow error from np.asarray
  seed = list(map(_uint_to_int, seed))
  seed = np.asarray(seed, dtype=STATE_TYPE)
  if len(seed.shape) != 1:
    raise ValueError(
        "seed should only have one dimension; got shape: %s" % seed.shape)
  seed = seed[0:PHILOX_STATE_SIZE]
  # Padding with zeros on the right if too short
  seed_size = seed.shape[0]
  if seed_size < PHILOX_STATE_SIZE:
    seed = np.pad(
        seed, [(0, PHILOX_STATE_SIZE - seed_size)],
        mode="constant",
        constant_values=0)
  assert seed.shape == (PHILOX_STATE_SIZE,), "Wrong seed.shape: %s" % seed.shape
  return seed


def _make_state_from_seed(seed, algorithm):
  if algorithm == RNG_ALG_PHILOX:
    return _make_philox_state(seed)
  else:
    raise ValueError("Unsupported algorithm id: %s" % algorithm)


def create_rng_state(seed, algorithm=None):
  """Creates a RNG state.

  Args:
    seed: an integer or 1-D tensor.
    algorithm: (optional) an integer representing the RNG algorithm. If None, an
      algorithm will be auto-selected.

  Returns:
    a 1-D tensor "rng_state" with:
    * rng_state[0] is a value that identifies the RNG algorithm;
    * rng_state[1:] holds the RNG state itself (size dependent on the
        algorithm).
  """
  if algorithm is None:
    # TODO(wangpeng): more sophisticated algorithm selection
    algorithm = DEFAULT_ALGORITHM
  state = _make_state_from_seed(seed, algorithm)
  return np.concatenate((np.array([algorithm], dtype=STATE_TYPE), state),
                        axis=None)


def _shape_tensor(shape):
  """Convert to an int32 or int64 tensor, defaulting to int64 if empty."""
  if isinstance(shape, (tuple, list)) and not shape:
    dtype = dtypes.int64
  else:
    dtype = None
  return ops.convert_to_tensor(shape, dtype=dtype, name="shape")


@tf_export("random.experimental.Generator")
class Generator(tracking.Checkpointable):
  """Random-number generator.

  It uses Variable to manage its internal state.
  """

  def __init__(self, copy_from=None, seed=None, algorithm=None):
    if copy_from is None:
      if seed is None:
        seed = non_deterministic_seed()
      state = create_rng_state(seed, algorithm)
      self._state_var = variables.Variable(state, dtype=STATE_TYPE)
    else:
      assert seed is None
      state = copy_from.state
      self._state_var = variables.Variable(state, dtype=STATE_TYPE)

  def reset(self, seed):
    algorithm = int(self.algorithm)
    state = create_rng_state(seed, algorithm)
    self._state_var.assign(state)

  @property
  def state(self):
    return self._state_var

  @property
  def algorithm(self):
    return self._state_var[0]

  # The following functions return a tensor and as a side effect update
  # self._state_var.
  def standard_normal(self, shape, dtype=dtypes.float32):
    return gen_stateful_random_ops.stateful_standard_normal(
        self.state.handle, shape, dtype)

  def normal(self, shape, mean=0.0, stddev=1.0, dtype=dtypes.float32,
             name=None):
    with ops.name_scope(name, "stateful_normal", [shape, mean, stddev]) as name:
      shape = _shape_tensor(shape)
      mean = ops.convert_to_tensor(mean, dtype=dtype, name="mean")
      stddev = ops.convert_to_tensor(stddev, dtype=dtype, name="stddev")
      rnd = self.standard_normal(shape, dtype)
      return math_ops.add(rnd * stddev, mean, name=name)

  # TODO(wangpeng): implement other distributions (`uniform`,
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
    algorithm = int(global_generator.algorithm)  # preserve the old algorithm
  global_generator = Generator(seed=seed, algorithm=algorithm)
