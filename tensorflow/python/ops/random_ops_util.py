# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Utilities for random ops to share common usages."""

import enum

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import bitwise_ops
from tensorflow.python.ops import gen_stateless_random_ops_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.util.tf_export import tf_export


@tf_export("random.Algorithm", "random.experimental.Algorithm")
class Algorithm(enum.Enum):
  """A random-number-generation (RNG) algorithm.

  Many random-number generators (e.g. the `alg` argument of
  `tf.random.Generator` and `tf.random.stateless_uniform`) in TF allow
  you to choose the algorithm used to generate the (pseudo-)random
  numbers. You can set the algorithm to be one of the options below.

  * `PHILOX`: The Philox algorithm introduced in the paper ["Parallel
    Random Numbers: As Easy as 1, 2,
    3"](https://www.thesalmons.org/john/random123/papers/random123sc11.pdf).
  * `THREEFRY`: The ThreeFry algorithm introduced in the paper
    ["Parallel Random Numbers: As Easy as 1, 2,
    3"](https://www.thesalmons.org/john/random123/papers/random123sc11.pdf).
  * `AUTO_SELECT`: Allow TF to automatically select the algorithm
    depending on the accelerator device. Note that with this option,
    running the same TF program on different devices may result in
    different random numbers. Also note that TF may select an
    algorithm that is different from `PHILOX` and `THREEFRY`.
  """

  # The numbers here must match framework/rng_alg.h
  PHILOX = 1
  THREEFRY = 2
  AUTO_SELECT = 3


def convert_alg_to_int(alg):
  """Converts algorithm to an integer.

  Args:
    alg: can be one of these types: integer, Algorithm, Tensor, string. Allowed
      strings are "philox" and "threefry".

  Returns:
    An integer, unless the input is a Tensor in which case a Tensor is returned.
  """
  if isinstance(alg, int):
    return alg
  if isinstance(alg, Algorithm):
    return alg.value
  if isinstance(alg, tensor.Tensor):
    return alg
  if isinstance(alg, str):
    # canonicalized alg
    canon_alg = alg.strip().lower().replace("-", "").replace("_", "")
    if canon_alg == "philox":
      return Algorithm.PHILOX.value
    elif canon_alg == "threefry":
      return Algorithm.THREEFRY.value
    elif canon_alg == "autoselect":
      return Algorithm.AUTO_SELECT.value
    else:
      raise ValueError(unsupported_alg_error_msg(alg))
  else:
    raise TypeError(
        f"Can't convert argument `alg` (of value {alg} and type {type(alg)}) "
        "to int."
    )


def _get_key_counter(seed, alg):
  """Calculates the key and counter to pass to raw RNG ops.

  This function calculates the key and counter that will be passed to
  the raw RNG ops like `StatelessRandomUniformV2`. Depending on the
  input `alg`, the key and counter may be scrambled or copied from
  `seed`. If `alg` is `"auto_select"`, the key and counter will be
  determined at runtime based on device type.

  Args:
    seed: An integer tensor of shape [2]. The seed to calculate the key and
      counter from.
    alg: The RNG algorithm. See `tf.random.stateless_uniform` for an
      explanation.

  Returns:
    A pair (key, counter) suitable for V2 stateless RNG ops like
    `StatelessRandomUniformV2`.
  """
  if alg == Algorithm.AUTO_SELECT.value:
    key, counter = gen_stateless_random_ops_v2.stateless_random_get_key_counter(
        seed
    )
  elif alg == Algorithm.PHILOX.value:
    key, counter = _philox_scramble_seed(seed)
  elif alg == Algorithm.THREEFRY.value:
    key = array_ops.reshape(
        _uint32s_to_uint64(math_ops.cast(seed, dtypes.uint32)), [1]
    )
    counter = array_ops.zeros([1], dtypes.uint64)
  else:
    raise ValueError(unsupported_alg_error_msg(alg))
  return key, counter


def get_key_counter_alg(seed, alg):
  """Calculates the key, counter and algorithm to pass to raw RNG ops.

  This function calculates the key and counter, and determines the algorithm
  that will be passed to the raw RNG ops like `StatelessRandomUniformV2`.
  Depending on the input `alg`, the key and counter may be scrambled or copied
  from `seed`. If `alg` is `"auto_select"`, the key and counter will be
  determined at runtime based on device type.

  Args:
    seed: An integer tensor of shape [2]. The seed to calculate the key and
      counter from.
    alg: The RNG algorithm. See `tf.random.stateless_uniform` for an
      explanation.

  Returns:
    A pair (key, counter, algorithm) suitable for V2 stateless RNG ops like
    `StatelessRandomUniformV2`.
  """
  if alg is None:
    alg = Algorithm.AUTO_SELECT.value
  alg = convert_alg_to_int(alg)
  key, counter = _get_key_counter(seed, alg)
  return key, counter, alg


def _uint32s_to_uint64(x):
  return bitwise_ops.bitwise_or(
      math_ops.cast(x[0], dtypes.uint64),
      bitwise_ops.left_shift(
          math_ops.cast(x[1], dtypes.uint64),
          constant_op.constant(32, dtypes.uint64),
      ),
  )


def unsupported_alg_error_msg(alg):
  """Produces the unsupported-algorithm error message."""
  if isinstance(alg, int):
    philox = Algorithm.PHILOX.value
    threefry = Algorithm.THREEFRY.value
    auto_select = Algorithm.AUTO_SELECT.value
  elif isinstance(alg, str):
    philox = "philox"
    threefry = "threefry"
    auto_select = "auto_select"
  else:
    philox = Algorithm.PHILOX
    threefry = Algorithm.THREEFRY
    auto_select = Algorithm.AUTO_SELECT
  return (
      f"Argument `alg` got unsupported value {alg}. Supported values are "
      f"{philox} for the Philox algorithm, "
      f"{threefry} for the ThreeFry algorithm, and "
      f"{auto_select} for auto-selection."
  )


def _philox_scramble_seed(seed):
  """Determines the key and counter for Philox PRNG with the given seed.

  Args:
    seed: An integer tensor of shape [2]. The seed to calculate the key and
      counter from.

  Returns:
    A pair (key, counter) suitable for V2 stateless RNG ops like
    `StatelessRandomUniformV2`.
  """
  # the same scrambling procedure as core/kernels/stateless_random_ops.cc
  key = constant_op.constant([0x02461E293EC8F720], dtypes.uint64)
  counter = math_ops.cast(seed, dtypes.uint64)
  mix = gen_stateless_random_ops_v2.stateless_random_uniform_full_int_v2(
      [4],
      key=key,
      counter=counter,
      dtype=dtypes.uint32,
      alg=Algorithm.PHILOX.value,
  )
  key = array_ops.reshape(_uint32s_to_uint64(mix[:2]), [1])
  counter = array_ops_stack.stack([0, _uint32s_to_uint64(mix[2:])], axis=0)
  return key, counter
