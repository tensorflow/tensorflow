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
"""Internal ops related to the Graphcore IPU."""
from functools import wraps
from tensorflow.python.platform import tf_logging as logging
from tensorflow.compiler.plugin.poplar.ops import gen_poputil_ops
from tensorflow.compiler.plugin.poplar.ops import gen_poprand_ops

import tensorflow as tf
import numpy as np


def dropout(x, seed=None, rate=0.5, scale=1, seed_modifier=1, name=None):
  """This targets the poplibs popnn dropout operation, optimized for execution on the IPU.

  Args:
    x: The input tensor.
    rate: The probability that a given element will be zeroed out.
    scale: An optional factor to apply to all other elements.
    seed_modifier: An optional parameter given to poplar which uses it to modify the seed.
    name: Optional op name.

  Returns:
    A `Tensor` which has some nodes set to zero, as randomly selected based on other parameters.
  """

  # Rate is a probability between 0 and 1. Specifically the rate that a variable will be droppe d out.
  if rate > 1.0 or rate < 0.0:
    raise ValueError("Rate must be between 0.0 and 1.0" % rate)

  is_using_user_seed = True
  if seed is None:
    is_using_user_seed = False
    # Create empty placeholder we will generate a random one internally.
    seed = tf.zeros([2], tf.int32)

  # We transfrom rate to be the change an individual node will dropout as ipu_dropout
  # is using the old tensorflow method that rate is the probability that value is kept
  # rather than disgarded.
  return gen_poprand_ops.ipu_dropout(
      x,
      seed=seed,
      user_seed=1,
      rate=(1 - rate),
      scale=scale,
      name=name,
      is_using_user_seed=is_using_user_seed,
      seed_modifier=seed_modifier)[0]
