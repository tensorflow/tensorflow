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
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numbers

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops


def alpha_dropout(x, keep_prob, noise_shape=None, seed=None, name=None): # pylint: disable=invalid-name
  """Computes alpha dropout.

  Alpha Dropout is a dropout that maintains the self-normalizing property. For
  an input with zero mean and unit standard deviation, the output of
  Alpha Dropout maintains the original mean and standard deviation of the input.

  See [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)

  Args:
    x: A tensor.
    keep_prob: A scalar `Tensor` with the same type as x. The probability
      that each element is kept.
    noise_shape: A 1-D `Tensor` of type `int32`, representing the
      shape for randomly generated keep/drop flags.
    seed: A Python integer. Used to create random seeds. See
      `tf.compat.v1.set_random_seed` for behavior.
    name: A name for this operation (optional).

  Returns:
    A Tensor of the same shape of `x`.

  Raises:
    ValueError: If `keep_prob` is not in `(0, 1]`.

  """
  with ops.name_scope(name, "alpha_dropout", [x]) as name:
    x = ops.convert_to_tensor(x, name="x")
    if isinstance(keep_prob, numbers.Real) and not 0 < keep_prob <= 1.:
      raise ValueError("keep_prob must be a scalar tensor or a float in the "
                       "range (0, 1], got %g" % keep_prob)
    keep_prob = ops.convert_to_tensor(keep_prob,
                                      dtype=x.dtype,
                                      name="keep_prob")
    keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

    # Do nothing if we know keep_prob == 1
    if tensor_util.constant_value(keep_prob) == 1:
      return x

    alpha = -1.7580993408473766

    noise_shape = noise_shape if noise_shape is not None else array_ops.shape(x)
    random_tensor = random_ops.random_uniform(noise_shape,
                                              seed=seed,
                                              dtype=x.dtype)
    kept_idx = gen_math_ops.greater_equal(random_tensor, 1 - keep_prob)
    kept_idx = math_ops.cast(kept_idx, x.dtype)
    # Mask
    x = x * kept_idx + alpha * (1 - kept_idx)

    # Affine transformation parameters
    a = (keep_prob + keep_prob * (1 - keep_prob) * alpha ** 2) ** -0.5
    b = -a * alpha * (1 - keep_prob)

    # Affine transformation
    return a * x + b
