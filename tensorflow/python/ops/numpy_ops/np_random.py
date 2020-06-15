# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Random functions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import random_seed
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.numpy_ops import np_utils

DEFAULT_RANDN_DTYPE = np.float32


def randn(*args):
  """Returns samples from a normal distribution.

  Uses `tf.random_normal`.

  Args:
    *args: The shape of the output array.

  Returns:
    An ndarray with shape `args` and dtype `float64`.
  """
  # TODO(wangpeng): Use new stateful RNG
  if np_utils.isscalar(args):
    args = (args,)
  return np_utils.tensor_to_ndarray(
      random_ops.random_normal(args, dtype=DEFAULT_RANDN_DTYPE))


def seed(s):
  """Sets the seed for the random number generator.

  Uses `tf.set_random_seed`.

  Args:
    s: an integer.
  """
  # TODO(wangpeng): make the signature the same as numpy
  random_seed.set_seed(s)
