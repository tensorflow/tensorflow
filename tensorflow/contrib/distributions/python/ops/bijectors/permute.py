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
"""Permutation bijectors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.distributions import bijector
from tensorflow.python.util import deprecation


__all__ = [
    "Permute",
]


class Permute(bijector.Bijector):
  """Permutes the rightmost dimension of a `Tensor`.

  ```python
  tfd = tf.contrib.distributions

  reverse = tfd.bijectors.Permute(permutation=[2, 1, 0])

  reverse.forward([-1., 0., 1.])
  # ==> [1., 0., -1]

  reverse.inverse([1., 0., -1])
  # ==> [-1., 0., 1.]

  reverse.forward_log_det_jacobian(any_value)
  # ==> 0.

  reverse.inverse_log_det_jacobian(any_value)
  # ==> 0.
  ```

  Warning: `tf.estimator` may repeatedly build the graph thus
  `Permute(np.random.permutation(event_size)).astype("int32"))` is not a
  reliable parameterization (nor would it be even if using `tf.constant`). A
  safe alternative is to use `tf.get_variable` to achieve "init once" behavior,
  i.e.,

  ```python
  def init_once(x, name):
    return tf.get_variable(name, initializer=x, trainable=False)

  Permute(permutation=init_once(
      np.random.permutation(event_size).astype("int32"),
      name="permutation"))
  ```

  """

  @deprecation.deprecated(
      "2018-10-01",
      "The TensorFlow Distributions library has moved to "
      "TensorFlow Probability "
      "(https://github.com/tensorflow/probability). You "
      "should update all references to use `tfp.distributions` "
      "instead of `tf.contrib.distributions`.",
      warn_once=True)
  def __init__(self, permutation, validate_args=False, name=None):
    """Creates the `Permute` bijector.

    Args:
      permutation: An `int`-like vector-shaped `Tensor` representing the
        permutation to apply to the rightmost dimension of the transformed
        `Tensor`.
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
      name: Python `str`, name given to ops managed by this object.

    Raises:
      TypeError: if `not permutation.dtype.is_integer`.
      ValueError: if `permutation` does not contain exactly one of each of
        `{0, 1, ..., d}`.
    """
    with ops.name_scope(name, "permute", values=[permutation]):
      permutation = ops.convert_to_tensor(
          permutation,
          name="permutation")
      if not permutation.dtype.is_integer:
        raise TypeError("permutation.dtype ({}) should be `int`-like.".format(
            permutation.dtype.name))
      p = tensor_util.constant_value(permutation)
      if p is not None:
        if set(p) != set(np.arange(p.size)):
          raise ValueError("Permutation over `d` must contain exactly one of "
                           "each of `{0, 1, ..., d}`.")
      elif validate_args:
        p, _ = nn_ops.top_k(-permutation,
                            k=array_ops.shape(permutation)[-1],
                            sorted=True)
        permutation = control_flow_ops.with_dependencies([
            check_ops.assert_equal(
                -p, math_ops.range(array_ops.size(p)),
                message=("Permutation over `d` must contain exactly one of "
                         "each of `{0, 1, ..., d}`.")),
        ], permutation)
      self._permutation = permutation
      super(Permute, self).__init__(
          forward_min_event_ndims=1,
          is_constant_jacobian=True,
          validate_args=validate_args,
          name=name or "permute")

  @property
  def permutation(self):
    return self._permutation

  def _forward(self, x):
    return array_ops.gather(x, self.permutation, axis=-1)

  def _inverse(self, y):
    return array_ops.gather(
        y,
        array_ops.invert_permutation(self.permutation),
        axis=-1)

  def _inverse_log_det_jacobian(self, y):
    # is_constant_jacobian = True for this bijector, hence the
    # `log_det_jacobian` need only be specified for a single input, as this will
    # be tiled to match `event_ndims`.
    return constant_op.constant(0., dtype=y.dtype.base_dtype)

  def _forward_log_det_jacobian(self, x):
    return constant_op.constant(0., dtype=x.dtype.base_dtype)
