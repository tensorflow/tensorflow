# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""FillTriangular bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import bijector
from tensorflow.python.ops.distributions import util as dist_util


__all__ = [
    "FillTriangular",
]


class FillTriangular(bijector.Bijector):
  """Transforms vectors to triangular.

  Triangular matrix elements are filled in a clockwise spiral.

  Given input with shape `batch_shape + [d]`, produces output with
  shape `batch_shape + [n, n]`, where
   `n = (-1 + sqrt(1 + 8 * d))/2`.
  This follows by solving the quadratic equation
   `d = 1 + 2 + ... + n = n * (n + 1)/2`.

  #### Example

  ```python
  b = tfb.FillTriangular(upper=False)
  b.forward([1, 2, 3, 4, 5, 6])
  # ==> [[4, 0, 0],
  #      [6, 5, 0],
  #      [3, 2, 1]]

  b = tfb.FillTriangular(upper=True)
  b.forward([1, 2, 3, 4, 5, 6])
  # ==> [[1, 2, 3],
  #      [0, 5, 6],
  #      [0, 0, 4]]

  ```
  """

  def __init__(self,
               upper=False,
               validate_args=False,
               name="fill_triangular"):
    """Instantiates the `FillTriangular` bijector.

    Args:
      upper: Python `bool` representing whether output matrix should be upper
        triangular (`True`) or lower triangular (`False`, default).
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
      name: Python `str` name given to ops managed by this object.
    """
    self._upper = upper
    super(FillTriangular, self).__init__(
        forward_min_event_ndims=1,
        inverse_min_event_ndims=2,
        validate_args=validate_args,
        name=name)

  def _forward(self, x):
    return dist_util.fill_triangular(x, upper=self._upper)

  def _inverse(self, y):
    return dist_util.fill_triangular_inverse(y, upper=self._upper)

  def _forward_log_det_jacobian(self, x):
    return array_ops.zeros_like(x[..., 0])

  def _inverse_log_det_jacobian(self, y):
    return array_ops.zeros_like(y[..., 0, 0])

  def _forward_event_shape(self, input_shape):
    batch_shape, d = input_shape[:-1], input_shape[-1].value
    if d is None:
      n = None
    else:
      n = vector_size_to_square_matrix_size(d, self.validate_args)
    return batch_shape.concatenate([n, n])

  def _inverse_event_shape(self, output_shape):
    batch_shape, n1, n2 = (output_shape[:-2],
                           output_shape[-2].value,
                           output_shape[-1].value)
    if n1 is None or n2 is None:
      m = None
    elif n1 != n2:
      raise ValueError("Matrix must be square. (saw [{}, {}])".format(n1, n2))
    else:
      m = n1 * (n1 + 1) / 2
    return batch_shape.concatenate([m])

  def _forward_event_shape_tensor(self, input_shape_tensor):
    batch_shape, d = input_shape_tensor[:-1], input_shape_tensor[-1]
    n = vector_size_to_square_matrix_size(d, self.validate_args)
    return array_ops.concat([batch_shape, [n, n]], axis=0)

  def _inverse_event_shape_tensor(self, output_shape_tensor):
    batch_shape, n = output_shape_tensor[:-2], output_shape_tensor[-1]
    if self.validate_args:
      is_square_matrix = check_ops.assert_equal(
          n, output_shape_tensor[-2], message="Matrix must be square.")
      with ops.control_dependencies([is_square_matrix]):
        n = array_ops.identity(n)
    d = math_ops.cast(n * (n + 1) / 2, output_shape_tensor.dtype)
    return array_ops.concat([batch_shape, [d]], axis=0)


def vector_size_to_square_matrix_size(d, validate_args, name=None):
  """Convert a vector size to a matrix size."""
  if isinstance(d, (float, int, np.generic, np.ndarray)):
    n = (-1 + np.sqrt(1 + 8 * d)) / 2.
    if float(int(n)) != n:
      raise ValueError("Vector length is not a triangular number.")
    return int(n)
  else:
    with ops.name_scope(name, "vector_size_to_square_matrix_size", [d]) as name:
      n = (-1. + math_ops.sqrt(1 + 8. * math_ops.to_float(d))) / 2.
      if validate_args:
        with ops.control_dependencies([check_ops.assert_equal(
            math_ops.to_float(math_ops.to_int32(n)), n,
            message="Vector length is not a triangular number")]):
          n = array_ops.identity(n)
      return math_ops.cast(n, d.dtype)
