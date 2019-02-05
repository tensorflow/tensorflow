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
"""TransformDiagonal bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import array_ops
from tensorflow.python.ops.distributions import bijector
from tensorflow.python.util import deprecation

__all__ = [
    "TransformDiagonal",
]


class TransformDiagonal(bijector.Bijector):
  """Applies a Bijector to the diagonal of a matrix.

  #### Example

  ```python
  b = tfb.TransformDiagonal(diag_bijector=tfb.Exp())

  b.forward([[1., 0.],
             [0., 1.]])
  # ==> [[2.718, 0.],
         [0., 2.718]]
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
  def __init__(self,
               diag_bijector,
               validate_args=False,
               name="transform_diagonal"):
    """Instantiates the `TransformDiagonal` bijector.

    Args:
      diag_bijector: `Bijector` instance used to transform the diagonal.
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
      name: Python `str` name given to ops managed by this object.
    """
    self._diag_bijector = diag_bijector
    super(TransformDiagonal, self).__init__(
        forward_min_event_ndims=2,
        inverse_min_event_ndims=2,
        validate_args=validate_args,
        name=name)

  def _forward(self, x):
    diag = self._diag_bijector.forward(array_ops.matrix_diag_part(x))
    return array_ops.matrix_set_diag(x, diag)

  def _inverse(self, y):
    diag = self._diag_bijector.inverse(array_ops.matrix_diag_part(y))
    return array_ops.matrix_set_diag(y, diag)

  def _forward_log_det_jacobian(self, x):
    # We formulate the Jacobian with respect to the flattened matrices
    # `vec(x)` and `vec(y)`. Suppose for notational convenience that
    # the first `n` entries of `vec(x)` are the diagonal of `x`, and
    # the remaining `n**2-n` entries are the off-diagonals in
    # arbitrary order. Then the Jacobian is a block-diagonal matrix,
    # with the Jacobian of the diagonal bijector in the first block,
    # and the identity Jacobian for the remaining entries (since this
    # bijector acts as the identity on non-diagonal entries):
    #
    # J_vec(x) (vec(y)) =
    # -------------------------------
    # | J_diag(x) (diag(y))      0  | n entries
    # |                             |
    # | 0                        I  | n**2-n entries
    # -------------------------------
    #   n                     n**2-n
    #
    # Since the log-det of the second (identity) block is zero, the
    # overall log-det-jacobian is just the log-det of first block,
    # from the diagonal bijector.
    #
    # Note that for elementwise operations (exp, softplus, etc) the
    # first block of the Jacobian will itself be a diagonal matrix,
    # but our implementation does not require this to be true.
    return self._diag_bijector.forward_log_det_jacobian(
        array_ops.matrix_diag_part(x), event_ndims=1)

  def _inverse_log_det_jacobian(self, y):
    return self._diag_bijector.inverse_log_det_jacobian(
        array_ops.matrix_diag_part(y), event_ndims=1)
