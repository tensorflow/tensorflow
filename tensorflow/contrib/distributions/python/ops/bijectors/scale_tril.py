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
"""ScaleTriL bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.distributions.python.ops.bijectors import affine_scalar
from tensorflow.contrib.distributions.python.ops.bijectors import chain
from tensorflow.contrib.distributions.python.ops.bijectors import fill_triangular
from tensorflow.contrib.distributions.python.ops.bijectors import softplus
from tensorflow.contrib.distributions.python.ops.bijectors import transform_diagonal

__all__ = [
    "ScaleTriL",
]


class ScaleTriL(chain.Chain):
  """Transforms unconstrained vectors to TriL matrices with positive diagonal.

  This is implemented as a simple `tfb.Chain` of `tfb.FillTriangular`
  followed by `tfb.TransformDiagonal`, and provided mostly as a
  convenience. The default setup is somewhat opinionated, using a
  Softplus transformation followed by a small shift (`1e-5`) which
  attempts to avoid numerical issues from zeros on the diagonal.

  #### Examples

  ```python
  tfb = tf.contrib.distributions.bijectors
  b = tfb.ScaleTriL(
       diag_bijector=tfb.Exp(),
       diag_shift=None)
  b.forward(x=[0., 0., 0.])
  # Result: [[1., 0.],
  #          [0., 1.]]
  b.inverse(y=[[1., 0],
               [.5, 2]])
  # Result: [log(2), .5, log(1)]

  # Define a distribution over PSD matrices of shape `[3, 3]`,
  # with `1 + 2 + 3 = 6` degrees of freedom.
  dist = tfd.TransformedDistribution(
          tfd.Normal(tf.zeros(6), tf.ones(6)),
          tfb.Chain([tfb.CholeskyOuterProduct(), tfb.ScaleTriL()]))

  # Using an identity transformation, ScaleTriL is equivalent to
  # tfb.FillTriangular.
  b = tfb.ScaleTriL(
       diag_bijector=tfb.Identity(),
       diag_shift=None)

  # For greater control over initialization, one can manually encode
  # pre- and post- shifts inside of `diag_bijector`.
  b = tfb.ScaleTriL(
       diag_bijector=tfb.Chain([
         tfb.AffineScalar(shift=1e-3),
         tfb.Softplus(),
         tfb.AffineScalar(shift=0.5413)]),  # softplus_inverse(1.)
                                            #  = log(expm1(1.)) = 0.5413
       diag_shift=None)
  ```
  """

  def __init__(self,
               diag_bijector=None,
               diag_shift=1e-5,
               validate_args=False,
               name="scale_tril"):
    """Instantiates the `ScaleTriL` bijector.

    Args:
      diag_bijector: `Bijector` instance, used to transform the output diagonal
        to be positive.
        Default value: `None` (i.e., `tfb.Softplus()`).
      diag_shift: Float value broadcastable and added to all diagonal entries
        after applying the `diag_bijector`. Setting a positive
        value forces the output diagonal entries to be positive, but
        prevents inverting the transformation for matrices with
        diagonal entries less than this value.
        Default value: `1e-5` (i.e., no shift is applied).
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
        Default value: `False` (i.e., arguments are not validated).
      name: Python `str` name given to ops managed by this object.
        Default value: `scale_tril`.
    """

    if diag_bijector is None:
      diag_bijector = softplus.Softplus(validate_args=validate_args)

    if diag_shift is not None:
      diag_bijector = chain.Chain([affine_scalar.AffineScalar(shift=diag_shift),
                                   diag_bijector])

    super(ScaleTriL, self).__init__(
        [transform_diagonal.TransformDiagonal(diag_bijector=diag_bijector),
         fill_triangular.FillTriangular()],
        validate_args=validate_args,
        name=name)
