# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""An API for reversible (bijective) transformations of random variables."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops


class _Bijector(object):
  """An interface for transforming random variable(s).

  A bijector is characterized by three operations:

  1) Forward Evaluation
     Useful for turning one random outcome into another random outcome from a
     different distribution.

  2) Inverse Evaluation
     Useful for "reversing" a transformation to compute one probability in terms
     of another.

  3) (log o det o Jacobian o inverse)(x)
     "The log of the determinant of the matrix of all first-order partial
     derivatives of the inverse function."
     Useful for inverting a transformation to compute one probability in terms
     of another.  Geometrically, the det(Jacobian) is the volume of the
     transformation and is used to scale the probability.

  By convention, transformations of random variables are named in terms of the
  forward transformation. The forward transformation creates samples, the
  inverse is useful for computing probabilities.

  Example transformations:
    "Exponential"

      ```
      Y = g(X) = exp(X)
      X ~ Normal(0, 1)  # Univariate.
      ```

      Implies:

      ```
        g^{-1}(Y) = log(Y)
        |Jacobian(g^{-1})(y)| = 1 / y
        Y ~ LogNormal(0, 1), i.e.,
        prob(Y=y) = |Jacobian(g^{-1})(y)| * prob(X=g^{-1}(y))
                  = (1 / y) Normal(log(y); 0, 1)
      ```

    "ShiftAndScale"

      ```
      Y = g(X) = sqrtSigma * X + mu
      X ~ MultivariateNormal(0, I_d)
      ```

      Implies:

      ```
        g^{-1}(Y) = inv(sqrtSigma) * (Y - mu)
        |Jacobian(g^{-1})(y)| = det(inv(sqrtSigma))
        Y ~ MultivariateNormal(mu, sqrtSigma) , i.e.,
        prob(Y=y) = |Jacobian(g^{-1})(y)| * prob(X=g^{-1}(y))
                  = det(sqrtSigma)^(-d) *
                    MultivariateNormal(inv(sqrtSigma) * (y - mu); 0, I_d)
      ```

  Example use:
    Basic properties:

    ```python
    x = ... # A tensor.
    # Evaluate forward transformation.
    fwd_x = my_bijector.forward(x)
    x != my_bijector.forward(fwd_x)  # Not equal because g(x) != g(g(x)).
    x == my_bijector.inverse(fwd_x)
    ```

    Computing a log-likelihood:

    ```python
    def transformed_log_pdf(bijector, log_pdf, x):
      return (bijector.inverse_log_det_jacobian(x) +
              log_pdf(bijector.inverse(x)))
    ```

    Transforming a random outcome:

    ```python
    def transformed_sample(bijector, x):
      return bijector.forward(x)
    ```

  """

  # TODO(b/30476956): Try to remove constructor dependence on shape util.
  def __init__(self, shaper=None, name=None):
    """Constructs Bijector.

    A bijector transforms random variables into new random variables. Managing
    shape is typically an important piece of this so a Bijector is usually
    composed of ShapeUtil. The ShapeUtil object handles input shape checks as
    well as reshaping/transposing for easier linear algebra operations.

    Example:
    ```python
    # Create the Y = g(X) = X transform which operates on 4-Tensors of vectors.
    identity = Identity(ShapeUtil(batch_ndims=4, event_ndims=1))

    # Create the Y = g(X) = exp(X) transform which operates on matrices.
    exp = Exp(ShapeUtil(batch_ndims=0, event_ndims=2))
    ```

    See Bijector subclass doc for more details and examples.

    Args:
      shaper: object used for managing and manipulating shape, typically an
        instance of ShapeUtil.
      name: The name to give Ops created by the initializer.
    """
    self._shaper = shaper
    self._name = name or type(self).__name__

  @property
  def shaper(self):
    """Returns shape object used to manage shape constraints."""
    return self._shaper

  @property
  def name(self):
    """Returns the string name of this bijector."""
    return self._name

  def forward(self, x, name='forward'):
    """Returns the forward bijector evaluation, i.e., X = g(Y).

    Args:
      x: `Tensor`. The input to the "forward" evaluation.
      name: The name to give this op.

    Returns:
      `Tensor`.
    """
    with ops.name_scope(self.name):
      with ops.op_scope([x], name):
        x = ops.convert_to_tensor(x)
        return self._forward(x)

  def inverse(self, x, name='inverse'):
    """Returns the inverse bijector evaluation, i.e., X = g^{-1}(Y).

    Args:
      x: `Tensor`. The input to the "inverse" evaluation.
      name: The name to give this op.

    Returns:
      `Tensor`.
    """
    with ops.name_scope(self.name):
      with ops.op_scope([x], name):
        x = ops.convert_to_tensor(x)
        try:
          return self._inverse(x)
        except NotImplementedError:
          return self._inverse_and_inverse_log_det_jacobian(x)[0]

  def inverse_log_det_jacobian(self, x, name='inverse_log_det_jacobian'):
    """Returns the (log o det o Jacobian o inverse)(x).

    Mathematically, returns: log(det(dY/dX g^{-1}))(Y).

    Args:
      x: `Tensor`. The input to the "inverse" Jacobian evaluation.
      name: The name to give this op.

    Returns:
      `Tensor`.
    """
    with ops.name_scope(self.name):
      with ops.op_scope([x], name):
        x = ops.convert_to_tensor(x)
        try:
          return self._inverse_log_det_jacobian(x)
        except NotImplementedError:
          return self._inverse_and_inverse_log_det_jacobian(x)[1]

  def inverse_and_inverse_log_det_jacobian(
      self, x, name='inverse_and_inverse_log_det_jacobian'):
    """Returns both the inverse evaluation and inverse_log_det_jacobian.

    Enables possibly more efficient calculation when both inverse and
    corresponding Jacobian are needed.

    See `inverse()`, `inverse_log_det_jacobian()` for more details.

    Args:
      x: `Tensor`. The input to the "inverse" Jacobian evaluation.
      name: The name to give this op.

    Returns:
      `Tensor`.
    """
    with ops.name_scope(self.name):
      with ops.op_scope([x], name):
        x = ops.convert_to_tensor(x)
        try:
          return self._inverse_and_inverse_log_det_jacobian(x)
        except NotImplementedError:
          return self._inverse(x), self._inverse_log_det_jacobian(x)

  # Subclass interface.
  def _forward(self, x):
    """Subclass implementation of forward().

    Args:
      x: `Tensor`. The input to the "forward" evaluation.

    Raises:
      `NotImplementedError`: if subclass implementation not provided

    Returns:
      `Tensor`.
    """
    raise NotImplementedError('_forward not implemented')

  def _inverse(self, x):
    """Subclass implementation of inverse().

    Args:
      x: `Tensor`. The input to the "inverse" evaluation.

    Raises:
      `NotImplementedError`: if subclass implementation not provided

    Returns:
      `Tensor`.
    """
    raise NotImplementedError('_inverse not implemented')

  def _inverse_log_det_jacobian(self, x):
    """Subclass implementation of inverse_log_det_jacobian().

    Args:
      x: `Tensor`. The input to the "inverse" Jacobian evaluation.

    Raises:
      `NotImplementedError`: if subclass implementation not provided

    Returns:
      `Tensor`.
    """
    raise NotImplementedError('_inverse_log_det_jacobian not implemented')

  def _inverse_and_inverse_log_det_jacobian(self, x):
    """Subclass implementation of inverse_and_inverse_log_det_jacobian().

    Args:
      x: `Tensor`. The input to the "inverse" evaluation.

    Returns:
      List of two `Tensor` items, inverse and inverse_log_det_jacobian.
    """
    raise NotImplementedError(
        '_inverse_and_inverse_log_det_jacobian not implemented')


class _Identity(_Bijector):
  """Bijector which computes Y = g(X) = X.

    Example Use:
    ```python
    # Create the Y=g(X)=X transform which works only on Tensors with 1 batch
    # ndims and 1 event ndim (i.e., vector of vectors).
    identity = Identity(ShapeUtil(batch_ndims=1, event_ndims=1))
    x = [[1., 2],
         [3, 4]]
    x == identity.forward(x) == identity.inverse(x)
    ```

  """

  # TODO(b/30476956): Try to remove constructor dependence on shape util.
  def __init__(self, shaper=None, name='Identity'):
    super(_Identity, self).__init__(shaper, name)

  def _forward(self, x):
    return x

  def _inverse(self, x):
    return x

  def _inverse_log_det_jacobian(self, x):
    result_shape = self.shaper.get_shape(
        x, sample=True, batch=True, event=False)
    return array_ops.zeros(result_shape, dtype=x.dtype)


class _Exp(_Bijector):
  """Bijector which computes Y = g(X) = exp(X).

    Example Use:
    ```python
    # Create the Y=g(X)=exp(X) transform which works only on Tensors with 1
    # batch ndims and 2 event ndim (i.e., vector of matrices).
    exp = Exp(ShapeUtil(batch_ndims=1, event_ndims=2))
    x = [[[1., 2],
           [3, 4]],
          [[5, 6],
           [7, 8]]]
    exp(x) == exp.forward(x)
    log(x) == exp.inverse(x)
    ```

  """

  # TODO(b/30476956): Try to remove constructor dependence on shape util.
  def __init__(self, shaper=None, name='Exp'):
    super(_Exp, self).__init__(shaper, name)

  def _forward(self, x):
    return math_ops.exp(x)

  def _inverse(self, x):
    return math_ops.log(x)

  def _inverse_log_det_jacobian(self, x):
    d = self.shaper.get_event_dims(x)
    return -math_ops.reduce_sum(math_ops.log(x), d)

  def _inverse_and_inverse_log_det_jacobian(self, x):
    y = math_ops.log(x)
    d = self.shaper.get_event_dims(x)
    return y, -math_ops.reduce_sum(y, d)
