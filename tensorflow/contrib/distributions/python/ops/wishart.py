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
"""The Wishart distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np

from tensorflow.contrib.distributions.python.ops import distribution_util
from tensorflow.contrib.framework.python.framework import tensor_util as contrib_tensor_util
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.distributions import distribution
from tensorflow.python.ops.linalg import linalg
from tensorflow.python.util import deprecation

__all__ = [
    "WishartCholesky",
    "WishartFull",
]


class _WishartLinearOperator(distribution.Distribution):
  """The matrix Wishart distribution on positive definite matrices.

  This distribution is defined by a scalar number of degrees of freedom `df` and
  an instance of `LinearOperator`, which provides matrix-free access to a
  symmetric positive definite operator, which defines the scale matrix.

  #### Mathematical Details

  The probability density function (pdf) is,

  ```none
  pdf(X; df, scale) = det(X)**(0.5 (df-k-1)) exp(-0.5 tr[inv(scale) X]) / Z
  Z = 2**(0.5 df k) |det(scale)|**(0.5 df) Gamma_k(0.5 df)
  ```

  where:

  * `df >= k` denotes the degrees of freedom,
  * `scale` is a symmetric, positive definite, `k x k` matrix,
  * `Z` is the normalizing constant, and,
  * `Gamma_k` is the [multivariate Gamma function](
    https://en.wikipedia.org/wiki/Multivariate_gamma_function).

  #### Examples

  See `WishartFull`, `WishartCholesky` for examples of initializing and using
  this class.
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
               df,
               scale_operator,
               cholesky_input_output_matrices=False,
               validate_args=False,
               allow_nan_stats=True,
               name=None):
    """Construct Wishart distributions.

    Args:
      df: `float` or `double` tensor, the degrees of freedom of the
        distribution(s). `df` must be greater than or equal to `k`.
      scale_operator: `float` or `double` instance of `LinearOperator`.
      cholesky_input_output_matrices: Python `bool`. Any function which whose
        input or output is a matrix assumes the input is Cholesky and returns a
        Cholesky factored matrix. Example `log_prob` input takes a Cholesky and
        `sample_n` returns a Cholesky when
        `cholesky_input_output_matrices=True`.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
        result is undefined. When `False`, an exception is raised if one or
        more of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.

    Raises:
      TypeError: if scale is not floating-type
      TypeError: if scale.dtype != df.dtype
      ValueError: if df < k, where scale operator event shape is
        `(k, k)`
    """
    parameters = dict(locals())
    self._cholesky_input_output_matrices = cholesky_input_output_matrices
    with ops.name_scope(name) as name:
      with ops.name_scope("init", values=[df, scale_operator]):
        if not scale_operator.dtype.is_floating:
          raise TypeError(
              "scale_operator.dtype=%s is not a floating-point type" %
              scale_operator.dtype)
        if not scale_operator.is_square:
          print(scale_operator.to_dense().eval())
          raise ValueError("scale_operator must be square.")

        self._scale_operator = scale_operator
        self._df = ops.convert_to_tensor(
            df,
            dtype=scale_operator.dtype,
            name="df")
        contrib_tensor_util.assert_same_float_dtype(
            (self._df, self._scale_operator))
        if (self._scale_operator.shape.ndims is None or
            self._scale_operator.shape.dims[-1].value is None):
          self._dimension = math_ops.cast(
              self._scale_operator.domain_dimension_tensor(),
              dtype=self._scale_operator.dtype, name="dimension")
        else:
          self._dimension = ops.convert_to_tensor(
              self._scale_operator.shape.dims[-1].value,
              dtype=self._scale_operator.dtype, name="dimension")
        df_val = tensor_util.constant_value(self._df)
        dim_val = tensor_util.constant_value(self._dimension)
        if df_val is not None and dim_val is not None:
          df_val = np.asarray(df_val)
          if not df_val.shape:
            df_val = [df_val]
          if any(df_val < dim_val):
            raise ValueError(
                "Degrees of freedom (df = %s) cannot be less than "
                "dimension of scale matrix (scale.dimension = %s)"
                % (df_val, dim_val))
        elif validate_args:
          assertions = check_ops.assert_less_equal(
              self._dimension, self._df,
              message=("Degrees of freedom (df = %s) cannot be "
                       "less than dimension of scale matrix "
                       "(scale.dimension = %s)" %
                       (self._dimension, self._df)))
          self._df = control_flow_ops.with_dependencies(
              [assertions], self._df)
    super(_WishartLinearOperator, self).__init__(
        dtype=self._scale_operator.dtype,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        reparameterization_type=distribution.FULLY_REPARAMETERIZED,
        parameters=parameters,
        graph_parents=([self._df, self._dimension] +
                       self._scale_operator.graph_parents),
        name=name)

  @property
  def df(self):
    """Wishart distribution degree(s) of freedom."""
    return self._df

  def _square_scale_operator(self):
    return self.scale_operator.matmul(
        self.scale_operator.to_dense(), adjoint_arg=True)

  def scale(self):
    """Wishart distribution scale matrix."""
    if self._cholesky_input_output_matrices:
      return self.scale_operator.to_dense()
    else:
      return self._square_scale_operator()

  @property
  def scale_operator(self):
    """Wishart distribution scale matrix as an Linear Operator."""
    return self._scale_operator

  @property
  def cholesky_input_output_matrices(self):
    """Boolean indicating if `Tensor` input/outputs are Cholesky factorized."""
    return self._cholesky_input_output_matrices

  @property
  def dimension(self):
    """Dimension of underlying vector space. The `p` in `R^(p*p)`."""
    return self._dimension

  def _event_shape_tensor(self):
    dimension = self.scale_operator.domain_dimension_tensor()
    return array_ops.stack([dimension, dimension])

  def _event_shape(self):
    dimension = self.scale_operator.domain_dimension
    return tensor_shape.TensorShape([dimension, dimension])

  def _batch_shape_tensor(self):
    return self.scale_operator.batch_shape_tensor()

  def _batch_shape(self):
    return self.scale_operator.batch_shape

  def _sample_n(self, n, seed):
    batch_shape = self.batch_shape_tensor()
    event_shape = self.event_shape_tensor()
    batch_ndims = array_ops.shape(batch_shape)[0]

    ndims = batch_ndims + 3  # sample_ndims=1, event_ndims=2
    shape = array_ops.concat([[n], batch_shape, event_shape], 0)

    # Complexity: O(nbk**2)
    x = random_ops.random_normal(shape=shape,
                                 mean=0.,
                                 stddev=1.,
                                 dtype=self.dtype,
                                 seed=seed)

    # Complexity: O(nbk)
    # This parametrization is equivalent to Chi2, i.e.,
    # ChiSquared(k) == Gamma(alpha=k/2, beta=1/2)
    expanded_df = self.df * array_ops.ones(
        self.scale_operator.batch_shape_tensor(),
        dtype=self.df.dtype.base_dtype)
    g = random_ops.random_gamma(shape=[n],
                                alpha=self._multi_gamma_sequence(
                                    0.5 * expanded_df, self.dimension),
                                beta=0.5,
                                dtype=self.dtype,
                                seed=distribution_util.gen_new_seed(
                                    seed, "wishart"))

    # Complexity: O(nbk**2)
    x = array_ops.matrix_band_part(x, -1, 0)  # Tri-lower.

    # Complexity: O(nbk)
    x = array_ops.matrix_set_diag(x, math_ops.sqrt(g))

    # Make batch-op ready.
    # Complexity: O(nbk**2)
    perm = array_ops.concat([math_ops.range(1, ndims), [0]], 0)
    x = array_ops.transpose(x, perm)
    shape = array_ops.concat([batch_shape, [event_shape[0]], [-1]], 0)
    x = array_ops.reshape(x, shape)

    # Complexity: O(nbM) where M is the complexity of the operator solving a
    # vector system. E.g., for LinearOperatorDiag, each matmul is O(k**2), so
    # this complexity is O(nbk**2). For LinearOperatorLowerTriangular,
    # each matmul is O(k^3) so this step has complexity O(nbk^3).
    x = self.scale_operator.matmul(x)

    # Undo make batch-op ready.
    # Complexity: O(nbk**2)
    shape = array_ops.concat([batch_shape, event_shape, [n]], 0)
    x = array_ops.reshape(x, shape)
    perm = array_ops.concat([[ndims - 1], math_ops.range(0, ndims - 1)], 0)
    x = array_ops.transpose(x, perm)

    if not self.cholesky_input_output_matrices:
      # Complexity: O(nbk^3)
      x = math_ops.matmul(x, x, adjoint_b=True)

    return x

  def _log_prob(self, x):
    if self.cholesky_input_output_matrices:
      x_sqrt = x
    else:
      # Complexity: O(nbk^3)
      x_sqrt = linalg_ops.cholesky(x)

    batch_shape = self.batch_shape_tensor()
    event_shape = self.event_shape_tensor()
    ndims = array_ops.rank(x_sqrt)
    # sample_ndims = ndims - batch_ndims - event_ndims
    sample_ndims = ndims - array_ops.shape(batch_shape)[0] - 2
    sample_shape = array_ops.strided_slice(
        array_ops.shape(x_sqrt), [0], [sample_ndims])

    # We need to be able to pre-multiply each matrix by its corresponding
    # batch scale matrix. Since a Distribution Tensor supports multiple
    # samples per batch, this means we need to reshape the input matrix `x`
    # so that the first b dimensions are batch dimensions and the last two
    # are of shape [dimension, dimensions*number_of_samples]. Doing these
    # gymnastics allows us to do a batch_solve.
    #
    # After we're done with sqrt_solve (the batch operation) we need to undo
    # this reshaping so what we're left with is a Tensor partitionable by
    # sample, batch, event dimensions.

    # Complexity: O(nbk**2) since transpose must access every element.
    scale_sqrt_inv_x_sqrt = x_sqrt
    perm = array_ops.concat([math_ops.range(sample_ndims, ndims),
                             math_ops.range(0, sample_ndims)], 0)
    scale_sqrt_inv_x_sqrt = array_ops.transpose(scale_sqrt_inv_x_sqrt, perm)
    shape = array_ops.concat(
        (batch_shape, (math_ops.cast(
            self.dimension, dtype=dtypes.int32), -1)),
        0)
    scale_sqrt_inv_x_sqrt = array_ops.reshape(scale_sqrt_inv_x_sqrt, shape)

    # Complexity: O(nbM*k) where M is the complexity of the operator solving
    # a vector system. E.g., for LinearOperatorDiag, each solve is O(k), so
    # this complexity is O(nbk**2). For LinearOperatorLowerTriangular,
    # each solve is O(k**2) so this step has complexity O(nbk^3).
    scale_sqrt_inv_x_sqrt = self.scale_operator.solve(
        scale_sqrt_inv_x_sqrt)

    # Undo make batch-op ready.
    # Complexity: O(nbk**2)
    shape = array_ops.concat([batch_shape, event_shape, sample_shape], 0)
    scale_sqrt_inv_x_sqrt = array_ops.reshape(scale_sqrt_inv_x_sqrt, shape)
    perm = array_ops.concat([math_ops.range(ndims - sample_ndims, ndims),
                             math_ops.range(0, ndims - sample_ndims)], 0)
    scale_sqrt_inv_x_sqrt = array_ops.transpose(scale_sqrt_inv_x_sqrt, perm)

    # Write V = SS', X = LL'. Then:
    # tr[inv(V) X] = tr[inv(S)' inv(S) L L']
    #              = tr[inv(S) L L' inv(S)']
    #              = tr[(inv(S) L) (inv(S) L)']
    #              = sum_{ik} (inv(S) L)_{ik}**2
    # The second equality follows from the cyclic permutation property.
    # Complexity: O(nbk**2)
    trace_scale_inv_x = math_ops.reduce_sum(
        math_ops.square(scale_sqrt_inv_x_sqrt),
        axis=[-2, -1])

    # Complexity: O(nbk)
    half_log_det_x = math_ops.reduce_sum(
        math_ops.log(array_ops.matrix_diag_part(x_sqrt)),
        axis=[-1])

    # Complexity: O(nbk**2)
    log_prob = ((self.df - self.dimension - 1.) * half_log_det_x -
                0.5 * trace_scale_inv_x -
                self.log_normalization())

    # Set shape hints.
    # Try to merge what we know from the input then what we know from the
    # parameters of this distribution.
    if x.get_shape().ndims is not None:
      log_prob.set_shape(x.get_shape()[:-2])
    if (log_prob.get_shape().ndims is not None and
        self.batch_shape.ndims is not None and
        self.batch_shape.ndims > 0):
      log_prob.get_shape()[-self.batch_shape.ndims:].merge_with(
          self.batch_shape)

    return log_prob

  def _prob(self, x):
    return math_ops.exp(self._log_prob(x))

  def _entropy(self):
    half_dp1 = 0.5 * self.dimension + 0.5
    half_df = 0.5 * self.df
    return (self.dimension * (half_df + half_dp1 * math.log(2.)) +
            2 * half_dp1 * self.scale_operator.log_abs_determinant() +
            self._multi_lgamma(half_df, self.dimension) +
            (half_dp1 - half_df) * self._multi_digamma(half_df, self.dimension))

  def _mean(self):
    if self.cholesky_input_output_matrices:
      return (math_ops.sqrt(self.df)
              * self.scale_operator.to_dense())
    return self.df * self._square_scale_operator()

  def _variance(self):
    x = math_ops.sqrt(self.df) * self._square_scale_operator()
    d = array_ops.expand_dims(array_ops.matrix_diag_part(x), -1)
    v = math_ops.square(x) + math_ops.matmul(d, d, adjoint_b=True)
    if self.cholesky_input_output_matrices:
      return linalg_ops.cholesky(v)
    return v

  def _stddev(self):
    if self.cholesky_input_output_matrices:
      raise ValueError(
          "Computing std. dev. when is cholesky_input_output_matrices=True "
          "does not make sense.")
    return linalg_ops.cholesky(self.variance())

  def _mode(self):
    s = self.df - self.dimension - 1.
    s = array_ops.where(
        math_ops.less(s, 0.),
        constant_op.constant(float("NaN"), dtype=self.dtype, name="nan"),
        s)
    if self.cholesky_input_output_matrices:
      return math_ops.sqrt(s) * self.scale_operator.to_dense()
    return s * self._square_scale_operator()

  def mean_log_det(self, name="mean_log_det"):
    """Computes E[log(det(X))] under this Wishart distribution."""
    with self._name_scope(name):
      return (self._multi_digamma(0.5 * self.df, self.dimension) +
              self.dimension * math.log(2.) +
              2 * self.scale_operator.log_abs_determinant())

  def log_normalization(self, name="log_normalization"):
    """Computes the log normalizing constant, log(Z)."""
    with self._name_scope(name):
      return (self.df * self.scale_operator.log_abs_determinant() +
              0.5 * self.df * self.dimension * math.log(2.) +
              self._multi_lgamma(0.5 * self.df, self.dimension))

  def _multi_gamma_sequence(self, a, p, name="multi_gamma_sequence"):
    """Creates sequence used in multivariate (di)gamma; shape = shape(a)+[p]."""
    with self._name_scope(name, values=[a, p]):
      # Linspace only takes scalars, so we'll add in the offset afterwards.
      seq = math_ops.linspace(
          constant_op.constant(0., dtype=self.dtype),
          0.5 - 0.5 * p,
          math_ops.cast(p, dtypes.int32))
      return seq + array_ops.expand_dims(a, [-1])

  def _multi_lgamma(self, a, p, name="multi_lgamma"):
    """Computes the log multivariate gamma function; log(Gamma_p(a))."""
    with self._name_scope(name, values=[a, p]):
      seq = self._multi_gamma_sequence(a, p)
      return (0.25 * p * (p - 1.) * math.log(math.pi) +
              math_ops.reduce_sum(math_ops.lgamma(seq),
                                  axis=[-1]))

  def _multi_digamma(self, a, p, name="multi_digamma"):
    """Computes the multivariate digamma function; Psi_p(a)."""
    with self._name_scope(name, values=[a, p]):
      seq = self._multi_gamma_sequence(a, p)
      return math_ops.reduce_sum(math_ops.digamma(seq),
                                 axis=[-1])


class WishartCholesky(_WishartLinearOperator):
  """The matrix Wishart distribution on positive definite matrices.

  This distribution is defined by a scalar degrees of freedom `df` and a
  lower, triangular Cholesky factor which characterizes the scale matrix.

  Using WishartCholesky is a constant-time improvement over WishartFull. It
  saves an O(nbk^3) operation, i.e., a matrix-product operation for sampling
  and a Cholesky factorization in log_prob. For most use-cases it often saves
  another O(nbk^3) operation since most uses of Wishart will also use the
  Cholesky factorization.

  #### Mathematical Details

  The probability density function (pdf) is,

  ```none
  pdf(X; df, scale) = det(X)**(0.5 (df-k-1)) exp(-0.5 tr[inv(scale) X]) / Z
  Z = 2**(0.5 df k) |det(scale)|**(0.5 df) Gamma_k(0.5 df)
  ```

  where:
  * `df >= k` denotes the degrees of freedom,
  * `scale` is a symmetric, positive definite, `k x k` matrix,
  * `Z` is the normalizing constant, and,
  * `Gamma_k` is the [multivariate Gamma function](
    https://en.wikipedia.org/wiki/Multivariate_gamma_function).


  #### Examples

  ```python
  import tensorflow_probability as tfp
  tfd = tfp.distributions

  # Initialize a single 3x3 Wishart with Cholesky factored scale matrix and 5
  # degrees-of-freedom.(*)
  df = 5
  chol_scale = tf.cholesky(...)  # Shape is [3, 3].
  dist = tfd.WishartCholesky(df=df, scale=chol_scale)

  # Evaluate this on an observation in R^3, returning a scalar.
  x = ...  # A 3x3 positive definite matrix.
  dist.prob(x)  # Shape is [], a scalar.

  # Evaluate this on a two observations, each in R^{3x3}, returning a length two
  # Tensor.
  x = [x0, x1]  # Shape is [2, 3, 3].
  dist.prob(x)  # Shape is [2].

  # Initialize two 3x3 Wisharts with Cholesky factored scale matrices.
  df = [5, 4]
  chol_scale = tf.cholesky(...)  # Shape is [2, 3, 3].
  dist = tfd.WishartCholesky(df=df, scale=chol_scale)

  # Evaluate this on four observations.
  x = [[x0, x1], [x2, x3]]  # Shape is [2, 2, 3, 3].
  dist.prob(x)  # Shape is [2, 2].

  # (*) - To efficiently create a trainable covariance matrix, see the example
  #   in tfp.distributions.matrix_diag_transform.
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
               df,
               scale,
               cholesky_input_output_matrices=False,
               validate_args=False,
               allow_nan_stats=True,
               name="WishartCholesky"):
    """Construct Wishart distributions.

    Args:
      df: `float` or `double` `Tensor`. Degrees of freedom, must be greater than
        or equal to dimension of the scale matrix.
      scale: `float` or `double` `Tensor`. The Cholesky factorization of
        the symmetric positive definite scale matrix of the distribution.
      cholesky_input_output_matrices: Python `bool`. Any function which whose
        input or output is a matrix assumes the input is Cholesky and returns a
        Cholesky factored matrix. Example `log_prob` input takes a Cholesky and
        `sample_n` returns a Cholesky when
        `cholesky_input_output_matrices=True`.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
        result is undefined. When `False`, an exception is raised if one or
        more of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.
    """
    parameters = dict(locals())
    with ops.name_scope(name, values=[scale]) as name:
      with ops.name_scope("init", values=[scale]):
        scale = ops.convert_to_tensor(scale)
        if validate_args:
          scale = control_flow_ops.with_dependencies([
              check_ops.assert_positive(
                  array_ops.matrix_diag_part(scale),
                  message="scale must be positive definite"),
              check_ops.assert_equal(
                  array_ops.shape(scale)[-1],
                  array_ops.shape(scale)[-2],
                  message="scale must be square")
          ] if validate_args else [], scale)

      super(WishartCholesky, self).__init__(
          df=df,
          scale_operator=linalg.LinearOperatorLowerTriangular(
              tril=scale,
              is_non_singular=True,
              is_positive_definite=True,
              is_square=True),
          cholesky_input_output_matrices=cholesky_input_output_matrices,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          name=name)
    self._parameters = parameters


class WishartFull(_WishartLinearOperator):
  """The matrix Wishart distribution on positive definite matrices.

  This distribution is defined by a scalar degrees of freedom `df` and a
  symmetric, positive definite scale matrix.

  Evaluation of the pdf, determinant, and sampling are all `O(k^3)` operations
  where `(k, k)` is the event space shape.

  #### Mathematical Details

  The probability density function (pdf) is,

  ```none
  pdf(X; df, scale) = det(X)**(0.5 (df-k-1)) exp(-0.5 tr[inv(scale) X]) / Z
  Z = 2**(0.5 df k) |det(scale)|**(0.5 df) Gamma_k(0.5 df)
  ```

  where:
  * `df >= k` denotes the degrees of freedom,
  * `scale` is a symmetric, positive definite, `k x k` matrix,
  * `Z` is the normalizing constant, and,
  * `Gamma_k` is the [multivariate Gamma function](
    https://en.wikipedia.org/wiki/Multivariate_gamma_function).

  #### Examples

  ```python
  import tensorflow_probability as tfp
  tfd = tfp.distributions

  # Initialize a single 3x3 Wishart with Full factored scale matrix and 5
  # degrees-of-freedom.(*)
  df = 5
  scale = ...  # Shape is [3, 3]; positive definite.
  dist = tfd.WishartFull(df=df, scale=scale)

  # Evaluate this on an observation in R^3, returning a scalar.
  x = ...  # A 3x3 positive definite matrix.
  dist.prob(x)  # Shape is [], a scalar.

  # Evaluate this on a two observations, each in R^{3x3}, returning a length two
  # Tensor.
  x = [x0, x1]  # Shape is [2, 3, 3].
  dist.prob(x)  # Shape is [2].

  # Initialize two 3x3 Wisharts with Full factored scale matrices.
  df = [5, 4]
  scale = ...  # Shape is [2, 3, 3].
  dist = tfd.WishartFull(df=df, scale=scale)

  # Evaluate this on four observations.
  x = [[x0, x1], [x2, x3]]  # Shape is [2, 2, 3, 3]; xi is positive definite.
  dist.prob(x)  # Shape is [2, 2].

  # (*) - To efficiently create a trainable covariance matrix, see the example
  #   in tfd.matrix_diag_transform.
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
               df,
               scale,
               cholesky_input_output_matrices=False,
               validate_args=False,
               allow_nan_stats=True,
               name="WishartFull"):
    """Construct Wishart distributions.

    Args:
      df: `float` or `double` `Tensor`. Degrees of freedom, must be greater than
        or equal to dimension of the scale matrix.
      scale: `float` or `double` `Tensor`. The symmetric positive definite
        scale matrix of the distribution.
      cholesky_input_output_matrices: Python `bool`. Any function which whose
        input or output is a matrix assumes the input is Cholesky and returns a
        Cholesky factored matrix. Example `log_prob` input takes a Cholesky and
        `sample_n` returns a Cholesky when
        `cholesky_input_output_matrices=True`.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
        result is undefined. When `False`, an exception is raised if one or
        more of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.
    """
    parameters = dict(locals())
    with ops.name_scope(name) as name:
      with ops.name_scope("init", values=[scale]):
        scale = ops.convert_to_tensor(scale)
        if validate_args:
          scale = distribution_util.assert_symmetric(scale)
        chol = linalg_ops.cholesky(scale)
        chol = control_flow_ops.with_dependencies([
            check_ops.assert_positive(array_ops.matrix_diag_part(chol))
        ] if validate_args else [], chol)
    super(WishartFull, self).__init__(
        df=df,
        scale_operator=linalg.LinearOperatorLowerTriangular(
            tril=chol,
            is_non_singular=True,
            is_positive_definite=True,
            is_square=True),
        cholesky_input_output_matrices=cholesky_input_output_matrices,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        name=name)
    self._parameters = parameters
