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
"""Student's t distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops.distributions import distribution
from tensorflow.python.ops.distributions import util as distribution_util
from tensorflow.python.util.tf_export import tf_export


__all__ = [
    "StudentT",
    "StudentTWithAbsDfSoftplusScale",
]


@tf_export("distributions.StudentT")
class StudentT(distribution.Distribution):
  """Student's t-distribution.

  This distribution has parameters: degree of freedom `df`, location `loc`,
  and `scale`.

  #### Mathematical details

  The probability density function (pdf) is,

  ```none
  pdf(x; df, mu, sigma) = (1 + y**2 / df)**(-0.5 (df + 1)) / Z
  where,
  y = (x - mu) / sigma
  Z = abs(sigma) sqrt(df pi) Gamma(0.5 df) / Gamma(0.5 (df + 1))
  ```

  where:
  * `loc = mu`,
  * `scale = sigma`, and,
  * `Z` is the normalization constant, and,
  * `Gamma` is the [gamma function](
    https://en.wikipedia.org/wiki/Gamma_function).

  The StudentT distribution is a member of the [location-scale family](
  https://en.wikipedia.org/wiki/Location-scale_family), i.e., it can be
  constructed as,

  ```none
  X ~ StudentT(df, loc=0, scale=1)
  Y = loc + scale * X
  ```

  Notice that `scale` has semantics more similar to standard deviation than
  variance. However it is not actually the std. deviation; the Student's
  t-distribution std. dev. is `scale sqrt(df / (df - 2))` when `df > 2`.

  #### Examples

  Examples of initialization of one or a batch of distributions.

  ```python
  # Define a single scalar Student t distribution.
  single_dist = tf.distributions.StudentT(df=3)

  # Evaluate the pdf at 1, returning a scalar Tensor.
  single_dist.prob(1.)

  # Define a batch of two scalar valued Student t's.
  # The first has degrees of freedom 2, mean 1, and scale 11.
  # The second 3, 2 and 22.
  multi_dist = tf.distributions.StudentT(df=[2, 3],
                                                 loc=[1, 2.],
                                                 scale=[11, 22.])

  # Evaluate the pdf of the first distribution on 0, and the second on 1.5,
  # returning a length two tensor.
  multi_dist.prob([0, 1.5])

  # Get 3 samples, returning a 3 x 2 tensor.
  multi_dist.sample(3)
  ```

  Arguments are broadcast when possible.

  ```python
  # Define a batch of two Student's t distributions.
  # Both have df 2 and mean 1, but different scales.
  dist = tf.distributions.StudentT(df=2, loc=1, scale=[11, 22.])

  # Evaluate the pdf of both distributions on the same point, 3.0,
  # returning a length 2 tensor.
  dist.prob(3.0)
  ```

  """
  # pylint: enable=line-too-long

  def __init__(self,
               df,
               loc,
               scale,
               validate_args=False,
               allow_nan_stats=True,
               name="StudentT"):
    """Construct Student's t distributions.

    The distributions have degree of freedom `df`, mean `loc`, and scale
    `scale`.

    The parameters `df`, `loc`, and `scale` must be shaped in a way that
    supports broadcasting (e.g. `df + loc + scale` is a valid operation).

    Args:
      df: Floating-point `Tensor`. The degrees of freedom of the
        distribution(s). `df` must contain only positive values.
      loc: Floating-point `Tensor`. The mean(s) of the distribution(s).
      scale: Floating-point `Tensor`. The scaling factor(s) for the
        distribution(s). Note that `scale` is not technically the standard
        deviation of this distribution but has semantics more similar to
        standard deviation than variance.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`,
        statistics (e.g., mean, mode, variance) use the value "`NaN`" to
        indicate the result is undefined. When `False`, an exception is raised
        if one or more of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.

    Raises:
      TypeError: if loc and scale are different dtypes.
    """
    parameters = locals()
    with ops.name_scope(name, values=[df, loc, scale]):
      with ops.control_dependencies([check_ops.assert_positive(df)]
                                    if validate_args else []):
        self._df = array_ops.identity(df, name="df")
        self._loc = array_ops.identity(loc, name="loc")
        self._scale = array_ops.identity(scale, name="scale")
        check_ops.assert_same_float_dtype(
            (self._df, self._loc, self._scale))
    super(StudentT, self).__init__(
        dtype=self._scale.dtype,
        reparameterization_type=distribution.NOT_REPARAMETERIZED,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        graph_parents=[self._df, self._loc, self._scale],
        name=name)

  @staticmethod
  def _param_shapes(sample_shape):
    return dict(
        zip(("df", "loc", "scale"), (
            [ops.convert_to_tensor(
                sample_shape, dtype=dtypes.int32)] * 3)))

  @property
  def df(self):
    """Degrees of freedom in these Student's t distribution(s)."""
    return self._df

  @property
  def loc(self):
    """Locations of these Student's t distribution(s)."""
    return self._loc

  @property
  def scale(self):
    """Scaling factors of these Student's t distribution(s)."""
    return self._scale

  def _batch_shape_tensor(self):
    return array_ops.broadcast_dynamic_shape(
        array_ops.shape(self.df),
        array_ops.broadcast_dynamic_shape(
            array_ops.shape(self.loc), array_ops.shape(self.scale)))

  def _batch_shape(self):
    return array_ops.broadcast_static_shape(
        array_ops.broadcast_static_shape(self.df.get_shape(),
                                         self.loc.get_shape()),
        self.scale.get_shape())

  def _event_shape_tensor(self):
    return constant_op.constant([], dtype=math_ops.int32)

  def _event_shape(self):
    return tensor_shape.scalar()

  def _sample_n(self, n, seed=None):
    # The sampling method comes from the fact that if:
    #   X ~ Normal(0, 1)
    #   Z ~ Chi2(df)
    #   Y = X / sqrt(Z / df)
    # then:
    #   Y ~ StudentT(df).
    shape = array_ops.concat([[n], self.batch_shape_tensor()], 0)
    normal_sample = random_ops.random_normal(shape, dtype=self.dtype, seed=seed)
    df = self.df * array_ops.ones(self.batch_shape_tensor(), dtype=self.dtype)
    gamma_sample = random_ops.random_gamma(
        [n],
        0.5 * df,
        beta=0.5,
        dtype=self.dtype,
        seed=distribution_util.gen_new_seed(seed, salt="student_t"))
    samples = normal_sample * math_ops.rsqrt(gamma_sample / df)
    return samples * self.scale + self.loc  # Abs(scale) not wanted.

  def _log_prob(self, x):
    return self._log_unnormalized_prob(x) - self._log_normalization()

  def _log_unnormalized_prob(self, x):
    y = (x - self.loc) / self.scale  # Abs(scale) superfluous.
    return -0.5 * (self.df + 1.) * math_ops.log1p(y**2. / self.df)

  def _log_normalization(self):
    return (math_ops.log(math_ops.abs(self.scale)) +
            0.5 * math_ops.log(self.df) +
            0.5 * np.log(np.pi) +
            math_ops.lgamma(0.5 * self.df) -
            math_ops.lgamma(0.5 * (self.df + 1.)))

  def _prob(self, x):
    return math_ops.exp(self._log_prob(x))

  def _cdf(self, x):
    # Take Abs(scale) to make subsequent where work correctly.
    y = (x - self.loc) / math_ops.abs(self.scale)
    x_t = self.df / (y**2. + self.df)
    neg_cdf = 0.5 * math_ops.betainc(0.5 * self.df, 0.5, x_t)
    return array_ops.where(math_ops.less(y, 0.), neg_cdf, 1. - neg_cdf)

  def _entropy(self):
    v = array_ops.ones(self.batch_shape_tensor(),
                       dtype=self.dtype)[..., array_ops.newaxis]
    u = v * self.df[..., array_ops.newaxis]
    beta_arg = array_ops.concat([u, v], -1) / 2.
    return (math_ops.log(math_ops.abs(self.scale)) +
            0.5 * math_ops.log(self.df) +
            special_math_ops.lbeta(beta_arg) +
            0.5 * (self.df + 1.) *
            (math_ops.digamma(0.5 * (self.df + 1.)) -
             math_ops.digamma(0.5 * self.df)))

  @distribution_util.AppendDocstring(
      """The mean of Student's T equals `loc` if `df > 1`, otherwise it is
      `NaN`. If `self.allow_nan_stats=True`, then an exception will be raised
      rather than returning `NaN`.""")
  def _mean(self):
    mean = self.loc * array_ops.ones(self.batch_shape_tensor(),
                                     dtype=self.dtype)
    if self.allow_nan_stats:
      nan = np.array(np.nan, dtype=self.dtype.as_numpy_dtype())
      return array_ops.where(
          math_ops.greater(
              self.df,
              array_ops.ones(self.batch_shape_tensor(), dtype=self.dtype)),
          mean,
          array_ops.fill(self.batch_shape_tensor(), nan, name="nan"))
    else:
      return control_flow_ops.with_dependencies(
          [
              check_ops.assert_less(
                  array_ops.ones([], dtype=self.dtype),
                  self.df,
                  message="mean not defined for components of df <= 1"),
          ],
          mean)

  @distribution_util.AppendDocstring("""
      The variance for Student's T equals

      ```
      df / (df - 2), when df > 2
      infinity, when 1 < df <= 2
      NaN, when df <= 1
      ```
      """)
  def _variance(self):
    # We need to put the tf.where inside the outer tf.where to ensure we never
    # hit a NaN in the gradient.
    denom = array_ops.where(math_ops.greater(self.df, 2.),
                            self.df - 2.,
                            array_ops.ones_like(self.df))
    # Abs(scale) superfluous.
    var = (array_ops.ones(self.batch_shape_tensor(), dtype=self.dtype) *
           math_ops.square(self.scale) * self.df / denom)
    # When 1 < df <= 2, variance is infinite.
    inf = np.array(np.inf, dtype=self.dtype.as_numpy_dtype())
    result_where_defined = array_ops.where(
        self.df > array_ops.fill(self.batch_shape_tensor(), 2.),
        var,
        array_ops.fill(self.batch_shape_tensor(), inf, name="inf"))

    if self.allow_nan_stats:
      nan = np.array(np.nan, dtype=self.dtype.as_numpy_dtype())
      return array_ops.where(
          math_ops.greater(
              self.df,
              array_ops.ones(self.batch_shape_tensor(), dtype=self.dtype)),
          result_where_defined,
          array_ops.fill(self.batch_shape_tensor(), nan, name="nan"))
    else:
      return control_flow_ops.with_dependencies(
          [
              check_ops.assert_less(
                  array_ops.ones([], dtype=self.dtype),
                  self.df,
                  message="variance not defined for components of df <= 1"),
          ],
          result_where_defined)

  def _mode(self):
    return array_ops.identity(self.loc)


class StudentTWithAbsDfSoftplusScale(StudentT):
  """StudentT with `df = floor(abs(df))` and `scale = softplus(scale)`."""

  def __init__(self,
               df,
               loc,
               scale,
               validate_args=False,
               allow_nan_stats=True,
               name="StudentTWithAbsDfSoftplusScale"):
    parameters = locals()
    with ops.name_scope(name, values=[df, scale]):
      super(StudentTWithAbsDfSoftplusScale, self).__init__(
          df=math_ops.floor(math_ops.abs(df)),
          loc=loc,
          scale=nn.softplus(scale, name="softplus_scale"),
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          name=name)
    self._parameters = parameters
