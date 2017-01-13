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
"""The Normal (Gaussian) distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from tensorflow.contrib.bayesflow.python.ops import special_math
from tensorflow.contrib.distributions.python.ops import distribution
from tensorflow.contrib.distributions.python.ops import kullback_leibler
from tensorflow.contrib.framework.python.framework import tensor_util as contrib_tensor_util
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import random_ops


class Normal(distribution.Distribution):
  """The scalar Normal distribution with mean and stddev parameters mu, sigma.

  #### Mathematical details

  The PDF of this distribution is:

  ```f(x) = sqrt(1/(2*pi*sigma^2)) exp(-(x-mu)^2/(2*sigma^2))```

  #### Examples

  Examples of initialization of one or a batch of distributions.

  ```python
  # Define a single scalar Normal distribution.
  dist = tf.contrib.distributions.Normal(mu=0., sigma=3.)

  # Evaluate the cdf at 1, returning a scalar.
  dist.cdf(1.)

  # Define a batch of two scalar valued Normals.
  # The first has mean 1 and standard deviation 11, the second 2 and 22.
  dist = tf.contrib.distributions.Normal(mu=[1, 2.], sigma=[11, 22.])

  # Evaluate the pdf of the first distribution on 0, and the second on 1.5,
  # returning a length two tensor.
  dist.pdf([0, 1.5])

  # Get 3 samples, returning a 3 x 2 tensor.
  dist.sample([3])
  ```

  Arguments are broadcast when possible.

  ```python
  # Define a batch of two scalar valued Normals.
  # Both have mean 1, but different standard deviations.
  dist = tf.contrib.distributions.Normal(mu=1., sigma=[11, 22.])

  # Evaluate the pdf of both distributions on the same point, 3.0,
  # returning a length 2 tensor.
  dist.pdf(3.0)
  ```

  """

  def __init__(self,
               mu,
               sigma,
               validate_args=False,
               allow_nan_stats=True,
               name="Normal"):
    """Construct Normal distributions with mean and stddev `mu` and `sigma`.

    The parameters `mu` and `sigma` must be shaped in a way that supports
    broadcasting (e.g. `mu + sigma` is a valid operation).

    Args:
      mu: Floating point tensor, the means of the distribution(s).
      sigma: Floating point tensor, the stddevs of the distribution(s).
        sigma must contain only positive values.
      validate_args: `Boolean`, default `False`.  Whether to assert that
        `sigma > 0`. If `validate_args` is `False`, correct output is not
        guaranteed when input is invalid.
      allow_nan_stats: `Boolean`, default `True`.  If `False`, raise an
        exception if a statistic (e.g. mean/mode/etc...) is undefined for any
        batch member.  If `True`, batch members with valid parameters leading to
        undefined statistics will return NaN for this statistic.
      name: The name to give Ops created by the initializer.

    Raises:
      TypeError: if mu and sigma are different dtypes.
    """
    parameters = locals()
    parameters.pop("self")
    with ops.name_scope(name, values=[mu, sigma]) as ns:
      with ops.control_dependencies([check_ops.assert_positive(sigma)] if
                                    validate_args else []):
        self._mu = array_ops.identity(mu, name="mu")
        self._sigma = array_ops.identity(sigma, name="sigma")
        contrib_tensor_util.assert_same_float_dtype((self._mu, self._sigma))
    super(Normal, self).__init__(
        dtype=self._sigma.dtype,
        is_continuous=True,
        is_reparameterized=True,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        graph_parents=[self._mu, self._sigma],
        name=ns)

  @staticmethod
  def _param_shapes(sample_shape):
    return dict(
        zip(("mu", "sigma"), ([ops.convert_to_tensor(
            sample_shape, dtype=dtypes.int32)] * 2)))

  @property
  def mu(self):
    """Distribution parameter for the mean."""
    return self._mu

  @property
  def sigma(self):
    """Distribution parameter for standard deviation."""
    return self._sigma

  def _batch_shape(self):
    return array_ops.broadcast_dynamic_shape(
        array_ops.shape(self.mu), array_ops.shape(self.sigma))

  def _get_batch_shape(self):
    return array_ops.broadcast_static_shape(
        self._mu.get_shape(), self.sigma.get_shape())

  def _event_shape(self):
    return constant_op.constant([], dtype=dtypes.int32)

  def _get_event_shape(self):
    return tensor_shape.scalar()

  def _sample_n(self, n, seed=None):
    shape = array_ops.concat(([n], array_ops.shape(self.mean())), 0)
    sampled = random_ops.random_normal(
        shape=shape, mean=0, stddev=1, dtype=self.mu.dtype, seed=seed)
    return sampled * self.sigma + self.mu

  def _log_prob(self, x):
    return (-0.5 * math.log(2. * math.pi) - math_ops.log(self.sigma)
            -0.5 * math_ops.square(self._z(x)))

  def _prob(self, x):
    return math_ops.exp(self._log_prob(x))

  def _log_cdf(self, x):
    return special_math.log_ndtr(self._z(x))

  def _cdf(self, x):
    return special_math.ndtr(self._z(x))

  def _log_survival_function(self, x):
    return special_math.log_ndtr(-self._z(x))

  def _survival_function(self, x):
    return special_math.ndtr(-self._z(x))

  def _entropy(self):
    # Use broadcasting rules to calculate the full broadcast sigma.
    sigma = self.sigma * array_ops.ones_like(self.mu)
    return 0.5 * math.log(2. * math.pi * math.e) + math_ops.log(sigma)

  def _mean(self):
    return self.mu * array_ops.ones_like(self.sigma)

  def _variance(self):
    return math_ops.square(self.std())

  def _std(self):
    return self.sigma * array_ops.ones_like(self.mu)

  def _mode(self):
    return self._mean()

  def _z(self, x):
    """Standardize input `x` to a unit normal."""
    with ops.name_scope("standardize", values=[x]):
      return (x - self.mu) / self.sigma


class NormalWithSoftplusSigma(Normal):
  """Normal with softplus applied to `sigma`."""

  def __init__(self,
               mu,
               sigma,
               validate_args=False,
               allow_nan_stats=True,
               name="NormalWithSoftplusSigma"):
    parameters = locals()
    parameters.pop("self")
    with ops.name_scope(name, values=[sigma]) as ns:
      super(NormalWithSoftplusSigma, self).__init__(
          mu=mu,
          sigma=nn.softplus(sigma, name="softplus_sigma"),
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          name=ns)
    self._parameters = parameters


@kullback_leibler.RegisterKL(Normal, Normal)
def _kl_normal_normal(n_a, n_b, name=None):
  """Calculate the batched KL divergence KL(n_a || n_b) with n_a and n_b Normal.

  Args:
    n_a: instance of a Normal distribution object.
    n_b: instance of a Normal distribution object.
    name: (optional) Name to use for created operations.
      default is "kl_normal_normal".

  Returns:
    Batchwise KL(n_a || n_b)
  """
  with ops.name_scope(name, "kl_normal_normal", [n_a.mu, n_b.mu]):
    one = constant_op.constant(1, dtype=n_a.dtype)
    two = constant_op.constant(2, dtype=n_a.dtype)
    half = constant_op.constant(0.5, dtype=n_a.dtype)
    s_a_squared = math_ops.square(n_a.sigma)
    s_b_squared = math_ops.square(n_b.sigma)
    ratio = s_a_squared / s_b_squared
    return (math_ops.square(n_a.mu - n_b.mu) / (two * s_b_squared) +
            half * (ratio - one - math_ops.log(ratio)))
