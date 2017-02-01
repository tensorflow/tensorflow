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
"""The RelaxedBernoulli distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.distributions.python.ops import bijector
from tensorflow.contrib.distributions.python.ops import distribution_util
from tensorflow.contrib.distributions.python.ops import logistic
from tensorflow.contrib.distributions.python.ops import transformed_distribution
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops


class _RelaxedBernoulli(transformed_distribution.TransformedDistribution):
  """RelaxedBernoulli distribution with temperature and logits parameters.

  The RelaxedBernoulli is a distribution over the unit interval (0,1), which
  continuously approximates a Bernoulli. The degree of approximation is
  controlled by a temperature: as the temperaturegoes to 0 the RelaxedBernoulli
  becomes discrete with a distribution described by the `logits` or `probs`
  parameters, as the temperature goes to infinity the RelaxedBernoulli
  becomes the constant distribution that is identically 0.5.

  The RelaxedBernoulli distribution is a reparameterized continuous
  distribution that is the binary special case of the RelaxedOneHotCategorical
  distribution (Maddison et al., 2016; Jang et al., 2016). For details on the
  binary special case see the appendix of Maddison et al. (2016) where it is
  referred to as BinConcrete. If you use this distribution, please cite both
  papers.

  Some care needs to be taken for loss functions that depend on the
  log-probability of RelaxedBernoullis, because computing log-probabilities of
  the RelaxedBernoulli can suffer from underflow issues. In many case loss
  functions such as these are invariant under invertible transformations of
  the random variables. The KL divergence, found in the variational autoencoder
  loss, is an example. Because RelaxedBernoullis are sampled by by a Logistic
  random variable followed by a `tf.sigmoid` op, one solution is to treat
  the Logistic as the random variable and `tf.sigmoid` as downstream. The
  KL divergences of two Logistics, which are always followed by a `tf.sigmoid`
  op, is equivalent to evaluating KL divergences of RelaxedBernoulli samples.
  See Maddison et al., 2016 for more details where this distribution is called
  the BinConcrete.

  #### Examples

  Creates three continuous distributions, which approximate 3 Bernoullis with
  probabilities (0.1, 0.5, 0.4). Samples from these distributions will be in
  the unit interval (0,1).

  ```python
  temperature = 0.5
  p = [0.1, 0.5, 0.4]
  dist = RelaxedBernoulli(temperature, probs=p)
  ```

  Creates three continuous distributions, which approximate 3 Bernoullis with
  logits (-2, 2, 0). Samples from these distributions will be in
  the unit interval (0,1).

  ```python
  temperature = 0.5
  logits = [-2, 2, 0]
  dist = RelaxedBernoulli(temperature, logits=logits)
  ```

  Creates three continuous distributions, whose sigmoid approximate 3 Bernoullis
  with logits (-2, 2, 0).

  ```python
  temperature = 0.5
  logits = [-2, 2, 0]
  dist = Logistic(logits/temperature, 1./temperature)
  samples = dist.sample()
  sigmoid_samples = tf.sigmoid(samples)
  # sigmoid_samples has the same distribution as samples from
  # RelaxedBernoulli(temperature, logits=logits)
  ```

  Creates three continuous distributions, which approximate 3 Bernoullis with
  logits (-2, 2, 0). Samples from these distributions will be in
  the unit interval (0,1). Because the temperature is very low, samples from
  these distributions are almost discrete, usually taking values very close to 0
  or 1.

  ```python
  temperature = 1e-5
  logits = [-2, 2, 0]
  dist = RelaxedBernoulli(temperature, logits=logits)
  ```

  Creates three continuous distributions, which approximate 3 Bernoullis with
  logits (-2, 2, 0). Samples from these distributions will be in
  the unit interval (0,1). Because the temperature is very high, samples from
  these distributions are usually close to the (0.5, 0.5, 0.5) vector.

  ```python
  temperature = 100
  logits = [-2, 2, 0]
  dist = RelaxedBernoulli(temperature, logits=logits)
  ```

  Chris J. Maddison, Andriy Mnih, and Yee Whye Teh. The Concrete Distribution:
  A Continuous Relaxation of Discrete Random Variables. 2016.

  Eric Jang, Shixiang Gu, and Ben Poole. Categorical Reparameterization with
  Gumbel-Softmax. 2016.
  """

  def __init__(self,
               temperature,
               logits=None,
               probs=None,
               validate_args=False,
               allow_nan_stats=True,
               name="RelaxedBernoulli"):
    """Construct RelaxedBernoulli distributions.

    Args:
      temperature: An 0-D `Tensor`, representing the temperature
        of a set of RelaxedBernoulli distributions. The temperature should be
        positive.
      logits: An N-D `Tensor` representing the log-odds
        of a positive event. Each entry in the `Tensor` parametrizes
        an independent RelaxedBernoulli distribution where the probability of an
        event is sigmoid(logits). Only one of `logits` or `probs` should be
        passed in.
      probs: An N-D `Tensor` representing the probability of a positive event.
        Each entry in the `Tensor` parameterizes an independent Bernoulli
        distribution. Only one of `logits` or `probs` should be passed in.
      validate_args: Python `Boolean`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `Boolean`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
        result is undefined.  When `False`, an exception is raised if one or
        more of the statistic's batch members are undefined.
      name: `String` name prefixed to Ops created by this class.

    Raises:
      ValueError: If both `probs` and `logits` are passed, or if neither.
    """
    parameters = locals()
    with ops.name_scope(name, values=[logits, probs, temperature]) as ns:
      with ops.control_dependencies([check_ops.assert_positive(temperature)]
                                    if validate_args else []):
        self._temperature = array_ops.identity(temperature, name="temperature")

      self._logits, self._probs = distribution_util.get_logits_and_probs(
          logits=logits, probs=probs, validate_args=validate_args)
      dist = logistic._Logistic(self._logits / self._temperature,
                                1. / self._temperature,
                                validate_args=validate_args,
                                allow_nan_stats=allow_nan_stats,
                                name=ns)
      self._parameters = parameters

    def inverse_log_det_jacobian_fn(y):
      return -math_ops.reduce_sum(math_ops.log(y) + math_ops.log1p(-y), -1)

    sigmoid_bijector = bijector.Inline(
        forward_fn=math_ops.sigmoid,
        inverse_fn=(lambda y: math_ops.log(y) - math_ops.log1p(-y)),
        inverse_log_det_jacobian_fn=inverse_log_det_jacobian_fn,
        name="sigmoid")
    super(_RelaxedBernoulli, self).__init__(dist,
                                            sigmoid_bijector,
                                            name=name)

  @staticmethod
  def _param_shapes(sample_shape):
    return {"logits": ops.convert_to_tensor(sample_shape, dtype=dtypes.int32)}

  @property
  def temperature(self):
    """Distribution parameter for the location."""
    return self._temperature

  @property
  def logits(self):
    """Log-odds of `1`."""
    return self._logits

  @property
  def probs(self):
    """Probability of `1`."""
    return self._probs
