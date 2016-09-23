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
"""A Transformed Distribution class."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.distributions.python.ops import distribution
from tensorflow.contrib.distributions.python.ops import distribution_util
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops


class TransformedDistribution(distribution.Distribution):
  """A Transformed Distribution.

  A Transformed Distribution models `p(y)` given a base distribution `p(x)`,
  an invertible transform, `y = f(x)`, and the determinant of the Jacobian of
  `f(x)`.

  Shapes, type, and reparameterization are taken from the base distribution.

  #### Mathematical details

  * `p(x)` - probability distribution for random variable X
  * `p(y)` - probability distribution for random variable Y
  * `f` - transform
  * `g` - inverse transform, `g(f(x)) = x`
  * `J(x)` - Jacobian of f(x)

  A Transformed Distribution exposes `sample` and `pdf`:

    * `sample`: `y = f(x)`, after drawing a sample of X.
    * `pdf`: `p(y) = p(x) / det|J(x)| = p(g(y)) / det|J(g(y))|`

  A simple example constructing a Log-Normal distribution from a Normal
  distribution:

  ```
  logit_normal = TransformedDistribution(
    base_dist_cls=tf.contrib.distributions.Normal,
    mu=mu,
    sigma=sigma,
    transform=lambda x: tf.sigmoid(x),
    inverse=lambda y: tf.log(y) - tf.log(1. - y),
    log_det_jacobian=(lambda x:
        tf.reduce_sum(tf.log(tf.sigmoid(x)) + tf.log(1. - tf.sigmoid(x)),
                      reduction_indices=[-1])))
    name="LogitNormalTransformedDistribution"
  )
  ```

  """

  def __init__(self,
               base_dist_cls,
               transform,
               inverse,
               log_det_jacobian,
               name="TransformedDistribution",
               **base_dist_args):
    """Construct a Transformed Distribution.

    Args:
      base_dist_cls: the base distribution class to transform. Must be a
          subclass of `Distribution`.
      transform: a callable that takes a `Tensor` sample from `base_dist` and
          returns a `Tensor` of the same shape and type. `x => y`.
      inverse: a callable that computes the inverse of transform. `y => x`. If
          None, users can only call `log_pdf` on values returned by `sample`.
      log_det_jacobian: a callable that takes a `Tensor` sample from `base_dist`
          and returns the log of the determinant of the Jacobian of `transform`.
      name: The name for the distribution.
      **base_dist_args: kwargs to pass on to dist_cls on construction.

    Raises:
      TypeError: if `base_dist_cls` is not a subclass of
          `Distribution`.
    """
    with ops.name_scope(name, values=base_dist_args.values()) as ns:
      self._base_dist = base_dist_cls(**base_dist_args)
      self._transform = transform
      self._inverse = inverse
      self._log_det_jacobian = log_det_jacobian
      self._inverse_cache = {}
      super(TransformedDistribution, self).__init__(
          dtype=self._base_dist.dtype,
          parameters={"base_dist_cls": base_dist_cls,
                      "transform": transform,
                      "inverse": inverse,
                      "log_det_jacobian": log_det_jacobian,
                      "base_dist_args": base_dist_args},
          is_continuous=self._base_dist.is_continuous,
          is_reparameterized=self._base_dist.is_reparameterized,
          validate_args=self._base_dist.validate_args,
          allow_nan_stats=self._base_dist.allow_nan_stats,
          name=ns)

  @property
  def base_distribution(self):
    """Base distribution, p(x)."""
    return self._base_dist

  @property
  def transform(self):
    """Function transforming x => y."""
    return self._transform

  @property
  def inverse(self):
    """Inverse function of transform, y => x."""
    return self._inverse

  @property
  def log_det_jacobian(self):
    """Function computing the log determinant of the Jacobian of transform."""
    return self._log_det_jacobian

  def _batch_shape(self):
    return self.base_distribution.batch_shape()

  def _get_batch_shape(self):
    return self.base_distribution.get_batch_shape()

  def _event_shape(self):
    return self.base_distribution.event_shape()

  def _get_event_shape(self):
    return self.base_distribution.get_event_shape()

  def _sample_n(self, n, seed=None):
    samples = self.base_distribution.sample_n(n=n, seed=seed)
    with ops.name_scope("transform"):
      transformed = self.transform(samples)
      self._inverse_cache[transformed] = samples
      return transformed

  def _log_prob(self, y):
    x = self._inverse_possibly_from_cache(y)
    with ops.name_scope("log_det_jacobian"):
      log_det_jacobian = self.log_det_jacobian(x)
    return self.base_distribution.log_prob(x) - log_det_jacobian

  def _prob(self, y):
    return math_ops.exp(self._log_prob(y))

  def _log_cdf(self, y):
    # If Y = f(X),
    # P[Y <= y] = P[f(X) <= y] = P[X <= f^{-1}(y)]
    x = self._inverse_possibly_from_cache(y)
    return self.base_distribution.log_cdf(x)

  def _cdf(self, y):
    x = self._inverse_possibly_from_cache(y)
    return self.base_distribution.cdf(x)

  def _log_survival_function(self, y):
    x = self._inverse_possibly_from_cache(y)
    return self.base_distribution.log_survival_function(x)

  def _survival_function(self, y):
    x = self._inverse_possibly_from_cache(y)
    return self.base_distribution.survival_function(x)

  def _inverse_possibly_from_cache(self, y):
    """Return `self._inverse(y)`, possibly using cached value."""
    y = ops.convert_to_tensor(y, name="y")
    with ops.name_scope("inverse"):
      if y in self._inverse_cache:
        x = self._inverse_cache[y]
      elif self.inverse:
        x = self.inverse(y)
      else:
        raise ValueError("No inverse function exists and input `y` was not "
                         "returned from `sample`.")
    return x


distribution_util.append_class_fun_doc(TransformedDistribution.batch_shape,
                                       doc_str="""

    The product of the dimensions of the `batch_shape` is the number of
    independent distributions of this kind the instance represents.

""")

distribution_util.append_class_fun_doc(TransformedDistribution.sample_n,
                                       doc_str="""

    Samples from the base distribution and then passes through the transform.
""")

distribution_util.append_class_fun_doc(TransformedDistribution.log_prob,
                                       doc_str="""

  `(log o p o g)(y) - (log o det o J o g)(y)`,
  where `g` is the inverse of `transform`.

  Raises:
    ValueError: if `inverse` was not provided to the distribution and `y` was
        not returned from `sample`.
""")

distribution_util.append_class_fun_doc(TransformedDistribution.prob,
                                       doc_str="""

  `p(g(y)) / det|J(g(y))|`, where `g` is the inverse of `transform`.

  Raises:
    ValueError: if `inverse` was not provided to the distribution and `y` was
        not returned from `sample`.
""")
