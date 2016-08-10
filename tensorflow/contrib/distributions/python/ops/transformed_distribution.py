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

from tensorflow.contrib.distributions.python.ops import distribution  # pylint: disable=line-too-long
from tensorflow.python.framework import ops


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
    base_dist=Normal(mu, sigma),
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
    if not issubclass(base_dist_cls, distribution.Distribution):
      raise TypeError("base_dist_cls must be a subclass of Distribution.")
    with ops.op_scope(base_dist_args.values(), name) as scope:
      self._name = scope
      self._base_dist = base_dist_cls(**base_dist_args)
    self._transform = transform
    self._inverse = inverse
    self._log_det_jacobian = log_det_jacobian
    self._inverse_cache = {}

  @property
  def name(self):
    return self._name

  @property
  def dtype(self):
    return self._base_dist.dtype

  def batch_shape(self, name="batch_shape"):
    """Batch dimensions of this instance as a 1-D int32 `Tensor`.

    The product of the dimensions of the `batch_shape` is the number of
    independent distributions of this kind the instance represents.

    Args:
      name: name to give to the op.

    Returns:
      `Tensor` `batch_shape`
    """
    with ops.name_scope(self.name):
      return self._base_dist.batch_shape(name)

  def get_batch_shape(self):
    """`TensorShape` available at graph construction time.

    Same meaning as `batch_shape`. May be only partially defined.

    Returns:
      batch shape
    """
    return self._base_dist.get_batch_shape()

  def event_shape(self, name="event_shape"):
    """Shape of a sample from a single distribution as a 1-D int32 `Tensor`.

    Args:
      name: name to give to the op.

    Returns:
      `Tensor` `event_shape`
    """
    with ops.name_scope(self.name):
      return self._base_dist.event_shape(name)

  def get_event_shape(self):
    """`TensorShape` available at graph construction time.

    Same meaning as `event_shape`. May be only partially defined.

    Returns:
      event shape
    """
    return self._base_dist.get_event_shape()

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

  def log_prob(self, y, name="log_prob"):
    """Log prob of observations in `y`.

    `log ( p(g(y)) / det|J(g(y))| )`, where `g` is the inverse of `transform`.

    Args:
      y: tensor of dtype `dtype`.
      name: The name to give this op.

    Returns:
      log_pdf: tensor of dtype `dtype`, the log-PDFs of `y`.

    Raises:
      ValueError: if `inverse` was not provided to the distribution and `y` was
          not returned from `sample`.
    """
    with ops.name_scope(self.name):
      with ops.op_scope([y], name):
        y = ops.convert_to_tensor(y)
        if y.dtype != self.dtype:
          raise TypeError("Input x dtype does not match dtype: %s vs. %s" %
                          (y.dtype, self.dtype))
        with ops.name_scope("inverse"):
          if y in self._inverse_cache:
            x = self._inverse_cache[y]
          elif self._inverse:
            x = self._inverse(y)
          else:
            raise ValueError("No inverse function exists and input `y` was not "
                             "returned from `sample`.")
        with ops.name_scope("log_det_jacobian"):
          log_det_jacobian = self._log_det_jacobian(x)
        return self._base_dist.log_prob(x) - log_det_jacobian

  def prob(self, y, name="prob"):
    """The prob of observations in `y`.

    `p(g(y)) / det|J(g(y))|`, where `g` is the inverse of `transform`.

    Args:
      y: `Tensor` of dtype `dtype`.
      name: The name to give this op.

    Returns:
      pdf: `Tensor` of dtype `dtype`, the pdf values of `y`.
    """
    return super(TransformedDistribution, self).prob(y, name=name)

  def sample_n(self, n, seed=None, name="sample_n"):
    """Sample `n` observations.

    Samples from the base distribution and then passes through the transform.

    Args:
      n: scalar, type int32, the number of observations to sample.
      seed: Python integer, the random seed.
      name: The name to give this op.

    Returns:
      samples: `[n, ...]`, a `Tensor` of `n` samples.
    """
    with ops.name_scope(self.name):
      with ops.name_scope(name):
        samples = self._base_dist.sample_n(n=n, seed=seed)
        with ops.name_scope("transform"):
          transformed = self._transform(samples)
          self._inverse_cache[transformed] = samples
          return transformed

  @property
  def is_reparameterized(self):
    return self._base_dist.is_reparameterized

  @property
  def allow_nan_stats(self):
    return self._base_dist.allow_nan_stats

  @property
  def validate_args(self):
    return self._base_dist.validate_args

  @property
  def is_continuous(self):
    return True
