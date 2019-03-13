# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Utilities for testing distributions and/or bijectors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import histogram_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables as variables_ops


__all__ = [
    "DiscreteScalarDistributionTestHelpers",
    "VectorDistributionTestHelpers",
]


class DiscreteScalarDistributionTestHelpers(object):
  """DiscreteScalarDistributionTestHelpers."""

  def run_test_sample_consistent_log_prob(
      self, sess_run_fn, dist,
      num_samples=int(1e5), num_threshold=int(1e3), seed=42,
      batch_size=None,
      rtol=1e-2, atol=0.):
    """Tests that sample/log_prob are consistent with each other.

    "Consistency" means that `sample` and `log_prob` correspond to the same
    distribution.

    Note: this test only verifies a necessary condition for consistency--it does
    does not verify sufficiency hence does not prove `sample`, `log_prob` truly
    are consistent.

    Args:
      sess_run_fn: Python `callable` taking `list`-like of `Tensor`s and
        returning a list of results after running one "step" of TensorFlow
        computation, typically set to `sess.run`.
      dist: Distribution instance or object which implements `sample`,
        `log_prob`, `event_shape_tensor` and `batch_shape_tensor`.
      num_samples: Python `int` scalar indicating the number of Monte-Carlo
        samples to draw from `dist`.
      num_threshold: Python `int` scalar indicating the number of samples a
        bucket must contain before being compared to the probability.
        Default value: 1e3; must be at least 1.
        Warning, set too high will cause test to falsely pass but setting too
        low will cause the test to falsely fail.
      seed: Python `int` indicating the seed to use when sampling from `dist`.
        In general it is not recommended to use `None` during a test as this
        increases the likelihood of spurious test failure.
      batch_size: Hint for unpacking result of samples. Default: `None` means
        batch_size is inferred.
      rtol: Python `float`-type indicating the admissible relative error between
        analytical and sample statistics.
      atol: Python `float`-type indicating the admissible absolute error between
        analytical and sample statistics.

    Raises:
      ValueError: if `num_threshold < 1`.
    """
    if num_threshold < 1:
      raise ValueError("num_threshold({}) must be at least 1.".format(
          num_threshold))
    # Histogram only supports vectors so we call it once per batch coordinate.
    y = dist.sample(num_samples, seed=seed)
    y = array_ops.reshape(y, shape=[num_samples, -1])
    if batch_size is None:
      batch_size = math_ops.reduce_prod(dist.batch_shape_tensor())
    batch_dims = array_ops.shape(dist.batch_shape_tensor())[0]
    edges_expanded_shape = 1 + array_ops.pad([-2], paddings=[[0, batch_dims]])
    for b, x in enumerate(array_ops.unstack(y, num=batch_size, axis=1)):
      counts, edges = self.histogram(x)
      edges = array_ops.reshape(edges, edges_expanded_shape)
      probs = math_ops.exp(dist.log_prob(edges))
      probs = array_ops.reshape(probs, shape=[-1, batch_size])[:, b]

      [counts_, probs_] = sess_run_fn([counts, probs])
      valid = counts_ > num_threshold
      probs_ = probs_[valid]
      counts_ = counts_[valid]
      self.assertAllClose(probs_, counts_ / num_samples,
                          rtol=rtol, atol=atol)

  def run_test_sample_consistent_mean_variance(
      self, sess_run_fn, dist,
      num_samples=int(1e5), seed=24,
      rtol=1e-2, atol=0.):
    """Tests that sample/mean/variance are consistent with each other.

    "Consistency" means that `sample`, `mean`, `variance`, etc all correspond
    to the same distribution.

    Args:
      sess_run_fn: Python `callable` taking `list`-like of `Tensor`s and
        returning a list of results after running one "step" of TensorFlow
        computation, typically set to `sess.run`.
      dist: Distribution instance or object which implements `sample`,
        `log_prob`, `event_shape_tensor` and `batch_shape_tensor`.
      num_samples: Python `int` scalar indicating the number of Monte-Carlo
        samples to draw from `dist`.
      seed: Python `int` indicating the seed to use when sampling from `dist`.
        In general it is not recommended to use `None` during a test as this
        increases the likelihood of spurious test failure.
      rtol: Python `float`-type indicating the admissible relative error between
        analytical and sample statistics.
      atol: Python `float`-type indicating the admissible absolute error between
        analytical and sample statistics.
    """
    x = math_ops.cast(dist.sample(num_samples, seed=seed), dtypes.float32)
    sample_mean = math_ops.reduce_mean(x, axis=0)
    sample_variance = math_ops.reduce_mean(
        math_ops.square(x - sample_mean), axis=0)
    sample_stddev = math_ops.sqrt(sample_variance)

    [
        sample_mean_,
        sample_variance_,
        sample_stddev_,
        mean_,
        variance_,
        stddev_
    ] = sess_run_fn([
        sample_mean,
        sample_variance,
        sample_stddev,
        dist.mean(),
        dist.variance(),
        dist.stddev(),
    ])

    self.assertAllClose(mean_, sample_mean_, rtol=rtol, atol=atol)
    self.assertAllClose(variance_, sample_variance_, rtol=rtol, atol=atol)
    self.assertAllClose(stddev_, sample_stddev_, rtol=rtol, atol=atol)

  def histogram(self, x, value_range=None, nbins=None, name=None):
    """Return histogram of values.

    Given the tensor `values`, this operation returns a rank 1 histogram
    counting the number of entries in `values` that fell into every bin. The
    bins are equal width and determined by the arguments `value_range` and
    `nbins`.

    Args:
      x: 1D numeric `Tensor` of items to count.
      value_range:  Shape [2] `Tensor`. `new_values <= value_range[0]` will be
        mapped to `hist[0]`, `values >= value_range[1]` will be mapped to
        `hist[-1]`. Must be same dtype as `x`.
      nbins:  Scalar `int32 Tensor`.  Number of histogram bins.
      name: Python `str` name prefixed to Ops created by this class.

    Returns:
      counts: 1D `Tensor` of counts, i.e.,
        `counts[i] = sum{ edges[i-1] <= values[j] < edges[i] : j }`.
      edges: 1D `Tensor` characterizing intervals used for counting.
    """
    with ops.name_scope(name, "histogram", [x]):
      x = ops.convert_to_tensor(x, name="x")
      if value_range is None:
        value_range = [math_ops.reduce_min(x), 1 + math_ops.reduce_max(x)]
      value_range = ops.convert_to_tensor(value_range, name="value_range")
      lo = value_range[0]
      hi = value_range[1]
      if nbins is None:
        nbins = math_ops.cast(hi - lo, dtypes.int32)
      delta = (hi - lo) / math_ops.cast(
          nbins, dtype=value_range.dtype.base_dtype)
      edges = math_ops.range(
          start=lo, limit=hi, delta=delta, dtype=x.dtype.base_dtype)
      counts = histogram_ops.histogram_fixed_width(
          x, value_range=value_range, nbins=nbins)
      return counts, edges


class VectorDistributionTestHelpers(object):
  """VectorDistributionTestHelpers helps test vector-event distributions."""

  def run_test_sample_consistent_log_prob(
      self,
      sess_run_fn,
      dist,
      num_samples=int(1e5),
      radius=1.,
      center=0.,
      seed=42,
      rtol=1e-2,
      atol=0.):
    """Tests that sample/log_prob are mutually consistent.

    "Consistency" means that `sample` and `log_prob` correspond to the same
    distribution.

    The idea of this test is to compute the Monte-Carlo estimate of the volume
    enclosed by a hypersphere, i.e., the volume of an `n`-ball. While we could
    choose an arbitrary function to integrate, the hypersphere's volume is nice
    because it is intuitive, has an easy analytical expression, and works for
    `dimensions > 1`.

    Technical Details:

    Observe that:

    ```none
    int_{R**d} dx [x in Ball(radius=r, center=c)]
    = E_{p(X)}[ [X in Ball(r, c)] / p(X) ]
    = lim_{m->infty} m**-1 sum_j^m [x[j] in Ball(r, c)] / p(x[j]),
        where x[j] ~iid p(X)
    ```

    Thus, for fixed `m`, the above is approximately true when `sample` and
    `log_prob` are mutually consistent.

    Furthermore, the above calculation has the analytical result:
    `pi**(d/2) r**d / Gamma(1 + d/2)`.

    Note: this test only verifies a necessary condition for consistency--it does
    does not verify sufficiency hence does not prove `sample`, `log_prob` truly
    are consistent. For this reason we recommend testing several different
    hyperspheres (assuming the hypersphere is supported by the distribution).
    Furthermore, we gain additional trust in this test when also tested `sample`
    against the first, second moments
    (`run_test_sample_consistent_mean_covariance`); it is probably unlikely that
    a "best-effort" implementation of `log_prob` would incorrectly pass both
    tests and for different hyperspheres.

    For a discussion on the analytical result (second-line) see:
      https://en.wikipedia.org/wiki/Volume_of_an_n-ball.

    For a discussion of importance sampling (fourth-line) see:
      https://en.wikipedia.org/wiki/Importance_sampling.

    Args:
      sess_run_fn: Python `callable` taking `list`-like of `Tensor`s and
        returning a list of results after running one "step" of TensorFlow
        computation, typically set to `sess.run`.
      dist: Distribution instance or object which implements `sample`,
        `log_prob`, `event_shape_tensor` and `batch_shape_tensor`. The
        distribution must have non-zero probability of sampling every point
        enclosed by the hypersphere.
      num_samples: Python `int` scalar indicating the number of Monte-Carlo
        samples to draw from `dist`.
      radius: Python `float`-type indicating the radius of the `n`-ball which
        we're computing the volume.
      center: Python floating-type vector (or scalar) indicating the center of
        the `n`-ball which we're computing the volume. When scalar, the value is
        broadcast to all event dims.
      seed: Python `int` indicating the seed to use when sampling from `dist`.
        In general it is not recommended to use `None` during a test as this
        increases the likelihood of spurious test failure.
      rtol: Python `float`-type indicating the admissible relative error between
        actual- and approximate-volumes.
      atol: Python `float`-type indicating the admissible absolute error between
        actual- and approximate-volumes. In general this should be zero since
        a typical radius implies a non-zero volume.
    """

    def actual_hypersphere_volume(dims, radius):
      # https://en.wikipedia.org/wiki/Volume_of_an_n-ball
      # Using tf.lgamma because we'd have to otherwise use SciPy which is not
      # a required dependency of core.
      radius = np.asarray(radius)
      dims = math_ops.cast(dims, dtype=radius.dtype)
      return math_ops.exp(
          (dims / 2.) * np.log(np.pi)
          - math_ops.lgamma(1. + dims / 2.)
          + dims * math_ops.log(radius))

    def is_in_ball(x, radius, center):
      return math_ops.cast(linalg_ops.norm(x - center, axis=-1) <= radius,
                           dtype=x.dtype)

    def monte_carlo_hypersphere_volume(dist, num_samples, radius, center):
      # https://en.wikipedia.org/wiki/Importance_sampling
      x = dist.sample(num_samples, seed=seed)
      x = array_ops.identity(x)  # Invalidate bijector cacheing.
      return math_ops.reduce_mean(
          math_ops.exp(-dist.log_prob(x)) * is_in_ball(x, radius, center),
          axis=0)

    # Build graph.
    with ops.name_scope(
        "run_test_sample_consistent_log_prob",
        values=[num_samples, radius, center] + dist._graph_parents):  # pylint: disable=protected-access
      batch_shape = dist.batch_shape_tensor()
      actual_volume = actual_hypersphere_volume(
          dims=dist.event_shape_tensor()[0],
          radius=radius)
      sample_volume = monte_carlo_hypersphere_volume(
          dist,
          num_samples=num_samples,
          radius=radius,
          center=center)
      init_op = variables_ops.global_variables_initializer()

    # Execute graph.
    sess_run_fn(init_op)
    [batch_shape_, actual_volume_, sample_volume_] = sess_run_fn([
        batch_shape, actual_volume, sample_volume])

    # Check results.
    self.assertAllClose(np.tile(actual_volume_, reps=batch_shape_),
                        sample_volume_,
                        rtol=rtol, atol=atol)

  def run_test_sample_consistent_mean_covariance(
      self,
      sess_run_fn,
      dist,
      num_samples=int(1e5),
      seed=24,
      rtol=1e-2,
      atol=0.1,
      cov_rtol=None,
      cov_atol=None):
    """Tests that sample/mean/covariance are consistent with each other.

    "Consistency" means that `sample`, `mean`, `covariance`, etc all correspond
    to the same distribution.

    Args:
      sess_run_fn: Python `callable` taking `list`-like of `Tensor`s and
        returning a list of results after running one "step" of TensorFlow
        computation, typically set to `sess.run`.
      dist: Distribution instance or object which implements `sample`,
        `log_prob`, `event_shape_tensor` and `batch_shape_tensor`.
      num_samples: Python `int` scalar indicating the number of Monte-Carlo
        samples to draw from `dist`.
      seed: Python `int` indicating the seed to use when sampling from `dist`.
        In general it is not recommended to use `None` during a test as this
        increases the likelihood of spurious test failure.
      rtol: Python `float`-type indicating the admissible relative error between
        analytical and sample statistics.
      atol: Python `float`-type indicating the admissible absolute error between
        analytical and sample statistics.
      cov_rtol: Python `float`-type indicating the admissible relative error
        between analytical and sample covariance. Default: rtol.
      cov_atol: Python `float`-type indicating the admissible absolute error
        between analytical and sample covariance. Default: atol.
    """

    x = dist.sample(num_samples, seed=seed)
    sample_mean = math_ops.reduce_mean(x, axis=0)
    sample_covariance = math_ops.reduce_mean(
        _vec_outer_square(x - sample_mean), axis=0)
    sample_variance = array_ops.matrix_diag_part(sample_covariance)
    sample_stddev = math_ops.sqrt(sample_variance)

    [
        sample_mean_,
        sample_covariance_,
        sample_variance_,
        sample_stddev_,
        mean_,
        covariance_,
        variance_,
        stddev_
    ] = sess_run_fn([
        sample_mean,
        sample_covariance,
        sample_variance,
        sample_stddev,
        dist.mean(),
        dist.covariance(),
        dist.variance(),
        dist.stddev(),
    ])

    self.assertAllClose(mean_, sample_mean_, rtol=rtol, atol=atol)
    self.assertAllClose(covariance_, sample_covariance_,
                        rtol=cov_rtol or rtol,
                        atol=cov_atol or atol)
    self.assertAllClose(variance_, sample_variance_, rtol=rtol, atol=atol)
    self.assertAllClose(stddev_, sample_stddev_, rtol=rtol, atol=atol)


def _vec_outer_square(x, name=None):
  """Computes the outer-product of a vector, i.e., x.T x."""
  with ops.name_scope(name, "vec_osquare", [x]):
    return x[..., :, array_ops.newaxis] * x[..., array_ops.newaxis, :]
