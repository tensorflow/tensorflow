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
"""Tests for BatchNorm Bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib import distributions
from tensorflow.contrib.distributions.python.ops import test_util
from tensorflow.contrib.distributions.python.ops.bijectors.batch_normalization import BatchNormalization
from tensorflow.contrib.distributions.python.ops.bijectors.invert import Invert
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.layers import normalization
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.distributions import normal as normal_lib
from tensorflow.python.ops.distributions import transformed_distribution as transformed_distribution_lib
from tensorflow.python.platform import test
from tensorflow.python.training import adam


class BatchNormTest(test_util.VectorDistributionTestHelpers,
                    test.TestCase):

  def _reduction_axes(self, input_shape, event_dims):
    if isinstance(event_dims, int):
      event_dims = [event_dims]
    ndims = len(input_shape)
    # Convert event_dims to non-negative indexing.
    event_dims = list(event_dims)
    for idx, x in enumerate(event_dims):
      if x < 0:
        event_dims[idx] = ndims + x
    return tuple(i for i in range(ndims) if i not in event_dims)

  def testForwardInverse(self):
    """Tests forward and backward passes with different event shapes.

    input_shape: Tuple of shapes for input tensor.
    event_dims: Tuple of dimension indices that will be normalized.
    training: Boolean of whether bijector runs in training or inference mode.
    """
    params = [
        ((5*2, 4), [-1], False),
        ((5, 2, 4), [-1], False),
        ((5, 2, 4), [1, 2], False),
        ((5, 2, 4), [0, 1], False),
        ((5*2, 4), [-1], True),
        ((5, 2, 4), [-1], True),
        ((5, 2, 4), [1, 2], True),
        ((5, 2, 4), [0, 1], True)
    ]
    for input_shape, event_dims, training in params:
      x_ = np.arange(5 * 4 * 2).astype(np.float32).reshape(input_shape)
      with self.cached_session() as sess:
        x = constant_op.constant(x_)
        # When training, memorize the exact mean of the last
        # minibatch that it normalized (instead of moving average assignment).
        layer = normalization.BatchNormalization(
            axis=event_dims, momentum=0., epsilon=0.)
        batch_norm = BatchNormalization(
            batchnorm_layer=layer, training=training)
        # Minibatch statistics are saved only after norm_x has been computed.
        norm_x = batch_norm.inverse(x)
        with ops.control_dependencies(batch_norm.batchnorm.updates):
          moving_mean = array_ops.identity(batch_norm.batchnorm.moving_mean)
          moving_var = array_ops.identity(batch_norm.batchnorm.moving_variance)
          denorm_x = batch_norm.forward(array_ops.identity(norm_x))
          fldj = batch_norm.forward_log_det_jacobian(
              x, event_ndims=len(event_dims))
          # Use identity to invalidate cache.
          ildj = batch_norm.inverse_log_det_jacobian(
              array_ops.identity(denorm_x), event_ndims=len(event_dims))
        variables.global_variables_initializer().run()
        # Update variables.
        norm_x_ = sess.run(norm_x)
        [
            norm_x_,
            moving_mean_,
            moving_var_,
            denorm_x_,
            ildj_,
            fldj_,
        ] = sess.run([
            norm_x,
            moving_mean,
            moving_var,
            denorm_x,
            ildj,
            fldj,
        ])
        self.assertEqual("batch_normalization", batch_norm.name)

        reduction_axes = self._reduction_axes(input_shape, event_dims)
        keepdims = len(event_dims) > 1

        expected_batch_mean = np.mean(
            x_, axis=reduction_axes, keepdims=keepdims)
        expected_batch_var = np.var(x_, axis=reduction_axes, keepdims=keepdims)

        if training:
          # When training=True, values become normalized across batch dim and
          # original values are recovered after de-normalizing.
          zeros = np.zeros_like(norm_x_)
          self.assertAllClose(np.mean(zeros, axis=reduction_axes),
                              np.mean(norm_x_, axis=reduction_axes))

          self.assertAllClose(expected_batch_mean, moving_mean_)
          self.assertAllClose(expected_batch_var, moving_var_)
          self.assertAllClose(x_, denorm_x_, atol=1e-5)
          # Since moving statistics are set to batch statistics after
          # normalization, ildj and -fldj should match.
          self.assertAllClose(ildj_, -fldj_)
          # ildj is computed with minibatch statistics.
          expected_ildj = np.sum(np.log(1.) - .5 * np.log(
              expected_batch_var + batch_norm.batchnorm.epsilon))
          self.assertAllClose(expected_ildj, ildj_)
        else:
          # When training=False, moving_mean, moving_var remain at their
          # initialized values (0., 1.), resulting in no scale/shift (a small
          # shift occurs if epsilon > 0.)
          self.assertAllClose(x_, norm_x_)
          self.assertAllClose(x_, denorm_x_, atol=1e-5)
          # ildj is computed with saved statistics.
          expected_ildj = np.sum(
              np.log(1.) - .5 * np.log(1. + batch_norm.batchnorm.epsilon))
          self.assertAllClose(expected_ildj, ildj_)

  def testMaximumLikelihoodTraining(self):
    # Test Maximum Likelihood training with default bijector.
    with self.cached_session() as sess:
      base_dist = distributions.MultivariateNormalDiag(loc=[0., 0.])
      batch_norm = BatchNormalization(training=True)
      dist = transformed_distribution_lib.TransformedDistribution(
          distribution=base_dist,
          bijector=batch_norm)
      target_dist = distributions.MultivariateNormalDiag(loc=[1., 2.])
      target_samples = target_dist.sample(100)
      dist_samples = dist.sample(3000)
      loss = -math_ops.reduce_mean(dist.log_prob(target_samples))
      with ops.control_dependencies(batch_norm.batchnorm.updates):
        train_op = adam.AdamOptimizer(1e-2).minimize(loss)
        moving_mean = array_ops.identity(batch_norm.batchnorm.moving_mean)
        moving_var = array_ops.identity(batch_norm.batchnorm.moving_variance)
      variables.global_variables_initializer().run()
      for _ in range(3000):
        sess.run(train_op)
      [
          dist_samples_,
          moving_mean_,
          moving_var_
      ] = sess.run([
          dist_samples,
          moving_mean,
          moving_var
      ])
      self.assertAllClose([1., 2.], np.mean(dist_samples_, axis=0), atol=5e-2)
      self.assertAllClose([1., 2.], moving_mean_, atol=5e-2)
      self.assertAllClose([1., 1.], moving_var_, atol=5e-2)

  def testLogProb(self):
    with self.cached_session() as sess:
      layer = normalization.BatchNormalization(epsilon=0.)
      batch_norm = BatchNormalization(batchnorm_layer=layer, training=False)
      base_dist = distributions.MultivariateNormalDiag(loc=[0., 0.])
      dist = transformed_distribution_lib.TransformedDistribution(
          distribution=base_dist,
          bijector=batch_norm,
          validate_args=True)
      samples = dist.sample(int(1e5))
      # No volume distortion since training=False, bijector is initialized
      # to the identity transformation.
      base_log_prob = base_dist.log_prob(samples)
      dist_log_prob = dist.log_prob(samples)
      variables.global_variables_initializer().run()
      base_log_prob_, dist_log_prob_ = sess.run([base_log_prob, dist_log_prob])
      self.assertAllClose(base_log_prob_, dist_log_prob_)

  def testMutuallyConsistent(self):
    # BatchNorm bijector is only mutually consistent when training=False.
    dims = 4
    with self.cached_session() as sess:
      layer = normalization.BatchNormalization(epsilon=0.)
      batch_norm = BatchNormalization(batchnorm_layer=layer, training=False)
      dist = transformed_distribution_lib.TransformedDistribution(
          distribution=normal_lib.Normal(loc=0., scale=1.),
          bijector=batch_norm,
          event_shape=[dims],
          validate_args=True)
      self.run_test_sample_consistent_log_prob(
          sess_run_fn=sess.run,
          dist=dist,
          num_samples=int(1e5),
          radius=2.,
          center=0.,
          rtol=0.02)

  def testInvertMutuallyConsistent(self):
    # BatchNorm bijector is only mutually consistent when training=False.
    dims = 4
    with self.cached_session() as sess:
      layer = normalization.BatchNormalization(epsilon=0.)
      batch_norm = Invert(
          BatchNormalization(batchnorm_layer=layer, training=False))
      dist = transformed_distribution_lib.TransformedDistribution(
          distribution=normal_lib.Normal(loc=0., scale=1.),
          bijector=batch_norm,
          event_shape=[dims],
          validate_args=True)
      self.run_test_sample_consistent_log_prob(
          sess_run_fn=sess.run,
          dist=dist,
          num_samples=int(1e5),
          radius=2.,
          center=0.,
          rtol=0.02)


if __name__ == "__main__":
  test.main()
