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
"""Tests for features.spectral_normalization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib import slim
from tensorflow.contrib.gan.python.features.python import spectral_normalization_impl as spectral_normalization
from tensorflow.contrib.layers.python.layers import layers as contrib_layers
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.keras.layers import convolutional as keras_convolutional
from tensorflow.python.keras.layers import core as keras_core
from tensorflow.python.layers import convolutional as layers_convolutional
from tensorflow.python.layers import core as layers_core
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class SpectralNormalizationTest(test.TestCase):

  def testComputeSpectralNorm(self):
    weights = variable_scope.get_variable(
        'w', dtype=dtypes.float32, shape=[2, 3, 50, 100])
    weights = math_ops.multiply(weights, 10.0)
    s = linalg_ops.svd(
        array_ops.reshape(weights, [-1, weights.shape[-1]]), compute_uv=False)
    true_sn = s[..., 0]
    estimated_sn = spectral_normalization.compute_spectral_norm(weights)

    with self.cached_session() as sess:
      sess.run(variables.global_variables_initializer())
      np_true_sn = sess.run(true_sn)
      for i in range(50):
        est = sess.run(estimated_sn)
        if i < 1:
          np_est_1 = est
        if i < 4:
          np_est_5 = est
        if i < 9:
          np_est_10 = est
        np_est_50 = est

      # Check that the estimate improves with more iterations.
      self.assertAlmostEqual(np_true_sn, np_est_50, 0)
      self.assertGreater(
          abs(np_true_sn - np_est_10), abs(np_true_sn - np_est_50))
      self.assertGreater(
          abs(np_true_sn - np_est_5), abs(np_true_sn - np_est_10))
      self.assertGreater(abs(np_true_sn - np_est_1), abs(np_true_sn - np_est_5))

  def testSpectralNormalize(self):
    weights = variable_scope.get_variable(
        'w', dtype=dtypes.float32, shape=[2, 3, 50, 100])
    weights = math_ops.multiply(weights, 10.0)
    normalized_weights = spectral_normalization.spectral_normalize(
        weights, power_iteration_rounds=1)

    unnormalized_sigma = linalg_ops.svd(
        array_ops.reshape(weights, [-1, weights.shape[-1]]),
        compute_uv=False)[..., 0]
    normalized_sigma = linalg_ops.svd(
        array_ops.reshape(normalized_weights, [-1, weights.shape[-1]]),
        compute_uv=False)[..., 0]

    with self.cached_session() as sess:
      sess.run(variables.global_variables_initializer())
      s0 = sess.run(unnormalized_sigma)

      for i in range(50):
        sigma = sess.run(normalized_sigma)
        if i < 1:
          s1 = sigma
        if i < 5:
          s5 = sigma
        if i < 10:
          s10 = sigma
        s50 = sigma

      self.assertAlmostEqual(1., s50, 0)
      self.assertGreater(abs(s10 - 1.), abs(s50 - 1.))
      self.assertGreater(abs(s5 - 1.), abs(s10 - 1.))
      self.assertGreater(abs(s1 - 1.), abs(s5 - 1.))
      self.assertGreater(abs(s0 - 1.), abs(s1 - 1.))

  def _testLayerHelper(self, build_layer_fn, w_shape, b_shape, is_keras=False):
    x = array_ops.placeholder(dtypes.float32, shape=[2, 10, 10, 3])

    w_initial = np.random.randn(*w_shape) * 10
    w_initializer = init_ops.constant_initializer(w_initial)
    b_initial = np.random.randn(*b_shape)
    b_initializer = init_ops.constant_initializer(b_initial)

    if is_keras:
      context_manager = spectral_normalization.keras_spectral_normalization()
    else:
      getter = spectral_normalization.spectral_normalization_custom_getter()
      context_manager = variable_scope.variable_scope('', custom_getter=getter)

    with context_manager:
      (net,
       expected_normalized_vars, expected_not_normalized_vars) = build_layer_fn(
           x, w_initializer, b_initializer)

    x_data = np.random.rand(*x.shape)

    with self.cached_session() as sess:
      sess.run(variables.global_variables_initializer())

      # Before running a forward pass we still expect the variables values to
      # differ from the initial value because of the normalizer.
      w_befores = []
      for name, var in expected_normalized_vars.items():
        w_before = sess.run(var)
        w_befores.append(w_before)
        self.assertFalse(
            np.allclose(w_initial, w_before),
            msg=('%s appears not to be normalized. Before: %s After: %s' %
                 (name, w_initial, w_before)))

      # Not true for the unnormalized variables.
      for name, var in expected_not_normalized_vars.items():
        b_before = sess.run(var)
        self.assertTrue(
            np.allclose(b_initial, b_before),
            msg=('%s appears to be unexpectedly normalized. '
                 'Before: %s After: %s' % (name, b_initial, b_before)))

      # Run a bunch of forward passes.
      for _ in range(1000):
        _ = sess.run(net, feed_dict={x: x_data})

      # We expect this to have improved the estimate of the spectral norm,
      # which should have changed the variable values and brought them close
      # to the true Spectral Normalized values.
      _, s, _ = np.linalg.svd(w_initial.reshape([-1, 3]))
      exactly_normalized = w_initial / s[0]
      for w_before, (name, var) in zip(w_befores,
                                       expected_normalized_vars.items()):
        w_after = sess.run(var)
        self.assertFalse(
            np.allclose(w_before, w_after, rtol=1e-8, atol=1e-8),
            msg=('%s did not improve over many iterations. '
                 'Before: %s After: %s' % (name, w_before, w_after)))
        self.assertAllClose(
            exactly_normalized,
            w_after,
            rtol=1e-4,
            atol=1e-4,
            msg=('Estimate of spectral norm for %s was innacurate. '
                 'Normalized matrices do not match.'
                 'Estimate: %s Actual: %s' % (name, w_after,
                                              exactly_normalized)))

  def testConv2D_Layers(self):

    def build_layer_fn(x, w_initializer, b_initializer):
      layer = layers_convolutional.Conv2D(
          filters=3,
          kernel_size=3,
          padding='same',
          kernel_initializer=w_initializer,
          bias_initializer=b_initializer)
      net = layer.apply(x)
      expected_normalized_vars = {'tf.layers.Conv2d.kernel': layer.kernel}
      expected_not_normalized_vars = {'tf.layers.Conv2d.bias': layer.bias}

      return net, expected_normalized_vars, expected_not_normalized_vars

    self._testLayerHelper(build_layer_fn, (3, 3, 3, 3), (3,))

  def testConv2D_ContribLayers(self):

    def build_layer_fn(x, w_initializer, b_initializer):
      var_collection = {
          'weights': ['CONTRIB_LAYERS_CONV2D_WEIGHTS'],
          'biases': ['CONTRIB_LAYERS_CONV2D_BIASES']
      }
      net = contrib_layers.conv2d(
          x,
          3,
          3,
          weights_initializer=w_initializer,
          biases_initializer=b_initializer,
          variables_collections=var_collection)
      weight_vars = ops.get_collection('CONTRIB_LAYERS_CONV2D_WEIGHTS')
      self.assertEquals(1, len(weight_vars))
      bias_vars = ops.get_collection('CONTRIB_LAYERS_CONV2D_BIASES')
      self.assertEquals(1, len(bias_vars))
      expected_normalized_vars = {
          'contrib.layers.conv2d.weights': weight_vars[0]
      }
      expected_not_normalized_vars = {
          'contrib.layers.conv2d.bias': bias_vars[0]
      }

      return net, expected_normalized_vars, expected_not_normalized_vars

    self._testLayerHelper(build_layer_fn, (3, 3, 3, 3), (3,))

  def testConv2D_Slim(self):

    def build_layer_fn(x, w_initializer, b_initializer):
      var_collection = {
          'weights': ['SLIM_CONV2D_WEIGHTS'],
          'biases': ['SLIM_CONV2D_BIASES']
      }
      net = slim.conv2d(
          x,
          3,
          3,
          weights_initializer=w_initializer,
          biases_initializer=b_initializer,
          variables_collections=var_collection)
      weight_vars = ops.get_collection('SLIM_CONV2D_WEIGHTS')
      self.assertEquals(1, len(weight_vars))
      bias_vars = ops.get_collection('SLIM_CONV2D_BIASES')
      self.assertEquals(1, len(bias_vars))
      expected_normalized_vars = {'slim.conv2d.weights': weight_vars[0]}
      expected_not_normalized_vars = {'slim.conv2d.bias': bias_vars[0]}

      return net, expected_normalized_vars, expected_not_normalized_vars

    self._testLayerHelper(build_layer_fn, (3, 3, 3, 3), (3,))

  def testConv2D_Keras(self):

    def build_layer_fn(x, w_initializer, b_initializer):
      layer = keras_convolutional.Conv2D(
          filters=3,
          kernel_size=3,
          padding='same',
          kernel_initializer=w_initializer,
          bias_initializer=b_initializer)
      net = layer.apply(x)
      expected_normalized_vars = {'keras.layers.Conv2d.kernel': layer.kernel}
      expected_not_normalized_vars = {'keras.layers.Conv2d.bias': layer.bias}

      return net, expected_normalized_vars, expected_not_normalized_vars

    self._testLayerHelper(build_layer_fn, (3, 3, 3, 3), (3,), is_keras=True)

  def testFC_Layers(self):

    def build_layer_fn(x, w_initializer, b_initializer):
      x = layers_core.Flatten()(x)
      layer = layers_core.Dense(
          units=3,
          kernel_initializer=w_initializer,
          bias_initializer=b_initializer)
      net = layer.apply(x)
      expected_normalized_vars = {'tf.layers.Dense.kernel': layer.kernel}
      expected_not_normalized_vars = {'tf.layers.Dense.bias': layer.bias}

      return net, expected_normalized_vars, expected_not_normalized_vars

    self._testLayerHelper(build_layer_fn, (300, 3), (3,))

  def testFC_ContribLayers(self):

    def build_layer_fn(x, w_initializer, b_initializer):
      var_collection = {
          'weights': ['CONTRIB_LAYERS_FC_WEIGHTS'],
          'biases': ['CONTRIB_LAYERS_FC_BIASES']
      }
      x = contrib_layers.flatten(x)
      net = contrib_layers.fully_connected(
          x,
          3,
          weights_initializer=w_initializer,
          biases_initializer=b_initializer,
          variables_collections=var_collection)
      weight_vars = ops.get_collection('CONTRIB_LAYERS_FC_WEIGHTS')
      self.assertEquals(1, len(weight_vars))
      bias_vars = ops.get_collection('CONTRIB_LAYERS_FC_BIASES')
      self.assertEquals(1, len(bias_vars))
      expected_normalized_vars = {
          'contrib.layers.fully_connected.weights': weight_vars[0]
      }
      expected_not_normalized_vars = {
          'contrib.layers.fully_connected.bias': bias_vars[0]
      }

      return net, expected_normalized_vars, expected_not_normalized_vars

    self._testLayerHelper(build_layer_fn, (300, 3), (3,))

  def testFC_Slim(self):

    def build_layer_fn(x, w_initializer, b_initializer):
      var_collection = {
          'weights': ['SLIM_FC_WEIGHTS'],
          'biases': ['SLIM_FC_BIASES']
      }
      x = slim.flatten(x)
      net = slim.fully_connected(
          x,
          3,
          weights_initializer=w_initializer,
          biases_initializer=b_initializer,
          variables_collections=var_collection)
      weight_vars = ops.get_collection('SLIM_FC_WEIGHTS')
      self.assertEquals(1, len(weight_vars))
      bias_vars = ops.get_collection('SLIM_FC_BIASES')
      self.assertEquals(1, len(bias_vars))
      expected_normalized_vars = {
          'slim.fully_connected.weights': weight_vars[0]
      }
      expected_not_normalized_vars = {'slim.fully_connected.bias': bias_vars[0]}

      return net, expected_normalized_vars, expected_not_normalized_vars

    self._testLayerHelper(build_layer_fn, (300, 3), (3,))

  def testFC_Keras(self):

    def build_layer_fn(x, w_initializer, b_initializer):
      x = keras_core.Flatten()(x)
      layer = keras_core.Dense(
          units=3,
          kernel_initializer=w_initializer,
          bias_initializer=b_initializer)
      net = layer.apply(x)
      expected_normalized_vars = {'keras.layers.Dense.kernel': layer.kernel}
      expected_not_normalized_vars = {'keras.layers.Dense.bias': layer.bias}

      return net, expected_normalized_vars, expected_not_normalized_vars

    self._testLayerHelper(build_layer_fn, (300, 3), (3,), is_keras=True)


if __name__ == '__main__':
  test.main()
