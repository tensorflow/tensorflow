# Copyright 2015 Google Inc. All Rights Reserved.
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
"""Tests for layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


# TODO(b/28426988): Add separate tests for non-legacy versions.
class FullyConnectedTest(tf.test.TestCase):

  def setUp(self):
    tf.test.TestCase.setUp(self)
    tf.set_random_seed(1234)
    self.input = tf.constant([[1., 2., 3.], [-4., 15., -6.]])
    self.input_3_dim_arr = [[[1., 1.1, 1.2],
                             [2., 2.1, 2.2],
                             [3., 3.1, 3.2],
                             [4., 4.1, 4.2]],
                            [[5., 5.1, 5.2],
                             [6., 6.1, 6.2],
                             [7., 7.1, 7.2],
                             [8., 8.1, 8.2]]]
    self.input_3_dim = tf.constant(self.input_3_dim_arr)

    assert not tf.get_collection(tf.GraphKeys.SUMMARIES)

  def _fully_connected_basic_use(self, x, num_output_units, expected_shape):
    output = tf.contrib.layers.legacy_fully_connected(x,
                                                      num_output_units,
                                                      activation_fn=tf.nn.relu)

    with tf.Session() as sess:
      with self.assertRaises(tf.errors.FailedPreconditionError):
        sess.run(output)

      tf.initialize_all_variables().run()
      out_value, shape_value = sess.run([output, tf.shape(output)])

    self.assertAllClose(shape_value, expected_shape)
    self.assertEquals(output.get_shape().as_list(), expected_shape)
    self.assertTrue(np.all(out_value >= 0),
                    'Relu should have all values >= 0.')

    self.assertEqual(2,
                     len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)))
    self.assertEqual(0,
                     len(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)))

  def test_fully_connected_basic_use(self):
    self._fully_connected_basic_use(self.input, 8, [2, 8])

  def test_fully_connected_basic_use_multi_dim(self):
    for last_dim in [1, 3]:
      self.setUp()
      self._fully_connected_basic_use(
          self.input_3_dim, last_dim, [2, 4, last_dim])

  def test_relu_layer_basic_use(self):
    output = tf.contrib.layers.legacy_relu(self.input, 8)

    with tf.Session() as sess:
      with self.assertRaises(tf.errors.FailedPreconditionError):
        sess.run(output)

      tf.initialize_all_variables().run()
      out_value = sess.run(output)

    self.assertEqual(output.get_shape().as_list(), [2, 8])
    self.assertTrue(np.all(out_value >= 0),
                    'Relu should have all values >= 0.')

    self.assertEqual(2,
                     len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)))
    self.assertEqual(0,
                     len(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)))

  def test_relu6_layer_basic_use(self):
    output = tf.contrib.layers.legacy_relu6(self.input, 8)

    with tf.Session() as sess:
      with self.assertRaises(tf.errors.FailedPreconditionError):
        sess.run(output)

      tf.initialize_all_variables().run()
      out_value = sess.run(output)

    self.assertEqual(output.get_shape().as_list(), [2, 8])
    self.assertTrue(np.all(out_value >= 0),
                    'Relu6 should have all values >= 0.')
    self.assertTrue(np.all(out_value <= 6),
                    'Relu6 should have all values <= 6.')

    self.assertEqual(2,
                     len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)))
    self.assertEqual(0,
                     len(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)))

  def test_variable_reuse_with_scope(self):
    with tf.variable_scope('test') as vs:
      output1 = tf.contrib.layers.legacy_relu(self.input, 8)
      output2 = tf.contrib.layers.legacy_relu(self.input, 8)

    with tf.variable_scope(vs, reuse=True):
      output3 = tf.contrib.layers.legacy_relu(self.input, 8)

    with tf.Session() as sess:
      tf.initialize_all_variables().run()
      out_value1, out_value2, out_value3 = sess.run([output1, output2, output3])

    self.assertFalse(np.allclose(out_value1, out_value2))
    self.assertAllClose(out_value1, out_value3)

  def test_variable_reuse_with_template(self):
    tmpl1 = tf.make_template('test',
                             tf.contrib.layers.legacy_fully_connected,
                             num_output_units=8)
    output1 = tmpl1(self.input)
    output2 = tmpl1(self.input)

    with tf.Session() as sess:
      tf.initialize_all_variables().run()
      out_value1, out_value2 = sess.run([output1, output2])
    self.assertAllClose(out_value1, out_value2)

  def _custom_initializers(self, x, num_output_units, expected_outputs):
    output = tf.contrib.layers.legacy_relu(
        x,
        num_output_units,
        weight_init=tf.constant_initializer(2.0),
        bias_init=tf.constant_initializer(1.0))

    with tf.Session() as sess:
      tf.initialize_all_variables().run()
      out_value = sess.run(output)

    self.assertAllClose(np.array(expected_outputs), out_value)

  def test_custom_initializers(self):
    self._custom_initializers(
        self.input, 2, [[13.0, 13.0], [11.0, 11.0]])

  def test_custom_initializers_multi_dim(self):
    self._custom_initializers(self.input_3_dim,
                              2,
                              [[[7.6, 7.6],
                                [13.6, 13.6],
                                [19.6, 19.6],
                                [25.6, 25.6]],
                               [[31.6, 31.6],
                                [37.6, 37.6],
                                [43.6, 43.6],
                                [49.6, 49.6]]])

  def test_custom_collections(self):
    tf.contrib.layers.legacy_relu(self.input,
                                  2,
                                  weight_collections=['unbiased'],
                                  bias_collections=['biased'],
                                  output_collections=['output'])

    self.assertEquals(1, len(tf.get_collection('unbiased')))
    self.assertEquals(1, len(tf.get_collection('biased')))
    self.assertEquals(1, len(tf.get_collection('output')))
    self.assertEquals(2, len(tf.get_collection(tf.GraphKeys.VARIABLES)))

  def test_all_custom_collections(self):
    tf.contrib.layers.legacy_relu(self.input,
                                  2,
                                  weight_collections=['unbiased', 'all'],
                                  bias_collections=['biased', 'all'])

    self.assertEquals(1, len(tf.get_collection('unbiased')))
    self.assertEquals(1, len(tf.get_collection('biased')))
    self.assertEquals(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES),
                      tf.get_collection('all'))

  def test_no_bias(self):
    tf.contrib.layers.legacy_relu(self.input, 2, bias_init=None)
    self.assertEqual(1, len(tf.get_collection(tf.GraphKeys.VARIABLES)))

  def test_no_activation(self):
    y = tf.contrib.layers.legacy_fully_connected(self.input, 2)
    self.assertEquals(2, len(tf.get_collection(tf.GraphKeys.VARIABLES)))
    self.assertEquals('BiasAdd', y.op.type)

  def test_no_activation_no_bias(self):
    y = tf.contrib.layers.legacy_fully_connected(self.input, 2, bias_init=None)
    self.assertEquals(1, len(tf.get_collection(tf.GraphKeys.VARIABLES)))
    self.assertEquals('MatMul', y.op.type)

  def test_regularizer(self):
    cnt = [0]
    tensor = tf.constant(5.0)
    def test_fn(_):
      cnt[0] += 1
      return tensor

    tf.contrib.layers.legacy_fully_connected(self.input,
                                             2,
                                             weight_regularizer=test_fn)

    self.assertEqual([tensor],
                     tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    self.assertEqual(1, cnt[0])

  def test_regularizer_with_multiple_variables(self):
    cnt = [0]
    tensor = tf.constant(5.0)
    def test_fn(_):
      cnt[0] += 1
      return tensor

    tf.contrib.layers.legacy_fully_connected(self.input,
                                             2,
                                             weight_regularizer=test_fn)
    tf.contrib.layers.legacy_fully_connected(self.input,
                                             2,
                                             weight_regularizer=test_fn)

    self.assertEqual([tensor, tensor],
                     tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    self.assertEqual(2, cnt[0])

  def test_regularizer_with_variable_reuse(self):
    cnt = [0]
    tensor = tf.constant(5.0)
    def test_fn(_):
      cnt[0] += 1
      return tensor

    with tf.variable_scope('test') as vs:
      tf.contrib.layers.legacy_fully_connected(self.input,
                                               2,
                                               weight_regularizer=test_fn)

    with tf.variable_scope(vs, reuse=True):
      tf.contrib.layers.legacy_fully_connected(self.input,
                                               2,
                                               weight_regularizer=test_fn)

    self.assertEqual([tensor],
                     tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    self.assertEqual(1, cnt[0])

  def test_empty_x_results_in_empty_output(self):
    # Empty x is common if someone masks their input with tf.boolean_mask in
    # order to drop missing entries, and in a particular batch all entries are
    # missing.
    with self.test_session():
      x = tf.constant([[]], shape=[0, 3])
      self.assertEqual(0, tf.size(x).eval())
      y = tf.contrib.layers.legacy_fully_connected(x,
                                                   2,
                                                   activation_fn=tf.nn.softmax)
      tf.initialize_all_variables().run()
      expected_y = np.array([]).reshape(0, 2)
      np.testing.assert_array_equal(expected_y, y.eval())

  def test_shapes_variable_first_dim(self):
    # first dimension is not known statically.
    x = tf.placeholder(tf.float32, shape=[None, 4, 3])
    y = tf.contrib.layers.legacy_fully_connected(x, 1)
    # in the output we still only know the 2nd and 3rd dimensions statically.
    self.assertEquals(y.get_shape().as_list(), [None, 4, 1])
    with self.test_session() as sess:
      tf.initialize_all_variables().run()
      # we can feed in input with first dimension 2
      shape_value = sess.run(tf.shape(y), feed_dict={x: self.input_3_dim_arr})
      self.assertAllClose(shape_value, [2, 4, 1])
      # we can feed in input with first dimension 1
      shape_value = sess.run(tf.shape(y),
                             feed_dict={x: [self.input_3_dim_arr[0]]})
      self.assertAllClose(shape_value, [1, 4, 1])
      # we cannot feed in input with inconsistent dimensions
      with self.assertRaises(ValueError):
        sess.run(tf.shape(y), feed_dict={x: [[[]]]})

  def _unknown_dim_invalid_input(self, last_dim):
    x = tf.placeholder(tf.float32, shape=[3, last_dim])
    tf.contrib.layers.legacy_fully_connected(x, 2, activation_fn=None)

  def test_known_dim_valid_input(self):
    self._unknown_dim_invalid_input(last_dim=3)

  def test_unknown_dim_invalid_input(self):
    with self.assertRaisesRegexp(
        ValueError, 'last dimension of x must be known but is None'):
      self._unknown_dim_invalid_input(last_dim=None)

  def test_1d_invalid_input(self):
    with self.test_session():
      with self.assertRaisesRegexp(ValueError,
                                   'rank of x must be at least 2 not: 1'):
        x = tf.constant([[]], shape=[0])
        tf.contrib.layers.legacy_fully_connected(x,
                                                 2,
                                                 activation_fn=tf.nn.softmax)


class Convolution2dTest(tf.test.TestCase):

  def setUp(self):
    tf.test.TestCase.setUp(self)
    tf.set_random_seed(1234)
    self.input = tf.constant(np.arange(2 * 3 * 3 * 4).reshape(
        [2, 3, 3, 4]).astype(np.float32))
    assert not tf.get_collection(tf.GraphKeys.SUMMARIES)

  def test_basic_use(self):
    output = tf.contrib.layers.legacy_convolution2d(self.input,
                                                    8, (3, 3),
                                                    activation_fn=tf.nn.relu)

    with tf.Session() as sess:
      with self.assertRaises(tf.errors.FailedPreconditionError):
        sess.run(output)

      tf.initialize_all_variables().run()
      out_value = sess.run(output)

    self.assertEqual(output.get_shape().as_list(), [2, 3, 3, 8])
    self.assertTrue(np.all(out_value >= 0),
                    'Relu should have capped all values.')
    self.assertEqual(2,
                     len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)))
    self.assertEqual(0,
                     len(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)))

  def test_variable_reuse_with_scope(self):
    with tf.variable_scope('test') as vs:
      output1 = tf.contrib.layers.legacy_convolution2d(self.input,
                                                       8, (3, 3),
                                                       activation_fn=tf.nn.relu)
      output2 = tf.contrib.layers.legacy_convolution2d(self.input,
                                                       8, (3, 3),
                                                       activation_fn=tf.nn.relu)

    with tf.variable_scope(vs, reuse=True):
      output3 = tf.contrib.layers.legacy_convolution2d(self.input,
                                                       8, (3, 3),
                                                       activation_fn=tf.nn.relu)

    with tf.Session() as sess:
      tf.initialize_all_variables().run()
      out_value1, out_value2, out_value3 = sess.run([output1, output2, output3])

    self.assertFalse(np.allclose(out_value1, out_value2))
    self.assertAllClose(out_value1, out_value3)

  def test_variable_reuse_with_template(self):
    tmpl1 = tf.make_template('test',
                             tf.contrib.layers.legacy_convolution2d,
                             kernel_size=(3, 3),
                             num_output_channels=8)
    output1 = tmpl1(self.input)
    output2 = tmpl1(self.input)

    with tf.Session() as sess:
      tf.initialize_all_variables().run()
      out_value1, out_value2 = sess.run([output1, output2])
    self.assertAllClose(out_value1, out_value2)

  def test_custom_initializers(self):
    output = tf.contrib.layers.legacy_convolution2d(
        self.input,
        2, (3, 3),
        activation_fn=tf.nn.relu,
        weight_init=tf.constant_initializer(2.0),
        bias_init=tf.constant_initializer(1.0),
        padding='VALID')

    with tf.Session() as sess:
      tf.initialize_all_variables().run()
      out_value = sess.run(output)

    self.assertAllClose(
        np.array([[[[1261., 1261.]]], [[[3853., 3853.]]]]), out_value)

  def test_custom_collections(self):
    tf.contrib.layers.legacy_convolution2d(self.input,
                                           2, (3, 3),
                                           activation_fn=tf.nn.relu,
                                           weight_collections=['unbiased'],
                                           bias_collections=['biased'])

    self.assertEquals(1, len(tf.get_collection('unbiased')))
    self.assertEquals(1, len(tf.get_collection('biased')))

  def test_all_custom_collections(self):
    tf.contrib.layers.legacy_convolution2d(
        self.input,
        2, (3, 3),
        activation_fn=tf.nn.relu,
        weight_collections=['unbiased', 'all'],
        bias_collections=['biased', 'all'])

    self.assertEquals(1, len(tf.get_collection('unbiased')))
    self.assertEquals(1, len(tf.get_collection('biased')))
    self.assertEquals(2, len(tf.get_collection('all')))
    self.assertEquals(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES),
                      tf.get_collection('all'))

  def test_regularizer(self):
    cnt = [0]
    tensor = tf.constant(5.0)
    def test_fn(_):
      cnt[0] += 1
      return tensor

    tf.contrib.layers.legacy_convolution2d(self.input,
                                           2, (3, 3),
                                           weight_regularizer=test_fn)

    self.assertEqual([tensor],
                     tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    self.assertEqual(1, cnt[0])

  def test_no_bias(self):
    tf.contrib.layers.legacy_convolution2d(self.input,
                                           2, (3, 3),
                                           bias_init=None)
    self.assertEqual(1,
                     len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)))


if __name__ == '__main__':
  tf.test.main()
