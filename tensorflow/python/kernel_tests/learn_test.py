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

"""Tests for tf.learn."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

import tensorflow.python.platform  # pylint: disable=unused-import,g-bad-import-order

import numpy as np
import six
import tensorflow as tf


def assert_summary_scope(regexp):
  """Assert that all generated summaries match regexp."""
  for summary in tf.get_collection(tf.GraphKeys.SUMMARIES):
    tag = tf.unsupported.constant_value(summary.op.inputs[0])
    assert tag is not None, 'All summaries must have constant tags'

    tag = str(tag)
    # Tags in Python 3 erroneously acquire the 'b\'' prefix and '\'' suffix
    # This is a temporary measure meant to expediently overcome test failure.
    # TODO(danmane): Get rid of the extraneous prefix and suffix.
    if tag.startswith('b\'') and tag.endswith('\''):
      tag = tag[2:-1]

    assert isinstance(tag[0], six.string_types), tag[0]
    assert re.match(regexp, tag), "tag doesn't match %s: %s" % (regexp, tag)


class FullyConnectedTest(tf.test.TestCase):

  def setUp(self):
    tf.test.TestCase.setUp(self)
    tf.set_random_seed(1234)
    self.input = tf.constant([[1., 2., 3.], [-4., 15., -6.]])
    assert not tf.get_collection(tf.GraphKeys.SUMMARIES)

  def test_fully_connected_basic_use(self):
    output = tf.learn.fully_connected(self.input, 8, activation_fn=tf.nn.relu)

    with tf.Session() as sess:
      with self.assertRaises(tf.errors.FailedPreconditionError):
        sess.run(output)

      tf.initialize_all_variables().run()
      out_value = sess.run(output)

    self.assertEqual(output.get_shape().as_list(), [2, 8])
    self.assertTrue(np.all(out_value >= 0),
                    'Relu should have all values >= 0.')

    self.assertGreater(
        len(tf.get_collection(tf.GraphKeys.SUMMARIES)), 0,
        'Some summaries should have been added.')
    self.assertEqual(2,
                     len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)))
    self.assertEqual(0,
                     len(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)))
    assert_summary_scope('fully_connected')

  def test_relu_layer_basic_use(self):
    output = tf.learn.relu_layer(self.input, 8)

    with tf.Session() as sess:
      with self.assertRaises(tf.errors.FailedPreconditionError):
        sess.run(output)

      tf.initialize_all_variables().run()
      out_value = sess.run(output)

    self.assertEqual(output.get_shape().as_list(), [2, 8])
    self.assertTrue(np.all(out_value >= 0),
                    'Relu should have all values >= 0.')

    self.assertGreater(
        len(tf.get_collection(tf.GraphKeys.SUMMARIES)), 0,
        'Some summaries should have been added.')
    self.assertEqual(2,
                     len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)))
    self.assertEqual(0,
                     len(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)))
    assert_summary_scope('fully_connected')

  def test_relu6_layer_basic_use(self):
    output = tf.learn.relu6_layer(self.input, 8)

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

    self.assertGreater(
        len(tf.get_collection(tf.GraphKeys.SUMMARIES)), 0,
        'Some summaries should have been added.')
    self.assertEqual(2,
                     len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)))
    self.assertEqual(0,
                     len(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)))
    assert_summary_scope('fully_connected')

  def test_variable_reuse_with_scope(self):
    with tf.variable_scope('test') as vs:
      output1 = tf.learn.fully_connected(self.input,
                                         8,
                                         activation_fn=tf.nn.relu)
      output2 = tf.learn.fully_connected(self.input,
                                         8,
                                         activation_fn=tf.nn.relu)

    with tf.variable_scope(vs, reuse=True):
      output3 = tf.learn.fully_connected(self.input,
                                         8,
                                         activation_fn=tf.nn.relu)

    with tf.Session() as sess:
      tf.initialize_all_variables().run()
      out_value1, out_value2, out_value3 = sess.run([output1, output2, output3])

    self.assertFalse(np.allclose(out_value1, out_value2))
    self.assertAllClose(out_value1, out_value3)

  def test_variable_reuse_with_template(self):
    tmpl1 = tf.make_template('test',
                             tf.learn.fully_connected,
                             num_output_nodes=8)
    output1 = tmpl1(self.input)
    output2 = tmpl1(self.input)

    with tf.Session() as sess:
      tf.initialize_all_variables().run()
      out_value1, out_value2 = sess.run([output1, output2])
    self.assertAllClose(out_value1, out_value2)
    assert_summary_scope(r'test(_\d)?/fully_connected')

  def test_custom_initializers(self):
    output = tf.learn.fully_connected(self.input,
                                      2,
                                      activation_fn=tf.nn.relu,
                                      weight_init=tf.constant_initializer(2.0),
                                      bias_init=tf.constant_initializer(1.0))

    with tf.Session() as sess:
      tf.initialize_all_variables().run()
      out_value = sess.run(output)

    self.assertAllClose(np.array([[13.0, 13.0], [11.0, 11.0]]), out_value)

  def test_custom_collections(self):
    tf.learn.fully_connected(self.input,
                             2,
                             activation_fn=tf.nn.relu,
                             weight_collections=['unbiased'],
                             bias_collections=['biased'])

    self.assertEqual(1, len(tf.get_collection('unbiased')))
    self.assertEqual(1, len(tf.get_collection('biased')))

  def test_all_custom_collections(self):
    tf.learn.fully_connected(self.input,
                             2,
                             activation_fn=tf.nn.relu,
                             weight_collections=['unbiased', 'all'],
                             bias_collections=['biased', 'all'])

    self.assertEqual(1, len(tf.get_collection('unbiased')))
    self.assertEqual(1, len(tf.get_collection('biased')))
    self.assertEqual(2, len(tf.get_collection('all')))
    self.assertEqual(
        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES),
        tf.get_collection('all'))

  def test_no_summaries(self):
    tf.learn.fully_connected(self.input,
                             2,
                             activation_fn=tf.nn.relu,
                             create_summaries=False)
    self.assertEqual([], tf.get_collection(tf.GraphKeys.SUMMARIES))

  # Verify fix of a bug where no_summaries + activation_fn=None led to a
  # NoneType exception.
  def test_no_summaries_no_activation(self):
    tf.learn.fully_connected(self.input,
                             2,
                             activation_fn=None,
                             create_summaries=False)
    self.assertEqual([], tf.get_collection(tf.GraphKeys.SUMMARIES))

  def test_regularizer(self):
    cnt = [0]
    tensor = tf.constant(5.0)
    def test_fn(_):
      cnt[0] += 1
      return tensor

    tf.learn.fully_connected(self.input, 2, weight_regularizer=test_fn)

    self.assertEqual([tensor],
                     tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    self.assertEqual(1, cnt[0])

  def test_shape_enforcement(self):
    place = tf.placeholder(tf.float32)
    with self.assertRaises(ValueError):
      tf.learn.fully_connected(place, 8)
    tf.learn.fully_connected(place, 8, num_input_nodes=5)  # No error

    place.set_shape([None, None])
    with self.assertRaises(ValueError):
      tf.learn.fully_connected(place, 8)
    tf.learn.fully_connected(place, 8, num_input_nodes=5)  # No error

    place.set_shape([None, 6])
    tf.learn.fully_connected(place, 8)  # No error
    with self.assertRaises(ValueError):
      tf.learn.fully_connected(place, 8, num_input_nodes=5)

    place = tf.placeholder(tf.float32)
    place.set_shape([2, 6, 5])
    with self.assertRaises(ValueError):
      tf.learn.fully_connected(place, 8)

  def test_no_bias(self):
    tf.learn.fully_connected(self.input, 2, bias_init=None)

    self.assertEqual(1,
                     len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)))


class Convolution2dTest(tf.test.TestCase):

  def setUp(self):
    tf.test.TestCase.setUp(self)
    tf.set_random_seed(1234)
    self.input = tf.constant(np.arange(2 * 3 * 3 * 4).reshape(
        [2, 3, 3, 4]).astype(np.float32))
    assert not tf.get_collection(tf.GraphKeys.SUMMARIES)

  def test_basic_use(self):
    output = tf.learn.convolution2d(self.input, 8, (3, 3),
                                    activation_fn=tf.nn.relu)

    with tf.Session() as sess:
      with self.assertRaises(tf.errors.FailedPreconditionError):
        sess.run(output)

      tf.initialize_all_variables().run()
      out_value = sess.run(output)

    self.assertEqual(output.get_shape().as_list(), [2, 3, 3, 8])
    self.assertTrue(np.all(out_value >= 0),
                    'Relu should have capped all values.')

    self.assertGreater(
        len(tf.get_collection(tf.GraphKeys.SUMMARIES)), 0,
        'Some summaries should have been added.')
    self.assertEqual(2,
                     len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)))
    self.assertEqual(0,
                     len(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)))
    assert_summary_scope('convolution2d')

  def test_variable_reuse_with_scope(self):
    with tf.variable_scope('test') as vs:
      output1 = tf.learn.convolution2d(self.input,
                                       8, (3, 3),
                                       activation_fn=tf.nn.relu)
      output2 = tf.learn.convolution2d(self.input,
                                       8, (3, 3),
                                       activation_fn=tf.nn.relu)

    with tf.variable_scope(vs, reuse=True):
      output3 = tf.learn.convolution2d(self.input,
                                       8, (3, 3),
                                       activation_fn=tf.nn.relu)

    with tf.Session() as sess:
      tf.initialize_all_variables().run()
      out_value1, out_value2, out_value3 = sess.run([output1, output2, output3])

    self.assertFalse(np.allclose(out_value1, out_value2))
    self.assertAllClose(out_value1, out_value3)

  def test_variable_reuse_with_template(self):
    tmpl1 = tf.make_template('test',
                             tf.learn.convolution2d,
                             kernel_size=(3, 3),
                             num_output_channels=8)
    output1 = tmpl1(self.input)
    output2 = tmpl1(self.input)

    with tf.Session() as sess:
      tf.initialize_all_variables().run()
      out_value1, out_value2 = sess.run([output1, output2])
    self.assertAllClose(out_value1, out_value2)
    assert_summary_scope(r'test(_\d)?/convolution2d')

  def test_custom_initializers(self):
    output = tf.learn.convolution2d(self.input,
                                    2,
                                    (3, 3),
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
    tf.learn.convolution2d(self.input,
                           2, (3, 3),
                           activation_fn=tf.nn.relu,
                           weight_collections=['unbiased'],
                           bias_collections=['biased'])

    self.assertEqual(1, len(tf.get_collection('unbiased')))
    self.assertEqual(1, len(tf.get_collection('biased')))

  def test_all_custom_collections(self):
    tf.learn.convolution2d(self.input,
                           2, (3, 3),
                           activation_fn=tf.nn.relu,
                           weight_collections=['unbiased', 'all'],
                           bias_collections=['biased', 'all'])

    self.assertEqual(1, len(tf.get_collection('unbiased')))
    self.assertEqual(1, len(tf.get_collection('biased')))
    self.assertEqual(2, len(tf.get_collection('all')))
    self.assertEqual(
        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES),
        tf.get_collection('all'))

  def test_no_summaries(self):
    tf.learn.convolution2d(self.input,
                           2, (3, 3),
                           activation_fn=tf.nn.relu,
                           create_summaries=False)
    self.assertEqual([], tf.get_collection(tf.GraphKeys.SUMMARIES))

  def test_regularizer(self):
    cnt = [0]
    tensor = tf.constant(5.0)
    def test_fn(_):
      cnt[0] += 1
      return tensor

    tf.learn.convolution2d(self.input, 2, (3, 3), weight_regularizer=test_fn)

    self.assertEqual([tensor],
                     tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    self.assertEqual(1, cnt[0])

  def test_shape_enforcement(self):
    place = tf.placeholder(tf.float32)
    with self.assertRaises(ValueError):
      tf.learn.convolution2d(place, 8, (3, 3))
    tf.learn.convolution2d(place, 8, (3, 3), num_input_channels=5)  # No error

    place.set_shape([None, None, None, None])
    with self.assertRaises(ValueError):
      tf.learn.convolution2d(place, 8, (3, 3))
    tf.learn.convolution2d(place, 8, (3, 3), num_input_channels=5)  # No error

    place.set_shape([None, None, None, 6])
    tf.learn.convolution2d(place, 8, (3, 3))  # No error
    with self.assertRaises(ValueError):
      tf.learn.convolution2d(place, 8, (3, 3), num_input_channels=5)

    place = tf.placeholder(tf.float32)
    place.set_shape([2, 6, 5])
    with self.assertRaises(ValueError):
      tf.learn.convolution2d(place, 8, (3, 3))

  def test_no_bias(self):
    tf.learn.convolution2d(self.input, 2, (3, 3), bias_init=None)

    self.assertEqual(1,
                     len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)))


class RegularizerTest(tf.test.TestCase):

  def test_l1(self):
    with self.assertRaises(ValueError):
      tf.learn.l1_regularizer(2.)
    with self.assertRaises(ValueError):
      tf.learn.l1_regularizer(-1.)
    with self.assertRaises(ValueError):
      tf.learn.l1_regularizer(0)

    self.assertIsNone(tf.learn.l1_regularizer(0.)(None))

    values = np.array([1., -1., 4., 2.])
    weights = tf.constant(values)
    with tf.Session() as sess:
      result = sess.run(tf.learn.l1_regularizer(.5)(weights))

    self.assertAllClose(np.abs(values).sum() * .5, result)

  def test_l2(self):
    with self.assertRaises(ValueError):
      tf.learn.l2_regularizer(2.)
    with self.assertRaises(ValueError):
      tf.learn.l2_regularizer(-1.)
    with self.assertRaises(ValueError):
      tf.learn.l2_regularizer(0)

    self.assertIsNone(tf.learn.l2_regularizer(0.)(None))

    values = np.array([1., -1., 4., 2.])
    weights = tf.constant(values)
    with tf.Session() as sess:
      result = sess.run(tf.learn.l2_regularizer(.42)(weights))

    self.assertAllClose(np.power(values, 2).sum() / 2.0 * .42, result)

if __name__ == '__main__':
  tf.test.main()
