# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for slim.nets.resnet_v2."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.contrib.slim.nets import resnet_utils
from tensorflow.contrib.slim.nets import resnet_v2

slim = tf.contrib.slim


def create_test_input(batch_size, height, width, channels):
  """Create test input tensor.

  Args:
    batch_size: The number of images per batch or `None` if unknown.
    height: The height of each image or `None` if unknown.
    width: The width of each image or `None` if unknown.
    channels: The number of channels per image or `None` if unknown.

  Returns:
    Either a placeholder `Tensor` of dimension
      [batch_size, height, width, channels] if any of the inputs are `None` or a
    constant `Tensor` with the mesh grid values along the spatial dimensions.
  """
  if None in [batch_size, height, width, channels]:
    return tf.placeholder(tf.float32, (batch_size, height, width, channels))
  else:
    return tf.to_float(
        np.tile(
            np.reshape(
                np.reshape(np.arange(height), [height, 1]) +
                np.reshape(np.arange(width), [1, width]),
                [1, height, width, 1]),
            [batch_size, 1, 1, channels]))


class ResnetUtilsTest(tf.test.TestCase):

  def testSubsampleThreeByThree(self):
    x = tf.reshape(tf.to_float(tf.range(9)), [1, 3, 3, 1])
    x = resnet_utils.subsample(x, 2)
    expected = tf.reshape(tf.constant([0, 2, 6, 8]), [1, 2, 2, 1])
    with self.test_session():
      self.assertAllClose(x.eval(), expected.eval())

  def testSubsampleFourByFour(self):
    x = tf.reshape(tf.to_float(tf.range(16)), [1, 4, 4, 1])
    x = resnet_utils.subsample(x, 2)
    expected = tf.reshape(tf.constant([0, 2, 8, 10]), [1, 2, 2, 1])
    with self.test_session():
      self.assertAllClose(x.eval(), expected.eval())

  def testConv2DSameEven(self):
    n, n2 = 4, 2

    # Input image.
    x = create_test_input(1, n, n, 1)

    # Convolution kernel.
    w = create_test_input(1, 3, 3, 1)
    w = tf.reshape(w, [3, 3, 1, 1])

    tf.get_variable('Conv/weights', initializer=w)
    tf.get_variable('Conv/biases', initializer=tf.zeros([1]))
    tf.get_variable_scope().reuse_variables()

    y1 = slim.conv2d(x, 1, [3, 3], stride=1, scope='Conv')
    y1_expected = tf.to_float([[14, 28, 43, 26],
                               [28, 48, 66, 37],
                               [43, 66, 84, 46],
                               [26, 37, 46, 22]])
    y1_expected = tf.reshape(y1_expected, [1, n, n, 1])

    y2 = resnet_utils.subsample(y1, 2)
    y2_expected = tf.to_float([[14, 43],
                               [43, 84]])
    y2_expected = tf.reshape(y2_expected, [1, n2, n2, 1])

    y3 = resnet_utils.conv2d_same(x, 1, 3, stride=2, scope='Conv')
    y3_expected = y2_expected

    y4 = slim.conv2d(x, 1, [3, 3], stride=2, scope='Conv')
    y4_expected = tf.to_float([[48, 37],
                               [37, 22]])
    y4_expected = tf.reshape(y4_expected, [1, n2, n2, 1])

    with self.test_session() as sess:
      sess.run(tf.initialize_all_variables())
      self.assertAllClose(y1.eval(), y1_expected.eval())
      self.assertAllClose(y2.eval(), y2_expected.eval())
      self.assertAllClose(y3.eval(), y3_expected.eval())
      self.assertAllClose(y4.eval(), y4_expected.eval())

  def testConv2DSameOdd(self):
    n, n2 = 5, 3

    # Input image.
    x = create_test_input(1, n, n, 1)

    # Convolution kernel.
    w = create_test_input(1, 3, 3, 1)
    w = tf.reshape(w, [3, 3, 1, 1])

    tf.get_variable('Conv/weights', initializer=w)
    tf.get_variable('Conv/biases', initializer=tf.zeros([1]))
    tf.get_variable_scope().reuse_variables()

    y1 = slim.conv2d(x, 1, [3, 3], stride=1, scope='Conv')
    y1_expected = tf.to_float([[14, 28, 43, 58, 34],
                               [28, 48, 66, 84, 46],
                               [43, 66, 84, 102, 55],
                               [58, 84, 102, 120, 64],
                               [34, 46, 55, 64, 30]])
    y1_expected = tf.reshape(y1_expected, [1, n, n, 1])

    y2 = resnet_utils.subsample(y1, 2)
    y2_expected = tf.to_float([[14, 43, 34],
                               [43, 84, 55],
                               [34, 55, 30]])
    y2_expected = tf.reshape(y2_expected, [1, n2, n2, 1])

    y3 = resnet_utils.conv2d_same(x, 1, 3, stride=2, scope='Conv')
    y3_expected = y2_expected

    y4 = slim.conv2d(x, 1, [3, 3], stride=2, scope='Conv')
    y4_expected = y2_expected

    with self.test_session() as sess:
      sess.run(tf.initialize_all_variables())
      self.assertAllClose(y1.eval(), y1_expected.eval())
      self.assertAllClose(y2.eval(), y2_expected.eval())
      self.assertAllClose(y3.eval(), y3_expected.eval())
      self.assertAllClose(y4.eval(), y4_expected.eval())

  def _resnet_plain(self, inputs, blocks, output_stride=None, scope=None):
    """A plain ResNet without extra layers before or after the ResNet blocks."""
    with tf.variable_scope(scope, values=[inputs]):
      with slim.arg_scope([slim.conv2d], outputs_collections='end_points'):
        net = resnet_utils.stack_blocks_dense(inputs, blocks, output_stride)
        end_points = slim.utils.convert_collection_to_dict('end_points')
        return net, end_points

  def testEndPointsV2(self):
    """Test the end points of a tiny v2 bottleneck network."""
    bottleneck = resnet_v2.bottleneck
    blocks = [resnet_utils.Block('block1', bottleneck, [(4, 1, 1), (4, 1, 2)]),
              resnet_utils.Block('block2', bottleneck, [(8, 2, 1), (8, 2, 1)])]
    inputs = create_test_input(2, 32, 16, 3)
    with slim.arg_scope(resnet_utils.resnet_arg_scope()):
      _, end_points = self._resnet_plain(inputs, blocks, scope='tiny')
    expected = [
        'tiny/block1/unit_1/bottleneck_v2/shortcut',
        'tiny/block1/unit_1/bottleneck_v2/conv1',
        'tiny/block1/unit_1/bottleneck_v2/conv2',
        'tiny/block1/unit_1/bottleneck_v2/conv3',
        'tiny/block1/unit_2/bottleneck_v2/conv1',
        'tiny/block1/unit_2/bottleneck_v2/conv2',
        'tiny/block1/unit_2/bottleneck_v2/conv3',
        'tiny/block2/unit_1/bottleneck_v2/shortcut',
        'tiny/block2/unit_1/bottleneck_v2/conv1',
        'tiny/block2/unit_1/bottleneck_v2/conv2',
        'tiny/block2/unit_1/bottleneck_v2/conv3',
        'tiny/block2/unit_2/bottleneck_v2/conv1',
        'tiny/block2/unit_2/bottleneck_v2/conv2',
        'tiny/block2/unit_2/bottleneck_v2/conv3']
    self.assertItemsEqual(expected, end_points)

  def _stack_blocks_nondense(self, net, blocks):
    """A simplified ResNet Block stacker without output stride control."""
    for block in blocks:
      with tf.variable_scope(block.scope, 'block', [net]):
        for i, unit in enumerate(block.args):
          depth, depth_bottleneck, stride = unit
          with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
            net = block.unit_fn(net,
                                depth=depth,
                                depth_bottleneck=depth_bottleneck,
                                stride=stride,
                                rate=1)
    return net

  def _atrousValues(self, bottleneck):
    """Verify the values of dense feature extraction by atrous convolution.

    Make sure that dense feature extraction by stack_blocks_dense() followed by
    subsampling gives identical results to feature extraction at the nominal
    network output stride using the simple self._stack_blocks_nondense() above.

    Args:
      bottleneck: The bottleneck function.
    """
    blocks = [
        resnet_utils.Block('block1', bottleneck, [(4, 1, 1), (4, 1, 2)]),
        resnet_utils.Block('block2', bottleneck, [(8, 2, 1), (8, 2, 2)]),
        resnet_utils.Block('block3', bottleneck, [(16, 4, 1), (16, 4, 2)]),
        resnet_utils.Block('block4', bottleneck, [(32, 8, 1), (32, 8, 1)])
    ]
    nominal_stride = 8

    # Test both odd and even input dimensions.
    height = 30
    width = 31
    with slim.arg_scope(resnet_utils.resnet_arg_scope(is_training=False)):
      for output_stride in [1, 2, 4, 8, None]:
        with tf.Graph().as_default():
          with self.test_session() as sess:
            tf.set_random_seed(0)
            inputs = create_test_input(1, height, width, 3)
            # Dense feature extraction followed by subsampling.
            output = resnet_utils.stack_blocks_dense(inputs,
                                                     blocks,
                                                     output_stride)
            if output_stride is None:
              factor = 1
            else:
              factor = nominal_stride // output_stride

            output = resnet_utils.subsample(output, factor)
            # Make the two networks use the same weights.
            tf.get_variable_scope().reuse_variables()
            # Feature extraction at the nominal network rate.
            expected = self._stack_blocks_nondense(inputs, blocks)
            sess.run(tf.initialize_all_variables())
            output, expected = sess.run([output, expected])
            self.assertAllClose(output, expected, atol=1e-4, rtol=1e-4)

  def testAtrousValuesBottleneck(self):
    self._atrousValues(resnet_v2.bottleneck)


class ResnetCompleteNetworkTest(tf.test.TestCase):
  """Tests with complete small ResNet v2 networks."""

  def _resnet_small(self,
                    inputs,
                    num_classes=None,
                    global_pool=True,
                    output_stride=None,
                    include_root_block=True,
                    reuse=None,
                    scope='resnet_v2_small'):
    """A shallow and thin ResNet v2 for faster tests."""
    bottleneck = resnet_v2.bottleneck
    blocks = [
        resnet_utils.Block(
            'block1', bottleneck, [(4, 1, 1)] * 2 + [(4, 1, 2)]),
        resnet_utils.Block(
            'block2', bottleneck, [(8, 2, 1)] * 2 + [(8, 2, 2)]),
        resnet_utils.Block(
            'block3', bottleneck, [(16, 4, 1)] * 2 + [(16, 4, 2)]),
        resnet_utils.Block(
            'block4', bottleneck, [(32, 8, 1)] * 2)]
    return resnet_v2.resnet_v2(inputs, blocks, num_classes, global_pool,
                               output_stride, include_root_block, reuse, scope)

  def testClassificationEndPoints(self):
    global_pool = True
    num_classes = 10
    inputs = create_test_input(2, 224, 224, 3)
    with slim.arg_scope(resnet_utils.resnet_arg_scope()):
      logits, end_points = self._resnet_small(inputs, num_classes, global_pool,
                                              scope='resnet')
    self.assertTrue(logits.op.name.startswith('resnet/logits'))
    self.assertListEqual(logits.get_shape().as_list(), [2, 1, 1, num_classes])
    self.assertTrue('predictions' in end_points)
    self.assertListEqual(end_points['predictions'].get_shape().as_list(),
                         [2, 1, 1, num_classes])

  def testClassificationShapes(self):
    global_pool = True
    num_classes = 10
    inputs = create_test_input(2, 224, 224, 3)
    with slim.arg_scope(resnet_utils.resnet_arg_scope()):
      _, end_points = self._resnet_small(inputs, num_classes, global_pool,
                                         scope='resnet')
      endpoint_to_shape = {
          'resnet/block1': [2, 28, 28, 4],
          'resnet/block2': [2, 14, 14, 8],
          'resnet/block3': [2, 7, 7, 16],
          'resnet/block4': [2, 7, 7, 32]}
      for endpoint in endpoint_to_shape:
        shape = endpoint_to_shape[endpoint]
        self.assertListEqual(end_points[endpoint].get_shape().as_list(), shape)

  def testFullyConvolutionalEndpointShapes(self):
    global_pool = False
    num_classes = 10
    inputs = create_test_input(2, 321, 321, 3)
    with slim.arg_scope(resnet_utils.resnet_arg_scope()):
      _, end_points = self._resnet_small(inputs, num_classes, global_pool,
                                         scope='resnet')
      endpoint_to_shape = {
          'resnet/block1': [2, 41, 41, 4],
          'resnet/block2': [2, 21, 21, 8],
          'resnet/block3': [2, 11, 11, 16],
          'resnet/block4': [2, 11, 11, 32]}
      for endpoint in endpoint_to_shape:
        shape = endpoint_to_shape[endpoint]
        self.assertListEqual(end_points[endpoint].get_shape().as_list(), shape)

  def testRootlessFullyConvolutionalEndpointShapes(self):
    global_pool = False
    num_classes = 10
    inputs = create_test_input(2, 128, 128, 3)
    with slim.arg_scope(resnet_utils.resnet_arg_scope()):
      _, end_points = self._resnet_small(inputs, num_classes, global_pool,
                                         include_root_block=False,
                                         scope='resnet')
      endpoint_to_shape = {
          'resnet/block1': [2, 64, 64, 4],
          'resnet/block2': [2, 32, 32, 8],
          'resnet/block3': [2, 16, 16, 16],
          'resnet/block4': [2, 16, 16, 32]}
      for endpoint in endpoint_to_shape:
        shape = endpoint_to_shape[endpoint]
        self.assertListEqual(end_points[endpoint].get_shape().as_list(), shape)

  def testAtrousFullyConvolutionalEndpointShapes(self):
    global_pool = False
    num_classes = 10
    output_stride = 8
    inputs = create_test_input(2, 321, 321, 3)
    with slim.arg_scope(resnet_utils.resnet_arg_scope()):
      _, end_points = self._resnet_small(inputs,
                                         num_classes,
                                         global_pool,
                                         output_stride=output_stride,
                                         scope='resnet')
      endpoint_to_shape = {
          'resnet/block1': [2, 41, 41, 4],
          'resnet/block2': [2, 41, 41, 8],
          'resnet/block3': [2, 41, 41, 16],
          'resnet/block4': [2, 41, 41, 32]}
      for endpoint in endpoint_to_shape:
        shape = endpoint_to_shape[endpoint]
        self.assertListEqual(end_points[endpoint].get_shape().as_list(), shape)

  def testAtrousFullyConvolutionalValues(self):
    """Verify dense feature extraction with atrous convolution."""
    nominal_stride = 32
    for output_stride in [4, 8, 16, 32, None]:
      with slim.arg_scope(resnet_utils.resnet_arg_scope(is_training=False)):
        with tf.Graph().as_default():
          with self.test_session() as sess:
            tf.set_random_seed(0)
            inputs = create_test_input(2, 81, 81, 3)
            # Dense feature extraction followed by subsampling.
            output, _ = self._resnet_small(inputs, None, global_pool=False,
                                           output_stride=output_stride)
            if output_stride is None:
              factor = 1
            else:
              factor = nominal_stride // output_stride
            output = resnet_utils.subsample(output, factor)
            # Make the two networks use the same weights.
            tf.get_variable_scope().reuse_variables()
            # Feature extraction at the nominal network rate.
            expected, _ = self._resnet_small(inputs, None, global_pool=False)
            sess.run(tf.initialize_all_variables())
            self.assertAllClose(output.eval(), expected.eval(),
                                atol=1e-4, rtol=1e-4)

  def testUnknownBatchSize(self):
    batch = 2
    height, width = 65, 65
    global_pool = True
    num_classes = 10
    inputs = create_test_input(None, height, width, 3)
    with slim.arg_scope(resnet_utils.resnet_arg_scope()):
      logits, _ = self._resnet_small(inputs, num_classes, global_pool,
                                     scope='resnet')
    self.assertTrue(logits.op.name.startswith('resnet/logits'))
    self.assertListEqual(logits.get_shape().as_list(),
                         [None, 1, 1, num_classes])
    images = create_test_input(batch, height, width, 3)
    with self.test_session() as sess:
      sess.run(tf.initialize_all_variables())
      output = sess.run(logits, {inputs: images.eval()})
      self.assertEqual(output.shape, (batch, 1, 1, num_classes))

  def testFullyConvolutionalUnknownHeightWidth(self):
    batch = 2
    height, width = 65, 65
    global_pool = False
    inputs = create_test_input(batch, None, None, 3)
    with slim.arg_scope(resnet_utils.resnet_arg_scope()):
      output, _ = self._resnet_small(inputs, None, global_pool)
    self.assertListEqual(output.get_shape().as_list(),
                         [batch, None, None, 32])
    images = create_test_input(batch, height, width, 3)
    with self.test_session() as sess:
      sess.run(tf.initialize_all_variables())
      output = sess.run(output, {inputs: images.eval()})
      self.assertEqual(output.shape, (batch, 3, 3, 32))

  def testAtrousFullyConvolutionalUnknownHeightWidth(self):
    batch = 2
    height, width = 65, 65
    global_pool = False
    output_stride = 8
    inputs = create_test_input(batch, None, None, 3)
    with slim.arg_scope(resnet_utils.resnet_arg_scope()):
      output, _ = self._resnet_small(inputs,
                                     None,
                                     global_pool,
                                     output_stride=output_stride)
    self.assertListEqual(output.get_shape().as_list(),
                         [batch, None, None, 32])
    images = create_test_input(batch, height, width, 3)
    with self.test_session() as sess:
      sess.run(tf.initialize_all_variables())
      output = sess.run(output, {inputs: images.eval()})
      self.assertEqual(output.shape, (batch, 9, 9, 32))


if __name__ == '__main__':
  tf.test.main()
