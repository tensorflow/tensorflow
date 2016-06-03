# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

# TODO(sguada) Expose tf.with_dependencies
from tensorflow.python.ops import control_flow_ops


class AvgPool2DTest(tf.test.TestCase):

  def testCreateAvgPool(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      output = tf.contrib.layers.avg_pool2d(images, [3, 3])
      self.assertEquals(output.op.name, 'AvgPool2D/AvgPool')
      self.assertListEqual(output.get_shape().as_list(), [5, 1, 1, 3])

  def testCollectOutputs(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      output = tf.contrib.layers.avg_pool2d(images, [3, 3],
                                            outputs_collections='outputs')
      self.assertEquals(('AvgPool2D', output),
                        tf.get_collection('outputs')[0])

  def testCreateSquareAvgPool(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      output = tf.contrib.layers.avg_pool2d(images, 3)
      self.assertEquals(output.op.name, 'AvgPool2D/AvgPool')
      self.assertListEqual(output.get_shape().as_list(), [5, 1, 1, 3])

  def testCreateAvgPoolWithScope(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      output = tf.contrib.layers.avg_pool2d(images, [3, 3], scope='pool1')
      self.assertEquals(output.op.name, 'pool1/AvgPool')

  def testCreateAvgPoolSAME(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      output = tf.contrib.layers.avg_pool2d(images, [3, 3], padding='SAME')
      self.assertListEqual(output.get_shape().as_list(), [5, 2, 2, 3])

  def testCreateAvgPoolStrideSAME(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      output = tf.contrib.layers.avg_pool2d(images, [3, 3], stride=1,
                                            padding='SAME')
      self.assertListEqual(output.get_shape().as_list(), [5, height, width, 3])

  def testGlobalAvgPool(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      output = tf.contrib.layers.avg_pool2d(images, images.get_shape()[1:3],
                                            stride=1)
      self.assertListEqual(output.get_shape().as_list(), [5, 1, 1, 3])


class BiasAddTest(tf.test.TestCase):

  def testCreate(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      output = tf.contrib.layers.bias_add(images)
      self.assertEquals(output.op.name, 'BiasAdd/BiasAdd')
      self.assertListEqual(output.get_shape().as_list(), [5, height, width, 3])

  def testCreateWithActivation(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      output = tf.contrib.layers.bias_add(images, activation_fn=tf.nn.relu)
      self.assertEquals(output.op.name, 'BiasAdd/Relu')
      self.assertListEqual(output.get_shape().as_list(), [5, height, width, 3])

  def testCreateDimensions(self):
    dims = (2, 3, 4)
    shape = [5, 2, 3, 4]
    with self.test_session():
      for d in dims:
        input_shape = shape[:d]
        inputs = tf.random_uniform(input_shape, seed=1)
        output = tf.contrib.layers.bias_add(inputs)
        self.assertListEqual(output.get_shape().as_list(), input_shape)
        biases = tf.contrib.framework.get_variables_by_name('biases')[-1]
        self.assertListEqual(biases.get_shape().as_list(), [input_shape[-1]])


class Convolution2dTest(tf.test.TestCase):

  def testCreateConv(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 4), seed=1)
      output = tf.contrib.layers.convolution2d(images, 32, [3, 3])
      self.assertEquals(output.op.name, 'Conv/Relu')
      self.assertListEqual(output.get_shape().as_list(), [5, height, width, 32])
      weights = tf.contrib.framework.get_variables_by_name('weights')[0]
      self.assertListEqual(weights.get_shape().as_list(), [3, 3, 4, 32])
      biases = tf.contrib.framework.get_variables_by_name('biases')[0]
      self.assertListEqual(biases.get_shape().as_list(), [32])

  def testCreateSquareConv(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      output = tf.contrib.layers.convolution2d(images, 32, 3)
      self.assertEquals(output.op.name, 'Conv/Relu')
      self.assertListEqual(output.get_shape().as_list(), [5, height, width, 32])

  def testCreateConvWithTensorShape(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      output = tf.contrib.layers.convolution2d(images, 32,
                                               images.get_shape()[1:3])
      self.assertEquals(output.op.name, 'Conv/Relu')
      self.assertListEqual(output.get_shape().as_list(), [5, height, width, 32])

  def testCreateFullyConv(self):
    height, width = 6, 6
    with self.test_session():
      images = tf.random_uniform((5, height, width, 32), seed=1)
      output = tf.contrib.layers.convolution2d(images, 64,
                                               images.get_shape()[1:3],
                                               padding='VALID')
      self.assertEquals(output.op.name, 'Conv/Relu')
      self.assertListEqual(output.get_shape().as_list(), [5, 1, 1, 64])
      biases = tf.contrib.framework.get_variables_by_name('biases')[0]
      self.assertListEqual(biases.get_shape().as_list(), [64])

  def testCreateVerticalConv(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 4), seed=1)
      output = tf.contrib.layers.convolution2d(images, 32, [3, 1])
      self.assertEquals(output.op.name, 'Conv/Relu')
      self.assertListEqual(output.get_shape().as_list(),
                           [5, height, width, 32])
      weights = tf.contrib.framework.get_variables_by_name('weights')[0]
      self.assertListEqual(weights.get_shape().as_list(), [3, 1, 4, 32])
      biases = tf.contrib.framework.get_variables_by_name('biases')[0]
      self.assertListEqual(biases.get_shape().as_list(), [32])

  def testCreateHorizontalConv(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 4), seed=1)
      output = tf.contrib.layers.convolution2d(images, 32, [1, 3])
      self.assertEquals(output.op.name, 'Conv/Relu')
      self.assertListEqual(output.get_shape().as_list(),
                           [5, height, width, 32])
      weights = tf.contrib.framework.get_variables_by_name('weights')[0]
      self.assertListEqual(weights.get_shape().as_list(), [1, 3, 4, 32])

  def testCreateConvWithStride(self):
    height, width = 6, 6
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      output = tf.contrib.layers.convolution2d(images, 32, [3, 3], stride=2)
      self.assertEquals(output.op.name, 'Conv/Relu')
      self.assertListEqual(output.get_shape().as_list(),
                           [5, height/2, width/2, 32])

  def testCreateConvCreatesWeightsAndBiasesVars(self):
    height, width = 3, 3
    images = tf.random_uniform((5, height, width, 3), seed=1)
    with self.test_session():
      self.assertFalse(tf.contrib.framework.get_variables('conv1/weights'))
      self.assertFalse(tf.contrib.framework.get_variables('conv1/biases'))
      tf.contrib.layers.convolution2d(images, 32, [3, 3], scope='conv1')
      self.assertTrue(tf.contrib.framework.get_variables('conv1/weights'))
      self.assertTrue(tf.contrib.framework.get_variables('conv1/biases'))

  def testCreateConvWithScope(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      output = tf.contrib.layers.convolution2d(images, 32, [3, 3],
                                               scope='conv1')
      self.assertEquals(output.op.name, 'conv1/Relu')

  def testCreateConvWithoutActivation(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      output = tf.contrib.layers.convolution2d(images, 32, [3, 3],
                                               activation_fn=None)
      self.assertEquals(output.op.name, 'Conv/BiasAdd')

  def testCreateConvValid(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      output = tf.contrib.layers.convolution2d(images, 32, [3, 3],
                                               padding='VALID')
      self.assertListEqual(output.get_shape().as_list(), [5, 1, 1, 32])

  def testCreateConvWithWD(self):
    height, width = 3, 3
    weight_decay = 0.01
    with self.test_session() as sess:
      images = tf.random_uniform((5, height, width, 3), seed=1)
      regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
      tf.contrib.layers.convolution2d(images, 32, [3, 3],
                                      weights_regularizer=regularizer)
      l2_loss = tf.nn.l2_loss(
          tf.contrib.framework.get_variables_by_name('weights')[0])
      wd = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)[0]
      self.assertEquals(wd.op.name,
                        'Conv/weights/Regularizer/l2_regularizer')
      sess.run(tf.initialize_all_variables())
      print(wd.eval())
      self.assertAlmostEqual(sess.run(wd), weight_decay * l2_loss.eval())

  def testCreateConvNoRegularizers(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      tf.contrib.layers.convolution2d(images, 32, [3, 3])
      self.assertEquals(
          tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), [])

  def testReuseVars(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      tf.contrib.layers.convolution2d(images, 32, [3, 3], scope='conv1')
      self.assertEquals(len(tf.contrib.framework.get_variables()), 2)
      tf.contrib.layers.convolution2d(images, 32, [3, 3], scope='conv1',
                                      reuse=True)
      self.assertEquals(len(tf.contrib.framework.get_variables()), 2)

  def testNonReuseVars(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      tf.contrib.layers.convolution2d(images, 32, [3, 3])
      self.assertEquals(len(tf.contrib.framework.get_variables()), 2)
      tf.contrib.layers.convolution2d(images, 32, [3, 3])
      self.assertEquals(len(tf.contrib.framework.get_variables()), 4)

  def testReuseConvWithWD(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      weight_decay = tf.contrib.layers.l2_regularizer(0.01)
      with tf.contrib.framework.arg_scope(
          [tf.contrib.layers.convolution2d],
          weights_regularizer=weight_decay):
        tf.contrib.layers.convolution2d(images, 32, [3, 3], scope='conv1')
        self.assertEquals(len(tf.contrib.framework.get_variables()), 2)
        self.assertEquals(
            len(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)), 1)
        tf.contrib.layers.convolution2d(images, 32, [3, 3], scope='conv1',
                                        reuse=True)
        self.assertEquals(len(tf.contrib.framework.get_variables()), 2)
        self.assertEquals(
            len(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)), 1)

  def testConvWithBatchNorm(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 32), seed=1)
      with tf.contrib.framework.arg_scope(
          [tf.contrib.layers.convolution2d],
          normalizer_fn=tf.contrib.layers.batch_norm,
          normalizer_params={'decay': 0.9}):
        net = tf.contrib.layers.convolution2d(images, 32, [3, 3])
        net = tf.contrib.layers.convolution2d(net, 32, [3, 3])
      self.assertEquals(len(tf.contrib.framework.get_variables()), 8)
      self.assertEquals(
          len(tf.contrib.framework.get_variables('Conv/BatchNorm')), 3)
      self.assertEquals(
          len(tf.contrib.framework.get_variables('Conv_1/BatchNorm')), 3)

  def testReuseConvWithBatchNorm(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 32), seed=1)
      with tf.contrib.framework.arg_scope(
          [tf.contrib.layers.convolution2d],
          normalizer_fn=tf.contrib.layers.batch_norm,
          normalizer_params={'decay': 0.9}):
        net = tf.contrib.layers.convolution2d(images, 32, [3, 3], scope='Conv')
        net = tf.contrib.layers.convolution2d(net, 32, [3, 3], scope='Conv',
                                              reuse=True)
      self.assertEquals(len(tf.contrib.framework.get_variables()), 4)
      self.assertEquals(
          len(tf.contrib.framework.get_variables('Conv/BatchNorm')), 3)
      self.assertEquals(
          len(tf.contrib.framework.get_variables('Conv_1/BatchNorm')), 0)


class DropoutTest(tf.test.TestCase):

  def testCreateDropout(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      output = tf.contrib.layers.dropout(images)
      self.assertEquals(output.op.name, 'Dropout/cond/Merge')
      output.get_shape().assert_is_compatible_with(images.get_shape())

  def testCollectOutputs(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      output = tf.contrib.layers.dropout(images, outputs_collections='outputs')
      self.assertEquals(('Dropout', output),
                        tf.get_collection('outputs')[0])

  def testDropout(self):
    height, width = 10, 10
    with self.test_session() as sess:
      images = tf.random_uniform((5, height, width, 3), seed=1, name='images')
      num_elem_initial = tf.reduce_mean(tf.to_float(images > 0))
      output = tf.contrib.layers.dropout(images)
      num_elem = tf.reduce_mean(tf.to_float(output > 0))
      sess.run(tf.initialize_all_variables())
      num_elem, num_elem_initial = sess.run([num_elem, num_elem_initial])
      self.assertLess(num_elem, num_elem_initial/2 + 0.1)
      self.assertGreater(num_elem, num_elem_initial/2 - 0.1)

  def testCreateDropoutNoTraining(self):
    height, width = 3, 3
    with self.test_session() as sess:
      images = tf.random_uniform((5, height, width, 3), seed=1, name='images')
      num_elem_initial = tf.reduce_mean(tf.to_float(images > 0))
      output = tf.contrib.layers.dropout(images, is_training=False)
      num_elem = tf.reduce_mean(tf.to_float(output > 0))
      sess.run(tf.initialize_all_variables())
      num_elem, num_elem_initial = sess.run([num_elem, num_elem_initial])
      self.assertEquals(num_elem, num_elem_initial)
      outputs, inputs = sess.run([output, images])
      self.assertAllClose(outputs, inputs)

  def testCreateFCFollowByDropout(self):
    height, width = 3, 3
    with self.test_session() as sess:
      images = tf.random_uniform((5, height, width, 3), seed=1, name='images')
      output = tf.contrib.layers.fully_connected(images, 50)
      num_elem_initial = tf.reduce_mean(tf.to_float(output > 0))
      output = tf.contrib.layers.dropout(output)
      num_elem = tf.reduce_mean(tf.to_float(output > 0))
      sess.run(tf.initialize_all_variables())
      num_elem, num_elem_initial = sess.run([num_elem, num_elem_initial])
      self.assertLess(num_elem, num_elem_initial/2 + 0.1)
      self.assertGreater(num_elem, num_elem_initial/2 - 0.1)

  def testCreateFCWithDropout(self):
    height, width = 3, 3
    with self.test_session() as sess:
      images = tf.random_uniform((5, height, width, 3), seed=1, name='images')
      output = tf.contrib.layers.fully_connected(
          images, 50, normalizer_fn=tf.contrib.layers.dropout)
      num_elem = tf.reduce_mean(tf.to_float(output > 0))
      sess.run(tf.initialize_all_variables())
      num_elem = sess.run(num_elem)
      self.assertLess(num_elem, 0.5)
      self.assertGreater(num_elem, 0.1)


class FlattenTest(tf.test.TestCase):

  def testCollectOutputs(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      output = tf.contrib.layers.flatten(images, outputs_collections='outputs')
      self.assertEquals(('Flatten', output),
                        tf.get_collection('outputs')[0])

  def testFlatten4D(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1, name='images')
      output = tf.contrib.layers.flatten(images)
      self.assertEquals(output.get_shape().num_elements(),
                        images.get_shape().num_elements())
      self.assertEqual(output.get_shape()[0], images.get_shape()[0])

  def testFlatten3D(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width), seed=1, name='images')
      output = tf.contrib.layers.flatten(images)
      self.assertEquals(output.get_shape().num_elements(),
                        images.get_shape().num_elements())
      self.assertEqual(output.get_shape()[0], images.get_shape()[0])

  def testFlattenBatchSize(self):
    height, width = 3, 3
    with self.test_session() as sess:
      images = tf.random_uniform((5, height, width, 3), seed=1, name='images')
      inputs = tf.placeholder(tf.int32, (None, height, width, 3))
      output = tf.contrib.layers.flatten(inputs)
      self.assertEquals(output.get_shape().as_list(),
                        [None, height * width * 3])
      output = sess.run(output, {inputs: images.eval()})
      self.assertEquals(output.size,
                        images.get_shape().num_elements())
      self.assertEqual(output.shape[0], images.get_shape()[0])


class FCTest(tf.test.TestCase):

  def testCreateFC(self):
    height, width = 3, 3
    for layer_fn in (tf.contrib.layers.fully_connected, tf.contrib.layers.relu):
      with tf.Graph().as_default() as g, self.test_session(g):
        inputs = tf.random_uniform((5, height * width * 3), seed=1)
        output = layer_fn(inputs, 32)
        self.assertEquals(output.op.name, 'fully_connected/Relu')
        self.assertListEqual(output.get_shape().as_list(), [5, 32])
        weights = tf.contrib.framework.get_variables_by_name('weights')[0]
        self.assertListEqual(weights.get_shape().as_list(), [3 * 3 * 3, 32])
        biases = tf.contrib.framework.get_variables_by_name('biases')[0]
        self.assertListEqual(biases.get_shape().as_list(), [32])

  def testCreateFCWithScope(self):
    height, width = 3, 3
    with self.test_session():
      inputs = tf.random_uniform((5, height * width * 3), seed=1)
      output = tf.contrib.layers.fully_connected(inputs, 32, scope='fc1')
      self.assertEquals(output.op.name, 'fc1/Relu')

  def testCreateFcCreatesWeightsAndBiasesVars(self):
    height, width = 3, 3
    inputs = tf.random_uniform((5, height * width * 3), seed=1)
    with self.test_session():
      self.assertFalse(tf.contrib.framework.get_variables('fc1/weights'))
      self.assertFalse(tf.contrib.framework.get_variables('fc1/biases'))
      tf.contrib.layers.fully_connected(inputs, 32, scope='fc1')
      self.assertTrue(tf.contrib.framework.get_variables('fc1/weights'))
      self.assertTrue(tf.contrib.framework.get_variables('fc1/biases'))

  def testReuseVars(self):
    height, width = 3, 3
    inputs = tf.random_uniform((5, height * width * 3), seed=1)
    with self.test_session():
      tf.contrib.layers.fully_connected(inputs, 32, scope='fc1')
      self.assertEquals(len(tf.contrib.framework.get_variables('fc1')), 2)
      tf.contrib.layers.fully_connected(inputs, 32, scope='fc1', reuse=True)
      self.assertEquals(len(tf.contrib.framework.get_variables('fc1')), 2)

  def testNonReuseVars(self):
    height, width = 3, 3
    inputs = tf.random_uniform((5, height * width * 3), seed=1)
    with self.test_session():
      tf.contrib.layers.fully_connected(inputs, 32)
      self.assertEquals(
          len(tf.contrib.framework.get_variables('fully_connected')), 2)
      tf.contrib.layers.fully_connected(inputs, 32)
      self.assertEquals(
          len(tf.contrib.framework.get_variables('fully_connected')), 4)

  def testCreateFCWithoutActivation(self):
    height, width = 3, 3
    with self.test_session():
      inputs = tf.random_uniform((5, height * width * 3), seed=1)
      output = tf.contrib.layers.fully_connected(inputs, 32, activation_fn=None)
      self.assertEquals(output.op.name, 'fully_connected/BiasAdd')

  def testCreateFCWithWD(self):
    height, width = 3, 3
    with self.test_session() as sess:
      inputs = tf.random_uniform((5, height * width * 3), seed=1)
      weight_decay = tf.contrib.layers.l2_regularizer(0.01)
      tf.contrib.layers.fully_connected(inputs, 32,
                                        weights_regularizer=weight_decay)
      wd = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)[0]
      self.assertEquals(wd.op.name,
                        'fully_connected/weights/Regularizer/l2_regularizer')
      sess.run(tf.initialize_all_variables())
      self.assertLess(sess.run(wd), 0.4)

  def testCreateNoRegularizers(self):
    height, width = 3, 3
    with self.test_session():
      inputs = tf.random_uniform((5, height * width * 3), seed=1)
      tf.contrib.layers.fully_connected(inputs, 32)
      self.assertEquals(
          tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), [])

  def testReuseFCWithWD(self):
    height, width = 3, 3
    with self.test_session():
      inputs = tf.random_uniform((5, height * width * 3), seed=1)
      weight_decay = tf.contrib.layers.l2_regularizer(0.01)
      tf.contrib.layers.fully_connected(inputs, 32,
                                        weights_regularizer=weight_decay,
                                        scope='FC')
      self.assertEquals(len(tf.contrib.framework.get_variables()), 2)
      self.assertEquals(
          len(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)), 1)
      tf.contrib.layers.fully_connected(inputs, 32,
                                        weights_regularizer=weight_decay,
                                        scope='FC',
                                        reuse=True)
      self.assertEquals(len(tf.contrib.framework.get_variables()), 2)
      self.assertEquals(
          len(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)), 1)

  def testFCWithBatchNorm(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height * width * 3), seed=1)
      with tf.contrib.framework.arg_scope(
          [tf.contrib.layers.fully_connected],
          normalizer_fn=tf.contrib.layers.batch_norm,
          normalizer_params={'decay': 0.9}):
        net = tf.contrib.layers.fully_connected(images, 27)
        net = tf.contrib.layers.fully_connected(net, 27)
      self.assertEquals(len(tf.contrib.framework.get_variables()), 8)
      self.assertEquals(len(tf.contrib.framework.get_variables(
          'fully_connected/BatchNorm')), 3)
      self.assertEquals(len(tf.contrib.framework.get_variables(
          'fully_connected_1/BatchNorm')), 3)

  def testReuseFCWithBatchNorm(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height * width * 3), seed=1)
      with tf.contrib.framework.arg_scope(
          [tf.contrib.layers.fully_connected],
          normalizer_fn=tf.contrib.layers.batch_norm,
          normalizer_params={'decay': 0.9}):
        net = tf.contrib.layers.fully_connected(images, 27, scope='fc1')
        net = tf.contrib.layers.fully_connected(net, 27, scope='fc1',
                                                reuse=True)
      self.assertEquals(len(tf.contrib.framework.get_variables()), 4)
      self.assertEquals(
          len(tf.contrib.framework.get_variables('fc1/BatchNorm')), 3)


class BatchNormTest(tf.test.TestCase):

  def testCreateOp(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      output = tf.contrib.layers.batch_norm(images)
      self.assertTrue(output.op.name.startswith('BatchNorm/batchnorm'))
      self.assertListEqual(output.get_shape().as_list(), [5, height, width, 3])

  def testCreateVariables(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      tf.contrib.layers.batch_norm(images, scale=True)
      beta = tf.contrib.framework.get_variables_by_name('beta')[0]
      gamma = tf.contrib.framework.get_variables_by_name('gamma')[0]
      self.assertEquals(beta.op.name, 'BatchNorm/beta')
      self.assertEquals(gamma.op.name, 'BatchNorm/gamma')
      moving_mean = tf.contrib.framework.get_variables_by_name('moving_mean')[0]
      moving_variance = tf.contrib.framework.get_variables_by_name(
          'moving_variance')[0]
      self.assertEquals(moving_mean.op.name, 'BatchNorm/moving_mean')
      self.assertEquals(moving_variance.op.name, 'BatchNorm/moving_variance')

  def testMovingAverageVariables(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      tf.contrib.layers.batch_norm(images, scale=True)
      moving_mean = tf.contrib.framework.get_variables_by_name('moving_mean')[0]
      moving_variance = tf.contrib.framework.get_variables_by_name(
          'moving_variance')[0]
      self.assertEquals(moving_mean.op.name, 'BatchNorm/moving_mean')
      self.assertEquals(moving_variance.op.name, 'BatchNorm/moving_variance')

  def testUpdatesCollection(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      tf.contrib.layers.batch_norm(images, updates_collections='my_update_ops')
      update_layers = tf.get_collection('my_update_ops')
      update_moving_mean = update_layers[0]
      update_moving_variance = update_layers[1]
      self.assertEquals(update_moving_mean.op.name,
                        'BatchNorm/AssignMovingAvg')
      self.assertEquals(update_moving_variance.op.name,
                        'BatchNorm/AssignMovingAvg_1')

  def testReuseVariables(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      tf.contrib.layers.batch_norm(images, scale=True, scope='bn')
      tf.contrib.layers.batch_norm(images, scale=True, scope='bn', reuse=True)
      beta = tf.contrib.framework.get_variables_by_name('beta')
      gamma = tf.contrib.framework.get_variables_by_name('gamma')
      self.assertEquals(len(beta), 1)
      self.assertEquals(len(gamma), 1)
      moving_mean = tf.contrib.framework.get_variables_by_name('moving_mean')
      moving_variance = tf.contrib.framework.get_variables_by_name(
          'moving_variance')
      moving_vars = moving_mean + moving_variance
      self.assertEquals(len(moving_vars), 2)

  def testReuseUpdateOps(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      with tf.contrib.framework.arg_scope([tf.contrib.layers.batch_norm],
                                          updates_collections='update_ops'):
        tf.contrib.layers.batch_norm(images, scope='bn')
        self.assertEquals(len(tf.get_collection('update_ops')), 2)
        tf.contrib.layers.batch_norm(images, scope='bn', reuse=True)
        self.assertEquals(len(tf.get_collection('update_ops')), 4)

  def testCreateMovingVars(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      _ = tf.contrib.layers.batch_norm(images)
      moving_mean = tf.contrib.framework.get_variables('BatchNorm/moving_mean')
      self.assertEquals(len(moving_mean), 1)
      self.assertEquals(moving_mean[0].op.name, 'BatchNorm/moving_mean')
      moving_variance = tf.contrib.framework.get_variables(
          'BatchNorm/moving_variance')
      self.assertEquals(len(moving_variance), 1)
      self.assertEquals(moving_variance[0].op.name, 'BatchNorm/moving_variance')

  def testForceUpdateMovingVars(self):
    height, width = 3, 3
    with self.test_session() as sess:
      image_shape = (10, height, width, 3)
      image_values = np.random.rand(*image_shape)
      expected_mean = np.mean(image_values, axis=(0, 1, 2))
      expected_var = np.var(image_values, axis=(0, 1, 2))
      images = tf.constant(image_values, shape=image_shape, dtype=tf.float32)
      output = tf.contrib.layers.batch_norm(images, decay=0.1,
                                            updates_collections=None)
      # Initialize all variables
      sess.run(tf.initialize_all_variables())
      moving_mean = tf.contrib.framework.get_variables(
          'BatchNorm/moving_mean')[0]
      moving_variance = tf.contrib.framework.get_variables(
          'BatchNorm/moving_variance')[0]
      mean, variance = sess.run([moving_mean, moving_variance])
      # After initialization moving_mean == 0 and moving_variance == 1.
      self.assertAllClose(mean, [0] * 3)
      self.assertAllClose(variance, [1] * 3)
      for _ in range(10):
        sess.run([output])
      mean = moving_mean.eval()
      variance = moving_variance.eval()
      # After 10 updates with decay 0.1 moving_mean == expected_mean and
      # moving_variance == expected_var.
      self.assertAllClose(mean, expected_mean)
      self.assertAllClose(variance, expected_var)

  def testDelayedUpdateMovingVars(self):
    height, width = 3, 3
    with self.test_session() as sess:
      image_shape = (10, height, width, 3)
      image_values = np.random.rand(*image_shape)
      expected_mean = np.mean(image_values, axis=(0, 1, 2))
      expected_var = np.var(image_values, axis=(0, 1, 2))
      images = tf.constant(image_values, shape=image_shape, dtype=tf.float32)
      output = tf.contrib.layers.batch_norm(images, decay=0.1)
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
        barrier = tf.no_op(name='barrier')
      output = control_flow_ops.with_dependencies([barrier], output)
      # Initialize all variables
      sess.run(tf.initialize_all_variables())
      moving_mean = tf.contrib.framework.get_variables(
          'BatchNorm/moving_mean')[0]
      moving_variance = tf.contrib.framework.get_variables(
          'BatchNorm/moving_variance')[0]
      mean, variance = sess.run([moving_mean, moving_variance])
      # After initialization moving_mean == 0 and moving_variance == 1.
      self.assertAllClose(mean, [0] * 3)
      self.assertAllClose(variance, [1] * 3)
      for _ in range(10):
        sess.run([output])
      mean = moving_mean.eval()
      variance = moving_variance.eval()
      # After 10 updates with decay 0.1 moving_mean == expected_mean and
      # moving_variance == expected_var.
      self.assertAllClose(mean, expected_mean)
      self.assertAllClose(variance, expected_var)

  def testEvalMovingVars(self):
    height, width = 3, 3
    with self.test_session() as sess:
      image_shape = (10, height, width, 3)
      image_values = np.random.rand(*image_shape)
      expected_mean = np.mean(image_values, axis=(0, 1, 2))
      expected_var = np.var(image_values, axis=(0, 1, 2))
      images = tf.constant(image_values, shape=image_shape, dtype=tf.float32)
      output = tf.contrib.layers.batch_norm(images,
                                            decay=0.1,
                                            is_training=False)
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      self.assertEquals(update_ops, [])
      # Initialize all variables
      sess.run(tf.initialize_all_variables())
      moving_mean = tf.contrib.framework.get_variables(
          'BatchNorm/moving_mean')[0]
      moving_variance = tf.contrib.framework.get_variables(
          'BatchNorm/moving_variance')[0]
      mean, variance = sess.run([moving_mean, moving_variance])
      # After initialization moving_mean == 0 and moving_variance == 1.
      self.assertAllClose(mean, [0] * 3)
      self.assertAllClose(variance, [1] * 3)
      # Simulate assigment from saver restore.
      init_assigns = [tf.assign(moving_mean, expected_mean),
                      tf.assign(moving_variance, expected_var)]
      sess.run(init_assigns)
      for _ in range(10):
        sess.run([output], {images: np.random.rand(*image_shape)})
      mean = moving_mean.eval()
      variance = moving_variance.eval()
      # Although we feed different images, the moving_mean and moving_variance
      # shouldn't change.
      self.assertAllClose(mean, expected_mean)
      self.assertAllClose(variance, expected_var)

  def testReuseVars(self):
    height, width = 3, 3
    with self.test_session() as sess:
      image_shape = (10, height, width, 3)
      image_values = np.random.rand(*image_shape)
      expected_mean = np.mean(image_values, axis=(0, 1, 2))
      expected_var = np.var(image_values, axis=(0, 1, 2))
      images = tf.constant(image_values, shape=image_shape, dtype=tf.float32)
      output = tf.contrib.layers.batch_norm(images,
                                            decay=0.1,
                                            is_training=False)
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      self.assertEquals(update_ops, [])
      # Initialize all variables
      sess.run(tf.initialize_all_variables())
      moving_mean = tf.contrib.framework.get_variables(
          'BatchNorm/moving_mean')[0]
      moving_variance = tf.contrib.framework.get_variables(
          'BatchNorm/moving_variance')[0]
      mean, variance = sess.run([moving_mean, moving_variance])
      # After initialization moving_mean == 0 and moving_variance == 1.
      self.assertAllClose(mean, [0] * 3)
      self.assertAllClose(variance, [1] * 3)
      # Simulate assigment from saver restore.
      init_assigns = [tf.assign(moving_mean, expected_mean),
                      tf.assign(moving_variance, expected_var)]
      sess.run(init_assigns)
      for _ in range(10):
        sess.run([output], {images: np.random.rand(*image_shape)})
      mean = moving_mean.eval()
      variance = moving_variance.eval()
      # Although we feed different images, the moving_mean and moving_variance
      # shouldn't change.
      self.assertAllClose(mean, expected_mean)
      self.assertAllClose(variance, expected_var)


class MaxPool2DTest(tf.test.TestCase):

  def testCreateMaxPool(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      output = tf.contrib.layers.max_pool2d(images, [3, 3])
      self.assertEquals(output.op.name, 'MaxPool2D/MaxPool')
      self.assertListEqual(output.get_shape().as_list(), [5, 1, 1, 3])

  def testCollectOutputs(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      output = tf.contrib.layers.max_pool2d(images, [3, 3],
                                            outputs_collections='outputs')
      self.assertEquals(('MaxPool2D', output),
                        tf.get_collection('outputs')[0])

  def testCreateSquareMaxPool(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      output = tf.contrib.layers.max_pool2d(images, 3)
      self.assertEquals(output.op.name, 'MaxPool2D/MaxPool')
      self.assertListEqual(output.get_shape().as_list(), [5, 1, 1, 3])

  def testCreateMaxPoolWithScope(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      output = tf.contrib.layers.max_pool2d(images, [3, 3], scope='pool1')
      self.assertEquals(output.op.name, 'pool1/MaxPool')

  def testCreateMaxPoolSAME(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      output = tf.contrib.layers.max_pool2d(images, [3, 3], padding='SAME')
      self.assertListEqual(output.get_shape().as_list(), [5, 2, 2, 3])

  def testCreateMaxPoolStrideSAME(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      output = tf.contrib.layers.max_pool2d(images, [3, 3], stride=1,
                                            padding='SAME')
      self.assertListEqual(output.get_shape().as_list(), [5, height, width, 3])

  def testGlobalMaxPool(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1)
      output = tf.contrib.layers.max_pool2d(images, images.get_shape()[1:3],
                                            stride=1)
      self.assertListEqual(output.get_shape().as_list(), [5, 1, 1, 3])


class OneHotEncodingTest(tf.test.TestCase):

  def testOneHotEncodingCreate(self):
    with self.test_session():
      labels = tf.constant([0, 1, 2])
      output = tf.contrib.layers.one_hot_encoding(labels, num_classes=3)
      self.assertEquals(output.op.name, 'OneHotEncoding/one_hot')
      self.assertListEqual(output.get_shape().as_list(), [3, 3])

  def testCollectOutputs(self):
    with self.test_session():
      labels = tf.constant([0, 1, 2])
      output = tf.contrib.layers.one_hot_encoding(labels, num_classes=3,
                                                  outputs_collections='outputs')
      self.assertEquals(('OneHotEncoding', output),
                        tf.get_collection('outputs')[0])

  def testOneHotEncoding(self):
    with self.test_session():
      labels = tf.constant([0, 1, 2])
      one_hot_labels = tf.constant([[1, 0, 0],
                                    [0, 1, 0],
                                    [0, 0, 1]])
      output = tf.contrib.layers.one_hot_encoding(labels, num_classes=3)
      self.assertAllClose(output.eval(), one_hot_labels.eval())

  def testOneHotEncodingInt32(self):
    with self.test_session():
      labels = tf.constant([0, 1, 2], dtype=tf.int32)
      one_hot_labels = tf.constant([[1, 0, 0],
                                    [0, 1, 0],
                                    [0, 0, 1]])
      output = tf.contrib.layers.one_hot_encoding(labels, num_classes=3)
      self.assertAllClose(output.eval(), one_hot_labels.eval())


class StackTests(tf.test.TestCase):

  def testStackFullyConnected(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height * width * 3), seed=1, name='images')
      output = tf.contrib.layers.stack(images,
                                       tf.contrib.layers.fully_connected,
                                       [10, 20, 30])
      self.assertEquals(output.op.name, 'Stack/fully_connected_3/Relu')
      self.assertListEqual(output.get_shape().as_list(), [5, 30])

  def testStackConvolution2d(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1, name='images')
      output = tf.contrib.layers.stack(images,
                                       tf.contrib.layers.convolution2d,
                                       [10, 20, 30],
                                       kernel_size=[3, 3],
                                       padding='SAME')
      self.assertEquals(output.op.name, 'Stack/convolution2d_3/Relu')
      self.assertListEqual(output.get_shape().as_list(), [5, 3, 3, 30])

  def testStackWithScope(self):
    height, width = 3, 3
    with self.test_session():
      images = tf.random_uniform((5, height, width, 3), seed=1, name='images')
      output = tf.contrib.layers.stack(images,
                                       tf.contrib.layers.convolution2d,
                                       [10, 20, 30],
                                       kernel_size=[3, 3],
                                       padding='SAME',
                                       scope='conv1')
      self.assertEquals(output.op.name, 'conv1/conv1_3/Relu')
      self.assertListEqual(output.get_shape().as_list(), [5, 3, 3, 30])


# TODO(b/28426988): Add separate tests for non-legacy versions.
class LegacyFullyConnectedTest(tf.test.TestCase):

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


if __name__ == '__main__':
  tf.test.main()
