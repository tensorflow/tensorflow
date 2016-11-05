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
"""Tests for slim.nets.vgg."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.contrib.slim.nets import vgg

slim = tf.contrib.slim


class VGGATest(tf.test.TestCase):

  def testBuild(self):
    batch_size = 5
    height, width = 224, 224
    num_classes = 1000
    with self.test_session():
      inputs = tf.random_uniform((batch_size, height, width, 3))
      logits, _ = vgg.vgg_a(inputs, num_classes)
      self.assertEquals(logits.op.name, 'vgg_a/fc8/squeezed')
      self.assertListEqual(logits.get_shape().as_list(),
                           [batch_size, num_classes])

  def testFullyConvolutional(self):
    batch_size = 1
    height, width = 256, 256
    num_classes = 1000
    with self.test_session():
      inputs = tf.random_uniform((batch_size, height, width, 3))
      logits, _ = vgg.vgg_a(inputs, num_classes, spatial_squeeze=False)
      self.assertEquals(logits.op.name, 'vgg_a/fc8/BiasAdd')
      self.assertListEqual(logits.get_shape().as_list(),
                           [batch_size, 2, 2, num_classes])

  def testEndPoints(self):
    batch_size = 5
    height, width = 224, 224
    num_classes = 1000
    for is_training in [True, False]:
      with tf.Graph().as_default():
        inputs = tf.random_uniform((batch_size, height, width, 3))
        _, end_points = vgg.vgg_a(inputs, num_classes, is_training=is_training)
        expected_names = ['vgg_a/conv1/conv1_1',
                          'vgg_a/pool1',
                          'vgg_a/conv2/conv2_1',
                          'vgg_a/pool2',
                          'vgg_a/conv3/conv3_1',
                          'vgg_a/conv3/conv3_2',
                          'vgg_a/pool3',
                          'vgg_a/conv4/conv4_1',
                          'vgg_a/conv4/conv4_2',
                          'vgg_a/pool4',
                          'vgg_a/conv5/conv5_1',
                          'vgg_a/conv5/conv5_2',
                          'vgg_a/pool5',
                          'vgg_a/fc6',
                          'vgg_a/fc7',
                          'vgg_a/fc8'
                         ]
        self.assertSetEqual(set(end_points.keys()), set(expected_names))

  def testModelVariables(self):
    batch_size = 5
    height, width = 224, 224
    num_classes = 1000
    with self.test_session():
      inputs = tf.random_uniform((batch_size, height, width, 3))
      vgg.vgg_a(inputs, num_classes)
      expected_names = ['vgg_a/conv1/conv1_1/weights',
                        'vgg_a/conv1/conv1_1/biases',
                        'vgg_a/conv2/conv2_1/weights',
                        'vgg_a/conv2/conv2_1/biases',
                        'vgg_a/conv3/conv3_1/weights',
                        'vgg_a/conv3/conv3_1/biases',
                        'vgg_a/conv3/conv3_2/weights',
                        'vgg_a/conv3/conv3_2/biases',
                        'vgg_a/conv4/conv4_1/weights',
                        'vgg_a/conv4/conv4_1/biases',
                        'vgg_a/conv4/conv4_2/weights',
                        'vgg_a/conv4/conv4_2/biases',
                        'vgg_a/conv5/conv5_1/weights',
                        'vgg_a/conv5/conv5_1/biases',
                        'vgg_a/conv5/conv5_2/weights',
                        'vgg_a/conv5/conv5_2/biases',
                        'vgg_a/fc6/weights',
                        'vgg_a/fc6/biases',
                        'vgg_a/fc7/weights',
                        'vgg_a/fc7/biases',
                        'vgg_a/fc8/weights',
                        'vgg_a/fc8/biases',
                       ]
      model_variables = [v.op.name for v in slim.get_model_variables()]
      self.assertSetEqual(set(model_variables), set(expected_names))

  def testEvaluation(self):
    batch_size = 2
    height, width = 224, 224
    num_classes = 1000
    with self.test_session():
      eval_inputs = tf.random_uniform((batch_size, height, width, 3))
      logits, _ = vgg.vgg_a(eval_inputs, is_training=False)
      self.assertListEqual(logits.get_shape().as_list(),
                           [batch_size, num_classes])
      predictions = tf.argmax(logits, 1)
      self.assertListEqual(predictions.get_shape().as_list(), [batch_size])

  def testTrainEvalWithReuse(self):
    train_batch_size = 2
    eval_batch_size = 1
    train_height, train_width = 224, 224
    eval_height, eval_width = 256, 256
    num_classes = 1000
    with self.test_session():
      train_inputs = tf.random_uniform(
          (train_batch_size, train_height, train_width, 3))
      logits, _ = vgg.vgg_a(train_inputs)
      self.assertListEqual(logits.get_shape().as_list(),
                           [train_batch_size, num_classes])
      tf.get_variable_scope().reuse_variables()
      eval_inputs = tf.random_uniform(
          (eval_batch_size, eval_height, eval_width, 3))
      logits, _ = vgg.vgg_a(eval_inputs, is_training=False,
                            spatial_squeeze=False)
      self.assertListEqual(logits.get_shape().as_list(),
                           [eval_batch_size, 2, 2, num_classes])
      logits = tf.reduce_mean(logits, [1, 2])
      predictions = tf.argmax(logits, 1)
      self.assertEquals(predictions.get_shape().as_list(), [eval_batch_size])

  def testForward(self):
    batch_size = 1
    height, width = 224, 224
    with self.test_session() as sess:
      inputs = tf.random_uniform((batch_size, height, width, 3))
      logits, _ = vgg.vgg_a(inputs)
      sess.run(tf.global_variables_initializer())
      output = sess.run(logits)
      self.assertTrue(output.any())


class VGG16Test(tf.test.TestCase):

  def testBuild(self):
    batch_size = 5
    height, width = 224, 224
    num_classes = 1000
    with self.test_session():
      inputs = tf.random_uniform((batch_size, height, width, 3))
      logits, _ = vgg.vgg_16(inputs, num_classes)
      self.assertEquals(logits.op.name, 'vgg_16/fc8/squeezed')
      self.assertListEqual(logits.get_shape().as_list(),
                           [batch_size, num_classes])

  def testFullyConvolutional(self):
    batch_size = 1
    height, width = 256, 256
    num_classes = 1000
    with self.test_session():
      inputs = tf.random_uniform((batch_size, height, width, 3))
      logits, _ = vgg.vgg_16(inputs, num_classes, spatial_squeeze=False)
      self.assertEquals(logits.op.name, 'vgg_16/fc8/BiasAdd')
      self.assertListEqual(logits.get_shape().as_list(),
                           [batch_size, 2, 2, num_classes])

  def testEndPoints(self):
    batch_size = 5
    height, width = 224, 224
    num_classes = 1000
    for is_training in [True, False]:
      with tf.Graph().as_default():
        inputs = tf.random_uniform((batch_size, height, width, 3))
        _, end_points = vgg.vgg_16(inputs, num_classes, is_training=is_training)
        expected_names = ['vgg_16/conv1/conv1_1',
                          'vgg_16/conv1/conv1_2',
                          'vgg_16/pool1',
                          'vgg_16/conv2/conv2_1',
                          'vgg_16/conv2/conv2_2',
                          'vgg_16/pool2',
                          'vgg_16/conv3/conv3_1',
                          'vgg_16/conv3/conv3_2',
                          'vgg_16/conv3/conv3_3',
                          'vgg_16/pool3',
                          'vgg_16/conv4/conv4_1',
                          'vgg_16/conv4/conv4_2',
                          'vgg_16/conv4/conv4_3',
                          'vgg_16/pool4',
                          'vgg_16/conv5/conv5_1',
                          'vgg_16/conv5/conv5_2',
                          'vgg_16/conv5/conv5_3',
                          'vgg_16/pool5',
                          'vgg_16/fc6',
                          'vgg_16/fc7',
                          'vgg_16/fc8'
                         ]
        self.assertSetEqual(set(end_points.keys()), set(expected_names))

  def testModelVariables(self):
    batch_size = 5
    height, width = 224, 224
    num_classes = 1000
    with self.test_session():
      inputs = tf.random_uniform((batch_size, height, width, 3))
      vgg.vgg_16(inputs, num_classes)
      expected_names = ['vgg_16/conv1/conv1_1/weights',
                        'vgg_16/conv1/conv1_1/biases',
                        'vgg_16/conv1/conv1_2/weights',
                        'vgg_16/conv1/conv1_2/biases',
                        'vgg_16/conv2/conv2_1/weights',
                        'vgg_16/conv2/conv2_1/biases',
                        'vgg_16/conv2/conv2_2/weights',
                        'vgg_16/conv2/conv2_2/biases',
                        'vgg_16/conv3/conv3_1/weights',
                        'vgg_16/conv3/conv3_1/biases',
                        'vgg_16/conv3/conv3_2/weights',
                        'vgg_16/conv3/conv3_2/biases',
                        'vgg_16/conv3/conv3_3/weights',
                        'vgg_16/conv3/conv3_3/biases',
                        'vgg_16/conv4/conv4_1/weights',
                        'vgg_16/conv4/conv4_1/biases',
                        'vgg_16/conv4/conv4_2/weights',
                        'vgg_16/conv4/conv4_2/biases',
                        'vgg_16/conv4/conv4_3/weights',
                        'vgg_16/conv4/conv4_3/biases',
                        'vgg_16/conv5/conv5_1/weights',
                        'vgg_16/conv5/conv5_1/biases',
                        'vgg_16/conv5/conv5_2/weights',
                        'vgg_16/conv5/conv5_2/biases',
                        'vgg_16/conv5/conv5_3/weights',
                        'vgg_16/conv5/conv5_3/biases',
                        'vgg_16/fc6/weights',
                        'vgg_16/fc6/biases',
                        'vgg_16/fc7/weights',
                        'vgg_16/fc7/biases',
                        'vgg_16/fc8/weights',
                        'vgg_16/fc8/biases',
                       ]
      model_variables = [v.op.name for v in slim.get_model_variables()]
      self.assertSetEqual(set(model_variables), set(expected_names))

  def testEvaluation(self):
    batch_size = 2
    height, width = 224, 224
    num_classes = 1000
    with self.test_session():
      eval_inputs = tf.random_uniform((batch_size, height, width, 3))
      logits, _ = vgg.vgg_16(eval_inputs, is_training=False)
      self.assertListEqual(logits.get_shape().as_list(),
                           [batch_size, num_classes])
      predictions = tf.argmax(logits, 1)
      self.assertListEqual(predictions.get_shape().as_list(), [batch_size])

  def testTrainEvalWithReuse(self):
    train_batch_size = 2
    eval_batch_size = 1
    train_height, train_width = 224, 224
    eval_height, eval_width = 256, 256
    num_classes = 1000
    with self.test_session():
      train_inputs = tf.random_uniform(
          (train_batch_size, train_height, train_width, 3))
      logits, _ = vgg.vgg_16(train_inputs)
      self.assertListEqual(logits.get_shape().as_list(),
                           [train_batch_size, num_classes])
      tf.get_variable_scope().reuse_variables()
      eval_inputs = tf.random_uniform(
          (eval_batch_size, eval_height, eval_width, 3))
      logits, _ = vgg.vgg_16(eval_inputs, is_training=False,
                             spatial_squeeze=False)
      self.assertListEqual(logits.get_shape().as_list(),
                           [eval_batch_size, 2, 2, num_classes])
      logits = tf.reduce_mean(logits, [1, 2])
      predictions = tf.argmax(logits, 1)
      self.assertEquals(predictions.get_shape().as_list(), [eval_batch_size])

  def testForward(self):
    batch_size = 1
    height, width = 224, 224
    with self.test_session() as sess:
      inputs = tf.random_uniform((batch_size, height, width, 3))
      logits, _ = vgg.vgg_16(inputs)
      sess.run(tf.global_variables_initializer())
      output = sess.run(logits)
      self.assertTrue(output.any())


class VGG19Test(tf.test.TestCase):

  def testBuild(self):
    batch_size = 5
    height, width = 224, 224
    num_classes = 1000
    with self.test_session():
      inputs = tf.random_uniform((batch_size, height, width, 3))
      logits, _ = vgg.vgg_19(inputs, num_classes)
      self.assertEquals(logits.op.name, 'vgg_19/fc8/squeezed')
      self.assertListEqual(logits.get_shape().as_list(),
                           [batch_size, num_classes])

  def testFullyConvolutional(self):
    batch_size = 1
    height, width = 256, 256
    num_classes = 1000
    with self.test_session():
      inputs = tf.random_uniform((batch_size, height, width, 3))
      logits, _ = vgg.vgg_19(inputs, num_classes, spatial_squeeze=False)
      self.assertEquals(logits.op.name, 'vgg_19/fc8/BiasAdd')
      self.assertListEqual(logits.get_shape().as_list(),
                           [batch_size, 2, 2, num_classes])

  def testEndPoints(self):
    batch_size = 5
    height, width = 224, 224
    num_classes = 1000
    for is_training in [True, False]:
      with tf.Graph().as_default():
        inputs = tf.random_uniform((batch_size, height, width, 3))
        _, end_points = vgg.vgg_19(inputs, num_classes, is_training=is_training)
        expected_names = [
            'vgg_19/conv1/conv1_1',
            'vgg_19/conv1/conv1_2',
            'vgg_19/pool1',
            'vgg_19/conv2/conv2_1',
            'vgg_19/conv2/conv2_2',
            'vgg_19/pool2',
            'vgg_19/conv3/conv3_1',
            'vgg_19/conv3/conv3_2',
            'vgg_19/conv3/conv3_3',
            'vgg_19/conv3/conv3_4',
            'vgg_19/pool3',
            'vgg_19/conv4/conv4_1',
            'vgg_19/conv4/conv4_2',
            'vgg_19/conv4/conv4_3',
            'vgg_19/conv4/conv4_4',
            'vgg_19/pool4',
            'vgg_19/conv5/conv5_1',
            'vgg_19/conv5/conv5_2',
            'vgg_19/conv5/conv5_3',
            'vgg_19/conv5/conv5_4',
            'vgg_19/pool5',
            'vgg_19/fc6',
            'vgg_19/fc7',
            'vgg_19/fc8'
        ]
        self.assertSetEqual(set(end_points.keys()), set(expected_names))

  def testModelVariables(self):
    batch_size = 5
    height, width = 224, 224
    num_classes = 1000
    with self.test_session():
      inputs = tf.random_uniform((batch_size, height, width, 3))
      vgg.vgg_19(inputs, num_classes)
      expected_names = [
          'vgg_19/conv1/conv1_1/weights',
          'vgg_19/conv1/conv1_1/biases',
          'vgg_19/conv1/conv1_2/weights',
          'vgg_19/conv1/conv1_2/biases',
          'vgg_19/conv2/conv2_1/weights',
          'vgg_19/conv2/conv2_1/biases',
          'vgg_19/conv2/conv2_2/weights',
          'vgg_19/conv2/conv2_2/biases',
          'vgg_19/conv3/conv3_1/weights',
          'vgg_19/conv3/conv3_1/biases',
          'vgg_19/conv3/conv3_2/weights',
          'vgg_19/conv3/conv3_2/biases',
          'vgg_19/conv3/conv3_3/weights',
          'vgg_19/conv3/conv3_3/biases',
          'vgg_19/conv3/conv3_4/weights',
          'vgg_19/conv3/conv3_4/biases',
          'vgg_19/conv4/conv4_1/weights',
          'vgg_19/conv4/conv4_1/biases',
          'vgg_19/conv4/conv4_2/weights',
          'vgg_19/conv4/conv4_2/biases',
          'vgg_19/conv4/conv4_3/weights',
          'vgg_19/conv4/conv4_3/biases',
          'vgg_19/conv4/conv4_4/weights',
          'vgg_19/conv4/conv4_4/biases',
          'vgg_19/conv5/conv5_1/weights',
          'vgg_19/conv5/conv5_1/biases',
          'vgg_19/conv5/conv5_2/weights',
          'vgg_19/conv5/conv5_2/biases',
          'vgg_19/conv5/conv5_3/weights',
          'vgg_19/conv5/conv5_3/biases',
          'vgg_19/conv5/conv5_4/weights',
          'vgg_19/conv5/conv5_4/biases',
          'vgg_19/fc6/weights',
          'vgg_19/fc6/biases',
          'vgg_19/fc7/weights',
          'vgg_19/fc7/biases',
          'vgg_19/fc8/weights',
          'vgg_19/fc8/biases',
      ]
      model_variables = [v.op.name for v in slim.get_model_variables()]
      self.assertSetEqual(set(model_variables), set(expected_names))

  def testEvaluation(self):
    batch_size = 2
    height, width = 224, 224
    num_classes = 1000
    with self.test_session():
      eval_inputs = tf.random_uniform((batch_size, height, width, 3))
      logits, _ = vgg.vgg_19(eval_inputs, is_training=False)
      self.assertListEqual(logits.get_shape().as_list(),
                           [batch_size, num_classes])
      predictions = tf.argmax(logits, 1)
      self.assertListEqual(predictions.get_shape().as_list(), [batch_size])

  def testTrainEvalWithReuse(self):
    train_batch_size = 2
    eval_batch_size = 1
    train_height, train_width = 224, 224
    eval_height, eval_width = 256, 256
    num_classes = 1000
    with self.test_session():
      train_inputs = tf.random_uniform(
          (train_batch_size, train_height, train_width, 3))
      logits, _ = vgg.vgg_19(train_inputs)
      self.assertListEqual(logits.get_shape().as_list(),
                           [train_batch_size, num_classes])
      tf.get_variable_scope().reuse_variables()
      eval_inputs = tf.random_uniform(
          (eval_batch_size, eval_height, eval_width, 3))
      logits, _ = vgg.vgg_19(eval_inputs, is_training=False,
                             spatial_squeeze=False)
      self.assertListEqual(logits.get_shape().as_list(),
                           [eval_batch_size, 2, 2, num_classes])
      logits = tf.reduce_mean(logits, [1, 2])
      predictions = tf.argmax(logits, 1)
      self.assertEquals(predictions.get_shape().as_list(), [eval_batch_size])

  def testForward(self):
    batch_size = 1
    height, width = 224, 224
    with self.test_session() as sess:
      inputs = tf.random_uniform((batch_size, height, width, 3))
      logits, _ = vgg.vgg_19(inputs)
      sess.run(tf.global_variables_initializer())
      output = sess.run(logits)
      self.assertTrue(output.any())

if __name__ == '__main__':
  tf.test.main()
