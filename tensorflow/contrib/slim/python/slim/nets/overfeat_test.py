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
"""Tests for slim.nets.overfeat."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.contrib.slim.nets import overfeat

slim = tf.contrib.slim


class OverFeatTest(tf.test.TestCase):

  def testBuild(self):
    batch_size = 5
    height, width = 231, 231
    num_classes = 1000
    with self.test_session():
      inputs = tf.random_uniform((batch_size, height, width, 3))
      logits, _ = overfeat.overfeat(inputs, num_classes)
      self.assertEquals(logits.op.name, 'overfeat/fc8/squeezed')
      self.assertListEqual(logits.get_shape().as_list(),
                           [batch_size, num_classes])

  def testFullyConvolutional(self):
    batch_size = 1
    height, width = 281, 281
    num_classes = 1000
    with self.test_session():
      inputs = tf.random_uniform((batch_size, height, width, 3))
      logits, _ = overfeat.overfeat(inputs, num_classes, spatial_squeeze=False)
      self.assertEquals(logits.op.name, 'overfeat/fc8/BiasAdd')
      self.assertListEqual(logits.get_shape().as_list(),
                           [batch_size, 2, 2, num_classes])

  def testEndPoints(self):
    batch_size = 5
    height, width = 231, 231
    num_classes = 1000
    with self.test_session():
      inputs = tf.random_uniform((batch_size, height, width, 3))
      _, end_points = overfeat.overfeat(inputs, num_classes)
      expected_names = ['overfeat/conv1',
                        'overfeat/pool1',
                        'overfeat/conv2',
                        'overfeat/pool2',
                        'overfeat/conv3',
                        'overfeat/conv4',
                        'overfeat/conv5',
                        'overfeat/pool5',
                        'overfeat/fc6',
                        'overfeat/fc7',
                        'overfeat/fc8'
                       ]
      self.assertSetEqual(set(end_points.keys()), set(expected_names))

  def testModelVariables(self):
    batch_size = 5
    height, width = 231, 231
    num_classes = 1000
    with self.test_session():
      inputs = tf.random_uniform((batch_size, height, width, 3))
      overfeat.overfeat(inputs, num_classes)
      expected_names = ['overfeat/conv1/weights',
                        'overfeat/conv1/biases',
                        'overfeat/conv2/weights',
                        'overfeat/conv2/biases',
                        'overfeat/conv3/weights',
                        'overfeat/conv3/biases',
                        'overfeat/conv4/weights',
                        'overfeat/conv4/biases',
                        'overfeat/conv5/weights',
                        'overfeat/conv5/biases',
                        'overfeat/fc6/weights',
                        'overfeat/fc6/biases',
                        'overfeat/fc7/weights',
                        'overfeat/fc7/biases',
                        'overfeat/fc8/weights',
                        'overfeat/fc8/biases',
                       ]
      model_variables = [v.op.name for v in slim.get_model_variables()]
      self.assertSetEqual(set(model_variables), set(expected_names))

  def testEvaluation(self):
    batch_size = 2
    height, width = 231, 231
    num_classes = 1000
    with self.test_session():
      eval_inputs = tf.random_uniform((batch_size, height, width, 3))
      logits, _ = overfeat.overfeat(eval_inputs, is_training=False)
      self.assertListEqual(logits.get_shape().as_list(),
                           [batch_size, num_classes])
      predictions = tf.argmax(logits, 1)
      self.assertListEqual(predictions.get_shape().as_list(), [batch_size])

  def testTrainEvalWithReuse(self):
    train_batch_size = 2
    eval_batch_size = 1
    train_height, train_width = 231, 231
    eval_height, eval_width = 281, 281
    num_classes = 1000
    with self.test_session():
      train_inputs = tf.random_uniform(
          (train_batch_size, train_height, train_width, 3))
      logits, _ = overfeat.overfeat(train_inputs)
      self.assertListEqual(logits.get_shape().as_list(),
                           [train_batch_size, num_classes])
      tf.get_variable_scope().reuse_variables()
      eval_inputs = tf.random_uniform(
          (eval_batch_size, eval_height, eval_width, 3))
      logits, _ = overfeat.overfeat(eval_inputs, is_training=False,
                                    spatial_squeeze=False)
      self.assertListEqual(logits.get_shape().as_list(),
                           [eval_batch_size, 2, 2, num_classes])
      logits = tf.reduce_mean(logits, [1, 2])
      predictions = tf.argmax(logits, 1)
      self.assertEquals(predictions.get_shape().as_list(), [eval_batch_size])

  def testForward(self):
    batch_size = 1
    height, width = 231, 231
    with self.test_session() as sess:
      inputs = tf.random_uniform((batch_size, height, width, 3))
      logits, _ = overfeat.overfeat(inputs)
      sess.run(tf.initialize_all_variables())
      output = sess.run(logits)
      self.assertTrue(output.any())

if __name__ == '__main__':
  tf.test.main()
