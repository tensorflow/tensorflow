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
"""Tests for convnet.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.contrib.kfac import layer_collection as lc
from tensorflow.contrib.kfac.examples import convnet


class ConvNetTest(tf.test.TestCase):

  def testConvLayer(self):
    with tf.Graph().as_default():
      pre, act, (w, b) = convnet.conv_layer(
          layer_id=1,
          inputs=tf.zeros([5, 3, 3, 2]),
          kernel_size=3,
          out_channels=5)
      self.assertShapeEqual(np.zeros([5, 3, 3, 5]), pre)
      self.assertShapeEqual(np.zeros([5, 3, 3, 5]), act)
      self.assertShapeEqual(np.zeros([3, 3, 2, 5]), tf.convert_to_tensor(w))
      self.assertShapeEqual(np.zeros([5]), tf.convert_to_tensor(b))
      self.assertIsInstance(w, tf.Variable)
      self.assertIsInstance(b, tf.Variable)
      self.assertIn("conv_1", w.op.name)
      self.assertIn("conv_1", b.op.name)

  def testMaxPoolLayer(self):
    with tf.Graph().as_default():
      act = convnet.max_pool_layer(
          layer_id=1, inputs=tf.zeros([5, 6, 6, 2]), kernel_size=5, stride=3)
      self.assertShapeEqual(np.zeros([5, 2, 2, 2]), act)
      self.assertEqual(act.op.name, "pool_1/pool")

  def testLinearLayer(self):
    with tf.Graph().as_default():
      act, (w, b) = convnet.linear_layer(
          layer_id=1, inputs=tf.zeros([5, 20]), output_size=5)
      self.assertShapeEqual(np.zeros([5, 5]), act)
      self.assertShapeEqual(np.zeros([20, 5]), tf.convert_to_tensor(w))
      self.assertShapeEqual(np.zeros([5]), tf.convert_to_tensor(b))
      self.assertIsInstance(w, tf.Variable)
      self.assertIsInstance(b, tf.Variable)
      self.assertIn("fc_1", w.op.name)
      self.assertIn("fc_1", b.op.name)

  def testBuildModel(self):
    with tf.Graph().as_default():
      x = tf.placeholder(tf.float32, [None, 6, 6, 3])
      y = tf.placeholder(tf.int64, [None])
      layer_collection = lc.LayerCollection()
      loss, accuracy = convnet.build_model(
          x, y, num_labels=5, layer_collection=layer_collection)

      # Ensure layers and logits were registered.
      self.assertEqual(len(layer_collection.fisher_blocks), 3)
      self.assertEqual(len(layer_collection.losses), 1)

      # Ensure inference doesn't crash.
      with self.test_session() as sess:
        sess.run(tf.global_variables_initializer())
        feed_dict = {
            x: np.random.randn(10, 6, 6, 3).astype(np.float32),
            y: np.random.randint(5, size=10).astype(np.int64),
        }
        sess.run([loss, accuracy], feed_dict=feed_dict)

  def _build_toy_problem(self):
    """Construct a toy linear regression problem.

    Initial loss should be,
      2.5 = 0.5 * (1^2 + 2^2)

    Returns:
      loss: 0-D Tensor representing loss to be minimized.
      accuracy: 0-D Tensors representing model accuracy.
      layer_collection: LayerCollection instance describing model architecture.
    """
    x = np.asarray([[1.], [2.]]).astype(np.float32)
    y = np.asarray([1., 2.]).astype(np.float32)
    x, y = (tf.data.Dataset.from_tensor_slices((x, y))
            .repeat(100).batch(2).make_one_shot_iterator().get_next())
    w = tf.get_variable("w", shape=[1, 1], initializer=tf.zeros_initializer())
    y_hat = tf.matmul(x, w)
    loss = tf.reduce_mean(0.5 * tf.square(y_hat - y))
    accuracy = loss

    layer_collection = lc.LayerCollection()
    layer_collection.register_fully_connected(params=w, inputs=x, outputs=y_hat)
    layer_collection.register_normal_predictive_distribution(y_hat)

    return loss, accuracy, layer_collection

  def testMinimizeLossSingleMachine(self):
    with tf.Graph().as_default():
      loss, accuracy, layer_collection = self._build_toy_problem()
      accuracy_ = convnet.minimize_loss_single_machine(
          loss, accuracy, layer_collection, device="/cpu:0")
      self.assertLess(accuracy_, 2.0)

  def testMinimizeLossDistributed(self):
    with tf.Graph().as_default():
      loss, accuracy, layer_collection = self._build_toy_problem()
      accuracy_ = convnet.distributed_grads_only_and_ops_chief_worker(
          task_id=0,
          is_chief=True,
          num_worker_tasks=1,
          num_ps_tasks=0,
          master="",
          checkpoint_dir=None,
          loss=loss,
          accuracy=accuracy,
          layer_collection=layer_collection)
      self.assertLess(accuracy_, 2.0)

  def testTrainMnistSingleMachine(self):
    with tf.Graph().as_default():
      # Ensure model training doesn't crash.
      #
      # Ideally, we should check that accuracy increases as the model converges,
      # but there are too few parameters for the model to effectively memorize
      # the training set the way an MLP can.
      convnet.train_mnist_single_machine(
          data_dir=None, num_epochs=1, use_fake_data=True, device="/cpu:0")

  def testTrainMnistMultitower(self):
    with tf.Graph().as_default():
      # Ensure model training doesn't crash.
      convnet.train_mnist_multitower(
          data_dir=None, num_epochs=1, num_towers=2, use_fake_data=True)

  def testTrainMnistDistributed(self):
    with tf.Graph().as_default():
      # Ensure model training doesn't crash.
      convnet.train_mnist_distributed_sync_replicas(
          task_id=0,
          is_chief=True,
          num_worker_tasks=1,
          num_ps_tasks=0,
          master="",
          data_dir=None,
          num_epochs=1,
          op_strategy="chief_worker",
          use_fake_data=True)


if __name__ == "__main__":
  tf.test.main()
