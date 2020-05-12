# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras import layers, optimizers


def _get_big_cnn_model(img_dim, n_channels, num_partitions,
                       blocks_per_partition):
  """Creates a test model whose activations are significantly larger than model size."""
  model = tf.keras.Sequential()
  model.add(layers.Input(shape=(img_dim, img_dim, n_channels)))
  for _ in range(num_partitions):
    for _ in range(blocks_per_partition):
      model.add(layers.Conv2D(10, 5, padding='same', activation=tf.nn.relu))
      model.add(layers.MaxPooling2D((1, 1), padding='same'))
      model.add(layers.Conv2D(40, 5, padding='same', activation=tf.nn.relu))
      model.add(layers.MaxPooling2D((1, 1), padding='same'))
      model.add(layers.Conv2D(20, 5, padding='same', activation=tf.nn.relu))
      model.add(layers.MaxPooling2D((1, 1), padding='same'))
  model.add(layers.Flatten())
  model.add(layers.Dense(32, activation=tf.nn.relu))
  model.add(layers.Dense(10))
  return model


def _get_split_cnn_model(img_dim, n_channels, num_partitions,
                         blocks_per_partition):
  """Creates a test model that is split into `num_partitions` smaller models"""
  models = [tf.keras.Sequential() for _ in range(num_partitions)]
  models[0].add(layers.Input(shape=(img_dim, img_dim, n_channels)))
  for i in range(num_partitions):
    model = models[i]
    if i > 0:
      last_shape = models[i - 1].layers[-1].output_shape
      model.add(layers.Input(shape=last_shape[1:]))
    for _ in range(blocks_per_partition):
      model.add(layers.Conv2D(10, 5, padding='same', activation=tf.nn.relu))
      model.add(layers.MaxPooling2D((1, 1), padding='same'))
      model.add(layers.Conv2D(40, 5, padding='same', activation=tf.nn.relu))
      model.add(layers.MaxPooling2D((1, 1), padding='same'))
      model.add(layers.Conv2D(20, 5, padding='same', activation=tf.nn.relu))
      model.add(layers.MaxPooling2D((1, 1), padding='same'))
  models[-1].add(layers.Flatten())
  models[-1].add(layers.Dense(32, activation=tf.nn.relu))
  models[-1].add(layers.Dense(10))
  return models


def _compute_loss(logits, labels):
  return tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                     labels=labels))


def _limit_gpu_memory():
  """Helper function to limit GPU memory for testing  """
  gpus = tf.config.experimental.list_physical_devices('GPU')
  if gpus:
    try:
      tf.config.experimental.set_virtual_device_configuration(
          gpus[0], [
              tf.config.experimental.VirtualDeviceConfiguration(
                  memory_limit=1024)
          ])
    except RuntimeError as e:
      print(e)


def _get_dummy_data(img_dim, n_channels, batch_size):
  inputs = tf.ones([batch_size, img_dim, img_dim, n_channels])
  labels = tf.ones([batch_size], dtype=tf.int64)
  return inputs, labels


def _train_no_recompute(n_steps):
  """Trains a single large model without gradient checkpointing."""
  _limit_gpu_memory()
  img_dim, n_channels, batch_size = 256, 1, 4
  x, y = _get_dummy_data(img_dim, n_channels, batch_size)
  model = _get_big_cnn_model(img_dim,
                             n_channels,
                             num_partitions=3,
                             blocks_per_partition=2)
  optimizer = optimizers.SGD()
  losses = []
  tr_vars = model.trainable_variables
  for _ in range(n_steps):
    with tf.GradientTape() as tape:
      logits = model(x)
      loss = _compute_loss(logits, y)
      losses.append(loss)
    grads = tape.gradient(loss, tr_vars)  # tr_vars
    optimizer.apply_gradients(zip(grads, tr_vars))
    del grads
  return losses


def _train_with_recompute(n_steps):
  """Trains a single large model with gradient checkpointing using tf.recompute_grad."""
  _limit_gpu_memory()
  img_dim, n_channels, batch_size = 256, 1, 4
  x, y = _get_dummy_data(img_dim, n_channels, batch_size)
  # This model is the same model as _get_big_cnn_model but split into 3 parts.
  models = _get_split_cnn_model(img_dim,
                                n_channels,
                                num_partitions=3,
                                blocks_per_partition=2)
  model1, model2, model3 = models
  # Apply gradient checkpointing to the submodels using tf.recompute_grad.
  model1_re = tf.recompute_grad(model1)
  model2_re = tf.recompute_grad(model2)
  model3_re = tf.recompute_grad(model3)
  optimizer = optimizers.SGD()
  tr_vars = (model1.trainable_variables + model2.trainable_variables +
             model3.trainable_variables)
  losses = []
  for _ in range(n_steps):
    with tf.GradientTape() as tape:
      logits1 = model1_re(x)
      logits2 = model2_re(logits1)
      logits3 = model3_re(logits2)
      loss = _compute_loss(logits3, y)
      losses.append(loss)
      grads = tape.gradient(loss, tr_vars)  # tr_vars
      optimizer.apply_gradients(zip(grads, tr_vars))
      del grads
  return losses


class GradientCheckpointTest(tf.test.TestCase):

  def test_raises_oom_exception(self):
    with self.assertRaises(Exception) as context:
      _train_no_recompute(1)
    self.assertTrue(
        context.exception.__class__.__name__ == 'ResourceExhaustedError')

  def test_does_not_raise_oom_exception(self):
    n_step = 2
    losses = _train_with_recompute(n_step)
    self.assertTrue(len(losses) == n_step)


if __name__ == "__main__":
  tf.test.main()
