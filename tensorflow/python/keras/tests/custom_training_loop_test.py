# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for custom training loops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensorflow.python import keras
from tensorflow.python.eager import backprop
from tensorflow.python.eager import def_function
from tensorflow.python.framework import ops
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class LayerWithLosses(keras.layers.Layer):

  def build(self, input_shape):
    self.v = self.add_weight(
        name='hey',
        shape=(),
        initializer='ones',
        regularizer=keras.regularizers.l1(100))

  def call(self, inputs):
    self.add_loss(math_ops.reduce_sum(inputs))
    return self.v * inputs


class LayerWithMetrics(keras.layers.Layer):

  def build(self, input_shape):
    self.mean = keras.metrics.Mean(name='mean_object')

  def call(self, inputs):
    self.add_metric(
        math_ops.reduce_mean(inputs), name='mean_tensor', aggregation='mean')
    self.add_metric(self.mean(inputs))
    return inputs


class LayerWithTrainingArg(keras.layers.Layer):

  def call(self, inputs, training=None):
    self.training = training
    if training:
      return inputs
    else:
      return 0. * inputs


def add_loss_step(defun):
  optimizer = keras.optimizer_v2.adam.Adam()
  model = testing_utils.get_model_from_layers([LayerWithLosses()],
                                              input_shape=(10,))

  def train_step(x):
    with backprop.GradientTape() as tape:
      model(x)
      assert len(model.losses) == 2
      loss = math_ops.reduce_sum(model.losses)
    gradients = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    return loss

  if defun:
    train_step = def_function.function(train_step)

  x = array_ops.ones((10, 10))
  return train_step(x)


def batch_norm_step(defun):
  optimizer = keras.optimizer_v2.adadelta.Adadelta()
  model = testing_utils.get_model_from_layers([
      keras.layers.BatchNormalization(momentum=0.9),
      keras.layers.Dense(1, kernel_initializer='zeros', activation='softmax')
  ],
                                              input_shape=(10,))

  def train_step(x, y):
    with backprop.GradientTape() as tape:
      y_pred = model(x, training=True)
      loss = keras.losses.binary_crossentropy(y, y_pred)
    gradients = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    return loss, model(x, training=False)

  if defun:
    train_step = def_function.function(train_step)

  x, y = array_ops.ones((10, 10)), array_ops.ones((10, 1))
  return train_step(x, y)


def add_metric_step(defun):
  optimizer = keras.optimizer_v2.rmsprop.RMSprop()
  model = testing_utils.get_model_from_layers([
      LayerWithMetrics(),
      keras.layers.Dense(1, kernel_initializer='zeros', activation='softmax')
  ],
                                              input_shape=(10,))

  def train_step(x, y):
    with backprop.GradientTape() as tape:
      y_pred_1 = model(x)
      y_pred_2 = model(2 * x)
      y_pred = y_pred_1 + y_pred_2
      loss = keras.losses.mean_squared_error(y, y_pred)
    gradients = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    assert len(model.metrics) == 2
    return [m.result() for m in model.metrics]

  if defun:
    train_step = def_function.function(train_step)

  x, y = array_ops.ones((10, 10)), array_ops.zeros((10, 1))
  metrics = train_step(x, y)
  assert np.allclose(metrics[0], 1.5)
  assert np.allclose(metrics[1], 1.5)
  return metrics


@keras_parameterized.run_with_all_model_types
class CustomTrainingLoopTest(keras_parameterized.TestCase):

  @parameterized.named_parameters(('add_loss_step', add_loss_step),
                                  ('add_metric_step', add_metric_step),
                                  ('batch_norm_step', batch_norm_step))
  def test_eager_and_tf_function(self, train_step):
    eager_result = train_step(defun=False)
    fn_result = train_step(defun=True)
    self.assertAllClose(eager_result, fn_result)

  @parameterized.named_parameters(('eager', False), ('defun', True))
  def test_training_arg_propagation(self, defun):

    model = testing_utils.get_model_from_layers([LayerWithTrainingArg()],
                                                input_shape=(1,))

    def train_step(x):
      return model(x), model(x, training=False), model(x, training=True)

    if defun:
      train_step = def_function.function(train_step)

    x = array_ops.ones((1, 1))
    results = train_step(x)
    self.assertAllClose(results[0], array_ops.zeros((1, 1)))
    self.assertAllClose(results[1], array_ops.zeros((1, 1)))
    self.assertAllClose(results[2], array_ops.ones((1, 1)))

  @parameterized.named_parameters(('eager', False), ('defun', True))
  def test_learning_phase_propagation(self, defun):

    class MyModel(keras.layers.Layer):

      def __init__(self):
        super(MyModel, self).__init__()
        self.layer = LayerWithTrainingArg()

      def call(self, inputs):
        return self.layer(inputs)

    model = MyModel()

    def train_step(x):
      no_learning_phase_out = model(x)
      self.assertFalse(model.layer.training)
      with keras.backend.learning_phase_scope(0):
        inf_learning_phase_out = model(x)
      self.assertEqual(model.layer.training, 0)
      with keras.backend.learning_phase_scope(1):
        train_learning_phase_out = model(x)
      self.assertEqual(model.layer.training, 1)
      return [
          no_learning_phase_out, inf_learning_phase_out,
          train_learning_phase_out
      ]

    if defun:
      train_step = def_function.function(train_step)

    x = array_ops.ones((1, 1))
    results = train_step(x)
    self.assertAllClose(results[0], array_ops.zeros((1, 1)))
    self.assertAllClose(results[1], array_ops.zeros((1, 1)))
    self.assertAllClose(results[2], array_ops.ones((1, 1)))

  @parameterized.named_parameters(('eager', False), ('defun', True))
  def test_training_arg_priorities(self, defun):

    class MyModel(keras.layers.Layer):

      def __init__(self):
        super(MyModel, self).__init__()
        self.layer = LayerWithTrainingArg()

      def call(self, inputs, training=False):
        return self.layer(inputs)

    model = MyModel()

    def train_step(x):
      explicit_out = model(x, training=True)
      default_out = model(x)
      with keras.backend.learning_phase_scope(1):
        parent_out = model(x, training=False)
        lr_out = model(x)
      return [explicit_out, default_out, parent_out, lr_out]

    if defun:
      train_step = def_function.function(train_step)

    x = array_ops.ones((1, 1))
    results = train_step(x)
    self.assertAllClose(results[0], array_ops.ones((1, 1)))
    self.assertAllClose(results[1], array_ops.zeros((1, 1)))
    self.assertAllClose(results[2], array_ops.zeros((1, 1)))
    self.assertAllClose(results[3], array_ops.ones((1, 1)))


if __name__ == '__main__':
  ops.enable_eager_execution()
  test.main()
