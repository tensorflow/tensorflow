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
"""Tests for compiled Model subclassing."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np

from tensorflow.python import keras
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import model_subclassing_test_util as model_util
from tensorflow.python.keras import testing_utils
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test

try:
  import h5py  # pylint:disable=g-import-not-at-top
except ImportError:
  h5py = None


@keras_parameterized.run_all_keras_modes
class ModelSubclassCompiledTest(keras_parameterized.TestCase):

  def test_single_io_workflow_with_np_arrays(self):
    num_classes = 2
    num_samples = 100
    input_dim = 50

    model = testing_utils.SmallSubclassMLP(
        num_hidden=32, num_classes=num_classes, use_dp=True, use_bn=True)
    model.compile(
        loss='mse',
        optimizer='rmsprop',
        metrics=['acc', keras.metrics.CategoricalAccuracy()],
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())

    x = np.ones((num_samples, input_dim))
    y = np.zeros((num_samples, num_classes))

    model.fit(x, y, epochs=2, batch_size=32, verbose=0)
    _ = model.evaluate(x, y, verbose=0)

  def test_multi_io_workflow_with_np_arrays(self):
    num_classes = (2, 3)
    num_samples = 1000
    input_dim = 50

    model = model_util.get_multi_io_subclass_model(
        num_classes=num_classes, use_dp=True, use_bn=True)
    model.compile(
        loss='mse',
        optimizer='rmsprop',
        metrics=['acc'],
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())

    x1 = np.ones((num_samples, input_dim))
    x2 = np.ones((num_samples, input_dim))
    y1 = np.zeros((num_samples, num_classes[0]))
    y2 = np.zeros((num_samples, num_classes[1]))

    model.fit([x1, x2], [y1, y2], epochs=2, batch_size=32, verbose=0)
    _ = model.evaluate([x1, x2], [y1, y2], verbose=0)

  def test_single_io_workflow_with_datasets(self):
    num_classes = 2
    num_samples = 10
    input_dim = 50

    with self.cached_session():
      model = testing_utils.SmallSubclassMLP(
          num_hidden=32, num_classes=num_classes, use_dp=True, use_bn=True)
      model.compile(
          loss='mse',
          optimizer='rmsprop',
          run_eagerly=testing_utils.should_run_eagerly(),
          experimental_run_tf_function=testing_utils.should_run_tf_function())

      x = np.ones((num_samples, input_dim), dtype=np.float32)
      y = np.zeros((num_samples, num_classes), dtype=np.float32)
      dataset = dataset_ops.Dataset.from_tensor_slices((x, y))
      dataset = dataset.repeat(100)
      dataset = dataset.batch(10)

      model.fit(dataset, epochs=2, steps_per_epoch=10, verbose=0)
      _ = model.evaluate(dataset, steps=10, verbose=0)

  def test_attributes(self):
    # layers, weights, trainable_weights, non_trainable_weights, inputs, outputs

    num_classes = (2, 3)
    num_samples = 100
    input_dim = 50

    model = model_util.get_multi_io_subclass_model(
        num_classes=num_classes, use_bn=True)

    x1 = np.ones((num_samples, input_dim))
    x2 = np.ones((num_samples, input_dim))
    y1 = np.zeros((num_samples, num_classes[0]))
    y2 = np.zeros((num_samples, num_classes[1]))

    self.assertEqual(model.name, 'test_model')
    self.assertEqual(model.built, False)
    self.assertEqual(len(model.weights), 0)

    model.compile(
        loss='mse',
        optimizer='rmsprop',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    model.train_on_batch([x1, x2], [y1, y2])

    self.assertEqual(model.built, True)
    self.assertEqual(len(model.layers), 4)
    self.assertEqual(len(model.weights), 10)
    self.assertEqual(len(model.trainable_weights), 8)
    self.assertEqual(len(model.non_trainable_weights), 2)
    self.assertEqual(len(model.inputs), 2)
    self.assertEqual(len(model.outputs), 2)

  def test_updates(self):
    # test that updates get run during training
    num_samples = 100
    input_dim = 50

    class BNNet(keras.Model):

      def __init__(self):
        super(BNNet, self).__init__()
        self.bn = keras.layers.BatchNormalization(beta_initializer='ones',
                                                  gamma_initializer='ones')

      def call(self, inputs):
        return self.bn(inputs)

    x = np.ones((num_samples, input_dim))
    y = np.ones((num_samples, input_dim))

    model = BNNet()
    model.compile(
        loss='mse',
        optimizer='rmsprop',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    y_ref = model.predict(x)

    model.train_on_batch(x, y)
    y_new = model.predict(x)
    self.assertGreater(np.sum(np.abs(y_ref - y_new)), 0.1)

  def test_training_and_inference_behavior(self):
    # test that dropout is applied in training and not inference

    num_samples = 100
    input_dim = 50

    class DPNet(keras.Model):

      def __init__(self):
        super(DPNet, self).__init__()
        self.dp = keras.layers.Dropout(0.5)
        self.dense = keras.layers.Dense(1,
                                        use_bias=False,
                                        kernel_initializer='ones')

      def call(self, inputs):
        x = self.dp(inputs)
        return self.dense(x)

    model = DPNet()
    x = np.ones((num_samples, input_dim))
    y = model.predict(x)
    self.assertEqual(np.sum(y), np.sum(x))
    model.compile(
        loss='mse',
        optimizer='rmsprop',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    loss = model.train_on_batch(x, y)
    self.assertGreater(loss, 0.1)

  def test_training_methods(self):
    # test fit, train_on_batch
    # on different input types: list, dict

    num_classes = (2, 3)
    num_samples = 100
    input_dim = 50

    x1 = np.ones((num_samples, input_dim))
    x2 = np.ones((num_samples, input_dim))
    y1 = np.zeros((num_samples, num_classes[0]))
    y2 = np.zeros((num_samples, num_classes[1]))

    model = model_util.get_multi_io_subclass_model(
        num_classes=num_classes, use_bn=True)
    model.compile(
        loss='mse',
        optimizer='rmsprop',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    model.fit([x1, x2], [y1, y2], epochs=2, batch_size=32, verbose=0)
    model.fit({'input_1': x1, 'input_2': x2},
              {'output_1': y1, 'output_2': y2},
              epochs=2, batch_size=32)
    model.fit([x1, x2], [y1, y2], epochs=2, batch_size=32, verbose=0,
              validation_data=([x1, x2], [y1, y2]))

    model = model_util.get_multi_io_subclass_model(
        num_classes=num_classes, use_bn=True)
    model.compile(
        loss='mse',
        optimizer='rmsprop',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    model.train_on_batch([x1, x2], [y1, y2])
    model.train_on_batch({'input_1': x1, 'input_2': x2},
                         {'output_1': y1, 'output_2': y2})

  def test_inference_methods(self):
    # test predict, evaluate, test_on_batch, predict_on_batch
    # on different input types: list, dict
    num_classes = (2, 3)
    num_samples = 100
    input_dim = 50

    x1 = np.ones((num_samples, input_dim))
    x2 = np.ones((num_samples, input_dim))
    y1 = np.zeros((num_samples, num_classes[0]))
    y2 = np.zeros((num_samples, num_classes[1]))

    model = model_util.get_multi_io_subclass_model(
        num_classes=num_classes, use_bn=True)
    model.compile(
        loss='mse',
        optimizer='rmsprop',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    model.evaluate([x1, x2], [y1, y2])
    model.test_on_batch([x1, x2], [y1, y2])

    model = model_util.get_multi_io_subclass_model(
        num_classes=num_classes, use_bn=True)
    model.predict([x1, x2])

    model = model_util.get_multi_io_subclass_model(
        num_classes=num_classes, use_bn=True)
    model.predict_on_batch([x1, x2])

  def test_saving(self):
    num_classes = (2, 3)
    num_samples = 100
    input_dim = 50

    x1 = np.ones((num_samples, input_dim))
    x2 = np.ones((num_samples, input_dim))
    y1 = np.zeros((num_samples, num_classes[0]))
    y2 = np.zeros((num_samples, num_classes[1]))

    model = model_util.get_multi_io_subclass_model(
        num_classes=num_classes, use_bn=True)
    model.compile(
        loss='mse',
        optimizer='rmsprop',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    model.fit([x1, x2], [y1, y2], epochs=2, batch_size=32, verbose=0)
    y_ref_1, y_ref_2 = model.predict([x1, x2])

    tf_format_name = os.path.join(self.get_temp_dir(), 'ckpt')
    model.save_weights(tf_format_name)
    if h5py is not None:
      hdf5_format_name = os.path.join(self.get_temp_dir(), 'weights.h5')
      model.save_weights(hdf5_format_name)

    model = model_util.get_multi_io_subclass_model(
        num_classes=num_classes, use_bn=True)

    if h5py is not None:
      with self.assertRaises(ValueError):
        model.load_weights(hdf5_format_name)

    model.load_weights(tf_format_name)

    y1, y2 = model.predict([x1, x2])
    self.assertAllClose(y_ref_1, y1, atol=1e-5)
    self.assertAllClose(y_ref_2, y2, atol=1e-5)

    if h5py is not None:
      model.load_weights(hdf5_format_name)

      y1, y2 = model.predict([x1, x2])
      self.assertAllClose(y_ref_1, y1, atol=1e-5)
      self.assertAllClose(y_ref_2, y2, atol=1e-5)

  def test_subclass_nested_in_subclass(self):
    num_classes = 2
    num_samples = 100
    input_dim = 50

    model = model_util.NestedTestModel1(num_classes=num_classes)
    model.compile(
        loss='mse',
        optimizer='rmsprop',
        metrics=['acc'],
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())

    x = np.ones((num_samples, input_dim))
    y = np.zeros((num_samples, num_classes))

    model.fit(x, y, epochs=2, batch_size=32, verbose=0)
    _ = model.evaluate(x, y, verbose=0)

    self.assertEqual(len(model.weights), 8 + len(model.test_net.weights))
    self.assertEqual(len(model.non_trainable_weights),
                     2 + len(model.test_net.non_trainable_weights))
    self.assertEqual(len(model.trainable_weights),
                     6 + len(model.test_net.trainable_weights))

  def test_graph_nested_in_subclass(self):
    num_classes = 2
    num_samples = 100
    input_dim = 50

    model = model_util.NestedTestModel2(num_classes=num_classes)
    model.compile(
        loss='mse',
        optimizer='rmsprop',
        metrics=['acc'],
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())

    x = np.ones((num_samples, input_dim))
    y = np.zeros((num_samples, num_classes))

    model.fit(x, y, epochs=2, batch_size=32, verbose=0)
    _ = model.evaluate(x, y, verbose=0)

    self.assertEqual(len(model.weights), 8 + len(model.test_net.weights))
    self.assertEqual(len(model.non_trainable_weights),
                     2 + len(model.test_net.non_trainable_weights))
    self.assertEqual(len(model.trainable_weights),
                     6 + len(model.test_net.trainable_weights))

  def test_subclass_nested_in_graph(self):
    num_classes = 2
    num_samples = 100
    input_dim = 50

    model = model_util.get_nested_model_3(
        input_dim=input_dim, num_classes=num_classes)
    model.compile(
        loss='mse',
        optimizer='rmsprop',
        metrics=['acc'],
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())

    x = np.ones((num_samples, input_dim))
    y = np.zeros((num_samples, num_classes))

    model.fit(x, y, epochs=2, batch_size=32, verbose=0)
    _ = model.evaluate(x, y, verbose=0)

    self.assertEqual(len(model.weights), 16)
    self.assertEqual(len(model.non_trainable_weights), 4)
    self.assertEqual(len(model.trainable_weights), 12)

  def test_subclass_nested_in_sequential(self):
    num_classes = 2
    num_samples = 100
    input_dim = 50

    class Inner(keras.Model):

      def __init__(self):
        super(Inner, self).__init__()
        self.dense1 = keras.layers.Dense(32, activation='relu')
        self.dense2 = keras.layers.Dense(num_classes, activation='relu')
        self.bn = keras.layers.BatchNormalization()

      def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.bn(x)

    model = keras.Sequential([Inner()])
    model.compile(
        loss='mse',
        optimizer='rmsprop',
        metrics=['acc'],
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())

    x = np.ones((num_samples, input_dim))
    y = np.zeros((num_samples, num_classes))
    model.fit(x, y, epochs=2, batch_size=32, verbose=0)
    _ = model.evaluate(x, y, verbose=0)

    self.assertEqual(len(model.weights), 8)
    self.assertEqual(len(model.non_trainable_weights), 2)
    self.assertEqual(len(model.trainable_weights), 6)

  def test_support_for_manual_training_arg(self):
    # In most cases, the `training` argument is left unspecified, in which
    # case it defaults to value corresponding to the Model method being used
    # (fit -> True, predict -> False, etc).
    # If the user writes their model `call` method to take
    # an explicit `training` argument, we must check that the correct value
    # is being passed to the model for each method call.

    class DPNet(keras.Model):

      def __init__(self):
        super(DPNet, self).__init__()
        self.dp = keras.layers.Dropout(0.5)
        self.dense = keras.layers.Dense(1,
                                        use_bias=False,
                                        kernel_initializer='ones')

      def call(self, inputs, training=False):
        x = self.dp(inputs, training=training)
        return self.dense(x)

    model = DPNet()
    x = np.ones((10, 10))
    y = model.predict(x)
    self.assertEqual(np.sum(y), np.sum(x))
    model.compile(
        loss='mse',
        optimizer='rmsprop',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    loss = model.train_on_batch(x, y)
    self.assertGreater(loss, 0.1)

  def test_no_loss_in_compile(self):

    class InternalLossModel(keras.Model):

      def __init__(self):
        super(InternalLossModel, self).__init__()
        self.dense = keras.layers.Dense(1)

      def call(self, inputs):
        out = self.dense(inputs)
        self.add_loss(math_ops.reduce_sum(out))
        return out

    model = InternalLossModel()
    x = np.ones((10, 10))
    model.predict(x)
    model.compile(
        optimizer='rmsprop',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    model.fit(x)
    model.evaluate(x)


if __name__ == '__main__':
  test.main()
