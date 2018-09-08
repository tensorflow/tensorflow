# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tf.keras models using DistributionStrategy."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

from tensorflow.contrib.distribute.python import mirrored_strategy
from tensorflow.contrib.distribute.python import values
from tensorflow.python import keras
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.estimator import keras as keras_lib
from tensorflow.python.estimator import run_config as run_config_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.engine import distributed_training_utils
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.summary.writer import writer_cache
from tensorflow.python.training import gradient_descent
from tensorflow.python.training import rmsprop


_RANDOM_SEED = 1337
_TRAIN_SIZE = 200
_INPUT_SIZE = (10,)
_NUM_CLASS = 2


# TODO(anjalisridhar): Add a decorator that will allow us to run these tests as
# part of the tf.keras unit tests suite.
def simple_sequential_model():
  model = keras.models.Sequential()
  model.add(keras.layers.Dense(16, activation='relu', input_shape=_INPUT_SIZE))
  model.add(keras.layers.Dropout(0.1))
  model.add(keras.layers.Dense(_NUM_CLASS, activation='softmax'))
  return model


def simple_functional_model():
  a = keras.layers.Input(shape=_INPUT_SIZE)
  b = keras.layers.Dense(16, activation='relu')(a)
  b = keras.layers.Dropout(0.1)(b)
  b = keras.layers.Dense(_NUM_CLASS, activation='softmax')(b)
  model = keras.models.Model(inputs=[a], outputs=[b])
  return model


def get_ds_train_input_fn():
  np.random.seed(_RANDOM_SEED)
  (x_train, y_train), _ = testing_utils.get_test_data(
      train_samples=_TRAIN_SIZE,
      test_samples=50,
      input_shape=_INPUT_SIZE,
      num_classes=_NUM_CLASS)
  y_train = keras.utils.to_categorical(y_train)

  dataset = dataset_ops.Dataset.from_tensor_slices((x_train, y_train))
  dataset = dataset.batch(32)
  return dataset


def get_ds_test_input_fn():
  np.random.seed(_RANDOM_SEED)
  _, (x_test, y_test) = testing_utils.get_test_data(
      train_samples=_TRAIN_SIZE,
      test_samples=50,
      input_shape=_INPUT_SIZE,
      num_classes=_NUM_CLASS)
  y_test = keras.utils.to_categorical(y_test)

  dataset = dataset_ops.Dataset.from_tensor_slices((x_test, y_test))
  dataset = dataset.batch(32)
  return dataset


class TestEstimatorDistributionStrategy(test_util.TensorFlowTestCase):

  def setUp(self):
    self._base_dir = os.path.join(self.get_temp_dir(),
                                  'keras_mirrored_strategy_test')
    gfile.MakeDirs(self._base_dir)
    self._config = run_config_lib.RunConfig(
        tf_random_seed=_RANDOM_SEED, model_dir=self._base_dir)

  def tearDown(self):
    writer_cache.FileWriterCache.clear()
    if os.path.isdir(self._base_dir):
      gfile.DeleteRecursively(self._base_dir)

  def test_train_functional_with_distribution_strategy(self):
    dist = mirrored_strategy.MirroredStrategy(
        devices=['/device:GPU:0', '/device:GPU:1'])
    keras_model = simple_functional_model()
    keras_model.compile(
        loss='categorical_crossentropy',
        optimizer=rmsprop.RMSPropOptimizer(learning_rate=0.01))
    config = run_config_lib.RunConfig(tf_random_seed=_RANDOM_SEED,
                                      model_dir=self._base_dir,
                                      train_distribute=dist,
                                      eval_distribute=dist)
    with self.cached_session():
      est_keras = keras_lib.model_to_estimator(
          keras_model=keras_model, config=config)
      before_eval_results = est_keras.evaluate(
          input_fn=get_ds_test_input_fn, steps=1)
      est_keras.train(input_fn=get_ds_train_input_fn, steps=_TRAIN_SIZE / 16)
      after_eval_results = est_keras.evaluate(input_fn=get_ds_test_input_fn,
                                              steps=1)
      self.assertLess(after_eval_results['loss'], before_eval_results['loss'])

    writer_cache.FileWriterCache.clear()
    gfile.DeleteRecursively(self._config.model_dir)

  def test_train_sequential_with_distribution_strategy(self):
    dist = mirrored_strategy.MirroredStrategy(
        devices=['/device:GPU:0', '/device:GPU:1'])
    keras_model = simple_sequential_model()
    keras_model.compile(
        loss='categorical_crossentropy',
        optimizer=rmsprop.RMSPropOptimizer(learning_rate=0.01))
    config = run_config_lib.RunConfig(tf_random_seed=_RANDOM_SEED,
                                      model_dir=self._base_dir,
                                      train_distribute=dist)
    with self.cached_session():
      est_keras = keras_lib.model_to_estimator(
          keras_model=keras_model, config=config)
      before_eval_results = est_keras.evaluate(
          input_fn=get_ds_test_input_fn, steps=1)
      est_keras.train(input_fn=get_ds_train_input_fn, steps=_TRAIN_SIZE / 16)
      after_eval_results = est_keras.evaluate(input_fn=get_ds_test_input_fn,
                                              steps=1)
      self.assertLess(after_eval_results['loss'], before_eval_results['loss'])

    writer_cache.FileWriterCache.clear()
    gfile.DeleteRecursively(self._config.model_dir)

  def test_keras_optimizer_with_distribution_strategy(self):
    dist = mirrored_strategy.MirroredStrategy(
        devices=['/device:GPU:0', '/device:GPU:1'])
    keras_model = simple_sequential_model()
    keras_model.compile(
        loss='categorical_crossentropy',
        optimizer=keras.optimizers.rmsprop(lr=0.01))

    config = run_config_lib.RunConfig(tf_random_seed=_RANDOM_SEED,
                                      model_dir=self._base_dir,
                                      train_distribute=dist)
    with self.cached_session():
      est_keras = keras_lib.model_to_estimator(keras_model=keras_model,
                                               config=config)
      with self.assertRaisesRegexp(ValueError,
                                   'Only TensorFlow native optimizers are '
                                   'supported with DistributionStrategy.'):
        est_keras.train(input_fn=get_ds_train_input_fn, steps=_TRAIN_SIZE / 16)

    writer_cache.FileWriterCache.clear()
    gfile.DeleteRecursively(self._config.model_dir)


class TestWithDistributionStrategy(test.TestCase):

  def test_validating_dataset_input_tensors_with_shape_mismatch(self):
    with self.cached_session():
      strategy = mirrored_strategy.MirroredStrategy(['/device:GPU:0',
                                                     '/device:CPU:0'])
      a = constant_op.constant([1, 2], shape=(1, 2))
      b = constant_op.constant([[1, 2], [1, 2]], shape=(2, 2))
      x = values.DistributedValues({'/device:CPU:0': a, '/device:GPU:0': b})
      y = values.DistributedValues({'/device:CPU:0': a, '/device:GPU:0': a})
      with strategy.scope():
        # Removed device and input tensor shape details from the error message
        # since the order of the device and the corresponding input tensor shape
        # is not deterministic over different runs.
        with self.assertRaisesRegexp(ValueError,
                                     'Input tensor shapes do not match for '
                                     'distributed tensor inputs '
                                     'DistributedValues:.+'):
          distributed_training_utils.validate_distributed_dataset_inputs(
              strategy, x, y)

  def test_validating_dataset_input_tensors_with_dtype_mismatch(self):
    with self.cached_session():
      strategy = mirrored_strategy.MirroredStrategy(['/device:GPU:0',
                                                     '/device:CPU:0'])
      a = constant_op.constant([1, 2], shape=(1, 2), dtype=dtypes.int32)
      b = constant_op.constant([1, 2], shape=(1, 2), dtype=dtypes.float64)
      x = values.DistributedValues({'/device:CPU:0': a, '/device:GPU:0': b})
      y = values.DistributedValues({'/device:CPU:0': a, '/device:GPU:0': a})
      with strategy.scope():
        # Removed device and input tensor dtype details from the error message
        # since the order of the device and the corresponding input tensor dtype
        # is not deterministic over different runs.
        with self.assertRaisesRegexp(ValueError,
                                     'Input tensor dtypes do not match for '
                                     'distributed tensor inputs '
                                     'DistributedValues:.+'):
          distributed_training_utils.validate_distributed_dataset_inputs(
              strategy, x, y)

  def test_calling_model_on_same_dataset(self):
    with self.cached_session():
      x = keras.layers.Input(shape=(3,), name='input')
      y = keras.layers.Dense(4, name='dense')(x)
      model = keras.Model(x, y)

      optimizer = gradient_descent.GradientDescentOptimizer(0.001)
      loss = 'mse'
      metrics = ['mae']
      strategy = mirrored_strategy.MirroredStrategy(['/device:GPU:1',
                                                     '/device:GPU:0'])
      model.compile(optimizer, loss, metrics=metrics, distribute=strategy)

      inputs = np.zeros((10, 3), dtype=np.float32)
      targets = np.zeros((10, 4), dtype=np.float32)
      dataset = dataset_ops.Dataset.from_tensor_slices((inputs, targets))
      dataset = dataset.repeat(100)
      dataset = dataset.batch(10)

      # Call fit with validation data
      model.fit(dataset, epochs=1, steps_per_epoch=2, verbose=0,
                validation_data=dataset, validation_steps=2)
      model.fit(dataset, epochs=1, steps_per_epoch=2, verbose=0,
                validation_data=dataset, validation_steps=2)
      model.predict(dataset, steps=2)

  def test_fit_with_tuple_and_dict_dataset_inputs(self):
    with self.cached_session():
      a = keras.layers.Input(shape=(3,), name='input_a')
      b = keras.layers.Input(shape=(3,), name='input_b')

      dense = keras.layers.Dense(4, name='dense')
      c = dense(a)
      d = dense(b)
      e = keras.layers.Dropout(0.5, name='dropout')(c)

      model = keras.models.Model([a, b], [d, e])

      optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=0.001)
      loss = 'mse'
      metrics = ['mae']
      strategy = mirrored_strategy.MirroredStrategy(['/device:GPU:0',
                                                     '/device:CPU:0'])
      model.compile(optimizer, loss, metrics=metrics, distribute=strategy)

      input_a_np = np.random.random((10, 3))
      input_b_np = np.random.random((10, 3))
      output_d_np = np.random.random((10, 4))
      output_e_np = np.random.random((10, 4))

      # Test with tuples
      dataset_tuple = dataset_ops.Dataset.from_tensor_slices((
          (input_a_np, input_b_np), (output_d_np, output_e_np)))
      dataset_tuple = dataset_tuple.repeat(100)
      dataset_tuple = dataset_tuple.batch(10)

      model.fit(dataset_tuple, epochs=1, steps_per_epoch=2, verbose=1)

      # Test with dict
      dataset_dict = dataset_ops.Dataset.from_tensor_slices((
          {'input_a': input_a_np, 'input_b': input_b_np},
          (output_d_np, output_e_np)))
      dataset_dict = dataset_dict.repeat(100)
      dataset_dict = dataset_dict.batch(10)

      model.fit(dataset_dict, epochs=1, steps_per_epoch=2, verbose=1)

  def test_fit_eval_and_predict_methods_on_dataset(self):
    with self.cached_session():
      x = keras.layers.Input(shape=(3,), name='input')
      y = keras.layers.Dense(4, name='dense')(x)
      model = keras.Model(x, y)

      optimizer = gradient_descent.GradientDescentOptimizer(0.001)
      loss = 'mse'
      metrics = ['mae']
      strategy = mirrored_strategy.MirroredStrategy(['/device:GPU:0',
                                                     '/device:CPU:0'])

      model.compile(optimizer, loss, metrics=metrics, distribute=strategy)

      inputs = np.zeros((10, 3), dtype=np.float32)
      targets = np.zeros((10, 4), dtype=np.float32)
      dataset = dataset_ops.Dataset.from_tensor_slices((inputs, targets))
      dataset = dataset.repeat(100)
      dataset = dataset.batch(10)

      model.fit(dataset, epochs=1, steps_per_epoch=2, verbose=1)
      model.evaluate(dataset, steps=2, verbose=1)
      model.predict(dataset, steps=2)
      # Test with validation data
      model.fit(dataset, epochs=1, steps_per_epoch=2, verbose=0,
                validation_data=dataset, validation_steps=2)

  def test_raise_error_for_stateful_metrics(self):

    class ExampleStatefulMetric(keras.layers.Layer):

      def __init__(self, name='true_positives', **kwargs):
        super(ExampleStatefulMetric, self).__init__(name=name, **kwargs)
        self.stateful = True

      def __call__(self, y_true, y_pred):
        return y_pred - y_true

    with self.cached_session():
      x = keras.layers.Input(shape=(3,), name='input')
      y = keras.layers.Dense(4, name='dense')(x)
      model = keras.Model(x, y)

      optimizer = gradient_descent.GradientDescentOptimizer(0.001)
      loss = 'mse'
      metrics = ['mae', ExampleStatefulMetric()]
      strategy = mirrored_strategy.MirroredStrategy(['/device:GPU:1',
                                                     '/device:GPU:0'])
      with self.assertRaisesRegexp(
          NotImplementedError, 'Stateful metrics are not supported with '
                               'DistributionStrategy.'):
        model.compile(optimizer, loss, metrics=metrics, distribute=strategy)

  def test_unsupported_features(self):
    with self.cached_session():
      x = keras.layers.Input(shape=(3,), name='input')
      y = keras.layers.Dense(4, name='dense')(x)
      model = keras.Model(x, y)

      optimizer = gradient_descent.GradientDescentOptimizer(0.001)
      loss = 'mse'
      metrics = ['mae']
      strategy = mirrored_strategy.MirroredStrategy(['/device:GPU:1',
                                                     '/device:GPU:0'])

      model.compile(optimizer, loss, metrics=metrics, distribute=strategy)

      inputs = np.zeros((10, 3), dtype=np.float32)
      targets = np.zeros((10, 4), dtype=np.float32)
      dataset = dataset_ops.Dataset.from_tensor_slices((inputs, targets))
      dataset = dataset.repeat(100)
      dataset = dataset.batch(10)

      # Test with validation split
      with self.assertRaisesRegexp(
          ValueError, '`validation_split` argument is not '
                      'supported when input `x` is a dataset or a '
                      'dataset iterator.+'):
        model.fit(dataset,
                  epochs=1, steps_per_epoch=2, verbose=0,
                  validation_split=0.5, validation_steps=2)

      # Test with sample weight.
      sample_weight = np.random.random((10,))
      with self.assertRaisesRegexp(
          NotImplementedError, '`sample_weight` is currently not supported '
                               'when using DistributionStrategy.'):
        model.fit(
            dataset,
            epochs=1,
            steps_per_epoch=2,
            verbose=0,
            sample_weight=sample_weight)

      # Test with not specifying the `steps` argument.
      with self.assertRaisesRegexp(
          ValueError, 'you should specify the `steps_per_epoch` argument'):
        model.fit(dataset, epochs=1, verbose=0)
      with self.assertRaisesRegexp(ValueError,
                                   'you should specify the `steps` argument'):
        model.evaluate(dataset, verbose=0)

      with self.assertRaisesRegexp(ValueError,
                                   'you should specify the `steps` argument'):
        model.predict(dataset, verbose=0)

  def test_calling_with_unsupported_predefined_callbacks(self):
    with self.cached_session():
      x = keras.layers.Input(shape=(3,), name='input')
      y = keras.layers.Dense(4, name='dense')(x)
      model = keras.Model(x, y)

      optimizer = gradient_descent.GradientDescentOptimizer(0.001)
      loss = 'mse'
      metrics = ['mae']
      strategy = mirrored_strategy.MirroredStrategy(['/device:GPU:1',
                                                     '/device:GPU:0'])
      model.compile(optimizer, loss, metrics=metrics, distribute=strategy)

      inputs = np.zeros((10, 3), dtype=np.float32)
      targets = np.zeros((10, 4), dtype=np.float32)
      dataset = dataset_ops.Dataset.from_tensor_slices((inputs, targets))
      dataset = dataset.repeat(100)
      dataset = dataset.batch(10)

      def schedule(_):
        return 0.001
      with self.assertRaisesRegexp(ValueError,
                                   'LearningRateScheduler callback is not '
                                   'supported with DistributionStrategy.'):
        model.fit(dataset, epochs=1, steps_per_epoch=2, verbose=0,
                  callbacks=[keras.callbacks.LearningRateScheduler(schedule)])

      with self.assertRaisesRegexp(ValueError,
                                   'ReduceLROnPlateau callback is not '
                                   'supported with DistributionStrategy.'):
        model.fit(dataset, epochs=1, steps_per_epoch=2, verbose=0,
                  callbacks=[keras.callbacks.ReduceLROnPlateau()])
      with self.assertRaisesRegexp(ValueError,
                                   'histogram_freq in the TensorBoard callback '
                                   'is not supported when using '
                                   'DistributionStrategy.'):
        model.fit(dataset, epochs=1, steps_per_epoch=2, verbose=0,
                  callbacks=[keras.callbacks.TensorBoard(histogram_freq=10)])

  def test_dataset_input_shape_validation(self):
    with self.cached_session():
      x = keras.layers.Input(shape=(3,), name='input')
      y = keras.layers.Dense(4, name='dense')(x)
      model = keras.Model(x, y)

      optimizer = rmsprop.RMSPropOptimizer(learning_rate=0.001)
      loss = 'mse'
      strategy = mirrored_strategy.MirroredStrategy(['/device:GPU:1',
                                                     '/device:GPU:0'])

      model.compile(optimizer, loss, distribute=strategy)

      # User forgets to batch the dataset
      inputs = np.zeros((10, 3), dtype=np.float32)
      targets = np.zeros((10, 4), dtype=np.float32)
      dataset = dataset_ops.Dataset.from_tensor_slices((inputs, targets))
      dataset = dataset.repeat(100)

      with self.assertRaisesRegexp(ValueError, 'expected input to have shape'):
        model.fit(dataset, epochs=1, steps_per_epoch=2, verbose=0)

      # Wrong input shape
      inputs = np.zeros((10, 5), dtype=np.float32)
      targets = np.zeros((10, 4), dtype=np.float32)
      dataset = dataset_ops.Dataset.from_tensor_slices((inputs, targets))
      dataset = dataset.repeat(100)
      dataset = dataset.batch(10)

      with self.assertRaisesRegexp(ValueError,
                                   'expected input to have shape'):
        model.fit(dataset, epochs=1, steps_per_epoch=2, verbose=0)

  def test_learning_phase_value(self):
    # TODO(anjalisridhar): Modify this test to use Lambdas since we can compare
    # meaningful values. Currently we don't pass the learning phase if the
    # Lambda layer uses the learning phase.
    with self.cached_session():
      x = keras.layers.Input(shape=(16,), name='input')
      y = keras.layers.Dense(16)(x)
      z = keras.layers.Dropout(0.9999)(y)
      model = keras.Model(x, z)

      optimizer = gradient_descent.GradientDescentOptimizer(0.005)
      loss = 'mse'
      metrics = ['acc']
      strategy = mirrored_strategy.MirroredStrategy(['/device:GPU:0',
                                                     '/device:CPU:0'])

      model.compile(optimizer, loss, metrics=metrics, distribute=strategy)

      inputs = np.random.rand(10, 16)
      targets = np.ones((10, 16), dtype=np.float32)
      dataset = dataset_ops.Dataset.from_tensor_slices((inputs, targets))
      dataset = dataset.repeat(100)
      dataset = dataset.batch(8)

      hist = model.fit(dataset, epochs=5, steps_per_epoch=20, verbose=1)
      self.assertEqual(hist.history['acc'][0], 1)

      evaluate_output = model.evaluate(dataset, steps=20)
      self.assertEqual(evaluate_output[1], 0)

      predict_output = model.predict(dataset, steps=1)
      self.assertNotEqual(np.mean(predict_output), 0)


class LossMaskingWithDistributionStrategyTest(test.TestCase):

  def test_masking(self):
    with self.cached_session():
      np.random.seed(1337)
      x = np.array([[[1], [1]], [[0], [0]]])
      model = keras.models.Sequential()
      model.add(keras.layers.Masking(mask_value=0, input_shape=(2, 1)))
      model.add(
          keras.layers.TimeDistributed(
              keras.layers.Dense(1, kernel_initializer='one')))
      strategy = mirrored_strategy.MirroredStrategy(['/device:GPU:1',
                                                     '/device:GPU:0'])

      model.compile(loss='mse',
                    optimizer=gradient_descent.GradientDescentOptimizer(0.01),
                    distribute=strategy)
      y = np.array([[[1], [1]], [[1], [1]]])
      dataset = dataset_ops.Dataset.from_tensor_slices((x, y))
      dataset = dataset.repeat(100)
      dataset = dataset.batch(10)
      hist = model.fit(x=dataset, epochs=1, steps_per_epoch=2)
      self.assertEqual(hist.history['loss'][0], 0)


class NormalizationLayerWithDistributionStrategyTest(test.TestCase):

  def test_batchnorm_correctness(self):
    with self.cached_session():
      model = keras.models.Sequential()
      norm = keras.layers.BatchNormalization(input_shape=(10,), momentum=0.8)
      model.add(norm)
      strategy = mirrored_strategy.MirroredStrategy(['/device:CPU:0',
                                                     '/device:GPU:0'])
      model.compile(loss='mse',
                    optimizer=gradient_descent.GradientDescentOptimizer(0.01),
                    distribute=strategy)

      # centered on 5.0, variance 10.0
      x = np.random.normal(loc=5.0, scale=10.0, size=(1000, 10))
      dataset = dataset_ops.Dataset.from_tensor_slices((x, x))
      dataset = dataset.repeat(100)
      dataset = dataset.batch(32)

      model.fit(dataset, epochs=4, verbose=0, steps_per_epoch=10)
      out = model.predict(dataset, steps=2)
      out -= keras.backend.eval(norm.beta)
      out /= keras.backend.eval(norm.gamma)
      np.testing.assert_allclose(out.mean(), 0.0, atol=1e-1)
      np.testing.assert_allclose(out.std(), 1.0, atol=1e-1)


class CorrectnessWithDistributionStrategyTest(test.TestCase):

  def test_correctness(self):
    with self.cached_session():
      keras.backend.set_image_data_format('channels_last')
      num_samples = 10000
      x_train = np.random.rand(num_samples, 1)
      y_train = 3 * x_train
      x_train = x_train.astype('float32')
      y_train = y_train.astype('float32')

      model = keras.Sequential()
      model.add(keras.layers.Dense(1, input_shape=(1,)))

      # With DistributionStrategy
      dataset_with = dataset_ops.Dataset.from_tensor_slices((x_train, y_train))
      dataset_with = dataset_with.batch(32)
      strategy = mirrored_strategy.MirroredStrategy(devices=['/device:CPU:0',
                                                             '/device:GPU:0'])

      model.compile(loss=keras.losses.mean_squared_error,
                    optimizer=gradient_descent.GradientDescentOptimizer(0.5),
                    distribute=strategy)
      model.fit(x=dataset_with, epochs=1, steps_per_epoch=310)
      wts_with_ds = model.get_weights()

      x_predict = [[1], [2], [3], [4]]
      predict_dataset_with = dataset_ops.Dataset.from_tensor_slices((x_predict,
                                                                     x_predict))
      predict_dataset_with = predict_dataset_with.batch(2)
      predict_with_ds = model.predict(predict_dataset_with, steps=1)
      predict_with_ds = np.reshape(predict_with_ds, (4, 1))

      # Without DistributionStrategy
      dataset_without = dataset_ops.Dataset.from_tensor_slices((x_train,
                                                                y_train))
      dataset_without = dataset_without.batch(64)

      model.compile(loss=keras.losses.mean_squared_error,
                    optimizer=gradient_descent.GradientDescentOptimizer(0.5))
      model.fit(x=dataset_without, epochs=1, steps_per_epoch=310)
      wts_without_ds = model.get_weights()

      x_predict = [[1], [2], [3], [4]]
      predict_dataset_without = dataset_ops.Dataset.from_tensor_slices((
          x_predict, x_predict))
      predict_dataset_without = predict_dataset_without.batch(4)
      predict_without_ds = model.predict(predict_dataset_without, steps=1)

      # Verify that the weights are the same within some limits of tolerance.
      np.testing.assert_allclose(wts_with_ds[0], wts_without_ds[0], rtol=1e-3)
      # Verify that the predicted outputs are the same within some limits of
      # tolerance.
      np.testing.assert_allclose(predict_with_ds, predict_without_ds, rtol=1e-3)


if __name__ == '__main__':
  test.main()
