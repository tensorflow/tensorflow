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
"""Tests for tf.keras models with callbacks, checkpointing with dist strategy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import tempfile
from absl.testing import parameterized
import numpy as np
from tensorflow.python import keras
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.distribute import values
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.keras.distribute import distribute_strategy_test as keras_test_lib
from tensorflow.python.keras.distribute import distributed_training_utils
from tensorflow.python.keras.optimizer_v2 import rmsprop as rms_prop_keras
from tensorflow.python.training import gradient_descent


class Counter(keras.callbacks.Callback):
  """Counts the number of times each callback method was run.

  Attributes:
    method_counts: dict. Contains the counts of time  each callback method was
      run.
  """

  def __init__(self):
    self.method_counts = collections.defaultdict(int)
    methods_to_count = [
        'on_batch_begin', 'on_batch_end', 'on_epoch_begin', 'on_epoch_end',
        'on_predict_batch_begin', 'on_predict_batch_end', 'on_predict_begin',
        'on_predict_end', 'on_test_batch_begin', 'on_test_batch_end',
        'on_test_begin', 'on_test_end', 'on_train_batch_begin',
        'on_train_batch_end', 'on_train_begin', 'on_train_end'
    ]
    for method_name in methods_to_count:
      setattr(self, method_name,
              self.wrap_with_counts(method_name, getattr(self, method_name)))

  def wrap_with_counts(self, method_name, method):

    def _call_and_count(*args, **kwargs):
      self.method_counts[method_name] += 1
      return method(*args, **kwargs)

    return _call_and_count


class TestDistributionStrategyWithCallbacks(test.TestCase,
                                            parameterized.TestCase):

  @combinations.generate(keras_test_lib.all_strategy_combinations())
  def test_callbacks_in_fit(self, distribution):
    with distribution.scope():
      model = keras_test_lib.get_model()
      model.compile(optimizer='sgd', loss='mse', metrics=['mae'])

    dataset = keras_test_lib.get_dataset(distribution)
    counter = Counter()

    epochs = 2
    steps_per_epoch = 5
    validation_steps = 3

    model.fit(
        dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        verbose=0,
        validation_data=dataset,
        validation_steps=validation_steps,
        callbacks=[counter])

    if isinstance(distribution, tpu_strategy.TPUStrategy):
      # TPU Strategy can have multi step training, from extended.steps_per_run
      # if steps_per_run = 1, then num_batch_call_per_epoch = steps_per_epoch
      steps_per_run = distribution.extended.steps_per_run
      num_batch_call_per_epoch = steps_per_epoch // steps_per_run
      if steps_per_epoch % steps_per_run:
        num_batch_call_per_epoch += 1
    else:
      num_batch_call_per_epoch = steps_per_epoch

    self.assertDictEqual(
        counter.method_counts, {
            'on_batch_begin': epochs * num_batch_call_per_epoch,
            'on_batch_end': epochs * num_batch_call_per_epoch,
            'on_epoch_begin': epochs,
            'on_epoch_end': epochs,
            'on_test_batch_begin': epochs * validation_steps,
            'on_test_batch_end': epochs * validation_steps,
            'on_test_begin': epochs,
            'on_test_end': epochs,
            'on_train_batch_begin': epochs * num_batch_call_per_epoch,
            'on_train_batch_end': epochs * num_batch_call_per_epoch,
            'on_train_begin': 1,
            'on_train_end': 1
        })

  @combinations.generate(keras_test_lib.all_strategy_combinations())
  def test_callbacks_in_eval(self, distribution):
    with distribution.scope():
      model = keras_test_lib.get_model()
      model.compile(optimizer='sgd', loss='mse', metrics=['mae'])

    dataset = keras_test_lib.get_dataset(distribution)
    counter = Counter()

    model.evaluate(dataset, steps=5, callbacks=[counter])

    self.assertDictEqual(
        counter.method_counts, {
            'on_test_batch_begin': 5,
            'on_test_batch_end': 5,
            'on_test_begin': 1,
            'on_test_end': 1
        })

  @combinations.generate(keras_test_lib.all_strategy_combinations())
  def test_callbacks_in_predict(self, distribution):
    with distribution.scope():
      model = keras_test_lib.get_model()
      model.compile(optimizer='sgd', loss='mse', metrics=['mae'])

    dataset = keras_test_lib.get_dataset(distribution)
    counter = Counter()

    model.predict(
        keras_test_lib.get_predict_dataset(dataset),
        steps=5,
        callbacks=[counter])

    self.assertDictEqual(
        counter.method_counts, {
            'on_predict_batch_begin': 5,
            'on_predict_batch_end': 5,
            'on_predict_begin': 1,
            'on_predict_end': 1
        })


class TestDistributionStrategyErrorCases(test.TestCase, parameterized.TestCase):

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
          ],
          mode=['graph', 'eager']))
  def test_validating_dataset_input_tensors_with_shape_mismatch(
      self, distribution):
    with self.cached_session():
      a = constant_op.constant([1, 2], shape=(1, 2))
      b = constant_op.constant([[1, 2], [1, 2]], shape=(2, 2))
      device_map = values.ReplicaDeviceMap(('/device:CPU:0', '/device:GPU:0'))
      x = values.DistributedValues(device_map, (a, b))
      y = values.DistributedValues(device_map, (a, a))
      # Removed device and input tensor shape details from the error message
      # since the order of the device and the corresponding input tensor shape
      # is not deterministic over different runs.
      with self.assertRaisesRegexp(
          ValueError, 'Input tensor shapes do not match for '
          'distributed tensor inputs '
          'DistributedValues:.+'):
        with distribution.scope():
          distributed_training_utils.validate_distributed_dataset_inputs(
              distribution, x, y)

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
          ],
          mode=['graph', 'eager']))
  def test_validating_dataset_input_tensors_with_dtype_mismatch(
      self, distribution):
    with self.cached_session():
      a = constant_op.constant([1, 2], shape=(1, 2), dtype=dtypes.int32)
      b = constant_op.constant([1, 2], shape=(1, 2), dtype=dtypes.float64)
      device_map = values.ReplicaDeviceMap(('/device:CPU:0', '/device:GPU:0'))
      x = values.DistributedValues(device_map, (a, b))
      y = values.DistributedValues(device_map, (a, a))
      # Removed device and input tensor dtype details from the error message
      # since the order of the device and the corresponding input tensor dtype
      # is not deterministic over different runs.
      with self.assertRaisesRegexp(
          ValueError, 'Input tensor dtypes do not match for '
          'distributed tensor inputs '
          'DistributedValues:.+'):
        with distribution.scope():
          distributed_training_utils.validate_distributed_dataset_inputs(
              distribution, x, y)

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
          ],
          mode=['graph', 'eager']))
  def test_unsupported_features(self, distribution):
    with self.cached_session():
      with distribution.scope():
        model = keras_test_lib.get_model()
        optimizer = gradient_descent.GradientDescentOptimizer(0.001)
        loss = 'mse'
        metrics = ['mae']
        model.compile(optimizer, loss, metrics=metrics)

      dataset = keras_test_lib.get_dataset(distribution)

      # Test with validation split
      with self.assertRaisesRegexp(
          ValueError, '`validation_split` argument is not '
          'supported when input `x` is a dataset or a '
          'dataset iterator.+'):
        model.fit(
            dataset,
            epochs=1,
            steps_per_epoch=2,
            verbose=0,
            validation_split=0.5,
            validation_steps=2)

      # Test with sample weight.
      sample_weight = np.random.random((10,))
      with self.assertRaisesRegexp(
          ValueError, '`sample_weight` argument is not supported when input '
          '`x` is a dataset or a dataset iterator.'):
        model.fit(
            dataset,
            epochs=1,
            steps_per_epoch=2,
            verbose=0,
            sample_weight=sample_weight)

      # Test with not specifying the `steps` argument for dataset with infinite
      # cardinality.
      dataset = dataset.repeat()
      with self.assertRaisesRegexp(
          ValueError, 'When passing an infinitely '
          'repeating dataset, you must specify the '
          '`steps_per_epoch` argument'):
        model.fit(dataset, epochs=1, verbose=0)
      with self.assertRaisesRegexp(
          ValueError, 'When passing an infinitely '
          'repeating dataset, you must specify the '
          '`steps` argument'):
        model.evaluate(dataset, verbose=0)

      with self.assertRaisesRegexp(
          ValueError, 'When passing an infinitely '
          'repeating dataset, you must specify the '
          '`steps` argument'):
        model.predict(dataset, verbose=0)

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
          ],
          mode=['graph', 'eager']))
  def test_calling_with_unsupported_predefined_callbacks(self, distribution):
    with self.cached_session():
      with distribution.scope():
        model = keras_test_lib.get_model()
        optimizer = gradient_descent.GradientDescentOptimizer(0.001)
        loss = 'mse'
        metrics = ['mae']
        model.compile(optimizer, loss, metrics=metrics)

      dataset = keras_test_lib.get_dataset(distribution)

      def schedule(_):
        return 0.001

      with self.assertRaisesRegexp(
          ValueError, 'You must specify a Keras Optimizer V2 when '
          'using'):
        model.fit(
            dataset,
            epochs=1,
            steps_per_epoch=2,
            verbose=0,
            callbacks=[keras.callbacks.LearningRateScheduler(schedule)])

      with self.assertRaisesRegexp(
          ValueError, 'You must specify a Keras Optimizer V2 when '
          'using'):
        model.fit(
            dataset,
            epochs=1,
            steps_per_epoch=2,
            verbose=0,
            callbacks=[keras.callbacks.ReduceLROnPlateau()])

  @combinations.generate(
      combinations.combine(
          distribution=[strategy_combinations.one_device_strategy],
          mode=['eager']))
  def test_distribution_strategy_with_run_eagerly(self, distribution):
    with distribution.scope():
      x = keras.layers.Input(shape=(1,))
      y = keras.layers.Dense(1, kernel_initializer='ones')(x)
      model = keras.models.Model(x, y)

      err_msg = ('We currently do not support enabling `run_eagerly` with '
                 'distribution strategy.')
      with self.assertRaisesRegex(ValueError, err_msg):
        model.compile('sgd', run_eagerly=True)

  # TODO(b/124377929): Remove error assertions once subclassed models
  # are supported in DistributedStrategy.
  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
          ],
          mode=['graph', 'eager']))
  def test_distribution_strategy_on_subclassed_model(self, distribution):
    with distribution.scope():

      class _SimpleMLP(keras.Model):

        def __init__(self, num_labels):
          super(_SimpleMLP, self).__init__()
          self.dense = keras.layers.Dense(num_labels)

        def call(self, inputs):
          return self.dense(inputs)

      model = _SimpleMLP(3)

      with self.assertRaisesRegexp(
          ValueError,
          'We currently do not support distribution strategy with a '
          '`Sequential` model that is created without '
          '`input_shape`/`input_dim` set in its first layer or '
          'a subclassed model.'):
        model.compile('sgd')

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
          ],
          mode=['graph', 'eager']))
  def test_distribution_strategy_on_deferred_sequential_model(
      self, distribution):
    with distribution.scope():
      model = keras.models.Sequential()
      model.add(keras.layers.Dense(16, activation='relu'))
      model.add(keras.layers.Dense(3, activation='softmax'))

      with self.assertRaisesRegexp(
          ValueError,
          'We currently do not support distribution strategy with a '
          '`Sequential` model that is created without '
          '`input_shape`/`input_dim` set in its first layer or '
          'a subclassed model.'):
        model.compile('sgd')


class TestDistributionStrategyWithLossMasking(test.TestCase,
                                              parameterized.TestCase):

  # TODO(priyag): Enable all strategies for this test. Currently it does not
  # work for TPU due to some invalid datatype.
  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
          ],
          mode=['graph', 'eager']))
  def test_masking(self, distribution):
    with self.cached_session():
      np.random.seed(1337)
      x = np.array([[[1], [1]], [[0], [0]]])
      with distribution.scope():
        model = keras.models.Sequential()
        model.add(keras.layers.Masking(mask_value=0, input_shape=(2, 1)))
        model.add(
            keras.layers.TimeDistributed(
                keras.layers.Dense(1, kernel_initializer='one')))
        model.compile(
            loss='mse',
            optimizer=gradient_descent.GradientDescentOptimizer(0.01))
      y = np.array([[[1], [1]], [[1], [1]]])
      dataset = dataset_ops.Dataset.from_tensor_slices((x, y))
      dataset = dataset.repeat(100)
      dataset = dataset.batch(10)
      hist = model.fit(x=dataset, epochs=1, steps_per_epoch=2)
      self.assertEqual(hist.history['loss'][0], 0)


class TestDistributionStrategyWithNormalizationLayer(test.TestCase,
                                                     parameterized.TestCase):

  @combinations.generate(
      combinations.times(keras_test_lib.all_strategy_combinations(),
                         combinations.combine(fused=[True, False])))
  def test_batchnorm_correctness(self, distribution, fused):
    with self.cached_session():
      with distribution.scope():
        model = keras.models.Sequential()
        norm = keras.layers.BatchNormalization(
            input_shape=(
                10,
                20,
                30,
            ), momentum=0.8, fused=fused)
        model.add(norm)
        model.compile(
            loss='mse',
            optimizer=gradient_descent.GradientDescentOptimizer(0.01))

      # centered on 5.0, variance 10.0
      x = np.random.normal(loc=5.0, scale=10.0, size=(1000, 10, 20, 30))
      x = x.astype('float32')
      dataset = dataset_ops.Dataset.from_tensor_slices((x, x))
      dataset = dataset.repeat(100)
      dataset = keras_test_lib.batch_wrapper(dataset, 32, distribution)

      predict_dataset = dataset_ops.Dataset.from_tensor_slices(x)
      predict_dataset = predict_dataset.repeat(100)
      predict_dataset = keras_test_lib.batch_wrapper(predict_dataset, 32,
                                                     distribution)

      model.fit(dataset, epochs=4, verbose=0, steps_per_epoch=10)
      out = model.predict(predict_dataset, steps=2)
      out -= keras.backend.eval(norm.beta)
      out /= keras.backend.eval(norm.gamma)
      np.testing.assert_allclose(out.mean(), 0.0, atol=1e-1)
      np.testing.assert_allclose(out.std(), 1.0, atol=1e-1)


class TestDistributionStrategySaveLoadWeights(test.TestCase,
                                              parameterized.TestCase):

  @combinations.generate(
      keras_test_lib.all_strategy_combinations_minus_default())
  def test_save_load_h5(self, distribution):
    with self.cached_session():
      dataset = keras_test_lib.get_dataset(distribution)
      with distribution.scope():
        model = keras_test_lib.get_model()
        model.compile(rms_prop_keras.RMSprop(learning_rate=0.01), 'mse')
        model.fit(dataset, epochs=1, steps_per_epoch=1)

        weights_file = tempfile.mktemp('.h5')
        model.save_weights(weights_file)

        model_2 = keras_test_lib.get_model()
        model_2.compile(rms_prop_keras.RMSprop(learning_rate=0.01), 'mse')
        model_2.load_weights(weights_file)
        model_2.predict(
            keras_test_lib.get_predict_dataset(distribution), steps=2)
        model_2.fit(dataset, epochs=1, steps_per_epoch=1)

  @combinations.generate(
      keras_test_lib.all_strategy_combinations_minus_default())
  def test_save_load_trackable(self, distribution):
    # TODO(b/123533246): Enable the test for TPU once bug is fixed
    if (isinstance(distribution, tpu_strategy.TPUStrategy) and
        distribution.extended.steps_per_run > 1):
      self.skipTest('MultiStep TPU Strategy deadlocks with optimizer restore.')
    with self.cached_session():
      dataset = keras_test_lib.get_dataset(distribution)
      with distribution.scope():
        model = keras_test_lib.get_model()
        model.compile(rms_prop_keras.RMSprop(learning_rate=0.01), 'mse')
        model.fit(dataset, epochs=1, steps_per_epoch=1)

        weights_file = tempfile.mktemp()
        model.save_weights(weights_file)

        model_2 = keras_test_lib.get_model()
        model_2.compile(rms_prop_keras.RMSprop(learning_rate=0.01), 'mse')
        model_2.load_weights(weights_file)
        model_2.predict(
            keras_test_lib.get_predict_dataset(distribution), steps=2)
        model_2.fit(dataset, epochs=1, steps_per_epoch=1)


class TestDistributionStrategyValidation(test.TestCase, parameterized.TestCase):

  @combinations.generate(
      keras_test_lib.all_strategy_combinations_minus_default())
  def test_layer_outside_scope(self, distribution):
    with self.cached_session():
      with self.assertRaisesRegexp(
          ValueError, 'was not created in the distribution strategy'):
        x = keras.layers.Input(shape=(3,), name='input')
        y = keras.layers.Dense(4, name='dense')(x)
        with distribution.scope():
          model = keras.Model(x, y)
          optimizer = gradient_descent.GradientDescentOptimizer(0.001)
          loss = 'mse'
          metrics = ['mae', keras.metrics.CategoricalAccuracy()]
          model.compile(optimizer, loss, metrics=metrics)

  @combinations.generate(
      keras_test_lib.all_strategy_combinations_minus_default())
  def test_model_outside_scope(self, distribution):
    with self.cached_session():
      with self.assertRaisesRegexp(
          ValueError, 'was not created in the distribution strategy'):
        x = keras.layers.Input(shape=(3,), name='input')
        y = keras.layers.Dense(4, name='dense')(x)
        model = keras.Model(x, y)
        with distribution.scope():
          optimizer = gradient_descent.GradientDescentOptimizer(0.001)
          loss = 'mse'
          metrics = ['mae', keras.metrics.CategoricalAccuracy()]
          model.compile(optimizer, loss, metrics=metrics)


if __name__ == '__main__':
  test.main()
