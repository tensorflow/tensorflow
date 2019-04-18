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
"""Tests for training routines."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import sys

import numpy as np
import six

from tensorflow.python import keras
from tensorflow.python.data.experimental.ops import cardinality
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import test_util as tf_test_util
from tensorflow.python.keras import callbacks
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import metrics as metrics_module
from tensorflow.python.keras import testing_utils
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging


class BatchCounterCallback(callbacks.Callback):

  def __init__(self):
    self.batch_count = 0

  def on_batch_end(self, *args, **kwargs):
    self.batch_count += 1


class TestTrainingWithDatasetIterators(keras_parameterized.TestCase):

  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.run_all_keras_modes
  def test_training_and_eval_methods_on_iterators_single_io(self):
    model = testing_utils.get_small_mlp(1, 4, input_dim=3)
    optimizer = 'rmsprop'
    loss = 'mse'
    metrics = ['mae', metrics_module.CategoricalAccuracy()]
    model.compile(optimizer, loss, metrics=metrics,
                  run_eagerly=testing_utils.should_run_eagerly())

    inputs = np.zeros((10, 3), np.float32)
    targets = np.zeros((10, 4), np.float32)
    dataset = dataset_ops.Dataset.from_tensor_slices((inputs, targets))
    dataset = dataset.repeat(100)
    dataset = dataset.batch(10)
    iterator = dataset_ops.make_one_shot_iterator(dataset)

    model.fit(iterator, epochs=1, steps_per_epoch=2, verbose=1)
    model.evaluate(iterator, steps=2, verbose=1)
    model.predict(iterator, steps=2)

    # Test with validation data
    model.fit(iterator,
              epochs=1, steps_per_epoch=2, verbose=0,
              validation_data=iterator, validation_steps=2)
    # Test with validation split
    with self.assertRaisesRegexp(
        ValueError, '`validation_split` argument is not supported '
        'when input `x` is a dataset or a dataset iterator'):
      model.fit(iterator,
                epochs=1, steps_per_epoch=2, verbose=0,
                validation_split=0.5, validation_steps=2)

    # Test with sample weight.
    sample_weight = np.random.random((10,))
    with self.assertRaisesRegexp(
        ValueError, '`sample_weight` argument is not supported '
        'when input `x` is a dataset or a dataset iterator'):
      model.fit(
          iterator,
          epochs=1,
          steps_per_epoch=2,
          verbose=0,
          sample_weight=sample_weight)

    # Test invalid usage
    with self.assertRaisesRegexp(ValueError,
                                 'you should not specify a target'):
      model.fit(iterator, iterator,
                epochs=1, steps_per_epoch=2, verbose=0)

    with self.assertRaisesRegexp(
        ValueError, 'the `steps_per_epoch` argument'):
      model.fit(iterator, epochs=1, verbose=0)
    with self.assertRaisesRegexp(ValueError,
                                 'the `steps` argument'):
      model.evaluate(iterator, verbose=0)
    with self.assertRaisesRegexp(ValueError,
                                 'the `steps` argument'):
      model.predict(iterator, verbose=0)

  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.run_all_keras_modes
  def test_iterators_running_out_of_data(self):
    model = testing_utils.get_small_mlp(1, 4, input_dim=3)
    optimizer = 'rmsprop'
    loss = 'mse'
    metrics = ['mae']
    model.compile(optimizer, loss, metrics=metrics,
                  run_eagerly=testing_utils.should_run_eagerly())

    inputs = np.zeros((10, 3), np.float32)
    targets = np.zeros((10, 4), np.float32)
    dataset = dataset_ops.Dataset.from_tensor_slices((inputs, targets))
    dataset = dataset.repeat(2)
    dataset = dataset.batch(10)
    iterator = dataset_ops.make_one_shot_iterator(dataset)

    with test.mock.patch.object(logging, 'warning') as mock_log:
      model.fit(iterator, epochs=1, steps_per_epoch=3, verbose=0)
      self.assertRegexpMatches(
          str(mock_log.call_args),
          'dataset iterator ran out of data')


class TestTrainingWithDataset(keras_parameterized.TestCase):

  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.run_all_keras_modes
  def test_calling_model_on_same_dataset(self):
    if ((not testing_utils.should_run_eagerly())
        and testing_utils.get_model_type() == 'subclass'
        and context.executing_eagerly()):
      self.skipTest('b/120673224')

    model = testing_utils.get_small_mlp(1, 4, input_dim=3)
    optimizer = 'rmsprop'
    loss = 'mse'
    metrics = ['mae']
    model.compile(optimizer, loss, metrics=metrics,
                  run_eagerly=testing_utils.should_run_eagerly())

    inputs = np.zeros((10, 3), np.float32)
    targets = np.zeros((10, 4), np.float32)
    dataset = dataset_ops.Dataset.from_tensor_slices((inputs, targets))
    dataset = dataset.repeat(100)
    dataset = dataset.batch(10)

    # Call fit with validation data
    model.fit(dataset, epochs=1, steps_per_epoch=2, verbose=0,
              validation_data=dataset, validation_steps=2)
    model.fit(dataset, epochs=1, steps_per_epoch=2, verbose=0,
              validation_data=dataset, validation_steps=2)

  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.run_all_keras_modes
  def test_training_and_eval_methods_on_dataset(self):
    model = testing_utils.get_small_mlp(1, 4, input_dim=3)
    optimizer = 'rmsprop'
    loss = 'mse'
    metrics = ['mae', metrics_module.CategoricalAccuracy()]
    model.compile(optimizer, loss, metrics=metrics,
                  run_eagerly=testing_utils.should_run_eagerly())

    inputs = np.zeros((10, 3), np.float32)
    targets = np.zeros((10, 4), np.float32)
    dataset = dataset_ops.Dataset.from_tensor_slices((inputs, targets))
    dataset = dataset.repeat()  # Infinite dataset.
    dataset = dataset.batch(10)

    model.fit(dataset, epochs=1, steps_per_epoch=2, verbose=1)
    model.evaluate(dataset, steps=2, verbose=1)
    model.predict(dataset, steps=2)

    # Test with validation data
    model.fit(dataset, epochs=1, steps_per_epoch=2, verbose=0,
              validation_data=dataset, validation_steps=2)

    # Test with validation split
    with self.assertRaisesRegexp(
        ValueError, '`validation_split` argument is not supported '
        'when input `x` is a dataset or a dataset iterator'):
      model.fit(dataset,
                epochs=1, steps_per_epoch=2, verbose=0,
                validation_split=0.5, validation_steps=2)

    # Test with sample weight.
    sample_weight = np.random.random((10,))
    with self.assertRaisesRegexp(
        ValueError, '`sample_weight` argument is not supported '
        'when input `x` is a dataset or a dataset iterator'):
      model.fit(
          dataset,
          epochs=1,
          steps_per_epoch=2,
          verbose=0,
          sample_weight=sample_weight)

    # Test invalid usage
    with self.assertRaisesRegexp(ValueError, 'The `batch_size` argument'
                                 ' must not be specified when using dataset'
                                 ' as an input.'):
      model.fit(dataset, batch_size=10, epochs=1, steps_per_epoch=2,
                verbose=0)
    with self.assertRaisesRegexp(ValueError, 'The `batch_size` argument'
                                 ' must not be specified when using dataset'
                                 ' as an input.'):
      model.predict(dataset, batch_size=10, steps=2, verbose=0)
    with self.assertRaisesRegexp(ValueError, 'The `batch_size` argument'
                                 ' must not be specified when using dataset'
                                 ' as an input.'):
      model.evaluate(dataset, batch_size=10, steps=2, verbose=0)

    with self.assertRaisesRegexp(ValueError,
                                 'you should not specify a target'):
      model.fit(dataset, dataset,
                epochs=1, steps_per_epoch=2, verbose=0)

    # With an infinite dataset, `steps_per_epoch`/`steps` argument is required.
    with self.assertRaisesRegexp(
        ValueError, 'the `steps_per_epoch` argument'):
      model.fit(dataset, epochs=1, verbose=0)
    with self.assertRaisesRegexp(ValueError,
                                 'the `steps` argument'):
      model.evaluate(dataset, verbose=0)
    with self.assertRaisesRegexp(ValueError,
                                 'the `steps` argument'):
      model.predict(dataset, verbose=0)

  # TODO(b/123531973): Include tests using dataset_v1.
  @keras_parameterized.run_with_all_model_types(exclude_models='sequential')
  @keras_parameterized.run_all_keras_modes(always_skip_v1=True)
  def test_training_and_eval_methods_on_multi_input_output_dataset(self):
    input_a = keras.layers.Input(shape=(3,), name='input_1')
    input_b = keras.layers.Input(shape=(3,), name='input_2')
    dense = keras.layers.Dense(4, name='dense')
    dropout = keras.layers.Dropout(0.5, name='dropout')
    branch_a = [input_a, dense]
    branch_b = [input_b, dense, dropout]

    model = testing_utils.get_multi_io_model(branch_a, branch_b)
    model.compile(
        optimizer='rmsprop',
        loss='mse',
        run_eagerly=testing_utils.should_run_eagerly())

    input_a_np = np.random.random((10, 3)).astype(dtype=np.float32)
    input_b_np = np.random.random((10, 3)).astype(dtype=np.float32)
    output_d_np = np.random.random((10, 4)).astype(dtype=np.float32)
    output_e_np = np.random.random((10, 4)).astype(dtype=np.float32)

    # Test with tuples
    dataset_tuple = dataset_ops.Dataset.from_tensor_slices((
        (input_a_np, input_b_np), (output_d_np, output_e_np)))
    dataset_tuple = dataset_tuple.repeat(100)
    dataset_tuple = dataset_tuple.batch(10)

    model.fit(dataset_tuple, epochs=1, steps_per_epoch=2, verbose=1)
    model.evaluate(dataset_tuple, steps=2, verbose=1)

    predict_dataset_tuple = dataset_ops.Dataset.from_tensor_slices(
        (input_a_np, input_b_np))
    # TODO(b/123360757): Remove below assertion once predict() supports
    # muti-input datasets.
    with self.assertRaisesRegexp(ValueError,
                                 'Error when checking model input'):
      model.predict(predict_dataset_tuple, steps=1)

    # Test with dict
    input_dict = {'input_1': input_a_np, 'input_2': input_b_np}
    if testing_utils.get_model_type() == 'subclass':
      output_dict = {'output_1': output_d_np, 'output_2': output_e_np}
    else:
      output_dict = {'dense': output_d_np, 'dropout': output_e_np}

    dataset_dict = dataset_ops.Dataset.from_tensor_slices((
        input_dict, output_dict))
    dataset_dict = dataset_dict.repeat(100)
    dataset_dict = dataset_dict.batch(10)

    model.fit(dataset_dict, epochs=1, steps_per_epoch=2, verbose=1)
    model.evaluate(dataset_dict, steps=2, verbose=1)

    predict_dataset_dict = dataset_ops.Dataset.from_tensor_slices(
        input_dict)
    predict_dataset_dict = predict_dataset_dict.repeat(100)
    predict_dataset_dict = predict_dataset_dict.batch(10)
    model.predict(predict_dataset_dict, steps=1)

  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.run_all_keras_modes
  def test_dataset_with_sample_weights(self):
    model = testing_utils.get_small_mlp(1, 4, input_dim=3)
    optimizer = 'rmsprop'
    loss = 'mse'
    metrics = ['mae', metrics_module.CategoricalAccuracy()]
    model.compile(optimizer, loss, metrics=metrics,
                  run_eagerly=testing_utils.should_run_eagerly())

    inputs = np.zeros((10, 3), np.float32)
    targets = np.zeros((10, 4), np.float32)
    sample_weights = np.ones((10), np.float32)
    dataset = dataset_ops.Dataset.from_tensor_slices((inputs, targets,
                                                      sample_weights))
    dataset = dataset.repeat(100)
    dataset = dataset.batch(10)

    model.fit(dataset, epochs=1, steps_per_epoch=2, verbose=1)
    model.evaluate(dataset, steps=2, verbose=1)
    model.predict(dataset, steps=2)

  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.run_all_keras_modes
  def test_dataset_with_sparse_labels(self):
    model = testing_utils.get_small_mlp(1, 4, input_dim=3)
    optimizer = 'rmsprop'
    model.compile(
        optimizer,
        loss='sparse_categorical_crossentropy',
        run_eagerly=testing_utils.should_run_eagerly())

    inputs = np.zeros((10, 3), dtype=np.float32)
    targets = np.random.randint(0, 4, size=10, dtype=np.int32)
    dataset = dataset_ops.Dataset.from_tensor_slices((inputs, targets))
    dataset = dataset.repeat(100)
    dataset = dataset.batch(10)

    model.fit(dataset, epochs=1, steps_per_epoch=2, verbose=1)

  @keras_parameterized.run_all_keras_modes
  def test_dataset_fit_correctness(self):

    class SumLayer(keras.layers.Layer):

      def build(self, _):
        self.w = self.add_weight('w', ())

      def call(self, inputs):
        return keras.backend.sum(inputs) + self.w * 0

    model = keras.Sequential([SumLayer(input_shape=(2,))])
    model.compile(
        'rmsprop', loss='mae', run_eagerly=testing_utils.should_run_eagerly())

    inputs = np.zeros((40, 2), dtype=np.float32)
    inputs[10:20, :] = 2
    inputs[20:30, :] = 1
    inputs[30:, :] = 4
    targets = np.zeros((40, 1), dtype=np.float32)

    # Test correctness with `steps_per_epoch`.
    train_dataset = dataset_ops.Dataset.from_tensor_slices(
        (inputs, targets)).batch(10)
    val_dataset = dataset_ops.Dataset.from_tensor_slices(
        (inputs, targets)).batch(10)
    history = model.fit(train_dataset,
                        epochs=2, steps_per_epoch=2, verbose=1,
                        validation_data=val_dataset, validation_steps=2)
    self.assertListEqual(history.history['loss'],
                         [inputs[:20].sum() / 2, inputs[20:].sum() / 2])
    # The validation dataset will be reset at the end of each validation run.
    self.assertListEqual(history.history['val_loss'],
                         [inputs[:20].sum() / 2, inputs[:20].sum() / 2])

    # Test correctness with dataset reset.
    train_dataset = dataset_ops.Dataset.from_tensor_slices(
        (inputs, targets)).batch(10)
    val_dataset = dataset_ops.Dataset.from_tensor_slices(
        (inputs, targets)).batch(10)
    history = model.fit(train_dataset,
                        epochs=2, verbose=1, validation_data=val_dataset)
    self.assertListEqual(history.history['loss'],
                         [inputs.sum() / 4, inputs.sum() / 4])
    self.assertListEqual(history.history['val_loss'],
                         [inputs.sum() / 4, inputs.sum() / 4])

  @tf_test_util.run_deprecated_v1
  def test_dataset_input_shape_validation(self):
    with self.cached_session():
      model = testing_utils.get_small_functional_mlp(1, 4, input_dim=3)
      model.compile(optimizer='rmsprop', loss='mse')

      # User forgets to batch the dataset
      inputs = np.zeros((10, 3))
      targets = np.zeros((10, 4))
      dataset = dataset_ops.Dataset.from_tensor_slices((inputs, targets))
      dataset = dataset.repeat(100)

      with self.assertRaisesRegexp(
          ValueError,
          r'expected (.*?) to have shape \(3,\) but got array with shape \(1,\)'
      ):
        model.train_on_batch(dataset)

      # Wrong input shape
      inputs = np.zeros((10, 5))
      targets = np.zeros((10, 4))
      dataset = dataset_ops.Dataset.from_tensor_slices((inputs, targets))
      dataset = dataset.repeat(100)
      dataset = dataset.batch(10)

      with self.assertRaisesRegexp(ValueError,
                                   r'expected (.*?) to have shape \(3,\)'):
        model.train_on_batch(dataset)

  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.run_all_keras_modes
  def test_finite_dataset_known_cardinality_no_steps_arg(self):
    model = testing_utils.get_small_mlp(1, 4, input_dim=3)
    model.compile('rmsprop', 'mse',
                  run_eagerly=testing_utils.should_run_eagerly())

    inputs = np.zeros((100, 3), dtype=np.float32)
    targets = np.random.randint(0, 4, size=100, dtype=np.int32)
    dataset = dataset_ops.Dataset.from_tensor_slices((inputs, targets))
    dataset = dataset.batch(10)

    batch_counter = BatchCounterCallback()
    history = model.fit(dataset, epochs=2, verbose=1, callbacks=[batch_counter])

    self.assertLen(history.history['loss'], 2)
    self.assertEqual(batch_counter.batch_count, 20)
    model.evaluate(dataset)
    out = model.predict(dataset)
    self.assertEqual(out.shape[0], 100)

  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.run_all_keras_modes
  def test_finite_dataset_unknown_cardinality_no_steps_arg(self):
    model = testing_utils.get_small_mlp(1, 4, input_dim=3)
    model.compile('rmsprop', 'mse',
                  run_eagerly=testing_utils.should_run_eagerly())

    inputs = np.zeros((100, 3), dtype=np.float32)
    targets = np.random.randint(0, 4, size=100, dtype=np.int32)
    dataset = dataset_ops.Dataset.from_tensor_slices((inputs, targets))
    dataset = dataset.filter(lambda x, y: True).batch(10)
    self.assertEqual(keras.backend.get_value(cardinality.cardinality(dataset)),
                     cardinality.UNKNOWN)

    batch_counter = BatchCounterCallback()
    history = model.fit(dataset, epochs=2, verbose=1, callbacks=[batch_counter])

    self.assertLen(history.history['loss'], 2)
    self.assertEqual(batch_counter.batch_count, 20)
    model.evaluate(dataset)
    out = model.predict(dataset)
    self.assertEqual(out.shape[0], 100)

  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.run_all_keras_modes(always_skip_v1=True)
  def test_finite_dataset_unknown_cardinality_no_step_with_train_and_val(self):

    class CaptureStdout(object):

      def __enter__(self):
        self._stdout = sys.stdout
        string_io = six.StringIO()
        sys.stdout = string_io
        self._stringio = string_io
        return self

      def __exit__(self, *args):
        self.output = self._stringio.getvalue()
        sys.stdout = self._stdout

    model = testing_utils.get_small_mlp(1, 4, input_dim=3)
    model.compile(
        'rmsprop', 'mse', run_eagerly=testing_utils.should_run_eagerly())

    inputs = np.zeros((100, 3), dtype=np.float32)
    targets = np.random.randint(0, 4, size=100, dtype=np.int32)
    dataset = dataset_ops.Dataset.from_tensor_slices((inputs, targets))
    dataset = dataset.filter(lambda x, y: True).batch(10)
    self.assertEqual(
        keras.backend.get_value(cardinality.cardinality(dataset)),
        cardinality.UNKNOWN)

    batch_counter = BatchCounterCallback()
    with CaptureStdout() as capture:
      history = model.fit(
          dataset,
          epochs=2,
          callbacks=[batch_counter],
          validation_data=dataset.take(3))

    lines = capture.output.splitlines()

    self.assertIn('1/Unknown', lines[2])
    self.assertIn('10/10', lines[-1])

    self.assertLen(history.history['loss'], 2)
    self.assertEqual(batch_counter.batch_count, 20)
    model.evaluate(dataset)
    out = model.predict(dataset)
    self.assertEqual(out.shape[0], 100)

  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.run_all_keras_modes
  def test_finite_dataset_unknown_cardinality_out_of_data(self):
    model = testing_utils.get_small_mlp(1, 4, input_dim=3)
    model.compile('rmsprop', 'mse',
                  run_eagerly=testing_utils.should_run_eagerly())

    inputs = np.zeros((100, 3), dtype=np.float32)
    targets = np.random.randint(0, 4, size=100, dtype=np.int32)
    dataset = dataset_ops.Dataset.from_tensor_slices((inputs, targets))
    dataset = dataset.filter(lambda x, y: True).batch(10)
    self.assertEqual(
        keras.backend.get_value(cardinality.cardinality(dataset)),
        cardinality.UNKNOWN)

    batch_counter = BatchCounterCallback()
    with test.mock.patch.object(logging, 'warning') as mock_log:
      # steps_per_epoch (200) is greater than the dataset size (100). As this is
      # unexpected, training will stop and not make it to the second epoch.
      history = model.fit(
          dataset,
          epochs=2,
          verbose=1,
          callbacks=[batch_counter],
          steps_per_epoch=200)
      self.assertIn(
          'Your dataset ran out of data; interrupting training. '
          'Make sure that your dataset can generate at least '
          '`steps_per_epoch * epochs` batches (in this case, 400 batches). '
          'You may need to use the repeat() function when '
          'building your dataset.', str(mock_log.call_args))

    self.assertLen(history.history['loss'], 1)
    self.assertEqual(batch_counter.batch_count, 10)
    model.evaluate(dataset)
    out = model.predict(dataset)
    self.assertEqual(out.shape[0], 100)


class TestMetricsWithDatasetIterators(keras_parameterized.TestCase):

  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.run_all_keras_modes
  def test_metrics_correctness_with_iterator(self):
    layers = [
        keras.layers.Dense(8, activation='relu', input_dim=4,
                           kernel_initializer='ones'),
        keras.layers.Dense(1, activation='sigmoid', kernel_initializer='ones')
    ]

    model = testing_utils.get_model_from_layers(layers, (4,))

    model.compile(
        loss='binary_crossentropy',
        metrics=['accuracy', metrics_module.BinaryAccuracy()],
        optimizer='rmsprop',
        run_eagerly=testing_utils.should_run_eagerly())

    np.random.seed(123)
    x = np.random.randint(10, size=(100, 4)).astype(np.float32)
    y = np.random.randint(2, size=(100, 1)).astype(np.float32)
    dataset = dataset_ops.Dataset.from_tensor_slices((x, y))
    dataset = dataset.batch(10)
    iterator = dataset_ops.make_one_shot_iterator(dataset)
    outs = model.evaluate(iterator, steps=10)
    self.assertEqual(np.around(outs[1], decimals=1), 0.5)
    self.assertEqual(np.around(outs[2], decimals=1), 0.5)

    y = np.zeros((100, 1), dtype=np.float32)
    dataset = dataset_ops.Dataset.from_tensor_slices((x, y))
    dataset = dataset.repeat(100)
    dataset = dataset.batch(10)
    iterator = dataset_ops.make_one_shot_iterator(dataset)
    outs = model.evaluate(iterator, steps=10)
    self.assertEqual(outs[1], 0.)
    self.assertEqual(outs[2], 0.)


if __name__ == '__main__':
  test.main()
