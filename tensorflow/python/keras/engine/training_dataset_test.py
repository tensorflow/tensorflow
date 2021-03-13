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

import sys

import numpy as np
import six

from tensorflow.python import keras
from tensorflow.python.data.experimental.ops import cardinality
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import ops
from tensorflow.python.keras import callbacks
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import metrics as metrics_module
from tensorflow.python.keras import testing_utils
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging


class BatchCounterCallback(callbacks.Callback):

  def __init__(self):
    self.batch_begin_count = 0
    self.batch_end_count = 0

  def on_batch_begin(self, *args, **kwargs):
    self.batch_begin_count += 1

  def on_batch_end(self, *args, **kwargs):
    self.batch_end_count += 1


class TestTrainingWithDataset(keras_parameterized.TestCase):

  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.run_all_keras_modes
  def test_calling_model_on_same_dataset(self):
    model = testing_utils.get_small_mlp(1, 4, input_dim=3)
    optimizer = 'rmsprop'
    loss = 'mse'
    metrics = ['mae']
    model.compile(
        optimizer,
        loss,
        metrics=metrics,
        run_eagerly=testing_utils.should_run_eagerly())

    inputs = np.zeros((10, 3), np.float32)
    targets = np.zeros((10, 4), np.float32)
    dataset = dataset_ops.Dataset.from_tensor_slices((inputs, targets))
    dataset = dataset.repeat(100)
    dataset = dataset.batch(10)

    # Call fit with validation data
    model.fit(
        dataset,
        epochs=1,
        steps_per_epoch=2,
        verbose=0,
        validation_data=dataset,
        validation_steps=2)
    model.fit(
        dataset,
        epochs=1,
        steps_per_epoch=2,
        verbose=0,
        validation_data=dataset,
        validation_steps=2)

  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.run_all_keras_modes
  def test_training_and_eval_methods_on_dataset(self):
    model = testing_utils.get_small_mlp(1, 4, input_dim=3)
    optimizer = 'rmsprop'
    loss = 'mse'
    metrics = ['mae', metrics_module.CategoricalAccuracy()]
    model.compile(
        optimizer,
        loss,
        metrics=metrics,
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
    model.fit(
        dataset,
        epochs=1,
        steps_per_epoch=2,
        verbose=0,
        validation_data=dataset,
        validation_steps=2)

    # Test with validation split
    with self.assertRaises(ValueError):
      model.fit(
          dataset,
          epochs=1,
          steps_per_epoch=2,
          verbose=0,
          validation_split=0.5,
          validation_steps=2)

    # Test with sample weight.
    sample_weight = np.random.random((10,))
    with self.assertRaisesRegex(
        ValueError, r'`sample_weight` argument is not supported .+dataset'):
      model.fit(
          dataset,
          epochs=1,
          steps_per_epoch=2,
          verbose=0,
          sample_weight=sample_weight)

    with self.assertRaisesRegex(
        ValueError, '(you should not specify a target)|'
        '(`y` argument is not supported when using dataset as input.)'):
      model.fit(dataset, dataset, epochs=1, steps_per_epoch=2, verbose=0)

    # With an infinite dataset, `steps_per_epoch`/`steps` argument is required.
    with self.assertRaises(ValueError):
      model.fit(dataset, epochs=1, verbose=0)
    with self.assertRaises(ValueError):
      model.evaluate(dataset, verbose=0)
    with self.assertRaises(ValueError):
      model.predict(dataset, verbose=0)

  @keras_parameterized.run_with_all_model_types(exclude_models='sequential')
  @keras_parameterized.run_all_keras_modes
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
    dataset_tuple = dataset_ops.Dataset.from_tensor_slices(
        ((input_a_np, input_b_np), (output_d_np, output_e_np)))
    dataset_tuple = dataset_tuple.repeat(100)
    dataset_tuple = dataset_tuple.batch(10)

    model.fit(dataset_tuple, epochs=1, steps_per_epoch=2, verbose=1)
    model.evaluate(dataset_tuple, steps=2, verbose=1)

    # Test with dict
    input_dict = {'input_1': input_a_np, 'input_2': input_b_np}
    if testing_utils.get_model_type() == 'subclass':
      output_dict = {'output_1': output_d_np, 'output_2': output_e_np}
    else:
      output_dict = {'dense': output_d_np, 'dropout': output_e_np}

    dataset_dict = dataset_ops.Dataset.from_tensor_slices(
        (input_dict, output_dict))
    dataset_dict = dataset_dict.repeat(100)
    dataset_dict = dataset_dict.batch(10)

    model.fit(dataset_dict, epochs=1, steps_per_epoch=2, verbose=1)
    model.evaluate(dataset_dict, steps=2, verbose=1)

    predict_dataset_dict = dataset_ops.Dataset.from_tensor_slices(input_dict)
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
    model.compile(
        optimizer,
        loss,
        metrics=metrics,
        run_eagerly=testing_utils.should_run_eagerly())

    inputs = np.zeros((10, 3), np.float32)
    targets = np.zeros((10, 4), np.float32)
    sample_weights = np.ones((10), np.float32)
    dataset = dataset_ops.Dataset.from_tensor_slices(
        (inputs, targets, sample_weights))
    dataset = dataset.repeat(100)
    dataset = dataset.batch(10)

    model.fit(dataset, epochs=1, steps_per_epoch=2, verbose=1)
    model.evaluate(dataset, steps=2, verbose=1)
    model.predict(dataset, steps=2)

  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.run_all_keras_modes
  def test_dataset_with_sample_weights_correctness(self):
    x = keras.layers.Input(shape=(1,), name='input')
    y = keras.layers.Dense(
        1, kernel_initializer='ones', bias_initializer='zeros', name='dense')(
            x)
    model = keras.Model(x, y)
    optimizer = 'rmsprop'
    loss = 'mse'
    model.compile(optimizer, loss)
    inputs = np.array([[0], [1], [2], [3]], np.float32)
    targets = np.array([[2], [4], [6], [8]], np.float32)
    sample_weights = np.array([0.25, 0.5, 0.75, 1], np.float32)
    ds = dataset_ops.Dataset.from_tensor_slices(
        (inputs, targets, sample_weights)).batch(2)
    result = model.evaluate(ds, verbose=1)
    # The per sample loss is multipled by the corresponding sample weight. The
    # average of these weighted losses is the return value of the `evaluate`
    # call. For example, in the test above the average weighted loss is
    # calculated in the following manner:
    # ((2-0)^2) * 0.25 + ((4-1)^2) * 0.5 + ((6-2)^2 * 0.75) + ((8-3)^2 * 1)
    #  equals 42.5 / 4 = 10.625
    self.assertEqual(result, 10.625)

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
        return keras.backend.sum(inputs, axis=1, keepdims=True) + self.w * 0

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
    history = model.fit(
        train_dataset,
        epochs=2,
        steps_per_epoch=2,
        verbose=1,
        validation_data=val_dataset,
        validation_steps=2)
    self.assertAllClose(history.history['loss'],
                        [inputs[:20].sum() / 20, inputs[20:].sum() / 20])
    # The validation dataset will be reset at the end of each validation run.
    self.assertAllClose(history.history['val_loss'],
                        [inputs[:20].sum() / 20, inputs[:20].sum() / 20])

    # Test correctness with dataset reset.
    train_dataset = dataset_ops.Dataset.from_tensor_slices(
        (inputs, targets)).batch(10)
    val_dataset = dataset_ops.Dataset.from_tensor_slices(
        (inputs, targets)).batch(10)
    history = model.fit(
        train_dataset, epochs=2, verbose=1, validation_data=val_dataset)
    self.assertAllClose(
        history.history['loss'],
        [inputs.sum() / 40, inputs.sum() / 40])
    self.assertAllClose(
        history.history['val_loss'],
        [inputs.sum() / 40, inputs.sum() / 40])

  def test_dataset_input_shape_validation(self):
    with ops.get_default_graph().as_default(), self.cached_session():
      model = testing_utils.get_small_functional_mlp(1, 4, input_dim=3)
      model.compile(optimizer='rmsprop', loss='mse')

      # User forgets to batch the dataset
      inputs = np.zeros((10, 3))
      targets = np.zeros((10, 4))
      dataset = dataset_ops.Dataset.from_tensor_slices((inputs, targets))
      dataset = dataset.repeat(100)

      with self.assertRaisesRegex(
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

      with self.assertRaisesRegex(ValueError,
                                  r'expected (.*?) to have shape \(3,\)'):
        model.train_on_batch(dataset)

  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.run_all_keras_modes
  def test_finite_dataset_known_cardinality_no_steps_arg(self):
    model = testing_utils.get_small_mlp(1, 4, input_dim=3)
    model.compile(
        'rmsprop', 'mse', run_eagerly=testing_utils.should_run_eagerly())

    inputs = np.zeros((100, 3), dtype=np.float32)
    targets = np.random.randint(0, 4, size=100, dtype=np.int32)
    dataset = dataset_ops.Dataset.from_tensor_slices((inputs, targets))
    dataset = dataset.batch(10)

    batch_counter = BatchCounterCallback()
    history = model.fit(dataset, epochs=2, verbose=1, callbacks=[batch_counter])

    self.assertLen(history.history['loss'], 2)
    self.assertEqual(batch_counter.batch_end_count, 20)
    model.evaluate(dataset)
    out = model.predict(dataset)
    self.assertEqual(out.shape[0], 100)

  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.run_all_keras_modes
  def test_finite_dataset_unknown_cardinality_no_steps_arg(self):
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
    history = model.fit(dataset, epochs=2, verbose=1, callbacks=[batch_counter])

    self.assertLen(history.history['loss'], 2)
    self.assertEqual(batch_counter.batch_end_count, 20)
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

    self.assertIn('10/10', lines[-1])

    self.assertLen(history.history['loss'], 2)
    self.assertEqual(batch_counter.batch_begin_count, 21)
    self.assertEqual(batch_counter.batch_end_count, 20)
    model.evaluate(dataset)
    out = model.predict(dataset)
    self.assertEqual(out.shape[0], 100)

  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.run_all_keras_modes
  def test_finite_dataset_unknown_cardinality_out_of_data(self):
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
    with test.mock.patch.object(logging, 'warning') as mock_log:
      # steps_per_epoch (200) is greater than the dataset size (100). As this is
      # unexpected, training will stop and not make it to the second epoch.
      history = model.fit(
          dataset,
          epochs=2,
          verbose=1,
          callbacks=[batch_counter],
          steps_per_epoch=200)
      self.assertIn('ran out of data; interrupting training.',
                    str(mock_log.call_args))
      self.assertIn(
          'can generate at least '
          '`steps_per_epoch * epochs` batches (in this case, 400 batches). '
          'You may need to use the repeat() function when '
          'building your dataset.', str(mock_log.call_args))

    self.assertLen(history.history['loss'], 1)
    self.assertEqual(batch_counter.batch_end_count, 10)
    model.evaluate(dataset)
    out = model.predict(dataset)
    self.assertEqual(out.shape[0], 100)

  @keras_parameterized.run_all_keras_modes
  def test_with_external_loss(self):
    inp = keras.Input(shape=(4,), name='inp1')
    out = keras.layers.Dense(2)(inp)
    model = keras.Model(inp, out)
    model.add_loss(math_ops.reduce_mean(out))
    model.compile('rmsprop')
    x = np.ones((10, 4))

    # dataset contains only features, no labels.
    dataset = dataset_ops.Dataset.from_tensor_slices(x).repeat(10).batch(10)
    model.fit(dataset)

  @keras_parameterized.run_all_keras_modes(always_skip_v1=True)
  def test_train_eval_with_steps(self):
    # See b/142880049 for more details.
    inp = keras.Input(shape=(4,), name='inp1')
    out = keras.layers.Dense(2)(inp)
    model = keras.Model(inp, out)
    model.compile(
        'rmsprop', loss='mse', run_eagerly=testing_utils.should_run_eagerly())

    inputs = np.zeros((100, 4), dtype=np.float32)
    targets = np.random.randint(0, 2, size=100, dtype=np.int32)
    training_ds = dataset_ops.Dataset.from_tensor_slices(
        (inputs, targets)).repeat().batch(10)

    # Create eval dataset with generator, so that dataset won't contain the
    # overall size metadata. Without eval_steps, we expect to run through all
    # the data in this dataset every epoch.
    def gen():
      for _ in range(100):
        yield (np.zeros(4, dtype=np.float32),
               np.random.randint(0, 2, size=1, dtype=np.int32))

    eval_ds = dataset_ops.Dataset.from_generator(
        generator=gen,
        output_types=('float64', 'int32'),
        output_shapes=([4], [1])).batch(100)
    batch_counter = BatchCounterCallback()

    model.fit(
        training_ds,
        steps_per_epoch=10,
        epochs=10,
        validation_data=eval_ds,
        callbacks=[batch_counter])

    # Expect 10 batch from training per epoch.
    self.assertEqual(batch_counter.batch_end_count, 100)


class TestMetricsWithDatasets(keras_parameterized.TestCase):

  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.run_all_keras_modes
  def test_metrics_correctness_with_dataset(self):
    layers = [
        keras.layers.Dense(
            8, activation='relu', input_dim=4, kernel_initializer='ones'),
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
    outs = model.evaluate(dataset, steps=10)
    self.assertEqual(np.around(outs[1], decimals=1), 0.5)
    self.assertEqual(np.around(outs[2], decimals=1), 0.5)

    y = np.zeros((100, 1), dtype=np.float32)
    dataset = dataset_ops.Dataset.from_tensor_slices((x, y))
    dataset = dataset.repeat(100)
    dataset = dataset.batch(10)
    outs = model.evaluate(dataset, steps=10)
    self.assertEqual(outs[1], 0.)
    self.assertEqual(outs[2], 0.)


if __name__ == '__main__':
  test.main()
