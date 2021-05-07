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
"""Tests for training routines."""

import itertools

from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.eager import context
from tensorflow.python.keras import combinations
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import layers as layers_module
from tensorflow.python.keras import losses
from tensorflow.python.keras import metrics as metrics_module
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.engine import input_layer
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.engine import training_generator_v1
from tensorflow.python.keras.optimizer_v2 import rmsprop
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.platform import test
from tensorflow.python.util import nest


def custom_generator(mode=2):
  batch_size = 10
  num_samples = 50
  arr_data = np.random.random((num_samples, 2))
  arr_labels = np.random.random((num_samples, 4))
  arr_weights = np.random.random((num_samples,))
  i = 0
  while True:
    batch_index = i * batch_size % num_samples
    i += 1
    start = batch_index
    end = start + batch_size
    x = arr_data[start: end]
    y = arr_labels[start: end]
    w = arr_weights[start: end]
    if mode == 1:
      yield x
    elif mode == 2:
      yield x, y
    else:
      yield x, y, w


def custom_generator_changing_batch_size(mode=2):
  batch_size = 10
  cur_batch_size = 11
  num_samples = 50
  arr_data = np.random.random((num_samples, 2))
  arr_labels = np.random.random((num_samples, 4))
  arr_weights = np.random.random((num_samples,))
  i = 0
  while True:
    if cur_batch_size > 1:
      cur_batch_size -= 1
    batch_index = i * batch_size % num_samples
    i += 1
    start = batch_index
    end = start + cur_batch_size
    x = arr_data[start: end]
    y = arr_labels[start: end]
    w = arr_weights[start: end]
    if mode == 1:
      yield x
    elif mode == 2:
      yield x, y
    else:
      yield x, y, w

custom_generator_threads = data_utils.threadsafe_generator(custom_generator)


class TestGeneratorMethods(keras_parameterized.TestCase):

  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.run_all_keras_modes
  @data_utils.dont_use_multiprocessing_pool
  def test_fit_generator_method(self):
    model = testing_utils.get_small_mlp(
        num_hidden=3, num_classes=4, input_dim=2)
    model.compile(
        loss='mse',
        optimizer=rmsprop.RMSprop(1e-3),
        metrics=['mae', metrics_module.CategoricalAccuracy()])

    model.fit_generator(custom_generator_threads(),
                        steps_per_epoch=5,
                        epochs=1,
                        verbose=1,
                        max_queue_size=10,
                        workers=4,
                        use_multiprocessing=True)
    model.fit_generator(custom_generator(),
                        steps_per_epoch=5,
                        epochs=1,
                        verbose=1,
                        max_queue_size=10,
                        use_multiprocessing=False)
    model.fit_generator(custom_generator(),
                        steps_per_epoch=5,
                        epochs=1,
                        verbose=1,
                        max_queue_size=10,
                        use_multiprocessing=False,
                        validation_data=custom_generator(),
                        validation_steps=10)
    model.fit_generator(custom_generator(),
                        steps_per_epoch=5,
                        validation_data=custom_generator(),
                        validation_steps=1,
                        workers=0)

  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.run_all_keras_modes
  @data_utils.dont_use_multiprocessing_pool
  def test_evaluate_generator_method(self):
    model = testing_utils.get_small_mlp(
        num_hidden=3, num_classes=4, input_dim=2)
    model.compile(
        loss='mse',
        optimizer=rmsprop.RMSprop(1e-3),
        metrics=['mae', metrics_module.CategoricalAccuracy()],
        run_eagerly=testing_utils.should_run_eagerly())

    model.evaluate_generator(custom_generator_threads(),
                             steps=5,
                             max_queue_size=10,
                             workers=2,
                             verbose=1,
                             use_multiprocessing=True)
    model.evaluate_generator(custom_generator(),
                             steps=5,
                             max_queue_size=10,
                             use_multiprocessing=False)
    model.evaluate_generator(custom_generator(),
                             steps=5,
                             max_queue_size=10,
                             use_multiprocessing=False,
                             workers=0)

  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.run_all_keras_modes
  @data_utils.dont_use_multiprocessing_pool
  def test_predict_generator_method(self):
    model = testing_utils.get_small_mlp(
        num_hidden=3, num_classes=4, input_dim=2)
    model.run_eagerly = testing_utils.should_run_eagerly()

    model.predict_generator(custom_generator_threads(),
                            steps=5,
                            max_queue_size=10,
                            workers=2,
                            use_multiprocessing=True)
    model.predict_generator(custom_generator(),
                            steps=5,
                            max_queue_size=10,
                            use_multiprocessing=False)
    model.predict_generator(custom_generator(),
                            steps=5,
                            max_queue_size=10,
                            workers=0)
    # Test generator with just inputs (no targets)
    model.predict_generator(custom_generator_threads(mode=1),
                            steps=5,
                            max_queue_size=10,
                            workers=2,
                            use_multiprocessing=True)
    model.predict_generator(custom_generator(mode=1),
                            steps=5,
                            max_queue_size=10,
                            use_multiprocessing=False)
    model.predict_generator(custom_generator(mode=1),
                            steps=5,
                            max_queue_size=10,
                            workers=0)

  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.run_all_keras_modes
  def test_generator_methods_with_sample_weights(self):
    model = testing_utils.get_small_mlp(
        num_hidden=3, num_classes=4, input_dim=2)
    model.compile(
        loss='mse',
        optimizer=rmsprop.RMSprop(1e-3),
        metrics=['mae', metrics_module.CategoricalAccuracy()],
        run_eagerly=testing_utils.should_run_eagerly())

    model.fit_generator(custom_generator(mode=3),
                        steps_per_epoch=5,
                        epochs=1,
                        verbose=1,
                        max_queue_size=10,
                        use_multiprocessing=False)
    model.fit_generator(custom_generator(mode=3),
                        steps_per_epoch=5,
                        epochs=1,
                        verbose=1,
                        max_queue_size=10,
                        use_multiprocessing=False,
                        validation_data=custom_generator(mode=3),
                        validation_steps=10)
    model.predict_generator(custom_generator(mode=3),
                            steps=5,
                            max_queue_size=10,
                            use_multiprocessing=False)
    model.evaluate_generator(custom_generator(mode=3),
                             steps=5,
                             max_queue_size=10,
                             use_multiprocessing=False)

  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.run_all_keras_modes
  def test_generator_methods_invalid_use_case(self):
    def invalid_generator():
      while 1:
        yield (0, 0, 0, 0)

    model = testing_utils.get_small_mlp(
        num_hidden=3, num_classes=4, input_dim=2)
    model.compile(
        loss='mse',
        optimizer=rmsprop.RMSprop(1e-3),
        run_eagerly=testing_utils.should_run_eagerly())

    with self.assertRaises(ValueError):
      model.fit_generator(invalid_generator(),
                          steps_per_epoch=5,
                          epochs=1,
                          verbose=1,
                          max_queue_size=10,
                          use_multiprocessing=False)
    with self.assertRaises(ValueError):
      model.fit_generator(custom_generator(),
                          steps_per_epoch=5,
                          epochs=1,
                          verbose=1,
                          max_queue_size=10,
                          use_multiprocessing=False,
                          validation_data=invalid_generator(),
                          validation_steps=10)
    with self.assertRaises(ValueError):
      model.predict_generator(invalid_generator(),
                              steps=5,
                              max_queue_size=10,
                              use_multiprocessing=False)
    with self.assertRaises(ValueError):
      model.evaluate_generator(invalid_generator(),
                               steps=5,
                               max_queue_size=10,
                               use_multiprocessing=False)

  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.run_all_keras_modes
  def test_generator_input_to_fit_eval_predict(self):
    val_data = np.ones([10, 10], np.float32), np.ones([10, 1], np.float32)

    def ones_generator():
      while True:
        yield np.ones([10, 10], np.float32), np.ones([10, 1], np.float32)

    model = testing_utils.get_small_mlp(
        num_hidden=10, num_classes=1, input_dim=10)

    model.compile(
        rmsprop.RMSprop(0.001),
        'binary_crossentropy',
        run_eagerly=testing_utils.should_run_eagerly())
    model.fit(
        ones_generator(),
        steps_per_epoch=2,
        validation_data=val_data,
        epochs=2)
    model.evaluate(ones_generator(), steps=2)
    model.predict(ones_generator(), steps=2)

    # Test with a changing batch size
    model = testing_utils.get_small_mlp(
        num_hidden=3, num_classes=4, input_dim=2)
    model.compile(
        loss='mse',
        optimizer=rmsprop.RMSprop(1e-3),
        metrics=['mae', metrics_module.CategoricalAccuracy()])
    model.fit_generator(custom_generator_changing_batch_size(),
                        steps_per_epoch=5,
                        epochs=1,
                        verbose=1,
                        max_queue_size=10,
                        use_multiprocessing=False)
    model.fit_generator(custom_generator_changing_batch_size(),
                        steps_per_epoch=5,
                        epochs=1,
                        verbose=1,
                        max_queue_size=10,
                        use_multiprocessing=False,
                        validation_data=custom_generator_changing_batch_size(),
                        validation_steps=10)

    model.fit(
        custom_generator_changing_batch_size(),
        steps_per_epoch=5,
        validation_data=custom_generator_changing_batch_size(),
        validation_steps=10,
        epochs=2)
    model.evaluate(custom_generator_changing_batch_size(), steps=5)
    model.predict(custom_generator_changing_batch_size(), steps=5)

  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.run_all_keras_modes
  @data_utils.dont_use_multiprocessing_pool
  def test_generator_dynamic_shapes(self):

    x = [
        'I think juice is great',
        'unknown is the best language since slicedbread',
        'a a a a a a a',
        'matmul'
        'Yaks are also quite nice',
    ]
    y = [1, 0, 0, 1, 1]

    vocab = {
        word: i + 1 for i, word in
        enumerate(
            sorted(set(itertools.chain(*[i.split() for i in x]))))
    }

    def data_gen(batch_size=2):
      np.random.seed(0)
      data = list(zip(x, y)) * 10
      np.random.shuffle(data)

      def pack_and_pad(queue):
        x = [[vocab[j] for j in i[0].split()] for i in queue]
        pad_len = max(len(i) for i in x)
        x = np.array([i + [0] * (pad_len - len(i)) for i in x])
        y = np.array([i[1] for i in queue])
        del queue[:]
        return x, y[:, np.newaxis]

      queue = []
      for i, element in enumerate(data):
        queue.append(element)
        if not (i + 1) % batch_size:
          yield pack_and_pad(queue)

      if queue:
        # Last partial batch
        yield pack_and_pad(queue)

    model = testing_utils.get_model_from_layers([
        layers_module.Embedding(input_dim=len(vocab) + 1, output_dim=4),
        layers_module.SimpleRNN(units=1),
        layers_module.Activation('sigmoid')
    ],
                                                input_shape=(None,))

    model.compile(loss=losses.binary_crossentropy, optimizer='sgd')
    model.fit(data_gen(), epochs=1, steps_per_epoch=5)


class TestGeneratorMethodsWithSequences(keras_parameterized.TestCase):

  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.run_all_keras_modes
  @data_utils.dont_use_multiprocessing_pool
  def test_training_with_sequences(self):

    class DummySequence(data_utils.Sequence):

      def __getitem__(self, idx):
        return np.zeros([10, 2]), np.ones([10, 4])

      def __len__(self):
        return 10

    model = testing_utils.get_small_mlp(
        num_hidden=3, num_classes=4, input_dim=2)
    model.compile(loss='mse', optimizer=rmsprop.RMSprop(1e-3))

    model.fit_generator(DummySequence(),
                        steps_per_epoch=10,
                        validation_data=custom_generator(),
                        validation_steps=1,
                        max_queue_size=10,
                        workers=0,
                        use_multiprocessing=True)
    model.fit_generator(DummySequence(),
                        steps_per_epoch=10,
                        validation_data=custom_generator(),
                        validation_steps=1,
                        max_queue_size=10,
                        workers=0,
                        use_multiprocessing=False)

  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.run_all_keras_modes
  @data_utils.dont_use_multiprocessing_pool
  def test_sequence_input_to_fit_eval_predict(self):
    val_data = np.ones([10, 10], np.float32), np.ones([10, 1], np.float32)

    class CustomSequence(data_utils.Sequence):

      def __getitem__(self, idx):
        return np.ones([10, 10], np.float32), np.ones([10, 1], np.float32)

      def __len__(self):
        return 2

    class CustomSequenceChangingBatchSize(data_utils.Sequence):

      def __getitem__(self, idx):
        batch_size = 10 - idx
        return (np.ones([batch_size, 10], np.float32),
                np.ones([batch_size, 1], np.float32))

      def __len__(self):
        return 2

    model = testing_utils.get_small_mlp(
        num_hidden=10, num_classes=1, input_dim=10)

    model.compile(rmsprop.RMSprop(0.001), 'binary_crossentropy')
    model.fit(CustomSequence(), validation_data=val_data, epochs=2)
    model.evaluate(CustomSequence())
    model.predict(CustomSequence())

    with self.assertRaisesRegex(ValueError, '`y` argument is not supported'):
      model.fit(CustomSequence(), y=np.ones([10, 1]))

    with self.assertRaisesRegex(ValueError,
                                '`sample_weight` argument is not supported'):
      model.fit(CustomSequence(), sample_weight=np.ones([10, 1]))

    model.compile(rmsprop.RMSprop(0.001), 'binary_crossentropy')
    model.fit(CustomSequenceChangingBatchSize(),
              validation_data=val_data, epochs=2)
    model.evaluate(CustomSequenceChangingBatchSize())
    model.predict(CustomSequenceChangingBatchSize())

  @keras_parameterized.run_all_keras_modes(always_skip_v1=True)
  def test_sequence_on_epoch_end(self):

    class MySequence(data_utils.Sequence):

      def __init__(self):
        self.epochs = 0

      def __getitem__(self, idx):
        return np.ones([10, 10], np.float32), np.ones([10, 1], np.float32)

      def __len__(self):
        return 2

      def on_epoch_end(self):
        self.epochs += 1

    inputs = input_layer.Input(10)
    outputs = layers_module.Dense(1)(inputs)
    model = training.Model(inputs, outputs)
    model.compile('sgd', 'mse')
    my_seq = MySequence()
    model.fit(my_seq, epochs=2)
    self.assertEqual(my_seq.epochs, 2)


@combinations.generate(combinations.combine(mode=['graph', 'eager']))
class TestConvertToGeneratorLike(test.TestCase, parameterized.TestCase):
  simple_inputs = (np.ones((10, 10)), np.ones((10, 1)))
  nested_inputs = ((np.ones((10, 10)), np.ones((10, 20))), (np.ones((10, 1)),
                                                            np.ones((10, 3))))

  def _make_dataset(self, inputs, batches):
    return dataset_ops.DatasetV2.from_tensors(inputs).repeat(batches)

  def _make_iterator(self, inputs, batches):
    return dataset_ops.make_one_shot_iterator(
        self._make_dataset(inputs, batches))

  def _make_generator(self, inputs, batches):

    def _gen():
      for _ in range(batches):
        yield inputs

    return _gen()

  def _make_numpy(self, inputs, _):
    return inputs

  @parameterized.named_parameters(
      ('simple_dataset', _make_dataset, simple_inputs),
      ('simple_iterator', _make_iterator, simple_inputs),
      ('simple_generator', _make_generator, simple_inputs),
      ('simple_numpy', _make_numpy, simple_inputs),
      ('nested_dataset', _make_dataset, nested_inputs),
      ('nested_iterator', _make_iterator, nested_inputs),
      ('nested_generator', _make_generator, nested_inputs),
      ('nested_numpy', _make_numpy, nested_inputs))
  def test_convert_to_generator_like(self, input_fn, inputs):
    expected_batches = 5
    data = input_fn(self, inputs, expected_batches)

    # Dataset and Iterator not supported in Legacy Graph mode.
    if (not context.executing_eagerly() and
        isinstance(data, (dataset_ops.DatasetV2, iterator_ops.Iterator))):
      return

    generator, steps = training_generator_v1.convert_to_generator_like(
        data, batch_size=2, steps_per_epoch=expected_batches)
    self.assertEqual(steps, expected_batches)

    for _ in range(expected_batches):
      outputs = next(generator)
    nest.assert_same_structure(outputs, inputs)


if __name__ == '__main__':
  test.main()
