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
"""TPU Estimator Signalling Tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.tpu.python.tpu import tpu_estimator
from tensorflow.python.client import session
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.platform import test


def make_input_fn(num_samples):
  a = np.linspace(0, 100.0, num=num_samples)
  b = np.reshape(np.array(a, dtype=np.float32), (len(a), 1))

  def input_fn(params):
    batch_size = params['batch_size']
    da1 = dataset_ops.Dataset.from_tensor_slices(a)
    da2 = dataset_ops.Dataset.from_tensor_slices(b)

    dataset = dataset_ops.Dataset.zip((da1, da2))
    dataset = dataset.map(lambda fa, fb: {'a': fa, 'b': fb})
    dataset = dataset.batch(batch_size)
    return dataset
  return input_fn, (a, b)


def make_input_fn_with_labels(num_samples):
  a = np.linspace(0, 100.0, num=num_samples)
  b = np.reshape(np.array(a, dtype=np.float32), (len(a), 1))

  def input_fn(params):
    batch_size = params['batch_size']
    da1 = dataset_ops.Dataset.from_tensor_slices(a)
    da2 = dataset_ops.Dataset.from_tensor_slices(b)

    dataset = dataset_ops.Dataset.zip((da1, da2))
    dataset = dataset.map(lambda fa, fb: ({'a': fa}, fb))
    dataset = dataset.batch(batch_size)
    return dataset
  return input_fn, (a, b)


class TPUEstimatorStoppingSignalsTest(test.TestCase):

  def test_normal_output_without_signals(self):
    num_samples = 4
    batch_size = 2

    params = {'batch_size': batch_size}
    input_fn, (a, b) = make_input_fn(num_samples=num_samples)

    with ops.Graph().as_default():
      dataset = input_fn(params)
      features = dataset_ops.make_one_shot_iterator(dataset).get_next()

      # With tf.data.Dataset.batch, the batch is None, i.e., dynamic shape.
      self.assertIsNone(features['a'].shape.as_list()[0])

      with session.Session() as sess:
        result = sess.run(features)
        self.assertAllEqual(a[:batch_size], result['a'])
        self.assertAllEqual(b[:batch_size], result['b'])

        # This run should work as num_samples / batch_size = 2.
        result = sess.run(features)
        self.assertAllEqual(a[batch_size:num_samples], result['a'])
        self.assertAllEqual(b[batch_size:num_samples], result['b'])

        with self.assertRaises(errors.OutOfRangeError):
          # Given num_samples and batch_size, this run should fail.
          sess.run(features)

  def test_output_with_stopping_signals(self):
    num_samples = 4
    batch_size = 2

    params = {'batch_size': batch_size}
    input_fn, (a, b) = make_input_fn(num_samples=num_samples)

    with ops.Graph().as_default():
      dataset = input_fn(params)
      inputs = tpu_estimator._InputsWithStoppingSignals(dataset, batch_size)
      dataset_initializer = inputs.dataset_initializer()
      features, _ = inputs.features_and_labels()
      signals = inputs.signals()

      # With tf.data.Dataset.batch, the batch is None, i.e., dynamic shape.
      self.assertIsNone(features['a'].shape.as_list()[0])

      with session.Session() as sess:
        sess.run(dataset_initializer)

        result, evaluated_signals = sess.run([features, signals])
        self.assertAllEqual(a[:batch_size], result['a'])
        self.assertAllEqual(b[:batch_size], result['b'])
        self.assertAllEqual([[0.]] * batch_size, evaluated_signals['stopping'])

        # This run should work as num_samples / batch_size = 2.
        result, evaluated_signals = sess.run([features, signals])
        self.assertAllEqual(a[batch_size:num_samples], result['a'])
        self.assertAllEqual(b[batch_size:num_samples], result['b'])
        self.assertAllEqual([[0.]] * batch_size, evaluated_signals['stopping'])

        # This run should work, *but* see STOP ('1') as signals
        _, evaluated_signals = sess.run([features, signals])
        self.assertAllEqual([[1.]] * batch_size, evaluated_signals['stopping'])

        with self.assertRaises(errors.OutOfRangeError):
          sess.run(features)


class TPUEstimatorStoppingSignalsWithPaddingTest(test.TestCase):

  def test_num_samples_divisible_by_batch_size(self):
    num_samples = 4
    batch_size = 2

    params = {'batch_size': batch_size}
    input_fn, (a, b) = make_input_fn(num_samples=num_samples)

    with ops.Graph().as_default():
      dataset = input_fn(params)
      inputs = tpu_estimator._InputsWithStoppingSignals(dataset, batch_size,
                                                        add_padding=True)
      dataset_initializer = inputs.dataset_initializer()
      features, _ = inputs.features_and_labels()
      signals = inputs.signals()

      # With padding, all shapes are static now.
      self.assertEqual(batch_size, features['a'].shape.as_list()[0])

      with session.Session() as sess:
        sess.run(dataset_initializer)

        result, evaluated_signals = sess.run([features, signals])
        self.assertAllEqual(a[:batch_size], result['a'])
        self.assertAllEqual(b[:batch_size], result['b'])
        self.assertAllEqual([[0.]] * batch_size, evaluated_signals['stopping'])
        self.assertAllEqual([0.] * batch_size,
                            evaluated_signals['padding_mask'])

        # This run should work as num_samples / batch_size = 2.
        result, evaluated_signals = sess.run([features, signals])
        self.assertAllEqual(a[batch_size:num_samples], result['a'])
        self.assertAllEqual(b[batch_size:num_samples], result['b'])
        self.assertAllEqual([[0.]] * batch_size, evaluated_signals['stopping'])
        self.assertAllEqual([0.] * batch_size,
                            evaluated_signals['padding_mask'])

        # This run should work, *but* see STOP ('1') as signals
        _, evaluated_signals = sess.run([features, signals])
        self.assertAllEqual([[1.]] * batch_size, evaluated_signals['stopping'])

        with self.assertRaises(errors.OutOfRangeError):
          sess.run(features)

  def test_num_samples_not_divisible_by_batch_size(self):
    num_samples = 5
    batch_size = 2

    params = {'batch_size': batch_size}
    input_fn, (a, b) = make_input_fn_with_labels(num_samples=num_samples)

    with ops.Graph().as_default():
      dataset = input_fn(params)
      inputs = tpu_estimator._InputsWithStoppingSignals(dataset, batch_size,
                                                        add_padding=True)
      dataset_initializer = inputs.dataset_initializer()
      features, labels = inputs.features_and_labels()
      signals = inputs.signals()

      # With padding, all shapes are static.
      self.assertEqual(batch_size, features['a'].shape.as_list()[0])

      with session.Session() as sess:
        sess.run(dataset_initializer)

        evaluated_features, evaluated_labels, evaluated_signals = (
            sess.run([features, labels, signals]))
        self.assertAllEqual(a[:batch_size], evaluated_features['a'])
        self.assertAllEqual(b[:batch_size], evaluated_labels)
        self.assertAllEqual([[0.]] * batch_size, evaluated_signals['stopping'])
        self.assertAllEqual([0.] * batch_size,
                            evaluated_signals['padding_mask'])

        # This run should work as num_samples / batch_size >= 2.
        evaluated_features, evaluated_labels, evaluated_signals = (
            sess.run([features, labels, signals]))
        self.assertAllEqual(a[batch_size:2*batch_size], evaluated_features['a'])
        self.assertAllEqual(b[batch_size:2*batch_size], evaluated_labels)
        self.assertAllEqual([[0.]] * batch_size, evaluated_signals['stopping'])
        self.assertAllEqual([0.] * batch_size,
                            evaluated_signals['padding_mask'])

        # This is the final partial batch.
        evaluated_features, evaluated_labels, evaluated_signals = (
            sess.run([features, labels, signals]))
        real_batch_size = num_samples % batch_size

        # Assert the real part.
        self.assertAllEqual(a[2*batch_size:num_samples],
                            evaluated_features['a'][:real_batch_size])
        self.assertAllEqual(b[2*batch_size:num_samples],
                            evaluated_labels[:real_batch_size])
        # Assert the padded part.
        self.assertAllEqual([0.0] * (batch_size - real_batch_size),
                            evaluated_features['a'][real_batch_size:])
        self.assertAllEqual([[0.0]] * (batch_size - real_batch_size),
                            evaluated_labels[real_batch_size:])

        self.assertAllEqual([[0.]] * batch_size, evaluated_signals['stopping'])

        padding = ([.0] * real_batch_size
                   + [1.] * (batch_size - real_batch_size))
        self.assertAllEqual(padding, evaluated_signals['padding_mask'])

        # This run should work, *but* see STOP ('1') as signals
        _, evaluated_signals = sess.run([features, signals])
        self.assertAllEqual([[1.]] * batch_size, evaluated_signals['stopping'])

        with self.assertRaises(errors.OutOfRangeError):
          sess.run(features)

  def test_slice(self):
    num_samples = 3
    batch_size = 2

    params = {'batch_size': batch_size}
    input_fn, (a, b) = make_input_fn(num_samples=num_samples)

    with ops.Graph().as_default():
      dataset = input_fn(params)
      inputs = tpu_estimator._InputsWithStoppingSignals(dataset, batch_size,
                                                        add_padding=True)
      dataset_initializer = inputs.dataset_initializer()
      features, _ = inputs.features_and_labels()
      signals = inputs.signals()

      sliced_features = (
          tpu_estimator._PaddingSignals.slice_tensor_or_dict(
              features, signals))

      with session.Session() as sess:
        sess.run(dataset_initializer)

        result, evaluated_signals = sess.run([sliced_features, signals])
        self.assertAllEqual(a[:batch_size], result['a'])
        self.assertAllEqual(b[:batch_size], result['b'])
        self.assertAllEqual([[0.]] * batch_size, evaluated_signals['stopping'])

        # This is the final partial batch.
        result, evaluated_signals = sess.run([sliced_features, signals])
        self.assertEqual(1, len(result['a']))
        self.assertAllEqual(a[batch_size:num_samples], result['a'])
        self.assertAllEqual(b[batch_size:num_samples], result['b'])
        self.assertAllEqual([[0.]] * batch_size, evaluated_signals['stopping'])

        # This run should work, *but* see STOP ('1') as signals
        _, evaluated_signals = sess.run([sliced_features, signals])
        self.assertAllEqual([[1.]] * batch_size, evaluated_signals['stopping'])

        with self.assertRaises(errors.OutOfRangeError):
          sess.run(sliced_features)

  def test_slice_with_multi_invocations_per_step(self):
    num_samples = 3
    batch_size = 2

    params = {'batch_size': batch_size}
    input_fn, (a, b) = make_input_fn(num_samples=num_samples)

    with ops.Graph().as_default():
      dataset = input_fn(params)
      inputs = tpu_estimator._InputsWithStoppingSignals(
          dataset, batch_size, add_padding=True, num_invocations_per_step=2)
      dataset_initializer = inputs.dataset_initializer()
      features, _ = inputs.features_and_labels()
      signals = inputs.signals()

      sliced_features = (
          tpu_estimator._PaddingSignals.slice_tensor_or_dict(features, signals))

      with session.Session() as sess:
        sess.run(dataset_initializer)

        result, evaluated_signals = sess.run([sliced_features, signals])
        self.assertAllEqual(a[:batch_size], result['a'])
        self.assertAllEqual(b[:batch_size], result['b'])
        self.assertAllEqual([[0.]] * batch_size, evaluated_signals['stopping'])

        # This is the final partial batch.
        result, evaluated_signals = sess.run([sliced_features, signals])
        self.assertEqual(1, len(result['a']))
        self.assertAllEqual(a[batch_size:num_samples], result['a'])
        self.assertAllEqual(b[batch_size:num_samples], result['b'])
        self.assertAllEqual([[0.]] * batch_size, evaluated_signals['stopping'])

        # We should see 3 continuous batches with STOP ('1') as signals and all
        # of them have mask 1.
        _, evaluated_signals = sess.run([sliced_features, signals])
        self.assertAllEqual([[1.]] * batch_size, evaluated_signals['stopping'])
        self.assertAllEqual([1.] * batch_size,
                            evaluated_signals['padding_mask'])

        _, evaluated_signals = sess.run([sliced_features, signals])
        self.assertAllEqual([[1.]] * batch_size, evaluated_signals['stopping'])
        self.assertAllEqual([1.] * batch_size,
                            evaluated_signals['padding_mask'])

        _, evaluated_signals = sess.run([sliced_features, signals])
        self.assertAllEqual([[1.]] * batch_size, evaluated_signals['stopping'])
        self.assertAllEqual([1.] * batch_size,
                            evaluated_signals['padding_mask'])
        with self.assertRaises(errors.OutOfRangeError):
          sess.run(sliced_features)


if __name__ == '__main__':
  test.main()
