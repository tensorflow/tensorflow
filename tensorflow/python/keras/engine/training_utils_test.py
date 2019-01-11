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
"""Tests for training utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np


from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import readers
from tensorflow.python.eager import context
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.platform import test


class ModelInputsTest(test.TestCase):

  def test_single_thing(self):
    a = np.ones(10)
    model_inputs = training_utils.ModelInputs(a)
    self.assertEqual(['input_1'], model_inputs.get_input_names())
    vals = model_inputs.get_symbolic_inputs()
    self.assertTrue(tensor_util.is_tensor(vals))
    vals = model_inputs.get_symbolic_inputs(return_single_as_list=True)
    self.assertEqual(1, len(vals))
    self.assertTrue(tensor_util.is_tensor(vals[0]))

  def test_single_thing_eager(self):
    with context.eager_mode():
      a = np.ones(10)
      model_inputs = training_utils.ModelInputs(a)
      self.assertEqual(['input_1'], model_inputs.get_input_names())
      val = model_inputs.get_symbolic_inputs()
      self.assertTrue(tf_utils.is_symbolic_tensor(val))
      vals = model_inputs.get_symbolic_inputs(return_single_as_list=True)
      self.assertEqual(1, len(vals))
      self.assertTrue(tf_utils.is_symbolic_tensor(vals[0]))

  def test_list(self):
    a = [np.ones(10), np.ones(20)]
    model_inputs = training_utils.ModelInputs(a)
    self.assertEqual(['input_1', 'input_2'], model_inputs.get_input_names())
    vals = model_inputs.get_symbolic_inputs()
    self.assertTrue(tensor_util.is_tensor(vals[0]))
    self.assertTrue(tensor_util.is_tensor(vals[1]))

  def test_list_eager(self):
    with context.eager_mode():
      a = [np.ones(10), np.ones(20)]
      model_inputs = training_utils.ModelInputs(a)
      self.assertEqual(['input_1', 'input_2'], model_inputs.get_input_names())
      vals = model_inputs.get_symbolic_inputs()
      self.assertTrue(tf_utils.is_symbolic_tensor(vals[0]))
      self.assertTrue(tf_utils.is_symbolic_tensor(vals[1]))

  def test_dict(self):
    a = {'b': np.ones(10), 'a': np.ones(20)}
    model_inputs = training_utils.ModelInputs(a)
    self.assertEqual(['a', 'b'], model_inputs.get_input_names())
    vals = model_inputs.get_symbolic_inputs()
    self.assertTrue(tensor_util.is_tensor(vals['a']))
    self.assertTrue(tensor_util.is_tensor(vals['b']))

  def test_dict_eager(self):
    with context.eager_mode():
      a = {'b': np.ones(10), 'a': np.ones(20)}
      model_inputs = training_utils.ModelInputs(a)
      self.assertEqual(['a', 'b'], model_inputs.get_input_names())
      vals = model_inputs.get_symbolic_inputs()
      self.assertTrue(tf_utils.is_symbolic_tensor(vals['a']))
      self.assertTrue(tf_utils.is_symbolic_tensor(vals['b']))


class DatasetUtilsTest(test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      # pylint: disable=g-long-lambda
      ('Batch', lambda: dataset_ops.Dataset.range(5).batch(2), ValueError),
      ('Cache', lambda: dataset_ops.Dataset.range(5).cache()),
      ('Concatenate', lambda: dataset_ops.Dataset.range(5).concatenate(
          dataset_ops.Dataset.range(5))),
      ('FlatMap', lambda: dataset_ops.Dataset.range(5).flat_map(
          lambda _: dataset_ops.Dataset.from_tensors(0)), ValueError),
      ('Filter', lambda: dataset_ops.Dataset.range(5).filter(lambda _: True)),
      ('FixedLengthRecordDatasetV2',
       lambda: readers.FixedLengthRecordDatasetV2([], 42)),
      ('FromTensors', lambda: dataset_ops.Dataset.from_tensors(0)),
      ('FromTensorSlices',
       lambda: dataset_ops.Dataset.from_tensor_slices([0, 0, 0])),
      ('Interleave', lambda: dataset_ops.Dataset.range(5).interleave(
          lambda _: dataset_ops.Dataset.from_tensors(0), cycle_length=1),
       ValueError),
      ('ParallelInterleave', lambda: dataset_ops.Dataset.range(5).interleave(
          lambda _: dataset_ops.Dataset.from_tensors(0),
          cycle_length=1,
          num_parallel_calls=1), ValueError),
      ('Map', lambda: dataset_ops.Dataset.range(5).map(lambda x: x)),
      ('Options',
       lambda: dataset_ops.Dataset.range(5).with_options(dataset_ops.Options())
      ),
      ('PaddedBatch', lambda: dataset_ops.Dataset.range(5).padded_batch(2, []),
       ValueError),
      ('ParallelMap', lambda: dataset_ops.Dataset.range(5).map(
          lambda x: x, num_parallel_calls=1)),
      ('Prefetch', lambda: dataset_ops.Dataset.range(5).prefetch(1)),
      ('Range', lambda: dataset_ops.Dataset.range(0)),
      ('Repeat', lambda: dataset_ops.Dataset.range(0).repeat(0)),
      ('Shuffle', lambda: dataset_ops.Dataset.range(5).shuffle(1)),
      ('Skip', lambda: dataset_ops.Dataset.range(5).skip(2)),
      ('Take', lambda: dataset_ops.Dataset.range(5).take(2)),
      ('TextLineDataset', lambda: readers.TextLineDatasetV2([])),
      ('TFRecordDataset', lambda: readers.TFRecordDatasetV2([])),
      ('Window', lambda: dataset_ops.Dataset.range(5).window(2), ValueError),
      ('Zip', lambda: dataset_ops.Dataset.zip(dataset_ops.Dataset.range(5))),
      # pylint: enable=g-long-lambda
  )
  def test_assert_not_batched(self, dataset_fn, expected_error=None):
    if expected_error is None:
      training_utils.assert_not_batched(dataset_fn())
    else:
      with self.assertRaises(expected_error):
        training_utils.assert_not_batched(dataset_fn())

  @parameterized.named_parameters(
      # pylint: disable=g-long-lambda
      ('Batch', lambda: dataset_ops.Dataset.range(5).batch(2)),
      ('Cache', lambda: dataset_ops.Dataset.range(5).cache()),
      ('Concatenate', lambda: dataset_ops.Dataset.range(5).concatenate(
          dataset_ops.Dataset.range(5))),
      ('FlatMap', lambda: dataset_ops.Dataset.range(5).flat_map(
          lambda _: dataset_ops.Dataset.from_tensors(0)), ValueError),
      ('Filter', lambda: dataset_ops.Dataset.range(5).filter(lambda _: True)),
      ('FixedLengthRecordDatasetV2',
       lambda: readers.FixedLengthRecordDatasetV2([], 42)),
      ('FromTensors', lambda: dataset_ops.Dataset.from_tensors(0)),
      ('FromTensorSlices',
       lambda: dataset_ops.Dataset.from_tensor_slices([0, 0, 0])),
      ('Interleave', lambda: dataset_ops.Dataset.range(5).interleave(
          lambda _: dataset_ops.Dataset.from_tensors(0), cycle_length=1),
       ValueError),
      ('Map', lambda: dataset_ops.Dataset.range(5).map(lambda x: x)),
      ('Options',
       lambda: dataset_ops.Dataset.range(5).with_options(dataset_ops.Options())
      ),
      ('PaddedBatch', lambda: dataset_ops.Dataset.range(5).padded_batch(2, [])),
      ('ParallelInterleave', lambda: dataset_ops.Dataset.range(5).interleave(
          lambda _: dataset_ops.Dataset.from_tensors(0),
          cycle_length=1,
          num_parallel_calls=1), ValueError),
      ('ParallelMap', lambda: dataset_ops.Dataset.range(5).map(
          lambda x: x, num_parallel_calls=1)),
      ('Prefetch', lambda: dataset_ops.Dataset.range(5).prefetch(1)),
      ('Range', lambda: dataset_ops.Dataset.range(0)),
      ('Repeat', lambda: dataset_ops.Dataset.range(0).repeat(0)),
      ('Shuffle', lambda: dataset_ops.Dataset.range(5).shuffle(1), ValueError),
      ('Skip', lambda: dataset_ops.Dataset.range(5).skip(2)),
      ('Take', lambda: dataset_ops.Dataset.range(5).take(2)),
      ('TextLineDataset', lambda: readers.TextLineDatasetV2([])),
      ('TFRecordDataset', lambda: readers.TFRecordDatasetV2([])),
      ('Window', lambda: dataset_ops.Dataset.range(5).window(2)),
      ('Zip', lambda: dataset_ops.Dataset.zip(dataset_ops.Dataset.range(5))),
      # pylint: enable=g-long-lambda
  )
  def test_assert_not_shuffled(self, dataset_fn, expected_error=None):
    if expected_error is None:
      training_utils.assert_not_shuffled(dataset_fn())
    else:
      with self.assertRaises(expected_error):
        training_utils.assert_not_shuffled(dataset_fn())


class StandardizeWeightsTest(keras_parameterized.TestCase):

  def test_sample_weights(self):
    y = np.array([0, 1, 0, 0, 2])
    sample_weights = np.array([0.5, 1., 1., 0., 2.])
    weights = training_utils.standardize_weights(y, sample_weights)
    self.assertAllClose(weights, sample_weights)

  def test_class_weights(self):
    y = np.array([0, 1, 0, 0, 2])
    class_weights = {0: 0.5, 1: 1., 2: 1.5}
    weights = training_utils.standardize_weights(y, class_weight=class_weights)
    self.assertAllClose(weights, np.array([0.5, 1., 0.5, 0.5, 1.5]))

  def test_sample_weights_and_class_weights(self):
    y = np.array([0, 1, 0, 0, 2])
    sample_weights = np.array([0.5, 1., 1., 0., 2.])
    class_weights = {0: 0.5, 1: 1., 2: 1.5}
    weights = training_utils.standardize_weights(y, sample_weights,
                                                 class_weights)
    expected = sample_weights * np.array([0.5, 1., 0.5, 0.5, 1.5])
    self.assertAllClose(weights, expected)


if __name__ == '__main__':
  test.main()
