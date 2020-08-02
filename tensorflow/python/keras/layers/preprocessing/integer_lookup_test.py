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
"""Tests for Keras text vectorization preprocessing layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import os
import random

from absl.testing import parameterized
import numpy as np

from tensorflow.python import keras
from tensorflow.python import tf2

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.layers.preprocessing import integer_lookup
from tensorflow.python.keras.layers.preprocessing import integer_lookup_v1
from tensorflow.python.keras.layers.preprocessing import preprocessing_test_utils
from tensorflow.python.keras.saving import save
from tensorflow.python.keras.utils.generic_utils import CustomObjectScope
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test


def get_layer_class():
  if context.executing_eagerly():
    return integer_lookup.IntegerLookup
  else:
    return integer_lookup_v1.IntegerLookup


def _get_end_to_end_test_cases():
  test_cases = (
      {
          "testcase_name":
              "test_ints_soft_vocab_cap",
          # Create an array where 1138 is the most frequent term, followed by
          # 1729, then 725, then 42. This ensures that the vocab accumulator
          # is sorting by frequency.
          "vocab_data":
              np.array([[42], [1138], [1138], [1138], [1138], [1729], [1729],
                        [1729], [725], [725]],
                       dtype=np.int64),
          "input_data":
              np.array([[1138], [1729], [725], [42], [42], [725], [1138], [4]],
                       dtype=np.int64),
          "kwargs": {
              "max_values": None,
              "dtype": dtypes.int64,
          },
          "expected_output": [[2], [3], [4], [5], [5], [4], [2], [1]],
          "input_dtype":
              dtypes.int64
      },)

  crossed_test_cases = []
  # Cross above test cases with use_dataset in (True, False)
  for use_dataset in (True, False):
    for case in test_cases:
      case = case.copy()
      if use_dataset:
        case["testcase_name"] = case["testcase_name"] + "_with_dataset"
      case["use_dataset"] = use_dataset
      crossed_test_cases.append(case)

  return crossed_test_cases


@keras_parameterized.run_all_keras_modes
class IntegerLookupLayerTest(keras_parameterized.TestCase,
                             preprocessing_test_utils.PreprocessingLayerTest):

  @parameterized.named_parameters(*_get_end_to_end_test_cases())
  def test_layer_end_to_end_with_adapt(self, vocab_data, input_data, kwargs,
                                       use_dataset, expected_output,
                                       input_dtype):
    cls = get_layer_class()
    expected_output_dtype = dtypes.int64
    input_shape = input_data.shape

    if use_dataset:
      # Keras APIs expect batched datasets.
      # TODO(rachelim): `model.predict` predicts the result on each
      # dataset batch separately, then tries to concatenate the results
      # together. When the results have different shapes on the non-concat
      # axis (which can happen in the output_mode = INT case for
      # IntegerLookup), the concatenation fails. In real use cases, this may
      # not be an issue because users are likely to pipe the preprocessing layer
      # into other keras layers instead of predicting it directly. A workaround
      # for these unit tests is to have the dataset only contain one batch, so
      # no concatenation needs to happen with the result. For consistency with
      # numpy input, we should make `predict` join differently shaped results
      # together sensibly, with 0 padding.
      input_data = dataset_ops.Dataset.from_tensor_slices(input_data).batch(
          input_shape[0])
      vocab_data = dataset_ops.Dataset.from_tensor_slices(vocab_data).batch(
          input_shape[0])

    with CustomObjectScope({"IntegerLookup": cls}):
      output_data = testing_utils.layer_test(
          cls,
          kwargs=kwargs,
          input_shape=input_shape,
          input_data=input_data,
          input_dtype=input_dtype,
          expected_output_dtype=expected_output_dtype,
          validate_training=False,
          adapt_data=vocab_data)
    self.assertAllClose(expected_output, output_data)


@keras_parameterized.run_all_keras_modes
class CategoricalEncodingInputTest(
    keras_parameterized.TestCase,
    preprocessing_test_utils.PreprocessingLayerTest):

  def test_sparse_int_input(self):
    vocab_data = np.array([10, 11, 12, 13], dtype=np.int64)
    input_array = sparse_tensor.SparseTensor(
        indices=[[0, 0], [1, 2]],
        values=np.array([13, 32], dtype=np.int64),
        dense_shape=[3, 4])

    expected_indices = [[0, 0], [1, 2]]
    expected_values = [5, 1]
    expected_dense_shape = [3, 4]

    input_data = keras.Input(shape=(None,), dtype=dtypes.int64, sparse=True)
    layer = get_layer_class()(max_values=None)
    layer.set_vocabulary(vocab_data)
    int_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=int_data)
    output_data = model.predict(input_array, steps=1)
    self.assertAllEqual(expected_indices, output_data.indices)
    self.assertAllEqual(expected_values, output_data.values)
    self.assertAllEqual(expected_dense_shape, output_data.dense_shape)

  def test_ragged_int_input(self):
    vocab_data = np.array([10, 11, 12, 13], dtype=np.int64)
    input_array = ragged_factory_ops.constant([[10, 11, 13], [13, 12, 10, 42]],
                                              dtype=np.int64)
    expected_output = [[2, 3, 5], [5, 4, 2, 1]]

    input_data = keras.Input(shape=(None,), dtype=dtypes.int64, ragged=True)
    layer = get_layer_class()(max_values=None)
    layer.set_vocabulary(vocab_data)
    int_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_array)
    self.assertAllEqual(expected_output, output_dataset)


@keras_parameterized.run_all_keras_modes
class CategoricalEncodingMultiOOVTest(
    keras_parameterized.TestCase,
    preprocessing_test_utils.PreprocessingLayerTest):

  def test_sparse_int_input_multi_bucket(self):
    vocab_data = np.array([10, 11, 12, 13], dtype=np.int64)
    input_array = sparse_tensor.SparseTensor(
        indices=[[0, 0], [1, 2]],
        values=np.array([13, 133], dtype=np.int64),
        dense_shape=[3, 4])

    expected_indices = [[0, 0], [1, 2]]
    expected_values = [6, 2]
    expected_dense_shape = [3, 4]

    input_data = keras.Input(shape=(None,), dtype=dtypes.int64, sparse=True)
    layer = get_layer_class()(
        max_values=None,
        dtype=dtypes.int64,
        num_oov_indices=2,
        mask_value=0,
        oov_value=-1)
    layer.set_vocabulary(vocab_data)
    int_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=int_data)
    output_data = model.predict(input_array, steps=1)
    self.assertAllEqual(expected_indices, output_data.indices)
    self.assertAllEqual(expected_values, output_data.values)
    self.assertAllEqual(expected_dense_shape, output_data.dense_shape)

  def test_ragged_int_input_multi_bucket(self):
    vocab_data = np.array([10, 11, 12, 13], dtype=np.int64)
    input_array = ragged_factory_ops.constant([[10, 11, 13], [13, 12, 10, 133]],
                                              dtype=np.int64)
    expected_output = [[3, 4, 6], [6, 5, 3, 2]]

    input_data = keras.Input(shape=(None,), dtype=dtypes.int64, ragged=True)
    layer = get_layer_class()(max_values=None, num_oov_indices=2)
    layer.set_vocabulary(vocab_data)
    int_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_array)
    self.assertAllEqual(expected_output, output_dataset)


@keras_parameterized.run_all_keras_modes
class CategoricalEncodingAdaptTest(
    keras_parameterized.TestCase,
    preprocessing_test_utils.PreprocessingLayerTest):

  def test_sparse_adapt(self):
    vocab_data = sparse_tensor.SparseTensor(
        indices=[[0, 0], [0, 1], [1, 2]],
        values=[203, 1729, 203],
        dense_shape=[3, 4])
    vocab_dataset = dataset_ops.Dataset.from_tensors(vocab_data)

    layer = get_layer_class()()
    layer.adapt(vocab_dataset)
    expected_vocabulary = [0, -1, 203, 1729]
    self.assertAllEqual(expected_vocabulary, layer.get_vocabulary())

  def test_ragged_adapt(self):
    vocab_data = ragged_factory_ops.constant([[203], [1729, 203]])
    vocab_dataset = dataset_ops.Dataset.from_tensors(vocab_data)

    layer = get_layer_class()()
    layer.adapt(vocab_dataset)
    expected_vocabulary = [0, -1, 203, 1729]
    self.assertAllEqual(expected_vocabulary, layer.get_vocabulary())

  def test_sparse_int_input(self):
    vocab_data = np.array([10, 11, 12, 13], dtype=np.int64)
    input_array = sparse_tensor.SparseTensor(
        indices=[[0, 0], [1, 2]],
        values=np.array([13, 32], dtype=np.int64),
        dense_shape=[3, 4])

    expected_indices = [[0, 0], [1, 2]]
    expected_values = [5, 1]
    expected_dense_shape = [3, 4]

    input_data = keras.Input(shape=(None,), dtype=dtypes.int64, sparse=True)
    layer = get_layer_class()(max_values=None)
    layer.set_vocabulary(vocab_data)
    int_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=int_data)
    output_data = model.predict(input_array, steps=1)
    self.assertAllEqual(expected_indices, output_data.indices)
    self.assertAllEqual(expected_values, output_data.values)
    self.assertAllEqual(expected_dense_shape, output_data.dense_shape)

  def test_ragged_int_input(self):
    vocab_data = np.array([10, 11, 12, 13], dtype=np.int64)
    input_array = ragged_factory_ops.constant([[10, 11, 13], [13, 12, 10, 42]],
                                              dtype=np.int64)
    expected_output = [[2, 3, 5], [5, 4, 2, 1]]

    input_data = keras.Input(shape=(None,), dtype=dtypes.int64, ragged=True)
    layer = get_layer_class()(max_values=None)
    layer.set_vocabulary(vocab_data)
    int_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_array)
    self.assertAllEqual(expected_output, output_dataset)

  def test_single_int_generator_dataset(self):

    def word_gen():
      for _ in itertools.count(1):
        yield random.randint(0, 100)

    ds = dataset_ops.Dataset.from_generator(word_gen, dtypes.int64,
                                            tensor_shape.TensorShape([]))
    batched_ds = ds.take(2)
    input_t = keras.Input(shape=(), dtype=dtypes.int64)
    layer = get_layer_class()(
        max_values=10, num_oov_indices=0, mask_value=None, oov_value=None)
    _ = layer(input_t)
    layer.adapt(batched_ds)


@keras_parameterized.run_all_keras_modes
class IntegerLookupOutputTest(keras_parameterized.TestCase,
                              preprocessing_test_utils.PreprocessingLayerTest):

  def test_int_output(self):
    vocab_data = [42, 1138, 725, 1729]
    input_array = np.array([[42, 1138, 725, 1729], [1729, 725, 42, 203]])
    expected_output = [[2, 3, 4, 5], [5, 4, 2, 1]]

    input_data = keras.Input(shape=(None,), dtype=dtypes.int64)
    layer = get_layer_class()()
    layer.set_vocabulary(vocab_data)
    int_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_array)
    self.assertAllEqual(expected_output, output_dataset)

  def test_output_shape(self):
    input_data = keras.Input(shape=(4,), dtype=dtypes.int64)
    layer = get_layer_class()(max_values=None, num_oov_indices=1)
    int_data = layer(input_data)
    self.assertAllEqual(int_data.shape[1:], input_data.shape[1:])

  def test_int_output_no_reserved_zero(self):
    vocab_data = [42, 1138, 725, 1729]
    input_array = np.array([[42, 1138, 725, 1729], [1729, 725, 42, 203]])
    expected_output = [[1, 2, 3, 4], [4, 3, 1, 0]]

    input_data = keras.Input(shape=(None,), dtype=dtypes.int64)
    layer = get_layer_class()(max_values=None, mask_value=None)
    layer.set_vocabulary(vocab_data)
    int_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_array)
    self.assertAllEqual(expected_output, output_dataset)

  def test_int_output_explicit_vocab(self):
    vocab_data = [42, 1138, 725, 1729]
    input_array = np.array([[42, 1138, 725, 1729], [1729, 725, 42, 203]])
    expected_output = [[2, 3, 4, 5], [5, 4, 2, 1]]

    input_data = keras.Input(shape=(None,), dtype=dtypes.int64)
    layer = get_layer_class()(
        vocabulary=vocab_data,
        max_values=None,
    )
    int_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_array)
    self.assertAllEqual(expected_output, output_dataset)

  def test_inverse_output(self):
    vocab_data = [0, -1, 42, 1138, 725, 1729]
    input_array = np.array([[2, 3, 4, 5], [5, 4, 2, 1]])
    expected_output = np.array([[42, 1138, 725, 1729], [1729, 725, 42, -1]])

    input_data = keras.Input(shape=(None,), dtype=dtypes.int64)
    layer = get_layer_class()(invert=True)
    layer.set_vocabulary(vocab_data)
    int_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_array)
    self.assertAllEqual(expected_output, output_dataset)

  def test_forward_backward_output(self):
    vocab_data = [42, 1138, 725, 1729]
    input_array = np.array([[42, 1138, 725, 1729], [1729, 725, 42, 203]])
    expected_output = np.array([[42, 1138, 725, 1729], [1729, 725, 42, -1]])

    input_data = keras.Input(shape=(None,), dtype=dtypes.int64)
    layer = get_layer_class()()
    inverse_layer = get_layer_class()()
    layer.set_vocabulary(vocab_data)
    inverse_layer = get_layer_class()(
        vocabulary=layer.get_vocabulary(), invert=True)
    int_data = layer(input_data)
    inverse_data = inverse_layer(int_data)
    model = keras.Model(inputs=input_data, outputs=inverse_data)
    output_dataset = model.predict(input_array)
    self.assertAllEqual(expected_output, output_dataset)


@keras_parameterized.run_all_keras_modes
class IntegerLookupVocabularyTest(
    keras_parameterized.TestCase,
    preprocessing_test_utils.PreprocessingLayerTest):

  def _write_to_temp_file(self, file_name, vocab_list):
    vocab_path = os.path.join(self.get_temp_dir(), file_name + ".txt")
    with gfile.GFile(vocab_path, "w") as writer:
      for vocab in vocab_list:
        writer.write(str(vocab) + "\n")
      writer.flush()
      writer.close()
    return vocab_path

  def test_int_output_explicit_vocab(self):
    vocab_data = [42, 1138, 725, 1729]
    input_array = np.array([[42, 1138, 725, 1729], [1729, 725, 42, 203]])
    expected_output = [[2, 3, 4, 5], [5, 4, 2, 1]]

    input_data = keras.Input(shape=(None,), dtype=dtypes.int64)
    layer = get_layer_class()(vocabulary=vocab_data)
    int_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_array)
    self.assertAllEqual(expected_output, output_dataset)

  def test_get_vocab_returns_int(self):
    vocab_data = [42, 1138, 725, 1729]
    expected_vocab = [0, -1, 42, 1138, 725, 1729]
    layer = get_layer_class()(vocabulary=vocab_data)
    layer_vocab = layer.get_vocabulary()
    self.assertAllEqual(expected_vocab, layer_vocab)
    self.assertIsInstance(layer_vocab[0], np.int64)

  def test_int_output_explicit_vocab_from_file(self):
    vocab_list = [42, 1138, 725, 1729]
    vocab_path = self._write_to_temp_file("vocab_file", vocab_list)

    input_array = np.array([[42, 1138, 725, 1729], [1729, 725, 42, 203]])
    expected_output = [[2, 3, 4, 5], [5, 4, 2, 1]]

    input_data = keras.Input(shape=(None,), dtype=dtypes.int64)
    layer = get_layer_class()(vocabulary=vocab_path)
    int_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_array)
    self.assertAllEqual(expected_output, output_dataset)

  def test_non_unique_vocab_fails(self):
    vocab_data = [42, 1138, 725, 1729, 1729]
    with self.assertRaisesRegex(ValueError, ".*repeated term.*1729.*"):
      _ = get_layer_class()(vocabulary=vocab_data)

  def test_non_unique_vocab_from_file_fails(self):
    vocab_list = [42, 1138, 725, 1729, 42]
    vocab_path = self._write_to_temp_file("repeat_vocab_file", vocab_list)
    with self.assertRaisesRegex(ValueError, ".*repeated term.*42.*"):
      _ = get_layer_class()(vocabulary=vocab_path)


@keras_parameterized.run_all_keras_modes(always_skip_eager=True)
class IntegerLookupSaveableTest(keras_parameterized.TestCase,
                                preprocessing_test_utils.PreprocessingLayerTest
                               ):

  def test_ops_are_not_added_with_multiple_get_set_weights(self):
    vocab_data = [42, 1138, 725, 1729]

    input_data = keras.Input(shape=(None,), dtype=dtypes.int64)
    layer = get_layer_class()(max_values=10)
    layer.set_vocabulary(vocab_data)
    int_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=int_data)
    weights = model.get_weights()
    model.set_weights(weights)
    keras.backend.get_session().graph.finalize()
    weights = model.get_weights()
    model.set_weights(weights)

  def test_layer_saving_with_h5(self):
    vocab_data = [42, 1138, 725, 1729]

    input_data = keras.Input(shape=(None,), dtype=dtypes.int64)
    layer = get_layer_class()(max_values=10)
    layer.set_vocabulary(vocab_data)
    int_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=int_data)
    path = os.path.join(self.get_temp_dir(), "model")
    with self.assertRaisesRegex(NotImplementedError,
                                "Save or restore weights that is not.*"):
      save.save_model(model, path, save_format="h5")


@keras_parameterized.run_all_keras_modes
class IntegerLookupErrorTest(keras_parameterized.TestCase,
                             preprocessing_test_utils.PreprocessingLayerTest):

  def test_too_long_vocab_fails_in_single_setting(self):
    vocab_data = [42, 1138, 725, 1729]

    layer = get_layer_class()(max_values=4, num_oov_indices=1)
    with self.assertRaisesRegex(ValueError,
                                "vocabulary larger than the maximum vocab.*"):
      layer.set_vocabulary(vocab_data)

  def test_zero_max_values_fails(self):
    with self.assertRaisesRegex(ValueError, ".*max_values.*"):
      _ = get_layer_class()(max_values=0, num_oov_indices=1)


@keras_parameterized.run_all_keras_modes
class IntegerLookupSavingTest(keras_parameterized.TestCase,
                              preprocessing_test_utils.PreprocessingLayerTest):

  def test_vocabulary_persistence_across_saving(self):
    vocab_data = [42, 1138, 725, 1729]
    input_array = np.array([[42, 1138, 725, 1729], [1729, 725, 42, 203]])
    expected_output = [[2, 3, 4, 5], [5, 4, 2, 1]]

    # Build and validate a golden model.
    input_data = keras.Input(shape=(None,), dtype=dtypes.int64)
    layer = get_layer_class()(max_values=None, num_oov_indices=1)
    layer.set_vocabulary(vocab_data)
    int_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_array)
    self.assertAllEqual(output_dataset, expected_output)

    # Save the model to disk.
    output_path = os.path.join(self.get_temp_dir(), "tf_keras_saved_model")
    model.save(output_path, save_format="tf")

    # Delete the session and graph to ensure that the loaded model is generated
    # from scratch.
    # TODO(b/149526183): Can't clear session when TF2 is disabled.
    if tf2.enabled():
      keras.backend.clear_session()

    loaded_model = keras.models.load_model(
        output_path, custom_objects={"IntegerLookup": get_layer_class()})

    # Ensure that the loaded model is unique (so that the save/load is real)
    self.assertIsNot(model, loaded_model)

    # Validate correctness of the new model.
    new_output_dataset = loaded_model.predict(input_array)
    self.assertAllEqual(new_output_dataset, expected_output)


if __name__ == "__main__":
  test.main()
