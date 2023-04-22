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

import os
from absl.testing import parameterized
import numpy as np

from tensorflow.python import keras

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import errors_impl
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.layers.preprocessing import preprocessing_test_utils
from tensorflow.python.keras.layers.preprocessing import string_lookup
from tensorflow.python.keras.utils.generic_utils import CustomObjectScope
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test


def _get_end_to_end_test_cases():
  test_cases = (
      {
          "testcase_name": "test_strings_soft_vocab_cap",
          # Create an array where 'earth' is the most frequent term, followed by
          # 'wind', then 'and', then 'fire'. This ensures that the vocab
          # accumulator is sorting by frequency.
          "vocab_data":
              np.array([["fire"], ["earth"], ["earth"], ["earth"], ["earth"],
                        ["wind"], ["wind"], ["wind"], ["and"], ["and"]]),
          "input_data":
              np.array([["earth"], ["wind"], ["and"], ["fire"], ["fire"],
                        ["and"], ["earth"], ["michigan"]]),
          "kwargs": {
              "max_tokens": None,
          },
          "expected_output": [[1], [2], [3], [4], [4], [3], [1], [0]],
          "input_dtype":
              dtypes.string
      },
  )

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


@keras_parameterized.run_all_keras_modes(always_skip_v1=True)
class StringLookupLayerTest(keras_parameterized.TestCase,
                            preprocessing_test_utils.PreprocessingLayerTest):

  @parameterized.named_parameters(*_get_end_to_end_test_cases())
  def test_layer_end_to_end_with_adapt(self, vocab_data, input_data, kwargs,
                                       use_dataset, expected_output,
                                       input_dtype):
    cls = string_lookup.StringLookup
    expected_output_dtype = dtypes.int64
    input_shape = input_data.shape

    if use_dataset:
      # Keras APIs expect batched datasets.
      # TODO(rachelim): `model.predict` predicts the result on each
      # dataset batch separately, then tries to concatenate the results
      # together. When the results have different shapes on the non-concat
      # axis (which can happen in the output_mode = INT case for
      # StringLookup), the concatenation fails. In real use cases, this may
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

    with CustomObjectScope({"StringLookup": cls}):
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


@keras_parameterized.run_all_keras_modes(always_skip_v1=True)
class StringLookupVocabularyTest(keras_parameterized.TestCase,
                                 preprocessing_test_utils.PreprocessingLayerTest
                                ):

  def _write_to_temp_file(self, file_name, vocab_list):
    vocab_path = os.path.join(self.get_temp_dir(), file_name + ".txt")
    with gfile.GFile(vocab_path, "w") as writer:
      for vocab in vocab_list:
        writer.write(vocab + "\n")
      writer.flush()
      writer.close()
    return vocab_path

  def test_int_output_explicit_vocab(self):
    vocab_data = ["earth", "wind", "and", "fire"]
    input_array = np.array([["earth", "wind", "and", "fire"],
                            ["fire", "and", "earth", "michigan"]])
    expected_output = [[1, 2, 3, 4], [4, 3, 1, 0]]

    input_data = keras.Input(shape=(None,), dtype=dtypes.string)
    layer = string_lookup.StringLookup(vocabulary=vocab_data)
    int_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=int_data)
    output_data = model.predict(input_array)
    self.assertAllEqual(expected_output, output_data)

  def test_int_output_explicit_vocab_with_special_tokens(self):
    vocab_data = ["", "[UNK]", "earth", "wind", "and", "fire"]
    input_array = np.array([["earth", "wind", "and", "fire"],
                            ["fire", "and", "earth", "michigan"]])
    expected_output = [[2, 3, 4, 5], [5, 4, 2, 1]]

    input_data = keras.Input(shape=(None,), dtype=dtypes.string)
    layer = string_lookup.StringLookup(vocabulary=vocab_data, mask_token="")
    int_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=int_data)
    output_data = model.predict(input_array)
    self.assertAllEqual(expected_output, output_data)

  def test_int_output_no_oov(self):
    vocab_data = ["earth", "wind", "and", "fire"]
    valid_input = np.array([["earth", "wind", "and", "fire"],
                            ["fire", "and", "earth", ""]])
    invalid_input = np.array([["earth", "wind", "and", "michigan"],
                              ["fire", "and", "earth", "michigan"]])
    expected_output = [[1, 2, 3, 4], [4, 3, 1, 0]]

    input_data = keras.Input(shape=(None,), dtype=dtypes.string)
    layer = string_lookup.StringLookup(
        vocabulary=vocab_data, mask_token="", num_oov_indices=0)
    int_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=int_data)
    output_data = model.predict(valid_input)
    self.assertAllEqual(expected_output, output_data)
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "found OOV values.*michigan"):
      _ = model.predict(invalid_input)

  def test_no_vocab(self):
    with self.assertRaisesRegex(
        ValueError, "You must set the layer's vocabulary"):
      layer = string_lookup.StringLookup()
      layer([["a"]])

  def test_one_hot_output(self):
    vocab_data = ["earth", "wind", "and", "fire"]
    input_array = np.array(["earth", "wind", "and", "fire", "michigan"])
    expected_output = [
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0],
    ]

    input_data = keras.Input(shape=(1,), dtype=dtypes.string)
    layer = string_lookup.StringLookup(
        vocabulary=vocab_data, output_mode="one_hot")
    res = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=res)
    output_data = model.predict(input_array)
    self.assertAllEqual(expected_output, output_data)

  def test_multi_hot_output(self):
    vocab_data = ["earth", "wind", "and", "fire"]
    input_array = np.array([["earth", "wind", "and", "fire"],
                            ["fire", "and", "earth", "michigan"]])
    expected_output = [[0, 1, 1, 1, 1], [1, 1, 0, 1, 1]]

    input_data = keras.Input(shape=(None,), dtype=dtypes.string)
    layer = string_lookup.StringLookup(
        vocabulary=vocab_data, output_mode="multi_hot")
    res = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=res)
    output_data = model.predict(input_array)
    self.assertAllEqual(expected_output, output_data)

  def test_count_output(self):
    vocab_data = ["earth", "wind", "and", "fire"]
    input_array = np.array([["earth", "earth", "fire", "fire"],
                            ["fire", "and", "earth", "michigan"]])
    expected_output = [[0, 2, 0, 0, 2], [1, 1, 0, 1, 1]]

    input_data = keras.Input(shape=(None,), dtype=dtypes.string)
    layer = string_lookup.StringLookup(
        vocabulary=vocab_data, output_mode="count")
    res = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=res)
    output_data = model.predict(input_array)
    self.assertAllEqual(expected_output, output_data)

  def test_sparse_output(self):
    vocab_data = ["earth", "wind", "and", "fire"]

    input_data = keras.Input(shape=(None,), dtype=dtypes.string)
    layer = string_lookup.StringLookup(
        vocabulary=vocab_data, output_mode="multi_hot", sparse=True)
    res = layer(input_data)
    self.assertTrue(res.__class__.__name__, "SparseKerasTensor")

  def test_get_vocab_returns_str(self):
    vocab_data = ["earth", "wind", "and", "fire"]
    expected_vocab = ["[UNK]", "earth", "wind", "and", "fire"]
    layer = string_lookup.StringLookup(vocabulary=vocab_data)
    layer_vocab = layer.get_vocabulary()
    self.assertAllEqual(expected_vocab, layer_vocab)
    self.assertIsInstance(layer_vocab[0], str)

    inverse_layer = string_lookup.StringLookup(
        vocabulary=layer.get_vocabulary(), invert=True)
    layer_vocab = inverse_layer.get_vocabulary()
    self.assertAllEqual(expected_vocab, layer_vocab)
    self.assertIsInstance(layer_vocab[0], str)

  def test_int_output_explicit_vocab_from_file(self):
    vocab_list = ["earth", "wind", "and", "fire"]
    vocab_path = self._write_to_temp_file("vocab_file", vocab_list)

    input_array = np.array([["earth", "wind", "and", "fire"],
                            ["fire", "and", "earth", "michigan"]])
    expected_output = [[1, 2, 3, 4], [4, 3, 1, 0]]

    input_data = keras.Input(shape=(None,), dtype=dtypes.string)
    layer = string_lookup.StringLookup(vocabulary=vocab_path)
    int_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=int_data)
    output_data = model.predict(input_array)
    self.assertAllEqual(expected_output, output_data)

  def test_int_output_explicit_vocab_from_file_via_setter(self):
    vocab_list = ["earth", "wind", "and", "fire"]
    vocab_path = self._write_to_temp_file("vocab_file", vocab_list)

    input_array = np.array([["earth", "wind", "and", "fire"],
                            ["fire", "and", "earth", "michigan"]])
    expected_output = [[1, 2, 3, 4], [4, 3, 1, 0]]

    input_data = keras.Input(shape=(None,), dtype=dtypes.string)
    layer = string_lookup.StringLookup()
    layer.set_vocabulary(vocab_path)
    int_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=int_data)
    output_data = model.predict(input_array)
    self.assertAllEqual(expected_output, output_data)

  def test_non_unique_vocab_fails(self):
    vocab_data = ["earth", "wind", "and", "fire", "fire"]
    with self.assertRaisesRegex(ValueError, ".*repeated term.*fire.*"):
      _ = string_lookup.StringLookup(vocabulary=vocab_data)

  def test_non_unique_vocab_from_file_fails(self):
    vocab_list = ["earth", "wind", "and", "fire", "earth"]
    vocab_path = self._write_to_temp_file("repeat_vocab_file", vocab_list)
    with self.assertRaisesRegex(
        errors_impl.FailedPreconditionError,
        "HashTable has different value for same key.*earth"):
      _ = string_lookup.StringLookup(vocabulary=vocab_path)

  def test_inverse_layer(self):
    vocab_data = ["earth", "wind", "and", "fire"]
    input_array = np.array([[2, 3, 4, 5], [5, 4, 2, 0]])
    expected_output = np.array([["earth", "wind", "and", "fire"],
                                ["fire", "and", "earth", ""]])

    input_data = keras.Input(shape=(None,), dtype=dtypes.int64)
    layer = string_lookup.StringLookup(
        vocabulary=vocab_data, invert=True, mask_token="")
    int_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=int_data)
    output_data = model.predict(input_array)
    self.assertAllEqual(expected_output, output_data)

  def test_inverse_layer_from_file(self):
    vocab_data = ["earth", "wind", "and", "fire"]
    input_array = np.array([[1, 2, 3, 4], [4, 3, 1, 0]])
    expected_output = np.array([["earth", "wind", "and", "fire"],
                                ["fire", "and", "earth", "[UNK]"]])
    vocab_path = self._write_to_temp_file("vocab_file", vocab_data)

    input_data = keras.Input(shape=(None,), dtype=dtypes.int64)
    layer = string_lookup.StringLookup(vocabulary=vocab_path, invert=True)
    int_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=int_data)
    output_data = model.predict(input_array)
    self.assertAllEqual(expected_output, output_data)

  def test_inverse_layer_from_file_with_mask(self):
    vocab_data = ["earth", "wind", "and", "fire"]
    input_array = np.array([[2, 3, 4, 5], [5, 4, 2, 0]])
    expected_output = np.array([["earth", "wind", "and", "fire"],
                                ["fire", "and", "earth", "[M]"]])
    vocab_path = self._write_to_temp_file("vocab_file", vocab_data)

    input_data = keras.Input(shape=(None,), dtype=dtypes.int64)
    layer = string_lookup.StringLookup(
        vocabulary=vocab_path, invert=True, mask_token="[M]")
    int_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=int_data)
    output_data = model.predict(input_array)
    self.assertAllEqual(expected_output, output_data)

  def test_forward_backward_explicit_vocab(self):
    vocab_data = ["earth", "wind", "and", "fire"]
    input_array = np.array([["earth", "wind", "and", "fire"],
                            ["fire", "and", "earth", "michigan"]])
    expected_output = np.array([["earth", "wind", "and", "fire"],
                                ["fire", "and", "earth", "[UNK]"]])

    input_data = keras.Input(shape=(None,), dtype=dtypes.string)
    layer = string_lookup.StringLookup(vocabulary=vocab_data)
    invert_layer = string_lookup.StringLookup(
        vocabulary=vocab_data, invert=True)
    int_data = layer(input_data)
    out_data = invert_layer(int_data)
    model = keras.Model(inputs=input_data, outputs=out_data)
    output_data = model.predict(input_array)
    self.assertAllEqual(expected_output, output_data)

  def test_forward_backward_adapted_vocab(self):
    adapt_data = ["earth", "wind", "and", "fire"]
    input_array = np.array([["earth", "wind", "and", "fire"],
                            ["fire", "and", "earth", "michigan"]])
    expected_output = np.array([["earth", "wind", "and", "fire"],
                                ["fire", "and", "earth", "[UNK]"]])

    input_data = keras.Input(shape=(None,), dtype=dtypes.string)
    layer = string_lookup.StringLookup()
    layer.adapt(adapt_data)
    invert_layer = string_lookup.StringLookup(
        vocabulary=layer.get_vocabulary(), invert=True)
    int_data = layer(input_data)
    out_data = invert_layer(int_data)
    model = keras.Model(inputs=input_data, outputs=out_data)
    output_data = model.predict(input_array)
    self.assertAllEqual(expected_output, output_data)

  def test_ragged_string_input_multi_bucket(self):
    vocab_data = ["earth", "wind", "and", "fire"]
    input_array = ragged_factory_ops.constant([["earth", "wind", "fire"],
                                               ["fire", "and", "earth",
                                                "ohio"]])
    expected_output = [[2, 3, 5], [5, 4, 2, 1]]

    input_data = keras.Input(shape=(None,), dtype=dtypes.string, ragged=True)
    layer = string_lookup.StringLookup(num_oov_indices=2)
    layer.set_vocabulary(vocab_data)
    int_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=int_data)
    output_data = model.predict(input_array)
    self.assertAllEqual(expected_output, output_data)

  def test_tensor_vocab(self):
    vocab_data = ["[UNK]", "wind", "and", "fire"]
    vocab_tensor = constant_op.constant(vocab_data)
    layer = string_lookup.StringLookup(vocabulary=vocab_tensor)
    returned_vocab = layer.get_vocabulary()
    self.assertAllEqual(vocab_data, returned_vocab)
    self.assertAllEqual(layer.vocabulary_size(), 4)
    fn = def_function.function(lambda: layer.set_vocabulary(vocab_tensor))
    with self.assertRaisesRegex(RuntimeError, "Cannot set a tensor vocabulary"):
      fn()

if __name__ == "__main__":
  test.main()
