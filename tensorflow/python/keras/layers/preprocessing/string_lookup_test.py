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

import os

from absl.testing import parameterized
import numpy as np
import six

from tensorflow.python import keras

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.layers.preprocessing import preprocessing_test_utils
from tensorflow.python.keras.layers.preprocessing import string_lookup
from tensorflow.python.keras.layers.preprocessing import string_lookup_v1
from tensorflow.python.keras.saving import save
from tensorflow.python.keras.utils.generic_utils import CustomObjectScope
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test


def get_layer_class():
  if context.executing_eagerly():
    return string_lookup.StringLookup
  else:
    return string_lookup_v1.StringLookup


def _get_end_to_end_test_cases():
  test_cases = (
      {
          "testcase_name":
              "test_strings_soft_vocab_cap",
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
          "expected_output": [[2], [3], [4], [5], [5], [4], [2], [1]],
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


@keras_parameterized.run_all_keras_modes
class StringLookupLayerTest(keras_parameterized.TestCase,
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


@keras_parameterized.run_all_keras_modes
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
    expected_output = [[2, 3, 4, 5], [5, 4, 2, 1]]

    input_data = keras.Input(shape=(None,), dtype=dtypes.string)
    layer = get_layer_class()(vocabulary=vocab_data)
    int_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_array)
    self.assertAllEqual(expected_output, output_dataset)

  def test_get_vocab_returns_str(self):
    vocab_data = ["earth", "wind", "and", "fire"]
    expected_vocab = ["", "[OOV]", "earth", "wind", "and", "fire"]
    layer = get_layer_class()(vocabulary=vocab_data)
    layer_vocab = layer.get_vocabulary()
    self.assertAllEqual(expected_vocab, layer_vocab)
    self.assertIsInstance(layer_vocab[0], six.text_type)

  def test_int_output_explicit_vocab_from_file(self):
    vocab_list = ["earth", "wind", "and", "fire"]
    vocab_path = self._write_to_temp_file("vocab_file", vocab_list)

    input_array = np.array([["earth", "wind", "and", "fire"],
                            ["fire", "and", "earth", "michigan"]])
    expected_output = [[2, 3, 4, 5], [5, 4, 2, 1]]

    input_data = keras.Input(shape=(None,), dtype=dtypes.string)
    layer = get_layer_class()(vocabulary=vocab_path)
    int_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_array)
    self.assertAllEqual(expected_output, output_dataset)

  def test_non_unique_vocab_fails(self):
    vocab_data = ["earth", "wind", "and", "fire", "fire"]
    with self.assertRaisesRegex(ValueError, ".*repeated term.*fire.*"):
      _ = get_layer_class()(vocabulary=vocab_data)

  def test_non_unique_vocab_from_file_fails(self):
    vocab_list = ["earth", "wind", "and", "fire", "earth"]
    vocab_path = self._write_to_temp_file("repeat_vocab_file", vocab_list)
    with self.assertRaisesRegex(ValueError, ".*repeated term.*earth.*"):
      _ = get_layer_class()(vocabulary=vocab_path)


@keras_parameterized.run_all_keras_modes(always_skip_eager=True)
class StringLookupSaveableTest(keras_parameterized.TestCase,
                               preprocessing_test_utils.PreprocessingLayerTest):

  def test_ops_are_not_added_with_multiple_get_set_weights(self):
    vocab_data = ["earth", "wind", "and", "fire"]

    input_data = keras.Input(shape=(None,), dtype=dtypes.string)
    layer = get_layer_class()(max_tokens=10)
    layer.set_vocabulary(vocab_data)
    int_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=int_data)
    weights = model.get_weights()
    model.set_weights(weights)
    keras.backend.get_session().graph.finalize()
    weights = model.get_weights()
    model.set_weights(weights)

  def test_layer_saving_with_h5(self):
    vocab_data = ["earth", "wind", "and", "fire"]

    input_data = keras.Input(shape=(None,), dtype=dtypes.string)
    layer = get_layer_class()(max_tokens=10)
    layer.set_vocabulary(vocab_data)
    int_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=int_data)
    path = os.path.join(self.get_temp_dir(), "model")
    with self.assertRaisesRegex(NotImplementedError,
                                "Save or restore weights that is not.*"):
      save.save_model(model, path, save_format="h5")


if __name__ == "__main__":
  test.main()
