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

from tensorflow.python import keras

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.layers.preprocessing import index_lookup
from tensorflow.python.keras.layers.preprocessing import index_lookup_v1
from tensorflow.python.keras.layers.preprocessing import preprocessing_test_utils
from tensorflow.python.keras.saving import saved_model_experimental as saving
from tensorflow.python.keras.utils.generic_utils import CustomObjectScope
from tensorflow.python.platform import test


def get_layer_class():
  if context.executing_eagerly():
    return index_lookup.IndexLookup
  else:
    return index_lookup_v1.IndexLookup


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
              "max_tokens": None,
              "dtype": dtypes.int64,
          },
          "expected_output": [[2], [3], [4], [5], [5], [4], [2], [1]],
          "input_dtype":
              dtypes.int64
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
class IndexLookupLayerTest(keras_parameterized.TestCase,
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
      # IndexLookup), the concatenation fails. In real use cases, this may
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

    with CustomObjectScope({"IndexLookup": cls}):
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
class IndexLookupOutputTest(keras_parameterized.TestCase,
                            preprocessing_test_utils.PreprocessingLayerTest):

  def test_int_output(self):
    vocab_data = ["earth", "wind", "and", "fire"]
    input_array = np.array([["earth", "wind", "and", "fire"],
                            ["fire", "and", "earth", "michigan"]])
    expected_output = [[2, 3, 4, 5], [5, 4, 2, 1]]

    input_data = keras.Input(shape=(None,), dtype=dtypes.string)
    layer = get_layer_class()(max_tokens=None)
    layer.set_vocabulary(vocab_data)
    int_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_array)
    self.assertAllEqual(expected_output, output_dataset)

  def test_int_output_no_reserved_zero(self):
    vocab_data = ["earth", "wind", "and", "fire"]
    input_array = np.array([["earth", "wind", "and", "fire"],
                            ["fire", "and", "earth", "michigan"]])
    expected_output = [[1, 2, 3, 4], [4, 3, 1, 0]]

    input_data = keras.Input(shape=(None,), dtype=dtypes.string)
    layer = get_layer_class()(max_tokens=None, reserve_zero=False)
    layer.set_vocabulary(vocab_data)
    int_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_array)
    self.assertAllEqual(expected_output, output_dataset)

  def test_vocab_appending(self):
    vocab_data = [["earth", "wind"], ["and", "fire"]]
    input_array = np.array([["earth", "wind", "and", "fire"],
                            ["fire", "and", "earth", "michigan"]])
    expected_output = [[2, 3, 4, 5], [5, 4, 2, 1]]

    input_data = keras.Input(shape=(None,), dtype=dtypes.string)
    layer = get_layer_class()(max_tokens=5)
    layer.set_vocabulary(vocab_data[0])
    layer.set_vocabulary(vocab_data[1], append=True)
    int_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_array)
    self.assertAllClose(expected_output, output_dataset)


@keras_parameterized.run_all_keras_modes(always_skip_eager=True)
class IndexLookupSaveableTest(keras_parameterized.TestCase,
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


@keras_parameterized.run_all_keras_modes
class IndexLookupErrorTest(keras_parameterized.TestCase,
                           preprocessing_test_utils.PreprocessingLayerTest):

  def test_too_long_vocab_fails_in_single_setting(self):
    vocab_data = ["earth", "wind", "and", "fire"]

    layer = get_layer_class()(max_tokens=4)
    with self.assertRaisesRegex(ValueError,
                                "vocabulary larger than the maximum vocab.*"):
      layer.set_vocabulary(vocab_data)

  def test_too_long_vocab_fails_in_multiple_settings(self):
    vocab_data = [["earth", "wind"], ["and", "fire"]]
    layer = get_layer_class()(max_tokens=4)

    # The first time we call set_vocabulary, we're under the max_tokens
    # so it should be fine.
    layer.set_vocabulary(vocab_data[0])
    with self.assertRaisesRegex(ValueError,
                                "vocabulary larger than the maximum vocab.*"):
      layer.set_vocabulary(vocab_data[1], append=True)

  def test_zero_max_tokens_fails(self):
    with self.assertRaisesRegex(ValueError, ".*max_tokens.*"):
      _ = get_layer_class()(max_tokens=0)


@keras_parameterized.run_all_keras_modes
class IndexLookupSavingTest(keras_parameterized.TestCase,
                            preprocessing_test_utils.PreprocessingLayerTest):

  def test_saving_errors(self):
    vocab_data = ["earth", "wind", "and", "fire"]

    # Build and validate a golden model.
    input_data = keras.Input(shape=(None,), dtype=dtypes.string)
    layer = get_layer_class()(max_tokens=None)
    layer.set_vocabulary(vocab_data)
    int_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=int_data)

    # Save the model to disk.
    output_path = os.path.join(self.get_temp_dir(), "tf_keras_saved_model")

    with self.assertRaisesRegex(NotImplementedError, ".*Saving is not yet.*"):
      model.save(output_path, save_format="tf")

  def test_saving_errors_when_nested(self):
    vocab_data = ["earth", "wind", "and", "fire"]

    # Build and validate a golden model.
    input_data = keras.Input(shape=(None,), dtype=dtypes.string)
    layer = get_layer_class()(max_tokens=None)
    layer.set_vocabulary(vocab_data)
    int_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=int_data)

    outer_input = keras.Input(shape=(None,), dtype=dtypes.string)
    outer_output = model(outer_input)
    outer_model = keras.Model(inputs=outer_input, outputs=outer_output)

    # Save the model to disk.
    output_path = os.path.join(self.get_temp_dir(), "tf_keras_saved_model")

    with self.assertRaisesRegex(NotImplementedError, ".*Saving is not yet.*"):
      outer_model.save(output_path, save_format="tf")

  def DISABLED_test_vocabulary_persistence_across_saving(self):
    vocab_data = ["earth", "wind", "and", "fire"]
    input_array = np.array([["earth", "wind", "and", "fire"],
                            ["fire", "and", "earth", "michigan"]])
    expected_output = [[2, 3, 4, 5], [5, 4, 2, 1]]

    # Build and validate a golden model.
    input_data = keras.Input(shape=(None,), dtype=dtypes.string)
    layer = get_layer_class()(max_tokens=None)
    layer.set_vocabulary(vocab_data)
    int_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_array)
    self.assertAllEqual(output_dataset, expected_output)

    # Save the model to disk.
    output_path = os.path.join(self.get_temp_dir(), "tf_keras_saved_model")
    model.save(output_path, save_format="tf")
    loaded_model = saving.load_from_saved_model(
        output_path, custom_objects={"IndexLookup": get_layer_class()})

    # Ensure that the loaded model is unique (so that the save/load is real)
    self.assertIsNot(model, loaded_model)

    # Validate correctness of the new model.
    new_output_dataset = loaded_model.predict(input_array)
    self.assertAllEqual(new_output_dataset, expected_output)

  def DISABLED_test_vocabulary_persistence_across_saving_with_tfidf(self):
    vocab_data = ["earth", "wind", "and", "fire"]
    tfidf_data = [.5, .25, .2, .125]
    input_array = np.array([["earth", "wind", "and", "earth"],
                            ["ohio", "fire", "earth", "michigan"]])

    # pyformat: disable
    # pylint: disable=bad-whitespace
    expected_output = [[ 0,  1, .25, .2,    0],
                       [.1, .5,   0,  0, .125]]
    # pylint: enable=bad-whitespace
    # pyformat: enable

    # Build and validate a golden model.
    input_data = keras.Input(shape=(None,), dtype=dtypes.string)
    layer = get_layer_class()(
        max_tokens=5,
        standardize=None,
        split=None,
        output_mode=index_lookup.TFIDF)
    layer.set_vocabulary(vocab_data, df_data=tfidf_data, oov_df_value=.05)

    int_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_array)
    self.assertAllClose(output_dataset, expected_output)

    # Save the model to disk.
    output_path = os.path.join(self.get_temp_dir(), "tf_keras_saved_model")
    model.save(output_path, save_format="tf")
    loaded_model = saving.load_from_saved_model(
        output_path, custom_objects={"IndexLookup": get_layer_class()})

    # Ensure that the loaded model is unique (so that the save/load is real)
    self.assertIsNot(model, loaded_model)

    # Validate correctness of the new model.
    new_output_dataset = loaded_model.predict(input_array)
    self.assertAllClose(new_output_dataset, expected_output)


@keras_parameterized.run_all_keras_modes
class IndexLookupCombinerTest(keras_parameterized.TestCase,
                              preprocessing_test_utils.PreprocessingLayerTest):

  def compare_text_accumulators(self, a, b, msg=None):
    if a is None or b is None:
      self.assertAllEqual(a, b, msg=msg)

    self.assertAllEqual(a.count_dict, b.count_dict, msg=msg)

  compare_accumulators = compare_text_accumulators

  def update_accumulator(self, accumulator, data):
    accumulator.count_dict.update(dict(zip(data["vocab"], data["counts"])))

    return accumulator

  def test_combiner_api_compatibility_int_mode(self):
    data = np.array([["earth", "wind", "and", "fire"],
                     ["earth", "wind", "and", "michigan"]])
    combiner = index_lookup._IndexLookupCombiner()
    expected_accumulator_output = {
        "vocab": np.array(["and", "earth", "wind", "fire", "michigan"]),
        "counts": np.array([2, 2, 2, 1, 1]),
    }
    expected_extract_output = {
        "vocab": np.array(["wind", "earth", "and", "michigan", "fire"]),
    }
    expected_accumulator = combiner._create_accumulator()
    expected_accumulator = self.update_accumulator(expected_accumulator,
                                                   expected_accumulator_output)
    self.validate_accumulator_serialize_and_deserialize(combiner, data,
                                                        expected_accumulator)
    self.validate_accumulator_uniqueness(combiner, data)
    self.validate_accumulator_extract(combiner, data, expected_extract_output)

  # TODO(askerryryan): Add tests confirming equivalence to behavior of
  # existing tf.keras.preprocessing.text.Tokenizer.
  @parameterized.named_parameters(
      {
          "testcase_name":
              "top_k_smaller_than_full_vocab",
          "data":
              np.array([["earth", "wind"], ["fire", "wind"], ["and"],
                        ["fire", "wind"]]),
          "vocab_size":
              3,
          "expected_accumulator_output": {
              "vocab": np.array(["wind", "fire", "earth", "and"]),
              "counts": np.array([3, 2, 1, 1]),
          },
          "expected_extract_output": {
              "vocab": np.array(["wind", "fire", "earth"]),
          },
      },
      {
          "testcase_name":
              "top_k_larger_than_full_vocab",
          "data":
              np.array([["earth", "wind"], ["fire", "wind"], ["and"],
                        ["fire", "wind"]]),
          "vocab_size":
              10,
          "expected_accumulator_output": {
              "vocab": np.array(["wind", "fire", "earth", "and"]),
              "counts": np.array([3, 2, 1, 1]),
          },
          "expected_extract_output": {
              "vocab": np.array(["wind", "fire", "earth", "and"]),
          },
      },
      {
          "testcase_name":
              "no_top_k",
          "data":
              np.array([["earth", "wind"], ["fire", "wind"], ["and"],
                        ["fire", "wind"]]),
          "vocab_size":
              None,
          "expected_accumulator_output": {
              "vocab": np.array(["wind", "fire", "earth", "and"]),
              "counts": np.array([3, 2, 1, 1]),
          },
          "expected_extract_output": {
              "vocab": np.array(["wind", "fire", "earth", "and"]),
          },
      },
      {
          "testcase_name": "single_element_per_row",
          "data": np.array([["earth"], ["wind"], ["fire"], ["wind"], ["and"]]),
          "vocab_size": 3,
          "expected_accumulator_output": {
              "vocab": np.array(["wind", "and", "earth", "fire"]),
              "counts": np.array([2, 1, 1, 1]),
          },
          "expected_extract_output": {
              "vocab": np.array(["wind", "fire", "earth"]),
          },
      },
      # Which tokens are retained are based on global frequency, and thus are
      # sensitive to frequency within a document. In contrast, because idf only
      # considers the presence of a token in a document, it is insensitive
      # to the frequency of the token within the document.
      {
          "testcase_name":
              "retained_tokens_sensitive_to_within_document_frequency",
          "data":
              np.array([["earth", "earth"], ["wind", "wind"], ["fire", "fire"],
                        ["wind", "wind"], ["and", "michigan"]]),
          "vocab_size":
              3,
          "expected_accumulator_output": {
              "vocab": np.array(["wind", "earth", "fire", "and", "michigan"]),
              "counts": np.array([4, 2, 2, 1, 1]),
          },
          "expected_extract_output": {
              "vocab": np.array(["wind", "fire", "earth"]),
          },
      })
  def test_combiner_computation(self, data, vocab_size,
                                expected_accumulator_output,
                                expected_extract_output):
    combiner = index_lookup._IndexLookupCombiner(vocab_size=vocab_size)
    expected_accumulator = combiner._create_accumulator()
    expected_accumulator = self.update_accumulator(expected_accumulator,
                                                   expected_accumulator_output)
    self.validate_accumulator_computation(combiner, data, expected_accumulator)
    self.validate_accumulator_extract(combiner, data, expected_extract_output)


if __name__ == "__main__":
  test.main()
