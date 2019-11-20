# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
from tensorflow.python.keras import backend
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.layers import convolutional
from tensorflow.python.keras.layers import core
from tensorflow.python.keras.layers import embeddings
from tensorflow.python.keras.layers.preprocessing import text_vectorization
from tensorflow.python.keras.layers.preprocessing import text_vectorization_v1
from tensorflow.python.keras.layers.preprocessing import preprocessing_test_utils
from tensorflow.python.keras.saving import saved_model_experimental as saving
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils.generic_utils import CustomObjectScope
from tensorflow.python.ops import gen_string_ops
from tensorflow.python.ops.ragged import ragged_string_ops
from tensorflow.python.platform import test


def get_layer_class():
  if context.executing_eagerly():
    return text_vectorization.TextVectorization
  else:
    return text_vectorization_v1.TextVectorization


def _get_end_to_end_test_cases():
  test_cases = (
      {
          "testcase_name":
              "test_simple_tokens_int_mode",
          # Create an array where 'earth' is the most frequent term, followed by
          # 'wind', then 'and', then 'fire'. This ensures that the vocab accumulator
          # is sorting by frequency.
          "vocab_data":
              np.array([["fire"], ["earth"], ["earth"], ["earth"], ["earth"],
                        ["wind"], ["wind"], ["wind"], ["and"], ["and"]]),
          "input_data":
              np.array([["earth"], ["wind"], ["and"], ["fire"], ["fire"],
                        ["and"], ["earth"], ["michigan"]]),
          "kwargs": {
              "max_tokens": None,
              "standardize": None,
              "split": None,
              "output_mode": text_vectorization.INT
          },
          "expected_output": [[2], [3], [4], [5], [5], [4], [2], [1]],
      },
      {
          "testcase_name":
              "test_documents_int_mode",
          "vocab_data":
              np.array([["fire earth earth"], ["earth earth"], ["wind wind"],
                        ["and wind and"]]),
          "input_data":
              np.array([["earth wind and"], ["fire fire"], ["and earth"],
                        ["michigan"]]),
          "kwargs": {
              "max_tokens": None,
              "standardize": None,
              "split": text_vectorization.SPLIT_ON_WHITESPACE,
              "output_mode": text_vectorization.INT
          },
          "expected_output": [[2, 3, 4], [5, 5, 0], [4, 2, 0], [1, 0, 0]],
      },
      {
          "testcase_name":
              "test_documents_1d_input_int_mode",
          "vocab_data":
              np.array([
                  "fire earth earth", "earth earth", "wind wind", "and wind and"
              ]),
          "input_data":
              np.array([["earth wind and"], ["fire fire"], ["and earth"],
                        ["michigan"]]),
          "kwargs": {
              "max_tokens": None,
              "standardize": None,
              "split": text_vectorization.SPLIT_ON_WHITESPACE,
              "output_mode": text_vectorization.INT
          },
          "expected_output": [[2, 3, 4], [5, 5, 0], [4, 2, 0], [1, 0, 0]],
      },
      {
          "testcase_name":
              "test_simple_tokens_binary_mode",
          "vocab_data":
              np.array([["fire"], ["earth"], ["earth"], ["earth"], ["earth"],
                        ["wind"], ["wind"], ["wind"], ["and"], ["and"]]),
          "input_data":
              np.array([["earth"], ["wind"], ["and"], ["fire"], ["fire"],
                        ["and"], ["earth"], ["michigan"]]),
          "kwargs": {
              "max_tokens": 5,
              "standardize": None,
              "split": None,
              "output_mode": text_vectorization.BINARY
          },
          "expected_output": [[0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 1], [0, 0, 0, 0, 1], [0, 0, 0, 1, 0],
                              [0, 1, 0, 0, 0], [1, 0, 0, 0, 0]],
      },
      {
          "testcase_name":
              "test_documents_binary_mode",
          "vocab_data":
              np.array([["fire earth earth"], ["earth earth"], ["wind wind"],
                        ["and wind and"]]),
          "input_data":
              np.array([["earth wind"], ["and"], ["fire fire"],
                        ["earth michigan"]]),
          "kwargs": {
              "max_tokens": 5,
              "standardize": None,
              "split": text_vectorization.SPLIT_ON_WHITESPACE,
              "output_mode": text_vectorization.BINARY
          },
          "expected_output": [[0, 1, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1],
                              [1, 1, 0, 0, 0]],
      },
      {
          "testcase_name":
              "test_simple_tokens_count_mode",
          "vocab_data":
              np.array([["fire"], ["earth"], ["earth"], ["earth"], ["earth"],
                        ["wind"], ["wind"], ["wind"], ["and"], ["and"]]),
          "input_data":
              np.array([["earth"], ["wind"], ["and"], ["fire"], ["fire"],
                        ["and"], ["earth"], ["michigan"]]),
          "kwargs": {
              "max_tokens": 5,
              "standardize": None,
              "split": None,
              "output_mode": text_vectorization.COUNT
          },
          "expected_output": [[0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 1], [0, 0, 0, 0, 1], [0, 0, 0, 1, 0],
                              [0, 1, 0, 0, 0], [1, 0, 0, 0, 0]],
      },
      {
          "testcase_name":
              "test_documents_count_mode",
          "vocab_data":
              np.array([["fire earth earth"], ["earth earth"], ["wind wind"],
                        ["and wind and"]]),
          "input_data":
              np.array([["earth wind"], ["and"], ["fire fire"],
                        ["earth michigan"]]),
          "kwargs": {
              "max_tokens": 5,
              "standardize": None,
              "split": text_vectorization.SPLIT_ON_WHITESPACE,
              "output_mode": text_vectorization.COUNT
          },
          "expected_output": [[0, 1, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 2],
                              [1, 1, 0, 0, 0]],
      },
      {
          "testcase_name":
              "test_tokens_idf_mode",
          "vocab_data":
              np.array([["fire"], ["earth"], ["earth"], ["earth"], ["earth"],
                        ["wind"], ["wind"], ["wind"], ["and"], ["and"]]),
          "input_data":
              np.array([["earth"], ["wind"], ["and"], ["fire"], ["fire"],
                        ["and"], ["earth"], ["michigan"]]),
          "kwargs": {
              "max_tokens": 5,
              "standardize": None,
              "split": None,
              "output_mode": text_vectorization.TFIDF
          },
          "expected_output": [[0, 1.098612, 0, 0, 0], [0, 0, 1.252763, 0, 0],
                              [0, 0, 0, 1.466337, 0], [0, 0, 0, 0, 1.7917595],
                              [0, 0, 0, 0, 1.7917595], [0, 0, 0, 1.4663371, 0],
                              [0, 1.098612, 0, 0, 0], [2.3978953, 0, 0, 0, 0]],
      },
      {
          "testcase_name":
              "test_documents_idf_mode",
          "vocab_data":
              np.array([["fire earth earth"], ["earth earth"], ["wind wind"],
                        ["and wind and"]]),
          "input_data":
              np.array([["earth wind"], ["and"], ["fire fire"],
                        ["earth michigan"]]),
          "kwargs": {
              "max_tokens": 5,
              "standardize": None,
              "split": text_vectorization.SPLIT_ON_WHITESPACE,
              "output_mode": text_vectorization.TFIDF
          },
          "expected_output": [[0., 0.847298, 0.847298, 0., 0.],
                              [0., 0., 0., 1.098612, 0.],
                              [0., 0., 0., 0., 2.197225],
                              [1.609438, 0.847298, 0., 0., 0.]],
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
class TextVectorizationLayerTest(keras_parameterized.TestCase,
                                 preprocessing_test_utils.PreprocessingLayerTest
                                ):

  @parameterized.named_parameters(*_get_end_to_end_test_cases())
  def test_layer_end_to_end_with_adapt(self, vocab_data, input_data, kwargs,
                                       use_dataset, expected_output):
    cls = get_layer_class()
    if kwargs.get("output_mode") == text_vectorization.TFIDF:
      expected_output_dtype = dtypes.float32
    else:
      expected_output_dtype = dtypes.int64
    input_shape = input_data.shape

    if use_dataset:
      # Keras APIs expect batched datasets.
      # TODO(rachelim): `model.predict` predicts the result on each
      # dataset batch separately, then tries to concatenate the results
      # together. When the results have different shapes on the non-concat
      # axis (which can happen in the output_mode = INT case for
      # TextVectorization), the concatenation fails. In real use cases, this may
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

    with CustomObjectScope({"TextVectorization": cls}):
      output_data = testing_utils.layer_test(
          cls,
          kwargs=kwargs,
          input_shape=input_shape,
          input_data=input_data,
          input_dtype=dtypes.string,
          expected_output_dtype=expected_output_dtype,
          validate_training=False,
          adapt_data=vocab_data)
    self.assertAllClose(expected_output, output_data)


@keras_parameterized.run_all_keras_modes
class TextVectorizationPreprocessingTest(
    keras_parameterized.TestCase,
    preprocessing_test_utils.PreprocessingLayerTest):

  def test_normalization(self):
    input_array = np.array([["Earth", "wInD", "aNd", "firE"],
                            ["fire|", "an<>d", "{earth}", "michigan@%$"]])
    expected_output = np.array([[b"earth", b"wind", b"and", b"fire"],
                                [b"fire", b"and", b"earth", b"michigan"]])

    input_data = keras.Input(shape=(None,), dtype=dtypes.string)
    layer = get_layer_class()(
        max_tokens=None,
        standardize=text_vectorization.LOWER_AND_STRIP_PUNCTUATION,
        split=None,
        ngrams=None,
        output_mode=None)
    int_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_array)
    self.assertAllEqual(expected_output, output_dataset)

  def test_custom_normalization(self):
    input_array = np.array([["Earth", "wInD", "aNd", "firE"],
                            ["fire|", "an<>d", "{earth}", "michigan@%$"]])
    expected_output = np.array(
        [[b"earth", b"wind", b"and", b"fire"],
         [b"fire|", b"an<>d", b"{earth}", b"michigan@%$"]])

    custom_standardization = gen_string_ops.string_lower
    input_data = keras.Input(shape=(None,), dtype=dtypes.string)
    layer = get_layer_class()(
        max_tokens=None,
        standardize=custom_standardization,
        split=None,
        ngrams=None,
        output_mode=None)
    int_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_array)
    self.assertAllEqual(expected_output, output_dataset)

  def test_string_splitting(self):
    input_array = np.array([["earth wind and fire"],
                            ["\tfire\tand\nearth    michigan  "]])
    expected_output = [[b"earth", b"wind", b"and", b"fire"],
                       [b"fire", b"and", b"earth", b"michigan"]]

    input_data = keras.Input(shape=(1,), dtype=dtypes.string)
    layer = get_layer_class()(
        max_tokens=None,
        standardize=None,
        split=text_vectorization.SPLIT_ON_WHITESPACE,
        ngrams=None,
        output_mode=None)
    int_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_array)
    self.assertAllEqual(expected_output, output_dataset)

  def test_custom_string_splitting(self):
    input_array = np.array([["earth>wind>and fire"],
                            ["\tfire>and\nearth>michigan"]])
    expected_output = [[b"earth", b"wind", b"and fire"],
                       [b"\tfire", b"and\nearth", b"michigan"]]

    custom_split = lambda x: ragged_string_ops.string_split_v2(x, sep=">")
    input_data = keras.Input(shape=(1,), dtype=dtypes.string)
    layer = get_layer_class()(
        max_tokens=None,
        standardize=None,
        split=custom_split,
        ngrams=None,
        output_mode=None)
    int_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_array)
    self.assertAllEqual(expected_output, output_dataset)

  def test_single_ngram_value(self):
    input_array = np.array([["earth", "wind", "and", "fire"],
                            ["fire", "and", "earth", "michigan"]])
    # pyformat: disable
    expected_output = [[b"earth", b"wind", b"and", b"fire",
                        b"earth wind", b"wind and", b"and fire",
                        b"earth wind and", b"wind and fire"],
                       [b"fire", b"and", b"earth", b"michigan",
                        b"fire and", b"and earth", b"earth michigan",
                        b"fire and earth", b"and earth michigan"]]
    # pyformat: enable

    input_data = keras.Input(shape=(4,), dtype=dtypes.string)
    layer = get_layer_class()(
        max_tokens=None,
        standardize=None,
        split=None,
        ngrams=3,
        output_mode=None)
    int_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_array)
    self.assertAllEqual(expected_output, output_dataset)

  def test_multiple_ngram_values(self):
    input_array = np.array([["earth", "wind", "and", "fire"],
                            ["fire", "and", "earth", "michigan"]])
    # pyformat: disable
    expected_output = [[b"earth wind", b"wind and", b"and fire",
                        b"earth wind and", b"wind and fire"],
                       [b"fire and", b"and earth", b"earth michigan",
                        b"fire and earth", b"and earth michigan"]]
    # pyformat: enable

    input_data = keras.Input(shape=(4,), dtype=dtypes.string)
    layer = get_layer_class()(
        max_tokens=None,
        standardize=None,
        split=None,
        ngrams=(2, 3),
        output_mode=None)
    int_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_array)
    self.assertAllEqual(expected_output, output_dataset)

  def test_string_multiple_preprocessing_steps(self):
    input_array = np.array([["earth wInD and firE"],
                            ["\tfire\tand\nearth!!    michig@n  "]])
    expected_output = [[
        b"earth",
        b"wind",
        b"and",
        b"fire",
        b"earth wind",
        b"wind and",
        b"and fire",
    ],
                       [
                           b"fire",
                           b"and",
                           b"earth",
                           b"michign",
                           b"fire and",
                           b"and earth",
                           b"earth michign",
                       ]]

    input_data = keras.Input(shape=(1,), dtype=dtypes.string)
    layer = get_layer_class()(
        max_tokens=None,
        standardize=text_vectorization.LOWER_AND_STRIP_PUNCTUATION,
        split=text_vectorization.SPLIT_ON_WHITESPACE,
        ngrams=2,
        output_mode=None)
    int_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_array)
    self.assertAllEqual(expected_output, output_dataset)

  def test_string_splitting_with_non_1d_array_fails(self):
    input_data = keras.Input(shape=(None,), dtype=dtypes.string)
    layer = get_layer_class()(
        max_tokens=None,
        standardize=None,
        split=text_vectorization.SPLIT_ON_WHITESPACE,
        output_mode=None)
    with self.assertRaisesRegex(RuntimeError,
                                ".*tokenize strings, the first dimension.*"):
      _ = layer(input_data)

  def test_standardization_with_invalid_standardize_arg(self):
    input_data = keras.Input(shape=(1,), dtype=dtypes.string)
    layer = get_layer_class()()
    layer._standardize = "unsupported"
    with self.assertRaisesRegex(ValueError,
                                ".*is not a supported standardization.*"):
      _ = layer(input_data)

  def test_splitting_with_invalid_split_arg(self):
    input_data = keras.Input(shape=(1,), dtype=dtypes.string)
    layer = get_layer_class()()
    layer._split = "unsuppported"
    with self.assertRaisesRegex(ValueError, ".*is not a supported splitting.*"):
      _ = layer(input_data)


@keras_parameterized.run_all_keras_modes
class TextVectorizationOutputTest(
    keras_parameterized.TestCase,
    preprocessing_test_utils.PreprocessingLayerTest):

  def test_int_output(self):
    vocab_data = ["earth", "wind", "and", "fire"]
    input_array = np.array([["earth", "wind", "and", "fire"],
                            ["fire", "and", "earth", "michigan"]])
    expected_output = [[2, 3, 4, 5], [5, 4, 2, 1]]

    input_data = keras.Input(shape=(None,), dtype=dtypes.string)
    layer = get_layer_class()(
        max_tokens=None,
        standardize=None,
        split=None,
        output_mode=text_vectorization.INT)
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
    layer = get_layer_class()(
        max_tokens=5,
        standardize=None,
        split=None,
        output_mode=text_vectorization.INT)
    layer.set_vocabulary(vocab_data[0])
    layer.set_vocabulary(vocab_data[1], append=True)
    int_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_array)
    self.assertAllClose(expected_output, output_dataset)

  def test_int_output_densifies_with_zeros(self):
    vocab_data = ["earth", "wind", "and", "fire"]
    # Create an input array that has 5 elements in the first example and 4 in
    # the second. This should output a 2x5 tensor with a padding value in the
    # second example.
    input_array = np.array([["earth wind and also fire"],
                            ["fire and earth michigan"]])
    expected_output = [[2, 3, 4, 1, 5], [5, 4, 2, 1, 0]]

    # This test doesn't explicitly set an output shape, so the 2nd dimension
    # should stay 'None'.
    expected_output_shape = [None, None]

    # The input shape here is explicitly 1 because we're tokenizing.
    input_data = keras.Input(shape=(1,), dtype=dtypes.string)
    layer = get_layer_class()(
        max_tokens=None,
        standardize=None,
        split=text_vectorization.SPLIT_ON_WHITESPACE,
        output_mode=text_vectorization.INT)
    layer.set_vocabulary(vocab_data)
    int_data = layer(input_data)
    self.assertAllEqual(expected_output_shape, int_data.shape.as_list())

    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_array)
    self.assertAllEqual(expected_output, output_dataset)

  def test_int_output_densifies_with_zeros_and_pads(self):
    vocab_data = ["earth", "wind", "and", "fire"]
    # Create an input array that has 5 elements in the first example and 4 in
    # the second. This should output a 2x6 tensor with a padding value in the
    # second example, since output_sequence_length is set to 6.
    input_array = np.array([["earth wind and also fire"],
                            ["fire and earth michigan"]])
    expected_output = [[2, 3, 4, 1, 5, 0], [5, 4, 2, 1, 0, 0]]

    output_sequence_length = 6
    expected_output_shape = [None, output_sequence_length]

    # The input shape here is explicitly 1 because we're tokenizing.
    input_data = keras.Input(shape=(1,), dtype=dtypes.string)
    layer = get_layer_class()(
        max_tokens=None,
        standardize=None,
        split=text_vectorization.SPLIT_ON_WHITESPACE,
        output_mode=text_vectorization.INT,
        output_sequence_length=output_sequence_length)
    layer.set_vocabulary(vocab_data)
    int_data = layer(input_data)
    self.assertAllEqual(expected_output_shape, int_data.shape.as_list())

    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_array)
    self.assertAllEqual(expected_output, output_dataset)

  def test_int_output_densifies_with_zeros_and_strips(self):
    vocab_data = ["earth", "wind", "and", "fire"]
    # Create an input array that has 5 elements in the first example and 4 in
    # the second. This should output a 2x3 tensor with a padding value in the
    # second example, since output_sequence_length is set to 3.
    input_array = np.array([["earth wind and also fire"],
                            ["fire and earth michigan"]])
    expected_output = [[2, 3, 4], [5, 4, 2]]
    output_sequence_length = 3
    expected_output_shape = [None, output_sequence_length]

    # The input shape here is explicitly 1 because we're tokenizing.
    input_data = keras.Input(shape=(1,), dtype=dtypes.string)
    layer = get_layer_class()(
        max_tokens=None,
        standardize=None,
        split=text_vectorization.SPLIT_ON_WHITESPACE,
        output_mode=text_vectorization.INT,
        output_sequence_length=output_sequence_length)
    layer.set_vocabulary(vocab_data)
    int_data = layer(input_data)
    self.assertAllEqual(expected_output_shape, int_data.shape.as_list())

    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_array)
    self.assertAllEqual(expected_output, output_dataset)

  def test_int_output_dynamically_strips_and_pads(self):
    vocab_data = ["earth", "wind", "and", "fire"]
    # Create an input array that has 5 elements in the first example and 4 in
    # the second. This should output a 2x3 tensor with a padding value in the
    # second example, since output_sequence_length is set to 3.
    input_array = np.array([["earth wind and also fire"],
                            ["fire and earth michigan"]])
    expected_output = [[2, 3, 4], [5, 4, 2]]
    output_sequence_length = 3
    expected_output_shape = [None, output_sequence_length]

    # The input shape here is explicitly 1 because we're tokenizing.
    input_data = keras.Input(shape=(1,), dtype=dtypes.string)
    layer = get_layer_class()(
        max_tokens=None,
        standardize=None,
        split=text_vectorization.SPLIT_ON_WHITESPACE,
        output_mode=text_vectorization.INT,
        output_sequence_length=output_sequence_length)
    layer.set_vocabulary(vocab_data)
    int_data = layer(input_data)
    self.assertAllEqual(expected_output_shape, int_data.shape.as_list())

    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_array)
    self.assertAllEqual(expected_output, output_dataset)

    # Create an input array that has 1 element in the first example and 2 in
    # the second. This should output a 2x3 tensor with a padding value in the
    # second example, since output_sequence_length is set to 3.
    input_array_2 = np.array([["wind"], ["fire and"]])
    expected_output_2 = [[3, 0, 0], [5, 4, 0]]
    output_dataset = model.predict(input_array_2)
    self.assertAllEqual(expected_output_2, output_dataset)

  def test_binary_output_hard_maximum(self):
    vocab_data = ["earth", "wind", "and", "fire"]
    input_array = np.array([["earth", "wind", "and", "earth"],
                            ["ohio", "and", "earth", "michigan"]])

    # pyformat: disable
    expected_output = [[0, 1, 1, 1, 0, 0],
                       [1, 1, 0, 1, 0, 0]]
    # pyformat: enable
    max_tokens = 6
    expected_output_shape = [None, max_tokens]

    input_data = keras.Input(shape=(None,), dtype=dtypes.string)
    layer = get_layer_class()(
        max_tokens=max_tokens,
        standardize=None,
        split=None,
        output_mode=text_vectorization.BINARY,
        pad_to_max_tokens=True)
    layer.set_vocabulary(vocab_data)
    int_data = layer(input_data)
    self.assertAllEqual(expected_output_shape, int_data.shape.as_list())

    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_array)
    self.assertAllEqual(expected_output, output_dataset)

  def test_binary_output_soft_maximum(self):
    vocab_data = ["earth", "wind", "and", "fire"]
    input_array = np.array([["earth", "wind", "and", "earth"],
                            ["ohio", "and", "earth", "michigan"]])

    # pyformat: disable
    expected_output = [[0, 1, 1, 1, 0],
                       [1, 1, 0, 1, 0]]
    # pyformat: enable
    max_tokens = 5
    expected_output_shape = [None, max_tokens]

    input_data = keras.Input(shape=(None,), dtype=dtypes.string)
    layer = get_layer_class()(
        max_tokens=10,
        standardize=None,
        split=None,
        output_mode=text_vectorization.BINARY,
        pad_to_max_tokens=False)
    layer.set_vocabulary(vocab_data)
    int_data = layer(input_data)
    self.assertAllEqual(expected_output_shape, int_data.shape.as_list())

    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_array)
    self.assertAllEqual(expected_output, output_dataset)

  def test_bag_output_hard_maximum_set_vocabulary_after_build(self):
    vocab_data = ["earth", "wind", "and", "fire"]
    input_array = np.array([["earth", "wind", "and", "earth"],
                            ["ohio", "and", "earth", "michigan"]])

    # pyformat: disable
    expected_output = [[0, 1, 1, 1, 0],
                       [1, 1, 0, 1, 0]]
    # pyformat: enable
    max_tokens = 5
    expected_output_shape = [None, max_tokens]

    input_data = keras.Input(shape=(None,), dtype=dtypes.string)
    layer = get_layer_class()(
        max_tokens=max_tokens,
        standardize=None,
        split=None,
        output_mode=text_vectorization.BINARY,
        pad_to_max_tokens=True)
    int_data = layer(input_data)
    layer.set_vocabulary(vocab_data)
    self.assertAllEqual(expected_output_shape, int_data.shape.as_list())

    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_array)
    self.assertAllEqual(expected_output, output_dataset)

  def test_bag_output_hard_maximum_adapt_after_build(self):
    vocab_data = np.array([
        "earth", "earth", "earth", "earth", "wind", "wind", "wind", "and",
        "and", "fire"
    ])
    input_array = np.array([["earth", "wind", "and", "earth"],
                            ["ohio", "and", "earth", "michigan"]])

    # pyformat: disable
    expected_output = [[0, 1, 1, 1, 0],
                       [1, 1, 0, 1, 0]]
    # pyformat: enable
    max_tokens = 5
    expected_output_shape = [None, max_tokens]

    input_data = keras.Input(shape=(None,), dtype=dtypes.string)
    layer = get_layer_class()(
        max_tokens=max_tokens,
        standardize=None,
        split=None,
        output_mode=text_vectorization.BINARY,
        pad_to_max_tokens=True)
    int_data = layer(input_data)
    layer.adapt(vocab_data)
    self.assertAllEqual(expected_output_shape, int_data.shape.as_list())

    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_array)
    self.assertAllEqual(expected_output, output_dataset)

  def test_bag_output_hard_maximum_set_state_variables_after_build(self):
    state_variables = {
        text_vectorization._VOCAB_NAME: ["earth", "wind", "and", "fire"]
    }
    input_array = np.array([["earth", "wind", "and", "earth"],
                            ["ohio", "and", "earth", "michigan"]])

    # pyformat: disable
    expected_output = [[0, 1, 1, 1, 0],
                       [1, 1, 0, 1, 0]]
    # pyformat: enable
    max_tokens = 5
    expected_output_shape = [None, max_tokens]

    input_data = keras.Input(shape=(None,), dtype=dtypes.string)
    layer = get_layer_class()(
        max_tokens=max_tokens,
        standardize=None,
        split=None,
        output_mode=text_vectorization.BINARY,
        pad_to_max_tokens=True)
    int_data = layer(input_data)
    layer._set_state_variables(state_variables)
    self.assertAllEqual(expected_output_shape, int_data.shape.as_list())

    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_array)
    self.assertAllEqual(expected_output, output_dataset)

  def test_bag_output_soft_maximum_set_state_after_build(self):
    vocab_data = ["earth", "wind", "and", "fire"]
    input_array = np.array([["earth", "wind", "and", "earth"],
                            ["ohio", "and", "earth", "michigan"]])

    # pyformat: disable
    expected_output = [[0, 1, 1, 1, 0],
                       [1, 1, 0, 1, 0]]
    # pyformat: enable
    max_tokens = 5
    expected_output_shape = [None, max_tokens]

    input_data = keras.Input(shape=(None,), dtype=dtypes.string)
    layer = get_layer_class()(
        max_tokens=10,
        standardize=None,
        split=None,
        output_mode=text_vectorization.BINARY,
        pad_to_max_tokens=False)
    layer.build(input_data.shape)
    layer.set_vocabulary(vocab_data)
    int_data = layer(input_data)
    self.assertAllEqual(expected_output_shape, int_data.shape.as_list())

    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_array)
    self.assertAllEqual(expected_output, output_dataset)

  def test_bag_output_soft_maximum_set_vocabulary_after_call_fails(self):
    vocab_data = ["earth", "wind", "and", "fire"]

    input_data = keras.Input(shape=(None,), dtype=dtypes.string)
    layer = get_layer_class()(
        max_tokens=None,
        standardize=None,
        split=None,
        output_mode=text_vectorization.BINARY,
        pad_to_max_tokens=False)
    _ = layer(input_data)
    with self.assertRaisesRegex(RuntimeError, "vocabulary cannot be changed"):
      layer.set_vocabulary(vocab_data)

  def test_bag_output_soft_maximum_adapt_after_call_fails(self):
    vocab_data = np.array([
        "earth", "earth", "earth", "earth", "wind", "wind", "wind", "and",
        "and", "fire"
    ])

    input_data = keras.Input(shape=(None,), dtype=dtypes.string)
    layer = get_layer_class()(
        max_tokens=None,
        standardize=None,
        split=None,
        output_mode=text_vectorization.BINARY,
        pad_to_max_tokens=False)
    _ = layer(input_data)
    with self.assertRaisesRegex(RuntimeError, "vocabulary cannot be changed"):
      layer.adapt(vocab_data)

  def test_bag_output_soft_maximum_set_state_variables_after_call_fails(self):
    state_variables = {
        text_vectorization._VOCAB_NAME: ["earth", "wind", "and", "fire"]
    }

    input_data = keras.Input(shape=(None,), dtype=dtypes.string)
    layer = get_layer_class()(
        max_tokens=None,
        standardize=None,
        split=None,
        output_mode=text_vectorization.BINARY,
        pad_to_max_tokens=False)
    _ = layer(input_data)
    with self.assertRaisesRegex(RuntimeError, "vocabulary cannot be changed"):
      layer._set_state_variables(state_variables)

  def test_count_output_hard_maximum(self):
    vocab_data = ["earth", "wind", "and", "fire"]
    input_array = np.array([["earth", "wind", "and", "earth"],
                            ["ohio", "and", "earth", "michigan"]])

    # pyformat: disable
    expected_output = [[0, 2, 1, 1, 0, 0],
                       [2, 1, 0, 1, 0, 0]]
    # pyformat: enable
    max_tokens = 6
    expected_output_shape = [None, max_tokens]

    input_data = keras.Input(shape=(None,), dtype=dtypes.string)
    layer = get_layer_class()(
        max_tokens=6,
        standardize=None,
        split=None,
        output_mode=text_vectorization.COUNT)
    layer.set_vocabulary(vocab_data)
    int_data = layer(input_data)
    self.assertAllEqual(expected_output_shape, int_data.shape.as_list())

    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_array)
    self.assertAllEqual(expected_output, output_dataset)

  def test_count_output_soft_maximum(self):
    vocab_data = ["earth", "wind", "and", "fire"]
    input_array = np.array([["earth", "wind", "and", "earth"],
                            ["ohio", "and", "earth", "michigan"]])

    # pyformat: disable
    expected_output = [[0, 2, 1, 1, 0],
                       [2, 1, 0, 1, 0]]
    # pyformat: enable
    max_tokens = 5
    expected_output_shape = [None, max_tokens]

    input_data = keras.Input(shape=(None,), dtype=dtypes.string)
    layer = get_layer_class()(
        max_tokens=10,
        standardize=None,
        split=None,
        output_mode=text_vectorization.COUNT,
        pad_to_max_tokens=False)
    layer.set_vocabulary(vocab_data)
    int_data = layer(input_data)
    self.assertAllEqual(expected_output_shape, int_data.shape.as_list())

    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_array)
    self.assertAllEqual(expected_output, output_dataset)

  def test_tfidf_output_hard_maximum(self):
    vocab_data = ["earth", "wind", "and", "fire"]
    tfidf_data = [.5, .25, .2, .125]
    input_array = np.array([["earth", "wind", "and", "earth"],
                            ["ohio", "fire", "earth", "michigan"]])

    # pyformat: disable
    # pylint: disable=bad-whitespace
    expected_output = [[ 0,  1, .25, .2,    0, 0],
                       [.1, .5,   0,  0, .125, 0]]
    # pylint: enable=bad-whitespace
    # pyformat: enable
    max_tokens = 6
    expected_output_shape = [None, max_tokens]

    input_data = keras.Input(shape=(None,), dtype=dtypes.string)
    layer = get_layer_class()(
        max_tokens=6,
        standardize=None,
        split=None,
        output_mode=text_vectorization.TFIDF,
        pad_to_max_tokens=True)
    layer.set_vocabulary(vocab_data, df_data=tfidf_data, oov_df_value=.05)
    int_data = layer(input_data)
    self.assertAllEqual(expected_output_shape, int_data.shape.as_list())

    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_array)
    self.assertAllClose(expected_output, output_dataset)

  def test_tfidf_output_soft_maximum(self):
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
    max_tokens = 5
    expected_output_shape = [None, max_tokens]

    input_data = keras.Input(shape=(None,), dtype=dtypes.string)
    layer = get_layer_class()(
        max_tokens=10,
        standardize=None,
        split=None,
        output_mode=text_vectorization.TFIDF,
        pad_to_max_tokens=False)
    layer.set_vocabulary(vocab_data, df_data=tfidf_data, oov_df_value=.05)
    int_data = layer(input_data)
    self.assertAllEqual(expected_output_shape, int_data.shape.as_list())

    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_array)
    self.assertAllClose(expected_output, output_dataset)

  def test_tfidf_appending(self):
    vocab_data = [["earth", "wind"], ["and", "fire"]]
    tfidf_data = [[.5, .25], [.2, .125]]
    input_array = np.array([["earth", "wind", "and", "earth"],
                            ["ohio", "fire", "earth", "michigan"]])

    # pyformat: disable
    # pylint: disable=bad-whitespace
    expected_output = [[ 0,  1, .25, .2,    0],
                       [.1, .5,   0,  0, .125]]
    # pylint: enable=bad-whitespace
    # pyformat: enable

    input_data = keras.Input(shape=(None,), dtype=dtypes.string)
    layer = get_layer_class()(
        max_tokens=5,
        standardize=None,
        split=None,
        output_mode=text_vectorization.TFIDF)
    layer.set_vocabulary(vocab_data[0], df_data=tfidf_data[0], oov_df_value=.05)
    layer.set_vocabulary(vocab_data[1], df_data=tfidf_data[1], append=True)
    int_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_array)
    self.assertAllClose(expected_output, output_dataset)

  def test_tfidf_appending_with_oov_replacement(self):
    vocab_data = [["earth", "wind"], ["and", "fire"]]
    tfidf_data = [[.5, .25], [.2, .125]]
    input_array = np.array([["earth", "wind", "and", "earth"],
                            ["ohio", "fire", "earth", "michigan"]])

    # pyformat: disable
    # pylint: disable=bad-whitespace
    expected_output = [[ 0,  1, .25, .2,    0],
                       [1.5, .5,   0,  0, .125]]
    # pylint: enable=bad-whitespace
    # pyformat: enable

    input_data = keras.Input(shape=(None,), dtype=dtypes.string)
    layer = get_layer_class()(
        max_tokens=5,
        standardize=None,
        split=None,
        output_mode=text_vectorization.TFIDF)
    layer.set_vocabulary(vocab_data[0], df_data=tfidf_data[0], oov_df_value=.05)
    # Note that here we've replaced the OOV vaue.
    layer.set_vocabulary(
        vocab_data[1], df_data=tfidf_data[1], oov_df_value=.75, append=True)
    int_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_array)
    self.assertAllClose(expected_output, output_dataset)


@keras_parameterized.run_all_keras_modes
class TextVectorizationModelBuildingTest(
    keras_parameterized.TestCase,
    preprocessing_test_utils.PreprocessingLayerTest):

  @parameterized.named_parameters(
      {
          "testcase_name": "count_hard_max",
          "pad_to_max_tokens": True,
          "output_mode": text_vectorization.COUNT
      }, {
          "testcase_name": "count_soft_max",
          "pad_to_max_tokens": False,
          "output_mode": text_vectorization.COUNT
      }, {
          "testcase_name": "binary_hard_max",
          "pad_to_max_tokens": True,
          "output_mode": text_vectorization.BINARY
      }, {
          "testcase_name": "binary_soft_max",
          "pad_to_max_tokens": False,
          "output_mode": text_vectorization.BINARY
      }, {
          "testcase_name": "tfidf_hard_max",
          "pad_to_max_tokens": True,
          "output_mode": text_vectorization.TFIDF
      }, {
          "testcase_name": "tfidf_soft_max",
          "pad_to_max_tokens": False,
          "output_mode": text_vectorization.TFIDF
      })
  def test_end_to_end_bagged_modeling(self, output_mode, pad_to_max_tokens):
    vocab_data = ["earth", "wind", "and", "fire"]
    tfidf_data = [.5, .25, .2, .125]
    input_array = np.array([["earth", "wind", "and", "earth"],
                            ["ohio", "and", "earth", "michigan"]])

    input_data = keras.Input(shape=(None,), dtype=dtypes.string)
    layer = get_layer_class()(
        max_tokens=10,
        standardize=None,
        split=None,
        output_mode=output_mode,
        pad_to_max_tokens=pad_to_max_tokens)
    if output_mode == text_vectorization.TFIDF:
      layer.set_vocabulary(vocab_data, df_data=tfidf_data, oov_df_value=.05)
    else:
      layer.set_vocabulary(vocab_data)

    int_data = layer(input_data)
    float_data = backend.cast(int_data, dtype="float32")
    output_data = core.Dense(64)(float_data)
    model = keras.Model(inputs=input_data, outputs=output_data)
    _ = model.predict(input_array)

  def test_end_to_end_vocab_modeling(self):
    vocab_data = ["earth", "wind", "and", "fire"]
    input_array = np.array([["earth wind and also fire"],
                            ["fire and earth michigan"]])
    output_sequence_length = 6
    max_tokens = 5

    # The input shape here is explicitly 1 because we're tokenizing.
    input_data = keras.Input(shape=(1,), dtype=dtypes.string)
    layer = get_layer_class()(
        max_tokens=None,
        standardize=None,
        split=text_vectorization.SPLIT_ON_WHITESPACE,
        output_mode=text_vectorization.INT,
        output_sequence_length=output_sequence_length)
    layer.set_vocabulary(vocab_data)
    int_data = layer(input_data)
    embedded_data = embeddings.Embedding(
        input_dim=max_tokens + 1, output_dim=32)(
            int_data)
    output_data = convolutional.Conv1D(
        250, 3, padding="valid", activation="relu", strides=1)(
            embedded_data)

    model = keras.Model(inputs=input_data, outputs=output_data)
    _ = model.predict(input_array)


@keras_parameterized.run_all_keras_modes(always_skip_eager=True)
class TextVectorizationSaveableTest(
    keras_parameterized.TestCase,
    preprocessing_test_utils.PreprocessingLayerTest):

  def test_ops_are_not_added_with_multiple_saves(self):
    vocab_data = ["earth", "wind", "and", "fire"]

    input_data = keras.Input(shape=(None,), dtype=dtypes.string)
    layer = get_layer_class()(
        max_tokens=10,
        standardize=None,
        split=None,
        output_mode=text_vectorization.COUNT,
        pad_to_max_tokens=False)
    layer.set_vocabulary(vocab_data)
    int_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=int_data)
    weights = model.get_weights()
    model.set_weights(weights)
    keras.backend.get_session().graph.finalize()
    weights = model.get_weights()
    model.set_weights(weights)


@keras_parameterized.run_all_keras_modes
class TextVectorizationErrorTest(keras_parameterized.TestCase,
                                 preprocessing_test_utils.PreprocessingLayerTest
                                ):

  def test_too_long_vocab_fails_in_single_setting(self):
    vocab_data = ["earth", "wind", "and", "fire"]

    layer = get_layer_class()(
        max_tokens=4,
        standardize=None,
        split=None,
        output_mode=text_vectorization.INT)
    with self.assertRaisesRegex(ValueError,
                                "vocabulary larger than the maximum vocab.*"):
      layer.set_vocabulary(vocab_data)

  def test_too_long_vocab_fails_in_multiple_settings(self):
    vocab_data = [["earth", "wind"], ["and", "fire"]]

    layer = get_layer_class()(
        max_tokens=4,
        standardize=None,
        split=None,
        output_mode=text_vectorization.INT)

    # The first time we call set_vocabulary, we're under the max_tokens limit
    # so it should be fine.
    layer.set_vocabulary(vocab_data[0])
    with self.assertRaisesRegex(ValueError,
                                "vocabulary larger than the maximum vocab.*"):
      layer.set_vocabulary(vocab_data[1], append=True)

  def test_setting_vocab_without_tfidf_data_fails_in_tfidf_mode(self):
    vocab_data = ["earth", "wind", "and", "fire"]

    layer = get_layer_class()(
        max_tokens=5,
        standardize=None,
        split=None,
        output_mode=text_vectorization.TFIDF)
    with self.assertRaisesRegex(ValueError,
                                "df_data must be set if output_mode is TFIDF"):
      layer.set_vocabulary(vocab_data)

  def test_tfidf_data_length_mismatch_fails(self):
    vocab_data = ["earth", "wind", "and", "fire"]
    df_data = [1, 2, 3]
    layer = get_layer_class()(
        max_tokens=5,
        standardize=None,
        split=None,
        output_mode=text_vectorization.TFIDF)
    with self.assertRaisesRegex(ValueError,
                                "df_data must be the same length as vocab.*"):
      layer.set_vocabulary(vocab_data, df_data)

  def test_tfidf_set_vocab_with_no_oov_fails(self):
    vocab_data = ["earth", "wind", "and", "fire"]
    df_data = [1, 2, 3, 4]
    layer = get_layer_class()(
        max_tokens=5,
        standardize=None,
        split=None,
        output_mode=text_vectorization.TFIDF)
    with self.assertRaisesRegex(ValueError,
                                "You must pass an oov_df_value.*"):
      layer.set_vocabulary(vocab_data, df_data)

  def test_tfidf_set_vocab_with_no_oov_fails_with_append_set(self):
    vocab_data = ["earth", "wind", "and", "fire"]
    df_data = [1, 2, 3, 4]
    layer = get_layer_class()(
        max_tokens=5,
        standardize=None,
        split=None,
        output_mode=text_vectorization.TFIDF)
    with self.assertRaisesRegex(ValueError,
                                "You must pass an oov_df_value.*"):
      layer.set_vocabulary(vocab_data, df_data, append=True)

  def test_set_tfidf_in_non_tfidf_fails(self):
    vocab_data = ["earth", "wind", "and", "fire"]
    df_data = [1, 2, 3, 4]
    layer = get_layer_class()(
        max_tokens=5,
        standardize=None,
        split=None,
        output_mode=text_vectorization.BINARY)
    with self.assertRaisesRegex(ValueError,
                                ".*df_data should only be set if.*"):
      layer.set_vocabulary(vocab_data, df_data)

  def test_zero_max_tokens_fails(self):
    with self.assertRaisesRegex(ValueError, ".*max_tokens.*"):
      _ = get_layer_class()(max_tokens=0)

  def test_non_string_dtype_fails(self):
    with self.assertRaisesRegex(ValueError, ".*dtype of string.*"):
      _ = get_layer_class()(dtype=dtypes.int64)

  def test_unknown_standardize_arg_fails(self):
    with self.assertRaisesRegex(ValueError,
                                ".*standardize arg.*unsupported_value.*"):
      _ = get_layer_class()(standardize="unsupported_value")

  def test_unknown_split_arg_fails(self):
    with self.assertRaisesRegex(ValueError, ".*split arg.*unsupported_value.*"):
      _ = get_layer_class()(split="unsupported_value")

  def test_unknown_output_mode_arg_fails(self):
    with self.assertRaisesRegex(ValueError,
                                ".*output_mode arg.*unsupported_value.*"):
      _ = get_layer_class()(output_mode="unsupported_value")

  def test_unknown_ngrams_arg_fails(self):
    with self.assertRaisesRegex(ValueError, ".*ngrams.*unsupported_value.*"):
      _ = get_layer_class()(ngrams="unsupported_value")

  def test_float_ngrams_arg_fails(self):
    with self.assertRaisesRegex(ValueError, ".*ngrams.*2.9.*"):
      _ = get_layer_class()(ngrams=2.9)

  def test_float_tuple_ngrams_arg_fails(self):
    with self.assertRaisesRegex(ValueError, ".*ngrams.*(1.3, 2.9).*"):
      _ = get_layer_class()(ngrams=(1.3, 2.9))

  def test_non_int_output_sequence_length_dtype_fails(self):
    with self.assertRaisesRegex(ValueError, ".*output_sequence_length.*2.0.*"):
      _ = get_layer_class()(output_mode="int", output_sequence_length=2.0)

  def test_non_none_output_sequence_length_fails_if_output_type_not_int(self):
    with self.assertRaisesRegex(ValueError,
                                ".*`output_sequence_length` must not be set.*"):
      _ = get_layer_class()(output_mode="count", output_sequence_length=2)

# Custom functions for the custom callable serialization test. Declared here
# to avoid multiple registrations from run_all_keras_modes().
@generic_utils.register_keras_serializable(package="Test")
def custom_standardize_fn(x):
  return gen_string_ops.string_lower(x)


@generic_utils.register_keras_serializable(package="Test")
def custom_split_fn(x):
  return ragged_string_ops.string_split_v2(x, sep=">")


@keras_parameterized.run_all_keras_modes
class TextVectorizationSavingTest(
    keras_parameterized.TestCase,
    preprocessing_test_utils.PreprocessingLayerTest):

  def test_saving_errors(self):
    vocab_data = ["earth", "wind", "and", "fire"]

    # Build and validate a golden model.
    input_data = keras.Input(shape=(None,), dtype=dtypes.string)
    layer = get_layer_class()(
        max_tokens=None,
        standardize=None,
        split=None,
        output_mode=text_vectorization.INT)
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
    layer = get_layer_class()(
        max_tokens=None,
        standardize=None,
        split=None,
        output_mode=text_vectorization.INT)
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

  def test_serialization_with_custom_callables(self):
    input_array = np.array([["earth>wind>and Fire"],
                            ["\tfire>And\nearth>michigan"]])
    expected_output = [[b"earth", b"wind", b"and fire"],
                       [b"\tfire", b"and\nearth", b"michigan"]]

    input_data = keras.Input(shape=(1,), dtype=dtypes.string)
    layer = get_layer_class()(
        max_tokens=None,
        standardize=custom_standardize_fn,
        split=custom_split_fn,
        ngrams=None,
        output_mode=None)
    int_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_array)
    self.assertAllEqual(expected_output, output_dataset)

    serialized_model_data = model.get_config()
    with CustomObjectScope({"TextVectorization": get_layer_class()}):
      new_model = keras.Model.from_config(serialized_model_data)
    new_output_dataset = new_model.predict(input_array)
    self.assertAllEqual(expected_output, new_output_dataset)

  def DISABLED_test_vocabulary_persistence_across_saving(self):
    vocab_data = ["earth", "wind", "and", "fire"]
    input_array = np.array([["earth", "wind", "and", "fire"],
                            ["fire", "and", "earth", "michigan"]])
    expected_output = [[2, 3, 4, 5], [5, 4, 2, 1]]

    # Build and validate a golden model.
    input_data = keras.Input(shape=(None,), dtype=dtypes.string)
    layer = get_layer_class()(
        max_tokens=None,
        standardize=None,
        split=None,
        output_mode=text_vectorization.INT)
    layer.set_vocabulary(vocab_data)
    int_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_array)
    self.assertAllEqual(output_dataset, expected_output)

    # Save the model to disk.
    output_path = os.path.join(self.get_temp_dir(), "tf_keras_saved_model")
    model.save(output_path, save_format="tf")
    loaded_model = saving.load_from_saved_model(
        output_path, custom_objects={"TextVectorization": get_layer_class()})

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
        output_mode=text_vectorization.TFIDF)
    layer.set_vocabulary(vocab_data, df_data=tfidf_data, oov_df_value=.05)

    int_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_array)
    self.assertAllClose(output_dataset, expected_output)

    # Save the model to disk.
    output_path = os.path.join(self.get_temp_dir(), "tf_keras_saved_model")
    model.save(output_path, save_format="tf")
    loaded_model = saving.load_from_saved_model(
        output_path, custom_objects={"TextVectorization": get_layer_class()})

    # Ensure that the loaded model is unique (so that the save/load is real)
    self.assertIsNot(model, loaded_model)

    # Validate correctness of the new model.
    new_output_dataset = loaded_model.predict(input_array)
    self.assertAllClose(new_output_dataset, expected_output)


@keras_parameterized.run_all_keras_modes
class TextVectorizationCombinerTest(
    keras_parameterized.TestCase,
    preprocessing_test_utils.PreprocessingLayerTest):

  def compare_text_accumulators(self, a, b, msg=None):
    if a is None or b is None:
      self.assertAllEqual(a, b, msg=msg)

    self.assertAllEqual(a.count_dict, b.count_dict, msg=msg)
    self.assertAllEqual(a.metadata, b.metadata, msg=msg)

    if a.per_doc_count_dict is not None:

      def per_doc_counts(accumulator):
        count_values = [
            count_dict["count"]
            for count_dict in accumulator.per_doc_count_dict.values()
        ]
        return dict(zip(accumulator.per_doc_count_dict.keys(), count_values))

      self.assertAllEqual(per_doc_counts(a), per_doc_counts(b), msg=msg)

  compare_accumulators = compare_text_accumulators

  def update_accumulator(self, accumulator, data):
    accumulator.count_dict.update(dict(zip(data["vocab"], data["counts"])))
    accumulator.metadata[0] = data["num_documents"]

    if "document_counts" in data:
      create_dict = lambda x: {"count": x, "last_doc_id": -1}
      idf_count_dicts = [
          create_dict(count) for count in data["document_counts"]
      ]
      idf_dict = dict(zip(data["vocab"], idf_count_dicts))

      accumulator.per_doc_count_dict.update(idf_dict)

    return accumulator

  def test_combiner_api_compatibility_int_mode(self):
    data = np.array([["earth", "wind", "and", "fire"],
                     ["earth", "wind", "and", "michigan"]])
    combiner = text_vectorization._TextVectorizationCombiner(compute_idf=False)
    expected_accumulator_output = {
        "vocab": np.array(["and", "earth", "wind", "fire", "michigan"]),
        "counts": np.array([2, 2, 2, 1, 1]),
        "num_documents": np.array(2),
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

  def test_combiner_api_compatibility_tfidf_mode(self):
    data = np.array([["earth", "wind", "and", "fire"],
                     ["earth", "wind", "and", "michigan"]])
    combiner = text_vectorization._TextVectorizationCombiner(compute_idf=True)
    expected_extract_output = {
        "vocab": np.array(["wind", "earth", "and", "michigan", "fire"]),
        "idf": np.array([0.510826, 0.510826, 0.510826, 0.693147, 0.693147]),
        "oov_idf": np.array([1.098612])
    }
    expected_accumulator_output = {
        "vocab": np.array(["wind", "earth", "and", "michigan", "fire"]),
        "counts": np.array([2, 2, 2, 1, 1]),
        "document_counts": np.array([2, 2, 2, 1, 1]),
        "num_documents": np.array(2),
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
              "document_counts": np.array([3, 2, 1, 1]),
              "num_documents": np.array(4),
          },
          "expected_extract_output": {
              "vocab": np.array(["wind", "fire", "earth"]),
              "idf": np.array([0.693147, 0.847298, 1.098612]),
              "oov_idf": np.array([1.609438]),
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
              "document_counts": np.array([3, 2, 1, 1]),
              "num_documents": np.array(4),
          },
          "expected_extract_output": {
              "vocab": np.array(["wind", "fire", "earth", "and"]),
              "idf": np.array([0.693147, 0.847298, 1.098612, 1.098612]),
              "oov_idf": np.array([1.609438]),
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
              "document_counts": np.array([3, 2, 1, 1]),
              "num_documents": np.array(4),
          },
          "expected_extract_output": {
              "vocab": np.array(["wind", "fire", "earth", "and"]),
              "idf": np.array([0.693147, 0.847298, 1.098612, 1.098612]),
              "oov_idf": np.array([1.609438]),
          },
      },
      {
          "testcase_name": "single_element_per_row",
          "data": np.array([["earth"], ["wind"], ["fire"], ["wind"], ["and"]]),
          "vocab_size": 3,
          "expected_accumulator_output": {
              "vocab": np.array(["wind", "and", "earth", "fire"]),
              "counts": np.array([2, 1, 1, 1]),
              "document_counts": np.array([2, 1, 1, 1]),
              "num_documents": np.array(5),
          },
          "expected_extract_output": {
              "vocab": np.array(["wind", "fire", "earth"]),
              "idf": np.array([0.980829, 1.252763, 1.252763]),
              "oov_idf": np.array([1.791759]),
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
              "document_counts": np.array([2, 1, 1, 1, 1]),
              "num_documents": np.array(5),
          },
          "expected_extract_output": {
              "vocab": np.array(["wind", "fire", "earth"]),
              "idf": np.array([0.980829, 1.252763, 1.252763]),
              "oov_idf": np.array([1.791759]),
          },
      })
  def test_combiner_computation(self,
                                data,
                                vocab_size,
                                expected_accumulator_output,
                                expected_extract_output,
                                compute_idf=True):
    combiner = text_vectorization._TextVectorizationCombiner(
        vocab_size=vocab_size, compute_idf=compute_idf)
    expected_accumulator = combiner._create_accumulator()
    expected_accumulator = self.update_accumulator(expected_accumulator,
                                                   expected_accumulator_output)
    self.validate_accumulator_computation(combiner, data, expected_accumulator)
    self.validate_accumulator_extract(combiner, data, expected_extract_output)



if __name__ == "__main__":
  test.main()
