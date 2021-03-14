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
import string

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
from tensorflow.python.keras.layers.preprocessing import index_lookup
from tensorflow.python.keras.layers.preprocessing import index_lookup_v1
from tensorflow.python.keras.layers.preprocessing import preprocessing_test_utils
from tensorflow.python.keras.saving import save
from tensorflow.python.keras.utils.generic_utils import CustomObjectScope
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import gfile
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
              "num_oov_indices": 1,
              "mask_token": "",
              "oov_token": "[OOV]",
              "dtype": dtypes.string,
          },
          "expected_output": [[2], [3], [4], [5], [5], [4], [2], [1]],
          "input_dtype":
              dtypes.string
      },
      {
          "testcase_name":
              "test_inverse_strings_soft_vocab_cap",
          # Create an array where 'earth' is the most frequent term, followed by
          # 'wind', then 'and', then 'fire'. This ensures that the vocab
          # accumulator is sorting by frequency.
          "vocab_data":
              np.array([["fire"], ["earth"], ["earth"], ["earth"], ["earth"],
                        ["wind"], ["wind"], ["wind"], ["and"], ["and"]]),
          "input_data":
              np.array([[2], [3], [4], [1], [1], [4], [2], [5]]),
          "kwargs": {
              "max_tokens": None,
              "num_oov_indices": 1,
              "mask_token": "",
              "oov_token": "[OOV]",
              "dtype": dtypes.string,
              "invert": True
          },
          "expected_output":
              np.array([[b"earth"], [b"wind"], [b"and"], [b"[OOV]"], [b"[OOV]"],
                        [b"and"], [b"earth"], [b"fire"]]),
          "input_dtype":
              dtypes.int64
      },
      {
          "testcase_name":
              "test_strings_with_special_tokens",
          # Mask and oov values in the vocab data should be dropped, and mapped
          # to 0 and 1 respectively when calling the layer.
          "vocab_data":
              np.array([["fire"], ["earth"], ["earth"], ["earth"], ["earth"],
                        [""], [""], [""], ["[OOV]"], ["[OOV]"], ["[OOV]"],
                        ["wind"], ["wind"], ["wind"], ["and"], ["and"]]),
          "input_data":
              np.array([["earth"], [""], ["wind"], ["[OOV]"], ["and"], [""],
                        ["fire"], ["and"], ["[OOV]"], ["michigan"]]),
          "kwargs": {
              "max_tokens": None,
              "num_oov_indices": 1,
              "mask_token": "",
              "oov_token": "[OOV]",
              "dtype": dtypes.string,
          },
          "expected_output": [[2], [0], [3], [1], [4], [0], [5], [4], [1], [1]],
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
              "num_oov_indices": 1,
              "mask_token": 0,
              "oov_token": -1,
              "dtype": dtypes.int64,
          },
          "expected_output": [[2], [3], [4], [5], [5], [4], [2], [1]],
          "input_dtype":
              dtypes.int64
      },
      {
          "testcase_name":
              "test_ints_with_special_tokens",
          # Mask and oov values in the vocab data should be dropped, and mapped
          # to 0 and 1 respectively when calling the layer.
          "vocab_data":
              np.array([[42], [1138], [1138], [1138], [1138], [0], [0], [0],
                        [-1], [-1], [-1], [1729], [1729], [1729], [725], [725]],
                       dtype=np.int64),
          "input_data":
              np.array([[1138], [0], [1729], [-1], [725], [0], [42], [725],
                        [-1], [4]],
                       dtype=np.int64),
          "kwargs": {
              "max_tokens": None,
              "num_oov_indices": 1,
              "mask_token": 0,
              "oov_token": -1,
              "dtype": dtypes.int64,
          },
          "expected_output": [[2], [0], [3], [1], [4], [0], [5], [4], [1], [1]],
          "input_dtype":
              dtypes.int64
      },
      {
          "testcase_name":
              "test_strings_hard_vocab_cap",
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
              "max_tokens": 5,
              "num_oov_indices": 1,
              "mask_token": "",
              "oov_token": "[OOV]",
              "dtype": dtypes.string,
          },
          "expected_output": [[2], [3], [4], [1], [1], [4], [2], [1]],
          "input_dtype":
              dtypes.string
      },
      {
          "testcase_name":
              "test_inverse_strings_hard_vocab_cap",
          # Create an array where 'earth' is the most frequent term, followed by
          # 'wind', then 'and', then 'fire'. This ensures that the vocab
          # accumulator is sorting by frequency.
          "vocab_data":
              np.array([["fire"], ["earth"], ["earth"], ["earth"], ["earth"],
                        ["wind"], ["wind"], ["wind"], ["and"], ["and"]]),
          "input_data":
              np.array([[2], [3], [4], [1], [1], [4], [2], [5]]),
          "kwargs": {
              "max_tokens": 5,
              "num_oov_indices": 1,
              "mask_token": "",
              "oov_token": "[OOV]",
              "dtype": dtypes.string,
              "invert": True
          },
          "expected_output":
              np.array([[b"earth"], [b"wind"], [b"and"], [b"[OOV]"], [b"[OOV]"],
                        [b"and"], [b"earth"], [b"[OOV]"]]),
          "input_dtype":
              dtypes.int64
      },
      {
          "testcase_name":
              "test_ints_hard_vocab_cap",
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
              "max_tokens": 5,
              "num_oov_indices": 1,
              "mask_token": 0,
              "oov_token": -1,
              "dtype": dtypes.int64,
          },
          "expected_output": [[2], [3], [4], [1], [1], [4], [2], [1]],
          "input_dtype":
              dtypes.int64
      },
      {
          "testcase_name":
              "test_ints_tf_idf_output",
          "vocab_data":
              np.array([[42], [1138], [1138], [1138], [1138], [1729], [1729],
                        [1729], [725], [725]]),
          "input_data":
              np.array([[1138], [1729], [725], [42], [42], [725], [1138], [4]]),
          "kwargs": {
              "max_tokens": 5,
              "num_oov_indices": 1,
              "mask_token": 0,
              "oov_token": -1,
              "output_mode": index_lookup.TFIDF,
              "dtype": dtypes.int64,
          },
          "expected_output": [[0, 1.098612, 0, 0, 0], [0, 0, 1.252763, 0, 0],
                              [0, 0, 0, 1.466337, 0], [0, 0, 0, 0, 1.7917595],
                              [0, 0, 0, 0, 1.7917595], [0, 0, 0, 1.4663371, 0],
                              [0, 1.098612, 0, 0, 0], [1.402368, 0, 0, 0, 0]],
          "input_dtype":
              dtypes.int64
      },
      {
          "testcase_name":
              "test_strings_tf_idf_output",
          "vocab_data":
              np.array([["fire"], ["earth"], ["earth"], ["earth"], ["earth"],
                        ["wind"], ["wind"], ["wind"], ["and"], ["and"]]),
          "input_data":
              np.array([["earth"], ["wind"], ["and"], ["fire"], ["fire"],
                        ["and"], ["earth"], ["michigan"]]),
          "kwargs": {
              "max_tokens": 5,
              "num_oov_indices": 1,
              "mask_token": "",
              "oov_token": "[OOV]",
              "output_mode": index_lookup.TFIDF,
              "dtype": dtypes.string,
          },
          "expected_output": [[0, 1.098612, 0, 0, 0], [0, 0, 1.252763, 0, 0],
                              [0, 0, 0, 1.466337, 0], [0, 0, 0, 0, 1.7917595],
                              [0, 0, 0, 0, 1.7917595], [0, 0, 0, 1.4663371, 0],
                              [0, 1.098612, 0, 0, 0], [1.402368, 0, 0, 0, 0]],
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
class IndexLookupLayerTest(keras_parameterized.TestCase,
                           preprocessing_test_utils.PreprocessingLayerTest):

  @parameterized.named_parameters(*_get_end_to_end_test_cases())
  def test_layer_end_to_end_with_adapt(self, vocab_data, input_data, kwargs,
                                       use_dataset, expected_output,
                                       input_dtype):
    cls = get_layer_class()
    if "invert" in kwargs and kwargs["invert"]:
      expected_output_dtype = kwargs["dtype"]
    elif "output_mode" in kwargs and kwargs["output_mode"] != index_lookup.INT:
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
    if "invert" in kwargs and kwargs["invert"]:
      self.assertAllEqual(expected_output, output_data)
    else:
      self.assertAllClose(expected_output, output_data)


@keras_parameterized.run_all_keras_modes
class CategoricalEncodingInputTest(
    keras_parameterized.TestCase,
    preprocessing_test_utils.PreprocessingLayerTest):

  def test_sparse_string_input(self):
    vocab_data = ["earth", "wind", "and", "fire"]
    input_array = sparse_tensor.SparseTensor(
        indices=[[0, 0], [1, 2]],
        values=["fire", "michigan"],
        dense_shape=[3, 4])

    expected_indices = [[0, 0], [1, 2]]
    expected_values = [5, 1]
    expected_dense_shape = [3, 4]

    input_data = keras.Input(shape=(None,), dtype=dtypes.string, sparse=True)
    layer = get_layer_class()(
        max_tokens=None,
        num_oov_indices=1,
        mask_token="",
        oov_token="[OOV]",
        dtype=dtypes.string)
    layer.set_vocabulary(vocab_data)
    int_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=int_data)
    output_data = model.predict(input_array, steps=1)
    self.assertAllEqual(expected_indices, output_data.indices)
    self.assertAllEqual(expected_values, output_data.values)
    self.assertAllEqual(expected_dense_shape, output_data.dense_shape)

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
    layer = get_layer_class()(
        max_tokens=None,
        dtype=dtypes.int64,
        num_oov_indices=1,
        mask_token=0,
        oov_token=-1)
    layer.set_vocabulary(vocab_data)
    int_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=int_data)
    output_data = model.predict(input_array, steps=1)
    self.assertAllEqual(expected_indices, output_data.indices)
    self.assertAllEqual(expected_values, output_data.values)
    self.assertAllEqual(expected_dense_shape, output_data.dense_shape)

  def test_ragged_string_input(self):
    vocab_data = ["earth", "wind", "and", "fire"]
    input_array = ragged_factory_ops.constant(
        [["earth", "wind", "fire"], ["fire", "and", "earth", "michigan"]])
    expected_output = [[2, 3, 5], [5, 4, 2, 1]]

    input_data = keras.Input(shape=(None,), dtype=dtypes.string, ragged=True)
    layer = get_layer_class()(
        max_tokens=None,
        num_oov_indices=1,
        mask_token="",
        oov_token="[OOV]",
        dtype=dtypes.string)
    layer.set_vocabulary(vocab_data)
    int_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_array)
    self.assertAllEqual(expected_output, output_dataset)

  def test_ragged_int_input(self):
    vocab_data = np.array([10, 11, 12, 13], dtype=np.int64)
    input_array = ragged_factory_ops.constant([[10, 11, 13], [13, 12, 10, 42]],
                                              dtype=np.int64)
    expected_output = [[2, 3, 5], [5, 4, 2, 1]]

    input_data = keras.Input(shape=(None,), dtype=dtypes.int64, ragged=True)
    layer = get_layer_class()(
        max_tokens=None,
        dtype=dtypes.int64,
        num_oov_indices=1,
        mask_token=0,
        oov_token=-1)
    layer.set_vocabulary(vocab_data)
    int_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_array)
    self.assertAllEqual(expected_output, output_dataset)

  def test_int32_input_with_int64_keys(self):
    vocab_data = np.array([10, 11, 12, 13], dtype=np.int64)
    input_array = ragged_factory_ops.constant([[10, 11, 13], [13, 12, 10, 42]],
                                              dtype=np.int32)
    expected_output = [[2, 3, 5], [5, 4, 2, 1]]

    input_data = keras.Input(shape=(None,), dtype=dtypes.int32, ragged=True)
    layer = get_layer_class()(
        max_tokens=None,
        dtype=dtypes.int64,
        num_oov_indices=1,
        mask_token=0,
        oov_token=-1)
    layer.set_vocabulary(vocab_data)
    int_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_array)
    self.assertAllEqual(expected_output, output_dataset)


@keras_parameterized.run_all_keras_modes
class CategoricalEncodingMultiOOVTest(
    keras_parameterized.TestCase,
    preprocessing_test_utils.PreprocessingLayerTest):

  def test_sparse_string_input_multi_bucket(self):
    vocab_data = ["earth", "wind", "and", "fire"]
    input_array = sparse_tensor.SparseTensor(
        indices=[[0, 0], [1, 2]],
        values=["fire", "ohio"],
        dense_shape=[3, 4])

    expected_indices = [[0, 0], [1, 2]]
    expected_values = [6, 2]
    expected_dense_shape = [3, 4]

    input_data = keras.Input(shape=(None,), dtype=dtypes.string, sparse=True)
    layer = get_layer_class()(
        max_tokens=None,
        num_oov_indices=2,
        mask_token="",
        oov_token="[OOV]",
        dtype=dtypes.string)
    layer.set_vocabulary(vocab_data)
    int_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=int_data)
    output_data = model.predict(input_array, steps=1)
    self.assertAllEqual(expected_indices, output_data.indices)
    self.assertAllEqual(expected_values, output_data.values)
    self.assertAllEqual(expected_dense_shape, output_data.dense_shape)

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
        max_tokens=None,
        dtype=dtypes.int64,
        num_oov_indices=2,
        mask_token=0,
        oov_token=-1)
    layer.set_vocabulary(vocab_data)
    int_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=int_data)
    output_data = model.predict(input_array, steps=1)
    self.assertAllEqual(expected_indices, output_data.indices)
    self.assertAllEqual(expected_values, output_data.values)
    self.assertAllEqual(expected_dense_shape, output_data.dense_shape)

  def test_ragged_string_input_multi_bucket(self):
    vocab_data = ["earth", "wind", "and", "fire"]
    input_array = ragged_factory_ops.constant(
        [["earth", "wind", "fire"], ["fire", "and", "earth", "ohio"]])
    expected_output = [[3, 4, 6], [6, 5, 3, 2]]

    input_data = keras.Input(shape=(None,), dtype=dtypes.string, ragged=True)
    layer = get_layer_class()(
        max_tokens=None,
        num_oov_indices=2,
        mask_token="",
        oov_token="[OOV]",
        dtype=dtypes.string)
    layer.set_vocabulary(vocab_data)
    int_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_array)
    self.assertAllEqual(expected_output, output_dataset)

  def test_ragged_int_input_multi_bucket(self):
    vocab_data = np.array([10, 11, 12, 13], dtype=np.int64)
    input_array = ragged_factory_ops.constant([[10, 11, 13], [13, 12, 10, 133]],
                                              dtype=np.int64)
    expected_output = [[3, 4, 6], [6, 5, 3, 2]]

    input_data = keras.Input(shape=(None,), dtype=dtypes.int64, ragged=True)
    layer = get_layer_class()(
        max_tokens=None,
        dtype=dtypes.int64,
        num_oov_indices=2,
        mask_token=0,
        oov_token=-1)
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
        values=["michigan", "fire", "michigan"],
        dense_shape=[3, 4])
    vocab_dataset = dataset_ops.Dataset.from_tensors(vocab_data)

    layer = get_layer_class()(
        max_tokens=None,
        num_oov_indices=1,
        mask_token="",
        oov_token="[OOV]",
        dtype=dtypes.string)
    layer.adapt(vocab_dataset)
    expected_vocabulary = ["", "[OOV]", "michigan", "fire"]
    self.assertAllEqual(expected_vocabulary, layer.get_vocabulary())

  def test_ragged_adapt(self):
    vocab_data = ragged_factory_ops.constant([["michigan"],
                                              ["fire", "michigan"]])
    vocab_dataset = dataset_ops.Dataset.from_tensors(vocab_data)

    layer = get_layer_class()(
        max_tokens=None,
        num_oov_indices=1,
        mask_token="",
        oov_token="[OOV]",
        dtype=dtypes.string)
    layer.adapt(vocab_dataset)
    expected_vocabulary = ["", "[OOV]", "michigan", "fire"]
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
    layer = get_layer_class()(
        max_tokens=None,
        dtype=dtypes.int64,
        num_oov_indices=1,
        mask_token=0,
        oov_token=-1)
    layer.set_vocabulary(vocab_data)
    int_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=int_data)
    output_data = model.predict(input_array, steps=1)
    self.assertAllEqual(expected_indices, output_data.indices)
    self.assertAllEqual(expected_values, output_data.values)
    self.assertAllEqual(expected_dense_shape, output_data.dense_shape)

  def test_ragged_string_input(self):
    vocab_data = ["earth", "wind", "and", "fire"]
    input_array = ragged_factory_ops.constant(
        [["earth", "wind", "fire"], ["fire", "and", "earth", "michigan"]])
    expected_output = [[2, 3, 5], [5, 4, 2, 1]]

    input_data = keras.Input(shape=(None,), dtype=dtypes.string, ragged=True)
    layer = get_layer_class()(
        max_tokens=None,
        num_oov_indices=1,
        mask_token="",
        oov_token="[OOV]",
        dtype=dtypes.string)
    layer.set_vocabulary(vocab_data)
    int_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_array)
    self.assertAllEqual(expected_output, output_dataset)

  def test_ragged_int_input(self):
    vocab_data = np.array([10, 11, 12, 13], dtype=np.int64)
    input_array = ragged_factory_ops.constant([[10, 11, 13], [13, 12, 10, 42]],
                                              dtype=np.int64)
    expected_output = [[2, 3, 5], [5, 4, 2, 1]]

    input_data = keras.Input(shape=(None,), dtype=dtypes.int64, ragged=True)
    layer = get_layer_class()(
        max_tokens=None,
        dtype=dtypes.int64,
        num_oov_indices=1,
        mask_token=0,
        oov_token=-1)
    layer.set_vocabulary(vocab_data)
    int_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_array)
    self.assertAllEqual(expected_output, output_dataset)

  def test_single_string_generator_dataset(self):

    def word_gen():
      for _ in itertools.count(1):
        yield "".join(random.choice(string.ascii_letters) for i in range(2))

    ds = dataset_ops.Dataset.from_generator(word_gen, dtypes.string,
                                            tensor_shape.TensorShape([]))
    batched_ds = ds.take(2)
    input_t = keras.Input(shape=(), dtype=dtypes.string)
    layer = get_layer_class()(
        max_tokens=10,
        num_oov_indices=0,
        mask_token=None,
        oov_token=None,
        dtype=dtypes.string)
    _ = layer(input_t)
    layer.adapt(batched_ds)


@keras_parameterized.run_all_keras_modes
class IndexLookupOutputTest(keras_parameterized.TestCase,
                            preprocessing_test_utils.PreprocessingLayerTest):

  def _write_to_temp_file(self, file_name, vocab_list):
    vocab_path = os.path.join(self.get_temp_dir(), file_name + ".txt")
    with gfile.GFile(vocab_path, "w") as writer:
      for vocab in vocab_list:
        writer.write(vocab + "\n")
      writer.flush()
      writer.close()
    return vocab_path

  def test_int_output(self):
    vocab_data = ["earth", "wind", "and", "fire"]
    input_array = np.array([["earth", "wind", "and", "fire"],
                            ["fire", "and", "earth", "michigan"]])
    expected_output = [[2, 3, 4, 5], [5, 4, 2, 1]]

    input_data = keras.Input(shape=(None,), dtype=dtypes.string)
    layer = get_layer_class()(
        max_tokens=None,
        num_oov_indices=1,
        mask_token="",
        oov_token="[OOV]",
        dtype=dtypes.string)
    layer.set_vocabulary(vocab_data)
    int_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_array)
    self.assertAllEqual(expected_output, output_dataset)

  def test_int_output_shape(self):
    input_data = keras.Input(batch_size=16, shape=(4,), dtype=dtypes.string)
    layer = get_layer_class()(
        max_tokens=2,
        num_oov_indices=1,
        mask_token="",
        oov_token="[OOV]",
        dtype=dtypes.string)
    int_data = layer(input_data)
    self.assertAllEqual(int_data.shape.as_list(), [16, 4])

  def test_int_output_no_reserved_zero(self):
    vocab_data = ["earth", "wind", "and", "fire"]
    input_array = np.array([["earth", "wind", "and", "fire"],
                            ["fire", "and", "earth", "michigan"]])
    expected_output = [[1, 2, 3, 4], [4, 3, 1, 0]]

    input_data = keras.Input(shape=(None,), dtype=dtypes.string)
    layer = get_layer_class()(
        max_tokens=None,
        num_oov_indices=1,
        mask_token=None,
        oov_token="[OOV]",
        dtype=dtypes.string)
    layer.set_vocabulary(vocab_data)
    int_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_array)
    self.assertAllEqual(expected_output, output_dataset)

  def test_int_output_no_oov(self):
    vocab_data = ["earth", "wind", "and", "fire"]
    input_array = np.array([["earth", "wind", "and", "ohio"],
                            ["fire", "and", "earth", "michigan"]])
    expected_output = [[1, 2, 3, -1], [4, 3, 1, -1]]

    input_data = keras.Input(shape=(None,), dtype=dtypes.string)
    layer = get_layer_class()(
        max_tokens=None,
        num_oov_indices=0,
        mask_token="",
        oov_token="[OOV]",
        dtype=dtypes.string)
    layer.set_vocabulary(vocab_data)
    int_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_array)
    self.assertAllEqual(expected_output, output_dataset)

  def test_int_output_explicit_vocab(self):
    vocab_data = ["earth", "wind", "and", "fire"]
    input_array = np.array([["earth", "wind", "and", "fire"],
                            ["fire", "and", "earth", "michigan"]])
    expected_output = [[2, 3, 4, 5], [5, 4, 2, 1]]

    input_data = keras.Input(shape=(None,), dtype=dtypes.string)
    layer = get_layer_class()(
        vocabulary=vocab_data,
        max_tokens=None,
        num_oov_indices=1,
        mask_token="",
        oov_token="[OOV]",
        dtype=dtypes.string)
    int_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_array)
    self.assertAllEqual(expected_output, output_dataset)

  def test_binary_output_hard_maximum(self):
    """Check binary output when pad_to_max_tokens=True."""
    vocab_data = ["earth", "wind", "and", "fire"]
    input_array = np.array([["earth", "wind", "and", "fire", ""],
                            ["fire", "fire", "and", "earth", "michigan"]])
    expected_output = [
        [0, 1, 1, 1, 1, 0],
        [1, 1, 0, 1, 1, 0],
    ]

    input_data = keras.Input(shape=(None,), dtype=dtypes.string)
    layer = get_layer_class()(
        max_tokens=6,
        num_oov_indices=1,
        mask_token="",
        oov_token="[OOV]",
        output_mode=index_lookup.BINARY,
        pad_to_max_tokens=True,
        dtype=dtypes.string)
    layer.set_vocabulary(vocab_data)
    binary_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=binary_data)
    output_dataset = model.predict(input_array)
    self.assertAllEqual(expected_output, output_dataset)

  def test_binary_output_no_oov(self):
    """Check binary output when pad_to_max_tokens=True."""
    vocab_data = ["earth", "wind", "and", "fire"]
    input_array = np.array([["earth", "wind", "and", "fire", "ohio"],
                            ["fire", "fire", "and", "earth", "michigan"]])
    expected_output = [
        [1, 1, 1, 1, 0],
        [1, 0, 1, 1, 0],
    ]

    input_data = keras.Input(shape=(None,), dtype=dtypes.string)
    layer = get_layer_class()(
        max_tokens=5,
        num_oov_indices=0,
        mask_token="",
        oov_token="[OOV]",
        output_mode=index_lookup.BINARY,
        pad_to_max_tokens=True,
        dtype=dtypes.string)
    layer.set_vocabulary(vocab_data)
    binary_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=binary_data)
    output_dataset = model.predict(input_array)
    self.assertAllEqual(expected_output, output_dataset)

  def test_binary_output_hard_maximum_multiple_adapts(self):
    input_array = np.array([["earth", "wind", "and", "earth"],
                            ["ohio", "and", "earth", "michigan"]])
    adapt_data = ["earth", "earth", "earth", "earth", "wind", "wind", "wind"]
    first_expected_output = [
        [1, 1, 1, 0, 0],
        [1, 1, 0, 0, 0],
    ]
    second_adapt_data = [
        "earth", "earth", "earth", "earth", "wind", "wind", "wind", "and",
        "and", "fire"
    ]
    second_expected_output = [
        [0, 1, 1, 1, 0],
        [1, 1, 0, 1, 0],
    ]

    input_data = keras.Input(shape=(None,), dtype=dtypes.string)
    layer = get_layer_class()(
        max_tokens=5,
        num_oov_indices=1,
        mask_token="",
        oov_token="[OOV]",
        output_mode=index_lookup.BINARY,
        pad_to_max_tokens=True,
        dtype=dtypes.string)
    int_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=int_data)

    # Test the first adapt
    layer.adapt(adapt_data)
    first_output = model.predict(input_array)
    # Test the second adapt
    layer.adapt(second_adapt_data)
    second_output = model.predict(input_array)
    self.assertAllEqual(first_expected_output, first_output)
    self.assertAllEqual(second_expected_output, second_output)

  def test_binary_output_soft_maximum(self):
    """Check binary output when pad_to_max_tokens=False."""
    vocab_data = ["earth", "wind", "and", "fire"]
    input_array = np.array([["earth", "wind", "and", "fire", ""],
                            ["fire", "and", "earth", "michigan", ""]])
    expected_output = [
        [0, 1, 1, 1, 1],
        [1, 1, 0, 1, 1],
    ]

    input_data = keras.Input(shape=(None,), dtype=dtypes.string)
    layer = get_layer_class()(
        max_tokens=None,
        num_oov_indices=1,
        mask_token="",
        oov_token="[OOV]",
        output_mode=index_lookup.BINARY,
        dtype=dtypes.string)
    layer.set_vocabulary(vocab_data)
    binary_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=binary_data)
    output_dataset = model.predict(input_array)
    self.assertAllEqual(expected_output, output_dataset)

  def test_binary_output_shape(self):
    input_data = keras.Input(batch_size=16, shape=(4,), dtype=dtypes.string)
    layer = get_layer_class()(
        max_tokens=2,
        num_oov_indices=1,
        mask_token="",
        oov_token="[OOV]",
        output_mode=index_lookup.BINARY,
        dtype=dtypes.string)
    binary_data = layer(input_data)
    self.assertAllEqual(binary_data.shape.as_list(), [16, 2])

  def test_count_output_hard_maxiumum(self):
    """Check count output when pad_to_max_tokens=True."""
    vocab_data = ["earth", "wind", "and", "fire"]
    input_array = np.array([["earth", "wind", "and", "wind", ""],
                            ["fire", "fire", "fire", "michigan", ""]])
    expected_output = [
        [0, 1, 2, 1, 0, 0],
        [1, 0, 0, 0, 3, 0],
    ]

    input_data = keras.Input(shape=(None,), dtype=dtypes.string)
    layer = get_layer_class()(
        max_tokens=6,
        num_oov_indices=1,
        mask_token="",
        oov_token="[OOV]",
        output_mode=index_lookup.COUNT,
        pad_to_max_tokens=True,
        dtype=dtypes.string)
    layer.set_vocabulary(vocab_data)
    count_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=count_data)
    output_dataset = model.predict(input_array)
    self.assertAllEqual(expected_output, output_dataset)

  def test_count_output_soft_maximum(self):
    """Check count output when pad_to_max_tokens=False."""
    vocab_data = ["earth", "wind", "and", "fire"]
    input_array = np.array([["earth", "wind", "and", "wind", ""],
                            ["fire", "fire", "fire", "michigan", ""]])
    expected_output = [
        [0, 1, 2, 1, 0],
        [1, 0, 0, 0, 3],
    ]

    input_data = keras.Input(shape=(None,), dtype=dtypes.string)
    layer = get_layer_class()(
        max_tokens=None,
        num_oov_indices=1,
        mask_token="",
        oov_token="[OOV]",
        output_mode=index_lookup.COUNT,
        dtype=dtypes.string)
    layer.set_vocabulary(vocab_data)
    count_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=count_data)
    output_dataset = model.predict(input_array)
    self.assertAllEqual(expected_output, output_dataset)

  def test_count_output_shape(self):
    input_data = keras.Input(batch_size=16, shape=(4,), dtype=dtypes.string)
    layer = get_layer_class()(
        max_tokens=2,
        num_oov_indices=1,
        mask_token="",
        oov_token="[OOV]",
        output_mode=index_lookup.COUNT,
        dtype=dtypes.string)
    count_data = layer(input_data)
    self.assertAllEqual(count_data.shape.as_list(), [16, 2])

  def test_ifidf_output_hard_maximum(self):
    """Check tf-idf output when pad_to_max_tokens=True."""
    vocab_data = ["earth", "wind", "and", "fire"]
    # OOV idf weight (bucket 0) should 0.5, the average of passed weights.
    idf_weights = [.4, .25, .75, .6]
    input_array = np.array([["earth", "wind", "and", "earth", ""],
                            ["ohio", "fire", "earth", "michigan", ""]])
    expected_output = [
        [0.00, 0.80, 0.25, 0.75, 0.00, 0.00],
        [1.00, 0.40, 0.00, 0.00, 0.60, 0.00],
    ]

    input_data = keras.Input(shape=(None,), dtype=dtypes.string)
    layer = get_layer_class()(
        max_tokens=6,
        num_oov_indices=1,
        mask_token="",
        oov_token="[OOV]",
        output_mode=index_lookup.TFIDF,
        pad_to_max_tokens=True,
        dtype=dtypes.string)
    layer.set_vocabulary(vocab_data, idf_weights=idf_weights)
    layer_output = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=layer_output)
    output_dataset = model.predict(input_array)
    self.assertAllClose(expected_output, output_dataset)

  def test_ifidf_output_soft_maximum(self):
    """Check tf-idf output when pad_to_max_tokens=False."""
    vocab_data = ["earth", "wind", "and", "fire"]
    # OOV idf weight (bucket 0) should 0.5, the average of passed weights.
    idf_weights = [.4, .25, .75, .6]
    input_array = np.array([["earth", "wind", "and", "earth", ""],
                            ["ohio", "fire", "earth", "michigan", ""]])
    expected_output = [
        [0.00, 0.80, 0.25, 0.75, 0.00],
        [1.00, 0.40, 0.00, 0.00, 0.60],
    ]

    input_data = keras.Input(shape=(None,), dtype=dtypes.string)
    layer = get_layer_class()(
        max_tokens=None,
        num_oov_indices=1,
        mask_token="",
        oov_token="[OOV]",
        output_mode=index_lookup.TFIDF,
        dtype=dtypes.string)
    layer.set_vocabulary(vocab_data, idf_weights=idf_weights)
    layer_output = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=layer_output)
    output_dataset = model.predict(input_array)
    self.assertAllClose(expected_output, output_dataset)

  def test_ifidf_output_shape(self):
    input_data = keras.Input(batch_size=16, shape=(4,), dtype=dtypes.string)
    layer = get_layer_class()(
        max_tokens=2,
        num_oov_indices=1,
        mask_token="",
        oov_token="[OOV]",
        output_mode=index_lookup.COUNT,
        dtype=dtypes.string)
    layer_output = layer(input_data)
    self.assertAllEqual(layer_output.shape.as_list(), [16, 2])

  def test_int_output_file_vocab(self):
    vocab_data = ["earth", "wind", "and", "fire"]
    input_array = np.array([["earth", "wind", "and", "fire"],
                            ["fire", "", "earth", "michigan"]])
    expected_output = [[2, 3, 4, 5], [5, 0, 2, 1]]

    vocab_file = self._write_to_temp_file("temp", vocab_data)
    vocabulary_initializer = lookup_ops.TextFileInitializer(
        filename=vocab_file,
        key_dtype=dtypes.string,
        key_index=lookup_ops.TextFileIndex.WHOLE_LINE,
        value_dtype=dtypes.int64,
        value_index=lookup_ops.TextFileIndex.LINE_NUMBER,
        value_index_offset=2)

    input_data = keras.Input(shape=(None,), dtype=dtypes.string)
    layer = get_layer_class()(
        vocabulary=vocabulary_initializer,
        max_tokens=None,
        num_oov_indices=1,
        mask_token="",
        oov_token="[OOV]",
        dtype=dtypes.string)
    int_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_array)
    self.assertAllEqual(expected_output, output_dataset)

  def test_int_output_int_file_vocab(self):
    vocab_data = ["10", "20", "30", "40"]
    input_array = np.array([[10, 20, 30, 40], [40, 0, 10, 42]])
    expected_output = [[2, 3, 4, 5], [5, 0, 2, 1]]

    vocab_file = self._write_to_temp_file("temp", vocab_data)
    vocabulary_initializer = lookup_ops.TextFileInitializer(
        filename=vocab_file,
        key_dtype=dtypes.int64,
        key_index=lookup_ops.TextFileIndex.WHOLE_LINE,
        value_dtype=dtypes.int64,
        value_index=lookup_ops.TextFileIndex.LINE_NUMBER,
        value_index_offset=2)

    input_data = keras.Input(shape=(None,), dtype=dtypes.int64)
    layer = get_layer_class()(
        vocabulary=vocabulary_initializer,
        max_tokens=None,
        num_oov_indices=1,
        mask_token=0,
        oov_token=-1,
        dtype=dtypes.int64)
    int_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_array)
    self.assertAllEqual(expected_output, output_dataset)


@keras_parameterized.run_all_keras_modes
class IndexLookupVocabularyTest(keras_parameterized.TestCase,
                                preprocessing_test_utils.PreprocessingLayerTest
                               ):

  def test_int_output_explicit_vocab(self):
    vocab_data = ["earth", "wind", "and", "fire"]
    input_array = np.array([["earth", "wind", "and", "fire"],
                            ["fire", "and", "earth", "michigan"]])
    expected_output = [[2, 3, 4, 5], [5, 4, 2, 1]]

    input_data = keras.Input(shape=(None,), dtype=dtypes.string)
    layer = get_layer_class()(
        vocabulary=vocab_data,
        max_tokens=None,
        num_oov_indices=1,
        mask_token="",
        oov_token="[OOV]",
        dtype=dtypes.string)
    int_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_array)
    self.assertAllEqual(expected_output, output_dataset)

  def test_int_output_explicit_vocab_with_special_tokens(self):
    vocab_data = ["", "[OOV]", "earth", "wind", "and", "fire"]
    input_array = np.array([["earth", "wind", "and", "fire"],
                            ["fire", "and", "earth", "michigan"]])
    expected_output = [[2, 3, 4, 5], [5, 4, 2, 1]]

    input_data = keras.Input(shape=(None,), dtype=dtypes.string)
    layer = get_layer_class()(
        vocabulary=vocab_data,
        max_tokens=None,
        num_oov_indices=1,
        mask_token="",
        oov_token="[OOV]",
        dtype=dtypes.string)
    int_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_array)
    self.assertAllEqual(expected_output, output_dataset)

  def test_vocab_with_max_cap(self):
    vocab_data = ["", "[OOV]", "wind", "and", "fire"]
    layer = get_layer_class()(
        max_tokens=5,
        num_oov_indices=1,
        mask_token="",
        oov_token="[OOV]",
        dtype=dtypes.string)
    layer.set_vocabulary(vocab_data)
    returned_vocab = layer.get_vocabulary()
    self.assertAllEqual(vocab_data, returned_vocab)
    self.assertAllEqual(layer.vocab_size(), 5)

  def test_int_vocab_with_max_cap(self):
    vocab_data = [0, -1, 42, 1276, 1138]
    layer = get_layer_class()(
        max_tokens=5,
        num_oov_indices=1,
        mask_token=0,
        oov_token=-1,
        dtype=dtypes.int64)
    layer.set_vocabulary(vocab_data)
    returned_vocab = layer.get_vocabulary()
    self.assertAllEqual(vocab_data, returned_vocab)
    self.assertAllEqual(layer.vocab_size(), 5)

  def test_vocab_with_multiple_oov_indices(self):
    vocab_data = ["", "[OOV]", "[OOV]", "[OOV]", "wind"]
    layer = get_layer_class()(
        max_tokens=None,
        num_oov_indices=3,
        mask_token="",
        oov_token="[OOV]",
        dtype=dtypes.string)
    layer.set_vocabulary(vocab_data)
    returned_vocab = layer.get_vocabulary()
    self.assertAllEqual(vocab_data, returned_vocab)

  def test_int_vocab_with_multiple_oov_indices(self):
    vocab_data = [0, -1, -1, -1, 42]
    layer = get_layer_class()(
        max_tokens=None,
        num_oov_indices=3,
        mask_token=0,
        oov_token=-1,
        dtype=dtypes.int64)
    layer.set_vocabulary(vocab_data)
    returned_vocab = layer.get_vocabulary()
    self.assertAllEqual(vocab_data, returned_vocab)

  def test_non_unique_vocab_fails(self):
    vocab_data = ["earth", "wind", "and", "fire", "fire"]
    with self.assertRaisesRegex(ValueError, ".*repeated term.*fire.*"):
      _ = get_layer_class()(
          vocabulary=vocab_data,
          max_tokens=None,
          num_oov_indices=1,
          mask_token="",
          oov_token="[OOV]",
          dtype=dtypes.string)

  def test_vocab_with_oov_and_wrong_mask_fails(self):
    vocab_data = ["custom_mask", "[OOV]", "earth", "wind", "and", "fire"]
    layer = get_layer_class()(
        max_tokens=None,
        num_oov_indices=1,
        mask_token="",
        oov_token="[OOV]",
        dtype=dtypes.string)
    with self.assertRaisesRegex(ValueError, ".*does not have the mask token.*"):
      layer.set_vocabulary(vocab_data)

  def test_vocab_with_oov_and_no_mask_fails(self):
    vocab_data = ["[OOV]", "earth", "wind", "and", "fire"]
    layer = get_layer_class()(
        max_tokens=None,
        num_oov_indices=1,
        mask_token="",
        oov_token="[OOV]",
        dtype=dtypes.string)
    with self.assertRaisesRegex(ValueError, ".*Reserved OOV.*"):
      layer.set_vocabulary(vocab_data)

  def test_vocab_with_mask_but_no_oov_fails(self):
    vocab_data = ["", "earth", "wind", "and", "fire"]
    layer = get_layer_class()(
        max_tokens=None,
        num_oov_indices=1,
        mask_token="",
        oov_token="[OOV]",
        dtype=dtypes.string)
    with self.assertRaisesRegex(ValueError, ".*does not have the OOV token.*"):
      layer.set_vocabulary(vocab_data)

  def test_vocab_with_repeated_element_fails(self):
    vocab_data = ["earth", "earth", "wind", "and", "fire"]
    layer = get_layer_class()(
        max_tokens=None,
        num_oov_indices=1,
        mask_token="",
        oov_token="[OOV]",
        dtype=dtypes.string)
    with self.assertRaisesRegex(ValueError, ".*repeated term.*earth.*"):
      layer.set_vocabulary(vocab_data)

  def test_vocab_with_reserved_oov_element_fails(self):
    vocab_data = ["earth", "test", "[OOV]", "wind", "and", "fire"]
    layer = get_layer_class()(
        max_tokens=None,
        num_oov_indices=1,
        mask_token="",
        oov_token="[OOV]",
        dtype=dtypes.string)
    with self.assertRaisesRegex(ValueError, ".*Reserved OOV.*"):
      layer.set_vocabulary(vocab_data)

  def test_vocab_with_reserved_mask_element_fails(self):
    vocab_data = ["earth", "mask_token", "wind", "and", "fire"]
    layer = get_layer_class()(
        max_tokens=None,
        num_oov_indices=1,
        mask_token="mask_token",
        oov_token="[OOV]",
        dtype=dtypes.string)
    with self.assertRaisesRegex(ValueError, ".*Reserved mask.*"):
      layer.set_vocabulary(vocab_data)

  def test_vocab_set_after_call_pad_to_max_false_fails(self):
    vocab_data = ["earth", "wind", "and", "fire"]
    layer = get_layer_class()(
        max_tokens=None,
        num_oov_indices=1,
        mask_token="",
        oov_token="[OOV]",
        pad_to_max_tokens=False,
        output_mode=index_lookup.BINARY,
        dtype=dtypes.string)
    layer.set_vocabulary(vocab_data)
    # Calling the layer should lock the vocabulary.
    _ = layer([["earth"]])
    with self.assertRaisesRegex(RuntimeError, "vocabulary cannot be changed"):
      layer.set_vocabulary(vocab_data)

  def test_vocab_with_idf_weights_non_tfidf_output_fails(self):
    vocab_data = ["earth", "wind", "and", "fire"]
    weight_data = [1, 1, 1, 1, 1]
    layer = get_layer_class()(
        max_tokens=None,
        num_oov_indices=1,
        mask_token="",
        oov_token="[OOV]",
        output_mode=index_lookup.BINARY,
        dtype=dtypes.string)
    with self.assertRaisesRegex(ValueError,
                                "`idf_weights` should only be set if"):
      layer.set_vocabulary(vocab_data, idf_weights=weight_data)

  def test_vocab_with_idf_weights_length_mismatch_fails(self):
    vocab_data = ["earth", "wind", "and", "fire"]
    weight_data = [1, 1, 1, 1, 1]  # too long
    layer = get_layer_class()(
        max_tokens=None,
        num_oov_indices=1,
        mask_token="",
        oov_token="[OOV]",
        output_mode=index_lookup.TFIDF,
        dtype=dtypes.string)
    with self.assertRaisesRegex(
        ValueError, "`idf_weights` must be the same length as vocab"):
      layer.set_vocabulary(vocab_data, idf_weights=weight_data)

  def test_vocab_without_idf_weights_tfidf_output_fails(self):
    vocab_data = ["earth", "wind", "and", "fire"]
    layer = get_layer_class()(
        max_tokens=None,
        num_oov_indices=1,
        mask_token="",
        oov_token="[OOV]",
        output_mode=index_lookup.TFIDF,
        dtype=dtypes.string)
    with self.assertRaisesRegex(
        ValueError, "`idf_weights` must be set if output_mode is TFIDF"):
      layer.set_vocabulary(vocab_data)

  def test_non_unique_int_vocab_fails(self):
    vocab_data = [12, 13, 14, 15, 15]
    with self.assertRaisesRegex(ValueError, "repeated term.*15"):
      _ = get_layer_class()(
          vocabulary=vocab_data,
          max_tokens=None,
          num_oov_indices=1,
          mask_token=0,
          oov_token=-1,
          dtype=dtypes.int64)

  def test_int_vocab_with_oov_and_wrong_mask_fails(self):
    vocab_data = [1234, -1, 11, 21, 13, 14]
    layer = get_layer_class()(
        max_tokens=None,
        num_oov_indices=1,
        mask_token=0,
        oov_token=-1,
        dtype=dtypes.int64)
    with self.assertRaisesRegex(ValueError, "does not have the mask token `0`"):
      layer.set_vocabulary(vocab_data)

  def test_int_vocab_with_oov_and_no_mask_fails(self):
    vocab_data = [-1, 11, 12, 13, 14]
    layer = get_layer_class()(
        max_tokens=None,
        num_oov_indices=1,
        mask_token=0,
        oov_token=-1,
        dtype=dtypes.int64)
    with self.assertRaisesRegex(ValueError, "Reserved OOV"):
      layer.set_vocabulary(vocab_data)

  def test_int_vocab_with_mask_but_no_oov_fails(self):
    vocab_data = [0, 11, 12, 13, 14]
    layer = get_layer_class()(
        max_tokens=None,
        num_oov_indices=1,
        mask_token=0,
        oov_token=-1,
        dtype=dtypes.int64)
    with self.assertRaisesRegex(ValueError, "does not have the OOV token `-1`"):
      layer.set_vocabulary(vocab_data)

  def test_int_vocab_with_repeated_element_fails(self):
    vocab_data = [11, 11, 34, 23, 124]
    layer = get_layer_class()(
        max_tokens=None,
        num_oov_indices=1,
        mask_token=0,
        oov_token=-1,
        dtype=dtypes.int64)
    with self.assertRaisesRegex(ValueError, "repeated term.*11"):
      layer.set_vocabulary(vocab_data)

  def test_int_vocab_with_reserved_oov_element_fails(self):
    vocab_data = [14, 38, -1, 34, 3, 84]
    layer = get_layer_class()(
        max_tokens=None,
        num_oov_indices=1,
        mask_token=0,
        oov_token=-1,
        dtype=dtypes.int64)
    with self.assertRaisesRegex(ValueError, "Reserved OOV"):
      layer.set_vocabulary(vocab_data)

  def test_int_vocab_with_reserved_mask_element_fails(self):
    vocab_data = [125, 0, 3, 4, 94]
    layer = get_layer_class()(
        max_tokens=None,
        num_oov_indices=1,
        mask_token=0,
        oov_token=-1,
        dtype=dtypes.int64)
    with self.assertRaisesRegex(ValueError, "Reserved mask"):
      layer.set_vocabulary(vocab_data)


@keras_parameterized.run_all_keras_modes
class IndexLookupInverseVocabularyTest(
    keras_parameterized.TestCase,
    preprocessing_test_utils.PreprocessingLayerTest):

  def test_int_output_explicit_vocab(self):
    vocab_data = ["", "[OOV]", "earth", "wind", "and", "fire"]
    input_array = np.array([[2, 3, 4, 5], [5, 4, 2, 1]])
    expected_output = np.array([["earth", "wind", "and", "fire"],
                                ["fire", "and", "earth", "[OOV]"]])

    input_data = keras.Input(shape=(None,), dtype=dtypes.int64)
    layer = get_layer_class()(
        vocabulary=vocab_data,
        max_tokens=None,
        num_oov_indices=1,
        mask_token="",
        oov_token="[OOV]",
        dtype=dtypes.string,
        invert=True)
    int_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_array)
    self.assertAllEqual(expected_output, output_dataset)

  def test_vocab_with_max_cap(self):
    vocab_data = ["", "[OOV]", "wind", "and", "fire"]
    layer = get_layer_class()(
        max_tokens=5,
        num_oov_indices=1,
        mask_token="",
        oov_token="[OOV]",
        dtype=dtypes.string,
        invert=True)
    layer.set_vocabulary(vocab_data)
    returned_vocab = layer.get_vocabulary()
    self.assertAllEqual(vocab_data, returned_vocab)

  def test_int_vocab_with_max_cap(self):
    vocab_data = [0, -1, 42, 1276, 1138]
    layer = get_layer_class()(
        max_tokens=5,
        num_oov_indices=1,
        mask_token=0,
        oov_token=-1,
        dtype=dtypes.int64,
        invert=True)
    layer.set_vocabulary(vocab_data)
    returned_vocab = layer.get_vocabulary()
    self.assertAllEqual(vocab_data, returned_vocab)

  def test_non_unique_vocab_fails(self):
    vocab_data = ["earth", "wind", "and", "fire", "fire"]
    with self.assertRaisesRegex(ValueError, ".*repeated term.*fire.*"):
      _ = get_layer_class()(
          vocabulary=vocab_data,
          max_tokens=None,
          num_oov_indices=1,
          mask_token="",
          oov_token="[OOV]",
          dtype=dtypes.string,
          invert=True)

  def test_non_int_output_fails(self):
    with self.assertRaisesRegex(ValueError, "`output_mode` must be int"):
      _ = get_layer_class()(
          max_tokens=None,
          num_oov_indices=1,
          mask_token="",
          oov_token="[OOV]",
          dtype=dtypes.string,
          output_mode=index_lookup.COUNT,
          invert=True)

  def test_vocab_with_repeated_element_fails(self):
    vocab_data = ["earth", "earth", "wind", "and", "fire"]
    layer = get_layer_class()(
        max_tokens=None,
        num_oov_indices=1,
        mask_token="",
        oov_token="[OOV]",
        dtype=dtypes.string,
        invert=True)
    with self.assertRaisesRegex(ValueError, ".*repeated term.*earth.*"):
      layer.set_vocabulary(vocab_data)

  def test_vocab_with_reserved_mask_element_fails(self):
    vocab_data = ["earth", "mask_token", "wind", "and", "fire"]
    layer = get_layer_class()(
        max_tokens=None,
        num_oov_indices=1,
        mask_token="mask_token",
        oov_token="[OOV]",
        dtype=dtypes.string,
        invert=True)
    with self.assertRaisesRegex(ValueError, ".*Reserved mask.*"):
      layer.set_vocabulary(vocab_data)

  def test_non_unique_int_vocab_fails(self):
    vocab_data = [12, 13, 14, 15, 15]
    with self.assertRaisesRegex(ValueError, ".*repeated term.*15.*"):
      _ = get_layer_class()(
          vocabulary=vocab_data,
          max_tokens=None,
          num_oov_indices=1,
          mask_token=0,
          oov_token=-1,
          dtype=dtypes.int64,
          invert=True)

  def test_int_vocab_with_repeated_element_fails(self):
    vocab_data = [11, 11, 34, 23, 124]
    layer = get_layer_class()(
        max_tokens=None,
        num_oov_indices=1,
        mask_token=0,
        oov_token=-1,
        dtype=dtypes.int64,
        invert=True)
    with self.assertRaisesRegex(ValueError, ".*repeated term.*11.*"):
      layer.set_vocabulary(vocab_data)


@keras_parameterized.run_all_keras_modes(always_skip_eager=True)
class IndexLookupSaveableTest(keras_parameterized.TestCase,
                              preprocessing_test_utils.PreprocessingLayerTest):

  def test_ops_are_not_added_with_multiple_get_set_weights(self):
    vocab_data = ["earth", "wind", "and", "fire"]

    input_data = keras.Input(shape=(None,), dtype=dtypes.string)
    layer = get_layer_class()(
        max_tokens=10,
        num_oov_indices=1,
        mask_token="",
        oov_token="[OOV]",
        dtype=dtypes.string)
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
    layer = get_layer_class()(
        max_tokens=10,
        num_oov_indices=1,
        mask_token="",
        oov_token="[OOV]",
        dtype=dtypes.string)
    layer.set_vocabulary(vocab_data)
    int_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=int_data)
    path = os.path.join(self.get_temp_dir(), "model")
    with self.assertRaisesRegex(NotImplementedError,
                                "Save or restore weights that is not.*"):
      save.save_model(model, path, save_format="h5")


@keras_parameterized.run_all_keras_modes
class IndexLookupErrorTest(keras_parameterized.TestCase,
                           preprocessing_test_utils.PreprocessingLayerTest):

  def test_too_long_vocab_fails_in_single_setting(self):
    vocab_data = ["earth", "wind", "and", "fire"]

    layer = get_layer_class()(
        max_tokens=4,
        num_oov_indices=1,
        mask_token="",
        oov_token="[OOV]",
        dtype=dtypes.string)
    with self.assertRaisesRegex(ValueError,
                                "vocabulary larger than the maximum vocab.*"):
      layer.set_vocabulary(vocab_data)

  def test_zero_max_tokens_fails(self):
    with self.assertRaisesRegex(ValueError, ".*max_tokens.*"):
      _ = get_layer_class()(
          max_tokens=0,
          num_oov_indices=1,
          mask_token="",
          oov_token="[OOV]",
          dtype=dtypes.string)


@keras_parameterized.run_all_keras_modes
class IndexLookupSavingTest(keras_parameterized.TestCase,
                            preprocessing_test_utils.PreprocessingLayerTest):

  def test_vocabulary_persistence_across_saving(self):
    vocab_data = ["earth", "wind", "and", "fire"]
    input_array = np.array([["earth", "wind", "and", "fire"],
                            ["fire", "and", "earth", "michigan"]])
    expected_output = [[2, 3, 4, 5], [5, 4, 2, 1]]

    # Build and validate a golden model.
    input_data = keras.Input(shape=(None,), dtype=dtypes.string)
    layer = get_layer_class()(
        max_tokens=None,
        num_oov_indices=1,
        mask_token="",
        oov_token="[OOV]",
        dtype=dtypes.string)
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
        output_path, custom_objects={"IndexLookup": get_layer_class()})

    # Ensure that the loaded model is unique (so that the save/load is real)
    self.assertIsNot(model, loaded_model)

    # Validate correctness of the new model.
    new_output_dataset = loaded_model.predict(input_array)
    self.assertAllEqual(new_output_dataset, expected_output)


@keras_parameterized.run_all_keras_modes
class IndexLookupStringCombinerTest(
    keras_parameterized.TestCase,
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
        "idf_weights": None,
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
              "idf_weights": None,
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
              "idf_weights": None,
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
              "idf_weights": None,
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
              "idf_weights": None,
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
              "idf_weights": None,
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


@keras_parameterized.run_all_keras_modes
class IndexLookupIntCombinerTest(keras_parameterized.TestCase,
                                 preprocessing_test_utils.PreprocessingLayerTest
                                ):

  def compare_text_accumulators(self, a, b, msg=None):
    if a is None or b is None:
      self.assertAllEqual(a, b, msg=msg)

    self.assertAllEqual(a.count_dict, b.count_dict, msg=msg)

  compare_accumulators = compare_text_accumulators

  def update_accumulator(self, accumulator, data):
    accumulator.count_dict.update(dict(zip(data["vocab"], data["counts"])))

    return accumulator

  def test_combiner_api_compatibility_int_mode(self):
    data = np.array([[42, 1138, 725, 1729], [42, 1138, 725, 203]])
    combiner = index_lookup._IndexLookupCombiner()
    expected_accumulator_output = {
        "vocab": np.array([1138, 725, 42, 1729, 203]),
        "counts": np.array([2, 2, 2, 1, 1]),
    }
    expected_extract_output = {
        "vocab": np.array([1138, 725, 42, 1729, 203]),
        "idf_weights": None,
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
          "testcase_name": "top_k_smaller_than_full_vocab",
          "data": np.array([[42, 1138], [1729, 1138], [725], [1729, 1138]]),
          "vocab_size": 3,
          "expected_accumulator_output": {
              "vocab": np.array([1138, 1729, 725, 42]),
              "counts": np.array([3, 2, 1, 1]),
          },
          "expected_extract_output": {
              "vocab": np.array([1138, 1729, 725]),
              "idf_weights": None,
          },
      },
      {
          "testcase_name": "top_k_larger_than_full_vocab",
          "data": np.array([[42, 1138], [1729, 1138], [725], [1729, 1138]]),
          "vocab_size": 10,
          "expected_accumulator_output": {
              "vocab": np.array([1138, 1729, 725, 42]),
              "counts": np.array([3, 2, 1, 1]),
          },
          "expected_extract_output": {
              "vocab": np.array([1138, 1729, 725, 42]),
              "idf_weights": None,
          },
      },
      {
          "testcase_name": "no_top_k",
          "data": np.array([[42, 1138], [1729, 1138], [725], [1729, 1138]]),
          "vocab_size": None,
          "expected_accumulator_output": {
              "vocab": np.array([1138, 1729, 725, 42]),
              "counts": np.array([3, 2, 1, 1]),
          },
          "expected_extract_output": {
              "vocab": np.array([1138, 1729, 725, 42]),
              "idf_weights": None,
          },
      },
      {
          "testcase_name": "single_element_per_row",
          "data": np.array([[42], [1138], [1729], [1138], [725]]),
          "vocab_size": 3,
          "expected_accumulator_output": {
              "vocab": np.array([1138, 1729, 725, 42]),
              "counts": np.array([2, 1, 1, 1]),
          },
          "expected_extract_output": {
              "vocab": np.array([1138, 1729, 725]),
              "idf_weights": None,
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
              np.array([[42, 42], [1138, 1138], [1729, 1729], [1138, 1138],
                        [725, 203]]),
          "vocab_size":
              3,
          "expected_accumulator_output": {
              "vocab": np.array([1138, 42, 1729, 725, 203]),
              "counts": np.array([4, 2, 2, 1, 1]),
          },
          "expected_extract_output": {
              "vocab": np.array([1138, 1729, 42]),
              "idf_weights": None,
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
