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
"""Tests for Keras text categorical_encoding preprocessing layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from absl.testing import parameterized
import numpy as np

from tensorflow.python import keras

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.keras import backend
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras.layers import core
from tensorflow.python.keras.layers.preprocessing import categorical_encoding
from tensorflow.python.keras.layers.preprocessing import categorical_encoding_v1
from tensorflow.python.keras.layers.preprocessing import preprocessing_test_utils
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import test


def get_layer_class():
  if context.executing_eagerly():
    return categorical_encoding.CategoricalEncoding
  else:
    return categorical_encoding_v1.CategoricalEncoding


@keras_parameterized.run_all_keras_modes(always_skip_v1=True)
class CategoricalEncodingInputTest(
    keras_parameterized.TestCase,
    preprocessing_test_utils.PreprocessingLayerTest):

  def test_dense_input_sparse_output(self):
    input_array = constant_op.constant([[1, 2, 3], [3, 3, 0]])

    # The expected output should be (X for missing value):
    # [[X, 1, 1, 1]
    #  [1, X, X, X]
    #  [X, X, X, 2]]
    expected_indices = [[0, 1], [0, 2], [0, 3], [1, 0], [1, 3]]
    expected_values = [1, 1, 1, 1, 2]
    max_tokens = 6

    input_data = keras.Input(shape=(None,), dtype=dtypes.int32)
    layer = get_layer_class()(
        max_tokens=max_tokens,
        output_mode=categorical_encoding.COUNT,
        sparse=True)
    int_data = layer(input_data)

    model = keras.Model(inputs=input_data, outputs=int_data)
    sp_output_dataset = model.predict(input_array, steps=1)
    self.assertAllEqual(expected_values, sp_output_dataset.values)
    self.assertAllEqual(expected_indices, sp_output_dataset.indices)

    # Assert sparse output is same as dense output.
    layer = get_layer_class()(
        max_tokens=max_tokens,
        output_mode=categorical_encoding.COUNT,
        sparse=False)
    int_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_array, steps=1)
    self.assertAllEqual(
        sparse_ops.sparse_tensor_to_dense(sp_output_dataset, default_value=0),
        output_dataset)

  def test_sparse_input(self):
    input_array = np.array([[1, 2, 3, 0], [0, 3, 1, 0]], dtype=np.int64)
    sparse_tensor_data = sparse_ops.from_dense(input_array)

    # pyformat: disable
    expected_output = [[0, 1, 1, 1, 0, 0],
                       [0, 1, 0, 1, 0, 0]]
    # pyformat: enable
    max_tokens = 6
    expected_output_shape = [None, max_tokens]

    input_data = keras.Input(shape=(None,), dtype=dtypes.int64, sparse=True)

    layer = get_layer_class()(
        max_tokens=max_tokens, output_mode=categorical_encoding.BINARY)
    int_data = layer(input_data)
    self.assertAllEqual(expected_output_shape, int_data.shape.as_list())

    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(sparse_tensor_data, steps=1)
    self.assertAllEqual(expected_output, output_dataset)

  def test_sparse_input_sparse_output(self):
    sp_inp = sparse_tensor.SparseTensor(
        indices=[[0, 0], [1, 1], [2, 0], [2, 1], [3, 1]],
        values=[0, 2, 1, 1, 0],
        dense_shape=[4, 2])
    input_data = keras.Input(shape=(None,), dtype=dtypes.int64, sparse=True)

    # The expected output should be (X for missing value):
    # [[1, X, X, X]
    #  [X, X, 1, X]
    #  [X, 2, X, X]
    #  [1, X, X, X]]
    expected_indices = [[0, 0], [1, 2], [2, 1], [3, 0]]
    expected_values = [1, 1, 2, 1]
    max_tokens = 6

    layer = get_layer_class()(
        max_tokens=max_tokens,
        output_mode=categorical_encoding.COUNT,
        sparse=True)
    int_data = layer(input_data)

    model = keras.Model(inputs=input_data, outputs=int_data)
    sp_output_dataset = model.predict(sp_inp, steps=1)
    self.assertAllEqual(expected_values, sp_output_dataset.values)
    self.assertAllEqual(expected_indices, sp_output_dataset.indices)

    # Assert sparse output is same as dense output.
    layer = get_layer_class()(
        max_tokens=max_tokens,
        output_mode=categorical_encoding.COUNT,
        sparse=False)
    int_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(sp_inp, steps=1)
    self.assertAllEqual(
        sparse_ops.sparse_tensor_to_dense(sp_output_dataset, default_value=0),
        output_dataset)

  def test_ragged_input(self):
    input_array = ragged_factory_ops.constant([[1, 2, 3], [3, 1]])

    # pyformat: disable
    expected_output = [[0, 1, 1, 1, 0, 0],
                       [0, 1, 0, 1, 0, 0]]
    # pyformat: enable
    max_tokens = 6
    expected_output_shape = [None, max_tokens]

    input_data = keras.Input(shape=(None,), dtype=dtypes.int32, ragged=True)

    layer = get_layer_class()(
        max_tokens=max_tokens, output_mode=categorical_encoding.BINARY)
    int_data = layer(input_data)

    self.assertAllEqual(expected_output_shape, int_data.shape.as_list())

    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_array, steps=1)
    self.assertAllEqual(expected_output, output_dataset)

  def test_ragged_input_sparse_output(self):
    input_array = ragged_factory_ops.constant([[1, 2, 3], [3, 3]])

    # The expected output should be (X for missing value):
    # [[X, 1, 1, 1]
    #  [X, X, X, 2]]
    expected_indices = [[0, 1], [0, 2], [0, 3], [1, 3]]
    expected_values = [1, 1, 1, 2]
    max_tokens = 6

    input_data = keras.Input(shape=(None,), dtype=dtypes.int32, ragged=True)
    layer = get_layer_class()(
        max_tokens=max_tokens,
        output_mode=categorical_encoding.COUNT,
        sparse=True)
    int_data = layer(input_data)

    model = keras.Model(inputs=input_data, outputs=int_data)
    sp_output_dataset = model.predict(input_array, steps=1)
    self.assertAllEqual(expected_values, sp_output_dataset.values)
    self.assertAllEqual(expected_indices, sp_output_dataset.indices)

    # Assert sparse output is same as dense output.
    layer = get_layer_class()(
        max_tokens=max_tokens,
        output_mode=categorical_encoding.COUNT,
        sparse=False)
    int_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_array, steps=1)
    self.assertAllEqual(
        sparse_ops.sparse_tensor_to_dense(sp_output_dataset, default_value=0),
        output_dataset)

  # Keras functional model doesn't support dense layer stacked with sparse out.
  def DISABLED_test_sparse_output_and_dense_layer(self):
    input_array = constant_op.constant([[1, 2, 3], [3, 3, 0]])

    max_tokens = 4

    input_data = keras.Input(shape=(None,), dtype=dtypes.int32)
    encoding_layer = get_layer_class()(
        max_tokens=max_tokens,
        output_mode=categorical_encoding.COUNT,
        sparse=True)
    int_data = encoding_layer(input_data)
    output_data = math_ops.cast(int_data, dtypes.float32)
    weights = variables.Variable([[.1], [.2], [.3], [.4]], dtype=dtypes.float32)
    weights_mult = lambda x: sparse_ops.sparse_tensor_dense_matmul(x, weights)
    output_data = keras.layers.Lambda(weights_mult)(output_data)

    model = keras.Model(inputs=input_data, outputs=output_data)
    _ = model.predict(input_array, steps=1)


@keras_parameterized.run_all_keras_modes
class CategoricalEncodingAdaptTest(
    keras_parameterized.TestCase,
    preprocessing_test_utils.PreprocessingLayerTest):

  def test_sparse_adapt(self):
    vocab_data = sparse_ops.from_dense(
        np.array([[1, 1, 0, 1, 1, 2, 2, 0, 2, 3, 3, 0, 4]], dtype=np.int64))
    vocab_dataset = dataset_ops.Dataset.from_tensors(vocab_data)
    input_array = sparse_ops.from_dense(
        np.array([[1, 2, 3, 0], [0, 3, 1, 0]], dtype=np.int64))

    # pyformat: disable
    expected_output = [[0, 1, 1, 1, 0],
                       [0, 1, 0, 1, 0]]
    # pyformat: enable
    max_tokens = 5
    expected_output_shape = [None, max_tokens]

    input_data = keras.Input(shape=(None,), dtype=dtypes.int64, sparse=True)
    layer = get_layer_class()(
        max_tokens=None, output_mode=categorical_encoding.BINARY)
    layer.adapt(vocab_dataset)
    int_data = layer(input_data)
    self.assertAllEqual(expected_output_shape, int_data.shape.as_list())

    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_array, steps=1)
    self.assertAllEqual(expected_output, output_dataset)

  def test_ragged_adapt(self):
    vocab_data = ragged_factory_ops.constant(
        np.array([[1, 1, 0, 1, 1], [2, 2], [0, 2, 3], [0, 4]]))
    vocab_dataset = dataset_ops.Dataset.from_tensors(vocab_data)
    input_array = ragged_factory_ops.constant([[1, 2, 3], [3, 1]])

    # pyformat: disable
    expected_output = [[0, 1, 1, 1, 0],
                       [0, 1, 0, 1, 0]]
    # pyformat: enable
    max_tokens = 5
    expected_output_shape = [None, max_tokens]

    input_data = keras.Input(shape=(None,), dtype=dtypes.int32, ragged=True)

    layer = get_layer_class()(
        max_tokens=None, output_mode=categorical_encoding.BINARY)
    layer.adapt(vocab_dataset)
    int_data = layer(input_data)

    self.assertAllEqual(expected_output_shape, int_data.shape.as_list())

    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_array, steps=1)
    self.assertAllEqual(expected_output, output_dataset)

  def test_adapt_after_build(self):
    vocab_data = np.array([[1, 1, 1, 1, 2, 2, 2, 3, 3, 4]])
    input_array = np.array([[1, 2, 3, 1], [0, 3, 1, 0]])

    # pyformat: disable
    expected_output = [[0, 1, 1, 1, 0],
                       [1, 1, 0, 1, 0]]
    # pyformat: enable
    max_tokens = 5
    expected_output_shape = [None, max_tokens]

    input_data = keras.Input(shape=(None,), dtype=dtypes.int32)
    layer = get_layer_class()(
        max_tokens=max_tokens, output_mode=categorical_encoding.BINARY)
    int_data = layer(input_data)
    layer.adapt(vocab_data)
    self.assertAllEqual(expected_output_shape, int_data.shape.as_list())

    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_array)
    self.assertAllEqual(expected_output, output_dataset)

  def test_hard_maximum_set_state_variables_after_build(self):
    state_variables = {categorical_encoding._NUM_ELEMENTS_NAME: 5}
    input_array = np.array([[1, 2, 3, 1], [0, 3, 1, 0]])

    # pyformat: disable
    expected_output = [[0, 1, 1, 1, 0],
                       [1, 1, 0, 1, 0]]
    # pyformat: enable
    max_tokens = 5
    expected_output_shape = [None, max_tokens]

    input_data = keras.Input(shape=(None,), dtype=dtypes.int32)
    layer = get_layer_class()(
        max_tokens=max_tokens, output_mode=categorical_encoding.BINARY)
    int_data = layer(input_data)
    layer._set_state_variables(state_variables)
    self.assertAllEqual(expected_output_shape, int_data.shape.as_list())

    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_array)
    self.assertAllEqual(expected_output, output_dataset)

  def test_soft_maximum_set_state_after_build(self):
    input_array = np.array([[1, 2, 3, 1], [0, 3, 1, 0]])

    # pyformat: disable
    expected_output = [[0, 1, 1, 1, 0],
                       [1, 1, 0, 1, 0]]
    # pyformat: enable
    max_tokens = 5
    expected_output_shape = [None, max_tokens]

    input_data = keras.Input(shape=(None,), dtype=dtypes.int32)
    layer = get_layer_class()(
        max_tokens=None, output_mode=categorical_encoding.BINARY)
    layer.build(input_data.shape)
    layer.set_num_elements(max_tokens)
    int_data = layer(input_data)
    self.assertAllEqual(expected_output_shape, int_data.shape.as_list())

    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_array)
    self.assertAllEqual(expected_output, output_dataset)

  def test_set_weights_fails_on_wrong_size_weights(self):
    tfidf_data = [.05, .5, .25, .2, .125]
    layer = get_layer_class()(
        max_tokens=6, output_mode=categorical_encoding.TFIDF)

    with self.assertRaisesRegex(ValueError, ".*Layer weight shape.*"):
      layer.set_weights([np.array(tfidf_data)])

  def test_set_num_elements_after_call_fails(self):
    input_data = keras.Input(shape=(None,), dtype=dtypes.int32)
    layer = get_layer_class()(
        max_tokens=None, output_mode=categorical_encoding.BINARY)
    _ = layer(input_data)
    with self.assertRaisesRegex(RuntimeError, "num_elements cannot be changed"):
      layer.set_num_elements(5)

  def test_adapt_after_call_fails(self):
    vocab_data = np.array([1, 1, 1, 1, 2, 2, 2, 3, 3, 4])

    input_data = keras.Input(shape=(None,), dtype=dtypes.int32)
    layer = get_layer_class()(
        max_tokens=None, output_mode=categorical_encoding.BINARY)
    _ = layer(input_data)
    with self.assertRaisesRegex(RuntimeError, "can't be adapted"):
      layer.adapt(vocab_data)

  def test_set_state_variables_after_call_fails(self):
    state_variables = {categorical_encoding._NUM_ELEMENTS_NAME: 5}

    input_data = keras.Input(shape=(None,), dtype=dtypes.int32)
    layer = get_layer_class()(
        max_tokens=None, output_mode=categorical_encoding.BINARY)
    _ = layer(input_data)
    with self.assertRaisesRegex(RuntimeError, "num_elements cannot be changed"):
      layer._set_state_variables(state_variables)


@keras_parameterized.run_all_keras_modes
@keras_parameterized.run_all_keras_modes
class CategoricalEncodingOutputTest(
    keras_parameterized.TestCase,
    preprocessing_test_utils.PreprocessingLayerTest):

  def test_binary_output_hard_maximum(self):
    input_array = np.array([[1, 2, 3, 1], [0, 3, 1, 0]])

    # pyformat: disable
    expected_output = [[0, 1, 1, 1, 0, 0],
                       [1, 1, 0, 1, 0, 0]]
    # pyformat: enable
    max_tokens = 6
    expected_output_shape = [None, max_tokens]

    input_data = keras.Input(shape=(None,), dtype=dtypes.int32)
    layer = get_layer_class()(
        max_tokens=max_tokens, output_mode=categorical_encoding.BINARY)
    int_data = layer(input_data)
    self.assertAllEqual(expected_output_shape, int_data.shape.as_list())

    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_array)
    self.assertAllEqual(expected_output, output_dataset)

  def test_binary_output_soft_maximum(self):
    input_array = np.array([[1, 2, 3, 1], [0, 3, 1, 0]])

    # pyformat: disable
    expected_output = [[0, 1, 1, 1, 0],
                       [1, 1, 0, 1, 0]]
    # pyformat: enable
    max_tokens = 5
    expected_output_shape = [None, max_tokens]

    input_data = keras.Input(shape=(None,), dtype=dtypes.int32)
    layer = get_layer_class()(
        max_tokens=None, output_mode=categorical_encoding.BINARY)
    layer.set_weights([np.array(max_tokens)])
    int_data = layer(input_data)
    self.assertAllEqual(expected_output_shape, int_data.shape.as_list())

    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_array)
    self.assertAllEqual(expected_output, output_dataset)

  def test_count_output_hard_maximum(self):
    input_array = np.array([[1, 2, 3, 1], [0, 3, 1, 0]])

    # pyformat: disable
    expected_output = [[0, 2, 1, 1, 0, 0],
                       [2, 1, 0, 1, 0, 0]]
    # pyformat: enable
    max_tokens = 6
    expected_output_shape = [None, max_tokens]

    input_data = keras.Input(shape=(None,), dtype=dtypes.int32)
    layer = get_layer_class()(
        max_tokens=6, output_mode=categorical_encoding.COUNT)
    int_data = layer(input_data)
    self.assertAllEqual(expected_output_shape, int_data.shape.as_list())

    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_array)
    self.assertAllEqual(expected_output, output_dataset)

  def test_count_output_soft_maximum(self):
    input_array = np.array([[1, 2, 3, 1], [0, 3, 1, 0]])

    # pyformat: disable
    expected_output = [[0, 2, 1, 1, 0],
                       [2, 1, 0, 1, 0]]
    # pyformat: enable
    max_tokens = 5
    expected_output_shape = [None, max_tokens]

    input_data = keras.Input(shape=(None,), dtype=dtypes.int32)
    layer = get_layer_class()(
        max_tokens=None, output_mode=categorical_encoding.COUNT)
    layer.set_weights([np.array(max_tokens)])
    int_data = layer(input_data)
    self.assertAllEqual(expected_output_shape, int_data.shape.as_list())

    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_array)
    self.assertAllEqual(expected_output, output_dataset)

  def test_tfidf_output_hard_maximum(self):
    tfidf_data = [.05, .5, .25, .2, .125]
    input_array = np.array([[1, 2, 3, 1], [0, 4, 1, 0]])

    # pyformat: disable
    # pylint: disable=bad-whitespace
    expected_output = [[ 0,  1, .25, .2,    0, 0],
                       [.1, .5,   0,  0, .125, 0]]
    # pylint: enable=bad-whitespace
    # pyformat: enable
    max_tokens = 6
    expected_output_shape = [None, max_tokens]

    input_data = keras.Input(shape=(None,), dtype=dtypes.int32)
    layer = get_layer_class()(
        max_tokens=6, output_mode=categorical_encoding.TFIDF)
    layer.set_tfidf_data(tfidf_data)
    int_data = layer(input_data)
    self.assertAllEqual(expected_output_shape, int_data.shape.as_list())

    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_array)
    self.assertAllClose(expected_output, output_dataset)

  def test_tfidf_output_soft_maximum(self):
    tfidf_data = [.05, .5, .25, .2, .125]
    input_array = np.array([[1, 2, 3, 1], [0, 4, 1, 0]])

    # pyformat: disable
    # pylint: disable=bad-whitespace
    expected_output = [[ 0,  1, .25, .2,    0],
                       [.1, .5,   0,  0, .125]]
    # pylint: enable=bad-whitespace
    # pyformat: enable
    max_tokens = 5
    expected_output_shape = [None, max_tokens]

    input_data = keras.Input(shape=(None,), dtype=dtypes.int32)
    layer = get_layer_class()(
        max_tokens=None, output_mode=categorical_encoding.TFIDF)
    layer.set_num_elements(max_tokens)
    layer.set_tfidf_data(tfidf_data)
    int_data = layer(input_data)
    self.assertAllEqual(expected_output_shape, int_data.shape.as_list())

    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_array)
    self.assertAllClose(expected_output, output_dataset)


class CategoricalEncodingModelBuildingTest(
    keras_parameterized.TestCase,
    preprocessing_test_utils.PreprocessingLayerTest):

  @parameterized.named_parameters(
      {
          "testcase_name": "count_hard_max",
          "max_tokens": 5,
          "output_mode": categorical_encoding.COUNT
      }, {
          "testcase_name": "count_soft_max",
          "max_tokens": None,
          "output_mode": categorical_encoding.COUNT
      }, {
          "testcase_name": "binary_hard_max",
          "max_tokens": 5,
          "output_mode": categorical_encoding.BINARY
      }, {
          "testcase_name": "binary_soft_max",
          "max_tokens": None,
          "output_mode": categorical_encoding.BINARY
      }, {
          "testcase_name": "tfidf_hard_max",
          "max_tokens": 5,
          "output_mode": categorical_encoding.TFIDF
      }, {
          "testcase_name": "tfidf_soft_max",
          "max_tokens": None,
          "output_mode": categorical_encoding.TFIDF
      })
  def test_end_to_end_bagged_modeling(self, output_mode, max_tokens):
    tfidf_data = np.array([.03, .5, .25, .2, .125])
    input_array = np.array([[1, 2, 3, 1], [0, 3, 1, 0]])

    input_data = keras.Input(shape=(None,), dtype=dtypes.int32)
    layer = get_layer_class()(max_tokens=max_tokens, output_mode=output_mode)

    weights = []
    if max_tokens is None:
      weights.append(np.array(5))
    if output_mode == categorical_encoding.TFIDF:
      weights.append(tfidf_data)

    layer.set_weights(weights)

    int_data = layer(input_data)
    float_data = backend.cast(int_data, dtype="float32")
    output_data = core.Dense(64)(float_data)
    model = keras.Model(inputs=input_data, outputs=output_data)
    _ = model.predict(input_array)


@keras_parameterized.run_all_keras_modes
class CategoricalEncodingCombinerTest(
    keras_parameterized.TestCase,
    preprocessing_test_utils.PreprocessingLayerTest):

  def compare_idf_accumulators(self, a, b, msg=None):
    if a is None or b is None:
      self.assertAllEqual(a, b, msg=msg)

    self.assertAllEqual(a.data, b.data, msg=msg)

    if a.per_doc_count_dict is not None:

      def per_doc_counts(accumulator):
        count_values = [
            count_dict["count"]
            for count_dict in accumulator.per_doc_count_dict.values()
        ]
        return dict(zip(accumulator.per_doc_count_dict.keys(), count_values))

      self.assertAllEqual(per_doc_counts(a), per_doc_counts(b), msg=msg)

  compare_accumulators = compare_idf_accumulators

  def update_accumulator(self, accumulator, data):
    accumulator.data[1] = data["num_documents"]
    accumulator.data[0] = data["max_element"]

    if "document_counts" in data:
      create_dict = lambda x: {"count": x, "last_doc_id": -1}
      idf_dict = {}
      for i, count in enumerate(data["document_counts"]):
        if count > 0:
          idf_dict[i] = create_dict(count)

      accumulator.per_doc_count_dict.update(idf_dict)

    return accumulator

  def test_combiner_api_compatibility_int_mode(self):
    data = np.array([[1, 2, 3, 4], [1, 2, 3, 0]])
    combiner = categorical_encoding._CategoricalEncodingCombiner(
        compute_idf=False)
    expected_accumulator_output = {
        "max_element": np.array(4),
        "num_documents": np.array(2),
    }
    expected_extract_output = {
        "num_elements": np.array(5),
    }
    expected_accumulator = combiner._create_accumulator()
    expected_accumulator = self.update_accumulator(expected_accumulator,
                                                   expected_accumulator_output)
    self.validate_accumulator_serialize_and_deserialize(combiner, data,
                                                        expected_accumulator)
    self.validate_accumulator_uniqueness(combiner, data)
    self.validate_accumulator_extract(combiner, data, expected_extract_output)

  def test_combiner_api_compatibility_tfidf_mode(self):
    data = np.array([[1, 2, 3, 4], [1, 2, 3, 0]])
    combiner = categorical_encoding._CategoricalEncodingCombiner(
        compute_idf=True)
    expected_accumulator_output = {
        "max_element": np.array(4),
        "document_counts": np.array([1, 2, 2, 2, 1]),
        "num_documents": np.array(2),
    }
    expected_extract_output = {
        "num_elements": np.array(5),
        "idf": np.array([0.693147, 0.510826, 0.510826, 0.510826, 0.693147]),
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
          "testcase_name": "no_top_k",
          "data": np.array([[1, 2], [4, 2], [3], [4, 2]]),
          "expected_accumulator_output": {
              "max_element": np.array(4),
              "document_counts": np.array([0, 1, 3, 1, 2]),
              "num_documents": np.array(4),
          },
          "expected_extract_output": {
              "num_elements":
                  np.array(5),
              "idf":
                  np.array([1.609438, 1.098612, 0.693147, 1.098612, 0.847298]),
          },
      }, {
          "testcase_name": "single_element_per_row",
          "data": np.array([[1], [2], [4], [2], [3]]),
          "expected_accumulator_output": {
              "max_element": np.array(4),
              "document_counts": np.array([0, 1, 2, 1, 1]),
              "num_documents": np.array(5),
          },
          "expected_extract_output": {
              "num_elements":
                  np.array(5),
              "idf":
                  np.array([1.791759, 1.252763, 0.980829, 1.252763, 1.252763]),
          },
      })
  def test_combiner_computation(self,
                                data,
                                expected_accumulator_output,
                                expected_extract_output,
                                compute_idf=True):
    combiner = categorical_encoding._CategoricalEncodingCombiner(
        compute_idf=compute_idf)
    expected_accumulator = combiner._create_accumulator()
    expected_accumulator = self.update_accumulator(expected_accumulator,
                                                   expected_accumulator_output)
    self.validate_accumulator_computation(combiner, data, expected_accumulator)
    self.validate_accumulator_extract(combiner, data, expected_extract_output)



if __name__ == "__main__":
  test.main()
