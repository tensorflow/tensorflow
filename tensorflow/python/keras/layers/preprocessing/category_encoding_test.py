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
"""Tests for Keras text category_encoding preprocessing layer."""

from absl.testing import parameterized
import numpy as np

from tensorflow.python import keras

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.keras import backend
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras.layers import core
from tensorflow.python.keras.layers.preprocessing import category_encoding
from tensorflow.python.keras.layers.preprocessing import preprocessing_test_utils
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import test


@keras_parameterized.run_all_keras_modes(always_skip_v1=True)
class CategoryEncodingInputTest(keras_parameterized.TestCase,
                                preprocessing_test_utils.PreprocessingLayerTest
                               ):

  def test_dense_input_sparse_output(self):
    input_array = constant_op.constant([[1, 2, 3], [3, 3, 0]])

    # The expected output should be (X for missing value):
    # [[X, 1, 1, 1, X, X]
    #  [1, X, X, 2, X, X]]
    expected_indices = [[0, 1], [0, 2], [0, 3], [1, 0], [1, 3]]
    expected_values = [1, 1, 1, 1, 2]
    num_tokens = 6

    input_data = keras.Input(shape=(None,), dtype=dtypes.int32)
    layer = category_encoding.CategoryEncoding(
        num_tokens=num_tokens, output_mode=category_encoding.COUNT, sparse=True)
    int_data = layer(input_data)

    model = keras.Model(inputs=input_data, outputs=int_data)
    sp_output_dataset = model.predict(input_array, steps=1)
    self.assertAllEqual(expected_values, sp_output_dataset.values)
    self.assertAllEqual(expected_indices, sp_output_dataset.indices)

    # Assert sparse output is same as dense output.
    layer = category_encoding.CategoryEncoding(
        num_tokens=num_tokens,
        output_mode=category_encoding.COUNT,
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
    num_tokens = 6
    expected_output_shape = [None, num_tokens]

    input_data = keras.Input(shape=(None,), dtype=dtypes.int64, sparse=True)

    layer = category_encoding.CategoryEncoding(
        num_tokens=num_tokens, output_mode=category_encoding.MULTI_HOT)
    int_data = layer(input_data)
    self.assertAllEqual(expected_output_shape, int_data.shape.as_list())

    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(sparse_tensor_data, steps=1)
    self.assertAllEqual(expected_output, output_dataset)

  def test_sparse_input_with_weights(self):
    input_array = np.array([[1, 2, 3, 4], [4, 3, 1, 4]], dtype=np.int64)
    weights_array = np.array([[.1, .2, .3, .4], [.2, .1, .4, .3]])
    sparse_tensor_data = sparse_ops.from_dense(input_array)
    sparse_weight_data = sparse_ops.from_dense(weights_array)

    # pyformat: disable
    expected_output = [[0, .1, .2, .3, .4, 0],
                       [0, .4, 0, .1, .5, 0]]
    # pyformat: enable
    num_tokens = 6
    expected_output_shape = [None, num_tokens]

    input_data = keras.Input(shape=(None,), dtype=dtypes.int64, sparse=True)
    weight_data = keras.Input(shape=(None,), dtype=dtypes.float32, sparse=True)

    layer = category_encoding.CategoryEncoding(
        num_tokens=num_tokens, output_mode=category_encoding.COUNT)
    int_data = layer(input_data, count_weights=weight_data)
    self.assertAllEqual(expected_output_shape, int_data.shape.as_list())

    model = keras.Model(inputs=[input_data, weight_data], outputs=int_data)
    output_dataset = model.predict([sparse_tensor_data, sparse_weight_data],
                                   steps=1)
    self.assertAllClose(expected_output, output_dataset)

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
    num_tokens = 6

    layer = category_encoding.CategoryEncoding(
        num_tokens=num_tokens, output_mode=category_encoding.COUNT, sparse=True)
    int_data = layer(input_data)

    model = keras.Model(inputs=input_data, outputs=int_data)
    sp_output_dataset = model.predict(sp_inp, steps=1)
    self.assertAllEqual(expected_values, sp_output_dataset.values)
    self.assertAllEqual(expected_indices, sp_output_dataset.indices)

    # Assert sparse output is same as dense output.
    layer = category_encoding.CategoryEncoding(
        num_tokens=num_tokens,
        output_mode=category_encoding.COUNT,
        sparse=False)
    int_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(sp_inp, steps=1)
    self.assertAllEqual(
        sparse_ops.sparse_tensor_to_dense(sp_output_dataset, default_value=0),
        output_dataset)

  def test_sparse_input_sparse_output_with_weights(self):
    indices = [[0, 0], [1, 1], [2, 0], [2, 1], [3, 1]]
    sp_inp = sparse_tensor.SparseTensor(
        indices=indices, values=[0, 2, 1, 1, 0], dense_shape=[4, 2])
    input_data = keras.Input(shape=(None,), dtype=dtypes.int64, sparse=True)
    sp_weight = sparse_tensor.SparseTensor(
        indices=indices, values=[.1, .2, .4, .3, .2], dense_shape=[4, 2])
    weight_data = keras.Input(shape=(None,), dtype=dtypes.float32, sparse=True)

    # The expected output should be (X for missing value):
    # [[1, X, X, X]
    #  [X, X, 1, X]
    #  [X, 2, X, X]
    #  [1, X, X, X]]
    expected_indices = [[0, 0], [1, 2], [2, 1], [3, 0]]
    expected_values = [.1, .2, .7, .2]
    num_tokens = 6

    layer = category_encoding.CategoryEncoding(
        num_tokens=num_tokens, output_mode=category_encoding.COUNT, sparse=True)
    int_data = layer(input_data, count_weights=weight_data)

    model = keras.Model(inputs=[input_data, weight_data], outputs=int_data)
    sp_output_dataset = model.predict([sp_inp, sp_weight], steps=1)
    self.assertAllClose(expected_values, sp_output_dataset.values)
    self.assertAllEqual(expected_indices, sp_output_dataset.indices)

  def test_ragged_input(self):
    input_array = ragged_factory_ops.constant([[1, 2, 3], [3, 1]])

    # pyformat: disable
    expected_output = [[0, 1, 1, 1, 0, 0],
                       [0, 1, 0, 1, 0, 0]]
    # pyformat: enable
    num_tokens = 6
    expected_output_shape = [None, num_tokens]

    input_data = keras.Input(shape=(None,), dtype=dtypes.int32, ragged=True)

    layer = category_encoding.CategoryEncoding(
        num_tokens=num_tokens, output_mode=category_encoding.MULTI_HOT)
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
    num_tokens = 6

    input_data = keras.Input(shape=(None,), dtype=dtypes.int32, ragged=True)
    layer = category_encoding.CategoryEncoding(
        num_tokens=num_tokens, output_mode=category_encoding.COUNT, sparse=True)
    int_data = layer(input_data)

    model = keras.Model(inputs=input_data, outputs=int_data)
    sp_output_dataset = model.predict(input_array, steps=1)
    self.assertAllEqual(expected_values, sp_output_dataset.values)
    self.assertAllEqual(expected_indices, sp_output_dataset.indices)

    # Assert sparse output is same as dense output.
    layer = category_encoding.CategoryEncoding(
        num_tokens=num_tokens,
        output_mode=category_encoding.COUNT,
        sparse=False)
    int_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_array, steps=1)
    self.assertAllEqual(
        sparse_ops.sparse_tensor_to_dense(sp_output_dataset, default_value=0),
        output_dataset)

  def test_sparse_output_and_dense_layer(self):
    input_array = constant_op.constant([[1, 2, 3], [3, 3, 0]])

    num_tokens = 4

    input_data = keras.Input(shape=(None,), dtype=dtypes.int32)
    encoding_layer = category_encoding.CategoryEncoding(
        num_tokens=num_tokens, output_mode=category_encoding.COUNT, sparse=True)
    int_data = encoding_layer(input_data)
    dense_layer = keras.layers.Dense(units=1)
    output_data = dense_layer(int_data)

    model = keras.Model(inputs=input_data, outputs=output_data)
    _ = model.predict(input_array, steps=1)

  def test_dense_oov_input(self):
    valid_array = constant_op.constant([[0, 1, 2], [0, 1, 2]])
    invalid_array = constant_op.constant([[0, 1, 2], [2, 3, 1]])
    num_tokens = 3
    expected_output_shape = [None, num_tokens]
    encoder_layer = category_encoding.CategoryEncoding(num_tokens)
    input_data = keras.Input(shape=(3,), dtype=dtypes.int32)
    int_data = encoder_layer(input_data)
    self.assertAllEqual(expected_output_shape, int_data.shape.as_list())
    model = keras.Model(inputs=input_data, outputs=int_data)
    # Call predict once on valid input to compile a graph and test control flow.
    _ = model.predict(valid_array, steps=1)
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        ".*must be in the range 0 <= values < num_tokens.*"):
      _ = model.predict(invalid_array, steps=1)

  def test_dense_negative(self):
    valid_array = constant_op.constant([[0, 1, 2], [0, 1, 2]])
    invalid_array = constant_op.constant([[1, 2, 0], [2, 2, -1]])
    num_tokens = 3
    expected_output_shape = [None, num_tokens]
    encoder_layer = category_encoding.CategoryEncoding(num_tokens)
    input_data = keras.Input(shape=(3,), dtype=dtypes.int32)
    int_data = encoder_layer(input_data)
    self.assertAllEqual(expected_output_shape, int_data.shape.as_list())
    model = keras.Model(inputs=input_data, outputs=int_data)
    # Call predict once on valid input to compile a graph and test control flow.
    _ = model.predict(valid_array, steps=1)
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        ".*must be in the range 0 <= values < num_tokens.*"):
      _ = model.predict(invalid_array, steps=1)

  def test_legacy_max_tokens_arg(self):
    input_array = np.array([[1, 2, 3, 1]])
    expected_output = [[0, 1, 1, 1, 0, 0]]
    num_tokens = 6
    expected_output_shape = [None, num_tokens]

    input_data = keras.Input(shape=(None,), dtype=dtypes.int32)
    layer = category_encoding.CategoryEncoding(
        max_tokens=num_tokens, output_mode=category_encoding.MULTI_HOT)
    int_data = layer(input_data)
    self.assertAllEqual(expected_output_shape, int_data.shape.as_list())

    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_array)
    self.assertAllEqual(expected_output, output_dataset)


@keras_parameterized.run_all_keras_modes
class CategoryEncodingOutputTest(keras_parameterized.TestCase,
                                 preprocessing_test_utils.PreprocessingLayerTest
                                ):

  def test_one_hot_output(self):
    input_data = np.array([[3], [2], [0], [1]])
    expected_output = [
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ]
    num_tokens = 4
    expected_output_shape = [None, num_tokens]

    layer = category_encoding.CategoryEncoding(
        num_tokens=num_tokens, output_mode=category_encoding.ONE_HOT)
    inputs = keras.Input(shape=(1,), dtype=dtypes.int32)
    outputs = layer(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)
    output_dataset = model(input_data)
    self.assertAllEqual(expected_output_shape, outputs.shape.as_list())
    self.assertAllEqual(expected_output, output_dataset)

  def test_one_hot_output_rank_one_input(self):
    input_data = np.array([3, 2, 0, 1])
    expected_output = [
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ]
    num_tokens = 4
    expected_output_shape = [None, num_tokens]

    # Test call on layer directly.
    layer = category_encoding.CategoryEncoding(
        num_tokens=num_tokens, output_mode=category_encoding.ONE_HOT)
    output_data = layer(input_data)
    self.assertAllEqual(expected_output, output_data)

    # Test call on model.
    inputs = keras.Input(shape=(1,), dtype=dtypes.int32)
    outputs = layer(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)
    output_data = model(input_data)
    self.assertAllEqual(expected_output_shape, outputs.shape.as_list())
    self.assertAllEqual(expected_output, output_data)

  def test_one_hot_output_rank_zero_input(self):
    input_data = np.array(3)
    expected_output = [0, 0, 0, 1]
    num_tokens = 4
    expected_output_shape = [None, num_tokens]

    # Test call on layer directly.
    layer = category_encoding.CategoryEncoding(
        num_tokens=num_tokens, output_mode=category_encoding.ONE_HOT)
    output_data = layer(input_data)
    self.assertAllEqual(expected_output, output_data)

    # Test call on model.
    inputs = keras.Input(shape=(1,), dtype=dtypes.int32)
    outputs = layer(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)
    output_data = model(input_data)

    self.assertAllEqual(expected_output_shape, outputs.shape.as_list())
    self.assertAllEqual(expected_output, output_data)

  def test_one_hot_rank_3_output_fails(self):
    layer = category_encoding.CategoryEncoding(
        num_tokens=4, output_mode=category_encoding.ONE_HOT)
    with self.assertRaisesRegex(ValueError, "only outputs up to rank 2"):
      _ = layer(keras.Input(shape=(4,), dtype=dtypes.int32))
    with self.assertRaisesRegex(ValueError, "only outputs up to rank 2"):
      _ = layer(np.array([[3, 2, 0, 1], [3, 2, 0, 1]]))

  def test_multi_hot_output(self):
    input_data = np.array([[1, 2, 3, 1], [0, 3, 1, 0]])
    expected_output = [
        [0, 1, 1, 1, 0, 0],
        [1, 1, 0, 1, 0, 0],
    ]
    num_tokens = 6
    expected_output_shape = [None, num_tokens]

    layer = category_encoding.CategoryEncoding(
        num_tokens=num_tokens, output_mode=category_encoding.MULTI_HOT)
    inputs = keras.Input(shape=(None,), dtype=dtypes.int32)
    outputs = layer(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)
    output_data = model.predict(input_data)
    self.assertAllEqual(expected_output_shape, outputs.shape.as_list())
    self.assertAllEqual(expected_output, output_data)

  def test_multi_hot_output_rank_one_input(self):
    input_data = np.array([3, 2, 0, 1])
    expected_output = [1, 1, 1, 1, 0, 0]
    num_tokens = 6
    expected_output_shape = [None, num_tokens]

    # Test call on layer directly.
    layer = category_encoding.CategoryEncoding(
        num_tokens=num_tokens, output_mode=category_encoding.MULTI_HOT)
    output_data = layer(input_data)
    self.assertAllEqual(expected_output, output_data)

    # Test call on model.
    inputs = keras.Input(shape=(4,), dtype=dtypes.int32)
    outputs = layer(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)
    output_data = model(input_data)
    self.assertAllEqual(expected_output_shape, outputs.shape.as_list())
    self.assertAllEqual(expected_output, output_data)

  def test_multi_hot_output_rank_zero_input(self):
    input_data = np.array(3)
    expected_output = [0, 0, 0, 1, 0, 0]
    num_tokens = 6
    expected_output_shape = [None, num_tokens]

    # Test call on layer directly.
    layer = category_encoding.CategoryEncoding(
        num_tokens=num_tokens, output_mode=category_encoding.MULTI_HOT)
    output_data = layer(input_data)
    self.assertAllEqual(expected_output, output_data)

    # Test call on model.
    inputs = keras.Input(shape=(4,), dtype=dtypes.int32)
    outputs = layer(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)
    output_data = model(input_data)
    self.assertAllEqual(expected_output_shape, outputs.shape.as_list())
    self.assertAllEqual(expected_output, output_data)

  def test_multi_hot_rank_3_output_fails(self):
    layer = category_encoding.CategoryEncoding(
        num_tokens=4, output_mode=category_encoding.ONE_HOT)
    with self.assertRaisesRegex(ValueError, "only outputs up to rank 2"):
      _ = layer(keras.Input(shape=(3, 4,), dtype=dtypes.int32))
    with self.assertRaisesRegex(ValueError, "only outputs up to rank 2"):
      _ = layer(np.array([[[3, 2, 0, 1], [3, 2, 0, 1]]]))

  def test_count_output(self):
    input_array = np.array([[1, 2, 3, 1], [0, 3, 1, 0]])

    # pyformat: disable
    expected_output = [[0, 2, 1, 1, 0, 0],
                       [2, 1, 0, 1, 0, 0]]
    # pyformat: enable
    num_tokens = 6
    expected_output_shape = [None, num_tokens]

    input_data = keras.Input(shape=(None,), dtype=dtypes.int32)
    layer = category_encoding.CategoryEncoding(
        num_tokens=6, output_mode=category_encoding.COUNT)
    int_data = layer(input_data)
    self.assertAllEqual(expected_output_shape, int_data.shape.as_list())

    model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_array)
    self.assertAllEqual(expected_output, output_dataset)


class CategoryEncodingModelBuildingTest(
    keras_parameterized.TestCase,
    preprocessing_test_utils.PreprocessingLayerTest):

  @parameterized.named_parameters(
      {
          "testcase_name": "count_output",
          "num_tokens": 5,
          "output_mode": category_encoding.COUNT
      }, {
          "testcase_name": "multi_hot_output",
          "num_tokens": 5,
          "output_mode": category_encoding.MULTI_HOT
      })
  def test_end_to_end_bagged_modeling(self, output_mode, num_tokens):
    input_array = np.array([[1, 2, 3, 1], [0, 3, 1, 0]])

    input_data = keras.Input(shape=(None,), dtype=dtypes.int32)
    layer = category_encoding.CategoryEncoding(
        num_tokens=num_tokens, output_mode=output_mode)

    weights = []
    if num_tokens is None:
      layer.set_num_elements(5)
    layer.set_weights(weights)

    int_data = layer(input_data)
    float_data = backend.cast(int_data, dtype="float32")
    output_data = core.Dense(64)(float_data)
    model = keras.Model(inputs=input_data, outputs=output_data)
    _ = model.predict(input_array)


if __name__ == "__main__":
  test.main()
