# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np

from tensorflow.python import keras
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.keras import backend
from tensorflow.python.keras import combinations
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import test


@combinations.generate(combinations.combine(mode=['graph', 'eager']))
class TrackableWeightHandlerTest(keras_parameterized.TestCase):

  def get_table_handler(self):
    # Note: There is some repetition in these tests' setup. However, Tensorflow
    # does not play nicely with a separate setUp() call (causing errors related
    # to graph building), so we have to use a called setup instead of a setUp()
    # call.
    table = lookup_ops.MutableHashTable(
        key_dtype=dtypes.string, value_dtype=dtypes.int32, default_value=0)
    return base_layer_utils.TrackableWeightHandler(table)

  def test_get_num_tensors(self):
    table_handler = self.get_table_handler()
    self.assertEqual(2, table_handler.num_tensors)

  def test_get_and_set_weights(self):
    table_handler = self.get_table_handler()

    table_data = {b'a': 1, b'b': 2, b'c': 3}
    table_handler.set_weights(
        [list(table_data.keys()),
         list(table_data.values())])
    weights = backend.batch_get_value(table_handler.get_tensors())
    weight_data = {key: value for key, value in zip(weights[0], weights[1])}
    self.assertDictEqual(table_data, weight_data)

  def test_get_and_set_weights_does_not_add_ops(self):
    table_handler = self.get_table_handler()
    table_data = {b'a': 1, b'b': 2, b'c': 3}
    table_handler.set_weights(
        [list(table_data.keys()),
         list(table_data.values())])
    _ = backend.batch_get_value(table_handler.get_tensors())
    backend.get_session().graph.finalize()
    table_handler.set_weights(
        [list(table_data.keys()),
         list(table_data.values())])
    _ = backend.batch_get_value(table_handler.get_tensors())


@combinations.generate(combinations.combine(mode=['eager']))
class OpLayerTest(keras_parameterized.TestCase):

  def test_tensor_op_layer(self):
    int_values = keras.Input(shape=(2,), dtype=dtypes.int32)
    float_values = math_ops.cast(int_values, dtypes.float32)
    model = keras.Model(int_values, float_values)
    model.compile(loss='mse')

    input_data = np.array([[1, 2], [3, 4]], dtype=np.int32)
    expected = [[1.0, 2.0], [3.0, 4.0]]
    output = model.predict(input_data)
    self.assertAllClose(expected, output)

  def test_ragged_op_layer_keras_tensors(self):
    int_values = keras.Input(shape=(None,), dtype=dtypes.int32, ragged=True)
    float_values = math_ops.cast(int_values, dtypes.float32)
    model = keras.Model(int_values, float_values)
    model.compile(loss='mse')

    input_data = ragged_factory_ops.constant(
        [[1, 2], [3, 4]], dtype=np.int32)
    expected = [[1.0, 2.0], [3.0, 4.0]]
    output = model.predict(input_data)
    self.assertIsInstance(output, ragged_tensor.RaggedTensor)
    self.assertAllClose(expected, output)

  def test_sparse_op_layer_keras_tensors(self):
    int_values = keras.Input(shape=(None,), dtype=dtypes.int32, sparse=True)
    float_values = math_ops.cast(int_values, dtypes.float32)
    _ = keras.Model(int_values, float_values)
    model = keras.Model(int_values, float_values)
    model.compile(loss='mse')

    input_data = sparse_ops.from_dense(
        np.array([[1, 2], [3, 4]], dtype=np.int32))
    expected = [[1.0, 2.0], [3.0, 4.0]]
    output = model.predict(input_data)
    self.assertIsInstance(output, sparse_tensor.SparseTensor)
    self.assertAllClose(expected, sparse_ops.sparse_tensor_to_dense(output))


if __name__ == '__main__':
  test.main()
