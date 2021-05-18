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
"""Tests for hashing layer."""

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.engine import input_layer
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.layers.preprocessing import hashing
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import test


@keras_parameterized.run_all_keras_modes(always_skip_v1=True)
class HashingTest(keras_parameterized.TestCase):

  def test_hash_single_bin(self):
    layer = hashing.Hashing(num_bins=1)
    inp = np.asarray([['A'], ['B'], ['C'], ['D'], ['E']])
    output = layer(inp)
    self.assertAllClose([[0], [0], [0], [0], [0]], output)

  def test_hash_dense_input_farmhash(self):
    layer = hashing.Hashing(num_bins=2)
    inp = np.asarray([['omar'], ['stringer'], ['marlo'], ['wire'],
                      ['skywalker']])
    output = layer(inp)
    # Assert equal for hashed output that should be true on all platforms.
    self.assertAllClose([[0], [0], [1], [0], [0]], output)

  def test_hash_dense_input_mask_value_farmhash(self):
    empty_mask_layer = hashing.Hashing(num_bins=3, mask_value='')
    omar_mask_layer = hashing.Hashing(num_bins=3, mask_value='omar')
    inp = np.asarray([['omar'], ['stringer'], ['marlo'], ['wire'],
                      ['skywalker']])
    empty_mask_output = empty_mask_layer(inp)
    omar_mask_output = omar_mask_layer(inp)
    # Outputs should be one more than test_hash_dense_input_farmhash (the zeroth
    # bin is now reserved for masks).
    self.assertAllClose([[1], [1], [2], [1], [1]], empty_mask_output)
    # 'omar' should map to 0.
    self.assertAllClose([[0], [1], [2], [1], [1]], omar_mask_output)

  def test_hash_dense_list_input_farmhash(self):
    layer = hashing.Hashing(num_bins=2)
    inp = [['omar'], ['stringer'], ['marlo'], ['wire'], ['skywalker']]
    output = layer(inp)
    # Assert equal for hashed output that should be true on all platforms.
    self.assertAllClose([[0], [0], [1], [0], [0]], output)

    inp = ['omar', 'stringer', 'marlo', 'wire', 'skywalker']
    output = layer(inp)
    # Assert equal for hashed output that should be true on all platforms.
    self.assertAllClose([0, 0, 1, 0, 0], output)

  def test_hash_dense_int_input_farmhash(self):
    layer = hashing.Hashing(num_bins=3)
    inp = np.asarray([[0], [1], [2], [3], [4]])
    output = layer(inp)
    # Assert equal for hashed output that should be true on all platforms.
    self.assertAllClose([[1], [0], [1], [0], [2]], output)

  def test_hash_dense_input_siphash(self):
    layer = hashing.Hashing(num_bins=2, salt=[133, 137])
    inp = np.asarray([['omar'], ['stringer'], ['marlo'], ['wire'],
                      ['skywalker']])
    output = layer(inp)
    # Assert equal for hashed output that should be true on all platforms.
    # Note the result is different from FarmHash.
    self.assertAllClose([[0], [1], [0], [1], [0]], output)

    layer_2 = hashing.Hashing(num_bins=2, salt=[211, 137])
    output_2 = layer_2(inp)
    # Note the result is different from (133, 137).
    self.assertAllClose([[1], [0], [1], [0], [1]], output_2)

  def test_hash_dense_int_input_siphash(self):
    layer = hashing.Hashing(num_bins=3, salt=[133, 137])
    inp = np.asarray([[0], [1], [2], [3], [4]])
    output = layer(inp)
    # Assert equal for hashed output that should be true on all platforms.
    self.assertAllClose([[1], [1], [2], [0], [1]], output)

  def test_hash_sparse_input_farmhash(self):
    layer = hashing.Hashing(num_bins=2)
    indices = [[0, 0], [1, 0], [1, 1], [2, 0], [2, 1]]
    inp = sparse_tensor.SparseTensor(
        indices=indices,
        values=['omar', 'stringer', 'marlo', 'wire', 'skywalker'],
        dense_shape=[3, 2])
    output = layer(inp)
    self.assertAllClose(indices, output.indices)
    self.assertAllClose([0, 0, 1, 0, 0], output.values)

  def test_hash_sparse_input_mask_value_farmhash(self):
    empty_mask_layer = hashing.Hashing(num_bins=3, mask_value='')
    omar_mask_layer = hashing.Hashing(num_bins=3, mask_value='omar')
    indices = [[0, 0], [1, 0], [1, 1], [2, 0], [2, 1]]
    inp = sparse_tensor.SparseTensor(
        indices=indices,
        values=['omar', 'stringer', 'marlo', 'wire', 'skywalker'],
        dense_shape=[3, 2])
    empty_mask_output = empty_mask_layer(inp)
    omar_mask_output = omar_mask_layer(inp)
    self.assertAllClose(indices, omar_mask_output.indices)
    self.assertAllClose(indices, empty_mask_output.indices)
    # Outputs should be one more than test_hash_sparse_input_farmhash (the
    # zeroth bin is now reserved for masks).
    self.assertAllClose([1, 1, 2, 1, 1], empty_mask_output.values)
    # 'omar' should map to 0.
    self.assertAllClose([0, 1, 2, 1, 1], omar_mask_output.values)

  def test_hash_sparse_int_input_farmhash(self):
    layer = hashing.Hashing(num_bins=3)
    indices = [[0, 0], [1, 0], [1, 1], [2, 0], [2, 1]]
    inp = sparse_tensor.SparseTensor(
        indices=indices, values=[0, 1, 2, 3, 4], dense_shape=[3, 2])
    output = layer(inp)
    self.assertAllClose(indices, output.indices)
    self.assertAllClose([1, 0, 1, 0, 2], output.values)

  def test_hash_sparse_input_siphash(self):
    layer = hashing.Hashing(num_bins=2, salt=[133, 137])
    indices = [[0, 0], [1, 0], [1, 1], [2, 0], [2, 1]]
    inp = sparse_tensor.SparseTensor(
        indices=indices,
        values=['omar', 'stringer', 'marlo', 'wire', 'skywalker'],
        dense_shape=[3, 2])
    output = layer(inp)
    self.assertAllClose(output.indices, indices)
    # The result should be same with test_hash_dense_input_siphash.
    self.assertAllClose([0, 1, 0, 1, 0], output.values)

    layer_2 = hashing.Hashing(num_bins=2, salt=[211, 137])
    output = layer_2(inp)
    # The result should be same with test_hash_dense_input_siphash.
    self.assertAllClose([1, 0, 1, 0, 1], output.values)

  def test_hash_sparse_int_input_siphash(self):
    layer = hashing.Hashing(num_bins=3, salt=[133, 137])
    indices = [[0, 0], [1, 0], [1, 1], [2, 0], [2, 1]]
    inp = sparse_tensor.SparseTensor(
        indices=indices, values=[0, 1, 2, 3, 4], dense_shape=[3, 2])
    output = layer(inp)
    self.assertAllClose(indices, output.indices)
    self.assertAllClose([1, 1, 2, 0, 1], output.values)

  def test_hash_ragged_string_input_farmhash(self):
    layer = hashing.Hashing(num_bins=2)
    inp_data = ragged_factory_ops.constant(
        [['omar', 'stringer', 'marlo', 'wire'], ['marlo', 'skywalker', 'wire']],
        dtype=dtypes.string)
    out_data = layer(inp_data)
    # Same hashed output as test_hash_sparse_input_farmhash
    expected_output = [[0, 0, 1, 0], [1, 0, 0]]
    self.assertAllEqual(expected_output, out_data)

    inp_t = input_layer.Input(shape=(None,), ragged=True, dtype=dtypes.string)
    out_t = layer(inp_t)
    model = training.Model(inputs=inp_t, outputs=out_t)
    self.assertAllClose(out_data, model.predict(inp_data))

  def test_hash_ragged_input_mask_value(self):
    empty_mask_layer = hashing.Hashing(num_bins=3, mask_value='')
    omar_mask_layer = hashing.Hashing(num_bins=3, mask_value='omar')
    inp_data = ragged_factory_ops.constant(
        [['omar', 'stringer', 'marlo', 'wire'], ['marlo', 'skywalker', 'wire']],
        dtype=dtypes.string)
    empty_mask_output = empty_mask_layer(inp_data)
    omar_mask_output = omar_mask_layer(inp_data)
    # Outputs should be one more than test_hash_ragged_string_input_farmhash
    # (the zeroth bin is now reserved for masks).
    expected_output = [[1, 1, 2, 1], [2, 1, 1]]
    self.assertAllClose(expected_output, empty_mask_output)
    # 'omar' should map to 0.
    expected_output = [[0, 1, 2, 1], [2, 1, 1]]
    self.assertAllClose(expected_output, omar_mask_output)

  def test_hash_ragged_int_input_farmhash(self):
    layer = hashing.Hashing(num_bins=3)
    inp_data = ragged_factory_ops.constant([[0, 1, 3, 4], [2, 1, 0]],
                                           dtype=dtypes.int64)
    out_data = layer(inp_data)
    # Same hashed output as test_hash_sparse_input_farmhash
    expected_output = [[1, 0, 0, 2], [1, 0, 1]]
    self.assertAllEqual(expected_output, out_data)

    inp_t = input_layer.Input(shape=(None,), ragged=True, dtype=dtypes.int64)
    out_t = layer(inp_t)
    model = training.Model(inputs=inp_t, outputs=out_t)
    self.assertAllClose(out_data, model.predict(inp_data))

  def test_hash_ragged_string_input_siphash(self):
    layer = hashing.Hashing(num_bins=2, salt=[133, 137])
    inp_data = ragged_factory_ops.constant(
        [['omar', 'stringer', 'marlo', 'wire'], ['marlo', 'skywalker', 'wire']],
        dtype=dtypes.string)
    out_data = layer(inp_data)
    # Same hashed output as test_hash_dense_input_siphash
    expected_output = [[0, 1, 0, 1], [0, 0, 1]]
    self.assertAllEqual(expected_output, out_data)

    inp_t = input_layer.Input(shape=(None,), ragged=True, dtype=dtypes.string)
    out_t = layer(inp_t)
    model = training.Model(inputs=inp_t, outputs=out_t)
    self.assertAllClose(out_data, model.predict(inp_data))

    layer_2 = hashing.Hashing(num_bins=2, salt=[211, 137])
    out_data = layer_2(inp_data)
    expected_output = [[1, 0, 1, 0], [1, 1, 0]]
    self.assertAllEqual(expected_output, out_data)

    out_t = layer_2(inp_t)
    model = training.Model(inputs=inp_t, outputs=out_t)
    self.assertAllClose(out_data, model.predict(inp_data))

  def test_hash_ragged_int_input_siphash(self):
    layer = hashing.Hashing(num_bins=3, salt=[133, 137])
    inp_data = ragged_factory_ops.constant([[0, 1, 3, 4], [2, 1, 0]],
                                           dtype=dtypes.int64)
    out_data = layer(inp_data)
    # Same hashed output as test_hash_sparse_input_farmhash
    expected_output = [[1, 1, 0, 1], [2, 1, 1]]
    self.assertAllEqual(expected_output, out_data)

    inp_t = input_layer.Input(shape=(None,), ragged=True, dtype=dtypes.int64)
    out_t = layer(inp_t)
    model = training.Model(inputs=inp_t, outputs=out_t)
    self.assertAllClose(out_data, model.predict(inp_data))

  def test_invalid_inputs(self):
    with self.assertRaisesRegex(ValueError, 'cannot be `None`'):
      _ = hashing.Hashing(num_bins=None)
    with self.assertRaisesRegex(ValueError, 'cannot be `None`'):
      _ = hashing.Hashing(num_bins=-1)
    with self.assertRaisesRegex(ValueError, 'can only be a tuple of size 2'):
      _ = hashing.Hashing(num_bins=2, salt='string')
    with self.assertRaisesRegex(ValueError, 'can only be a tuple of size 2'):
      _ = hashing.Hashing(num_bins=2, salt=[1])
    with self.assertRaisesRegex(ValueError, 'can only be a tuple of size 2'):
      _ = hashing.Hashing(num_bins=1, salt=constant_op.constant([133, 137]))

  def test_hash_compute_output_signature(self):
    input_shape = tensor_shape.TensorShape([2, 3])
    input_spec = tensor_spec.TensorSpec(input_shape, dtypes.string)
    layer = hashing.Hashing(num_bins=2)
    output_spec = layer.compute_output_signature(input_spec)
    self.assertEqual(output_spec.shape.dims, input_shape.dims)
    self.assertEqual(output_spec.dtype, dtypes.int64)

  @testing_utils.run_v2_only
  def test_config_with_custom_name(self):
    layer = hashing.Hashing(num_bins=2, name='hashing')
    config = layer.get_config()
    layer_1 = hashing.Hashing.from_config(config)
    self.assertEqual(layer_1.name, layer.name)


if __name__ == '__main__':
  test.main()
