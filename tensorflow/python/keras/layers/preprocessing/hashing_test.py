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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util as tf_test_util
from tensorflow.python.keras import keras_parameterized
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
    self.assertAllClose(np.asarray([[0], [0], [0], [0], [0]]), output)

  def test_hash_two_bins(self):
    layer = hashing.Hashing(num_bins=2)
    inp = np.asarray([['A'], ['B'], ['C'], ['D'], ['E']])
    output = layer(inp)
    self.assertEqual(output.numpy().max(), 1)
    self.assertEqual(output.numpy().min(), 0)

  def test_hash_sparse_input(self):
    layer = hashing.Hashing(num_bins=2)
    inp = sparse_tensor.SparseTensor(
        indices=[[0, 0], [1, 0], [1, 1], [2, 0], [2, 1]],
        values=['omar', 'stringer', 'marlo', 'wire', 'skywalker'],
        dense_shape=[3, 2])
    output = layer(inp)
    self.assertEqual(output.values.numpy().max(), 1)
    self.assertEqual(output.values.numpy().min(), 0)

  def test_hash_ragged_string_input(self):
    layer = hashing.Hashing(num_bins=2)
    inp_data = ragged_factory_ops.constant(
        [['omar', 'stringer', 'marlo', 'wire'], ['marlo', 'skywalker', 'wire']],
        dtype=dtypes.string)
    out_data = layer(inp_data)
    self.assertEqual(out_data.values.numpy().max(), 1)
    self.assertEqual(out_data.values.numpy().min(), 0)
    # hash of 'marlo' should be same.
    self.assertAllClose(out_data[0][2], out_data[1][0])
    # hash of 'wire' should be same.
    self.assertAllClose(out_data[0][3], out_data[1][2])

    inp_t = input_layer.Input(shape=(None,), ragged=True, dtype=dtypes.string)
    out_t = layer(inp_t)
    model = training.Model(inputs=inp_t, outputs=out_t)
    self.assertAllClose(out_data, model.predict(inp_data))

  def test_hash_compute_output_signature(self):
    input_shape = tensor_shape.TensorShape([2, 3])
    input_spec = tensor_spec.TensorSpec(input_shape, dtypes.string)
    layer = hashing.Hashing(num_bins=2)
    output_spec = layer.compute_output_signature(input_spec)
    self.assertEqual(output_spec.shape.dims, input_shape.dims)
    self.assertEqual(output_spec.dtype, dtypes.int64)

  @tf_test_util.run_v2_only
  def test_config_with_custom_name(self):
    layer = hashing.Hashing(num_bins=2, name='hashing')
    config = layer.get_config()
    layer_1 = hashing.Hashing.from_config(config)
    self.assertEqual(layer_1.name, layer.name)


if __name__ == '__main__':
  test.main()
