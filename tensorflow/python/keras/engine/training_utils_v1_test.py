# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for training utility functions."""

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.keras.engine import training_utils_v1
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class StandardizeInputDataTest(test.TestCase):

  @test_util.run_deprecated_v1
  def test_unknown_rank_tensor_does_not_crash(self):
    # Regression test: tensors of unknown rank (e.g. produced by
    # tf.numpy_function) used to crash with
    # "ValueError: as_list() is not defined on an unknown TensorShape".
    x = array_ops.placeholder(dtypes.float32, shape=None)
    result = training_utils_v1.standardize_input_data(
        [x], ['input_1'], shapes=[(None, 10)])
    self.assertIs(result[0], x)

  @test_util.run_deprecated_v1
  def test_unknown_rank_composite_tensor_does_not_crash(self):
    indices = array_ops.placeholder(dtypes.int64, shape=None)
    values = array_ops.placeholder(dtypes.float32, shape=None)
    dense_shape = array_ops.placeholder(dtypes.int64, shape=None)
    x = sparse_tensor.SparseTensor(indices, values, dense_shape)
    result = training_utils_v1.standardize_input_data(
        [x], ['input_1'], shapes=[(None, 10)])
    self.assertIs(result[0], x)


if __name__ == '__main__':
  test.main()
