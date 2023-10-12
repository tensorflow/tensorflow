# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for ragged_array_ops.bitcast."""

from absl.testing import parameterized

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops.ragged import ragged_array_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import googletest


@test_util.run_all_in_graph_and_eager_modes
class RaggedSplitOpTest(test_util.TensorFlowTestCase, parameterized.TestCase):

  @parameterized.parameters([
      #=========================================================================
      # Cast to same-size dtype.
      #=========================================================================
      dict(
          descr='int32 to int32 cast',
          inputs=ragged_factory_ops.constant_value(
              [[1, 2], [3]],
              dtype=dtypes.int32,
          ),
          outputs=ragged_factory_ops.constant_value(
              [[1, 2], [3]],
              dtype=dtypes.int32,
          )),
      dict(
          descr='int32 to uint32 cast',
          inputs=ragged_factory_ops.constant_value(
              [[1, 2], [-1]],
              dtype=dtypes.int32,
          ),
          outputs=ragged_factory_ops.constant_value(
              [[1, 2], [4294967295]],
              dtype=dtypes.uint32,
          )),
      dict(
          descr='uint32 to int32 cast',
          inputs=ragged_factory_ops.constant_value(
              [[1, 2], [4294967295]],
              dtype=dtypes.uint32,
          ),
          outputs=ragged_factory_ops.constant_value(
              [[1, 2], [-1]],
              dtype=dtypes.int32,
          )),
      #=========================================================================
      # Cast to larger dtype.
      #=========================================================================
      dict(
          descr='int32 to int64 cast',
          inputs=ragged_factory_ops.constant_value(
              [[[1, 0], [2, 0]], [[3, 0]]],
              dtype=dtypes.int32,
              ragged_rank=1,
          ),
          outputs=ragged_factory_ops.constant_value(
              [[1, 2], [3]],
              dtype=dtypes.int64,
          )),
      #=========================================================================
      # Cast to smaller dtype.
      #=========================================================================
      dict(
          descr='int64 to int32 cast',
          inputs=ragged_factory_ops.constant_value(
              [[1, 2], [3]],
              dtype=dtypes.int64,
          ),
          outputs=ragged_factory_ops.constant_value(
              [[[1, 0], [2, 0]], [[3, 0]]],
              dtype=dtypes.int32,
              ragged_rank=1,
          )),
  ])  # pyformat: disable
  def testBitcast(self, descr, inputs, outputs, name=None):
    result = ragged_array_ops.bitcast(inputs, outputs.dtype, name)
    self.assertEqual(result.dtype, outputs.dtype)
    self.assertEqual(result.ragged_rank, outputs.ragged_rank)
    self.assertAllEqual(result, outputs)

  @parameterized.parameters([
      dict(
          descr='Upcast requires uniform inner dimension',
          inputs=ragged_factory_ops.constant_value(
              [[[1, 0], [2, 0]], [[3, 0]]],
              dtype=dtypes.int32,
              ragged_rank=2,
          ),
          cast_to_dtype=dtypes.int64,
          exception=ValueError,
          message='`input.flat_values` is required to have rank >= 2'),
  ])  # pyformat: disable
  def testBitcastError(self,
                       descr,
                       inputs,
                       cast_to_dtype,
                       exception,
                       message,
                       name=None):
    with self.assertRaisesRegex(exception, message):
      result = ragged_array_ops.bitcast(inputs, cast_to_dtype, name)
      self.evaluate(result)


if __name__ == '__main__':
  googletest.main()
