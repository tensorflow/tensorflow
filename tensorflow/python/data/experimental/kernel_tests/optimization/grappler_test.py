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
"""Tests for the generic Grappler optimizations used within tf.data."""
from absl.testing import parameterized

from tensorflow.core.example import example_pb2
from tensorflow.core.example import feature_pb2
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.platform import test


class GrapplerTest(test_base.DatasetTestBase, parameterized.TestCase):

  @combinations.generate(test_base.default_test_combinations())
  def testConstantFoldingVarLenFeature(self):
    example = example_pb2.Example(features=feature_pb2.Features(feature={}))
    dataset = dataset_ops.Dataset.from_tensors(example.SerializeToString())

    def parse_fn(serialized):
      features = {"x": parsing_ops.VarLenFeature(dtypes.int64)}
      parsed = parsing_ops.parse_single_example(serialized, features)
      parsed = parsed["x"].values

      size = array_ops.size(parsed)
      value = math_ops.cast(parsed, dtypes.bool)
      return cond.cond(size > 0,
                       lambda: array_ops.reshape(value, []),
                       lambda: array_ops.zeros([], dtypes.bool))

    dataset = dataset.map(parse_fn)

    self.assertDatasetProduces(dataset, expected_output=[0])

  @combinations.generate(test_base.default_test_combinations())
  def testLayoutOptimizationConv2D(self):
    if not test_util.is_gpu_available():
      self.skipTest("No GPU available")

    # Compute convolution with input and filter of [1, 1, 1, 1] shape.
    # Verify that Grappler doesn't transpose Conv2D data format to NCHW.
    dataset = dataset_ops.Dataset.from_tensors((1, 1))

    def map_function(x, y):
      i = math_ops.cast(x, dtypes.float32)
      i = array_ops.reshape(i, [1, 1, 1, 1])
      f = math_ops.cast(y, dtypes.float32)
      f = array_ops.reshape(f, [1, 1, 1, 1])
      c = nn_ops.conv2d(i, f, strides=[1, 1, 1, 1], padding="VALID")
      return array_ops.reshape(c, ())

    dataset = dataset.map(map_function)
    self.assertDatasetProduces(dataset, expected_output=[1])


if __name__ == "__main__":
  test.main()
