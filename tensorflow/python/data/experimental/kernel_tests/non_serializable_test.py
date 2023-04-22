# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for `tf.data.experimental.non_serializable()`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.data.experimental.ops import testing
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.platform import test


class NonSerializableTest(test_base.DatasetTestBase, parameterized.TestCase):

  @combinations.generate(test_base.default_test_combinations())
  def testNonSerializable(self):
    dataset = dataset_ops.Dataset.from_tensors(0)
    dataset = dataset.apply(testing.assert_next(["FiniteSkip"]))
    dataset = dataset.skip(0)  # Should not be removed by noop elimination
    dataset = dataset.apply(testing.non_serializable())
    dataset = dataset.apply(testing.assert_next(["MemoryCacheImpl"]))
    dataset = dataset.skip(0)  # Should be removed by noop elimination
    dataset = dataset.cache()
    options = dataset_ops.Options()
    options.experimental_optimization.apply_default_optimizations = False
    options.experimental_optimization.noop_elimination = True
    dataset = dataset.with_options(options)
    self.assertDatasetProduces(dataset, expected_output=[0])

  @combinations.generate(test_base.default_test_combinations())
  def testNonSerializableAsDirectInput(self):
    """Tests that non-serializable dataset can be OptimizeDataset's input."""
    dataset = dataset_ops.Dataset.from_tensors(0)
    dataset = dataset.apply(testing.non_serializable())
    options = dataset_ops.Options()
    options.experimental_optimization.apply_default_optimizations = False
    options.experimental_optimization.noop_elimination = True
    dataset = dataset.with_options(options)
    self.assertDatasetProduces(dataset, expected_output=[0])


if __name__ == "__main__":
  test.main()
