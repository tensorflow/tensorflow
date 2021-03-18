# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for `tf.data.Dataset.skip()`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.platform import test


class SkipTest(test_base.DatasetTestBase, parameterized.TestCase):

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(count=[-1, 0, 4, 10, 25])))
  def testBasic(self, count):
    components = (np.arange(10),)
    dataset = dataset_ops.Dataset.from_tensor_slices(components).skip(count)
    self.assertEqual(
        [c.shape[1:] for c in components],
        [shape for shape in dataset_ops.get_legacy_output_shapes(dataset)])
    start_range = min(count, 10) if count != -1 else 10
    self.assertDatasetProduces(
        dataset,
        [tuple(components[0][i:i + 1]) for i in range(start_range, 10)])


class SkipDatasetCheckpointTest(checkpoint_test_base.CheckpointTestBase,
                                parameterized.TestCase):

  def _build_skip_dataset(self, count):
    components = (np.arange(10),)
    return dataset_ops.Dataset.from_tensor_slices(components).skip(count)

  @combinations.generate(test_base.default_test_combinations())
  def testSkipFewerThanInputs(self):
    count = 4
    num_outputs = 10 - count
    self.run_core_tests(lambda: self._build_skip_dataset(count), num_outputs)

  @combinations.generate(test_base.default_test_combinations())
  def testSkipVarious(self):
    # Skip more than inputs
    self.run_core_tests(lambda: self._build_skip_dataset(20), 0)
    # Skip exactly the input size
    self.run_core_tests(lambda: self._build_skip_dataset(10), 0)
    self.run_core_tests(lambda: self._build_skip_dataset(-1), 0)
    # Skip nothing
    self.run_core_tests(lambda: self._build_skip_dataset(0), 10)

  @combinations.generate(test_base.default_test_combinations())
  def testInvalidSkip(self):
    with self.assertRaisesRegex(ValueError,
                                "Shape must be rank 0 but is rank 1"):
      self.run_core_tests(lambda: self._build_skip_dataset([1, 2]), 0)


if __name__ == "__main__":
  test.main()
