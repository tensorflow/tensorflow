# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for the private `_RebatchDataset` transformation."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.data.experimental.ops import batching
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


@test_util.run_all_in_graph_and_eager_modes
class RebatchDatasetTest(test_base.DatasetTestBase):

  def testBasic(self):
    dataset = dataset_ops.Dataset.range(1024).batch(32, drop_remainder=True)
    rebatched_dataset = batching._RebatchDataset(dataset, num_workers=4)
    self.assertEqual(
        [[32]], [ts.as_list() for ts in nest.flatten(dataset.output_shapes)])
    self.assertEqual(
        [[8]],
        [ts.as_list() for ts in nest.flatten(rebatched_dataset.output_shapes)])

    expected_output = [[k for k in range(i, i + 8)] for i in range(0, 1024, 8)]  # pylint: disable=g-complex-comprehension
    self.assertDatasetProduces(rebatched_dataset, expected_output)

  def testScalarInputError(self):
    dataset = dataset_ops.Dataset.range(1024)
    with self.assertRaisesRegexp(ValueError, "at least one dimension"):
      batching._RebatchDataset(dataset, num_workers=4)

  def testUnknownBatchSizeError(self):
    dataset = dataset_ops.Dataset.range(1024).batch(32)
    with self.assertRaisesRegexp(ValueError, "unknown batch size datasets"):
      batching._RebatchDataset(dataset, num_workers=4)

  def testNotDivisibleError(self):
    dataset = dataset_ops.Dataset.range(1024).batch(32, drop_remainder=True)
    with self.assertRaisesRegexp(ValueError, "not divisible by"):
      batching._RebatchDataset(dataset, num_workers=5)


if __name__ == "__main__":
  test.main()
