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
"""Tests for checkpointing the ChooseFastestDataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.data.experimental.ops import optimization
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.platform import test


class ChooseFastestDatasetCheckpointTest(
    checkpoint_test_base.CheckpointTestBase, parameterized.TestCase):

  @combinations.generate(test_base.default_test_combinations())
  def testCore(self):
    num_outputs = 10
    batch_size = 2

    def build_ds():
      dataset = dataset_ops.Dataset.range(num_outputs)
      map_fn = lambda x: x * 2
      return optimization._ChooseFastestDataset([  # pylint: disable=protected-access
          dataset.map(map_fn).batch(batch_size),
          dataset.batch(batch_size).map(map_fn)
      ])

    self.run_core_tests(build_ds, num_outputs // 2)


if __name__ == "__main__":
  test.main()
