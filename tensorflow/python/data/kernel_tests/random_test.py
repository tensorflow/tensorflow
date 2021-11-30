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
"""Tests for `tf.data.Dataset.random()`."""
from absl.testing import parameterized

from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import random_seed
from tensorflow.python.platform import test


class RandomTest(test_base.DatasetTestBase, parameterized.TestCase):

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(global_seed=[None, 10], local_seed=[None, 20])))
  def testDeterminism(self, global_seed, local_seed):
    expect_determinism = (global_seed is not None) or (local_seed is not None)

    random_seed.set_random_seed(global_seed)
    ds = dataset_ops.Dataset.random(seed=local_seed).take(10)

    output_1 = self.getDatasetOutput(ds)
    ds = self.graphRoundTrip(ds)
    output_2 = self.getDatasetOutput(ds)

    if expect_determinism:
      self.assertEqual(output_1, output_2)
    else:
      # Technically not guaranteed since the two randomly-chosen int64 seeds
      # could match, but that is sufficiently unlikely (1/2^128 with perfect
      # random number generation).
      self.assertNotEqual(output_1, output_2)

  @combinations.generate(test_base.default_test_combinations())
  def testName(self):
    dataset = dataset_ops.Dataset.random(
        seed=42, name="random").take(1).map(lambda _: 42)
    self.assertDatasetProduces(dataset, [42])


if __name__ == "__main__":
  test.main()
