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
"""Tests for the `ShuffleAndRepeatFusion` optimization."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.data.experimental.ops import testing
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.platform import test


class ShuffleAndRepeatFusionTest(test_base.DatasetTestBase,
                                 parameterized.TestCase):

  @combinations.generate(test_base.default_test_combinations())
  def testShuffleAndRepeatFusion(self):
    expected = "ShuffleAndRepeat"
    dataset = dataset_ops.Dataset.range(10).apply(
        testing.assert_next([expected])).shuffle(10).repeat(2)
    options = dataset_ops.Options()
    options.experimental_optimization.apply_default_optimizations = False
    options.experimental_optimization.shuffle_and_repeat_fusion = True
    dataset = dataset.with_options(options)
    get_next = self.getNext(dataset)

    for _ in range(2):
      results = []
      for _ in range(10):
        results.append(self.evaluate(get_next()))
      self.assertAllEqual([x for x in range(10)], sorted(results))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())


if __name__ == "__main__":
  test.main()
