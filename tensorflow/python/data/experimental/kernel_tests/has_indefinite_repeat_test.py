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
"""Tests for `tf.data.experimental.has_indefinite_repeat()`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.data.experimental.ops import has_indefinite_repeat
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.platform import test


class HasIndefiniteRepeat(test_base.DatasetTestBase, parameterized.TestCase):
  """Tests for `tf.data.experimental.has_indefinite_repeat()`."""

  @parameterized.named_parameters(
      ("NoRepeat", dataset_ops.Dataset.range(10), False),
      ("FiniteRepeat", dataset_ops.Dataset.range(10).repeat(2), False),
      ("FiniteRepeatNotAtEnd", dataset_ops.Dataset.range(10).repeat(2).skip(1),
       False),
      ("InfiniteRepeat", dataset_ops.Dataset.range(10).repeat(), True),
      ("InfiniteRepeatNotAtEnd", dataset_ops.Dataset.range(10).repeat().skip(1),
       True),
      ("InfiniteRepeatThenFiniteRepeat",
       dataset_ops.Dataset.range(10).repeat().repeat(2), True),
      ("ConcatenateFiniteAndInfinite",
       dataset_ops.Dataset.range(10).repeat(2).concatenate(
           dataset_ops.Dataset.range(10).repeat()), True),
      ("AssumeFinite", dataset_ops.Dataset.range(10).repeat().apply(
          has_indefinite_repeat.assume_finite()), False),
  )
  def testHasIndefiniteRepeat(self, dataset, expected_result):
    self.assertEqual(
        has_indefinite_repeat.has_indefinite_repeat(dataset), expected_result)


if __name__ == "__main__":
  test.main()
