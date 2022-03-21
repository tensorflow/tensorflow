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
"""Tests for random index shuffle ops."""

import itertools

from absl.testing import parameterized
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import stateless_random_ops as stateless
from tensorflow.python.platform import test

_SEEDS = ((74, 117), (42, 5))
_MAX_VALUES = (129, 2_389)
_DTYPES = (dtypes.int32, dtypes.uint32, dtypes.int64, dtypes.uint64)


class StatelessOpsTest(test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      itertools.product(_SEEDS, _DTYPES, _MAX_VALUES, _DTYPES))
  def testUnbatched(self, seed, seed_dtype, max_index, index_dtype):
    if max_index > 200:
      self.skipTest('Too slow in graph mode.')
    seen = (max_index + 1) * [False]
    seed = math_ops.cast(seed, seed_dtype)
    for index in range(max_index + 1):
      new_index = stateless.index_shuffle(
          math_ops.cast(index, index_dtype),
          seed,
          max_index=math_ops.cast(max_index, index_dtype))
      self.assertEqual(new_index.dtype, index_dtype)
      new_index = self.evaluate(new_index)
      self.assertGreaterEqual(new_index, 0)
      self.assertLessEqual(new_index, max_index)
      self.assertFalse(seen[new_index])
      seen[new_index] = True

  @parameterized.parameters(
      itertools.product(_SEEDS, _DTYPES, _MAX_VALUES, _DTYPES))
  def testBatchedBroadcastSeedAndMaxval(self, seed, seed_dtype, max_index,
                                        index_dtype):
    seed = math_ops.cast(seed, seed_dtype)
    index = math_ops.cast(range(max_index + 1), index_dtype)
    new_index = stateless.index_shuffle(index, seed, max_index=max_index)
    self.assertEqual(new_index.dtype, index_dtype)
    new_index = self.evaluate(new_index)
    self.assertAllGreaterEqual(new_index, 0)
    self.assertAllLessEqual(new_index, max_index)
    self.assertLen(new_index, max_index + 1)
    self.assertLen(set(new_index), max_index + 1)


if __name__ == '__main__':
  test.main()
