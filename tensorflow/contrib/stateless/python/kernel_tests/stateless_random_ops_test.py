# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for stateless random ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensorflow.contrib import stateless
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test

CASES = [(stateless.stateless_random_uniform, random_ops.random_uniform),
         (stateless.stateless_random_normal, random_ops.random_normal),
         (stateless.stateless_truncated_normal, random_ops.truncated_normal)]


def invert_philox(key, value):
  """Invert the Philox bijection."""
  key = np.array(key, dtype=np.uint32)
  value = np.array(value, dtype=np.uint32)
  step = np.array([0x9E3779B9, 0xBB67AE85], dtype=np.uint32)
  for n in range(10)[::-1]:
    key0, key1 = key + n * step
    v0 = value[3] * 0x991a7cdb & 0xffffffff
    v2 = value[1] * 0x6d7cae67 & 0xffffffff
    hi0 = v0 * 0xD2511F53 >> 32
    hi1 = v2 * 0xCD9E8D57 >> 32
    v1 = hi1 ^ value[0] ^ key0
    v3 = hi0 ^ value[2] ^ key1
    value = v0, v1, v2, v3
  return np.array(value)


class StatelessOpsTest(test.TestCase):

  def testMatchStateful(self):
    # Stateless ops should be the same as stateful ops on the first call
    # after seed scrambling.
    key = 0x3ec8f720, 0x02461e29
    for seed in (7, 17), (11, 5), (2, 3):
      preseed = invert_philox(key, (seed[0], 0, seed[1], 0)).astype(np.uint64)
      preseed = preseed[::2] | preseed[1::2] << 32
      random_seed.set_random_seed(seed[0])
      with self.test_session(use_gpu=True):
        for stateless_op, stateful_op in CASES:
          for shape in (), (3,), (2, 5):
            stateful = stateful_op(shape, seed=seed[1])
            pure = stateless_op(shape, seed=preseed)
            self.assertAllEqual(stateful.eval(), pure.eval())

  def testDeterminism(self):
    # Stateless values should be equal iff the seeds are equal (roughly)
    with self.test_session(use_gpu=True):
      seed_t = array_ops.placeholder(dtypes.int64, shape=[2])
      seeds = [(x, y) for x in range(5) for y in range(5)] * 3
      for stateless_op, _ in CASES:
        for shape in (), (3,), (2, 5):
          pure = stateless_op(shape, seed=seed_t)
          values = [(seed, pure.eval(feed_dict={seed_t: seed}))
                    for seed in seeds]
          for s0, v0 in values:
            for s1, v1 in values:
              self.assertEqual(s0 == s1, np.all(v0 == v1))

  def testShapeType(self):
    with self.test_session(use_gpu=True):
      for shape_dtype in [dtypes.int32, dtypes.int64]:
        seed_t = array_ops.placeholder(dtypes.int64, shape=[2])
        seeds = [(x, y) for x in range(5) for y in range(5)] * 3
        for stateless_op, _ in CASES:
          for shape in (), (3,), (2, 5):
            pure = stateless_op(constant_op.constant(shape, dtype=shape_dtype),
                                seed=seed_t)
            values = [(seed, pure.eval(feed_dict={seed_t: seed}))
                      for seed in seeds]
            for s0, v0 in values:
              for s1, v1 in values:
                self.assertEqual(s0 == s1, np.all(v0 == v1))


if __name__ == '__main__':
  test.main()
