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
"""Long tests for Multinomial."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test


class MultinomialTest(test.TestCase):
  # check that events with tiny probabilities are not over-sampled

  def testLargeDynamicRange(self):
    random_seed.set_random_seed(10)
    counts_by_indices = {}
    with self.test_session(use_gpu=True) as sess:
      samples = random_ops.multinomial(
          constant_op.constant([[-30, 0]], dtype=dtypes.float32),
          num_samples=1000000,
          seed=15)
      for _ in range(100):
        x = sess.run(samples)
        indices, counts = np.unique(x, return_counts=True)
        for index, count in zip(indices, counts):
          if index in counts_by_indices.keys():
            counts_by_indices[index] += count
          else:
            counts_by_indices[index] = count
    self.assertEqual(counts_by_indices[1], 100000000)

  def testLargeDynamicRange2(self):
    random_seed.set_random_seed(10)
    counts_by_indices = {}
    with self.test_session(use_gpu=True) as sess:
      samples = random_ops.multinomial(
          constant_op.constant([[0, -30]], dtype=dtypes.float32),
          num_samples=1000000,
          seed=15)
      for _ in range(100):
        x = sess.run(samples)
        indices, counts = np.unique(x, return_counts=True)
        for index, count in zip(indices, counts):
          if index in counts_by_indices.keys():
            counts_by_indices[index] += count
          else:
            counts_by_indices[index] = count
    self.assertEqual(counts_by_indices[0], 100000000)

  def testLargeDynamicRange3(self):
    random_seed.set_random_seed(10)
    counts_by_indices = {}
    # here the cpu undersamples and won't pass this test either
    with self.test_session(use_gpu=True) as sess:
      samples = random_ops.multinomial(
          constant_op.constant([[0, -17]], dtype=dtypes.float32),
          num_samples=1000000,
          seed=22)

      # we'll run out of memory if we try to draw 1e9 samples directly
      # really should fit in 12GB of memory...
      for _ in range(100):
        x = sess.run(samples)
        indices, counts = np.unique(x, return_counts=True)
        for index, count in zip(indices, counts):
          if index in counts_by_indices.keys():
            counts_by_indices[index] += count
          else:
            counts_by_indices[index] = count
    self.assertGreater(counts_by_indices[1], 0)

if __name__ == "__main__":
  test.main()
