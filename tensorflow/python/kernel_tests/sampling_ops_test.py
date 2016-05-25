# Copyright 2016 Google Inc. All Rights Reserved.
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

"""Tests for tensorflow.ops.sampling_ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import sampling_ops


class BernoulliSampleTest(tf.test.TestCase):

  def testAllOrNothing(self):
    with self.test_session():
      p = 0.0
      a = np.random.randn(10)
      b = np.random.randn(10)

      output = sampling_ops.bernoulli_sample(p, a, b).eval()

      self.assertAllEqual(a, output)

      p = 1.0

      output = sampling_ops.bernoulli_sample(p, a, b).eval()

      self.assertAllEqual(b, output)


class SampleDistributionIndexTest(tf.test.TestCase):

  def testIndexRange(self):
    with self.test_session():
      batch_size = 10
      vocab_size = 100
      dist = np.random.randn(batch_size, vocab_size)
      idx = sampling_ops.sample_distribution_index(dist).eval()

      for b in range(batch_size):
        self.assertGreaterEqual(idx[b], 0)
        self.assertLess(idx[b], vocab_size)

  def testOneHot(self):
    with self.test_session():
      batch_size = 10
      vocab_size = 100
      dist = np.zeros([batch_size, vocab_size], dtype=np.float32)
      hot_idx = np.random.randint(0, vocab_size, size=batch_size)
      dist[range(batch_size), hot_idx] = 1.0

      idx = sampling_ops.sample_distribution_index(dist).eval()
      self.assertAllEqual(idx, hot_idx)

if __name__ == '__main__':
  tf.test.main()
