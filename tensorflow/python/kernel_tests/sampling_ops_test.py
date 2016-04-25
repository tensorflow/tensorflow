from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import sampling_ops
from tensorflow.python.ops import gen_sampling_ops


class BernoulliSampleTest(tf.test.TestCase):

  def testAllOrNothing(self):
    with self.test_session():
      p = 0.0
      a = np.random.randn(10)
      b = np.random.randn(10)

      output = gen_sampling_ops.bernoulli_sample(p, a, b).eval()

      self.assertAllEqual(a, output)

      p = 1.0

      output = gen_sampling_ops.bernoulli_sample(p, a, b).eval()

      self.assertAllEqual(b, output)


class SampleDistributionIndexTest(tf.test.TestCase):

  def testIndexRange(self):
    with self.test_session():
      batch_size = 10
      vocab_size = 100
      dist = np.random.randn(batch_size, vocab_size)
      idx = gen_sampling_ops.sample_distribution_index(dist).eval()

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

      idx = gen_sampling_ops.sample_distribution_index(dist).eval()
      self.assertAllEqual(idx, hot_idx)

if __name__ == '__main__':
  tf.test.main()
