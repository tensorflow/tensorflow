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
"""Tests for multinomial generation ops in the XLA JIT compiler."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.platform import googletest


# TODO(srvasude): Merge this with
# third_party/tensorflow/python/kernel_tests/random/multinomial_op_test.py.
class CategoricalTest(xla_test.XLATestCase):
  """Test cases for random-number generating operators."""

  def output_dtypes(self):
    return set(self.int_types).intersection([np.int32, np.int64])

  def _chi2(self, expected, actual):
    """Returns Chi2 GOF statistic."""
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    diff = actual - expected
    chi2 = np.sum(diff * diff / expected)
    return chi2

  def _do_sampling(self, logits, num_samples):
    """Categorical samples from given input.

    Args:
      logits: Numpy ndarray of shape [batch_size, num_classes].
      num_samples: Int; number of samples to draw.

    Returns:
      Frequencies from sampled classes; shape [batch_size, num_classes].
    """
    with self.cached_session() as sess, self.test_scope():
      random_seed.set_random_seed(1618)
      op = random_ops.multinomial(logits, num_samples,
                                  output_dtype=dtypes.int32)
      d = sess.run(op)

    batch_size, num_classes = logits.shape
    freqs_mat = []
    for i in range(batch_size):
      cnts = dict(collections.Counter(d[i, :]))

      # Requires drawn class labels be in range.
      self.assertLess(max(cnts.keys()), num_classes)
      self.assertGreaterEqual(min(cnts.keys()), 0)

      freqs = [(cnts[k] * 1. / num_samples if k in cnts else 0)
               for k in range(num_classes)]
      freqs_mat.append(freqs)

    return freqs_mat

  def _testRngIsNotConstant(self, rng, dtype, output_dtype):
    # Tests that 'rng' does not always return the same value.
    with self.cached_session() as sess:
      with self.test_scope():
        x = rng(dtype, output_dtype)

      # The random-number generator, if working correctly, should produce the
      # same output multiple times with low probability.
      y = sess.run(x)
      z = sess.run(x)
      w = sess.run(x)

      # We use exact equality here. If the random-number generator is producing
      # deterministic output, all three outputs will be bitwise identical.
      self.assertTrue((not np.array_equal(y, z)) or
                      (not np.array_equal(z, w)) or
                      (not np.array_equal(y, w)))

  def testCategoricalIsNotConstant(self):
    def rng(dtype, output_dtype):
      return random_ops.multinomial(np.array([[1., 1., 1.]], dtype=dtype), 10,
                                    output_dtype=output_dtype)

    dtype = np.float32
    for output_dtype in self.output_dtypes():
      self._testRngIsNotConstant(rng, dtype, output_dtype)

  def testCategoricalIsInRange(self):
    for dtype in self.float_types:
      for output_dtype in self.output_dtypes():
        with self.cached_session() as sess:
          with self.test_scope():
            x = random_ops.multinomial(
                array_ops.ones(shape=[1, 20], dtype=dtype), 1000,
                output_dtype=output_dtype)
          y = sess.run(x)
          self.assertTrue((y >= 0).sum() == 1000)
          self.assertTrue((y < 20).sum() == 1000)

  def testSamplingCorrectness(self):
    np.random.seed(1618)  # Make it reproducible.
    num_samples = 21000

    rand_probs = np.random.dirichlet([1., 1., 2., 3.])
    rand_probs2 = np.random.dirichlet([1., 4., 5.], size=3)  # batched
    for probs in [[.5, .5], [.85, .05, .1], rand_probs, rand_probs2]:
      probs = np.asarray(probs)
      if len(probs.shape) == 1:
        probs = probs.reshape(1, probs.size)  # singleton batch

      logits = np.log(probs).astype(np.float32)
      freqs = self._do_sampling(logits, num_samples)

      # the test here is similar to
      # python/kernel_tests/random/multinomial_op_test.py
      # Note that df >= 1 in all these cases. Choosing a cutoff of 1e-3
      # corresponds to an alpha value of 2.5% for df = 1, and smaller for larger
      # df.
      chi2 = self._chi2(probs, freqs)
      self.assertLess(chi2, 1e-3)

  def testStatelessMultinomialIsInRange(self):
    for dtype in self.float_types:
      for output_dtype in self.output_dtypes():
        with self.cached_session() as sess:
          with self.test_scope():
            seed_t = array_ops.placeholder(dtypes.int32, shape=[2])
            x = stateless_random_ops.stateless_multinomial(
                array_ops.ones(shape=[1, 20], dtype=dtype),
                1000,
                seed_t,
                output_dtype=output_dtype)
          y = sess.run(x, {seed_t: [0x12345678, 0xabcdef12]})
          self.assertTrue((y >= 0).sum() == 1000)
          self.assertTrue((y < 20).sum() == 1000)

  def testDeterminismMultinomial(self):
    # Stateless values should be equal iff the seeds are equal (roughly)
    num_samples = 10
    with self.cached_session(), self.test_scope():
      seed_t = array_ops.placeholder(dtypes.int32, shape=[2])
      seeds = [(x, y) for x in range(5) for y in range(5)] * 3
      for logits in ([[0.1, 0.25, 0.5, 0.15]], [[0.5, 0.5], [0.8, 0.2],
                                                [0.25, 0.75]]):
        pure = stateless_random_ops.stateless_multinomial(
            logits, num_samples, seed=seed_t)
        values = [(seed, pure.eval(feed_dict={seed_t: seed})) for seed in seeds]
        for s0, v0 in values:
          for s1, v1 in values:
            self.assertEqual(s0 == s1, np.all(v0 == v1))

  def testEmpty(self):
    with self.cached_session() as sess:
      with self.test_scope():
        x = random_ops.multinomial(
            array_ops.zeros([42, 40]), 0, output_dtype=dtypes.int32)
        y = sess.run(x)
        self.assertEqual(y.shape, (42, 0))

  def testEmptyStateless(self):
    with self.cached_session() as sess:
      with self.test_scope():
        seed_t = array_ops.placeholder(dtypes.int32, shape=[2])
        x = stateless_random_ops.stateless_multinomial(
            array_ops.zeros([42, 40]),
            0,
            seed=seed_t,
            output_dtype=dtypes.int32)
        y = sess.run(x, {seed_t: [0x12345678, 0xabcdef12]})
        self.assertEqual(y.shape, (42, 0))



if __name__ == '__main__':
  googletest.main()
