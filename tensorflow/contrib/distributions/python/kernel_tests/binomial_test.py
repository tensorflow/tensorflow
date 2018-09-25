# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy import stats
from tensorflow.contrib.distributions.python.ops import binomial
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class BinomialTest(test.TestCase):

  def testSimpleShapes(self):
    with self.cached_session():
      p = np.float32(np.random.beta(1, 1))
      binom = binomial.Binomial(total_count=1., probs=p)
      self.assertAllEqual([], binom.event_shape_tensor().eval())
      self.assertAllEqual([], binom.batch_shape_tensor().eval())
      self.assertEqual(tensor_shape.TensorShape([]), binom.event_shape)
      self.assertEqual(tensor_shape.TensorShape([]), binom.batch_shape)

  def testComplexShapes(self):
    with self.cached_session():
      p = np.random.beta(1, 1, size=(3, 2)).astype(np.float32)
      n = [[3., 2], [4, 5], [6, 7]]
      binom = binomial.Binomial(total_count=n, probs=p)
      self.assertAllEqual([], binom.event_shape_tensor().eval())
      self.assertAllEqual([3, 2], binom.batch_shape_tensor().eval())
      self.assertEqual(tensor_shape.TensorShape([]), binom.event_shape)
      self.assertEqual(
          tensor_shape.TensorShape([3, 2]), binom.batch_shape)

  def testNProperty(self):
    p = [[0.1, 0.2, 0.7], [0.2, 0.3, 0.5]]
    n = [[3.], [4]]
    with self.cached_session():
      binom = binomial.Binomial(total_count=n, probs=p)
      self.assertEqual((2, 1), binom.total_count.get_shape())
      self.assertAllClose(n, binom.total_count.eval())

  def testPProperty(self):
    p = [[0.1, 0.2, 0.7]]
    with self.cached_session():
      binom = binomial.Binomial(total_count=3., probs=p)
      self.assertEqual((1, 3), binom.probs.get_shape())
      self.assertEqual((1, 3), binom.logits.get_shape())
      self.assertAllClose(p, binom.probs.eval())

  def testLogitsProperty(self):
    logits = [[0., 9., -0.5]]
    with self.cached_session():
      binom = binomial.Binomial(total_count=3., logits=logits)
      self.assertEqual((1, 3), binom.probs.get_shape())
      self.assertEqual((1, 3), binom.logits.get_shape())
      self.assertAllClose(logits, binom.logits.eval())

  def testPmfAndCdfNandCountsAgree(self):
    p = [[0.1, 0.2, 0.7]]
    n = [[5.]]
    with self.cached_session():
      binom = binomial.Binomial(total_count=n, probs=p, validate_args=True)
      binom.prob([2., 3, 2]).eval()
      binom.prob([3., 1, 2]).eval()
      binom.cdf([2., 3, 2]).eval()
      binom.cdf([3., 1, 2]).eval()
      with self.assertRaisesOpError("Condition x >= 0.*"):
        binom.prob([-1., 4, 2]).eval()
      with self.assertRaisesOpError("Condition x <= y.*"):
        binom.prob([7., 3, 0]).eval()
      with self.assertRaisesOpError("Condition x >= 0.*"):
        binom.cdf([-1., 4, 2]).eval()
      with self.assertRaisesOpError("Condition x <= y.*"):
        binom.cdf([7., 3, 0]).eval()

  def testPmfAndCdfNonIntegerCounts(self):
    p = [[0.1, 0.2, 0.7]]
    n = [[5.]]
    with self.cached_session():
      # No errors with integer n.
      binom = binomial.Binomial(total_count=n, probs=p, validate_args=True)
      binom.prob([2., 3, 2]).eval()
      binom.prob([3., 1, 2]).eval()
      binom.cdf([2., 3, 2]).eval()
      binom.cdf([3., 1, 2]).eval()
      placeholder = array_ops.placeholder(dtypes.float32)
      # Both equality and integer checking fail.
      with self.assertRaisesOpError(
          "cannot contain fractional components."):
        binom.prob(placeholder).eval(feed_dict={placeholder: [1.0, 2.5, 1.5]})
      with self.assertRaisesOpError(
          "cannot contain fractional components."):
        binom.cdf(placeholder).eval(feed_dict={placeholder: [1.0, 2.5, 1.5]})

      binom = binomial.Binomial(total_count=n, probs=p, validate_args=False)
      binom.prob([1., 2., 3.]).eval()
      binom.cdf([1., 2., 3.]).eval()
      # Non-integer arguments work.
      binom.prob([1.0, 2.5, 1.5]).eval()
      binom.cdf([1.0, 2.5, 1.5]).eval()

  def testPmfAndCdfBothZeroBatches(self):
    with self.cached_session():
      # Both zero-batches.  No broadcast
      p = 0.5
      counts = 1.
      binom = binomial.Binomial(total_count=1., probs=p)
      pmf = binom.prob(counts)
      cdf = binom.cdf(counts)
      self.assertAllClose(0.5, pmf.eval())
      self.assertAllClose(stats.binom.cdf(counts, n=1, p=p), cdf.eval())
      self.assertEqual((), pmf.get_shape())
      self.assertEqual((), cdf.get_shape())

  def testPmfAndCdfBothZeroBatchesNontrivialN(self):
    with self.cached_session():
      # Both zero-batches.  No broadcast
      p = 0.1
      counts = 3.
      binom = binomial.Binomial(total_count=5., probs=p)
      pmf = binom.prob(counts)
      cdf = binom.cdf(counts)
      self.assertAllClose(stats.binom.pmf(counts, n=5., p=p), pmf.eval())
      self.assertAllClose(stats.binom.cdf(counts, n=5., p=p), cdf.eval())
      self.assertEqual((), pmf.get_shape())
      self.assertEqual((), cdf.get_shape())

  def testPmfAndCdfPStretchedInBroadcastWhenSameRank(self):
    with self.cached_session():
      p = [[0.1, 0.9]]
      counts = [[1., 2.]]
      binom = binomial.Binomial(total_count=3., probs=p)
      pmf = binom.prob(counts)
      cdf = binom.cdf(counts)
      self.assertAllClose(stats.binom.pmf(counts, n=3., p=p), pmf.eval())
      self.assertAllClose(stats.binom.cdf(counts, n=3., p=p), cdf.eval())
      self.assertEqual((1, 2), pmf.get_shape())
      self.assertEqual((1, 2), cdf.get_shape())

  def testPmfAndCdfPStretchedInBroadcastWhenLowerRank(self):
    with self.cached_session():
      p = [0.1, 0.4]
      counts = [[1.], [0.]]
      binom = binomial.Binomial(total_count=1., probs=p)
      pmf = binom.prob(counts)
      cdf = binom.cdf(counts)
      self.assertAllClose([[0.1, 0.4], [0.9, 0.6]], pmf.eval())
      self.assertAllClose([[1.0, 1.0], [0.9, 0.6]], cdf.eval())
      self.assertEqual((2, 2), pmf.get_shape())
      self.assertEqual((2, 2), cdf.get_shape())

  def testBinomialMean(self):
    with self.cached_session():
      n = 5.
      p = [0.1, 0.2, 0.7]
      binom = binomial.Binomial(total_count=n, probs=p)
      expected_means = stats.binom.mean(n, p)
      self.assertEqual((3,), binom.mean().get_shape())
      self.assertAllClose(expected_means, binom.mean().eval())

  def testBinomialVariance(self):
    with self.cached_session():
      n = 5.
      p = [0.1, 0.2, 0.7]
      binom = binomial.Binomial(total_count=n, probs=p)
      expected_variances = stats.binom.var(n, p)
      self.assertEqual((3,), binom.variance().get_shape())
      self.assertAllClose(expected_variances, binom.variance().eval())

  def testBinomialMode(self):
    with self.cached_session():
      n = 5.
      p = [0.1, 0.2, 0.7]
      binom = binomial.Binomial(total_count=n, probs=p)
      expected_modes = [0., 1, 4]
      self.assertEqual((3,), binom.mode().get_shape())
      self.assertAllClose(expected_modes, binom.mode().eval())

  def testBinomialMultipleMode(self):
    with self.cached_session():
      n = 9.
      p = [0.1, 0.2, 0.7]
      binom = binomial.Binomial(total_count=n, probs=p)
      # For the case where (n + 1) * p is an integer, the modes are:
      # (n + 1) * p and (n + 1) * p - 1. In this case, we get back
      # the larger of the two modes.
      expected_modes = [1., 2, 7]
      self.assertEqual((3,), binom.mode().get_shape())
      self.assertAllClose(expected_modes, binom.mode().eval())


if __name__ == "__main__":
  test.main()
