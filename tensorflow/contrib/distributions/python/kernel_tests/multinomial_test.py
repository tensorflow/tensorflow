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
import tensorflow as tf


class MultinomialTest(tf.test.TestCase):

  def testSimpleShapes(self):
    with self.test_session():
      p = [.1, .3, .6]
      dist = tf.contrib.distributions.Multinomial(n=1., p=p)
      self.assertEqual(3, dist.event_shape().eval())
      self.assertAllEqual([], dist.batch_shape().eval())
      self.assertEqual(tf.TensorShape([3]), dist.get_event_shape())
      self.assertEqual(tf.TensorShape([]), dist.get_batch_shape())

  def testComplexShapes(self):
    with self.test_session():
      p = 0.5 * np.ones([3, 2, 2], dtype=np.float32)
      n = [[3., 2], [4, 5], [6, 7]]
      dist = tf.contrib.distributions.Multinomial(n=n, p=p)
      self.assertEqual(2, dist.event_shape().eval())
      self.assertAllEqual([3, 2], dist.batch_shape().eval())
      self.assertEqual(tf.TensorShape([2]), dist.get_event_shape())
      self.assertEqual(tf.TensorShape([3, 2]), dist.get_batch_shape())

  def testNProperty(self):
    p = [[0.1, 0.2, 0.7], [0.2, 0.3, 0.5]]
    n = [[3.], [4]]
    with self.test_session():
      dist = tf.contrib.distributions.Multinomial(n=n, p=p)
      self.assertEqual((2, 1), dist.n.get_shape())
      self.assertAllClose(n, dist.n.eval())

  def testPProperty(self):
    p = [[0.1, 0.2, 0.7]]
    with self.test_session():
      dist = tf.contrib.distributions.Multinomial(n=3., p=p)
      self.assertEqual((1, 3), dist.p.get_shape())
      self.assertEqual((1, 3), dist.logits.get_shape())
      self.assertAllClose(p, dist.p.eval())

  def testLogitsProperty(self):
    logits = [[0., 9., -0.5]]
    with self.test_session():
      multinom = tf.contrib.distributions.Multinomial(n=3., logits=logits)
      self.assertEqual((1, 3), multinom.p.get_shape())
      self.assertEqual((1, 3), multinom.logits.get_shape())
      self.assertAllClose(logits, multinom.logits.eval())

  def testPmfNandCountsAgree(self):
    p = [[0.1, 0.2, 0.7]]
    n = [[5.]]
    with self.test_session():
      dist = tf.contrib.distributions.Multinomial(
          n=n, p=p, validate_args=True)
      dist.pmf([2., 3, 0]).eval()
      dist.pmf([3., 0, 2]).eval()
      with self.assertRaisesOpError("Condition x >= 0.*"):
        dist.pmf([-1., 4, 2]).eval()
      with self.assertRaisesOpError("counts do not sum to n"):
        dist.pmf([3., 3, 0]).eval()

  def testPmf_non_integer_counts(self):
    p = [[0.1, 0.2, 0.7]]
    n = [[5.]]
    with self.test_session():
      # No errors with integer n.
      multinom = tf.contrib.distributions.Multinomial(
          n=n, p=p, validate_args=True)
      multinom.pmf([2., 1, 2]).eval()
      multinom.pmf([3., 0, 2]).eval()
      # Counts don't sum to n.
      with self.assertRaisesOpError("counts do not sum to n"):
        multinom.pmf([2., 3, 2]).eval()
      # Counts are non-integers.
      with self.assertRaisesOpError("Condition x == y.*"):
        multinom.pmf([1.0, 2.5, 1.5]).eval()

      multinom = tf.contrib.distributions.Multinomial(
          n=n, p=p, validate_args=False)
      multinom.pmf([1., 2., 2.]).eval()
      # Non-integer arguments work.
      multinom.pmf([1.0, 2.5, 1.5]).eval()

  def testPmfBothZeroBatches(self):
    with self.test_session():
      # Both zero-batches.  No broadcast
      p = [0.5, 0.5]
      counts = [1., 0]
      pmf = tf.contrib.distributions.Multinomial(n=1., p=p).pmf(counts)
      self.assertAllClose(0.5, pmf.eval())
      self.assertEqual((), pmf.get_shape())

  def testPmfBothZeroBatchesNontrivialN(self):
    with self.test_session():
      # Both zero-batches.  No broadcast
      p = [0.1, 0.9]
      counts = [3., 2]
      dist = tf.contrib.distributions.Multinomial(n=5., p=p)
      pmf = dist.pmf(counts)
      # 5 choose 3 = 5 choose 2 = 10. 10 * (.9)^2 * (.1)^3 = 81/10000.
      self.assertAllClose(81./10000, pmf.eval())
      self.assertEqual((), pmf.get_shape())

  def testPmfPStretchedInBroadcastWhenSameRank(self):
    with self.test_session():
      p = [[0.1, 0.9]]
      counts = [[1., 0], [0, 1]]
      pmf = tf.contrib.distributions.Multinomial(n=1., p=p).pmf(counts)
      self.assertAllClose([0.1, 0.9], pmf.eval())
      self.assertEqual((2), pmf.get_shape())

  def testPmfPStretchedInBroadcastWhenLowerRank(self):
    with self.test_session():
      p = [0.1, 0.9]
      counts = [[1., 0], [0, 1]]
      pmf = tf.contrib.distributions.Multinomial(n=1., p=p).pmf(counts)
      self.assertAllClose([0.1, 0.9], pmf.eval())
      self.assertEqual((2), pmf.get_shape())

  def testPmfCountsStretchedInBroadcastWhenSameRank(self):
    with self.test_session():
      p = [[0.1, 0.9], [0.7, 0.3]]
      counts = [[1., 0]]
      pmf = tf.contrib.distributions.Multinomial(n=1., p=p).pmf(counts)
      self.assertAllClose(pmf.eval(), [0.1, 0.7])
      self.assertEqual((2), pmf.get_shape())

  def testPmfCountsStretchedInBroadcastWhenLowerRank(self):
    with self.test_session():
      p = [[0.1, 0.9], [0.7, 0.3]]
      counts = [1., 0]
      pmf = tf.contrib.distributions.Multinomial(n=1., p=p).pmf(counts)
      self.assertAllClose(pmf.eval(), [0.1, 0.7])
      self.assertEqual(pmf.get_shape(), (2))

  def testPmfShapeCountsStretched_N(self):
    with self.test_session():
      # [2, 2, 2]
      p = [[[0.1, 0.9], [0.1, 0.9]], [[0.7, 0.3], [0.7, 0.3]]]
      # [2, 2]
      n = [[3., 3], [3, 3]]
      # [2]
      counts = [2., 1]
      pmf = tf.contrib.distributions.Multinomial(n=n, p=p).pmf(counts)
      pmf.eval()
      self.assertEqual(pmf.get_shape(), (2, 2))

  def testPmfShapeCountsPStretched_N(self):
    with self.test_session():
      p = [0.1, 0.9]
      counts = [3., 2]
      n = np.full([4, 3], 5., dtype=np.float32)
      pmf = tf.contrib.distributions.Multinomial(n=n, p=p).pmf(counts)
      pmf.eval()
      self.assertEqual((4, 3), pmf.get_shape())

  def testMultinomialMean(self):
    with self.test_session():
      n = 5.
      p = [0.1, 0.2, 0.7]
      dist = tf.contrib.distributions.Multinomial(n=n, p=p)
      expected_means = 5 * np.array(p, dtype=np.float32)
      self.assertEqual((3,), dist.mean().get_shape())
      self.assertAllClose(expected_means, dist.mean().eval())

  def testMultinomialVariance(self):
    with self.test_session():
      n = 5.
      p = [0.1, 0.2, 0.7]
      dist = tf.contrib.distributions.Multinomial(n=n, p=p)
      expected_variances = [
          [9./20, -1/10, -7/20], [-1/10, 4/5, -7/10], [-7/20, -7/10, 21/20]]
      self.assertEqual((3, 3), dist.variance().get_shape())
      self.assertAllClose(expected_variances, dist.variance().eval())

  def testMultinomialVariance_batch(self):
    with self.test_session():
      # Shape [2]
      n = [5.] * 2
      # Shape [4, 1, 2]
      p = [[[0.1, 0.9]], [[0.1, 0.9]]] * 2
      dist = tf.contrib.distributions.Multinomial(n=n, p=p)
      # Shape [2, 2]
      inner_var = [[9./20, -9/20], [-9/20, 9/20]]
      # Shape [4, 2, 2, 2]
      expected_variances = [[inner_var, inner_var]] * 4
      self.assertEqual((4, 2, 2, 2), dist.variance().get_shape())
      self.assertAllClose(expected_variances, dist.variance().eval())

  def testVariance_multidimensional(self):
    # Shape [3, 5, 4]
    p = np.random.dirichlet([.25, .25, .25, .25], [3, 5]).astype(np.float32)
    # Shape [6, 3, 3]
    p2 = np.random.dirichlet([.3, .3, .4], [6, 3]).astype(np.float32)

    ns = np.random.randint(low=1, high=11, size=[3, 5]).astype(np.float32)
    ns2 = np.random.randint(low=1, high=11, size=[6, 1]).astype(np.float32)

    with self.test_session():
      dist = tf.contrib.distributions.Multinomial(ns, p)
      dist2 = tf.contrib.distributions.Multinomial(ns2, p2)

      variance = dist.variance()
      variance2 = dist2.variance()
      self.assertEqual((3, 5, 4, 4), variance.get_shape())
      self.assertEqual((6, 3, 3, 3), variance2.get_shape())

if __name__ == "__main__":
  tf.test.main()
