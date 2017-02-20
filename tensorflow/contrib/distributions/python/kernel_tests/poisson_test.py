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
from tensorflow.contrib.distributions.python.ops import poisson as poisson_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import test


class PoissonTest(test.TestCase):

  def testPoissonShape(self):
    with self.test_session():
      lam = constant_op.constant([3.0] * 5)
      poisson = poisson_lib.Poisson(rate=lam)

      self.assertEqual(poisson.batch_shape_tensor().eval(), (5,))
      self.assertEqual(poisson.batch_shape, tensor_shape.TensorShape([5]))
      self.assertAllEqual(poisson.event_shape_tensor().eval(), [])
      self.assertEqual(poisson.event_shape, tensor_shape.TensorShape([]))

  def testInvalidLam(self):
    invalid_lams = [-.01, 0, -2.]
    for lam in invalid_lams:
      with self.test_session():
        with self.assertRaisesOpError("Condition x > 0"):
          poisson = poisson_lib.Poisson(rate=lam, validate_args=True)
          poisson.rate.eval()

  def testPoissonLogPmf(self):
    with self.test_session():
      batch_size = 6
      lam = constant_op.constant([3.0] * batch_size)
      lam_v = 3.0
      x = [2., 3., 4., 5., 6., 7.]
      poisson = poisson_lib.Poisson(rate=lam)
      log_pmf = poisson.log_prob(x)
      self.assertEqual(log_pmf.get_shape(), (6,))
      self.assertAllClose(log_pmf.eval(), stats.poisson.logpmf(x, lam_v))

      pmf = poisson.prob(x)
      self.assertEqual(pmf.get_shape(), (6,))
      self.assertAllClose(pmf.eval(), stats.poisson.pmf(x, lam_v))

  def testPoissonLogPmfValidateArgs(self):
    with self.test_session():
      batch_size = 6
      lam = constant_op.constant([3.0] * batch_size)
      x = [2.5, 3.2, 4.3, 5.1, 6., 7.]
      poisson = poisson_lib.Poisson(rate=lam, validate_args=True)

      # Non-integer
      with self.assertRaisesOpError("cannot contain fractional components"):
        log_pmf = poisson.log_prob(x)
        log_pmf.eval()

      with self.assertRaisesOpError("Condition x >= 0"):
        log_pmf = poisson.log_prob([-1.])
        log_pmf.eval()

      poisson = poisson_lib.Poisson(rate=lam, validate_args=False)
      log_pmf = poisson.log_prob(x)
      self.assertEqual(log_pmf.get_shape(), (6,))
      pmf = poisson.prob(x)
      self.assertEqual(pmf.get_shape(), (6,))

  def testPoissonLogPmfMultidimensional(self):
    with self.test_session():
      batch_size = 6
      lam = constant_op.constant([[2.0, 4.0, 5.0]] * batch_size)
      lam_v = [2.0, 4.0, 5.0]
      x = np.array([[2., 3., 4., 5., 6., 7.]], dtype=np.float32).T

      poisson = poisson_lib.Poisson(rate=lam)
      log_pmf = poisson.log_prob(x)
      self.assertEqual(log_pmf.get_shape(), (6, 3))
      self.assertAllClose(log_pmf.eval(), stats.poisson.logpmf(x, lam_v))

      pmf = poisson.prob(x)
      self.assertEqual(pmf.get_shape(), (6, 3))
      self.assertAllClose(pmf.eval(), stats.poisson.pmf(x, lam_v))

  def testPoissonCDF(self):
    with self.test_session():
      batch_size = 6
      lam = constant_op.constant([3.0] * batch_size)
      lam_v = 3.0
      x = [2.2, 3.1, 4., 5.5, 6., 7.]

      poisson = poisson_lib.Poisson(rate=lam)
      log_cdf = poisson.log_cdf(x)
      self.assertEqual(log_cdf.get_shape(), (6,))
      self.assertAllClose(log_cdf.eval(), stats.poisson.logcdf(x, lam_v))

      cdf = poisson.cdf(x)
      self.assertEqual(cdf.get_shape(), (6,))
      self.assertAllClose(cdf.eval(), stats.poisson.cdf(x, lam_v))

  def testPoissonCdfMultidimensional(self):
    with self.test_session():
      batch_size = 6
      lam = constant_op.constant([[2.0, 4.0, 5.0]] * batch_size)
      lam_v = [2.0, 4.0, 5.0]
      x = np.array([[2.2, 3.1, 4., 5.5, 6., 7.]], dtype=np.float32).T

      poisson = poisson_lib.Poisson(rate=lam)
      log_cdf = poisson.log_cdf(x)
      self.assertEqual(log_cdf.get_shape(), (6, 3))
      self.assertAllClose(log_cdf.eval(), stats.poisson.logcdf(x, lam_v))

      cdf = poisson.cdf(x)
      self.assertEqual(cdf.get_shape(), (6, 3))
      self.assertAllClose(cdf.eval(), stats.poisson.cdf(x, lam_v))

  def testPoissonMean(self):
    with self.test_session():
      lam_v = [1.0, 3.0, 2.5]
      poisson = poisson_lib.Poisson(rate=lam_v)
      self.assertEqual(poisson.mean().get_shape(), (3,))
      self.assertAllClose(poisson.mean().eval(), stats.poisson.mean(lam_v))
      self.assertAllClose(poisson.mean().eval(), lam_v)

  def testPoissonVariance(self):
    with self.test_session():
      lam_v = [1.0, 3.0, 2.5]
      poisson = poisson_lib.Poisson(rate=lam_v)
      self.assertEqual(poisson.variance().get_shape(), (3,))
      self.assertAllClose(poisson.variance().eval(), stats.poisson.var(lam_v))
      self.assertAllClose(poisson.variance().eval(), lam_v)

  def testPoissonStd(self):
    with self.test_session():
      lam_v = [1.0, 3.0, 2.5]
      poisson = poisson_lib.Poisson(rate=lam_v)
      self.assertEqual(poisson.stddev().get_shape(), (3,))
      self.assertAllClose(poisson.stddev().eval(), stats.poisson.std(lam_v))
      self.assertAllClose(poisson.stddev().eval(), np.sqrt(lam_v))

  def testPoissonMode(self):
    with self.test_session():
      lam_v = [1.0, 3.0, 2.5, 3.2, 1.1, 0.05]
      poisson = poisson_lib.Poisson(rate=lam_v)
      self.assertEqual(poisson.mode().get_shape(), (6,))
      self.assertAllClose(poisson.mode().eval(), np.floor(lam_v))

  def testPoissonMultipleMode(self):
    with self.test_session():
      lam_v = [1.0, 3.0, 2.0, 4.0, 5.0, 10.0]
      poisson = poisson_lib.Poisson(rate=lam_v)
      # For the case where lam is an integer, the modes are: lam and lam - 1.
      # In this case, we get back the larger of the two modes.
      self.assertEqual((6,), poisson.mode().get_shape())
      self.assertAllClose(lam_v, poisson.mode().eval())

  def testPoissonSample(self):
    with self.test_session():
      lam_v = 4.0
      lam = constant_op.constant(lam_v)
      # Choosing `n >= (k/rtol)**2, roughly ensures our sample mean should be
      # within `k` std. deviations of actual up to rtol precision.
      n = int(100e3)
      poisson = poisson_lib.Poisson(rate=lam)
      samples = poisson.sample(n, seed=123456)
      sample_values = samples.eval()
      self.assertEqual(samples.get_shape(), (n,))
      self.assertEqual(sample_values.shape, (n,))
      self.assertAllClose(
          sample_values.mean(), stats.poisson.mean(lam_v), rtol=.01)
      self.assertAllClose(
          sample_values.var(), stats.poisson.var(lam_v), rtol=.01)

  def testPoissonSampleMultidimensionalMean(self):
    with self.test_session():
      lam_v = np.array([np.arange(1, 51, dtype=np.float32)])  # 1 x 50
      poisson = poisson_lib.Poisson(rate=lam_v)
      # Choosing `n >= (k/rtol)**2, roughly ensures our sample mean should be
      # within `k` std. deviations of actual up to rtol precision.
      n = int(100e3)
      samples = poisson.sample(n, seed=123456)
      sample_values = samples.eval()
      self.assertEqual(samples.get_shape(), (n, 1, 50))
      self.assertEqual(sample_values.shape, (n, 1, 50))
      self.assertAllClose(
          sample_values.mean(axis=0),
          stats.poisson.mean(lam_v),
          rtol=.01,
          atol=0)

  def testPoissonSampleMultidimensionalVariance(self):
    with self.test_session():
      lam_v = np.array([np.arange(5, 15, dtype=np.float32)])  # 1 x 10
      poisson = poisson_lib.Poisson(rate=lam_v)
      # Choosing `n >= 2 * lam * (k/rtol)**2, roughly ensures our sample
      # variance should be within `k` std. deviations of actual up to rtol
      # precision.
      n = int(300e3)
      samples = poisson.sample(n, seed=123456)
      sample_values = samples.eval()
      self.assertEqual(samples.get_shape(), (n, 1, 10))
      self.assertEqual(sample_values.shape, (n, 1, 10))

      self.assertAllClose(
          sample_values.var(axis=0), stats.poisson.var(lam_v), rtol=.03, atol=0)


if __name__ == "__main__":
  test.main()
