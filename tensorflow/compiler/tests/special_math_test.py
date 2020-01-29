# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for special math operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import flags
from absl.testing import parameterized

import numpy as np
import scipy.special as sps
import six

from tensorflow.compiler.tests import xla_test
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test

flags.DEFINE_bool('vary_seed', False,
                  ('Whether to vary the PRNG seed unpredictably.  '
                   'With --runs_per_test=N, produces N iid runs.'))

NUM_SAMPLES = int(1e3)


class IgammaTest(xla_test.XLATestCase, parameterized.TestCase):

  def setUp(self):
    if flags.FLAGS.vary_seed:
      entropy = os.urandom(64)
      if six.PY2:
        answer = int(entropy.encode('hex'), 16)
      else:
        answer = int.from_bytes(entropy, 'big')
      np.random.seed(answer)
    super(IgammaTest, self).setUp()

  @parameterized.parameters((np.float32, 1e-2, 1e-11),
                            (np.float64, 1e-4, 1e-30))
  def testIgammaSmallValues(self, dtype, rtol, atol):
    # Test values near zero.
    x = np.random.uniform(
        low=np.finfo(dtype).tiny, high=1., size=[NUM_SAMPLES]).astype(dtype)
    a = np.random.uniform(
        low=np.finfo(dtype).tiny, high=1., size=[NUM_SAMPLES]).astype(dtype)

    expected_values = sps.gammainc(a, x)
    with self.session() as sess:
      with self.test_scope():
        actual = sess.run(math_ops.igamma(a, x))
    self.assertAllClose(expected_values, actual, atol=atol, rtol=rtol)

  @parameterized.parameters((np.float32, 1e-2, 1e-11),
                            (np.float64, 1e-4, 1e-30))
  def testIgammaMediumValues(self, dtype, rtol, atol):
    # Test values near zero.
    x = np.random.uniform(low=1., high=100., size=[NUM_SAMPLES]).astype(dtype)
    a = np.random.uniform(low=1., high=100., size=[NUM_SAMPLES]).astype(dtype)

    expected_values = sps.gammainc(a, x)
    with self.session() as sess:
      with self.test_scope():
        actual = sess.run(math_ops.igamma(a, x))
    self.assertAllClose(expected_values, actual, atol=atol, rtol=rtol)

  @parameterized.parameters((np.float32, 2e-2, 1e-5), (np.float64, 1e-4, 1e-30))
  def testIgammaLargeValues(self, dtype, rtol, atol):
    # Test values near zero.
    x = np.random.uniform(
        low=100., high=int(1e4), size=[NUM_SAMPLES]).astype(dtype)
    a = np.random.uniform(
        low=100., high=int(1e4), size=[NUM_SAMPLES]).astype(dtype)

    expected_values = sps.gammainc(a, x)
    with self.session() as sess:
      with self.test_scope():
        actual = sess.run(math_ops.igamma(a, x))
    self.assertAllClose(expected_values, actual, atol=atol, rtol=rtol)


if __name__ == '__main__':
  os.environ['XLA_FLAGS'] = '--xla_cpu_enable_fast_math=false'
  test.main()
