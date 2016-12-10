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
"""Tests for distributions KL mechanism."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.contrib.distributions.python.ops import kullback_leibler

# pylint: disable=protected-access
_DIVERGENCES = kullback_leibler._DIVERGENCES
_registered_kl = kullback_leibler._registered_kl
# pylint: enable=protected-access


class KLTest(tf.test.TestCase):

  def testRegistration(self):
    class MyDist(tf.contrib.distributions.Normal):
      pass

    # Register KL to a lambda that spits out the name parameter
    @tf.contrib.distributions.RegisterKL(MyDist, MyDist)
    def _kl(a, b, name=None):  # pylint: disable=unused-argument,unused-variable
      return name

    a = MyDist(mu=0.0, sigma=1.0)
    # Run kl() with allow_nan=True because strings can't go through is_nan.
    self.assertEqual(
        "OK", tf.contrib.distributions.kl(a, a, allow_nan=True, name="OK"))

  def testDomainErrorExceptions(self):
    class MyDistException(tf.contrib.distributions.Normal):
      pass

    # Register KL to a lambda that spits out the name parameter
    @tf.contrib.distributions.RegisterKL(MyDistException, MyDistException)
    # pylint: disable=unused-argument,unused-variable
    def _kl(a, b, name=None):
      return tf.identity([float("nan")])
    # pylint: disable=unused-argument,unused-variable

    with self.test_session():
      a = MyDistException(mu=0.0, sigma=1.0)
      kl = tf.contrib.distributions.kl(a, a)
      with self.assertRaisesOpError(
          "KL calculation between .* and .* returned NaN values"):
        kl.eval()
      kl_ok = tf.contrib.distributions.kl(a, a, allow_nan=True)
      self.assertAllEqual([float("nan")], kl_ok.eval())

  def testRegistrationFailures(self):
    class MyDist(tf.contrib.distributions.Normal):
      pass

    with self.assertRaisesRegexp(TypeError, "must be callable"):
      tf.contrib.distributions.RegisterKL(MyDist, MyDist)("blah")

    # First registration is OK
    tf.contrib.distributions.RegisterKL(MyDist, MyDist)(lambda a, b: None)

    # Second registration fails
    with self.assertRaisesRegexp(ValueError, "has already been registered"):
      tf.contrib.distributions.RegisterKL(MyDist, MyDist)(lambda a, b: None)

  def testExactRegistrationsAllMatch(self):
    for (k, v) in _DIVERGENCES.items():
      self.assertEqual(v, _registered_kl(*k))

  def testIndirectRegistration(self):
    class Sub1(tf.contrib.distributions.Normal):
      pass

    class Sub2(tf.contrib.distributions.Normal):
      pass

    class Sub11(Sub1):
      pass

    # pylint: disable=unused-argument,unused-variable
    @tf.contrib.distributions.RegisterKL(Sub1, Sub1)
    def _kl11(a, b, name=None):
      return "sub1-1"

    @tf.contrib.distributions.RegisterKL(Sub1, Sub2)
    def _kl12(a, b, name=None):
      return "sub1-2"

    @tf.contrib.distributions.RegisterKL(Sub2, Sub1)
    def _kl21(a, b, name=None):
      return "sub2-1"
    # pylint: enable=unused-argument,unused_variable

    sub1 = Sub1(mu=0.0, sigma=1.0)
    sub2 = Sub2(mu=0.0, sigma=1.0)
    sub11 = Sub11(mu=0.0, sigma=1.0)

    self.assertEqual(
        "sub1-1", tf.contrib.distributions.kl(sub1, sub1, allow_nan=True))
    self.assertEqual(
        "sub1-2", tf.contrib.distributions.kl(sub1, sub2, allow_nan=True))
    self.assertEqual(
        "sub2-1", tf.contrib.distributions.kl(sub2, sub1, allow_nan=True))
    self.assertEqual(
        "sub1-1", tf.contrib.distributions.kl(sub11, sub11, allow_nan=True))
    self.assertEqual(
        "sub1-1", tf.contrib.distributions.kl(sub11, sub1, allow_nan=True))
    self.assertEqual(
        "sub1-2", tf.contrib.distributions.kl(sub11, sub2, allow_nan=True))
    self.assertEqual(
        "sub1-1", tf.contrib.distributions.kl(sub11, sub1, allow_nan=True))
    self.assertEqual(
        "sub1-2", tf.contrib.distributions.kl(sub11, sub2, allow_nan=True))
    self.assertEqual(
        "sub2-1", tf.contrib.distributions.kl(sub2, sub11, allow_nan=True))
    self.assertEqual(
        "sub1-1", tf.contrib.distributions.kl(sub1, sub11, allow_nan=True))


if __name__ == "__main__":
  tf.test.main()
