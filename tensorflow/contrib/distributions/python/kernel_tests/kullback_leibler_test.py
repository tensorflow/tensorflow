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


class KLTest(tf.test.TestCase):

  def testRegistration(self):
    class MyDist(tf.contrib.distributions.Normal):
      pass

    # Register KL to a lambda that spits out the name parameter
    @tf.contrib.distributions.RegisterKL(MyDist, MyDist)
    def _kl(unused_a, unused_b, name=None):  # pylint: disable=unused-variable
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
    # pylint: disable=unused-variable
    def _kl(unused_a, unused_b, name=None):  # pylint: disable=unused-argument
      return tf.identity([float("nan")])
    # pylint: disable=unused-variable

    with self.test_session():
      a = MyDistException(mu=0.0, sigma=1.0)
      kl = tf.contrib.distributions.kl(a, a)
      with self.assertRaisesOpError(
          "KL calculation between .* and .* returned NaN values"):
        kl.eval()
      kl_ok = tf.contrib.distributions.kl(a, a, allow_nan=True)
      self.assertAllEqual([float("nan")], kl_ok.eval())

  def testRegistrationFailures(self):
    with self.assertRaisesRegexp(TypeError, "is not a subclass of"):
      tf.contrib.distributions.RegisterKL(
          tf.contrib.distributions.Normal, object)(lambda x: x)
    with self.assertRaisesRegexp(TypeError, "is not a subclass of"):
      tf.contrib.distributions.RegisterKL(
          object, tf.contrib.distributions.Normal)(lambda x: x)

    class MyDist(tf.contrib.distributions.Normal):
      pass

    with self.assertRaisesRegexp(TypeError, "must be callable"):
      tf.contrib.distributions.RegisterKL(MyDist, MyDist)("blah")

    # First registration is OK
    tf.contrib.distributions.RegisterKL(MyDist, MyDist)(lambda a, b: None)

    # Second registration fails
    with self.assertRaisesRegexp(ValueError, "has already been registered"):
      tf.contrib.distributions.RegisterKL(MyDist, MyDist)(lambda a, b: None)


if __name__ == "__main__":
  tf.test.main()
