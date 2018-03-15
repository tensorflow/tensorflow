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
"""Tests for Bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import six

from tensorflow.python.framework import constant_op
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import bijector
from tensorflow.python.platform import test


class BaseBijectorTest(test.TestCase):
  """Tests properties of the Bijector base-class."""

  def testIsAbstract(self):
    with self.test_session():
      with self.assertRaisesRegexp(TypeError,
                                   ("Can't instantiate abstract class Bijector "
                                    "with abstract methods __init__")):
        bijector.Bijector()  # pylint: disable=abstract-class-instantiated

  def testDefaults(self):
    class _BareBonesBijector(bijector.Bijector):
      """Minimal specification of a `Bijector`."""

      def __init__(self):
        super(_BareBonesBijector, self).__init__()

    with self.test_session() as sess:
      bij = _BareBonesBijector()
      self.assertEqual(None, bij.event_ndims)
      self.assertEqual([], bij.graph_parents)
      self.assertEqual(False, bij.is_constant_jacobian)
      self.assertEqual(False, bij.validate_args)
      self.assertEqual(None, bij.dtype)
      self.assertEqual("bare_bones_bijector", bij.name)

      for shape in [[], [1, 2], [1, 2, 3]]:
        [
            forward_event_shape_,
            inverse_event_shape_,
        ] = sess.run([
            bij.inverse_event_shape_tensor(shape),
            bij.forward_event_shape_tensor(shape),
        ])
        self.assertAllEqual(shape, forward_event_shape_)
        self.assertAllEqual(shape, bij.forward_event_shape(shape))
        self.assertAllEqual(shape, inverse_event_shape_)
        self.assertAllEqual(shape, bij.inverse_event_shape(shape))

      for fn in ["forward",
                 "inverse",
                 "inverse_log_det_jacobian",
                 "forward_log_det_jacobian"]:
        with self.assertRaisesRegexp(
            NotImplementedError, fn + " not implemented"):
          getattr(bij, fn)(0)


class IntentionallyMissingError(Exception):
  pass


class BrokenBijector(bijector.Bijector):
  """Forward and inverse are not inverses of each other."""

  def __init__(self, forward_missing=False, inverse_missing=False):
    super(BrokenBijector, self).__init__(
        event_ndims=0, validate_args=False, name="broken")
    self._forward_missing = forward_missing
    self._inverse_missing = inverse_missing

  def _forward(self, x):
    if self._forward_missing:
      raise IntentionallyMissingError
    return 2 * x

  def _inverse(self, y):
    if self._inverse_missing:
      raise IntentionallyMissingError
    return y / 2.

  def _inverse_log_det_jacobian(self, y):  # pylint:disable=unused-argument
    if self._inverse_missing:
      raise IntentionallyMissingError
    return -math_ops.log(2.)

  def _forward_log_det_jacobian(self, x):  # pylint:disable=unused-argument
    if self._forward_missing:
      raise IntentionallyMissingError
    return math_ops.log(2.)


@six.add_metaclass(abc.ABCMeta)
class BijectorCachingTestBase(object):

  @abc.abstractproperty
  def broken_bijector_cls(self):
    # return a BrokenBijector type Bijector, since this will test the caching.
    raise IntentionallyMissingError("Not implemented")

  def testCachingOfForwardResults(self):
    broken_bijector = self.broken_bijector_cls(inverse_missing=True)
    with self.test_session():
      x = constant_op.constant(1.1)

      # Call forward and forward_log_det_jacobian one-by-one (not together).
      y = broken_bijector.forward(x)
      _ = broken_bijector.forward_log_det_jacobian(x)

      # Now, everything should be cached if the argument is y.
      try:
        broken_bijector.inverse(y)
        broken_bijector.inverse_log_det_jacobian(y)
      except IntentionallyMissingError:
        raise AssertionError("Tests failed! Cached values not used.")

  def testCachingOfInverseResults(self):
    broken_bijector = self.broken_bijector_cls(forward_missing=True)
    with self.test_session():
      y = constant_op.constant(1.1)

      # Call inverse and inverse_log_det_jacobian one-by-one (not together).
      x = broken_bijector.inverse(y)
      _ = broken_bijector.inverse_log_det_jacobian(y)

      # Now, everything should be cached if the argument is x.
      try:
        broken_bijector.forward(x)
        broken_bijector.forward_log_det_jacobian(x)
      except IntentionallyMissingError:
        raise AssertionError("Tests failed! Cached values not used.")


class BijectorCachingTest(BijectorCachingTestBase, test.TestCase):
  """Test caching with BrokenBijector."""

  @property
  def broken_bijector_cls(self):
    return BrokenBijector


if __name__ == "__main__":
  test.main()
