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
"""Tests for Substr op from string_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensorflow.python.framework import errors_impl
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import test


class SubstrOpTest(test.TestCase, parameterized.TestCase):

  def _testScalarString(self, dtype):
    test_string = b"Hello"
    position = np.array(1, dtype)
    length = np.array(3, dtype)
    expected_value = b"ell"

    substr_op = string_ops.substr(test_string, position, length)
    with self.cached_session():
      substr = substr_op.eval()
      self.assertAllEqual(substr, expected_value)

    # Negative position.
    test_string = b"Hello"
    position = np.array(-4, dtype)
    length = np.array(3, dtype)
    expected_value = b"ell"

    substr_op = string_ops.substr(test_string, position, length)
    with self.cached_session():
      substr = substr_op.eval()
      self.assertAllEqual(substr, expected_value)

    # Position is equal to the length of string.
    test_string = b""
    position = np.array(0, dtype)
    length = np.array(2, dtype)
    expected_value = b""

    substr_op = string_ops.substr(test_string, position, length)
    with self.cached_session():
      substr = substr_op.eval()
      self.assertAllEqual(substr, expected_value)

    # Negative position magnitude is equal to the length of string.
    test_string = b"yo"
    position = np.array(-2, dtype)
    length = np.array(1, dtype)
    expected_value = b"y"

    substr_op = string_ops.substr(test_string, position, length)
    with self.cached_session():
      substr = substr_op.eval()
      self.assertAllEqual(substr, expected_value)

  def _testVectorStrings(self, dtype):
    test_string = [b"Hello", b"World"]
    position = np.array(1, dtype)
    length = np.array(3, dtype)
    expected_value = [b"ell", b"orl"]

    substr_op = string_ops.substr(test_string, position, length)
    with self.cached_session():
      substr = substr_op.eval()
      self.assertAllEqual(substr, expected_value)

    # Negative position.
    test_string = [b"Hello", b"World"]
    position = np.array(-4, dtype)
    length = np.array(3, dtype)
    expected_value = [b"ell", b"orl"]

    substr_op = string_ops.substr(test_string, position, length)
    with self.cached_session():
      substr = substr_op.eval()
      self.assertAllEqual(substr, expected_value)

  def _testMatrixStrings(self, dtype):
    test_string = [[b"ten", b"eleven", b"twelve"],
                   [b"thirteen", b"fourteen", b"fifteen"],
                   [b"sixteen", b"seventeen", b"eighteen"]]
    position = np.array(1, dtype)
    length = np.array(4, dtype)
    expected_value = [[b"en", b"leve", b"welv"], [b"hirt", b"ourt", b"ifte"],
                      [b"ixte", b"even", b"ight"]]

    substr_op = string_ops.substr(test_string, position, length)
    with self.cached_session():
      substr = substr_op.eval()
      self.assertAllEqual(substr, expected_value)

    # Negative position
    test_string = [[b"ten", b"eleven", b"twelve"],
                   [b"thirteen", b"fourteen", b"fifteen"],
                   [b"sixteen", b"seventeen", b"eighteen"]]
    position = np.array(-2, dtype)
    length = np.array(2, dtype)
    expected_value = [[b"en", b"en", b"ve"], [b"en", b"en", b"en"],
                      [b"en", b"en", b"en"]]

    substr_op = string_ops.substr(test_string, position, length)
    with self.cached_session():
      substr = substr_op.eval()
      self.assertAllEqual(substr, expected_value)

  def _testElementWisePosLen(self, dtype):
    test_string = [[b"ten", b"eleven", b"twelve"],
                   [b"thirteen", b"fourteen", b"fifteen"],
                   [b"sixteen", b"seventeen", b"eighteen"]]
    position = np.array([[1, -4, 3], [1, 2, -4], [-5, 2, 3]], dtype)
    length = np.array([[2, 2, 4], [4, 3, 2], [5, 5, 5]], dtype)
    expected_value = [[b"en", b"ev", b"lve"], [b"hirt", b"urt", b"te"],
                      [b"xteen", b"vente", b"hteen"]]

    substr_op = string_ops.substr(test_string, position, length)
    with self.cached_session():
      substr = substr_op.eval()
      self.assertAllEqual(substr, expected_value)

  def _testBroadcast(self, dtype):
    # Broadcast pos/len onto input string
    test_string = [[b"ten", b"eleven", b"twelve"],
                   [b"thirteen", b"fourteen", b"fifteen"],
                   [b"sixteen", b"seventeen", b"eighteen"],
                   [b"nineteen", b"twenty", b"twentyone"]]
    position = np.array([1, -4, 3], dtype)
    length = np.array([1, 2, 3], dtype)
    expected_value = [[b"e", b"ev", b"lve"], [b"h", b"te", b"tee"],
                      [b"i", b"te", b"hte"], [b"i", b"en", b"nty"]]
    substr_op = string_ops.substr(test_string, position, length)
    with self.cached_session():
      substr = substr_op.eval()
      self.assertAllEqual(substr, expected_value)

    # Broadcast input string onto pos/len
    test_string = [b"thirteen", b"fourteen", b"fifteen"]
    position = np.array([[1, -2, 3], [-3, 2, 1], [5, 5, -5]], dtype)
    length = np.array([[3, 2, 1], [1, 2, 3], [2, 2, 2]], dtype)
    expected_value = [[b"hir", b"en", b"t"], [b"e", b"ur", b"ift"],
                      [b"ee", b"ee", b"ft"]]
    substr_op = string_ops.substr(test_string, position, length)
    with self.cached_session():
      substr = substr_op.eval()
      self.assertAllEqual(substr, expected_value)

    # Test 1D broadcast
    test_string = b"thirteen"
    position = np.array([1, -5, 7], dtype)
    length = np.array([3, 2, 1], dtype)
    expected_value = [b"hir", b"rt", b"n"]
    substr_op = string_ops.substr(test_string, position, length)
    with self.cached_session():
      substr = substr_op.eval()
      self.assertAllEqual(substr, expected_value)

  def _testBadBroadcast(self, dtype):
    test_string = [[b"ten", b"eleven", b"twelve"],
                   [b"thirteen", b"fourteen", b"fifteen"],
                   [b"sixteen", b"seventeen", b"eighteen"]]
    position = np.array([1, 2, -3, 4], dtype)
    length = np.array([1, 2, 3, 4], dtype)
    with self.assertRaises(ValueError):
      substr_op = string_ops.substr(test_string, position, length)

  def _testOutOfRangeError(self, dtype):
    # Scalar/Scalar
    test_string = b"Hello"
    position = np.array(7, dtype)
    length = np.array(3, dtype)
    substr_op = string_ops.substr(test_string, position, length)
    with self.cached_session():
      with self.assertRaises(errors_impl.InvalidArgumentError):
        substr = substr_op.eval()

    # Scalar/Scalar (with negative)
    test_string = b"Hello"
    position = np.array(-7, dtype)
    length = np.array(3, dtype)
    substr_op = string_ops.substr(test_string, position, length)
    with self.cached_session():
      with self.assertRaises(errors_impl.InvalidArgumentError):
        substr = substr_op.eval()

    # Vector/Scalar
    test_string = [b"good", b"good", b"bad", b"good"]
    position = np.array(4, dtype)
    length = np.array(1, dtype)
    substr_op = string_ops.substr(test_string, position, length)
    with self.cached_session():
      with self.assertRaises(errors_impl.InvalidArgumentError):
        substr = substr_op.eval()

    # Vector/Scalar (with negative)
    test_string = [b"good", b"good", b"bad", b"good"]
    position = np.array(-4, dtype)
    length = np.array(1, dtype)
    substr_op = string_ops.substr(test_string, position, length)
    with self.cached_session():
      with self.assertRaises(errors_impl.InvalidArgumentError):
        substr = substr_op.eval()

    # Matrix/Matrix
    test_string = [[b"good", b"good", b"good"], [b"good", b"good", b"bad"],
                   [b"good", b"good", b"good"]]
    position = np.array([[1, 2, 3], [1, 2, 4], [1, 2, 3]], dtype)
    length = np.array([[3, 2, 1], [1, 2, 3], [2, 2, 2]], dtype)
    substr_op = string_ops.substr(test_string, position, length)
    with self.cached_session():
      with self.assertRaises(errors_impl.InvalidArgumentError):
        substr = substr_op.eval()

    # Matrix/Matrix (with negative)
    test_string = [[b"good", b"good", b"good"], [b"good", b"good", b"bad"],
                   [b"good", b"good", b"good"]]
    position = np.array([[1, 2, -3], [1, 2, -4], [1, 2, -3]], dtype)
    length = np.array([[3, 2, 1], [1, 2, 3], [2, 2, 2]], dtype)
    substr_op = string_ops.substr(test_string, position, length)
    with self.cached_session():
      with self.assertRaises(errors_impl.InvalidArgumentError):
        substr = substr_op.eval()

    # Broadcast
    test_string = [[b"good", b"good", b"good"], [b"good", b"good", b"bad"]]
    position = np.array([1, 2, 4], dtype)
    length = np.array([1, 2, 3], dtype)
    substr_op = string_ops.substr(test_string, position, length)
    with self.cached_session():
      with self.assertRaises(errors_impl.InvalidArgumentError):
        substr = substr_op.eval()

    # Broadcast (with negative)
    test_string = [[b"good", b"good", b"good"], [b"good", b"good", b"bad"]]
    position = np.array([-1, -2, -4], dtype)
    length = np.array([1, 2, 3], dtype)
    substr_op = string_ops.substr(test_string, position, length)
    with self.cached_session():
      with self.assertRaises(errors_impl.InvalidArgumentError):
        substr = substr_op.eval()

  def _testMismatchPosLenShapes(self, dtype):
    test_string = [[b"ten", b"eleven", b"twelve"],
                   [b"thirteen", b"fourteen", b"fifteen"],
                   [b"sixteen", b"seventeen", b"eighteen"]]
    position = np.array([[1, 2, 3]], dtype)
    length = np.array([2, 3, 4], dtype)
    # Should fail: position/length have different rank
    with self.assertRaises(ValueError):
      substr_op = string_ops.substr(test_string, position, length)

    position = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]], dtype)
    length = np.array([[2, 3, 4]], dtype)
    # Should fail: position/length have different dimensionality
    with self.assertRaises(ValueError):
      substr_op = string_ops.substr(test_string, position, length)

    # Negative position.
    test_string = [[b"ten", b"eleven", b"twelve"],
                   [b"thirteen", b"fourteen", b"fifteen"],
                   [b"sixteen", b"seventeen", b"eighteen"]]
    position = np.array([[-1, -2, -3]], dtype)
    length = np.array([1, 2, 3], dtype)
    # Should fail: position/length have different rank
    with self.assertRaises(ValueError):
      substr_op = string_ops.substr(test_string, position, length)

  @parameterized.parameters(np.int32, np.int64)
  def testAll(self, dtype):
    self._testScalarString(dtype)
    self._testVectorStrings(dtype)
    self._testMatrixStrings(dtype)
    self._testElementWisePosLen(dtype)
    self._testBroadcast(dtype)
    self._testBadBroadcast(dtype)
    self._testOutOfRangeError(dtype)
    self._testMismatchPosLenShapes(dtype)

  def testWrongDtype(self):
    with self.cached_session():
      with self.assertRaises(TypeError):
        string_ops.substr(b"test", 3.0, 1)
      with self.assertRaises(TypeError):
        string_ops.substr(b"test", 3, 1.0)


if __name__ == "__main__":
  test.main()
