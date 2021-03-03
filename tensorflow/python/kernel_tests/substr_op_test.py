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
from tensorflow.python.framework import test_util
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import test


class SubstrOpTest(test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      (np.int32, 1, "BYTE"),
      (np.int64, 1, "BYTE"),
      (np.int32, -4, "BYTE"),
      (np.int64, -4, "BYTE"),
      (np.int32, 1, "UTF8_CHAR"),
      (np.int64, 1, "UTF8_CHAR"),
      (np.int32, -4, "UTF8_CHAR"),
      (np.int64, -4, "UTF8_CHAR"),
  )
  def testScalarString(self, dtype, pos, unit):
    test_string = {
        "BYTE": b"Hello",
        "UTF8_CHAR": u"He\xc3\xc3\U0001f604".encode("utf-8"),
    }[unit]
    expected_value = {
        "BYTE": b"ell",
        "UTF8_CHAR": u"e\xc3\xc3".encode("utf-8"),
    }[unit]
    position = np.array(pos, dtype)
    length = np.array(3, dtype)
    substr_op = string_ops.substr(test_string, position, length, unit=unit)
    with self.cached_session():
      substr = self.evaluate(substr_op)
      self.assertAllEqual(substr, expected_value)

  @parameterized.parameters(
      (np.int32, "BYTE"),
      (np.int64, "BYTE"),
      (np.int32, "UTF8_CHAR"),
      (np.int64, "UTF8_CHAR"),
  )
  def testScalarString_EdgeCases(self, dtype, unit):
    # Empty string
    test_string = {
        "BYTE": b"",
        "UTF8_CHAR": u"".encode("utf-8"),
    }[unit]
    expected_value = b""
    position = np.array(0, dtype)
    length = np.array(3, dtype)
    substr_op = string_ops.substr(test_string, position, length, unit=unit)
    with self.cached_session():
      substr = self.evaluate(substr_op)
      self.assertAllEqual(substr, expected_value)

    # Full string
    test_string = {
        "BYTE": b"Hello",
        "UTF8_CHAR": u"H\xc3ll\U0001f604".encode("utf-8"),
    }[unit]
    position = np.array(0, dtype)
    length = np.array(5, dtype)
    substr_op = string_ops.substr(test_string, position, length, unit=unit)
    with self.cached_session():
      substr = self.evaluate(substr_op)
      self.assertAllEqual(substr, test_string)

    # Full string (Negative)
    test_string = {
        "BYTE": b"Hello",
        "UTF8_CHAR": u"H\xc3ll\U0001f604".encode("utf-8"),
    }[unit]
    position = np.array(-5, dtype)
    length = np.array(5, dtype)
    substr_op = string_ops.substr(test_string, position, length, unit=unit)
    with self.cached_session():
      substr = self.evaluate(substr_op)
      self.assertAllEqual(substr, test_string)

    # Length is larger in magnitude than a negative position
    test_string = {
        "BYTE": b"Hello",
        "UTF8_CHAR": u"H\xc3ll\U0001f604".encode("utf-8"),
    }[unit]
    expected_string = {
        "BYTE": b"ello",
        "UTF8_CHAR": u"\xc3ll\U0001f604".encode("utf-8"),
    }[unit]
    position = np.array(-4, dtype)
    length = np.array(5, dtype)
    substr_op = string_ops.substr(test_string, position, length, unit=unit)
    with self.cached_session():
      substr = self.evaluate(substr_op)
      self.assertAllEqual(substr, expected_string)

  @parameterized.parameters(
      (np.int32, 1, "BYTE"),
      (np.int64, 1, "BYTE"),
      (np.int32, -4, "BYTE"),
      (np.int64, -4, "BYTE"),
      (np.int32, 1, "UTF8_CHAR"),
      (np.int64, 1, "UTF8_CHAR"),
      (np.int32, -4, "UTF8_CHAR"),
      (np.int64, -4, "UTF8_CHAR"),
  )
  def testVectorStrings(self, dtype, pos, unit):
    test_string = {
        "BYTE": [b"Hello", b"World"],
        "UTF8_CHAR": [x.encode("utf-8") for x in [u"H\xc3llo",
                                                  u"W\U0001f604rld"]],
    }[unit]
    expected_value = {
        "BYTE": [b"ell", b"orl"],
        "UTF8_CHAR": [x.encode("utf-8") for x in [u"\xc3ll", u"\U0001f604rl"]],
    }[unit]
    position = np.array(pos, dtype)
    length = np.array(3, dtype)
    substr_op = string_ops.substr(test_string, position, length, unit=unit)
    with self.cached_session():
      substr = self.evaluate(substr_op)
      self.assertAllEqual(substr, expected_value)

  @parameterized.parameters(
      (np.int32, "BYTE"),
      (np.int64, "BYTE"),
      (np.int32, "UTF8_CHAR"),
      (np.int64, "UTF8_CHAR"),
  )
  def testMatrixStrings(self, dtype, unit):
    test_string = {
        "BYTE": [[b"ten", b"eleven", b"twelve"],
                 [b"thirteen", b"fourteen", b"fifteen"],
                 [b"sixteen", b"seventeen", b"eighteen"]],
        "UTF8_CHAR": [[x.encode("utf-8") for x in [u"\U0001d229\U0001d227n",
                                                   u"\xc6\u053c\u025bv\u025bn",
                                                   u"tw\u0c1dlv\u025b"]],
                      [x.encode("utf-8") for x in [u"He\xc3\xc3o",
                                                   u"W\U0001f604rld",
                                                   u"d\xfcd\xea"]]],
    }[unit]
    position = np.array(1, dtype)
    length = np.array(4, dtype)
    expected_value = {
        "BYTE": [[b"en", b"leve", b"welv"], [b"hirt", b"ourt", b"ifte"],
                 [b"ixte", b"even", b"ight"]],
        "UTF8_CHAR": [[x.encode("utf-8") for x in [u"\U0001d227n",
                                                   u"\u053c\u025bv\u025b",
                                                   u"w\u0c1dlv"]],
                      [x.encode("utf-8") for x in [u"e\xc3\xc3o",
                                                   u"\U0001f604rld",
                                                   u"\xfcd\xea"]]],
    }[unit]
    substr_op = string_ops.substr(test_string, position, length, unit=unit)
    with self.cached_session():
      substr = self.evaluate(substr_op)
      self.assertAllEqual(substr, expected_value)

    position = np.array(-3, dtype)
    length = np.array(2, dtype)
    expected_value = {
        "BYTE": [[b"te", b"ve", b"lv"], [b"ee", b"ee", b"ee"],
                 [b"ee", b"ee", b"ee"]],
        "UTF8_CHAR": [[x.encode("utf-8") for x in [u"\U0001d229\U0001d227",
                                                   u"v\u025b", u"lv"]],
                      [x.encode("utf-8") for x in [u"\xc3\xc3", u"rl",
                                                   u"\xfcd"]]],
    }[unit]
    substr_op = string_ops.substr(test_string, position, length, unit=unit)
    with self.cached_session():
      substr = self.evaluate(substr_op)
      self.assertAllEqual(substr, expected_value)

  @parameterized.parameters(
      (np.int32, "BYTE"),
      (np.int64, "BYTE"),
      (np.int32, "UTF8_CHAR"),
      (np.int64, "UTF8_CHAR"),
  )
  def testElementWisePosLen(self, dtype, unit):
    test_string = {
        "BYTE": [[b"ten", b"eleven", b"twelve"],
                 [b"thirteen", b"fourteen", b"fifteen"],
                 [b"sixteen", b"seventeen", b"eighteen"]],
        "UTF8_CHAR": [[x.encode("utf-8") for x in [u"\U0001d229\U0001d227n",
                                                   u"\xc6\u053c\u025bv\u025bn",
                                                   u"tw\u0c1dlv\u025b"]],
                      [x.encode("utf-8") for x in [u"He\xc3\xc3o",
                                                   u"W\U0001f604rld",
                                                   u"d\xfcd\xea"]],
                      [x.encode("utf-8") for x in [u"sixt\xea\xean",
                                                   u"se\U00010299enteen",
                                                   u"ei\U0001e920h\x86een"]]],
    }[unit]
    position = np.array([[1, -4, 3], [1, 2, -4], [-5, 2, 3]], dtype)
    length = np.array([[2, 2, 4], [4, 3, 2], [5, 5, 5]], dtype)
    expected_value = {
        "BYTE": [[b"en", b"ev", b"lve"], [b"hirt", b"urt", b"te"],
                 [b"xteen", b"vente", b"hteen"]],
        "UTF8_CHAR": [[x.encode("utf-8") for x in [u"\U0001d227n",
                                                   u"\u025bv",
                                                   u"lv\u025b"]],
                      [x.encode("utf-8") for x in [u"e\xc3\xc3o",
                                                   u"rld",
                                                   u"d\xfc"]],
                      [x.encode("utf-8") for x in [u"xt\xea\xean",
                                                   u"\U00010299ente",
                                                   u"h\x86een"]]],
    }[unit]
    substr_op = string_ops.substr(test_string, position, length, unit=unit)
    with self.cached_session():
      substr = self.evaluate(substr_op)
      self.assertAllEqual(substr, expected_value)

  @parameterized.parameters(
      (np.int32, "BYTE"),
      (np.int64, "BYTE"),
      (np.int32, "UTF8_CHAR"),
      (np.int64, "UTF8_CHAR"),
  )
  def testBroadcast(self, dtype, unit):
    # Broadcast pos/len onto input string
    test_string = {
        "BYTE": [[b"ten", b"eleven", b"twelve"],
                 [b"thirteen", b"fourteen", b"fifteen"],
                 [b"sixteen", b"seventeen", b"eighteen"],
                 [b"nineteen", b"twenty", b"twentyone"]],
        "UTF8_CHAR": [[x.encode("utf-8") for x in [u"\U0001d229\U0001d227n",
                                                   u"\xc6\u053c\u025bv\u025bn",
                                                   u"tw\u0c1dlv\u025b"]],
                      [x.encode("utf-8") for x in [u"th\xcdrt\xea\xean",
                                                   u"f\U0001f604urt\xea\xean",
                                                   u"f\xcd\ua09ctee\ua0e4"]],
                      [x.encode("utf-8") for x in [u"s\xcdxt\xea\xean",
                                                   u"se\U00010299enteen",
                                                   u"ei\U0001e920h\x86een"]],
                      [x.encode("utf-8") for x in [u"nineteen",
                                                   u"twenty",
                                                   u"twentyone"]]],
    }[unit]
    position = np.array([1, -4, 3], dtype)
    length = np.array([1, 2, 3], dtype)
    expected_value = {
        "BYTE": [[b"e", b"ev", b"lve"], [b"h", b"te", b"tee"],
                 [b"i", b"te", b"hte"], [b"i", b"en", b"nty"]],
        "UTF8_CHAR": [[x.encode("utf-8") for x in [u"\U0001d227",
                                                   u"\u025bv", u"lv\u025b"]],
                      [x.encode("utf-8") for x in [u"h", u"t\xea", u"tee"]],
                      [x.encode("utf-8") for x in [u"\xcd", u"te", u"h\x86e"]],
                      [x.encode("utf-8") for x in [u"i", u"en", u"nty"]]],
    }[unit]
    substr_op = string_ops.substr(test_string, position, length, unit=unit)
    with self.cached_session():
      substr = self.evaluate(substr_op)
      self.assertAllEqual(substr, expected_value)

    # Broadcast input string onto pos/len
    test_string = {
        "BYTE": [b"thirteen", b"fourteen", b"fifteen"],
        "UTF8_CHAR": [x.encode("utf-8") for x in [u"th\xcdrt\xea\xean",
                                                  u"f\U0001f604urt\xea\xean",
                                                  u"f\xcd\ua09ctee\ua0e4"]],
    }[unit]
    position = np.array([[1, -2, 3], [-3, 2, 1], [5, 5, -5]], dtype)
    length = np.array([[3, 2, 1], [1, 2, 3], [2, 2, 2]], dtype)
    expected_value = {
        "BYTE": [[b"hir", b"en", b"t"], [b"e", b"ur", b"ift"],
                 [b"ee", b"ee", b"ft"]],
        "UTF8_CHAR": [[x.encode("utf-8") for x in [u"h\xcdr", u"\xean", u"t"]],
                      [x.encode("utf-8") for x in [u"\xea", u"ur",
                                                   u"\xcd\ua09ct"]],
                      [x.encode("utf-8") for x in [u"\xea\xea", u"\xea\xea",
                                                   u"\ua09ct"]]],
    }[unit]
    substr_op = string_ops.substr(test_string, position, length, unit=unit)
    with self.cached_session():
      substr = self.evaluate(substr_op)
      self.assertAllEqual(substr, expected_value)

    # Test 1D broadcast
    test_string = {
        "BYTE": b"thirteen",
        "UTF8_CHAR": u"th\xcdrt\xea\xean".encode("utf-8"),
    }[unit]
    position = np.array([1, -4, 7], dtype)
    length = np.array([3, 2, 1], dtype)
    expected_value = {
        "BYTE": [b"hir", b"te", b"n"],
        "UTF8_CHAR": [x.encode("utf-8") for x in [u"h\xcdr", u"t\xea", u"n"]],
    }[unit]
    substr_op = string_ops.substr(test_string, position, length, unit=unit)
    with self.cached_session():
      substr = self.evaluate(substr_op)
      self.assertAllEqual(substr, expected_value)

  @parameterized.parameters(
      (np.int32, "BYTE"),
      (np.int64, "BYTE"),
      (np.int32, "UTF8_CHAR"),
      (np.int64, "UTF8_CHAR"),
  )
  @test_util.run_deprecated_v1
  def testBadBroadcast(self, dtype, unit):
    test_string = [[b"ten", b"eleven", b"twelve"],
                   [b"thirteen", b"fourteen", b"fifteen"],
                   [b"sixteen", b"seventeen", b"eighteen"]]
    position = np.array([1, 2, -3, 4], dtype)
    length = np.array([1, 2, 3, 4], dtype)
    with self.assertRaises(ValueError):
      string_ops.substr(test_string, position, length, unit=unit)

  @parameterized.parameters(
      (np.int32, 6, "BYTE"),
      (np.int64, 6, "BYTE"),
      (np.int32, -6, "BYTE"),
      (np.int64, -6, "BYTE"),
      (np.int32, 6, "UTF8_CHAR"),
      (np.int64, 6, "UTF8_CHAR"),
      (np.int32, -6, "UTF8_CHAR"),
      (np.int64, -6, "UTF8_CHAR"),
  )
  @test_util.run_deprecated_v1
  def testOutOfRangeError_Scalar(self, dtype, pos, unit):
    # Scalar/Scalar
    test_string = {
        "BYTE": b"Hello",
        "UTF8_CHAR": u"H\xc3ll\U0001f604".encode("utf-8"),
    }[unit]
    position = np.array(pos, dtype)
    length = np.array(3, dtype)
    substr_op = string_ops.substr(test_string, position, length, unit=unit)
    with self.cached_session():
      with self.assertRaises(errors_impl.InvalidArgumentError):
        self.evaluate(substr_op)

  @parameterized.parameters(
      (np.int32, 4, "BYTE"),
      (np.int64, 4, "BYTE"),
      (np.int32, -4, "BYTE"),
      (np.int64, -4, "BYTE"),
      (np.int32, 4, "UTF8_CHAR"),
      (np.int64, 4, "UTF8_CHAR"),
      (np.int32, -4, "UTF8_CHAR"),
      (np.int64, -4, "UTF8_CHAR"),
  )
  @test_util.run_deprecated_v1
  def testOutOfRangeError_VectorScalar(self, dtype, pos, unit):
    # Vector/Scalar
    test_string = {
        "BYTE": [b"good", b"good", b"bad", b"good"],
        "UTF8_CHAR": [x.encode("utf-8") for x in [u"g\xc3\xc3d", u"b\xc3d",
                                                  u"g\xc3\xc3d"]],
    }[unit]
    position = np.array(pos, dtype)
    length = np.array(1, dtype)
    substr_op = string_ops.substr(test_string, position, length, unit=unit)
    with self.cached_session():
      with self.assertRaises(errors_impl.InvalidArgumentError):
        self.evaluate(substr_op)

  @parameterized.parameters(
      (np.int32, "BYTE"),
      (np.int64, "BYTE"),
      (np.int32, "UTF8_CHAR"),
      (np.int64, "UTF8_CHAR"),
  )
  @test_util.run_deprecated_v1
  def testOutOfRangeError_MatrixMatrix(self, dtype, unit):
    # Matrix/Matrix
    test_string = {
        "BYTE": [[b"good", b"good", b"good"], [b"good", b"good", b"bad"],
                 [b"good", b"good", b"good"]],
        "UTF8_CHAR": [[x.encode("utf-8") for x in [u"g\xc3\xc3d", u"g\xc3\xc3d",
                                                   u"g\xc3\xc3d"]],
                      [x.encode("utf-8") for x in [u"g\xc3\xc3d", u"g\xc3\xc3d",
                                                   u"b\xc3d"]],
                      [x.encode("utf-8") for x in [u"g\xc3\xc3d", u"g\xc3\xc3d",
                                                   u"g\xc3\xc3d"]]],
    }[unit]
    position = np.array([[1, 2, 3], [1, 2, 4], [1, 2, 3]], dtype)
    length = np.array([[3, 2, 1], [1, 2, 3], [2, 2, 2]], dtype)
    substr_op = string_ops.substr(test_string, position, length, unit=unit)
    with self.cached_session():
      with self.assertRaises(errors_impl.InvalidArgumentError):
        self.evaluate(substr_op)

    # Matrix/Matrix (with negative)
    position = np.array([[1, 2, -3], [1, 2, -4], [1, 2, -3]], dtype)
    length = np.array([[3, 2, 1], [1, 2, 3], [2, 2, 2]], dtype)
    substr_op = string_ops.substr(test_string, position, length, unit=unit)
    with self.cached_session():
      with self.assertRaises(errors_impl.InvalidArgumentError):
        self.evaluate(substr_op)

  @parameterized.parameters(
      (np.int32, "BYTE"),
      (np.int64, "BYTE"),
      (np.int32, "UTF8_CHAR"),
      (np.int64, "UTF8_CHAR"),
  )
  @test_util.run_deprecated_v1
  def testOutOfRangeError_Broadcast(self, dtype, unit):
    # Broadcast
    test_string = {
        "BYTE": [[b"good", b"good", b"good"], [b"good", b"good", b"bad"]],
        "UTF8_CHAR": [[x.encode("utf-8") for x in [u"g\xc3\xc3d", u"g\xc3\xc3d",
                                                   u"g\xc3\xc3d"]],
                      [x.encode("utf-8") for x in [u"g\xc3\xc3d", u"g\xc3\xc3d",
                                                   u"b\xc3d"]]],
    }[unit]
    position = np.array([1, 2, 4], dtype)
    length = np.array([1, 2, 3], dtype)
    substr_op = string_ops.substr(test_string, position, length, unit=unit)
    with self.cached_session():
      with self.assertRaises(errors_impl.InvalidArgumentError):
        self.evaluate(substr_op)

    # Broadcast (with negative)
    position = np.array([-1, -2, -4], dtype)
    length = np.array([1, 2, 3], dtype)
    substr_op = string_ops.substr(test_string, position, length, unit=unit)
    with self.cached_session():
      with self.assertRaises(errors_impl.InvalidArgumentError):
        self.evaluate(substr_op)

  @parameterized.parameters(
      (np.int32, "BYTE"),
      (np.int64, "BYTE"),
      (np.int32, "UTF8_CHAR"),
      (np.int64, "UTF8_CHAR"),
  )
  @test_util.run_deprecated_v1
  def testMismatchPosLenShapes(self, dtype, unit):
    test_string = {
        "BYTE": [[b"ten", b"eleven", b"twelve"],
                 [b"thirteen", b"fourteen", b"fifteen"],
                 [b"sixteen", b"seventeen", b"eighteen"]],
        "UTF8_CHAR": [[x.encode("utf-8") for x in [u"\U0001d229\U0001d227n",
                                                   u"\xc6\u053c\u025bv\u025bn",
                                                   u"tw\u0c1dlv\u025b"]],
                      [x.encode("utf-8") for x in [u"th\xcdrt\xea\xean",
                                                   u"f\U0001f604urt\xea\xean",
                                                   u"f\xcd\ua09ctee\ua0e4"]],
                      [x.encode("utf-8") for x in [u"s\xcdxt\xea\xean",
                                                   u"se\U00010299enteen",
                                                   u"ei\U0001e920h\x86een"]]],
    }[unit]
    position = np.array([[1, 2, 3]], dtype)
    length = np.array([2, 3, 4], dtype)
    # Should fail: position/length have different rank
    with self.assertRaises(ValueError):
      string_ops.substr(test_string, position, length)

    position = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]], dtype)
    length = np.array([[2, 3, 4]], dtype)
    # Should fail: position/length have different dimensionality
    with self.assertRaises(ValueError):
      string_ops.substr(test_string, position, length)

  @test_util.run_deprecated_v1
  def testWrongDtype(self):
    with self.cached_session():
      with self.assertRaises(TypeError):
        string_ops.substr(b"test", 3.0, 1)
      with self.assertRaises(TypeError):
        string_ops.substr(b"test", 3, 1.0)

  @test_util.run_deprecated_v1
  def testInvalidUnit(self):
    with self.cached_session():
      with self.assertRaises(ValueError):
        string_ops.substr(b"test", 3, 1, unit="UTF8")

  def testInvalidPos(self):
    # Test case for GitHub issue 46900.
    with self.assertRaises((ValueError, errors_impl.InvalidArgumentError)):
      x = string_ops.substr(b"abc", len=1, pos=[1, -1])
      self.evaluate(x)

    with self.assertRaises((ValueError, errors_impl.InvalidArgumentError)):
      x = string_ops.substr(b"abc", len=1, pos=[1, 2])
      self.evaluate(x)


if __name__ == "__main__":
  test.main()
