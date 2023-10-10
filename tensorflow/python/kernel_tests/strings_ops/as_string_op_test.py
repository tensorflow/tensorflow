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
"""Tests for as_string_op."""
import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import test


class AsStringOpTest(test.TestCase):

  def testFloat(self):
    float_inputs_ = [
        0, 1, -1, 0.5, 0.25, 0.125, float("INF"), float("NAN"), float("-INF")
    ]

    for dtype in (dtypes.half, dtypes.bfloat16, dtypes.float32, dtypes.float64):
      inputs = ops.convert_to_tensor(float_inputs_, dtype=dtype)
      s = lambda strs: [x.decode("ascii") for x in self.evaluate(strs)]
      result = string_ops.as_string(inputs, shortest=True)
      self.assertAllEqual(s(result), ["%g" % x for x in float_inputs_])

      result = string_ops.as_string(inputs, scientific=True)
      self.assertAllEqual(s(result), ["%e" % x for x in float_inputs_])

      result = string_ops.as_string(inputs)
      self.assertAllEqual(s(result), ["%f" % x for x in float_inputs_])

      result = string_ops.as_string(inputs, width=3)
      self.assertAllEqual(s(result), ["%3f" % x for x in float_inputs_])

      result = string_ops.as_string(inputs, width=3, fill="0")
      self.assertAllEqual(s(result), ["%03f" % x for x in float_inputs_])

      result = string_ops.as_string(inputs, width=3, fill="0", shortest=True)
      self.assertAllEqual(s(result), ["%03g" % x for x in float_inputs_])

      result = string_ops.as_string(inputs, precision=10, width=3)
      self.assertAllEqual(s(result), ["%03.10f" % x for x in float_inputs_])

      result = string_ops.as_string(
          inputs, precision=10, width=3, fill="0", shortest=True
      )
      self.assertAllEqual(s(result), ["%03.10g" % x for x in float_inputs_])

    with self.assertRaisesOpError("Cannot select both"):
      self.evaluate(
          string_ops.as_string(inputs, scientific=True, shortest=True)
      )

    with self.assertRaisesOpError("Fill string must be one or fewer"):
      self.evaluate(string_ops.as_string(inputs, fill="ab"))

  def testInt(self):
    # Cannot use values outside -128..127 for test, because we're also
    # testing int8
    int_inputs = [0, -1, 1, -128, 127, -101, 101, -0]
    int_dtypes = [dtypes.int8, dtypes.int32, dtypes.int64]
    uint_inputs = [0, 1, 127, 255, 101]
    uint_dtypes = [dtypes.uint8, dtypes.uint32, dtypes.uint64]
    s = lambda strs: [x.decode("ascii") for x in self.evaluate(strs)]

    for dtypes_, inputs_ in [
        (int_dtypes, int_inputs),
        (uint_dtypes, uint_inputs),
    ]:
      for dtype in dtypes_:
        inputs = ops.convert_to_tensor(inputs_, dtype=dtype)
        result = string_ops.as_string(inputs)
        self.assertAllEqual(s(result), ["%d" % x for x in inputs_])

        result = string_ops.as_string(inputs, width=3)
        self.assertAllEqual(s(result), ["%3d" % x for x in inputs_])

        result = string_ops.as_string(inputs, width=3, fill="0")
        self.assertAllEqual(s(result), ["%03d" % x for x in inputs_])

      with self.assertRaisesOpError("scientific and shortest"):
        self.evaluate(string_ops.as_string(inputs, scientific=True))

      with self.assertRaisesOpError("scientific and shortest"):
        self.evaluate(string_ops.as_string(inputs, shortest=True))

      with self.assertRaisesOpError("precision not supported"):
        self.evaluate(string_ops.as_string(inputs, precision=0))

  def testLargeInt(self):
    # Cannot use values outside -128..127 for test, because we're also
    # testing int8
    s = lambda strs: [x.decode("ascii") for x in self.evaluate(strs)]
    inputs = [np.iinfo(np.int32).min, np.iinfo(np.int32).max]
    result = string_ops.as_string(inputs)
    self.assertAllEqual(s(result), ["%d" % x for x in inputs])

    inputs = [np.iinfo(np.int64).min, np.iinfo(np.int64).max]
    result = string_ops.as_string(inputs)
    self.assertAllEqual(s(result), ["%d" % x for x in inputs])

  def testHalfInt(self):
    s = lambda strs: [x.decode("ascii") for x in self.evaluate(strs)]
    for dtype, np_dtype in [
        (dtypes.int16, np.int16),
        (dtypes.uint16, np.uint16),
    ]:
      inputs = [np.iinfo(np_dtype).min, np.iinfo(np_dtype).max]
      result = string_ops.as_string(ops.convert_to_tensor(inputs, dtype=dtype))
      self.assertAllEqual(s(result), ["%d" % x for x in inputs])

  def testBool(self):
    bool_inputs_ = [False, True]
    s = lambda strs: [x.decode("ascii") for x in self.evaluate(strs)]
    result = string_ops.as_string(bool_inputs_)
    self.assertAllEqual(s(result), ["false", "true"])

  def testComplex(self):
    inputs = [
        0,
        1,
        -1,
        0.5,
        0.25,
        0.125,
        complex("INF"),
        complex("NAN"),
        complex("-INF"),
    ]
    complex_inputs_ = [(x + (x + 1) * 1j) for x in inputs]

    for dtype in (dtypes.complex64, dtypes.complex128):
      inputs = ops.convert_to_tensor(complex_inputs_, dtype=dtype)

      def clean_nans(s_l):
        return [
            s.decode("ascii").replace("-nan", "nan") for s in self.evaluate(s_l)
        ]

      result = string_ops.as_string(inputs, shortest=True)
      self.assertAllEqual(
          clean_nans(result),
          ["(%g,%g)" % (x.real, x.imag) for x in complex_inputs_],
      )

      result = string_ops.as_string(inputs, scientific=True)
      self.assertAllEqual(
          clean_nans(result),
          ["(%e,%e)" % (x.real, x.imag) for x in complex_inputs_],
      )

      result = string_ops.as_string(inputs)
      self.assertAllEqual(
          clean_nans(result),
          ["(%f,%f)" % (x.real, x.imag) for x in complex_inputs_],
      )

      result = string_ops.as_string(inputs, width=3)
      self.assertAllEqual(
          clean_nans(result),
          ["(%03f,%03f)" % (x.real, x.imag) for x in complex_inputs_],
      )

      result = string_ops.as_string(inputs, width=3, fill="0", shortest=True)
      self.assertAllEqual(
          clean_nans(result),
          ["(%03g,%03g)" % (x.real, x.imag) for x in complex_inputs_],
      )

      result = string_ops.as_string(inputs, precision=10, width=3)
      self.assertAllEqual(
          clean_nans(result),
          ["(%03.10f,%03.10f)" % (x.real, x.imag) for x in complex_inputs_],
      )

      result = string_ops.as_string(
          inputs, precision=10, width=3, fill="0", shortest=True
      )
      self.assertAllEqual(
          clean_nans(result),
          ["(%03.10g,%03.10g)" % (x.real, x.imag) for x in complex_inputs_],
      )

    with self.assertRaisesOpError("Cannot select both"):
      self.evaluate(
          string_ops.as_string(inputs, scientific=True, shortest=True)
      )

  def testString(self):
    self.assertAllEqual(string_ops.as_string("hello, world!"), "hello, world!")
    widened_string = self.evaluate(
        string_ops.as_string("hello, world!", width=20)
    )
    self.assertAllEqual(widened_string, "       hello, world!")


if __name__ == "__main__":
  test.main()
