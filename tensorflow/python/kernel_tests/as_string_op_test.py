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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import test


class AsStringOpTest(test.TestCase):

  @test_util.run_deprecated_v1
  def testFloat(self):
    float_inputs_ = [
        0, 1, -1, 0.5, 0.25, 0.125, float("INF"), float("NAN"), float("-INF")
    ]

    with self.cached_session():
      for dtype in (dtypes.half, dtypes.bfloat16, dtypes.float32,
                    dtypes.float64):
        input_ = array_ops.placeholder(dtype)

        output = string_ops.as_string(input_, shortest=True)
        result = output.eval(feed_dict={input_: float_inputs_})
        s = lambda strs: [x.decode("ascii") for x in strs]
        self.assertAllEqual(s(result), ["%g" % x for x in float_inputs_])

        output = string_ops.as_string(input_, scientific=True)
        result = output.eval(feed_dict={input_: float_inputs_})
        self.assertAllEqual(s(result), ["%e" % x for x in float_inputs_])

        output = string_ops.as_string(input_)
        result = output.eval(feed_dict={input_: float_inputs_})
        self.assertAllEqual(s(result), ["%f" % x for x in float_inputs_])

        output = string_ops.as_string(input_, width=3)
        result = output.eval(feed_dict={input_: float_inputs_})
        self.assertAllEqual(s(result), ["%3f" % x for x in float_inputs_])

        output = string_ops.as_string(input_, width=3, fill="0")
        result = output.eval(feed_dict={input_: float_inputs_})
        self.assertAllEqual(s(result), ["%03f" % x for x in float_inputs_])

        output = string_ops.as_string(input_, width=3, fill="0", shortest=True)
        result = output.eval(feed_dict={input_: float_inputs_})
        self.assertAllEqual(s(result), ["%03g" % x for x in float_inputs_])

        output = string_ops.as_string(input_, precision=10, width=3)
        result = output.eval(feed_dict={input_: float_inputs_})
        self.assertAllEqual(s(result), ["%03.10f" % x for x in float_inputs_])

        output = string_ops.as_string(
            input_, precision=10, width=3, fill="0", shortest=True)
        result = output.eval(feed_dict={input_: float_inputs_})
        self.assertAllEqual(s(result), ["%03.10g" % x for x in float_inputs_])

      with self.assertRaisesOpError("Cannot select both"):
        output = string_ops.as_string(input_, scientific=True, shortest=True)
        output.eval(feed_dict={input_: float_inputs_})

      with self.assertRaisesOpError("Fill string must be one or fewer"):
        output = string_ops.as_string(input_, fill="ab")
        output.eval(feed_dict={input_: float_inputs_})

  @test_util.run_deprecated_v1
  def testInt(self):
    # Cannot use values outside -128..127 for test, because we're also
    # testing int8
    int_inputs = [0, -1, 1, -128, 127, -101, 101, -0]
    int_dtypes = [dtypes.int8, dtypes.int32, dtypes.int64]
    uint_inputs = [0, 1, 127, 255, 101]
    uint_dtypes = [dtypes.uint8, dtypes.uint32, dtypes.uint64]
    s = lambda strs: [x.decode("ascii") for x in strs]

    with self.cached_session():
      for dtypes_, inputs in [(int_dtypes, int_inputs),
                              (uint_dtypes, uint_inputs)]:
        for dtype in dtypes_:
          input_ = array_ops.placeholder(dtype)

          output = string_ops.as_string(input_)
          result = output.eval(feed_dict={input_: inputs})
          self.assertAllEqual(s(result), ["%d" % x for x in inputs])

          output = string_ops.as_string(input_, width=3)
          result = output.eval(feed_dict={input_: inputs})
          self.assertAllEqual(s(result), ["%3d" % x for x in inputs])

          output = string_ops.as_string(input_, width=3, fill="0")
          result = output.eval(feed_dict={input_: inputs})
          self.assertAllEqual(s(result), ["%03d" % x for x in inputs])

        with self.assertRaisesOpError("scientific and shortest"):
          output = string_ops.as_string(input_, scientific=True)
          output.eval(feed_dict={input_: inputs})

        with self.assertRaisesOpError("scientific and shortest"):
          output = string_ops.as_string(input_, shortest=True)
          output.eval(feed_dict={input_: inputs})

        with self.assertRaisesOpError("precision not supported"):
          output = string_ops.as_string(input_, precision=0)
          output.eval(feed_dict={input_: inputs})

  @test_util.run_deprecated_v1
  def testLargeInt(self):
    # Cannot use values outside -128..127 for test, because we're also
    # testing int8
    s = lambda strs: [x.decode("ascii") for x in strs]

    with self.cached_session():
      input_ = array_ops.placeholder(dtypes.int32)
      int_inputs_ = [np.iinfo(np.int32).min, np.iinfo(np.int32).max]
      output = string_ops.as_string(input_)
      result = output.eval(feed_dict={input_: int_inputs_})
      self.assertAllEqual(s(result), ["%d" % x for x in int_inputs_])

      input_ = array_ops.placeholder(dtypes.int64)
      int_inputs_ = [np.iinfo(np.int64).min, np.iinfo(np.int64).max]
      output = string_ops.as_string(input_)
      result = output.eval(feed_dict={input_: int_inputs_})
      self.assertAllEqual(s(result), ["%d" % x for x in int_inputs_])

  @test_util.run_deprecated_v1
  def testHalfInt(self):
    s = lambda strs: [x.decode("ascii") for x in strs]

    with self.cached_session():
      for dtype, np_dtype in [(dtypes.int16, np.int16),
                              (dtypes.uint16, np.uint16)]:
        input_ = array_ops.placeholder(dtype)
        int_inputs_ = [np.iinfo(np_dtype).min, np.iinfo(np_dtype).max]
        output = string_ops.as_string(input_)
        result = output.eval(feed_dict={input_: int_inputs_})
        self.assertAllEqual(s(result), ["%d" % x for x in int_inputs_])

  @test_util.run_deprecated_v1
  def testBool(self):
    bool_inputs_ = [False, True]
    s = lambda strs: [x.decode("ascii") for x in strs]

    with self.cached_session():
      for dtype in (dtypes.bool,):
        input_ = array_ops.placeholder(dtype)

        output = string_ops.as_string(input_)
        result = output.eval(feed_dict={input_: bool_inputs_})
        self.assertAllEqual(s(result), ["false", "true"])

  @test_util.run_deprecated_v1
  def testComplex(self):
    float_inputs_ = [
        0, 1, -1, 0.5, 0.25, 0.125, complex("INF"), complex("NAN"),
        complex("-INF")
    ]
    complex_inputs_ = [(x + (x + 1) * 1j) for x in float_inputs_]

    with self.cached_session():
      for dtype in (dtypes.complex64, dtypes.complex128):
        input_ = array_ops.placeholder(dtype)

        def clean_nans(s_l):
          return [s.decode("ascii").replace("-nan", "nan") for s in s_l]

        output = string_ops.as_string(input_, shortest=True)
        result = output.eval(feed_dict={input_: complex_inputs_})
        self.assertAllEqual(
            clean_nans(result),
            ["(%g,%g)" % (x.real, x.imag) for x in complex_inputs_])

        output = string_ops.as_string(input_, scientific=True)
        result = output.eval(feed_dict={input_: complex_inputs_})
        self.assertAllEqual(
            clean_nans(result),
            ["(%e,%e)" % (x.real, x.imag) for x in complex_inputs_])

        output = string_ops.as_string(input_)
        result = output.eval(feed_dict={input_: complex_inputs_})
        self.assertAllEqual(
            clean_nans(result),
            ["(%f,%f)" % (x.real, x.imag) for x in complex_inputs_])

        output = string_ops.as_string(input_, width=3)
        result = output.eval(feed_dict={input_: complex_inputs_})
        self.assertAllEqual(
            clean_nans(result),
            ["(%03f,%03f)" % (x.real, x.imag) for x in complex_inputs_])

        output = string_ops.as_string(input_, width=3, fill="0", shortest=True)
        result = output.eval(feed_dict={input_: complex_inputs_})
        self.assertAllEqual(
            clean_nans(result),
            ["(%03g,%03g)" % (x.real, x.imag) for x in complex_inputs_])

        output = string_ops.as_string(input_, precision=10, width=3)
        result = output.eval(feed_dict={input_: complex_inputs_})
        self.assertAllEqual(
            clean_nans(result),
            ["(%03.10f,%03.10f)" % (x.real, x.imag) for x in complex_inputs_])

        output = string_ops.as_string(
            input_, precision=10, width=3, fill="0", shortest=True)
        result = output.eval(feed_dict={input_: complex_inputs_})
        self.assertAllEqual(
            clean_nans(result),
            ["(%03.10g,%03.10g)" % (x.real, x.imag) for x in complex_inputs_])

      with self.assertRaisesOpError("Cannot select both"):
        output = string_ops.as_string(input_, scientific=True, shortest=True)
        output.eval(feed_dict={input_: complex_inputs_})


if __name__ == "__main__":
  test.main()
