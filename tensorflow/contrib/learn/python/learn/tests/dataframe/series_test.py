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
"""Tests of the Series class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

# TODO: #6568 Remove this hack that makes dlopen() not crash.
if hasattr(sys, "getdlopenflags") and hasattr(sys, "setdlopenflags"):
  import ctypes
  sys.setdlopenflags(sys.getdlopenflags() | ctypes.RTLD_GLOBAL)

from tensorflow.contrib.learn.python import learn
from tensorflow.contrib.learn.python.learn.tests.dataframe import mocks
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import test


class TransformedSeriesTest(test.TestCase):
  """Test of `TransformedSeries`."""

  def test_repr(self):
    col = learn.TransformedSeries(
        [mocks.MockSeries("foobar", [])],
        mocks.MockTwoOutputTransform("thb", "nth", "snt"), "qux")

    # note params are sorted by name
    expected = ("MockTransform({'param_one': 'thb', 'param_three': 'snt', "
                "'param_two': 'nth'})"
                "(foobar)[qux]")
    self.assertEqual(expected, repr(col))

  def test_build_no_output(self):

    def create_no_output_series():
      return learn.TransformedSeries(
          [mocks.MockSeries("foobar", [])],
          mocks.MockZeroOutputTransform("thb", "nth"), None)

    self.assertRaises(ValueError, create_no_output_series)

  def test_build_single_output(self):
    col = learn.TransformedSeries([mocks.MockSeries("foobar", [])],
                                  mocks.MockOneOutputTransform("thb", "nth"),
                                  "out1")

    result = col.build()
    expected = mocks.MockTensor("Mock Tensor 1", dtypes.int32)
    self.assertEqual(expected, result)

  def test_build_multiple_output(self):
    col = learn.TransformedSeries(
        [mocks.MockSeries("foobar", [])],
        mocks.MockTwoOutputTransform("thb", "nth", "snt"), "out2")

    result = col.build()
    expected = mocks.MockTensor("Mock Tensor 2", dtypes.int32)
    self.assertEqual(expected, result)


if __name__ == "__main__":
  test.main()
