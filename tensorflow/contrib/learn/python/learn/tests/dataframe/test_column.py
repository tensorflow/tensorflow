# pylint: disable=g-bad-file-header
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

"""Tests of the Column class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.contrib.learn.python import learn
from tensorflow.contrib.learn.python.learn.tests.dataframe import mocks


class TransformedColumnTest(tf.test.TestCase):
  """Test of `TransformedColumn`."""

  def test_repr(self):
    col = learn.TransformedColumn(
        [mocks.MockColumn("foobar", [])],
        mocks.MockTwoOutputTransform("thb", "nth", "snt"), "qux")

    # note params are sorted by name
    expected = ("MockTransform({'param_one': 'thb', 'param_three': 'snt', "
                "'param_two': 'nth'})"
                "(foobar)[qux]")
    self.assertEqual(expected, repr(col))

  def test_build_no_output(self):
    def create_no_output_column():
      return learn.TransformedColumn(
          [mocks.MockColumn("foobar", [])],
          mocks.MockZeroOutputTransform("thb", "nth"), None)

    self.assertRaises(ValueError, create_no_output_column)

  def test_build_single_output(self):
    col = learn.TransformedColumn(
        [mocks.MockColumn("foobar", [])],
        mocks.MockOneOutputTransform("thb", "nth"), "out1")

    result = col.build()
    expected = "Fake Tensor 1"
    self.assertEqual(expected, result)

  def test_build_multiple_output(self):
    col = learn.TransformedColumn(
        [mocks.MockColumn("foobar", [])],
        mocks.MockTwoOutputTransform("thb", "nth", "snt"), "out2")

    result = col.build()
    expected = "Fake Tensor 2"
    self.assertEqual(expected, result)


if __name__ == "__main__":
  tf.test.main()
