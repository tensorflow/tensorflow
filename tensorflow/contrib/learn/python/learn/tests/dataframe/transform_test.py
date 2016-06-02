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

"""Tests of the Transform class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.contrib.learn.python import learn
from tensorflow.contrib.learn.python.learn.dataframe.transform import _make_list_of_series
from tensorflow.contrib.learn.python.learn.tests.dataframe import mocks


class TransformTest(tf.test.TestCase):
  """Tests of the Transform class."""

  def test_make_list_of_column(self):
    col1 = mocks.MockSeries("foo", [])
    col2 = mocks.MockSeries("bar", [])

    self.assertEqual([], _make_list_of_series(None))
    self.assertEqual([col1], _make_list_of_series(col1))
    self.assertEqual([col1], _make_list_of_series([col1]))
    self.assertEqual([col1, col2], _make_list_of_series([col1, col2]))
    self.assertEqual([col1, col2], _make_list_of_series((col1, col2)))

  def test_cache(self):
    z = mocks.MockSeries("foobar", [])
    t = mocks.MockTwoOutputTransform("thb", "nth", "snt")
    cache = {}
    t.apply_transform([z], cache)
    self.assertEqual(2, len(cache))

    expected_keys = [
        "MockTransform("
        "{'param_one': 'thb', 'param_three': 'snt', 'param_two': 'nth'})"
        "(foobar)[out1]",
        "MockTransform("
        "{'param_one': 'thb', 'param_three': 'snt', 'param_two': 'nth'})"
        "(foobar)[out2]"]

    self.assertEqual(expected_keys, sorted(cache.keys()))

  def test_parameters(self):
    t = mocks.MockTwoOutputTransform("a", "b", "c")
    self.assertEqual({"param_one": "a", "param_three": "c", "param_two": "b"},
                     t.parameters())

  def test_parameters_inherited_combined(self):
    t = mocks.MockTwoOutputTransform("thb", "nth", "snt")

    expected = {"param_one": "thb", "param_two": "nth", "param_three": "snt"}
    self.assertEqual(expected, t.parameters())

  def test_return_type(self):
    t = mocks.MockTwoOutputTransform("a", "b", "c")

    rt = t.return_type
    self.assertEqual("ReturnType", rt.__name__)
    self.assertEqual(("out1", "out2"), rt._fields)

  def test_call(self):
    t = mocks.MockTwoOutputTransform("a", "b", "c")
    # MockTwoOutputTransform has input valency 1
    input1 = mocks.MockSeries("foobar", [])
    out1, out2 = t([input1])  # pylint: disable=not-callable

    self.assertEqual(learn.TransformedSeries, type(out1))
    # self.assertEqual(out1.transform, t)
    # self.assertEqual(out1.output_name, "output1")

    self.assertEqual(learn.TransformedSeries, type(out2))
    # self.assertEqual(out2.transform, t)
    # self.assertEqual(out2.output_name, "output2")


if __name__ == "__main__":
  tf.test.main()
