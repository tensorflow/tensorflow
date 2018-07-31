# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for some testing utils from strategy_test_lib."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
from absl.testing import parameterized

from tensorflow.contrib.distribute.python import combinations
from tensorflow.python.eager import test


class TestingCombinationsTest(test.TestCase):

  def test_combine(self):
    self.assertEqual([{
        "a": 1,
        "b": 2
    }, {
        "a": 1,
        "b": 3
    }, {
        "a": 2,
        "b": 2
    }, {
        "a": 2,
        "b": 3
    }], combinations.combine(a=[1, 2], b=[2, 3]))

  def test_combine_single_parameter(self):
    self.assertEqual([{
        "a": 1,
        "b": 2
    }, {
        "a": 2,
        "b": 2
    }], combinations.combine(a=[1, 2], b=2))

  def test_add(self):
    self.assertEqual(
        [{
            "a": 1
        }, {
            "a": 2
        }, {
            "b": 2
        }, {
            "b": 3
        }],
        combinations.combine(a=[1, 2]) +
        combinations.combine(b=[2, 3]))

  def test_times(self):
    c1 = combinations.combine(mode=["graph"], loss=["callable", "tensor"])
    c2 = combinations.combine(mode=["eager"], loss=["callable"])
    c3 = combinations.combine(distribution=["d1", "d2"])
    c4 = combinations.times(c3, c1 + c2)
    self.assertEqual([
        OrderedDict([("distribution", "d1"), ("loss", "callable"),
                     ("mode", "graph")]),
        OrderedDict([("distribution", "d1"), ("loss", "tensor"),
                     ("mode", "graph")]),
        OrderedDict([("distribution", "d1"), ("loss", "callable"),
                     ("mode", "eager")]),
        OrderedDict([("distribution", "d2"), ("loss", "callable"),
                     ("mode", "graph")]),
        OrderedDict([("distribution", "d2"), ("loss", "tensor"),
                     ("mode", "graph")]),
        OrderedDict([("distribution", "d2"), ("loss", "callable"),
                     ("mode", "eager")])
    ], c4)

  def test_times_variable_arguments(self):
    c1 = combinations.combine(mode=["graph", "eager"])
    c2 = combinations.combine(optimizer=["adam", "gd"])
    c3 = combinations.combine(distribution=["d1", "d2"])
    c4 = combinations.times(c3, c1, c2)
    self.assertEqual([
        OrderedDict([("distribution", "d1"), ("mode", "graph"),
                     ("optimizer", "adam")]),
        OrderedDict([("distribution", "d1"), ("mode", "graph"),
                     ("optimizer", "gd")]),
        OrderedDict([("distribution", "d1"), ("mode", "eager"),
                     ("optimizer", "adam")]),
        OrderedDict([("distribution", "d1"), ("mode", "eager"),
                     ("optimizer", "gd")]),
        OrderedDict([("distribution", "d2"), ("mode", "graph"),
                     ("optimizer", "adam")]),
        OrderedDict([("distribution", "d2"), ("mode", "graph"),
                     ("optimizer", "gd")]),
        OrderedDict([("distribution", "d2"), ("mode", "eager"),
                     ("optimizer", "adam")]),
        OrderedDict([("distribution", "d2"), ("mode", "eager"),
                     ("optimizer", "gd")])
    ], c4)
    self.assertEqual(
        combinations.combine(
            mode=["graph", "eager"],
            optimizer=["adam", "gd"],
            distribution=["d1", "d2"]), c4)

  def test_overlapping_keys(self):
    c1 = combinations.combine(mode=["graph"], loss=["callable", "tensor"])
    c2 = combinations.combine(mode=["eager"], loss=["callable"])
    with self.assertRaisesRegexp(ValueError, ".*Keys.+overlap.+"):
      _ = combinations.times(c1, c2)


@combinations.generate(combinations.combine(a=[1, 0], b=[2, 3], c=[1]))
class CombineTheTestSuite(parameterized.TestCase):

  def test_add_things(self, a, b, c):
    self.assertLessEqual(3, a + b + c)
    self.assertLessEqual(a + b + c, 5)

  def test_add_things_one_more(self, a, b, c):
    self.assertLessEqual(3, a + b + c)
    self.assertLessEqual(a + b + c, 5)

  def not_a_test(self, a=0, b=0, c=0):
    del a, b, c
    self.fail()

  def _test_but_private(self, a=0, b=0, c=0):
    del a, b, c
    self.fail()

  # Check that nothing funny happens to a non-callable that starts with "_test".
  test_member = 0


if __name__ == "__main__":
  test.main()
