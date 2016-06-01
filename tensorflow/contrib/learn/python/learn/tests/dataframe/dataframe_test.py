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
"""Tests of the DataFrame class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.contrib.learn.python import learn
from tensorflow.contrib.learn.python.learn.tests.dataframe import mocks


def setup_test_df():
  """Create a dataframe populated with some test columns."""
  df = learn.DataFrame()
  df["a"] = learn.TransformedSeries(
      [mocks.MockSeries("foobar", [])],
      mocks.MockTwoOutputTransform("iue", "eui", "snt"), "out1")
  df["b"] = learn.TransformedSeries(
      [mocks.MockSeries("foobar", [])],
      mocks.MockTwoOutputTransform("iue", "eui", "snt"), "out2")
  df["c"] = learn.TransformedSeries(
      [mocks.MockSeries("foobar", [])],
      mocks.MockTwoOutputTransform("iue", "eui", "snt"), "out1")
  return df


class DataFrameTest(tf.test.TestCase):
  """Test of `DataFrame`."""

  def test_create(self):
    df = setup_test_df()
    self.assertEqual(df.columns(), frozenset(["a", "b", "c"]))

  def test_select(self):
    df = setup_test_df()
    df2 = df.select(["a", "c"])
    self.assertEqual(df2.columns(), frozenset(["a", "c"]))

  def test_get_item(self):
    df = setup_test_df()
    c1 = df["b"]
    self.assertEqual("Fake Tensor 2", c1.build())

  def test_set_item_column(self):
    df = setup_test_df()
    self.assertEqual(3, len(df))
    col1 = mocks.MockSeries("QuackColumn", [])
    df["quack"] = col1
    self.assertEqual(4, len(df))
    col2 = df["quack"]
    self.assertEqual(col1, col2)

  def test_set_item_column_multi(self):
    df = setup_test_df()
    self.assertEqual(3, len(df))
    col1 = mocks.MockSeries("QuackColumn", [])
    col2 = mocks.MockSeries("MooColumn", [])
    df["quack", "moo"] = [col1, col2]
    self.assertEqual(5, len(df))
    col3 = df["quack"]
    self.assertEqual(col1, col3)
    col4 = df["moo"]
    self.assertEqual(col2, col4)

  def test_set_item_pandas(self):
    # TODO(jamieas)
    pass

  def test_set_item_numpy(self):
    # TODO(jamieas)
    pass

  def test_build(self):
    df = setup_test_df()
    result = df.build()
    expected = {"a": "Fake Tensor 1",
                "b": "Fake Tensor 2",
                "c": "Fake Tensor 1"}
    self.assertEqual(expected, result)

  def test_to_input_fn_all_features(self):
    df = setup_test_df()
    input_fn = df.to_input_fn()
    f, t = input_fn()
    expected_f = {"a": "Fake Tensor 1",
                  "b": "Fake Tensor 2",
                  "c": "Fake Tensor 1"}
    self.assertEqual(expected_f, f)

    expected_t = {}
    self.assertEqual(expected_t, t)

  def test_to_input_fn_features_only(self):
    df = setup_test_df()
    input_fn = df.to_input_fn(["b", "c"])
    f, t = input_fn()
    expected_f = {"b": "Fake Tensor 2", "c": "Fake Tensor 1"}
    self.assertEqual(expected_f, f)

    expected_t = {}
    self.assertEqual(expected_t, t)

  def test_to_input_fn_targets_only(self):
    df = setup_test_df()
    input_fn = df.to_input_fn(target_keys=["b", "c"])
    f, t = input_fn()
    expected_f = {"a": "Fake Tensor 1"}
    self.assertEqual(expected_f, f)

    expected_t = {"b": "Fake Tensor 2", "c": "Fake Tensor 1"}
    self.assertEqual(expected_t, t)

  def test_to_input_fn_both(self):
    df = setup_test_df()
    input_fn = df.to_input_fn(feature_keys=["a"], target_keys=["b"])
    f, t = input_fn()
    expected_f = {"a": "Fake Tensor 1"}
    self.assertEqual(expected_f, f)

    expected_t = {"b": "Fake Tensor 2"}
    self.assertEqual(expected_t, t)

  def test_to_input_fn_not_disjoint(self):
    df = setup_test_df()

    def get_not_disjoint():
      df.to_input_fn(feature_keys=["a", "b"], target_keys=["b"])

    self.assertRaises(ValueError, get_not_disjoint)


if __name__ == "__main__":
  tf.test.main()
