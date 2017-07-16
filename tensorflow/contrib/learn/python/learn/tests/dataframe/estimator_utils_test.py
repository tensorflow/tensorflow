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

import copy

from tensorflow.contrib.layers import feature_column
from tensorflow.contrib.learn.python import learn
from tensorflow.contrib.learn.python.learn.dataframe import estimator_utils
from tensorflow.contrib.learn.python.learn.tests.dataframe import mocks
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import parsing_ops
from tensorflow.python.platform import test


def setup_test_df():
  """Create a dataframe populated with some test columns."""
  df = learn.DataFrame()
  df["a"] = learn.TransformedSeries(
      [mocks.MockSeries("foobar", mocks.MockTensor("Tensor a", dtypes.int32))],
      mocks.MockTwoOutputTransform("iue", "eui", "snt"), "out1")
  df["b"] = learn.TransformedSeries(
      [mocks.MockSeries("foobar", mocks.MockTensor("Tensor b", dtypes.int32))],
      mocks.MockTwoOutputTransform("iue", "eui", "snt"), "out2")
  df["c"] = learn.TransformedSeries(
      [mocks.MockSeries("foobar", mocks.MockTensor("Tensor c", dtypes.int32))],
      mocks.MockTwoOutputTransform("iue", "eui", "snt"), "out1")
  return df


def setup_test_df_3layer():
  """Create a dataframe populated with some test columns."""
  df = learn.DataFrame()

  df["a"] = mocks.MockSeries("a_series",
                             mocks.MockTensor("Tensor a", dtypes.int32))
  df["b"] = mocks.MockSeries(
      "b_series", mocks.MockSparseTensor("SparseTensor b", dtypes.int32))
  df["c"] = mocks.MockSeries("c_series",
                             mocks.MockTensor("Tensor c", dtypes.int32))
  df["d"] = mocks.MockSeries(
      "d_series", mocks.MockSparseTensor("SparseTensor d", dtypes.int32))

  df["e"] = learn.TransformedSeries([df["a"], df["b"]],
                                    mocks.Mock2x2Transform("iue", "eui", "snt"),
                                    "out1")
  df["f"] = learn.TransformedSeries([df["c"], df["d"]],
                                    mocks.Mock2x2Transform("iue", "eui", "snt"),
                                    "out2")
  df["g"] = learn.TransformedSeries([df["e"], df["f"]],
                                    mocks.Mock2x2Transform("iue", "eui", "snt"),
                                    "out1")
  return df


class DataFrameColumnTest(test.TestCase):
  """Test of layers.DataFrameColumn."""

  def test_constructor(self):
    series = learn.PredefinedSeries(
        "a",
        parsing_ops.FixedLenFeature(
            tensor_shape.unknown_shape(), dtypes.int32, 1))
    column = feature_column.DataFrameColumn("a", series)
    self.assertEqual("a", column.column_name)
    self.assertEqual("a", column.name)
    self.assertEqual(series, column.series)

  def test_deepcopy(self):
    series = learn.PredefinedSeries(
        "a",
        parsing_ops.FixedLenFeature(
            tensor_shape.unknown_shape(), dtypes.int32, 1))
    column = feature_column.DataFrameColumn("a", series)
    column_copy = copy.deepcopy(column)
    self.assertEqual("a", column_copy.column_name)
    self.assertEqual("a", column_copy.name)
    self.assertEqual(series, column_copy.series)


class EstimatorUtilsTest(test.TestCase):
  """Test of estimator utils."""

  def test_to_feature_columns_and_input_fn(self):
    df = setup_test_df_3layer()
    feature_columns, input_fn = (
        estimator_utils.to_feature_columns_and_input_fn(
            df,
            base_input_keys_with_defaults={"a": 1,
                                           "b": 2,
                                           "c": 3,
                                           "d": 4},
            label_keys=["g"],
            feature_keys=["a", "b", "f"]))

    expected_feature_column_a = feature_column.DataFrameColumn(
        "a",
        learn.PredefinedSeries(
            "a",
            parsing_ops.FixedLenFeature(tensor_shape.unknown_shape(),
                                        dtypes.int32, 1)))
    expected_feature_column_b = feature_column.DataFrameColumn(
        "b",
        learn.PredefinedSeries("b", parsing_ops.VarLenFeature(dtypes.int32)))
    expected_feature_column_f = feature_column.DataFrameColumn(
        "f",
        learn.TransformedSeries([
            learn.PredefinedSeries("c",
                                   parsing_ops.FixedLenFeature(
                                       tensor_shape.unknown_shape(),
                                       dtypes.int32, 3)),
            learn.PredefinedSeries("d", parsing_ops.VarLenFeature(dtypes.int32))
        ], mocks.Mock2x2Transform("iue", "eui", "snt"), "out2"))

    expected_feature_columns = [
        expected_feature_column_a, expected_feature_column_b,
        expected_feature_column_f
    ]
    self.assertEqual(sorted(expected_feature_columns), sorted(feature_columns))

    base_features, labels = input_fn()
    expected_base_features = {
        "a": mocks.MockTensor("Tensor a", dtypes.int32),
        "b": mocks.MockSparseTensor("SparseTensor b", dtypes.int32),
        "c": mocks.MockTensor("Tensor c", dtypes.int32),
        "d": mocks.MockSparseTensor("SparseTensor d", dtypes.int32)
    }
    self.assertEqual(expected_base_features, base_features)

    expected_labels = mocks.MockTensor("Out iue", dtypes.int32)
    self.assertEqual(expected_labels, labels)

    self.assertEqual(3, len(feature_columns))

  def test_to_feature_columns_and_input_fn_no_labels(self):
    df = setup_test_df_3layer()
    feature_columns, input_fn = (
        estimator_utils.to_feature_columns_and_input_fn(
            df,
            base_input_keys_with_defaults={"a": 1,
                                           "b": 2,
                                           "c": 3,
                                           "d": 4},
            feature_keys=["a", "b", "f"]))

    base_features, labels = input_fn()
    expected_base_features = {
        "a": mocks.MockTensor("Tensor a", dtypes.int32),
        "b": mocks.MockSparseTensor("SparseTensor b", dtypes.int32),
        "c": mocks.MockTensor("Tensor c", dtypes.int32),
        "d": mocks.MockSparseTensor("SparseTensor d", dtypes.int32)
    }
    self.assertEqual(expected_base_features, base_features)

    expected_labels = {}
    self.assertEqual(expected_labels, labels)

    self.assertEqual(3, len(feature_columns))

  def test_to_estimator_not_disjoint(self):
    df = setup_test_df_3layer()

    # pylint: disable=unused-variable
    def get_not_disjoint():
      feature_columns, input_fn = (
          estimator_utils.to_feature_columns_and_input_fn(
              df,
              base_input_keys_with_defaults={"a": 1,
                                             "b": 2,
                                             "c": 3,
                                             "d": 4},
              label_keys=["f"],
              feature_keys=["a", "b", "f"]))

    self.assertRaises(ValueError, get_not_disjoint)


if __name__ == "__main__":
  test.main()
