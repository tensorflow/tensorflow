# Copyright 2016 Google Inc. All Rights Reserved.
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

"""Tests for learn.dataframe.tensorflow_dataframe."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import tempfile

import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.dataframe import tensorflow_dataframe as df
from tensorflow.core.example import example_pb2

# pylint: disable=g-import-not-at-top
try:
  import pandas as pd
  HAS_PANDAS = True
except ImportError:
  HAS_PANDAS = False


def _assert_df_equals_dict(expected_df, actual_dict):
  for col in expected_df:
    col_type = expected_df[col].dtype
    if col_type in [np.float32, np.float64]:
      assertion = np.testing.assert_allclose
    else:
      assertion = np.testing.assert_array_equal
    expected = expected_df[col].values
    actual = actual_dict[col]
    assertion(expected,
              actual,
              err_msg="Expected {} in column '{}'; got {}.".format(
                  expected_df[col].values, col, actual_dict[col]))


def _make_test_csv():
  f = tempfile.NamedTemporaryFile(delete=False, mode="w")
  w = csv.writer(f)
  w.writerow(["int", "float", "bin"])
  for _ in range(100):
    row = [np.random.randint(-10, 10), np.random.rand(), int(np.random.rand() >
                                                             0.3)]
    w.writerow(row)
  f.close()
  return f.name


def _make_test_tfrecord():
  f = tempfile.NamedTemporaryFile(delete=False)
  w = tf.python_io.TFRecordWriter(f.name)
  for i in range(100):
    ex = example_pb2.Example()
    ex.features.feature["var_len_int"].int64_list.value.extend(range((i % 3)))
    ex.features.feature["fixed_len_float"].float_list.value.extend(
        [float(i), 2 * float(i)])
    w.write(ex.SerializeToString())
  return f.name


class TensorFlowDataFrameTestCase(tf.test.TestCase):
  """Tests for `TensorFlowDataFrame`."""

  def _assert_pandas_equals_tensorflow(self,
                                       pandas_df,
                                       tensorflow_df,
                                       num_batches,
                                       batch_size):
    self.assertItemsEqual(
        list(pandas_df.columns) + ["index"], tensorflow_df.columns())

    for batch_num, batch in enumerate(tensorflow_df.run(num_batches)):
      row_numbers = [
          total_row_num % pandas_df.shape[0]
          for total_row_num in range(batch_size * batch_num,
                                     batch_size * (batch_num + 1))
      ]
      expected_df = pandas_df.iloc[row_numbers]
      _assert_df_equals_dict(expected_df, batch)

  def testInitFromPandas(self):
    """Test construction from Pandas DataFrame."""
    if not HAS_PANDAS:
      return

    pandas_df = pd.DataFrame({"sparrow": range(10), "ostrich": 1})
    tensorflow_df = df.TensorFlowDataFrame.from_pandas(
        pandas_df, batch_size=10, shuffle=False)

    batch = tensorflow_df.run_once()

    np.testing.assert_array_equal(pandas_df.index.values, batch["index"],
                                  "Expected index {}; got {}".format(
                                      pandas_df.index.values, batch["index"]))
    _assert_df_equals_dict(pandas_df, batch)

  def testBatch(self):
    """Tests `batch` method.

    `DataFrame.batch()` should iterate through the rows of the
    `pandas.DataFrame`, and should "wrap around" when it reaches the last row.
    """
    if not HAS_PANDAS:
      return
    pandas_df = pd.DataFrame({"albatross": range(10),
                              "bluejay": 1,
                              "cockatoo": range(0, 20, 2)})
    tensorflow_df = df.TensorFlowDataFrame.from_pandas(pandas_df, shuffle=False)

    # Rebatch `df` into the following sizes successively.
    batch_sizes = [8, 4, 7]
    num_batches = 10

    final_batch_size = batch_sizes[-1]

    for batch_size in batch_sizes:
      tensorflow_df = tensorflow_df.batch(batch_size, shuffle=False)

    self._assert_pandas_equals_tensorflow(pandas_df,
                                          tensorflow_df,
                                          num_batches=num_batches,
                                          batch_size=final_batch_size)

  def testFromNumpy(self):
    x = np.eye(20)
    tensorflow_df = df.TensorFlowDataFrame.from_numpy(x, batch_size=10)
    for batch in tensorflow_df.run(30):
      for ind, val in zip(batch["index"], batch["value"]):
        expected_val = np.zeros_like(val)
        expected_val[ind] = 1
        np.testing.assert_array_equal(expected_val, val)

  def testFromCSV(self):
    if not HAS_PANDAS:
      return

    num_batches = 100
    batch_size = 8

    data_path = _make_test_csv()
    default_values = [0, 0.0, 0]

    pandas_df = pd.read_csv(data_path)
    tensorflow_df = df.TensorFlowDataFrame.from_csv(
        [data_path],
        batch_size=batch_size,
        shuffle=False,
        default_values=default_values)
    self._assert_pandas_equals_tensorflow(pandas_df,
                                          tensorflow_df,
                                          num_batches=num_batches,
                                          batch_size=batch_size)

  def testFromExamples(self):
    num_batches = 77
    batch_size = 13

    data_path = _make_test_tfrecord()
    features = {
        "fixed_len_float": tf.FixedLenFeature(shape=[2],
                                              dtype=tf.float32,
                                              default_value=[0.0, 0.0]),
        "var_len_int": tf.VarLenFeature(dtype=tf.int64)
    }

    tensorflow_df = df.TensorFlowDataFrame.from_examples(data_path,
                                                         batch_size=batch_size,
                                                         features=features,
                                                         shuffle=False)

    # `test.tfrecord` contains 100 records with two features: var_len_int and
    # fixed_len_float. Entry n contains `range(n % 3)` and
    # `float(n)` for var_len_int and fixed_len_float,
    # respectively.
    num_records = 100
    def _expected_fixed_len_float(n):
      return np.array([float(n), 2 * float(n)])
    def _expected_var_len_int(n):
      return np.arange(n % 3)

    for batch_num, batch in enumerate(tensorflow_df.run(num_batches)):
      record_numbers = [
          n % num_records
          for n in range(batch_num * batch_size, (batch_num + 1) * batch_size)
      ]
      for i, j in enumerate(record_numbers):
        np.testing.assert_allclose(_expected_fixed_len_float(j),
                                   batch["fixed_len_float"][i])
        var_len_int = batch["var_len_int"]
      for i, ind in enumerate(var_len_int.indices):
        val = var_len_int.values[i]
        expected_row = _expected_var_len_int(record_numbers[ind[0]])
        expected_value = expected_row[ind[1]]
        np.testing.assert_array_equal(expected_value, val)

if __name__ == "__main__":
  tf.test.main()
