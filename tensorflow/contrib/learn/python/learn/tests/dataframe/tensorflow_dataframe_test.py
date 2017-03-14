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
"""Tests for learn.dataframe.tensorflow_dataframe."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import math
import tempfile

import numpy as np

from tensorflow.contrib.learn.python.learn.dataframe import tensorflow_dataframe as df
from tensorflow.contrib.learn.python.learn.dataframe.transforms import densify
from tensorflow.core.example import example_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.lib.io import tf_record
from tensorflow.python.ops import parsing_ops
from tensorflow.python.platform import test

# pylint: disable=g-import-not-at-top
try:
  import pandas as pd
  HAS_PANDAS = True
except ImportError:
  HAS_PANDAS = False


def _assert_df_equals_dict(expected_df, actual_dict):
  for col in expected_df:
    if expected_df[col].dtype in [np.float32, np.float64]:
      assertion = np.testing.assert_allclose
    else:
      assertion = np.testing.assert_array_equal

    if expected_df[col].dtype.kind in ["O", "S", "U"]:
      # Python 2/3 compatibility
      # TensorFlow always returns bytes, so we just convert the unicode
      # expectations to bytes also before comparing.
      expected_values = [x.encode("utf-8") for x in expected_df[col].values]
    else:
      expected_values = expected_df[col].values

    assertion(
        expected_values,
        actual_dict[col],
        err_msg="Expected {} in column '{}'; got {}.".format(expected_values,
                                                             col,
                                                             actual_dict[col]))


class TensorFlowDataFrameTestCase(test.TestCase):
  """Tests for `TensorFlowDataFrame`."""

  def _make_test_csv(self):
    f = tempfile.NamedTemporaryFile(
        dir=self.get_temp_dir(), delete=False, mode="w")
    w = csv.writer(f)
    w.writerow(["int", "float", "bool", "string"])
    for _ in range(100):
      intvalue = np.random.randint(-10, 10)
      floatvalue = np.random.rand()
      boolvalue = int(np.random.rand() > 0.3)
      stringvalue = "S: %.4f" % np.random.rand()

      row = [intvalue, floatvalue, boolvalue, stringvalue]
      w.writerow(row)
    f.close()
    return f.name

  def _make_test_csv_sparse(self):
    f = tempfile.NamedTemporaryFile(
        dir=self.get_temp_dir(), delete=False, mode="w")
    w = csv.writer(f)
    w.writerow(["int", "float", "bool", "string"])
    for _ in range(100):
      # leave columns empty; these will be read as default value (e.g. 0 or NaN)
      intvalue = np.random.randint(-10, 10) if np.random.rand() > 0.5 else ""
      floatvalue = np.random.rand() if np.random.rand() > 0.5 else ""
      boolvalue = int(np.random.rand() > 0.3) if np.random.rand() > 0.5 else ""
      stringvalue = (("S: %.4f" % np.random.rand()) if np.random.rand() > 0.5 else
                     "")

      row = [intvalue, floatvalue, boolvalue, stringvalue]
      w.writerow(row)
    f.close()
    return f.name

  def _make_test_tfrecord(self):
    f = tempfile.NamedTemporaryFile(dir=self.get_temp_dir(), delete=False)
    w = tf_record.TFRecordWriter(f.name)
    for i in range(100):
      ex = example_pb2.Example()
      ex.features.feature["var_len_int"].int64_list.value.extend(range((i % 3)))
      ex.features.feature["fixed_len_float"].float_list.value.extend(
          [float(i), 2 * float(i)])
      w.write(ex.SerializeToString())
    return f.name

  def _assert_pandas_equals_tensorflow(self, pandas_df, tensorflow_df,
                                       num_batches, batch_size):
    self.assertItemsEqual(
        list(pandas_df.columns) + ["index"], tensorflow_df.columns())

    for batch_num, batch in enumerate(tensorflow_df.run(num_batches)):
      row_numbers = [
          total_row_num % pandas_df.shape[0]
          for total_row_num in range(batch_size * batch_num, batch_size * (
              batch_num + 1))
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

    batch = tensorflow_df.run_one_batch()

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
    pandas_df = pd.DataFrame({
        "albatross": range(10),
        "bluejay": 1,
        "cockatoo": range(0, 20, 2),
        "penguin": list("abcdefghij")
    })
    tensorflow_df = df.TensorFlowDataFrame.from_pandas(pandas_df, shuffle=False)

    # Rebatch `df` into the following sizes successively.
    batch_sizes = [4, 7]
    num_batches = 3

    final_batch_size = batch_sizes[-1]

    for batch_size in batch_sizes:
      tensorflow_df = tensorflow_df.batch(batch_size, shuffle=False)

    self._assert_pandas_equals_tensorflow(
        pandas_df,
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
    enqueue_size = 7

    data_path = self._make_test_csv()
    default_values = [0, 0.0, 0, ""]

    pandas_df = pd.read_csv(data_path)
    tensorflow_df = df.TensorFlowDataFrame.from_csv(
        [data_path],
        enqueue_size=enqueue_size,
        batch_size=batch_size,
        shuffle=False,
        default_values=default_values)
    self._assert_pandas_equals_tensorflow(
        pandas_df,
        tensorflow_df,
        num_batches=num_batches,
        batch_size=batch_size)

  def testFromCSVLimitEpoch(self):
    batch_size = 8
    num_epochs = 17
    expected_num_batches = (num_epochs * 100) // batch_size

    data_path = self._make_test_csv()
    default_values = [0, 0.0, 0, ""]

    tensorflow_df = df.TensorFlowDataFrame.from_csv(
        [data_path],
        batch_size=batch_size,
        shuffle=False,
        default_values=default_values)
    result_batches = list(tensorflow_df.run(num_epochs=num_epochs))
    actual_num_batches = len(result_batches)
    self.assertEqual(expected_num_batches, actual_num_batches)

    # TODO(soergel): figure out how to dequeue the final small batch
    expected_rows = 1696  # num_epochs * 100
    actual_rows = sum([len(x["int"]) for x in result_batches])
    self.assertEqual(expected_rows, actual_rows)

  def testFromCSVWithFeatureSpec(self):
    if not HAS_PANDAS:
      return
    num_batches = 100
    batch_size = 8

    data_path = self._make_test_csv_sparse()
    feature_spec = {
        "int": parsing_ops.FixedLenFeature(None, dtypes.int16, np.nan),
        "float": parsing_ops.VarLenFeature(dtypes.float16),
        "bool": parsing_ops.VarLenFeature(dtypes.bool),
        "string": parsing_ops.FixedLenFeature(None, dtypes.string, "")
    }

    pandas_df = pd.read_csv(data_path, dtype={"string": object})
    # Pandas insanely uses NaN for empty cells in a string column.
    # And, we can't use Pandas replace() to fix them because nan != nan
    s = pandas_df["string"]
    for i in range(0, len(s)):
      if isinstance(s[i], float) and math.isnan(s[i]):
        pandas_df.set_value(i, "string", "")
    tensorflow_df = df.TensorFlowDataFrame.from_csv_with_feature_spec(
        [data_path],
        batch_size=batch_size,
        shuffle=False,
        feature_spec=feature_spec)

    # These columns were sparse; re-densify them for comparison
    tensorflow_df["float"] = densify.Densify(np.nan)(tensorflow_df["float"])
    tensorflow_df["bool"] = densify.Densify(np.nan)(tensorflow_df["bool"])

    self._assert_pandas_equals_tensorflow(
        pandas_df,
        tensorflow_df,
        num_batches=num_batches,
        batch_size=batch_size)

  def testFromExamples(self):
    num_batches = 77
    enqueue_size = 11
    batch_size = 13

    data_path = self._make_test_tfrecord()
    features = {
        "fixed_len_float":
            parsing_ops.FixedLenFeature(
                shape=[2], dtype=dtypes.float32, default_value=[0.0, 0.0]),
        "var_len_int":
            parsing_ops.VarLenFeature(dtype=dtypes.int64)
    }

    tensorflow_df = df.TensorFlowDataFrame.from_examples(
        data_path,
        enqueue_size=enqueue_size,
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
        np.testing.assert_allclose(
            _expected_fixed_len_float(j), batch["fixed_len_float"][i])
        var_len_int = batch["var_len_int"]
      for i, ind in enumerate(var_len_int.indices):
        val = var_len_int.values[i]
        expected_row = _expected_var_len_int(record_numbers[ind[0]])
        expected_value = expected_row[ind[1]]
        np.testing.assert_array_equal(expected_value, val)

  def testSplitString(self):
    batch_size = 8
    num_epochs = 17
    expected_num_batches = (num_epochs * 100) // batch_size

    data_path = self._make_test_csv()
    default_values = [0, 0.0, 0, ""]

    tensorflow_df = df.TensorFlowDataFrame.from_csv(
        [data_path],
        batch_size=batch_size,
        shuffle=False,
        default_values=default_values)

    a, b = tensorflow_df.split("string", 0.7)  # no rebatching

    total_result_batches = list(tensorflow_df.run(num_epochs=num_epochs))
    a_result_batches = list(a.run(num_epochs=num_epochs))
    b_result_batches = list(b.run(num_epochs=num_epochs))

    self.assertEqual(expected_num_batches, len(total_result_batches))
    self.assertEqual(expected_num_batches, len(a_result_batches))
    self.assertEqual(expected_num_batches, len(b_result_batches))

    total_rows = sum([len(x["int"]) for x in total_result_batches])
    a_total_rows = sum([len(x["int"]) for x in a_result_batches])
    b_total_rows = sum([len(x["int"]) for x in b_result_batches])

    print("Split rows: %s => %s, %s" % (total_rows, a_total_rows, b_total_rows))

    # TODO(soergel): figure out how to dequeue the final small batch
    expected_total_rows = 1696  # (num_epochs * 100)

    self.assertEqual(expected_total_rows, total_rows)
    self.assertEqual(1087, a_total_rows)  # stochastic but deterministic
    # self.assertEqual(int(total_rows * 0.7), a_total_rows)
    self.assertEqual(609, b_total_rows)  # stochastic but deterministic
    # self.assertEqual(int(total_rows * 0.3), b_total_rows)

    # The strings used for hashing were all unique in the original data, but
    # we ran 17 epochs, so each one should appear 17 times.  Each copy should
    # be hashed into the same partition, so there should be no overlap of the
    # keys.
    a_strings = set([s for x in a_result_batches for s in x["string"]])
    b_strings = set([s for x in b_result_batches for s in x["string"]])
    self.assertEqual(frozenset(), a_strings & b_strings)


if __name__ == "__main__":
  test.main()
