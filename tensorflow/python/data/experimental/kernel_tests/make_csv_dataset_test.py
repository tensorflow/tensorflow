# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for `tf.data.experimental.make_csv_dataset()`."""
import gzip
import os
import zlib

from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.experimental.ops import readers
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.platform import test


class MakeCsvDatasetTest(test_base.DatasetTestBase, parameterized.TestCase):

  def _make_csv_dataset(self, filenames, batch_size, num_epochs=1, **kwargs):
    return readers.make_csv_dataset(
        filenames, batch_size=batch_size, num_epochs=num_epochs, **kwargs)

  def _setup_files(self,
                   inputs,
                   linebreak="\n",
                   compression_type=None,
                   encoding="utf-8"):
    filenames = []
    for i, ip in enumerate(inputs):
      fn = os.path.join(self.get_temp_dir(), "temp_%d.csv" % i)
      contents = linebreak.join(ip).encode(encoding)
      if compression_type is None:
        with open(fn, "wb") as f:
          f.write(contents)
      elif compression_type == "GZIP":
        with gzip.GzipFile(fn, "wb") as f:
          f.write(contents)
      elif compression_type == "ZLIB":
        contents = zlib.compress(contents)
        with open(fn, "wb") as f:
          f.write(contents)
      else:
        raise ValueError("Unsupported compression_type", compression_type)
      filenames.append(fn)
    return filenames

  def _next_expected_batch(self, expected_output, expected_keys, batch_size,
                           num_epochs):
    features = {k: [] for k in expected_keys}
    for _ in range(num_epochs):
      for values in expected_output:
        for n, key in enumerate(expected_keys):
          features[key].append(values[n])
        if len(features[expected_keys[0]]) == batch_size:
          yield features
          features = {k: [] for k in expected_keys}
    if features[expected_keys[0]]:  # Leftover from the last batch
      yield features

  def _verify_output(
      self,
      dataset,
      batch_size,
      num_epochs,
      label_name,
      expected_output,
      expected_keys,
  ):
    get_next = self.getNext(dataset)

    for expected_features in self._next_expected_batch(
        expected_output,
        expected_keys,
        batch_size,
        num_epochs,
    ):
      actual_features = self.evaluate(get_next())

      if label_name is not None:
        expected_labels = expected_features.pop(label_name)
        self.assertAllEqual(expected_labels, actual_features[1])
        actual_features = actual_features[0]

      for k in expected_features.keys():
        # Compare features
        self.assertAllEqual(expected_features[k], actual_features[k])

    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(get_next())

  def _test_dataset(self,
                    inputs,
                    expected_output,
                    expected_keys,
                    batch_size=1,
                    num_epochs=1,
                    label_name=None,
                    encoding="utf-8",
                    **kwargs):
    """Checks that elements produced by CsvDataset match expected output."""
    # Convert str type because py3 tf strings are bytestrings
    filenames = self._setup_files(
        inputs,
        compression_type=kwargs.get("compression_type", None),
        encoding=encoding)
    dataset = self._make_csv_dataset(
        filenames,
        batch_size=batch_size,
        num_epochs=num_epochs,
        label_name=label_name,
        encoding=encoding,
        **kwargs)
    self._verify_output(dataset, batch_size, num_epochs, label_name,
                        expected_output, expected_keys)

  @combinations.generate(test_base.default_test_combinations())
  def testBasic(self):
    """Tests making a CSV dataset with keys and defaults provided."""
    record_defaults = [
        constant_op.constant([], dtypes.int32),
        constant_op.constant([], dtypes.int64),
        constant_op.constant([], dtypes.float32),
        constant_op.constant([], dtypes.float64),
        constant_op.constant([], dtypes.string)
    ]

    column_names = ["col%d" % i for i in range(5)]
    inputs = [[",".join(x for x in column_names), "0,1,2,3,4", "5,6,7,8,9"], [
        ",".join(x for x in column_names), "10,11,12,13,14", "15,16,17,18,19"
    ]]
    expected_output = [[0, 1, 2, 3, b"4"], [5, 6, 7, 8, b"9"],
                       [10, 11, 12, 13, b"14"], [15, 16, 17, 18, b"19"]]
    label = "col0"

    self._test_dataset(
        inputs,
        expected_output=expected_output,
        expected_keys=column_names,
        column_names=column_names,
        label_name=label,
        batch_size=1,
        num_epochs=1,
        shuffle=False,
        header=True,
        column_defaults=record_defaults,
    )

  @combinations.generate(test_base.default_test_combinations())
  def testEncoding(self):
    """Tests making a CSV dataset with an encoding except for utf-8."""
    record_defaults = [
        constant_op.constant([], dtypes.string),
        constant_op.constant([], dtypes.string)
    ]

    column_names = ["col%d" % i for i in range(2)]
    inputs = [[",".join(x for x in column_names), "さる,猿", "とり,鳥"],
              [",".join(x for x in column_names), "いぬ,犬", "ねこ,猫"]]
    expected_output = [["さる".encode("shift-jis"), "猿".encode("shift-jis")],
                       ["とり".encode("shift-jis"), "鳥".encode("shift-jis")],
                       ["いぬ".encode("shift-jis"), "犬".encode("shift-jis")],
                       ["ねこ".encode("shift-jis"), "猫".encode("shift-jis")]]
    label = "col0"

    self._test_dataset(
        inputs,
        expected_output=expected_output,
        expected_keys=column_names,
        column_names=column_names,
        label_name=label,
        batch_size=1,
        num_epochs=1,
        shuffle=False,
        header=True,
        column_defaults=record_defaults,
        encoding="shift-jis",
    )

  @combinations.generate(test_base.default_test_combinations())
  def testWithBatchSizeAndEpochs(self):
    """Tests making a CSV dataset with keys and defaults provided."""
    record_defaults = [
        constant_op.constant([], dtypes.int32),
        constant_op.constant([], dtypes.int64),
        constant_op.constant([], dtypes.float32),
        constant_op.constant([], dtypes.float64),
        constant_op.constant([], dtypes.string)
    ]

    column_names = ["col%d" % i for i in range(5)]
    inputs = [[",".join(x for x in column_names), "0,1,2,3,4", "5,6,7,8,9"], [
        ",".join(x for x in column_names), "10,11,12,13,14", "15,16,17,18,19"
    ]]
    expected_output = [[0, 1, 2, 3, b"4"], [5, 6, 7, 8, b"9"],
                       [10, 11, 12, 13, b"14"], [15, 16, 17, 18, b"19"]]
    label = "col0"

    self._test_dataset(
        inputs,
        expected_output=expected_output,
        expected_keys=column_names,
        column_names=column_names,
        label_name=label,
        batch_size=3,
        num_epochs=10,
        shuffle=False,
        header=True,
        column_defaults=record_defaults,
    )

  @combinations.generate(test_base.default_test_combinations())
  def testWithCompressionType(self):
    """Tests `compression_type` argument."""
    record_defaults = [
        constant_op.constant([], dtypes.int32),
        constant_op.constant([], dtypes.int64),
        constant_op.constant([], dtypes.float32),
        constant_op.constant([], dtypes.float64),
        constant_op.constant([], dtypes.string)
    ]

    column_names = ["col%d" % i for i in range(5)]
    inputs = [[",".join(x for x in column_names), "0,1,2,3,4", "5,6,7,8,9"], [
        ",".join(x for x in column_names), "10,11,12,13,14", "15,16,17,18,19"
    ]]
    expected_output = [[0, 1, 2, 3, b"4"], [5, 6, 7, 8, b"9"],
                       [10, 11, 12, 13, b"14"], [15, 16, 17, 18, b"19"]]
    label = "col0"

    for compression_type in ("GZIP", "ZLIB"):
      self._test_dataset(
          inputs,
          expected_output=expected_output,
          expected_keys=column_names,
          column_names=column_names,
          label_name=label,
          batch_size=1,
          num_epochs=1,
          shuffle=False,
          header=True,
          column_defaults=record_defaults,
          compression_type=compression_type,
      )

  @combinations.generate(test_base.default_test_combinations())
  def testWithCompressionTypeAndNoColumnNames(self):
    """Tests `compression_type` argument."""
    record_defaults = [
        constant_op.constant([], dtypes.int32),
        constant_op.constant([], dtypes.int64),
        constant_op.constant([], dtypes.float32),
        constant_op.constant([], dtypes.float64),
        constant_op.constant([], dtypes.string)
    ]

    column_names = ["col%d" % i for i in range(5)]
    inputs = [[",".join(x for x in column_names), "0,1,2,3,4", "5,6,7,8,9"],
              [
                  ",".join(x for x in column_names), "10,11,12,13,14",
                  "15,16,17,18,19"
              ]]
    expected_output = [[0, 1, 2, 3, b"4"], [5, 6, 7, 8, b"9"],
                       [10, 11, 12, 13, b"14"], [15, 16, 17, 18, b"19"]]
    label = "col0"

    self._test_dataset(
        inputs,
        expected_output=expected_output,
        expected_keys=column_names,
        label_name=label,
        batch_size=1,
        num_epochs=1,
        shuffle=False,
        header=True,
        column_defaults=record_defaults,
        compression_type="GZIP",
    )

    with self.assertRaisesRegex(ValueError,
                                "`compression_type` ZLIB is not supported"):
      self._test_dataset(
          inputs,
          expected_output=expected_output,
          expected_keys=column_names,
          label_name=label,
          batch_size=1,
          num_epochs=1,
          shuffle=False,
          header=True,
          column_defaults=record_defaults,
          compression_type="ZLIB",
      )

  @combinations.generate(test_base.default_test_combinations())
  def testWithBadInputs(self):
    """Tests that exception is raised when input is malformed.
    """
    record_defaults = [
        constant_op.constant([], dtypes.int32),
        constant_op.constant([], dtypes.int64),
        constant_op.constant([], dtypes.float32),
        constant_op.constant([], dtypes.float64),
        constant_op.constant([], dtypes.string)
    ]

    column_names = ["col%d" % i for i in range(5)]
    inputs = [[",".join(x for x in column_names), "0,1,2,3,4", "5,6,7,8,9"], [
        ",".join(x for x in column_names), "10,11,12,13,14", "15,16,17,18,19"
    ]]
    filenames = self._setup_files(inputs)

    # Duplicate column names
    with self.assertRaises(ValueError):
      self._make_csv_dataset(
          filenames,
          batch_size=1,
          column_defaults=record_defaults,
          label_name="col0",
          column_names=column_names * 2)

    # Label key not one of column names
    with self.assertRaises(ValueError):
      self._make_csv_dataset(
          filenames,
          batch_size=1,
          column_defaults=record_defaults,
          label_name="not_a_real_label",
          column_names=column_names)

  @combinations.generate(test_base.default_test_combinations())
  def testWithNoLabel(self):
    """Tests making a CSV dataset with no label provided."""
    record_defaults = [
        constant_op.constant([], dtypes.int32),
        constant_op.constant([], dtypes.int64),
        constant_op.constant([], dtypes.float32),
        constant_op.constant([], dtypes.float64),
        constant_op.constant([], dtypes.string)
    ]

    column_names = ["col%d" % i for i in range(5)]
    inputs = [[",".join(x for x in column_names), "0,1,2,3,4", "5,6,7,8,9"], [
        ",".join(x for x in column_names), "10,11,12,13,14", "15,16,17,18,19"
    ]]
    expected_output = [[0, 1, 2, 3, b"4"], [5, 6, 7, 8, b"9"],
                       [10, 11, 12, 13, b"14"], [15, 16, 17, 18, b"19"]]

    self._test_dataset(
        inputs,
        expected_output=expected_output,
        expected_keys=column_names,
        column_names=column_names,
        batch_size=1,
        num_epochs=1,
        shuffle=False,
        header=True,
        column_defaults=record_defaults,
    )

  @combinations.generate(test_base.default_test_combinations())
  def testWithNoHeader(self):
    """Tests that datasets can be created from CSV files with no header line.
    """
    record_defaults = [
        constant_op.constant([], dtypes.int32),
        constant_op.constant([], dtypes.int64),
        constant_op.constant([], dtypes.float32),
        constant_op.constant([], dtypes.float64),
        constant_op.constant([], dtypes.string)
    ]

    column_names = ["col%d" % i for i in range(5)]
    inputs = [["0,1,2,3,4", "5,6,7,8,9"], ["10,11,12,13,14", "15,16,17,18,19"]]
    expected_output = [[0, 1, 2, 3, b"4"], [5, 6, 7, 8, b"9"],
                       [10, 11, 12, 13, b"14"], [15, 16, 17, 18, b"19"]]
    label = "col0"

    self._test_dataset(
        inputs,
        expected_output=expected_output,
        expected_keys=column_names,
        column_names=column_names,
        label_name=label,
        batch_size=1,
        num_epochs=1,
        shuffle=False,
        header=False,
        column_defaults=record_defaults,
    )

  @combinations.generate(test_base.default_test_combinations())
  def testWithTypes(self):
    """Tests that defaults can be a dtype instead of a Tensor for required vals.
    """
    record_defaults = [
        dtypes.int32, dtypes.int64, dtypes.float32, dtypes.float64,
        dtypes.string
    ]

    column_names = ["col%d" % i for i in range(5)]
    inputs = [[",".join(x[0] for x in column_names), "0,1,2,3,4", "5,6,7,8,9"],
              [
                  ",".join(x[0] for x in column_names), "10,11,12,13,14",
                  "15,16,17,18,19"
              ]]
    expected_output = [[0, 1, 2, 3, b"4"], [5, 6, 7, 8, b"9"],
                       [10, 11, 12, 13, b"14"], [15, 16, 17, 18, b"19"]]
    label = "col0"

    self._test_dataset(
        inputs,
        expected_output=expected_output,
        expected_keys=column_names,
        column_names=column_names,
        label_name=label,
        batch_size=1,
        num_epochs=1,
        shuffle=False,
        header=True,
        column_defaults=record_defaults,
    )

  @combinations.generate(test_base.default_test_combinations())
  def testWithNoColNames(self):
    """Tests that datasets can be created when column names are not specified.

    In that case, we should infer the column names from the header lines.
    """
    record_defaults = [
        constant_op.constant([], dtypes.int32),
        constant_op.constant([], dtypes.int64),
        constant_op.constant([], dtypes.float32),
        constant_op.constant([], dtypes.float64),
        constant_op.constant([], dtypes.string)
    ]

    column_names = ["col%d" % i for i in range(5)]
    inputs = [[",".join(x for x in column_names), "0,1,2,3,4", "5,6,7,8,9"], [
        ",".join(x for x in column_names), "10,11,12,13,14", "15,16,17,18,19"
    ]]
    expected_output = [[0, 1, 2, 3, b"4"], [5, 6, 7, 8, b"9"],
                       [10, 11, 12, 13, b"14"], [15, 16, 17, 18, b"19"]]
    label = "col0"

    self._test_dataset(
        inputs,
        expected_output=expected_output,
        expected_keys=column_names,
        label_name=label,
        batch_size=1,
        num_epochs=1,
        shuffle=False,
        header=True,
        column_defaults=record_defaults,
    )

  @combinations.generate(test_base.default_test_combinations())
  def testWithTypeInferenceMismatch(self):
    # Test that error is thrown when num fields doesn't match columns
    column_names = ["col%d" % i for i in range(5)]
    inputs = [[",".join(x for x in column_names), "0,1,2,3,4", "5,6,7,8,9"], [
        ",".join(x for x in column_names), "10,11,12,13,14", "15,16,17,18,19"
    ]]
    filenames = self._setup_files(inputs)
    with self.assertRaises(ValueError):
      self._make_csv_dataset(
          filenames,
          column_names=column_names + ["extra_name"],
          column_defaults=None,
          batch_size=2,
          num_epochs=10)

  @combinations.generate(test_base.default_test_combinations())
  def testWithTypeInference(self):
    """Tests that datasets can be created when no defaults are specified.

    In that case, we should infer the types from the first N records.
    """
    column_names = ["col%d" % i for i in range(5)]
    str_int32_max = str(2**33)
    inputs = [[
        ",".join(x for x in column_names),
        "0,%s,2.0,3e50,rabbit" % str_int32_max
    ]]
    expected_output = [[0, 2**33, 2.0, 3e50, b"rabbit"]]
    label = "col0"

    self._test_dataset(
        inputs,
        expected_output=expected_output,
        expected_keys=column_names,
        column_names=column_names,
        label_name=label,
        batch_size=1,
        num_epochs=1,
        shuffle=False,
        header=True,
    )

  @combinations.generate(test_base.default_test_combinations())
  def testWithTypeInferenceFallthrough(self):
    """Tests that datasets can be created when no defaults are specified.

    Tests on a deliberately tricky file.
    """
    column_names = ["col%d" % i for i in range(5)]
    str_int32_max = str(2**33)
    inputs = [[
        ",".join(x for x in column_names),
        ",,,,",
        "0,0,0.0,0.0,0.0",
        "0,%s,2.0,3e50,rabbit" % str_int32_max,
        ",,,,",
    ]]
    expected_output = [[0, 0, 0, 0, b""], [0, 0, 0, 0, b"0.0"],
                       [0, 2**33, 2.0, 3e50, b"rabbit"], [0, 0, 0, 0, b""]]
    label = "col0"

    self._test_dataset(
        inputs,
        expected_output=expected_output,
        expected_keys=column_names,
        column_names=column_names,
        label_name=label,
        batch_size=1,
        num_epochs=1,
        shuffle=False,
        header=True,
    )

  @combinations.generate(test_base.default_test_combinations())
  def testWithNAValuesAndFieldDelim(self):
    """Tests that datasets can be created from different delim and na_value."""
    column_names = ["col%d" % i for i in range(5)]
    inputs = [["0 1 2 3 4", "5 6 7 8 9"], ["10 11 12 13 14", "15 16 17 ? 19"]]
    expected_output = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14],
                       [15, 16, 17, 0, 19]]
    label = "col0"

    self._test_dataset(
        inputs,
        expected_output=expected_output,
        expected_keys=column_names,
        column_names=column_names,
        label_name=label,
        batch_size=1,
        num_epochs=1,
        shuffle=False,
        header=False,
        na_value="?",
        field_delim=" ",
    )

  @combinations.generate(test_base.default_test_combinations())
  def testWithSelectCols(self):
    record_defaults = [
        constant_op.constant([], dtypes.int32),
        constant_op.constant([], dtypes.int64),
        constant_op.constant([], dtypes.float32),
        constant_op.constant([], dtypes.float64),
        constant_op.constant([], dtypes.string)
    ]
    column_names = ["col%d" % i for i in range(5)]
    str_int32_max = str(2**33)
    inputs = [[
        ",".join(x for x in column_names),
        "0,%s,2.0,3e50,rabbit" % str_int32_max
    ]]
    expected_output = [[0, 2**33, 2.0, 3e50, b"rabbit"]]

    select_cols = [1, 3, 4]
    self._test_dataset(
        inputs,
        expected_output=[[x[i] for i in select_cols] for x in expected_output],
        expected_keys=[column_names[i] for i in select_cols],
        column_names=column_names,
        column_defaults=[record_defaults[i] for i in select_cols],
        batch_size=1,
        num_epochs=1,
        shuffle=False,
        header=True,
        select_columns=select_cols,
    )

    # Can still do inference without provided defaults
    self._test_dataset(
        inputs,
        expected_output=[[x[i] for i in select_cols] for x in expected_output],
        expected_keys=[column_names[i] for i in select_cols],
        column_names=column_names,
        batch_size=1,
        num_epochs=1,
        shuffle=False,
        header=True,
        select_columns=select_cols,
    )

    # Can still do column name inference
    self._test_dataset(
        inputs,
        expected_output=[[x[i] for i in select_cols] for x in expected_output],
        expected_keys=[column_names[i] for i in select_cols],
        batch_size=1,
        num_epochs=1,
        shuffle=False,
        header=True,
        select_columns=select_cols,
    )

    # Can specify column names instead of indices
    self._test_dataset(
        inputs,
        expected_output=[[x[i] for i in select_cols] for x in expected_output],
        expected_keys=[column_names[i] for i in select_cols],
        column_names=column_names,
        batch_size=1,
        num_epochs=1,
        shuffle=False,
        header=True,
        select_columns=[column_names[i] for i in select_cols],
    )

  @combinations.generate(test_base.default_test_combinations())
  def testWithSelectColsError(self):
    record_defaults = [
        constant_op.constant([], dtypes.int32),
        constant_op.constant([], dtypes.int64),
        constant_op.constant([], dtypes.float32),
        constant_op.constant([], dtypes.float64),
        constant_op.constant([], dtypes.string)
    ]
    column_names = ["col%d" % i for i in range(5)]
    str_int32_max = str(2**33)
    inputs = [[
        ",".join(x for x in column_names),
        "0,%s,2.0,3e50,rabbit" % str_int32_max
    ]]

    select_cols = [1, 3, 4]
    filenames = self._setup_files(inputs)

    with self.assertRaises(ValueError):
      # Mismatch in number of defaults and number of columns selected,
      # should raise an error
      self._make_csv_dataset(
          filenames,
          batch_size=1,
          column_defaults=record_defaults,
          column_names=column_names,
          select_columns=select_cols)

    with self.assertRaises(ValueError):
      # Invalid column name should raise an error
      self._make_csv_dataset(
          filenames,
          batch_size=1,
          column_defaults=[[0]],
          column_names=column_names,
          label_name=None,
          select_columns=["invalid_col_name"])

  @combinations.generate(test_base.default_test_combinations())
  def testWithShuffle(self):
    record_defaults = [
        constant_op.constant([], dtypes.int32),
        constant_op.constant([], dtypes.int64),
        constant_op.constant([], dtypes.float32),
        constant_op.constant([], dtypes.float64),
        constant_op.constant([], dtypes.string)
    ]

    def str_series(st):
      return ",".join(str(i) for i in range(st, st + 5))

    column_names = ["col%d" % i for i in range(5)]
    inputs = [
        [",".join(x for x in column_names)
        ] + [str_series(5 * i) for i in range(15)],
        [",".join(x for x in column_names)] +
        [str_series(5 * i) for i in range(15, 20)],
    ]

    filenames = self._setup_files(inputs)

    total_records = 20
    for batch_size in [1, 2]:
      # Test that shuffling with the same seed produces the same result
      dataset1 = self._make_csv_dataset(
          filenames,
          column_defaults=record_defaults,
          column_names=column_names,
          batch_size=batch_size,
          header=True,
          shuffle=True,
          shuffle_seed=5,
          num_epochs=2,
      )
      dataset2 = self._make_csv_dataset(
          filenames,
          column_defaults=record_defaults,
          column_names=column_names,
          batch_size=batch_size,
          header=True,
          shuffle=True,
          shuffle_seed=5,
          num_epochs=2,
      )
      next1 = self.getNext(dataset1)
      next2 = self.getNext(dataset2)
      for _ in range(total_records // batch_size):
        batch1 = nest.flatten(self.evaluate(next1()))
        batch2 = nest.flatten(self.evaluate(next2()))
        for i in range(len(batch1)):
          self.assertAllEqual(batch1[i], batch2[i])

      # Test that shuffling with a different seed produces different results
      dataset1 = self._make_csv_dataset(
          filenames,
          column_defaults=record_defaults,
          column_names=column_names,
          batch_size=batch_size,
          header=True,
          shuffle=True,
          shuffle_seed=5,
          num_epochs=2,
      )
      dataset2 = self._make_csv_dataset(
          filenames,
          column_defaults=record_defaults,
          column_names=column_names,
          batch_size=batch_size,
          header=True,
          shuffle=True,
          shuffle_seed=6,
          num_epochs=2,
      )
      next1 = self.getNext(dataset1)
      next2 = self.getNext(dataset2)
      all_equal = False
      for _ in range(total_records // batch_size):
        batch1 = nest.flatten(self.evaluate(next1()))
        batch2 = nest.flatten(self.evaluate(next2()))
        for i in range(len(batch1)):
          all_equal = all_equal and np.array_equal(batch1[i], batch2[i])
      self.assertFalse(all_equal)

  @combinations.generate(test_base.default_test_combinations())
  def testIndefiniteRepeatShapeInference(self):
    column_names = ["col%d" % i for i in range(5)]
    inputs = [[",".join(x for x in column_names), "0,1,2,3,4", "5,6,7,8,9"], [
        ",".join(x for x in column_names), "10,11,12,13,14", "15,16,17,18,19"
    ]]
    filenames = self._setup_files(inputs)
    dataset = self._make_csv_dataset(filenames, batch_size=32, num_epochs=None)
    for shape in nest.flatten(dataset_ops.get_legacy_output_shapes(dataset)):
      self.assertEqual(32, shape[0])

  @combinations.generate(test_base.default_test_combinations())
  def testFieldOrder(self):
    data = [[
        "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19",
        "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19"
    ]]
    file_path = self._setup_files(data)

    ds = readers.make_csv_dataset(
        file_path, batch_size=1, shuffle=False, num_epochs=1)
    nxt = self.getNext(ds)

    result = list(self.evaluate(nxt()).values())

    self.assertEqual(result, sorted(result))


if __name__ == "__main__":
  test.main()
