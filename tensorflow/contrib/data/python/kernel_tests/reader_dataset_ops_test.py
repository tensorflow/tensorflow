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
"""Tests for the experimental input pipeline ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import zlib

import numpy as np

from tensorflow.contrib.data.python.kernel_tests import reader_dataset_ops_test_base
from tensorflow.contrib.data.python.ops import readers
from tensorflow.python.data.ops import readers as core_readers
from tensorflow.python.data.util import nest
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import test


class ReadBatchFeaturesTest(
    reader_dataset_ops_test_base.ReadBatchFeaturesTestBase):

  def testRead(self):
    for batch_size in [1, 2]:
      for num_epochs in [1, 10]:
        with ops.Graph().as_default() as g:
          with self.test_session(graph=g) as sess:
            # Basic test: read from file 0.
            self.outputs = self.make_batch_feature(
                filenames=self.test_filenames[0],
                num_epochs=num_epochs,
                batch_size=batch_size).make_one_shot_iterator().get_next()
            self.verify_records(sess, batch_size, 0, num_epochs=num_epochs)
            with self.assertRaises(errors.OutOfRangeError):
              self._next_actual_batch(sess)

        with ops.Graph().as_default() as g:
          with self.test_session(graph=g) as sess:
            # Basic test: read from file 1.
            self.outputs = self.make_batch_feature(
                filenames=self.test_filenames[1],
                num_epochs=num_epochs,
                batch_size=batch_size).make_one_shot_iterator().get_next()
            self.verify_records(sess, batch_size, 1, num_epochs=num_epochs)
            with self.assertRaises(errors.OutOfRangeError):
              self._next_actual_batch(sess)

        with ops.Graph().as_default() as g:
          with self.test_session(graph=g) as sess:
            # Basic test: read from both files.
            self.outputs = self.make_batch_feature(
                filenames=self.test_filenames,
                num_epochs=num_epochs,
                batch_size=batch_size).make_one_shot_iterator().get_next()
            self.verify_records(sess, batch_size, num_epochs=num_epochs)
            with self.assertRaises(errors.OutOfRangeError):
              self._next_actual_batch(sess)

  def testReadWithEquivalentDataset(self):
    features = {
        "file": parsing_ops.FixedLenFeature([], dtypes.int64),
        "record": parsing_ops.FixedLenFeature([], dtypes.int64),
    }
    dataset = (
        core_readers.TFRecordDataset(self.test_filenames)
        .map(lambda x: parsing_ops.parse_single_example(x, features))
        .repeat(10).batch(2))
    iterator = dataset.make_initializable_iterator()
    init_op = iterator.initializer
    next_element = iterator.get_next()

    with self.test_session() as sess:
      sess.run(init_op)
      for file_batch, _, _, _, record_batch in self._next_expected_batch(
          range(self._num_files), 2, 10):
        actual_batch = sess.run(next_element)
        self.assertAllEqual(file_batch, actual_batch["file"])
        self.assertAllEqual(record_batch, actual_batch["record"])
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(next_element)

  def testReadWithFusedShuffleRepeatDataset(self):
    num_epochs = 5
    total_records = num_epochs * self._num_records
    for batch_size in [1, 2]:
      # Test that shuffling with same seed produces the same result.
      with ops.Graph().as_default() as g:
        with self.test_session(graph=g) as sess:
          outputs1 = self.make_batch_feature(
              filenames=self.test_filenames[0],
              num_epochs=num_epochs,
              batch_size=batch_size,
              shuffle=True,
              shuffle_seed=5).make_one_shot_iterator().get_next()
          outputs2 = self.make_batch_feature(
              filenames=self.test_filenames[0],
              num_epochs=num_epochs,
              batch_size=batch_size,
              shuffle=True,
              shuffle_seed=5).make_one_shot_iterator().get_next()
          for _ in range(total_records // batch_size):
            batch1 = self._run_actual_batch(outputs1, sess)
            batch2 = self._run_actual_batch(outputs2, sess)
            for i in range(len(batch1)):
              self.assertAllEqual(batch1[i], batch2[i])

      # Test that shuffling with different seeds produces a different order.
      with ops.Graph().as_default() as g:
        with self.test_session(graph=g) as sess:
          outputs1 = self.make_batch_feature(
              filenames=self.test_filenames[0],
              num_epochs=num_epochs,
              batch_size=batch_size,
              shuffle=True,
              shuffle_seed=5).make_one_shot_iterator().get_next()
          outputs2 = self.make_batch_feature(
              filenames=self.test_filenames[0],
              num_epochs=num_epochs,
              batch_size=batch_size,
              shuffle=True,
              shuffle_seed=15).make_one_shot_iterator().get_next()
          all_equal = True
          for _ in range(total_records // batch_size):
            batch1 = self._run_actual_batch(outputs1, sess)
            batch2 = self._run_actual_batch(outputs2, sess)
            for i in range(len(batch1)):
              all_equal = all_equal and np.array_equal(batch1[i], batch2[i])
          self.assertFalse(all_equal)

  def testParallelReadersAndParsers(self):
    num_epochs = 5
    for batch_size in [1, 2]:
      for reader_num_threads in [2, 4]:
        for parser_num_threads in [2, 4]:
          with ops.Graph().as_default() as g:
            with self.test_session(graph=g) as sess:
              self.outputs = self.make_batch_feature(
                  filenames=self.test_filenames,
                  num_epochs=num_epochs,
                  batch_size=batch_size,
                  reader_num_threads=reader_num_threads,
                  parser_num_threads=parser_num_threads).make_one_shot_iterator(
                  ).get_next()
              self.verify_records(
                  sess,
                  batch_size,
                  num_epochs=num_epochs,
                  interleave_cycle_length=reader_num_threads)
              with self.assertRaises(errors.OutOfRangeError):
                self._next_actual_batch(sess)

  def testDropFinalBatch(self):
    for batch_size in [1, 2]:
      for num_epochs in [1, 10]:
        with ops.Graph().as_default():
          # Basic test: read from file 0.
          outputs = self.make_batch_feature(
              filenames=self.test_filenames[0],
              num_epochs=num_epochs,
              batch_size=batch_size,
              drop_final_batch=True).make_one_shot_iterator().get_next()
          for _, tensor in outputs.items():
            if isinstance(tensor, ops.Tensor):  # Guard against SparseTensor.
              self.assertEqual(tensor.shape[0], batch_size)

  def testIndefiniteRepeatShapeInference(self):
    dataset = self.make_batch_feature(
        filenames=self.test_filenames[0], num_epochs=None, batch_size=32)
    for shape, clazz in zip(nest.flatten(dataset.output_shapes),
                            nest.flatten(dataset.output_classes)):
      if issubclass(clazz, ops.Tensor):
        self.assertEqual(32, shape[0])


class MakeCsvDatasetTest(test.TestCase):

  def _make_csv_dataset(self, filenames, batch_size, num_epochs=1, **kwargs):
    return readers.make_csv_dataset(
        filenames, batch_size=batch_size, num_epochs=num_epochs, **kwargs)

  def _setup_files(self, inputs, linebreak="\n", compression_type=None):
    filenames = []
    for i, ip in enumerate(inputs):
      fn = os.path.join(self.get_temp_dir(), "temp_%d.csv" % i)
      contents = linebreak.join(ip).encode("utf-8")
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
      sess,
      dataset,
      batch_size,
      num_epochs,
      label_name,
      expected_output,
      expected_keys,
  ):
    nxt = dataset.make_one_shot_iterator().get_next()

    for expected_features in self._next_expected_batch(
        expected_output,
        expected_keys,
        batch_size,
        num_epochs,
    ):
      actual_features = sess.run(nxt)

      if label_name is not None:
        expected_labels = expected_features.pop(label_name)
        self.assertAllEqual(expected_labels, actual_features[1])
        actual_features = actual_features[0]

      for k in expected_features.keys():
        # Compare features
        self.assertAllEqual(expected_features[k], actual_features[k])

    with self.assertRaises(errors.OutOfRangeError):
      sess.run(nxt)

  def _test_dataset(self,
                    inputs,
                    expected_output,
                    expected_keys,
                    batch_size=1,
                    num_epochs=1,
                    label_name=None,
                    **kwargs):
    """Checks that elements produced by CsvDataset match expected output."""
    # Convert str type because py3 tf strings are bytestrings
    filenames = self._setup_files(
        inputs, compression_type=kwargs.get("compression_type", None))
    with ops.Graph().as_default() as g:
      with self.test_session(graph=g) as sess:
        dataset = self._make_csv_dataset(
            filenames,
            batch_size=batch_size,
            num_epochs=num_epochs,
            label_name=label_name,
            **kwargs)
        self._verify_output(sess, dataset, batch_size, num_epochs, label_name,
                            expected_output, expected_keys)

  def testMakeCSVDataset(self):
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

  def testMakeCSVDataset_withBatchSizeAndEpochs(self):
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

  def testMakeCSVDataset_withCompressionType(self):
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

  def testMakeCSVDataset_withBadInputs(self):
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

  def testMakeCSVDataset_withNoLabel(self):
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

  def testMakeCSVDataset_withNoHeader(self):
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

  def testMakeCSVDataset_withTypes(self):
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

  def testMakeCSVDataset_withNoColNames(self):
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

  def testMakeCSVDataset_withTypeInferenceMismatch(self):
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

  def testMakeCSVDataset_withTypeInference(self):
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

  def testMakeCSVDataset_withTypeInferenceFallthrough(self):
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

  def testMakeCSVDataset_withSelectCols(self):
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

  def testMakeCSVDataset_withSelectColsError(self):
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

  def testMakeCSVDataset_withShuffle(self):
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
      with ops.Graph().as_default() as g:
        with self.test_session(graph=g) as sess:
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
          outputs1 = dataset1.make_one_shot_iterator().get_next()
          outputs2 = dataset2.make_one_shot_iterator().get_next()
          for _ in range(total_records // batch_size):
            batch1 = nest.flatten(sess.run(outputs1))
            batch2 = nest.flatten(sess.run(outputs2))
            for i in range(len(batch1)):
              self.assertAllEqual(batch1[i], batch2[i])

      with ops.Graph().as_default() as g:
        with self.test_session(graph=g) as sess:
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
          outputs1 = dataset1.make_one_shot_iterator().get_next()
          outputs2 = dataset2.make_one_shot_iterator().get_next()
          all_equal = False
          for _ in range(total_records // batch_size):
            batch1 = nest.flatten(sess.run(outputs1))
            batch2 = nest.flatten(sess.run(outputs2))
            for i in range(len(batch1)):
              all_equal = all_equal and np.array_equal(batch1[i], batch2[i])
          self.assertFalse(all_equal)

  def testIndefiniteRepeatShapeInference(self):
    column_names = ["col%d" % i for i in range(5)]
    inputs = [[",".join(x for x in column_names), "0,1,2,3,4", "5,6,7,8,9"], [
        ",".join(x for x in column_names), "10,11,12,13,14", "15,16,17,18,19"
    ]]
    filenames = self._setup_files(inputs)
    dataset = self._make_csv_dataset(filenames, batch_size=32, num_epochs=None)
    for shape in nest.flatten(dataset.output_shapes):
      self.assertEqual(32, shape[0])


class MakeTFRecordDatasetTest(
    reader_dataset_ops_test_base.TFRecordDatasetTestBase):

  def _interleave(self, iterators, cycle_length):
    pending_iterators = iterators
    open_iterators = []
    num_open = 0
    for i in range(cycle_length):
      if pending_iterators:
        open_iterators.append(pending_iterators.pop(0))
        num_open += 1

    while num_open:
      for i in range(min(cycle_length, len(open_iterators))):
        if open_iterators[i] is None:
          continue
        try:
          yield next(open_iterators[i])
        except StopIteration:
          if pending_iterators:
            open_iterators[i] = pending_iterators.pop(0)
          else:
            open_iterators[i] = None
            num_open -= 1

  def _next_expected_batch(self,
                           file_indices,
                           batch_size,
                           num_epochs,
                           cycle_length,
                           drop_final_batch,
                           use_parser_fn):

    def _next_record(file_indices):
      for j in file_indices:
        for i in range(self._num_records):
          yield j, i

    def _next_record_interleaved(file_indices, cycle_length):
      return self._interleave([_next_record([i]) for i in file_indices],
                              cycle_length)

    record_batch = []
    batch_index = 0
    for _ in range(num_epochs):
      if cycle_length == 1:
        next_records = _next_record(file_indices)
      else:
        next_records = _next_record_interleaved(file_indices, cycle_length)
      for f, r in next_records:
        record = self._record(f, r)
        if use_parser_fn:
          record = record[1:]
        record_batch.append(record)
        batch_index += 1
        if len(record_batch) == batch_size:
          yield record_batch
          record_batch = []
          batch_index = 0
    if record_batch and not drop_final_batch:
      yield record_batch

  def _verify_records(self,
                      sess,
                      outputs,
                      batch_size,
                      file_index,
                      num_epochs,
                      interleave_cycle_length,
                      drop_final_batch,
                      use_parser_fn):
    if file_index is not None:
      file_indices = [file_index]
    else:
      file_indices = range(self._num_files)

    for expected_batch in self._next_expected_batch(
        file_indices, batch_size, num_epochs, interleave_cycle_length,
        drop_final_batch, use_parser_fn):
      actual_batch = sess.run(outputs)
      self.assertAllEqual(expected_batch, actual_batch)

  def _read_test(self, batch_size, num_epochs, file_index=None,
                 num_parallel_reads=1, drop_final_batch=False, parser_fn=False):
    if file_index is None:
      file_pattern = self.test_filenames
    else:
      file_pattern = self.test_filenames[file_index]

    if parser_fn:
      fn = lambda x: string_ops.substr(x, 1, 999)
    else:
      fn = None

    with ops.Graph().as_default() as g:
      with self.test_session(graph=g) as sess:
        outputs = readers.make_tf_record_dataset(
            file_pattern=file_pattern,
            num_epochs=num_epochs,
            batch_size=batch_size,
            parser_fn=fn,
            num_parallel_reads=num_parallel_reads,
            drop_final_batch=drop_final_batch,
            shuffle=False).make_one_shot_iterator().get_next()
        self._verify_records(
            sess, outputs, batch_size, file_index, num_epochs=num_epochs,
            interleave_cycle_length=num_parallel_reads,
            drop_final_batch=drop_final_batch, use_parser_fn=parser_fn)
        with self.assertRaises(errors.OutOfRangeError):
          sess.run(outputs)

  def testRead(self):
    for batch_size in [1, 2]:
      for num_epochs in [1, 3]:
        # Basic test: read from file 0.
        self._read_test(batch_size, num_epochs, 0)

        # Basic test: read from file 1.
        self._read_test(batch_size, num_epochs, 1)

        # Basic test: read from both files.
        self._read_test(batch_size, num_epochs)

        # Basic test: read from both files, with parallel reads.
        self._read_test(batch_size, num_epochs, num_parallel_reads=8)

  def testDropFinalBatch(self):
    for batch_size in [1, 2, 10]:
      for num_epochs in [1, 3]:
        # Read from file 0.
        self._read_test(batch_size, num_epochs, 0, drop_final_batch=True)

        # Read from both files.
        self._read_test(batch_size, num_epochs, drop_final_batch=True)

        # Read from both files, with parallel reads.
        self._read_test(batch_size, num_epochs, num_parallel_reads=8,
                        drop_final_batch=True)

  def testParserFn(self):
    for batch_size in [1, 2]:
      for num_epochs in [1, 3]:
        for drop_final_batch in [False, True]:
          self._read_test(batch_size, num_epochs, parser_fn=True,
                          drop_final_batch=drop_final_batch)
          self._read_test(batch_size, num_epochs, num_parallel_reads=8,
                          parser_fn=True, drop_final_batch=drop_final_batch)

  def _shuffle_test(self, batch_size, num_epochs, num_parallel_reads=1,
                    seed=None):
    with ops.Graph().as_default() as g:
      with self.test_session(graph=g) as sess:
        dataset = readers.make_tf_record_dataset(
            file_pattern=self.test_filenames,
            num_epochs=num_epochs,
            batch_size=batch_size,
            num_parallel_reads=num_parallel_reads,
            shuffle=True,
            shuffle_seed=seed)
        iterator = dataset.make_initializable_iterator()
        next_element = iterator.get_next()

        sess.run(iterator.initializer)
        first_batches = []
        try:
          while True:
            first_batches.append(sess.run(next_element))
        except errors.OutOfRangeError:
          pass

        sess.run(iterator.initializer)
        second_batches = []
        try:
          while True:
            second_batches.append(sess.run(next_element))
        except errors.OutOfRangeError:
          pass

        self.assertEqual(len(first_batches), len(second_batches))
        if seed is not None:
          # if you set a seed, should get the same results
          for i in range(len(first_batches)):
            self.assertAllEqual(first_batches[i], second_batches[i])

        expected = []
        for f in range(self._num_files):
          for r in range(self._num_records):
            expected.extend([self._record(f, r)] * num_epochs)

        for batches in (first_batches, second_batches):
          actual = []
          for b in batches:
            actual.extend(b)
          self.assertAllEqual(sorted(expected), sorted(actual))

  def testShuffle(self):
    for batch_size in [1, 2]:
      for num_epochs in [1, 3]:
        for num_parallel_reads in [1, 2]:
          # Test that all expected elements are produced
          self._shuffle_test(batch_size, num_epochs, num_parallel_reads)
          # Test that elements are produced in a consistent order if
          # you specify a seed.
          self._shuffle_test(batch_size, num_epochs, num_parallel_reads,
                             seed=21345)

  def testIndefiniteRepeatShapeInference(self):
    dataset = readers.make_tf_record_dataset(
        file_pattern=self.test_filenames, num_epochs=None, batch_size=32)
    for shape in nest.flatten(dataset.output_shapes):
      self.assertEqual(32, shape[0])


if __name__ == "__main__":
  test.main()
