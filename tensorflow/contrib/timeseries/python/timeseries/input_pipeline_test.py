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
"""Tests for the time series input pipeline."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import tempfile

import numpy

from tensorflow.contrib.timeseries.python.timeseries import input_pipeline
from tensorflow.contrib.timeseries.python.timeseries import test_utils
from tensorflow.contrib.timeseries.python.timeseries.feature_keys import TrainEvalFeatures

from tensorflow.core.example import example_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.lib.io import tf_record
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import coordinator as coordinator_lib
from tensorflow.python.training import queue_runner_impl


def _make_csv_temp_file(to_write, test_tmpdir):
  _, data_file = tempfile.mkstemp(dir=test_tmpdir)
  with open(data_file, "w") as f:
    csvwriter = csv.writer(f)
    for record in to_write:
      csvwriter.writerow(record)
  return data_file


def _make_csv_time_series(num_features, num_samples, test_tmpdir):
  filename = _make_csv_temp_file(
      [[i] + [float(i) * 2. + feature_number
              for feature_number in range(num_features)]
       for i in range(num_samples)],
      test_tmpdir=test_tmpdir)
  return filename


def _make_tfexample_series(num_features, num_samples, test_tmpdir):
  _, data_file = tempfile.mkstemp(dir=test_tmpdir)
  with tf_record.TFRecordWriter(data_file) as writer:
    for i in range(num_samples):
      example = example_pb2.Example()
      times = example.features.feature[TrainEvalFeatures.TIMES]
      times.int64_list.value.append(i)
      values = example.features.feature[TrainEvalFeatures.VALUES]
      values.float_list.value.extend(
          [float(i) * 2. + feature_number
           for feature_number in range(num_features)])
      writer.write(example.SerializeToString())
  return data_file


def _make_numpy_time_series(num_features, num_samples):
  times = numpy.arange(num_samples)
  values = times[:, None] * 2. + numpy.arange(num_features)[None, :]
  return {TrainEvalFeatures.TIMES: times,
          TrainEvalFeatures.VALUES: values}


class RandomWindowInputFnTests(test.TestCase):

  def _random_window_input_fn_test_template(
      self, time_series_reader, window_size, batch_size, num_features,
      discard_out_of_order=False):
    input_fn = input_pipeline.RandomWindowInputFn(
        time_series_reader=time_series_reader,
        window_size=window_size, batch_size=batch_size)
    result, _ = input_fn()
    init_op = variables.local_variables_initializer()
    with self.test_session() as session:
      coordinator = coordinator_lib.Coordinator()
      queue_runner_impl.start_queue_runners(session, coord=coordinator)
      session.run(init_op)
      features = session.run(result)
      coordinator.request_stop()
      coordinator.join()
    self.assertAllEqual([batch_size, window_size],
                        features[TrainEvalFeatures.TIMES].shape)
    for window_position in range(window_size - 1):
      for batch_position in range(batch_size):
        # Checks that all times are contiguous
        self.assertEqual(
            features[TrainEvalFeatures.TIMES][batch_position,
                                              window_position + 1],
            features[TrainEvalFeatures.TIMES][batch_position,
                                              window_position] + 1)
    self.assertAllEqual([batch_size, window_size, num_features],
                        features[TrainEvalFeatures.VALUES].shape)
    self.assertEqual("int64", features[TrainEvalFeatures.TIMES].dtype)
    for feature_number in range(num_features):
      self.assertAllEqual(
          features[TrainEvalFeatures.TIMES] * 2. + feature_number,
          features[TrainEvalFeatures.VALUES][:, :, feature_number])
    return features

  def _test_out_of_order(self, time_series_reader, discard_out_of_order):
    self._random_window_input_fn_test_template(
        time_series_reader=time_series_reader,
        num_features=1, window_size=2, batch_size=5,
        discard_out_of_order=discard_out_of_order)

  def test_csv_sort_out_of_order(self):
    filename = _make_csv_time_series(num_features=1, num_samples=50,
                                     test_tmpdir=self.get_temp_dir())
    time_series_reader = input_pipeline.CSVReader([filename])
    self._test_out_of_order(time_series_reader, discard_out_of_order=False)

  def test_tfexample_sort_out_of_order(self):
    filename = _make_tfexample_series(
        num_features=1, num_samples=50,
        test_tmpdir=self.get_temp_dir())
    time_series_reader = input_pipeline.TFExampleReader(
        [filename],
        features={
            TrainEvalFeatures.TIMES: parsing_ops.FixedLenFeature(
                shape=[], dtype=dtypes.int64),
            TrainEvalFeatures.VALUES: parsing_ops.FixedLenFeature(
                shape=[1], dtype=dtypes.float32)})
    self._test_out_of_order(time_series_reader, discard_out_of_order=False)

  def test_numpy_sort_out_of_order(self):
    data = _make_numpy_time_series(num_features=1, num_samples=50)
    time_series_reader = input_pipeline.NumpyReader(data)
    self._test_out_of_order(time_series_reader, discard_out_of_order=False)

  def test_csv_discard_out_of_order(self):
    filename = _make_csv_time_series(num_features=1, num_samples=50,
                                     test_tmpdir=self.get_temp_dir())
    time_series_reader = input_pipeline.CSVReader([filename])
    self._test_out_of_order(time_series_reader, discard_out_of_order=True)

  def test_csv_discard_out_of_order_window_equal(self):
    filename = _make_csv_time_series(num_features=1, num_samples=3,
                                     test_tmpdir=self.get_temp_dir())
    time_series_reader = input_pipeline.CSVReader([filename])
    self._random_window_input_fn_test_template(
        time_series_reader=time_series_reader,
        num_features=1, window_size=3, batch_size=5,
        discard_out_of_order=True)

  def test_csv_discard_out_of_order_window_too_large(self):
    filename = _make_csv_time_series(num_features=1, num_samples=2,
                                     test_tmpdir=self.get_temp_dir())
    time_series_reader = input_pipeline.CSVReader([filename])
    with self.assertRaises(errors.OutOfRangeError):
      self._random_window_input_fn_test_template(
          time_series_reader=time_series_reader,
          num_features=1, window_size=3, batch_size=5,
          discard_out_of_order=True)

  def test_csv_no_data(self):
    filename = _make_csv_time_series(num_features=1, num_samples=0,
                                     test_tmpdir=self.get_temp_dir())
    time_series_reader = input_pipeline.CSVReader([filename])
    with self.assertRaises(errors.OutOfRangeError):
      self._test_out_of_order(time_series_reader, discard_out_of_order=True)

  def test_numpy_discard_out_of_order(self):
    data = _make_numpy_time_series(num_features=1, num_samples=50)
    time_series_reader = input_pipeline.NumpyReader(data)
    self._test_out_of_order(time_series_reader, discard_out_of_order=True)

  def test_numpy_discard_out_of_order_window_equal(self):
    data = _make_numpy_time_series(num_features=1, num_samples=3)
    time_series_reader = input_pipeline.NumpyReader(data)
    self._random_window_input_fn_test_template(
        time_series_reader=time_series_reader,
        num_features=1, window_size=3, batch_size=5,
        discard_out_of_order=True)

  def test_numpy_discard_out_of_order_window_too_large(self):
    data = _make_numpy_time_series(num_features=1, num_samples=2)
    time_series_reader = input_pipeline.NumpyReader(data)
    with self.assertRaisesRegexp(ValueError, "only 2 records were available"):
      self._random_window_input_fn_test_template(
          time_series_reader=time_series_reader,
          num_features=1, window_size=3, batch_size=5,
          discard_out_of_order=True)

  def _test_multivariate(self, time_series_reader, num_features):
    self._random_window_input_fn_test_template(
        time_series_reader=time_series_reader,
        num_features=num_features,
        window_size=2,
        batch_size=5)

  def test_csv_multivariate(self):
    filename = _make_csv_time_series(num_features=2, num_samples=50,
                                     test_tmpdir=self.get_temp_dir())
    time_series_reader = input_pipeline.CSVReader(
        [filename],
        column_names=(TrainEvalFeatures.TIMES, TrainEvalFeatures.VALUES,
                      TrainEvalFeatures.VALUES))
    self._test_multivariate(time_series_reader=time_series_reader,
                            num_features=2)

  def test_tfexample_multivariate(self):
    filename = _make_tfexample_series(
        num_features=2, num_samples=50,
        test_tmpdir=self.get_temp_dir())
    time_series_reader = input_pipeline.TFExampleReader(
        [filename],
        features={
            TrainEvalFeatures.TIMES: parsing_ops.FixedLenFeature(
                shape=[], dtype=dtypes.int64),
            TrainEvalFeatures.VALUES: parsing_ops.FixedLenFeature(
                shape=[2], dtype=dtypes.float32)})
    self._test_multivariate(time_series_reader=time_series_reader,
                            num_features=2)

  def test_numpy_multivariate(self):
    data = _make_numpy_time_series(num_features=3, num_samples=50)
    time_series_reader = input_pipeline.NumpyReader(data)
    self._test_multivariate(time_series_reader, num_features=3)

  def test_numpy_withbatch(self):
    data_nobatch = _make_numpy_time_series(num_features=4, num_samples=100)
    data = {feature_name: feature_value[None]
            for feature_name, feature_value in data_nobatch.items()}
    time_series_reader = input_pipeline.NumpyReader(data)
    self._random_window_input_fn_test_template(
        time_series_reader=time_series_reader,
        num_features=4,
        window_size=3,
        batch_size=5)

  def test_numpy_nobatch_nofeatures(self):
    data = _make_numpy_time_series(num_features=1, num_samples=100)
    data[TrainEvalFeatures.VALUES] = data[TrainEvalFeatures.VALUES][:, 0]
    time_series_reader = input_pipeline.NumpyReader(data)
    self._random_window_input_fn_test_template(
        time_series_reader=time_series_reader,
        num_features=1,
        window_size=16,
        batch_size=16)


class WholeDatasetInputFnTests(test.TestCase):

  def _whole_dataset_input_fn_test_template(
      self, time_series_reader, num_features, num_samples):
    result, _ = input_pipeline.WholeDatasetInputFn(time_series_reader)()
    with self.test_session() as session:
      session.run(variables.local_variables_initializer())
      coordinator = coordinator_lib.Coordinator()
      queue_runner_impl.start_queue_runners(session, coord=coordinator)
      features = session.run(result)
      coordinator.request_stop()
      coordinator.join()
    self.assertEqual("int64", features[TrainEvalFeatures.TIMES].dtype)
    self.assertAllEqual(numpy.arange(num_samples, dtype=numpy.int64)[None, :],
                        features[TrainEvalFeatures.TIMES])
    for feature_number in range(num_features):
      self.assertAllEqual(
          features[TrainEvalFeatures.TIMES] * 2. + feature_number,
          features[TrainEvalFeatures.VALUES][:, :, feature_number])

  def test_csv(self):
    filename = _make_csv_time_series(num_features=3, num_samples=50,
                                     test_tmpdir=self.get_temp_dir())
    time_series_reader = input_pipeline.CSVReader(
        [filename],
        column_names=(TrainEvalFeatures.TIMES, TrainEvalFeatures.VALUES,
                      TrainEvalFeatures.VALUES, TrainEvalFeatures.VALUES))
    self._whole_dataset_input_fn_test_template(
        time_series_reader=time_series_reader, num_features=3, num_samples=50)

  def test_csv_no_data(self):
    filename = _make_csv_time_series(num_features=1, num_samples=0,
                                     test_tmpdir=self.get_temp_dir())
    time_series_reader = input_pipeline.CSVReader([filename])
    with self.assertRaises(errors.OutOfRangeError):
      self._whole_dataset_input_fn_test_template(
          time_series_reader=time_series_reader, num_features=1, num_samples=50)

  def test_tfexample(self):
    filename = _make_tfexample_series(
        num_features=4, num_samples=100,
        test_tmpdir=self.get_temp_dir())
    time_series_reader = input_pipeline.TFExampleReader(
        [filename],
        features={
            TrainEvalFeatures.TIMES: parsing_ops.FixedLenFeature(
                shape=[], dtype=dtypes.int64),
            TrainEvalFeatures.VALUES: parsing_ops.FixedLenFeature(
                shape=[4], dtype=dtypes.float32)})
    self._whole_dataset_input_fn_test_template(
        time_series_reader=time_series_reader, num_features=4, num_samples=100)

  def test_numpy(self):
    data = _make_numpy_time_series(num_features=4, num_samples=100)
    time_series_reader = input_pipeline.NumpyReader(data)
    self._whole_dataset_input_fn_test_template(
        time_series_reader=time_series_reader, num_features=4, num_samples=100)

  def test_numpy_withbatch(self):
    data_nobatch = _make_numpy_time_series(num_features=4, num_samples=100)
    data = {feature_name: feature_value[None]
            for feature_name, feature_value in data_nobatch.items()}
    time_series_reader = input_pipeline.NumpyReader(data)
    self._whole_dataset_input_fn_test_template(
        time_series_reader=time_series_reader, num_features=4, num_samples=100)

  def test_numpy_nobatch_nofeatures(self):
    data = _make_numpy_time_series(num_features=1, num_samples=100)
    data[TrainEvalFeatures.VALUES] = data[TrainEvalFeatures.VALUES][:, 0]
    time_series_reader = input_pipeline.NumpyReader(data)
    self._whole_dataset_input_fn_test_template(
        time_series_reader=time_series_reader, num_features=1, num_samples=100)


class AllWindowInputFnTests(test.TestCase):

  def _all_window_input_fn_test_template(
      self, time_series_reader, num_samples, window_size,
      original_numpy_features=None):
    input_fn = test_utils.AllWindowInputFn(
        time_series_reader=time_series_reader,
        window_size=window_size)
    features, _ = input_fn()
    init_op = variables.local_variables_initializer()
    with self.test_session() as session:
      coordinator = coordinator_lib.Coordinator()
      queue_runner_impl.start_queue_runners(session, coord=coordinator)
      session.run(init_op)
      chunked_times, chunked_values = session.run(
          [features[TrainEvalFeatures.TIMES],
           features[TrainEvalFeatures.VALUES]])
      coordinator.request_stop()
      coordinator.join()
    self.assertAllEqual([num_samples - window_size + 1, window_size],
                        chunked_times.shape)
    if original_numpy_features is not None:
      original_times = original_numpy_features[TrainEvalFeatures.TIMES]
      original_values = original_numpy_features[TrainEvalFeatures.VALUES]
      self.assertAllEqual(original_times, numpy.unique(chunked_times))
      self.assertAllEqual(original_values[chunked_times],
                          chunked_values)

  def test_csv(self):
    filename = _make_csv_time_series(num_features=1, num_samples=50,
                                     test_tmpdir=self.get_temp_dir())
    time_series_reader = input_pipeline.CSVReader(
        [filename],
        column_names=(TrainEvalFeatures.TIMES, TrainEvalFeatures.VALUES))
    self._all_window_input_fn_test_template(
        time_series_reader=time_series_reader, num_samples=50, window_size=10)

  def test_numpy(self):
    data = _make_numpy_time_series(num_features=2, num_samples=31)
    time_series_reader = input_pipeline.NumpyReader(data)
    self._all_window_input_fn_test_template(
        time_series_reader=time_series_reader, original_numpy_features=data,
        num_samples=31, window_size=5)


if __name__ == "__main__":
  test.main()
