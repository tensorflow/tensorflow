# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for SavedModel and Checkpoint metrics Python bindings."""

from tensorflow.core.framework import summary_pb2
from tensorflow.python.eager import test
from tensorflow.python.saved_model.pywrap_saved_model import metrics


class MetricsTest(test.TestCase):

  def _get_histogram_proto(self, proto_bytes):
    histogram_proto = summary_pb2.HistogramProto()
    histogram_proto.ParseFromString(proto_bytes)
    return histogram_proto

  def test_SM_increment_write(self):
    self.assertEqual(metrics.GetWrite(write_version="1"), 0)
    metrics.IncrementWriteApi("foo")
    self.assertEqual(metrics.GetWriteApi("foo"), 1)
    metrics.IncrementWrite(write_version="1")
    self.assertEqual(metrics.GetWrite(write_version="1"), 1)

  def test_SM_increment_read(self):
    self.assertEqual(metrics.GetRead(write_version="2"), 0)
    metrics.IncrementReadApi("bar")
    self.assertEqual(metrics.GetReadApi("bar"), 1)
    metrics.IncrementRead(write_version="2")
    self.assertEqual(metrics.GetRead(write_version="2"), 1)

  def test_checkpoint_add_write_duration(self):
    self.assertEqual(
        self._get_histogram_proto(
            metrics.GetCheckpointWriteDurations(api_label="foo")).num, 0)

    metrics.AddCheckpointWriteDuration(api_label="foo", microseconds=100)
    metrics.AddCheckpointWriteDuration(api_label="foo", microseconds=200)

    self.assertEqual(
        self._get_histogram_proto(
            metrics.GetCheckpointWriteDurations(api_label="foo")).num, 2)
    self.assertEqual(
        self._get_histogram_proto(
            metrics.GetCheckpointWriteDurations(api_label="foo")).min, 100)
    self.assertEqual(
        self._get_histogram_proto(
            metrics.GetCheckpointWriteDurations(api_label="foo")).max, 200)

  def test_checkpoint_add_read_duration(self):
    self.assertEqual(
        self._get_histogram_proto(
            metrics.GetCheckpointReadDurations(api_label="bar")).num, 0)

    metrics.AddCheckpointReadDuration(api_label="bar", microseconds=200)
    metrics.AddCheckpointReadDuration(api_label="bar", microseconds=20000)

    self.assertEqual(
        self._get_histogram_proto(
            metrics.GetCheckpointReadDurations(api_label="bar")).num, 2)
    self.assertEqual(
        self._get_histogram_proto(
            metrics.GetCheckpointReadDurations(api_label="bar")).min, 200)
    self.assertEqual(
        self._get_histogram_proto(
            metrics.GetCheckpointReadDurations(api_label="bar")).max, 20000)

  def test_training_time_saved(self):
    self.assertEqual(metrics.GetTrainingTimeSaved(api_label="baz"), 0)
    metrics.AddTrainingTimeSaved(api_label="baz", microseconds=1000)
    self.assertEqual(metrics.GetTrainingTimeSaved(api_label="baz"), 1000)


if __name__ == "__main__":
  test.main()
