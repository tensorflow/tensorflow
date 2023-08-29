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

import os

from tensorflow.core.framework import summary_pb2
from tensorflow.core.framework import versions_pb2
from tensorflow.core.protobuf import fingerprint_pb2
from tensorflow.python.eager import test
from tensorflow.python.saved_model.pywrap_saved_model import metrics


class MetricsTest(test.TestCase):

  def _get_histogram_proto(self, proto_bytes):
    histogram_proto = summary_pb2.HistogramProto()
    histogram_proto.ParseFromString(proto_bytes)
    return histogram_proto

  def _get_serialized_fingerprint_def(self):
    return fingerprint_pb2.FingerprintDef(
        saved_model_checksum=1,
        graph_def_program_hash=2,
        signature_def_hash=3,
        saved_object_graph_hash=4,
        checkpoint_hash=5,
        version=versions_pb2.VersionDef(producer=6)).SerializeToString()

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

  def test_async_checkpoint_add_write_duration(self):
    self.assertEqual(
        self._get_histogram_proto(
            metrics.GetAsyncCheckpointWriteDurations(api_label="foo")).num, 0)

    metrics.AddAsyncCheckpointWriteDuration(api_label="foo", microseconds=20)
    metrics.AddAsyncCheckpointWriteDuration(api_label="foo", microseconds=50)

    self.assertEqual(
        self._get_histogram_proto(
            metrics.GetAsyncCheckpointWriteDurations(api_label="foo")).num, 2)
    self.assertEqual(
        self._get_histogram_proto(
            metrics.GetAsyncCheckpointWriteDurations(api_label="foo")).min, 20)
    self.assertEqual(
        self._get_histogram_proto(
            metrics.GetAsyncCheckpointWriteDurations(api_label="foo")).max, 50)

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

  def test_checkpoint_size(self):
    self.assertEqual(
        metrics.GetCheckpointSize(api_label="baz", filesize=100), 0)
    metrics.RecordCheckpointSize(api_label="baz", filesize=100)
    metrics.RecordCheckpointSize(api_label="baz", filesize=100)
    self.assertEqual(
        metrics.GetCheckpointSize(api_label="baz", filesize=100), 2)

  def test_filesize(self):
    filename = os.path.join(self.get_temp_dir(), "test.txt")
    with open(filename, "w") as file:
      file.write("Hello! \n")
    self.assertEqual(metrics.CalculateFileSize(filename), 0)

  def test_invalid_file(self):
    self.assertEqual(metrics.CalculateFileSize("not_a_file.txt"), -1)

  def test_SM_read_fingerprint(self):
    self.assertEqual(metrics.GetReadFingerprint(), "")
    metrics.SetReadFingerprint(
        fingerprint=self._get_serialized_fingerprint_def())
    read_fingerprint = metrics.GetReadFingerprint()
    self.assertIn('"saved_model_checksum" : 1', read_fingerprint)
    self.assertIn('"graph_def_program_hash" : 2', read_fingerprint)
    self.assertIn('"signature_def_hash" : 3', read_fingerprint)
    self.assertIn('"saved_object_graph_hash" : 4', read_fingerprint)
    self.assertIn('"checkpoint_hash" : 5', read_fingerprint)

  def test_SM_write_fingerprint(self):
    self.assertEqual(metrics.GetWriteFingerprint(), "")
    metrics.SetWriteFingerprint(
        fingerprint=self._get_serialized_fingerprint_def())
    write_fingerprint = metrics.GetWriteFingerprint()
    self.assertIn('"saved_model_checksum" : 1', write_fingerprint)
    self.assertIn('"graph_def_program_hash" : 2', write_fingerprint)
    self.assertIn('"signature_def_hash" : 3', write_fingerprint)
    self.assertIn('"saved_object_graph_hash" : 4', write_fingerprint)
    self.assertIn('"checkpoint_hash" : 5', write_fingerprint)

  def test_SM_read_path(self):
    self.assertEqual(metrics.GetReadPath(), "")
    metrics.SetReadPath(saved_model_path="foo")
    self.assertEqual(metrics.GetReadPath(), "foo")

  def test_SM_write_path(self):
    self.assertEqual(metrics.GetWritePath(), "")
    metrics.SetWritePath(saved_model_path="foo")
    self.assertEqual(metrics.GetWritePath(), "foo")

  def test_SM_read_path_and_singleprint(self):
    self.assertEqual(metrics.GetReadPathAndSingleprint(), ("", ""))
    metrics.SetReadPathAndSingleprint(path="foo", singleprint="bar")
    self.assertEqual(metrics.GetReadPathAndSingleprint(), ("foo", "bar"))

  def test_SM_write_path_and_singleprint(self):
    self.assertEqual(metrics.GetWritePathAndSingleprint(), ("", ""))
    metrics.SetWritePathAndSingleprint(path="foo", singleprint="bar")
    self.assertEqual(metrics.GetWritePathAndSingleprint(), ("foo", "bar"))


if __name__ == "__main__":
  test.main()
