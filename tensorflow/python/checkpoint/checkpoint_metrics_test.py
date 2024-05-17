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
"""Tests for checking the checkpoint reading and writing metrics."""

import os
import time

from tensorflow.core.framework import summary_pb2
from tensorflow.python.checkpoint import checkpoint as util
from tensorflow.python.eager import context
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.platform import test
from tensorflow.python.saved_model.pywrap_saved_model import metrics


class CheckpointMetricTests(test.TestCase):

  def _get_write_histogram_proto(self, api_label):
    proto_bytes = metrics.GetCheckpointWriteDurations(api_label=api_label)
    histogram_proto = summary_pb2.HistogramProto()
    histogram_proto.ParseFromString(proto_bytes)
    return histogram_proto

  def _get_read_histogram_proto(self, api_label):
    proto_bytes = metrics.GetCheckpointReadDurations(api_label=api_label)
    histogram_proto = summary_pb2.HistogramProto()
    histogram_proto.ParseFromString(proto_bytes)
    return histogram_proto

  def _get_time_saved(self, api_label):
    return metrics.GetTrainingTimeSaved(api_label=api_label)

  def _get_checkpoint_size(self, api_label, filesize):
    return metrics.GetCheckpointSize(api_label=api_label, filesize=filesize)

  def test_metrics_v2(self):
    api_label = util._CHECKPOINT_V2
    prefix = os.path.join(self.get_temp_dir(), 'ckpt')

    with context.eager_mode():
      ckpt = util.Checkpoint(v=variables_lib.Variable(1.))
      self.assertEqual(self._get_time_saved(api_label), 0.0)
      self.assertEqual(self._get_write_histogram_proto(api_label).num, 0.0)

      for i in range(3):
        time_saved = self._get_time_saved(api_label)
        time.sleep(1)
        ckpt_path = ckpt.write(file_prefix=prefix)
        filesize = util._get_checkpoint_size(ckpt_path)
        self.assertEqual(self._get_checkpoint_size(api_label, filesize), i + 1)
        self.assertGreater(self._get_time_saved(api_label), time_saved)

    self.assertEqual(self._get_write_histogram_proto(api_label).num, 3.0)
    self.assertEqual(self._get_read_histogram_proto(api_label).num, 0.0)

    time_saved = self._get_time_saved(api_label)
    with context.eager_mode():
      ckpt.restore(ckpt_path)
    self.assertEqual(self._get_read_histogram_proto(api_label).num, 1.0)
    # Restoring a checkpoint in the same "job" does not increase training time
    # saved.
    self.assertEqual(self._get_time_saved(api_label), time_saved)

  def test_metrics_v1(self):
    api_label = util._CHECKPOINT_V1
    prefix = os.path.join(self.get_temp_dir(), 'ckpt')

    with self.cached_session():
      ckpt = util.CheckpointV1()
      v = variables_lib.Variable(1.)
      self.evaluate(v.initializer)
      ckpt.v = v
      self.assertEqual(self._get_time_saved(api_label), 0.0)
      self.assertEqual(self._get_write_histogram_proto(api_label).num, 0.0)

      for i in range(3):
        time_saved = self._get_time_saved(api_label)
        time.sleep(1)
        ckpt_path = ckpt.write(file_prefix=prefix)
        filesize = util._get_checkpoint_size(ckpt_path)
        self.assertEqual(self._get_checkpoint_size(api_label, filesize), i + 1)
        self.assertGreater(self._get_time_saved(api_label), time_saved)

    self.assertEqual(self._get_write_histogram_proto(api_label).num, 3.0)

    self.assertEqual(self._get_read_histogram_proto(api_label).num, 0.0)
    time_saved = self._get_time_saved(api_label)
    ckpt.restore(ckpt_path)
    self.assertEqual(self._get_read_histogram_proto(api_label).num, 1.0)
    # Restoring a checkpoint in the same "job" does not increase training time
    # saved.
    self.assertEqual(self._get_time_saved(api_label), time_saved)


if __name__ == '__main__':
  test.main()
