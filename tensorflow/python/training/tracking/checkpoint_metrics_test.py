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

from tensorflow.python.eager import context
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.platform import test
from tensorflow.python.training.tracking import util


class CheckpointMetricTests(test.TestCase):

  def _get_write_durations(self, label):
    return util._checkpoint_write_durations.get_cell(label).value()

  def _get_read_durations(self, label):
    return util._checkpoint_read_durations.get_cell(label).value()

  def _get_time_saved(self, label):
    return util._checkpoint_training_time_saved.get_cell(label).value()

  def test_metrics_v2(self):
    label = 'V2'
    prefix = os.path.join(self.get_temp_dir(), 'ckpt')

    with context.eager_mode():
      ckpt = util.Checkpoint(v=variables_lib.Variable(1.))
      self.assertEqual(self._get_time_saved(label), 0.0)
      self.assertEqual(self._get_write_durations(label).num, 0.0)
      for _ in range(3):
        time_saved = self._get_time_saved(label)
        ckpt_path = ckpt.write(file_prefix=prefix)
        self.assertGreater(self._get_time_saved(label), time_saved)

    self.assertEqual(self._get_write_durations(label).num, 3.0)

    self.assertEqual(self._get_read_durations(label).num, 0.0)
    time_saved = self._get_time_saved(label)
    with context.eager_mode():
      ckpt.restore(ckpt_path)
    self.assertEqual(self._get_read_durations(label).num, 1.0)
    # Restoring a checkpoint does not increase training time saved.
    self.assertEqual(self._get_time_saved(label), time_saved)

  def test_metrics_v1(self):
    label = 'V1'
    prefix = os.path.join(self.get_temp_dir(), 'ckpt')

    with self.cached_session():
      ckpt = util.CheckpointV1()
      v = variables_lib.Variable(1.)
      self.evaluate(v.initializer)
      ckpt.v = v
      self.assertEqual(self._get_time_saved(label), 0.0)
      self.assertEqual(self._get_write_durations(label).num, 0.0)
      for _ in range(3):
        time_saved = self._get_time_saved(label)
        ckpt_path = ckpt.write(file_prefix=prefix)
        self.assertGreaterEqual(self._get_time_saved(label), time_saved)

    self.assertEqual(self._get_write_durations(label).num, 3.0)

    self.assertEqual(self._get_read_durations(label).num, 0.0)
    time_saved = self._get_time_saved(label)
    ckpt.restore(ckpt_path)
    self.assertEqual(self._get_read_durations(label).num, 1.0)
    # Restoring a checkpoint does not increase training time saved.
    self.assertEqual(self._get_time_saved(label), time_saved)


if __name__ == '__main__':
  test.main()
