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
"""Tests for profiler_hook."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import shutil
import tempfile

from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.hooks.python.training import ProfilerHook
from tensorflow.python.framework import ops
from tensorflow.python.ops import state_ops
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.training import monitored_session


class ProfilerHookTest(test.TestCase):

  def setUp(self):
    super(ProfilerHookTest, self).setUp()
    self.output_dir = tempfile.mkdtemp()
    self.graph = ops.Graph()
    self.filepattern = os.path.join(self.output_dir, "timeline-*.json")
    with self.graph.as_default():
      self.global_step = variables.get_or_create_global_step()
      self.train_op = state_ops.assign_add(self.global_step, 1)

  def tearDown(self):
    super(ProfilerHookTest, self).tearDown()
    shutil.rmtree(self.output_dir, ignore_errors=True)

  def _count_timeline_files(self):
    return len(gfile.Glob(self.filepattern))

  def test_raise_in_both_secs_and_steps(self):
    with self.assertRaises(ValueError):
      ProfilerHook(save_secs=10, save_steps=20)

  def test_raise_in_none_secs_and_steps(self):
    with self.assertRaises(ValueError):
      ProfilerHook(save_secs=None, save_steps=None)

  def test_save_secs_saves_in_first_step(self):
    with self.graph.as_default():
      hook = ProfilerHook(save_secs=2, output_dir=self.output_dir)
      with monitored_session.SingularMonitoredSession(hooks=[hook]) as sess:
        sess.run(self.train_op)
        self.assertEqual(1, self._count_timeline_files())

  @test.mock.patch('time.time')
  def test_save_secs_saves_periodically(self, mock_time):
    # Pick a fixed start time.
    current_time = 1484863632.320497

    with self.graph.as_default():
      mock_time.return_value = current_time
      hook = ProfilerHook(save_secs=2, output_dir=self.output_dir)
      with monitored_session.SingularMonitoredSession(hooks=[hook]) as sess:
        sess.run(self.train_op)  # Saved.
        self.assertEqual(1, self._count_timeline_files())
        sess.run(self.train_op)  # Not saved.
        self.assertEqual(1, self._count_timeline_files())
        # Simulate 2.5 seconds of sleep.
        mock_time.return_value = current_time + 2.5
        sess.run(self.train_op)  # Saved.

        # Pretend some small amount of time has passed.
        mock_time.return_value = current_time + 0.1
        sess.run(self.train_op)  # Not saved.
        # Edge test just before we should save the timeline.
        mock_time.return_value = current_time + 1.9
        sess.run(self.train_op)  # Not saved.
        self.assertEqual(2, self._count_timeline_files())

        mock_time.return_value = current_time + 4.5
        sess.run(self.train_op)  # Saved.
        self.assertEqual(3, self._count_timeline_files())

  def test_save_steps_saves_in_first_step(self):
    with self.graph.as_default():
      hook = ProfilerHook(save_secs=2, output_dir=self.output_dir)
      with monitored_session.SingularMonitoredSession(hooks=[hook]) as sess:
        sess.run(self.train_op)  # Saved.
        sess.run(self.train_op)  # Not saved.
        self.assertEqual(1, self._count_timeline_files())

  def test_save_steps_saves_periodically(self):
    with self.graph.as_default():
      hook = ProfilerHook(save_steps=2, output_dir=self.output_dir)
      with monitored_session.SingularMonitoredSession(hooks=[hook]) as sess:
        self.assertEqual(0, self._count_timeline_files())
        sess.run(self.train_op)  # Saved.
        self.assertEqual(1, self._count_timeline_files())
        sess.run(self.train_op)  # Not saved.
        self.assertEqual(1, self._count_timeline_files())
        sess.run(self.train_op)  # Saved.
        self.assertEqual(2, self._count_timeline_files())
        sess.run(self.train_op)  # Not saved.
        self.assertEqual(2, self._count_timeline_files())
        sess.run(self.train_op)  # Saved.
        self.assertEqual(3, self._count_timeline_files())


if __name__ == '__main__':
  test.main()
