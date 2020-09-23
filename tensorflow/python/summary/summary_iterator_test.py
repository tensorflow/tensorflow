# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.python.summary.summary_iterator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os.path

from tensorflow.core.util import event_pb2
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tensorflow.python.summary import summary_iterator
from tensorflow.python.summary.writer import writer


class SummaryIteratorTestCase(test.TestCase):

  @test_util.run_deprecated_v1
  def testSummaryIteratorEventsAddedAfterEndOfFile(self):
    test_dir = os.path.join(self.get_temp_dir(), "events")
    with writer.FileWriter(test_dir) as w:
      session_log_start = event_pb2.SessionLog.START
      w.add_session_log(event_pb2.SessionLog(status=session_log_start), 1)
      w.flush()
      path = glob.glob(os.path.join(test_dir, "event*"))[0]
      rr = summary_iterator.summary_iterator(path)
      # The first event should list the file_version.
      ev = next(rr)
      self.assertEqual("brain.Event:2", ev.file_version)
      # The next event should be the START message.
      ev = next(rr)
      self.assertEqual(1, ev.step)
      self.assertEqual(session_log_start, ev.session_log.status)
      # Reached EOF.
      self.assertRaises(StopIteration, lambda: next(rr))
      w.add_session_log(event_pb2.SessionLog(status=session_log_start), 2)
      w.flush()
      # The new event is read, after previously seeing EOF.
      ev = next(rr)
      self.assertEqual(2, ev.step)
      self.assertEqual(session_log_start, ev.session_log.status)
      # Get EOF again.
      self.assertRaises(StopIteration, lambda: next(rr))

if __name__ == "__main__":
  test.main()
