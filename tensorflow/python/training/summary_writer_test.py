# pylint: disable=g-bad-file-header
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
"""Tests for Runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile
import threading
import time

import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.python.summary.summary_iterator import SummaryWriter


class SummaryWriterTest(tf.test.TestCase):
  """SummaryWriter tests."""

  def testLoggerThreadExit(self):
    writer = SummaryWriter(tempfile.mkdtemp(), graph=ops.get_default_graph())
    previous_thread_count = threading.active_count()
    writer.close()
    current_thread_count = threading.active_count()
    self.assertTrue(previous_thread_count == current_thread_count + 1)

  def testFlushEvents(self):
    """Test that all published events are flushed after close the
       `SummaryWriter`.
    """

    writer = SummaryWriter(tempfile.mkdtemp(), graph=ops.get_default_graph())

    # Create a producer thread to keep adding events.
    def put_events():
      with self.assertRaisesRegexp(Exception, "Event queue is closed."):
        for i in xrange(100000):
          event = tf.Event(wall_time=int(time.time()), step=i)
          writer.add_event(event)

    producer = threading.Thread(target=put_events)
    producer.start()

    # Ensure there are enough events be flushed because close the writer,
    # then we could continue to compare the number of published events and
    # flushed events.
    while writer._worker._success_events_counter <= 0:
      time.sleep(1)
    writer.close()

    producer.join()

    # Verify all published events are flushed after close the writer.
    self.assertEqual(writer._worker._success_events_counter, writer._add_events_counter)

if __name__ == '__main__':
  tf.test.main()
