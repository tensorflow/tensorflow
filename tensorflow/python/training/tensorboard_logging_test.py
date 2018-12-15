# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.python.framework.tensorboard_logging."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
import shutil
import tempfile
import time

from tensorflow.core.util import event_pb2
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import summary_iterator
from tensorflow.python.summary.writer import writer
from tensorflow.python.training import tensorboard_logging


@test_util.run_v1_only("b/120545219")
class EventLoggingTest(test.TestCase):

  def setUp(self):
    self._work_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
    self._sw = writer.FileWriter(self._work_dir)
    tensorboard_logging.set_summary_writer(self._sw)
    self.addCleanup(shutil.rmtree, self._work_dir)

    # Stop the clock to avoid test flakiness.
    now = time.time()
    time._real_time = time.time
    time.time = lambda: now

    # Mock out logging calls so we can verify that the right number of messages
    # get logged.
    self.logged_message_count = 0
    self._actual_log = logging.log

    def mockLog(*args, **kwargs):
      self.logged_message_count += 1
      self._actual_log(*args, **kwargs)

    logging.log = mockLog

  def tearDown(self):
    time.time = time._real_time
    logging.log = self._actual_log

  def assertLoggedMessagesAre(self, expected_messages):
    self._sw.close()
    event_paths = glob.glob(os.path.join(self._work_dir, "event*"))
    # If the tests runs multiple time in the same directory we can have
    # more than one matching event file.  We only want to read the last one.
    self.assertTrue(event_paths)
    event_reader = summary_iterator.summary_iterator(event_paths[-1])
    # Skip over the version event.
    next(event_reader)

    for level, message in expected_messages:
      event = next(event_reader)
      self.assertEqual(event.wall_time, time.time())
      self.assertEqual(event.log_message.level, level)
      self.assertEqual(event.log_message.message, message)

  def testBasic(self):
    tensorboard_logging.set_summary_writer(self._sw)
    tensorboard_logging.error("oh no!")
    tensorboard_logging.error("for%s", "mat")

    self.assertLoggedMessagesAre([(event_pb2.LogMessage.ERROR, "oh no!"),
                                  (event_pb2.LogMessage.ERROR, "format")])
    self.assertEqual(2, self.logged_message_count)

  @test_util.run_v1_only("b/120545219")
  def testVerbosity(self):
    tensorboard_logging.set_summary_writer(self._sw)
    tensorboard_logging.set_verbosity(tensorboard_logging.ERROR)
    tensorboard_logging.warn("warn")
    tensorboard_logging.error("error")
    tensorboard_logging.set_verbosity(tensorboard_logging.DEBUG)
    tensorboard_logging.debug("debug")

    self.assertLoggedMessagesAre([(event_pb2.LogMessage.ERROR, "error"),
                                  (event_pb2.LogMessage.DEBUGGING, "debug")])
    # All message should be logged because tensorboard_logging verbosity doesn't
    # affect logging verbosity.
    self.assertEqual(3, self.logged_message_count)

  def testBadVerbosity(self):
    with self.assertRaises(ValueError):
      tensorboard_logging.set_verbosity("failure")

    with self.assertRaises(ValueError):
      tensorboard_logging.log("bad", "dead")

  def testNoSummaryWriter(self):
    """Test that logging without a SummaryWriter succeeds."""
    tensorboard_logging.set_summary_writer(None)
    tensorboard_logging.warn("this should work")
    self.assertEqual(1, self.logged_message_count)

  @test_util.run_v1_only("b/120545219")
  def testSummaryWriterFailsAfterClear(self):
    tensorboard_logging._clear_summary_writer()
    with self.assertRaises(RuntimeError):
      tensorboard_logging.log(tensorboard_logging.ERROR, "failure")


if __name__ == "__main__":
  test.main()
