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


if __name__ == '__main__':
  tf.test.main()
