# -*- coding: utf-8 -*-
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
"""Tests for tensorflow.contrib.io."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import zmq

from tensorflow.python.platform import test
from tensorflow.python.util import compat
from tensorflow.contrib import io as io_ops


class IoOpsTest(test.TestCase):
  def testPollZmq(self):
    def _listener():
      context = zmq.Context()
      socket = context.socket(zmq.REP)
      socket.bind("tcp://*:5555")

      message = socket.recv()
      socket.send(b"World")
      self.assertEqual(message, b"Hello")

    # Start a listener thread
    thread = self.checkedThread(_listener)
    thread.start()

    with self.test_session() as sess:
      response = io_ops.poll_zmq("Hello", "tcp://localhost:5555")
      response_val = sess.run(response)
      self.assertEqual(response_val, b"World")

    thread.join()


if __name__ == '__main__':
  test.main()
