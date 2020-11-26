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
"""Tests for basic_loops.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil

from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.platform import test
from tensorflow.python.training import basic_loops
from tensorflow.python.training import supervisor


def _test_dir(test_name):
  test_dir = os.path.join(test.get_temp_dir(), test_name)
  if os.path.exists(test_dir):
    shutil.rmtree(test_dir)
  return test_dir


class BasicTrainLoopTest(test.TestCase):

  def testBasicTrainLoop(self):
    logdir = _test_dir("basic_train_loop")
    # Counts the number of calls.
    num_calls = [0]

    def train_fn(unused_sess, sv, y, a):
      num_calls[0] += 1
      self.assertEqual("y", y)
      self.assertEqual("A", a)
      if num_calls[0] == 3:
        sv.request_stop()

    with ops.Graph().as_default():
      sv = supervisor.Supervisor(logdir=logdir)
      basic_loops.basic_train_loop(
          sv, train_fn, args=(sv, "y"), kwargs={"a": "A"})
      self.assertEqual(3, num_calls[0])

  def testBasicTrainLoopExceptionAborts(self):
    logdir = _test_dir("basic_train_loop_exception_aborts")

    def train_fn(unused_sess):
      train_fn.counter += 1
      if train_fn.counter == 3:
        raise RuntimeError("Failed")

    # Function attribute use to count the number of calls.
    train_fn.counter = 0

    with ops.Graph().as_default():
      sv = supervisor.Supervisor(logdir=logdir)
      with self.assertRaisesRegex(RuntimeError, "Failed"):
        basic_loops.basic_train_loop(sv, train_fn)

  def testBasicTrainLoopRetryOnAborted(self):
    logdir = _test_dir("basic_train_loop_exception_aborts")

    class AbortAndRetry(object):

      def __init__(self):
        self.num_calls = 0
        self.retries_left = 2

      def train_fn(self, unused_sess):
        self.num_calls += 1
        if self.num_calls % 3 == 2:
          self.retries_left -= 1
        if self.retries_left > 0:
          raise errors_impl.AbortedError(None, None, "Aborted here")
        else:
          raise RuntimeError("Failed Again")

    with ops.Graph().as_default():
      sv = supervisor.Supervisor(logdir=logdir)
      aar = AbortAndRetry()
      with self.assertRaisesRegex(RuntimeError, "Failed Again"):
        basic_loops.basic_train_loop(sv, aar.train_fn)
      self.assertEqual(0, aar.retries_left)


if __name__ == "__main__":
  test.main()
