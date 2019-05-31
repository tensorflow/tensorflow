# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================
"""Test to make sure stack trace is generated in case of test failures."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import signal
import subprocess
import sys

from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging


# FLAGS defined at the bottom:
# child (bool) set to true if we are running in the child process.
FLAGS = None

_CHILD_FLAG_HELP = 'Boolean. Set to true if this is the child process.'


class StacktraceHandlerTest(test.TestCase):

  def testChildProcessKillsItself(self):
    if FLAGS.child:
      os.kill(os.getpid(), signal.SIGABRT)

  def testGeneratesStacktrace(self):
    if FLAGS.child:
      return

    # Subprocess sys.argv[0] with --child=True
    if sys.executable:
      child_process = subprocess.Popen(
          [sys.executable, sys.argv[0], '--child=True'], cwd=os.getcwd(),
          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    else:
      child_process = subprocess.Popen(
          [sys.argv[0], '--child=True'], cwd=os.getcwd(),
          stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Capture its output. capture both stdout and stderr and append them.
    # We are not worried about timing or order of messages in this test.
    child_stdout, child_stderr = child_process.communicate()
    child_output = child_stdout + child_stderr

    # Make sure the child process is dead before we proceed.
    child_process.wait()

    logging.info('Output from the child process:')
    logging.info(child_output)

    # Verify a stack trace is printed.
    self.assertIn(b'PyEval_EvalFrame', child_output)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--child', type=bool, default=False, help=_CHILD_FLAG_HELP)
  FLAGS, unparsed = parser.parse_known_args()

  # Now update argv, so that unittest library does not get confused.
  sys.argv = [sys.argv[0]] + unparsed
  test.main()
