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
"""Unit tests for local command-line-interface debug wrapper session."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import tempfile

from tensorflow.python.client import session
from tensorflow.python.debug.cli import debugger_cli_common
from tensorflow.python.debug.wrappers import local_cli_wrapper
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest


class LocalCLIDebuggerWrapperSessionForTest(
    local_cli_wrapper.LocalCLIDebugWrapperSession):
  """Subclasses the wrapper class for testing.

  Overrides its CLI-related methods for headless testing environments.
  Inserts observer variables for assertions.
  """

  def __init__(self, command_args_sequence, sess, dump_root=None):
    """Constructor of the for-test subclass.

    Args:
      command_args_sequence: (list of list of str) A list of arguments for the
        "run" command.
      sess: See the doc string of LocalCLIDebugWrapperSession.__init__.
      dump_root: See the doc string of LocalCLIDebugWrapperSession.__init__.
    """

    local_cli_wrapper.LocalCLIDebugWrapperSession.__init__(
        self, sess, dump_root=dump_root, log_usage=False)

    self._command_args_sequence = command_args_sequence
    self._response_pointer = 0

    # Observer variables.
    self.observers = {
        "debug_dumps": [],
        "tf_errors": [],
        "run_start_cli_run_numbers": [],
        "run_end_cli_run_numbers": [],
    }

  def _prep_cli_for_run_start(self):
    pass

  def _prep_cli_for_run_end(self, debug_dump, tf_error, passed_filter):
    self.observers["debug_dumps"].append(debug_dump)
    self.observers["tf_errors"].append(tf_error)

  def _launch_cli(self, is_run_start=False):
    if is_run_start:
      self.observers["run_start_cli_run_numbers"].append(self._run_call_count)
    else:
      self.observers["run_end_cli_run_numbers"].append(self._run_call_count)

    command_args = self._command_args_sequence[self._response_pointer]
    self._response_pointer += 1

    try:
      self._run_handler(command_args)
    except debugger_cli_common.CommandLineExit as e:
      response = e.exit_token

    return response


class LocalCLIDebugWrapperSessionTest(test_util.TensorFlowTestCase):

  def setUp(self):
    self._tmp_dir = tempfile.mktemp()

    self.v = variables.Variable(10.0, name="v")
    self.delta = constant_op.constant(1.0, name="delta")
    self.inc_v = state_ops.assign_add(self.v, self.delta, name="inc_v")

    self.ph = array_ops.placeholder(dtypes.float32, name="ph")
    self.xph = array_ops.transpose(self.ph, name="xph")
    self.m = constant_op.constant(
        [[0.0, 1.0, 2.0], [-4.0, -1.0, 0.0]], dtype=dtypes.float32, name="m")
    self.y = math_ops.matmul(self.m, self.xph, name="y")

    self.sess = session.Session()

    # Initialize variable.
    self.sess.run(self.v.initializer)

  def tearDown(self):
    ops.reset_default_graph()
    if os.path.isdir(self._tmp_dir):
      shutil.rmtree(self._tmp_dir)

  def testConstructWrapper(self):
    local_cli_wrapper.LocalCLIDebugWrapperSession(
        session.Session(), log_usage=False)

  def testConstructWrapperWithExistingEmptyDumpRoot(self):
    os.mkdir(self._tmp_dir)
    self.assertTrue(os.path.isdir(self._tmp_dir))

    local_cli_wrapper.LocalCLIDebugWrapperSession(
        session.Session(), dump_root=self._tmp_dir, log_usage=False)

  def testConstructWrapperWithExistingNonEmptyDumpRoot(self):
    os.mkdir(self._tmp_dir)
    dir_path = os.path.join(self._tmp_dir, "foo")
    os.mkdir(dir_path)
    self.assertTrue(os.path.isdir(dir_path))

    with self.assertRaisesRegexp(
        ValueError, "dump_root path points to a non-empty directory"):
      local_cli_wrapper.LocalCLIDebugWrapperSession(
          session.Session(), dump_root=self._tmp_dir, log_usage=False)

  def testConstructWrapperWithExistingFileDumpRoot(self):
    os.mkdir(self._tmp_dir)
    file_path = os.path.join(self._tmp_dir, "foo")
    open(file_path, "a").close()  # Create the file
    self.assertTrue(os.path.isfile(file_path))
    with self.assertRaisesRegexp(ValueError, "dump_root path points to a file"):
      local_cli_wrapper.LocalCLIDebugWrapperSession(
          session.Session(), dump_root=file_path, log_usage=False)

  def testRunsUnderDebugMode(self):
    # Test command sequence: run; run; run;
    wrapped_sess = LocalCLIDebuggerWrapperSessionForTest(
        [[], [], []], self.sess, dump_root=self._tmp_dir)

    # run under debug mode twice.
    wrapped_sess.run(self.inc_v)
    wrapped_sess.run(self.inc_v)

    # Verify that the assign_add op did take effect.
    self.assertAllClose(12.0, self.sess.run(self.v))

    # Assert correct run call numbers for which the CLI has been launched at
    # run-start and run-end.
    self.assertEqual([1], wrapped_sess.observers["run_start_cli_run_numbers"])
    self.assertEqual([1, 2], wrapped_sess.observers["run_end_cli_run_numbers"])

    # Verify that the dumps have been generated and picked up during run-end.
    self.assertEqual(2, len(wrapped_sess.observers["debug_dumps"]))

    # Verify that the TensorFlow runtime errors are picked up and in this case,
    # they should be both None.
    self.assertEqual([None, None], wrapped_sess.observers["tf_errors"])

  def testRunsUnderNonDebugMode(self):
    # Test command sequence: run -n; run -n; run -n;
    wrapped_sess = LocalCLIDebuggerWrapperSessionForTest(
        [["-n"], ["-n"], ["-n"]], self.sess, dump_root=self._tmp_dir)

    # run three times.
    wrapped_sess.run(self.inc_v)
    wrapped_sess.run(self.inc_v)
    wrapped_sess.run(self.inc_v)

    self.assertAllClose(13.0, self.sess.run(self.v))

    self.assertEqual([1, 2, 3],
                     wrapped_sess.observers["run_start_cli_run_numbers"])
    self.assertEqual([], wrapped_sess.observers["run_end_cli_run_numbers"])

  def testRunsUnderNonDebugThenDebugMode(self):
    # Test command sequence: run -n; run -n; run; run;
    # Do two NON_DEBUG_RUNs, followed by DEBUG_RUNs.
    wrapped_sess = LocalCLIDebuggerWrapperSessionForTest(
        [["-n"], ["-n"], [], []], self.sess, dump_root=self._tmp_dir)

    # run three times.
    wrapped_sess.run(self.inc_v)
    wrapped_sess.run(self.inc_v)
    wrapped_sess.run(self.inc_v)

    self.assertAllClose(13.0, self.sess.run(self.v))

    self.assertEqual([1, 2, 3],
                     wrapped_sess.observers["run_start_cli_run_numbers"])

    # Here, the CLI should have been launched only under the third run,
    # because the first and second runs are NON_DEBUG.
    self.assertEqual([3], wrapped_sess.observers["run_end_cli_run_numbers"])
    self.assertEqual(1, len(wrapped_sess.observers["debug_dumps"]))
    self.assertEqual([None], wrapped_sess.observers["tf_errors"])

  def testRunMultipleTimesWithinLimit(self):
    # Test command sequence: run -t 3; run;
    wrapped_sess = LocalCLIDebuggerWrapperSessionForTest(
        [["-t", "3"], []], self.sess, dump_root=self._tmp_dir)

    # run three times.
    wrapped_sess.run(self.inc_v)
    wrapped_sess.run(self.inc_v)
    wrapped_sess.run(self.inc_v)

    self.assertAllClose(13.0, self.sess.run(self.v))

    self.assertEqual([1], wrapped_sess.observers["run_start_cli_run_numbers"])
    self.assertEqual([3], wrapped_sess.observers["run_end_cli_run_numbers"])
    self.assertEqual(1, len(wrapped_sess.observers["debug_dumps"]))
    self.assertEqual([None], wrapped_sess.observers["tf_errors"])

  def testRunMultipleTimesOverLimit(self):
    # Test command sequence: run -t 3;
    wrapped_sess = LocalCLIDebuggerWrapperSessionForTest(
        [["-t", "3"]], self.sess, dump_root=self._tmp_dir)

    # run twice, which is less than the number of times specified by the
    # command.
    wrapped_sess.run(self.inc_v)
    wrapped_sess.run(self.inc_v)

    self.assertAllClose(12.0, self.sess.run(self.v))

    self.assertEqual([1], wrapped_sess.observers["run_start_cli_run_numbers"])
    self.assertEqual([], wrapped_sess.observers["run_end_cli_run_numbers"])
    self.assertEqual(0, len(wrapped_sess.observers["debug_dumps"]))
    self.assertEqual([], wrapped_sess.observers["tf_errors"])

  def testRunMixingDebugModeAndMultpleTimes(self):
    # Test command sequence: run -n; run -t 2; run; run;
    wrapped_sess = LocalCLIDebuggerWrapperSessionForTest(
        [["-n"], ["-t", "2"], [], []], self.sess, dump_root=self._tmp_dir)

    # run four times.
    wrapped_sess.run(self.inc_v)
    wrapped_sess.run(self.inc_v)
    wrapped_sess.run(self.inc_v)
    wrapped_sess.run(self.inc_v)

    self.assertAllClose(14.0, self.sess.run(self.v))

    self.assertEqual([1, 2],
                     wrapped_sess.observers["run_start_cli_run_numbers"])
    self.assertEqual([3, 4], wrapped_sess.observers["run_end_cli_run_numbers"])
    self.assertEqual(2, len(wrapped_sess.observers["debug_dumps"]))
    self.assertEqual([None, None], wrapped_sess.observers["tf_errors"])

  def testRuntimeErrorShouldBeCaught(self):
    wrapped_sess = LocalCLIDebuggerWrapperSessionForTest(
        [[], []], self.sess, dump_root=self._tmp_dir)

    # Do a run that should lead to an TensorFlow runtime error.
    wrapped_sess.run(self.y, feed_dict={self.ph: [[0.0], [1.0], [2.0]]})

    self.assertEqual([1], wrapped_sess.observers["run_start_cli_run_numbers"])
    self.assertEqual([1], wrapped_sess.observers["run_end_cli_run_numbers"])
    self.assertEqual(1, len(wrapped_sess.observers["debug_dumps"]))

    # Verify that the runtime error is caught by the wrapped session properly.
    self.assertEqual(1, len(wrapped_sess.observers["tf_errors"]))
    tf_error = wrapped_sess.observers["tf_errors"][0]
    self.assertEqual("y", tf_error.op.name)

  def testRunTillFilterPassesShouldLaunchCLIAtCorrectRun(self):
    # Test command sequence:
    #   run -f greater_than_twelve; run -f greater_than_twelve; run;
    wrapped_sess = LocalCLIDebuggerWrapperSessionForTest(
        [["-f", "v_greater_than_twelve"], ["-f", "v_greater_than_twelve"], []],
        self.sess,
        dump_root=self._tmp_dir)

    def v_greater_than_twelve(datum, tensor):
      return datum.node_name == "v" and tensor > 12.0

    wrapped_sess.add_tensor_filter("v_greater_than_twelve",
                                   v_greater_than_twelve)

    # run five times.
    wrapped_sess.run(self.inc_v)
    wrapped_sess.run(self.inc_v)
    wrapped_sess.run(self.inc_v)
    wrapped_sess.run(self.inc_v)
    wrapped_sess.run(self.inc_v)

    self.assertAllClose(15.0, self.sess.run(self.v))

    self.assertEqual([1], wrapped_sess.observers["run_start_cli_run_numbers"])

    # run-end CLI should NOT have been launched for run #2 and #3, because only
    # starting from run #4 v becomes greater than 12.0.
    self.assertEqual([4, 5], wrapped_sess.observers["run_end_cli_run_numbers"])

    self.assertEqual(2, len(wrapped_sess.observers["debug_dumps"]))
    self.assertEqual([None, None], wrapped_sess.observers["tf_errors"])


if __name__ == "__main__":
  googletest.main()
