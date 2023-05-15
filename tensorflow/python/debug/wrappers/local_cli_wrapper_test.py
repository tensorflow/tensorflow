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
import os
import tempfile

import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import session
from tensorflow.python.debug.cli import cli_config
from tensorflow.python.debug.cli import cli_shared
from tensorflow.python.debug.cli import debugger_cli_common
from tensorflow.python.debug.cli import ui_factory
from tensorflow.python.debug.wrappers import local_cli_wrapper
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
# Import resource_variable_ops for the variables-to-tensor implicit conversion.
from tensorflow.python.ops import resource_variable_ops  # pylint: disable=unused-import
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.training import monitored_session
from tensorflow.python.training import session_run_hook


class LocalCLIDebuggerWrapperSessionForTest(
    local_cli_wrapper.LocalCLIDebugWrapperSession):
  """Subclasses the wrapper class for testing.

  Overrides its CLI-related methods for headless testing environments.
  Inserts observer variables for assertions.
  """

  def __init__(self,
               command_sequence,
               sess,
               dump_root=None):
    """Constructor of the for-test subclass.

    Args:
      command_sequence: (list of list of str) A list of command arguments,
        including the command prefix, each element of the list is such as:
        ["run", "-n"],
        ["print_feed", "input:0"].
      sess: See the doc string of LocalCLIDebugWrapperSession.__init__.
      dump_root: See the doc string of LocalCLIDebugWrapperSession.__init__.
    """

    local_cli_wrapper.LocalCLIDebugWrapperSession.__init__(
        self, sess, dump_root=dump_root)

    self._command_sequence = command_sequence
    self._command_pointer = 0

    # Observer variables.
    self.observers = {
        "debug_dumps": [],
        "tf_errors": [],
        "run_start_cli_run_numbers": [],
        "run_end_cli_run_numbers": [],
        "print_feed_responses": [],
        "profiler_py_graphs": [],
        "profiler_run_metadata": [],
    }

  def _prep_cli_for_run_start(self):
    pass

  def _prep_debug_cli_for_run_end(self,
                                  debug_dump,
                                  tf_error,
                                  passed_filter,
                                  passed_filter_exclude_op_names):
    self.observers["debug_dumps"].append(debug_dump)
    self.observers["tf_errors"].append(tf_error)

  def _prep_profile_cli_for_run_end(self, py_graph, run_metadata):
    self.observers["profiler_py_graphs"].append(py_graph)
    self.observers["profiler_run_metadata"].append(run_metadata)

  def _launch_cli(self):
    if self._is_run_start:
      self.observers["run_start_cli_run_numbers"].append(self._run_call_count)
    else:
      self.observers["run_end_cli_run_numbers"].append(self._run_call_count)

    readline_cli = ui_factory.get_ui(
        "readline",
        config=cli_config.CLIConfig(
            config_file_path=os.path.join(tempfile.mkdtemp(), ".tfdbg_config")))
    self._register_this_run_info(readline_cli)

    while self._command_pointer < len(self._command_sequence):
      command = self._command_sequence[self._command_pointer]
      self._command_pointer += 1

      try:
        if command[0] == "run":
          self._run_handler(command[1:])
        elif command[0] == "print_feed":
          self.observers["print_feed_responses"].append(
              self._print_feed_handler(command[1:]))
        else:
          raise ValueError("Unrecognized command prefix: %s" % command[0])
      except debugger_cli_common.CommandLineExit as e:
        return e.exit_token


@test_util.run_v1_only("b/120545219")
class LocalCLIDebugWrapperSessionTest(test_util.TensorFlowTestCase):

  def setUp(self):
    self._tmp_dir = tempfile.mkdtemp()

    self.v = variable_v1.VariableV1(10.0, name="v")
    self.w = variable_v1.VariableV1(21.0, name="w")
    self.delta = constant_op.constant(1.0, name="delta")
    self.inc_v = state_ops.assign_add(self.v, self.delta, name="inc_v")

    self.w_int = control_flow_ops.with_dependencies(
        [self.inc_v],
        math_ops.cast(self.w, dtypes.int32, name="w_int_inner"),
        name="w_int_outer")

    self.ph = array_ops.placeholder(dtypes.float32, name="ph")
    self.xph = array_ops.transpose(self.ph, name="xph")
    self.m = constant_op.constant(
        [[0.0, 1.0, 2.0], [-4.0, -1.0, 0.0]], dtype=dtypes.float32, name="m")
    self.y = math_ops.matmul(self.m, self.xph, name="y")

    self.sparse_ph = array_ops.sparse_placeholder(
        dtypes.float32, shape=([5, 5]), name="sparse_placeholder")
    self.sparse_add = sparse_ops.sparse_add(self.sparse_ph, self.sparse_ph)

    rewriter_config = rewriter_config_pb2.RewriterConfig(
        disable_model_pruning=True,
        arithmetic_optimization=rewriter_config_pb2.RewriterConfig.OFF,
        dependency_optimization=rewriter_config_pb2.RewriterConfig.OFF)
    graph_options = config_pb2.GraphOptions(rewrite_options=rewriter_config)
    config_proto = config_pb2.ConfigProto(graph_options=graph_options)
    self.sess = session.Session(config=config_proto)

    # Initialize variable.
    self.sess.run(variables.global_variables_initializer())

  def tearDown(self):
    ops.reset_default_graph()
    if os.path.isdir(self._tmp_dir):
      file_io.delete_recursively(self._tmp_dir)

  def testConstructWrapper(self):
    local_cli_wrapper.LocalCLIDebugWrapperSession(session.Session())

  def testConstructWrapperWithExistingNonEmptyDumpRoot(self):
    dir_path = os.path.join(self._tmp_dir, "foo")
    os.mkdir(dir_path)
    self.assertTrue(os.path.isdir(dir_path))

    with self.assertRaisesRegex(
        ValueError, "dump_root path points to a non-empty directory"):
      local_cli_wrapper.LocalCLIDebugWrapperSession(
          session.Session(), dump_root=self._tmp_dir)

  def testConstructWrapperWithExistingFileDumpRoot(self):
    file_path = os.path.join(self._tmp_dir, "foo")
    open(file_path, "a").close()  # Create the file
    self.assertTrue(os.path.isfile(file_path))
    with self.assertRaisesRegex(ValueError, "dump_root path points to a file"):
      local_cli_wrapper.LocalCLIDebugWrapperSession(
          session.Session(), dump_root=file_path)

  def testRunsUnderDebugMode(self):
    wrapped_sess = LocalCLIDebuggerWrapperSessionForTest(
        [["run"], ["run"], ["run"]], self.sess, dump_root=self._tmp_dir)

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

  def testRunsWithEmptyStringDumpRootWorks(self):
    wrapped_sess = LocalCLIDebuggerWrapperSessionForTest(
        [["run"], ["run"]], self.sess, dump_root="")

    # run under debug mode.
    wrapped_sess.run(self.inc_v)

    self.assertAllClose(11.0, self.sess.run(self.v))

  def testRunInfoOutputAtRunEndIsCorrect(self):
    wrapped_sess = LocalCLIDebuggerWrapperSessionForTest(
        [["run"], ["run"], ["run"]], self.sess, dump_root=self._tmp_dir)

    wrapped_sess.run(self.inc_v)
    run_info_output = wrapped_sess._run_info_handler([])

    tfdbg_logo = cli_shared.get_tfdbg_logo()

    # The run_info output in the first run() call should contain the tfdbg logo.
    self.assertEqual(tfdbg_logo.lines,
                     run_info_output.lines[:len(tfdbg_logo.lines)])
    menu = run_info_output.annotations[debugger_cli_common.MAIN_MENU_KEY]
    self.assertIn("list_tensors", menu.captions())

    wrapped_sess.run(self.inc_v)
    run_info_output = wrapped_sess._run_info_handler([])

    # The run_info output in the second run() call should NOT contain the logo.
    self.assertNotEqual(tfdbg_logo.lines,
                        run_info_output.lines[:len(tfdbg_logo.lines)])
    menu = run_info_output.annotations[debugger_cli_common.MAIN_MENU_KEY]
    self.assertIn("list_tensors", menu.captions())

  def testRunsUnderNonDebugMode(self):
    wrapped_sess = LocalCLIDebuggerWrapperSessionForTest(
        [["run", "-n"], ["run", "-n"], ["run", "-n"]],
        self.sess, dump_root=self._tmp_dir)

    # run three times.
    wrapped_sess.run(self.inc_v)
    wrapped_sess.run(self.inc_v)
    wrapped_sess.run(self.inc_v)

    self.assertAllClose(13.0, self.sess.run(self.v))

    self.assertEqual([1, 2, 3],
                     wrapped_sess.observers["run_start_cli_run_numbers"])
    self.assertEqual([], wrapped_sess.observers["run_end_cli_run_numbers"])

  def testRunningWithSparsePlaceholderFeedWorks(self):
    wrapped_sess = LocalCLIDebuggerWrapperSessionForTest(
        [["run"], ["run"]], self.sess, dump_root=self._tmp_dir)

    sparse_feed = ([[0, 1], [0, 2]], [10.0, 20.0])
    sparse_result = wrapped_sess.run(
        self.sparse_add, feed_dict={self.sparse_ph: sparse_feed})
    self.assertAllEqual([[0, 1], [0, 2]], sparse_result.indices)
    self.assertAllClose([20.0, 40.0], sparse_result.values)

  def testRunsUnderNonDebugThenDebugMode(self):
    # Do two NON_DEBUG_RUNs, followed by DEBUG_RUNs.
    wrapped_sess = LocalCLIDebuggerWrapperSessionForTest(
        [["run", "-n"], ["run", "-n"], ["run"], ["run"]],
        self.sess, dump_root=self._tmp_dir)

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
    wrapped_sess = LocalCLIDebuggerWrapperSessionForTest(
        [["run", "-t", "3"], ["run"]],
        self.sess, dump_root=self._tmp_dir)

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
    wrapped_sess = LocalCLIDebuggerWrapperSessionForTest(
        [["run", "-t", "3"]], self.sess, dump_root=self._tmp_dir)

    # run twice, which is less than the number of times specified by the
    # command.
    wrapped_sess.run(self.inc_v)
    wrapped_sess.run(self.inc_v)

    self.assertAllClose(12.0, self.sess.run(self.v))

    self.assertEqual([1], wrapped_sess.observers["run_start_cli_run_numbers"])
    self.assertEqual([], wrapped_sess.observers["run_end_cli_run_numbers"])
    self.assertEqual(0, len(wrapped_sess.observers["debug_dumps"]))
    self.assertEqual([], wrapped_sess.observers["tf_errors"])

  def testRunMixingDebugModeAndMultipleTimes(self):
    wrapped_sess = LocalCLIDebuggerWrapperSessionForTest(
        [["run", "-n"], ["run", "-t", "2"], ["run"], ["run"]],
        self.sess, dump_root=self._tmp_dir)

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

  def testDebuggingMakeCallableTensorRunnerWorks(self):
    wrapped_sess = LocalCLIDebuggerWrapperSessionForTest(
        [["run"], ["run"]], self.sess, dump_root=self._tmp_dir)
    v = variable_v1.VariableV1(42)
    tensor_runner = wrapped_sess.make_callable(v)
    self.sess.run(v.initializer)

    self.assertAllClose(42, tensor_runner())
    self.assertEqual(1, len(wrapped_sess.observers["debug_dumps"]))

  def testDebuggingMakeCallableTensorRunnerWithCustomRunOptionsWorks(self):
    wrapped_sess = LocalCLIDebuggerWrapperSessionForTest(
        [["run"], ["run"]], self.sess, dump_root=self._tmp_dir)
    a = constant_op.constant(42)
    tensor_runner = wrapped_sess.make_callable(a)

    run_options = config_pb2.RunOptions(
        trace_level=config_pb2.RunOptions.FULL_TRACE)
    run_metadata = config_pb2.RunMetadata()
    self.assertAllClose(
        42, tensor_runner(options=run_options, run_metadata=run_metadata))
    self.assertEqual(1, len(wrapped_sess.observers["debug_dumps"]))
    self.assertGreater(len(run_metadata.step_stats.dev_stats), 0)

  def testDebuggingMakeCallableOperationRunnerWorks(self):
    wrapped_sess = LocalCLIDebuggerWrapperSessionForTest(
        [["run"], ["run"]], self.sess, dump_root=self._tmp_dir)
    v = variable_v1.VariableV1(10.0)
    inc_v = state_ops.assign_add(v, 1.0)
    op_runner = wrapped_sess.make_callable(inc_v.op)
    self.sess.run(v.initializer)

    op_runner()
    self.assertEqual(1, len(wrapped_sess.observers["debug_dumps"]))
    self.assertEqual(11.0, self.sess.run(v))

  def testDebuggingMakeCallableRunnerWithFeedListWorks(self):
    wrapped_sess = LocalCLIDebuggerWrapperSessionForTest(
        [["run"], ["run"]], self.sess, dump_root=self._tmp_dir)
    ph1 = array_ops.placeholder(dtypes.float32)
    ph2 = array_ops.placeholder(dtypes.float32)
    a = math_ops.add(ph1, ph2)
    tensor_runner = wrapped_sess.make_callable(a, feed_list=[ph1, ph2])

    self.assertAllClose(42.0, tensor_runner(41.0, 1.0))
    self.assertEqual(1, len(wrapped_sess.observers["debug_dumps"]))

  def testDebuggingMakeCallableFromOptionsWithZeroFeedWorks(self):
    variable_1 = variable_v1.VariableV1(
        10.5, dtype=dtypes.float32, name="variable_1")
    a = math_ops.add(variable_1, variable_1, "callable_a")
    math_ops.add(a, a, "callable_b")
    self.sess.run(variable_1.initializer)

    wrapped_sess = LocalCLIDebuggerWrapperSessionForTest(
        [["run"]] * 3, self.sess, dump_root=self._tmp_dir)
    callable_options = config_pb2.CallableOptions()
    callable_options.fetch.append("callable_b")
    sess_callable = wrapped_sess._make_callable_from_options(callable_options)

    for _ in range(2):
      callable_output = sess_callable()
      self.assertAllClose(np.array(42.0, dtype=np.float32), callable_output[0])

    debug_dumps = wrapped_sess.observers["debug_dumps"]
    self.assertEqual(2, len(debug_dumps))
    for debug_dump in debug_dumps:
      node_names = [datum.node_name for datum in debug_dump.dumped_tensor_data]
      self.assertItemsEqual(
          ["callable_a", "callable_b", "variable_1", "variable_1/read"],
          node_names)

  def testDebuggingMakeCallableFromOptionsWithOneFeedWorks(self):
    ph1 = array_ops.placeholder(dtypes.float32, name="callable_ph1")
    a = math_ops.add(ph1, ph1, "callable_a")
    math_ops.add(a, a, "callable_b")

    wrapped_sess = LocalCLIDebuggerWrapperSessionForTest(
        [["run"]] * 3, self.sess, dump_root=self._tmp_dir)
    callable_options = config_pb2.CallableOptions()
    callable_options.feed.append("callable_ph1")
    callable_options.fetch.append("callable_b")
    sess_callable = wrapped_sess._make_callable_from_options(callable_options)

    ph1_value = np.array([10.5, -10.5], dtype=np.float32)

    for _ in range(2):
      callable_output = sess_callable(ph1_value)
      self.assertAllClose(
          np.array([42.0, -42.0], dtype=np.float32), callable_output[0])

    debug_dumps = wrapped_sess.observers["debug_dumps"]
    self.assertEqual(2, len(debug_dumps))
    for debug_dump in debug_dumps:
      node_names = [datum.node_name for datum in debug_dump.dumped_tensor_data]
      self.assertIn("callable_a", node_names)
      self.assertIn("callable_b", node_names)

  def testDebuggingMakeCallableFromOptionsWithTwoFeedsWorks(self):
    ph1 = array_ops.placeholder(dtypes.float32, name="callable_ph1")
    ph2 = array_ops.placeholder(dtypes.float32, name="callable_ph2")
    a = math_ops.add(ph1, ph2, "callable_a")
    math_ops.add(a, a, "callable_b")

    wrapped_sess = LocalCLIDebuggerWrapperSessionForTest(
        [["run"]] * 3, self.sess, dump_root=self._tmp_dir)
    callable_options = config_pb2.CallableOptions()
    callable_options.feed.append("callable_ph1")
    callable_options.feed.append("callable_ph2")
    callable_options.fetch.append("callable_b")
    sess_callable = wrapped_sess._make_callable_from_options(callable_options)

    ph1_value = np.array(5.0, dtype=np.float32)
    ph2_value = np.array(16.0, dtype=np.float32)

    for _ in range(2):
      callable_output = sess_callable(ph1_value, ph2_value)
      self.assertAllClose(np.array(42.0, dtype=np.float32), callable_output[0])

    debug_dumps = wrapped_sess.observers["debug_dumps"]
    self.assertEqual(2, len(debug_dumps))
    for debug_dump in debug_dumps:
      node_names = [datum.node_name for datum in debug_dump.dumped_tensor_data]
      self.assertIn("callable_a", node_names)
      self.assertIn("callable_b", node_names)

  def testDebugMakeCallableFromOptionsWithCustomOptionsAndMetadataWorks(self):
    variable_1 = variable_v1.VariableV1(
        10.5, dtype=dtypes.float32, name="variable_1")
    a = math_ops.add(variable_1, variable_1, "callable_a")
    math_ops.add(a, a, "callable_b")
    self.sess.run(variable_1.initializer)

    wrapped_sess = LocalCLIDebuggerWrapperSessionForTest(
        [["run"], ["run"]], self.sess, dump_root=self._tmp_dir)
    callable_options = config_pb2.CallableOptions()
    callable_options.fetch.append("callable_b")
    callable_options.run_options.trace_level = config_pb2.RunOptions.FULL_TRACE

    sess_callable = wrapped_sess._make_callable_from_options(callable_options)

    run_metadata = config_pb2.RunMetadata()
    # Call the callable with a custom run_metadata.
    callable_output = sess_callable(run_metadata=run_metadata)
    # Verify that step_stats is populated in the custom run_metadata.
    self.assertTrue(run_metadata.step_stats)
    self.assertAllClose(np.array(42.0, dtype=np.float32), callable_output[0])

    debug_dumps = wrapped_sess.observers["debug_dumps"]
    self.assertEqual(1, len(debug_dumps))
    debug_dump = debug_dumps[0]
    node_names = [datum.node_name for datum in debug_dump.dumped_tensor_data]
    self.assertItemsEqual(
        ["callable_a", "callable_b", "variable_1", "variable_1/read"],
        node_names)

  def testRuntimeErrorShouldBeCaught(self):
    wrapped_sess = LocalCLIDebuggerWrapperSessionForTest(
        [["run"], ["run"]], self.sess, dump_root=self._tmp_dir)

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
    wrapped_sess = LocalCLIDebuggerWrapperSessionForTest(
        [["run", "-f", "v_greater_than_twelve"],
         ["run", "-f", "v_greater_than_twelve"],
         ["run"]],
        self.sess,
        dump_root=self._tmp_dir)

    def v_greater_than_twelve(datum, tensor):
      return datum.node_name == "v" and tensor > 12.0

    # Verify that adding the same tensor filter more than once is tolerated
    # (i.e., as if it were added only once).
    wrapped_sess.add_tensor_filter("v_greater_than_twelve",
                                   v_greater_than_twelve)
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

  def testRunTillFilterPassesWithExcludeOpNames(self):
    wrapped_sess = LocalCLIDebuggerWrapperSessionForTest(
        [["run", "-f", "greater_than_twelve",
          "--filter_exclude_node_names", "inc_v.*"],
         ["run"], ["run"]],
        self.sess,
        dump_root=self._tmp_dir)

    def greater_than_twelve(datum, tensor):
      del datum  # Unused.
      return tensor > 12.0

    # Verify that adding the same tensor filter more than once is tolerated
    # (i.e., as if it were added only once).
    wrapped_sess.add_tensor_filter("greater_than_twelve", greater_than_twelve)

    # run five times.
    wrapped_sess.run(self.inc_v)
    wrapped_sess.run(self.inc_v)
    wrapped_sess.run(self.inc_v)
    wrapped_sess.run(self.inc_v)

    self.assertAllClose(14.0, self.sess.run(self.v))

    self.assertEqual([1], wrapped_sess.observers["run_start_cli_run_numbers"])

    # Due to the --filter_exclude_op_names flag, the run-end CLI should show up
    # not after run 3, but after run 4.
    self.assertEqual([4], wrapped_sess.observers["run_end_cli_run_numbers"])

  def testRunTillFilterPassesWorksInConjunctionWithOtherNodeNameFilter(self):
    """Test that --.*_filter flags work in conjunction with -f.

    In other words, test that you can use a tensor filter on a subset of
    the tensors.
    """
    wrapped_sess = LocalCLIDebuggerWrapperSessionForTest(
        [["run", "-f", "v_greater_than_twelve", "--node_name_filter", "v$"],
         ["run", "-f", "v_greater_than_twelve", "--node_name_filter", "v$"],
         ["run"]],
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

    debug_dumps = wrapped_sess.observers["debug_dumps"]
    self.assertEqual(2, len(debug_dumps))
    self.assertEqual(1, len(debug_dumps[0].dumped_tensor_data))
    self.assertEqual("v:0", debug_dumps[0].dumped_tensor_data[0].tensor_name)
    self.assertEqual(1, len(debug_dumps[1].dumped_tensor_data))
    self.assertEqual("v:0", debug_dumps[1].dumped_tensor_data[0].tensor_name)

  def testRunsUnderDebugModeWithWatchFnFilteringNodeNames(self):
    wrapped_sess = LocalCLIDebuggerWrapperSessionForTest(
        [["run", "--node_name_filter", "inc.*"],
         ["run", "--node_name_filter", "delta"],
         ["run"]],
        self.sess, dump_root=self._tmp_dir)

    # run under debug mode twice.
    wrapped_sess.run(self.inc_v)
    wrapped_sess.run(self.inc_v)

    # Verify that the assign_add op did take effect.
    self.assertAllClose(12.0, self.sess.run(self.v))

    # Verify that the dumps have been generated and picked up during run-end.
    self.assertEqual(2, len(wrapped_sess.observers["debug_dumps"]))

    dumps = wrapped_sess.observers["debug_dumps"][0]
    self.assertEqual(1, dumps.size)
    self.assertEqual("inc_v", dumps.dumped_tensor_data[0].node_name)

    dumps = wrapped_sess.observers["debug_dumps"][1]
    self.assertEqual(1, dumps.size)
    self.assertEqual("delta", dumps.dumped_tensor_data[0].node_name)

  def testRunsUnderDebugModeWithWatchFnFilteringOpTypes(self):
    wrapped_sess = LocalCLIDebuggerWrapperSessionForTest(
        [["run", "--node_name_filter", "delta"],
         ["run", "--op_type_filter", "AssignAdd"],
         ["run"]],
        self.sess, dump_root=self._tmp_dir)

    # run under debug mode twice.
    wrapped_sess.run(self.inc_v)
    wrapped_sess.run(self.inc_v)

    # Verify that the assign_add op did take effect.
    self.assertAllClose(12.0, self.sess.run(self.v))

    # Verify that the dumps have been generated and picked up during run-end.
    self.assertEqual(2, len(wrapped_sess.observers["debug_dumps"]))

    dumps = wrapped_sess.observers["debug_dumps"][0]
    self.assertEqual(1, dumps.size)
    self.assertEqual("delta", dumps.dumped_tensor_data[0].node_name)

    dumps = wrapped_sess.observers["debug_dumps"][1]
    self.assertEqual(1, dumps.size)
    self.assertEqual("inc_v", dumps.dumped_tensor_data[0].node_name)

  def testRunsUnderDebugModeWithWatchFnFilteringTensorDTypes(self):
    wrapped_sess = LocalCLIDebuggerWrapperSessionForTest(
        [["run", "--op_type_filter", "Variable.*"],
         ["run", "--tensor_dtype_filter", "int32"],
         ["run"]],
        self.sess, dump_root=self._tmp_dir)

    # run under debug mode twice.
    wrapped_sess.run(self.w_int)
    wrapped_sess.run(self.w_int)

    # Verify that the dumps have been generated and picked up during run-end.
    self.assertEqual(2, len(wrapped_sess.observers["debug_dumps"]))

    dumps = wrapped_sess.observers["debug_dumps"][0]
    self.assertEqual(2, dumps.size)
    self.assertItemsEqual(
        ["v", "w"], [dumps.dumped_tensor_data[i].node_name for i in [0, 1]])

    dumps = wrapped_sess.observers["debug_dumps"][1]
    self.assertEqual(2, dumps.size)
    self.assertEqual(
        ["w_int_inner", "w_int_outer"],
        [dumps.dumped_tensor_data[i].node_name for i in [0, 1]])

  def testRunsUnderDebugModeWithWatchFnFilteringOpTypesAndTensorDTypes(self):
    wrapped_sess = LocalCLIDebuggerWrapperSessionForTest(
        [["run", "--op_type_filter", "Cast", "--tensor_dtype_filter", "int32"],
         ["run"]],
        self.sess, dump_root=self._tmp_dir)

    # run under debug mode twice.
    wrapped_sess.run(self.w_int)

    # Verify that the dumps have been generated and picked up during run-end.
    self.assertEqual(1, len(wrapped_sess.observers["debug_dumps"]))

    dumps = wrapped_sess.observers["debug_dumps"][0]
    self.assertEqual(1, dumps.size)
    self.assertEqual("w_int_inner", dumps.dumped_tensor_data[0].node_name)

  def testPrintFeedPrintsFeedValueForTensorFeedKey(self):
    wrapped_sess = LocalCLIDebuggerWrapperSessionForTest(
        [["print_feed", "ph:0"], ["run"], ["run"]], self.sess)

    self.assertAllClose(
        [[5.0], [-1.0]],
        wrapped_sess.run(self.y, feed_dict={self.ph: [[0.0, 1.0, 2.0]]}))
    print_feed_responses = wrapped_sess.observers["print_feed_responses"]
    self.assertEqual(1, len(print_feed_responses))
    self.assertEqual(
        ["Tensor \"ph:0 (feed)\":", "", "[[0.0, 1.0, 2.0]]"],
        print_feed_responses[0].lines)

  def testPrintFeedPrintsFeedValueForTensorNameFeedKey(self):
    wrapped_sess = LocalCLIDebuggerWrapperSessionForTest(
        [["print_feed", "ph:0"], ["run"], ["run"]], self.sess)

    self.assertAllClose(
        [[5.0], [-1.0]],
        wrapped_sess.run(self.y, feed_dict={"ph:0": [[0.0, 1.0, 2.0]]}))
    print_feed_responses = wrapped_sess.observers["print_feed_responses"]
    self.assertEqual(1, len(print_feed_responses))
    self.assertEqual(
        ["Tensor \"ph:0 (feed)\":", "", "[[0.0, 1.0, 2.0]]"],
        print_feed_responses[0].lines)

  def testPrintFeedPrintsErrorForInvalidFeedKey(self):
    wrapped_sess = LocalCLIDebuggerWrapperSessionForTest(
        [["print_feed", "spam"], ["run"], ["run"]], self.sess)

    self.assertAllClose(
        [[5.0], [-1.0]],
        wrapped_sess.run(self.y, feed_dict={"ph:0": [[0.0, 1.0, 2.0]]}))
    print_feed_responses = wrapped_sess.observers["print_feed_responses"]
    self.assertEqual(1, len(print_feed_responses))
    self.assertEqual(
        ["ERROR: The feed_dict of the current run does not contain the key "
         "spam"], print_feed_responses[0].lines)

  def testPrintFeedPrintsErrorWhenFeedDictIsNone(self):
    wrapped_sess = LocalCLIDebuggerWrapperSessionForTest(
        [["print_feed", "spam"], ["run"], ["run"]], self.sess)

    wrapped_sess.run(self.w_int)
    print_feed_responses = wrapped_sess.observers["print_feed_responses"]
    self.assertEqual(1, len(print_feed_responses))
    self.assertEqual(
        ["ERROR: The feed_dict of the current run is None or empty."],
        print_feed_responses[0].lines)

  def testRunUnderProfilerModeWorks(self):
    wrapped_sess = LocalCLIDebuggerWrapperSessionForTest(
        [["run", "-p"], ["run"]], self.sess)

    wrapped_sess.run(self.w_int)

    self.assertEqual(1, len(wrapped_sess.observers["profiler_run_metadata"]))
    self.assertTrue(
        wrapped_sess.observers["profiler_run_metadata"][0].step_stats)
    self.assertEqual(1, len(wrapped_sess.observers["profiler_py_graphs"]))
    self.assertIsInstance(
        wrapped_sess.observers["profiler_py_graphs"][0], ops.Graph)

  def testCallingHookDelBeforeAnyRun(self):
    wrapped_sess = LocalCLIDebuggerWrapperSessionForTest(
        [["run"], ["run"]], self.sess)
    del wrapped_sess

  def testCallingShouldStopMethodOnNonWrappedNonMonitoredSessionErrors(self):
    wrapped_sess = LocalCLIDebuggerWrapperSessionForTest(
        [["run"], ["run"]], self.sess)
    with self.assertRaisesRegex(
        ValueError,
        r"The wrapped session .* does not have a method .*should_stop.*"):
      wrapped_sess.should_stop()

  def testLocalCLIDebugWrapperSessionWorksOnMonitoredSession(self):
    monitored_sess = monitored_session.MonitoredSession()
    wrapped_monitored_sess = LocalCLIDebuggerWrapperSessionForTest(
        [["run"], ["run"]], monitored_sess)
    self.assertFalse(wrapped_monitored_sess.should_stop())

  def testRunsWithEmptyFetchWorks(self):
    wrapped_sess = LocalCLIDebuggerWrapperSessionForTest(
        [["run"]], self.sess, dump_root="")

    run_output = wrapped_sess.run([])
    self.assertEqual([], run_output)

  def testRunsWithEmptyNestedFetchWorks(self):
    wrapped_sess = LocalCLIDebuggerWrapperSessionForTest(
        [["run"]], self.sess, dump_root="")

    run_output = wrapped_sess.run({"foo": {"baz": []}, "bar": ()})
    self.assertEqual({"foo": {"baz": []}, "bar": ()}, run_output)

  def testSessionRunHook(self):
    a = array_ops.placeholder(dtypes.float32, [10])
    b = a + 1
    c = b * 2

    class Hook(session_run_hook.SessionRunHook):

      def before_run(self, _):
        return session_run_hook.SessionRunArgs(fetches=c)

    class Hook2(session_run_hook.SessionRunHook):

      def before_run(self, _):
        return session_run_hook.SessionRunArgs(fetches=b)

    sess = session.Session()
    sess = LocalCLIDebuggerWrapperSessionForTest([["run"], ["run"]], sess)

    class SessionCreator(object):

      def create_session(self):
        return sess

    final_sess = monitored_session.MonitoredSession(
        session_creator=SessionCreator(), hooks=[Hook(), Hook2()])

    final_sess.run(b, feed_dict={a: np.arange(10)})
    debug_dumps = sess.observers["debug_dumps"]
    self.assertEqual(1, len(debug_dumps))
    debug_dump = debug_dumps[0]
    node_names = [datum.node_name for datum in debug_dump.dumped_tensor_data]
    self.assertIn(b.op.name, node_names)


if __name__ == "__main__":
  googletest.main()
