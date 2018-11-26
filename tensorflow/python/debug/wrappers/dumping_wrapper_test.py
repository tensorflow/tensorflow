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
"""Unit Tests for classes in dumping_wrapper.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
import shutil
import tempfile
import threading

from tensorflow.python.client import session
from tensorflow.python.debug.lib import debug_data
from tensorflow.python.debug.lib import stepper
from tensorflow.python.debug.wrappers import dumping_wrapper
from tensorflow.python.debug.wrappers import framework
from tensorflow.python.debug.wrappers import hooks
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import googletest
from tensorflow.python.training import monitored_session


class DumpingDebugWrapperSessionTest(test_util.TensorFlowTestCase):

  def setUp(self):
    self.session_root = tempfile.mkdtemp()

    self.v = variables.VariableV1(10.0, dtype=dtypes.float32, name="v")
    self.delta = constant_op.constant(1.0, dtype=dtypes.float32, name="delta")
    self.eta = constant_op.constant(-1.4, dtype=dtypes.float32, name="eta")
    self.inc_v = state_ops.assign_add(self.v, self.delta, name="inc_v")
    self.dec_v = state_ops.assign_add(self.v, self.eta, name="dec_v")

    self.ph = array_ops.placeholder(dtypes.float32, shape=(), name="ph")
    self.inc_w_ph = state_ops.assign_add(self.v, self.ph, name="inc_w_ph")

    self.sess = session.Session()
    self.sess.run(self.v.initializer)

  def tearDown(self):
    ops.reset_default_graph()
    if os.path.isdir(self.session_root):
      shutil.rmtree(self.session_root)

  def _assert_correct_run_subdir_naming(self, run_subdir):
    self.assertStartsWith(run_subdir, "run_")
    self.assertEqual(2, run_subdir.count("_"))
    self.assertGreater(int(run_subdir.split("_")[1]), 0)

  def testConstructWrapperWithExistingNonEmptyRootDirRaisesException(self):
    dir_path = os.path.join(self.session_root, "foo")
    os.mkdir(dir_path)
    self.assertTrue(os.path.isdir(dir_path))

    with self.assertRaisesRegexp(
        ValueError, "session_root path points to a non-empty directory"):
      dumping_wrapper.DumpingDebugWrapperSession(
          session.Session(), session_root=self.session_root, log_usage=False)

  def testConstructWrapperWithExistingFileDumpRootRaisesException(self):
    file_path = os.path.join(self.session_root, "foo")
    open(file_path, "a").close()  # Create the file
    self.assertTrue(gfile.Exists(file_path))
    self.assertFalse(gfile.IsDirectory(file_path))
    with self.assertRaisesRegexp(ValueError,
                                 "session_root path points to a file"):
      dumping_wrapper.DumpingDebugWrapperSession(
          session.Session(), session_root=file_path, log_usage=False)

  def testConstructWrapperWithNonexistentSessionRootCreatesDirectory(self):
    new_dir_path = os.path.join(tempfile.mkdtemp(), "new_dir")
    dumping_wrapper.DumpingDebugWrapperSession(
        session.Session(), session_root=new_dir_path, log_usage=False)
    self.assertTrue(gfile.IsDirectory(new_dir_path))
    # Cleanup.
    gfile.DeleteRecursively(new_dir_path)

  def testDumpingOnASingleRunWorks(self):
    sess = dumping_wrapper.DumpingDebugWrapperSession(
        self.sess, session_root=self.session_root, log_usage=False)
    sess.run(self.inc_v)

    dump_dirs = glob.glob(os.path.join(self.session_root, "run_*"))
    self.assertEqual(1, len(dump_dirs))

    self._assert_correct_run_subdir_naming(os.path.basename(dump_dirs[0]))
    dump = debug_data.DebugDumpDir(dump_dirs[0])
    self.assertAllClose([10.0], dump.get_tensors("v", 0, "DebugIdentity"))

    self.assertEqual(repr(self.inc_v), dump.run_fetches_info)
    self.assertEqual(repr(None), dump.run_feed_keys_info)

  def testDumpingOnASingleRunWorksWithRelativePathForDebugDumpDir(self):
    sess = dumping_wrapper.DumpingDebugWrapperSession(
        self.sess, session_root=self.session_root, log_usage=False)
    sess.run(self.inc_v)
    dump_dirs = glob.glob(os.path.join(self.session_root, "run_*"))
    cwd = os.getcwd()
    try:
      os.chdir(self.session_root)
      dump = debug_data.DebugDumpDir(
          os.path.relpath(dump_dirs[0], self.session_root))
      self.assertAllClose([10.0], dump.get_tensors("v", 0, "DebugIdentity"))
    finally:
      os.chdir(cwd)

  def testDumpingOnASingleRunWithFeedDictWorks(self):
    sess = dumping_wrapper.DumpingDebugWrapperSession(
        self.sess, session_root=self.session_root, log_usage=False)
    feed_dict = {self.ph: 3.2}
    sess.run(self.inc_w_ph, feed_dict=feed_dict)

    dump_dirs = glob.glob(os.path.join(self.session_root, "run_*"))
    self.assertEqual(1, len(dump_dirs))

    self._assert_correct_run_subdir_naming(os.path.basename(dump_dirs[0]))
    dump = debug_data.DebugDumpDir(dump_dirs[0])
    self.assertAllClose([10.0], dump.get_tensors("v", 0, "DebugIdentity"))

    self.assertEqual(repr(self.inc_w_ph), dump.run_fetches_info)
    self.assertEqual(repr(feed_dict.keys()), dump.run_feed_keys_info)

  def testDumpingOnMultipleRunsWorks(self):
    sess = dumping_wrapper.DumpingDebugWrapperSession(
        self.sess, session_root=self.session_root, log_usage=False)
    for _ in range(3):
      sess.run(self.inc_v)

    dump_dirs = glob.glob(os.path.join(self.session_root, "run_*"))
    dump_dirs = sorted(
        dump_dirs, key=lambda x: int(os.path.basename(x).split("_")[1]))
    self.assertEqual(3, len(dump_dirs))
    for i, dump_dir in enumerate(dump_dirs):
      self._assert_correct_run_subdir_naming(os.path.basename(dump_dir))
      dump = debug_data.DebugDumpDir(dump_dir)
      self.assertAllClose([10.0 + 1.0 * i],
                          dump.get_tensors("v", 0, "DebugIdentity"))
      self.assertEqual(repr(self.inc_v), dump.run_fetches_info)
      self.assertEqual(repr(None), dump.run_feed_keys_info)

  def testUsingNonCallableAsWatchFnRaisesTypeError(self):
    bad_watch_fn = "bad_watch_fn"
    with self.assertRaisesRegexp(TypeError, "watch_fn is not callable"):
      dumping_wrapper.DumpingDebugWrapperSession(
          self.sess,
          session_root=self.session_root,
          watch_fn=bad_watch_fn,
          log_usage=False)

  def testDumpingWithLegacyWatchFnOnFetchesWorks(self):
    """Use a watch_fn that returns different whitelists for different runs."""

    def watch_fn(fetches, feeds):
      del feeds
      # A watch_fn that picks fetch name.
      if fetches.name == "inc_v:0":
        # If inc_v, watch everything.
        return "DebugIdentity", r".*", r".*"
      else:
        # If dec_v, watch nothing.
        return "DebugIdentity", r"$^", r"$^"

    sess = dumping_wrapper.DumpingDebugWrapperSession(
        self.sess,
        session_root=self.session_root,
        watch_fn=watch_fn,
        log_usage=False)

    for _ in range(3):
      sess.run(self.inc_v)
      sess.run(self.dec_v)

    dump_dirs = glob.glob(os.path.join(self.session_root, "run_*"))
    dump_dirs = sorted(
        dump_dirs, key=lambda x: int(os.path.basename(x).split("_")[1]))
    self.assertEqual(6, len(dump_dirs))

    for i, dump_dir in enumerate(dump_dirs):
      self._assert_correct_run_subdir_naming(os.path.basename(dump_dir))
      dump = debug_data.DebugDumpDir(dump_dir)
      if i % 2 == 0:
        self.assertGreater(dump.size, 0)
        self.assertAllClose([10.0 - 0.4 * (i / 2)],
                            dump.get_tensors("v", 0, "DebugIdentity"))
        self.assertEqual(repr(self.inc_v), dump.run_fetches_info)
        self.assertEqual(repr(None), dump.run_feed_keys_info)
      else:
        self.assertEqual(0, dump.size)
        self.assertEqual(repr(self.dec_v), dump.run_fetches_info)
        self.assertEqual(repr(None), dump.run_feed_keys_info)

  def testDumpingWithLegacyWatchFnWithNonDefaultDebugOpsWorks(self):
    """Use a watch_fn that specifies non-default debug ops."""

    def watch_fn(fetches, feeds):
      del fetches, feeds
      return ["DebugIdentity", "DebugNumericSummary"], r".*", r".*"

    sess = dumping_wrapper.DumpingDebugWrapperSession(
        self.sess,
        session_root=self.session_root,
        watch_fn=watch_fn,
        log_usage=False)

    sess.run(self.inc_v)

    dump_dirs = glob.glob(os.path.join(self.session_root, "run_*"))
    self.assertEqual(1, len(dump_dirs))
    dump = debug_data.DebugDumpDir(dump_dirs[0])

    self.assertAllClose([10.0], dump.get_tensors("v", 0, "DebugIdentity"))
    self.assertEqual(14,
                     len(dump.get_tensors("v", 0, "DebugNumericSummary")[0]))

  def testDumpingWithWatchFnWithNonDefaultDebugOpsWorks(self):
    """Use a watch_fn that specifies non-default debug ops."""

    def watch_fn(fetches, feeds):
      del fetches, feeds
      return framework.WatchOptions(
          debug_ops=["DebugIdentity", "DebugNumericSummary"],
          node_name_regex_whitelist=r"^v.*",
          op_type_regex_whitelist=r".*",
          tensor_dtype_regex_whitelist=".*_ref")

    sess = dumping_wrapper.DumpingDebugWrapperSession(
        self.sess,
        session_root=self.session_root,
        watch_fn=watch_fn,
        log_usage=False)

    sess.run(self.inc_v)

    dump_dirs = glob.glob(os.path.join(self.session_root, "run_*"))
    self.assertEqual(1, len(dump_dirs))
    dump = debug_data.DebugDumpDir(dump_dirs[0])

    self.assertAllClose([10.0], dump.get_tensors("v", 0, "DebugIdentity"))
    self.assertEqual(14,
                     len(dump.get_tensors("v", 0, "DebugNumericSummary")[0]))

    dumped_nodes = [dump.node_name for dump in dump.dumped_tensor_data]
    self.assertNotIn("inc_v", dumped_nodes)
    self.assertNotIn("delta", dumped_nodes)

  def testDumpingDebugHookWithoutWatchFnWorks(self):
    dumping_hook = hooks.DumpingDebugHook(self.session_root, log_usage=False)
    mon_sess = monitored_session._HookedSession(self.sess, [dumping_hook])
    mon_sess.run(self.inc_v)

    dump_dirs = glob.glob(os.path.join(self.session_root, "run_*"))
    self.assertEqual(1, len(dump_dirs))

    self._assert_correct_run_subdir_naming(os.path.basename(dump_dirs[0]))
    dump = debug_data.DebugDumpDir(dump_dirs[0])
    self.assertAllClose([10.0], dump.get_tensors("v", 0, "DebugIdentity"))

    self.assertEqual(repr(self.inc_v), dump.run_fetches_info)
    self.assertEqual(repr(None), dump.run_feed_keys_info)

  def testDumpingDebugHookWithStatefulWatchFnWorks(self):
    watch_fn_state = {"run_counter": 0}

    def counting_watch_fn(fetches, feed_dict):
      del fetches, feed_dict
      watch_fn_state["run_counter"] += 1
      if watch_fn_state["run_counter"] % 2 == 1:
        # If odd-index run (1-based), watch every ref-type tensor.
        return framework.WatchOptions(
            debug_ops="DebugIdentity",
            tensor_dtype_regex_whitelist=".*_ref")
      else:
        # If even-index run, watch nothing.
        return framework.WatchOptions(
            debug_ops="DebugIdentity",
            node_name_regex_whitelist=r"^$",
            op_type_regex_whitelist=r"^$")

    dumping_hook = hooks.DumpingDebugHook(
        self.session_root, watch_fn=counting_watch_fn, log_usage=False)
    mon_sess = monitored_session._HookedSession(self.sess, [dumping_hook])
    for _ in range(4):
      mon_sess.run(self.inc_v)

    dump_dirs = glob.glob(os.path.join(self.session_root, "run_*"))
    dump_dirs = sorted(
        dump_dirs, key=lambda x: int(os.path.basename(x).split("_")[1]))
    self.assertEqual(4, len(dump_dirs))

    for i, dump_dir in enumerate(dump_dirs):
      self._assert_correct_run_subdir_naming(os.path.basename(dump_dir))
      dump = debug_data.DebugDumpDir(dump_dir)
      if i % 2 == 0:
        self.assertAllClose([10.0 + 1.0 * i],
                            dump.get_tensors("v", 0, "DebugIdentity"))
        self.assertNotIn("delta",
                         [datum.node_name for datum in dump.dumped_tensor_data])
      else:
        self.assertEqual(0, dump.size)

      self.assertEqual(repr(self.inc_v), dump.run_fetches_info)
      self.assertEqual(repr(None), dump.run_feed_keys_info)

  def testDumpingDebugHookWithStatefulLegacyWatchFnWorks(self):
    watch_fn_state = {"run_counter": 0}

    def counting_watch_fn(fetches, feed_dict):
      del fetches, feed_dict
      watch_fn_state["run_counter"] += 1
      if watch_fn_state["run_counter"] % 2 == 1:
        # If odd-index run (1-based), watch everything.
        return "DebugIdentity", r".*", r".*"
      else:
        # If even-index run, watch nothing.
        return "DebugIdentity", r"$^", r"$^"

    dumping_hook = hooks.DumpingDebugHook(
        self.session_root, watch_fn=counting_watch_fn, log_usage=False)
    mon_sess = monitored_session._HookedSession(self.sess, [dumping_hook])
    for _ in range(4):
      mon_sess.run(self.inc_v)

    dump_dirs = glob.glob(os.path.join(self.session_root, "run_*"))
    dump_dirs = sorted(
        dump_dirs, key=lambda x: int(os.path.basename(x).split("_")[1]))
    self.assertEqual(4, len(dump_dirs))

    for i, dump_dir in enumerate(dump_dirs):
      self._assert_correct_run_subdir_naming(os.path.basename(dump_dir))
      dump = debug_data.DebugDumpDir(dump_dir)
      if i % 2 == 0:
        self.assertAllClose([10.0 + 1.0 * i],
                            dump.get_tensors("v", 0, "DebugIdentity"))
      else:
        self.assertEqual(0, dump.size)

      self.assertEqual(repr(self.inc_v), dump.run_fetches_info)
      self.assertEqual(repr(None), dump.run_feed_keys_info)

  def testDumpingFromMultipleThreadsObeysThreadNameFilter(self):
    sess = dumping_wrapper.DumpingDebugWrapperSession(
        self.sess, session_root=self.session_root, log_usage=False,
        thread_name_filter=r"MainThread$")

    self.assertAllClose(1.0, sess.run(self.delta))
    child_thread_result = []
    def child_thread_job():
      child_thread_result.append(sess.run(self.eta))

    thread = threading.Thread(name="ChildThread", target=child_thread_job)
    thread.start()
    thread.join()
    self.assertAllClose([-1.4], child_thread_result)

    dump_dirs = glob.glob(os.path.join(self.session_root, "run_*"))
    self.assertEqual(1, len(dump_dirs))
    dump = debug_data.DebugDumpDir(dump_dirs[0])
    self.assertEqual(1, dump.size)
    self.assertEqual("delta", dump.dumped_tensor_data[0].node_name)

  def testCallingInvokeNodeStepperOnDumpingWrapperRaisesException(self):
    sess = dumping_wrapper.DumpingDebugWrapperSession(
        self.sess, session_root=self.session_root, log_usage=False)
    node_stepper = stepper.NodeStepper(self.sess, self.inc_v)
    with self.assertRaisesRegexp(
        NotImplementedError,
        r"NonInteractiveDebugWrapperSession does not support node-stepper "
        r"mode\."):
      sess.invoke_node_stepper(node_stepper)

  def testDumpingWrapperWithEmptyFetchWorks(self):
    sess = dumping_wrapper.DumpingDebugWrapperSession(
        self.sess, session_root=self.session_root, log_usage=False)
    sess.run([])


if __name__ == "__main__":
  googletest.main()
