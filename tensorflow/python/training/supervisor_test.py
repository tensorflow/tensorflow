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
"""Tests for supervisor.py."""

import glob
import os
import shutil
import time
import uuid


from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.util import event_pb2
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.summary import summary
from tensorflow.python.summary import summary_iterator
from tensorflow.python.summary.writer import writer
from tensorflow.python.training import input as input_lib
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.training import server_lib
from tensorflow.python.training import session_manager as session_manager_lib
from tensorflow.python.training import supervisor


def _summary_iterator(test_dir):
  """Reads events from test_dir/events.

  Args:
    test_dir: Name of the test directory.

  Returns:
    A summary_iterator
  """
  event_paths = sorted(glob.glob(os.path.join(test_dir, "event*")))
  return summary_iterator.summary_iterator(event_paths[-1])


class SupervisorTest(test.TestCase):

  def _test_dir(self, test_name):
    test_dir = os.path.join(self.get_temp_dir(), test_name)
    if os.path.exists(test_dir):
      shutil.rmtree(test_dir)
    return test_dir

  def _wait_for_glob(self, pattern, timeout_secs, for_checkpoint=True):
    """Wait for a checkpoint file to appear.

    Args:
      pattern: A string.
      timeout_secs: How long to wait for in seconds.
      for_checkpoint: whether we're globbing for checkpoints.
    """
    end_time = time.time() + timeout_secs
    while time.time() < end_time:
      if for_checkpoint:
        if checkpoint_management.checkpoint_exists(pattern):
          return
      else:
        if len(gfile.Glob(pattern)) >= 1:
          return
      time.sleep(0.05)
    self.assertFalse(True, "Glob never matched any file: %s" % pattern)

  # This test does not test much.
  def testBasics(self):
    logdir = self._test_dir("basics")
    with ops.Graph().as_default():
      my_op = constant_op.constant(1.0)
      sv = supervisor.Supervisor(logdir=logdir)
      sess = sv.prepare_or_wait_for_session("")
      for _ in range(10):
        self.evaluate(my_op)
      sess.close()
      sv.stop()

  def testManagedSession(self):
    logdir = self._test_dir("managed_session")
    with ops.Graph().as_default():
      my_op = constant_op.constant(1.0)
      sv = supervisor.Supervisor(logdir=logdir)
      with sv.managed_session(""):
        for _ in range(10):
          self.evaluate(my_op)
      # Supervisor has been stopped.
      self.assertTrue(sv.should_stop())

  def testManagedSessionUserError(self):
    logdir = self._test_dir("managed_user_error")
    with ops.Graph().as_default():
      my_op = constant_op.constant(1.0)
      sv = supervisor.Supervisor(logdir=logdir)
      last_step = None
      with self.assertRaisesRegex(RuntimeError, "failing here"):
        with sv.managed_session("") as sess:
          for step in range(10):
            last_step = step
            if step == 1:
              raise RuntimeError("failing here")
            else:
              self.evaluate(my_op)
      # Supervisor has been stopped.
      self.assertTrue(sv.should_stop())
      self.assertEqual(1, last_step)

  def testManagedSessionIgnoreOutOfRangeError(self):
    logdir = self._test_dir("managed_out_of_range")
    with ops.Graph().as_default():
      my_op = constant_op.constant(1.0)
      sv = supervisor.Supervisor(logdir=logdir)
      last_step = None
      with sv.managed_session("") as sess:
        for step in range(10):
          last_step = step
          if step == 3:
            raise errors_impl.OutOfRangeError(my_op.op.node_def, my_op.op,
                                              "all done")
          else:
            self.evaluate(my_op)
      # Supervisor has been stopped.  OutOfRangeError was not thrown.
      self.assertTrue(sv.should_stop())
      self.assertEqual(3, last_step)

  def testManagedSessionDoNotKeepSummaryWriter(self):
    logdir = self._test_dir("managed_not_keep_summary_writer")
    with ops.Graph().as_default():
      summary.scalar("c1", constant_op.constant(1))
      summary.scalar("c2", constant_op.constant(2))
      summary.scalar("c3", constant_op.constant(3))
      summ = summary.merge_all()
      sv = supervisor.Supervisor(logdir=logdir, summary_op=None)
      with sv.managed_session(
          "", close_summary_writer=True, start_standard_services=False) as sess:
        sv.summary_computed(sess, sess.run(summ))
      # Sleep 1.2s to make sure that the next event file has a different name
      # than the current one.
      time.sleep(1.2)
      with sv.managed_session(
          "", close_summary_writer=True, start_standard_services=False) as sess:
        sv.summary_computed(sess, sess.run(summ))
    event_paths = sorted(glob.glob(os.path.join(logdir, "event*")))
    self.assertEqual(2, len(event_paths))
    # The two event files should have the same contents.
    for path in event_paths:
      # The summary iterator should report the summary once as we closed the
      # summary writer across the 2 sessions.
      rr = summary_iterator.summary_iterator(path)
      # The first event should list the file_version.
      ev = next(rr)
      self.assertEqual("brain.Event:2", ev.file_version)

      # The next one has the graph and metagraph.
      ev = next(rr)
      self.assertTrue(ev.graph_def)

      ev = next(rr)
      self.assertTrue(ev.meta_graph_def)

      # The next one should have the values from the summary.
      # But only once.
      ev = next(rr)
      self.assertProtoEquals("""
        value { tag: 'c1' simple_value: 1.0 }
        value { tag: 'c2' simple_value: 2.0 }
        value { tag: 'c3' simple_value: 3.0 }
        """, ev.summary)

      # The next one should be a stop message if we closed cleanly.
      ev = next(rr)
      self.assertEqual(event_pb2.SessionLog.STOP, ev.session_log.status)

      # We should be done.
      with self.assertRaises(StopIteration):
        next(rr)

  def testManagedSessionKeepSummaryWriter(self):
    logdir = self._test_dir("managed_keep_summary_writer")
    with ops.Graph().as_default():
      summary.scalar("c1", constant_op.constant(1))
      summary.scalar("c2", constant_op.constant(2))
      summary.scalar("c3", constant_op.constant(3))
      summ = summary.merge_all()
      sv = supervisor.Supervisor(logdir=logdir)
      with sv.managed_session(
          "", close_summary_writer=False,
          start_standard_services=False) as sess:
        sv.summary_computed(sess, sess.run(summ))
      with sv.managed_session(
          "", close_summary_writer=False,
          start_standard_services=False) as sess:
        sv.summary_computed(sess, sess.run(summ))
    # Now close the summary writer to flush the events.
    sv.summary_writer.close()
    # The summary iterator should report the summary twice as we reused
    # the same summary writer across the 2 sessions.
    rr = _summary_iterator(logdir)
    # The first event should list the file_version.
    ev = next(rr)
    self.assertEqual("brain.Event:2", ev.file_version)

    # The next one has the graph.
    ev = next(rr)
    self.assertTrue(ev.graph_def)

    ev = next(rr)
    self.assertTrue(ev.meta_graph_def)

    # The next one should have the values from the summary.
    ev = next(rr)
    self.assertProtoEquals("""
      value { tag: 'c1' simple_value: 1.0 }
      value { tag: 'c2' simple_value: 2.0 }
      value { tag: 'c3' simple_value: 3.0 }
      """, ev.summary)

    # The next one should also have the values from the summary.
    ev = next(rr)
    self.assertProtoEquals("""
      value { tag: 'c1' simple_value: 1.0 }
      value { tag: 'c2' simple_value: 2.0 }
      value { tag: 'c3' simple_value: 3.0 }
      """, ev.summary)

    # We should be done.
    self.assertRaises(StopIteration, lambda: next(rr))

  def _csv_data(self, logdir):
    # Create a small data file with 3 CSV records.
    data_path = os.path.join(logdir, "data.csv")
    with open(data_path, "w") as f:
      f.write("1,2,3\n")
      f.write("4,5,6\n")
      f.write("7,8,9\n")
    return data_path

  def testManagedEndOfInputOneQueue(self):
    # Tests that the supervisor finishes without an error when using
    # a fixed number of epochs, reading from a single queue.
    logdir = self._test_dir("managed_end_of_input_one_queue")
    os.makedirs(logdir)
    data_path = self._csv_data(logdir)
    with ops.Graph().as_default():
      # Create an input pipeline that reads the file 3 times.
      filename_queue = input_lib.string_input_producer(
          [data_path], num_epochs=3)
      reader = io_ops.TextLineReader()
      _, csv = reader.read(filename_queue)
      rec = parsing_ops.decode_csv(csv, record_defaults=[[1], [1], [1]])
      sv = supervisor.Supervisor(logdir=logdir)
      with sv.managed_session("") as sess:
        while not sv.should_stop():
          sess.run(rec)

  def testManagedEndOfInputTwoQueues(self):
    # Tests that the supervisor finishes without an error when using
    # a fixed number of epochs, reading from two queues, the second
    # one producing a batch from the first one.
    logdir = self._test_dir("managed_end_of_input_two_queues")
    os.makedirs(logdir)
    data_path = self._csv_data(logdir)
    with ops.Graph().as_default():
      # Create an input pipeline that reads the file 3 times.
      filename_queue = input_lib.string_input_producer(
          [data_path], num_epochs=3)
      reader = io_ops.TextLineReader()
      _, csv = reader.read(filename_queue)
      rec = parsing_ops.decode_csv(csv, record_defaults=[[1], [1], [1]])
      shuff_rec = input_lib.shuffle_batch(rec, 1, 6, 4)
      sv = supervisor.Supervisor(logdir=logdir)
      with sv.managed_session("") as sess:
        while not sv.should_stop():
          sess.run(shuff_rec)

  def testManagedMainErrorTwoQueues(self):
    # Tests that the supervisor correctly raises a main loop
    # error even when using multiple queues for input.
    logdir = self._test_dir("managed_main_error_two_queues")
    os.makedirs(logdir)
    data_path = self._csv_data(logdir)
    with self.assertRaisesRegex(RuntimeError, "fail at step 3"):
      with ops.Graph().as_default():
        # Create an input pipeline that reads the file 3 times.
        filename_queue = input_lib.string_input_producer(
            [data_path], num_epochs=3)
        reader = io_ops.TextLineReader()
        _, csv = reader.read(filename_queue)
        rec = parsing_ops.decode_csv(csv, record_defaults=[[1], [1], [1]])
        shuff_rec = input_lib.shuffle_batch(rec, 1, 6, 4)
        sv = supervisor.Supervisor(logdir=logdir)
        with sv.managed_session("") as sess:
          for step in range(9):
            if sv.should_stop():
              break
            elif step == 3:
              raise RuntimeError("fail at step 3")
            else:
              sess.run(shuff_rec)

  def testSessionConfig(self):
    logdir = self._test_dir("session_config")
    with ops.Graph().as_default():
      with ops.device("/cpu:1"):
        my_op = constant_op.constant([1.0])
      sv = supervisor.Supervisor(logdir=logdir)
      sess = sv.prepare_or_wait_for_session(
          "", config=config_pb2.ConfigProto(device_count={"CPU": 2}))
      for _ in range(10):
        self.evaluate(my_op)
      sess.close()
      sv.stop()

  def testChiefCanWriteEvents(self):
    logdir = self._test_dir("can_write")
    with ops.Graph().as_default():
      summary.scalar("c1", constant_op.constant(1))
      summary.scalar("c2", constant_op.constant(2))
      summary.scalar("c3", constant_op.constant(3))
      summ = summary.merge_all()
      sv = supervisor.Supervisor(is_chief=True, logdir=logdir, summary_op=None)
      meta_graph_def = meta_graph.create_meta_graph_def()
      sess = sv.prepare_or_wait_for_session("")
      sv.summary_computed(sess, sess.run(summ))
      sess.close()
      # Wait to make sure everything is written to file before stopping.
      time.sleep(1)
      sv.stop()

    rr = _summary_iterator(logdir)

    # The first event should list the file_version.
    ev = next(rr)
    self.assertEqual("brain.Event:2", ev.file_version)

    # The next one has the graph.
    ev = next(rr)
    ev_graph = graph_pb2.GraphDef()
    ev_graph.ParseFromString(ev.graph_def)
    self.assertProtoEquals(sess.graph.as_graph_def(add_shapes=True), ev_graph)

    # Stored MetaGraphDef
    ev = next(rr)
    ev_meta_graph = meta_graph_pb2.MetaGraphDef()
    ev_meta_graph.ParseFromString(ev.meta_graph_def)
    self.assertProtoEquals(meta_graph_def, ev_meta_graph)
    self.assertProtoEquals(
        sess.graph.as_graph_def(add_shapes=True), ev_meta_graph.graph_def)
    # The next one should have the values from the summary.
    ev = next(rr)
    self.assertProtoEquals("""
      value { tag: 'c1' simple_value: 1.0 }
      value { tag: 'c2' simple_value: 2.0 }
      value { tag: 'c3' simple_value: 3.0 }
      """, ev.summary)

    # The next one should be a stop message if we closed cleanly.
    ev = next(rr)
    self.assertEqual(event_pb2.SessionLog.STOP, ev.session_log.status)

    # We should be done.
    self.assertRaises(StopIteration, lambda: next(rr))

  def testNonChiefCannotWriteEvents(self):

    def _summary_computed():
      with ops.Graph().as_default():
        sv = supervisor.Supervisor(is_chief=False)
        sess = sv.prepare_or_wait_for_session("")
        summary.scalar("c1", constant_op.constant(1))
        summary.scalar("c2", constant_op.constant(2))
        summ = summary.merge_all()
        sv.summary_computed(sess, sess.run(summ))

    def _start_standard_services():
      with ops.Graph().as_default():
        sv = supervisor.Supervisor(is_chief=False)
        sess = sv.prepare_or_wait_for_session("")
        sv.start_standard_services(sess)

    self.assertRaises(RuntimeError, _summary_computed)
    self.assertRaises(RuntimeError, _start_standard_services)

  def testNoLogdirButWantSummary(self):
    with ops.Graph().as_default():
      summary.scalar("c1", constant_op.constant(1))
      summary.scalar("c2", constant_op.constant(2))
      summary.scalar("c3", constant_op.constant(3))
      summ = summary.merge_all()
      sv = supervisor.Supervisor(logdir="", summary_op=None)
      sess = sv.prepare_or_wait_for_session("")
      with self.assertRaisesRegex(RuntimeError, "requires a summary writer"):
        sv.summary_computed(sess, sess.run(summ))

  @test_util.run_v1_only("train.Supervisor is for v1 only")
  def testLogdirButExplicitlyNoSummaryWriter(self):
    logdir = self._test_dir("explicit_no_summary_writer")
    with ops.Graph().as_default():
      variable_v1.VariableV1([1.0], name="foo")
      summary.scalar("c1", constant_op.constant(1))
      summary.scalar("c2", constant_op.constant(2))
      summary.scalar("c3", constant_op.constant(3))
      summ = summary.merge_all()
      sv = supervisor.Supervisor(logdir=logdir, summary_writer=None)
      sess = sv.prepare_or_wait_for_session("")
      # Check that a checkpoint is still be generated.
      self._wait_for_glob(sv.save_path, 3.0)
      # Check that we cannot write a summary
      with self.assertRaisesRegex(RuntimeError, "requires a summary writer"):
        sv.summary_computed(sess, sess.run(summ))

  def testNoLogdirButExplicitSummaryWriter(self):
    logdir = self._test_dir("explicit_summary_writer")
    with ops.Graph().as_default():
      summary.scalar("c1", constant_op.constant(1))
      summary.scalar("c2", constant_op.constant(2))
      summary.scalar("c3", constant_op.constant(3))
      summ = summary.merge_all()
      sw = writer.FileWriter(logdir)
      sv = supervisor.Supervisor(logdir="", summary_op=None, summary_writer=sw)
      meta_graph_def = meta_graph.create_meta_graph_def()
      sess = sv.prepare_or_wait_for_session("")
      sv.summary_computed(sess, sess.run(summ))
      sess.close()
      # Wait to make sure everything is written to file before stopping.
      time.sleep(1)
      sv.stop()

    # Check the summary was written to 'logdir'
    rr = _summary_iterator(logdir)

    # The first event should list the file_version.
    ev = next(rr)
    self.assertEqual("brain.Event:2", ev.file_version)

    # The next one has the graph.
    ev = next(rr)
    ev_graph = graph_pb2.GraphDef()
    ev_graph.ParseFromString(ev.graph_def)
    self.assertProtoEquals(sess.graph.as_graph_def(add_shapes=True), ev_graph)

    # Stored MetaGraphDef
    ev = next(rr)
    ev_meta_graph = meta_graph_pb2.MetaGraphDef()
    ev_meta_graph.ParseFromString(ev.meta_graph_def)
    self.assertProtoEquals(meta_graph_def, ev_meta_graph)
    self.assertProtoEquals(
        sess.graph.as_graph_def(add_shapes=True), ev_meta_graph.graph_def)

    # The next one should have the values from the summary.
    ev = next(rr)
    self.assertProtoEquals("""
      value { tag: 'c1' simple_value: 1.0 }
      value { tag: 'c2' simple_value: 2.0 }
      value { tag: 'c3' simple_value: 3.0 }
      """, ev.summary)

    # The next one should be a stop message if we closed cleanly.
    ev = next(rr)
    self.assertEqual(event_pb2.SessionLog.STOP, ev.session_log.status)

    # We should be done.
    self.assertRaises(StopIteration, lambda: next(rr))

  def testNoLogdirSucceeds(self):
    with ops.Graph().as_default():
      variable_v1.VariableV1([1.0, 2.0, 3.0])
      sv = supervisor.Supervisor(logdir="", summary_op=None)
      sess = sv.prepare_or_wait_for_session("")
      sess.close()
      sv.stop()

  def testUseSessionManager(self):
    with ops.Graph().as_default():
      variable_v1.VariableV1([1.0, 2.0, 3.0])
      sm = session_manager_lib.SessionManager()
      # Pass in session_manager. The additional init_op is ignored.
      sv = supervisor.Supervisor(logdir="", session_manager=sm)
      sv.prepare_or_wait_for_session("")

  @test_util.run_v1_only("train.Supervisor is for v1 only")
  def testInitOp(self):
    logdir = self._test_dir("default_init_op")
    with ops.Graph().as_default():
      v = variable_v1.VariableV1([1.0, 2.0, 3.0])
      sv = supervisor.Supervisor(logdir=logdir)
      sess = sv.prepare_or_wait_for_session("")
      self.assertAllClose([1.0, 2.0, 3.0], sess.run(v))
      sv.stop()

  @test_util.run_v1_only("train.Supervisor is for v1 only")
  def testInitFn(self):
    logdir = self._test_dir("default_init_op")
    with ops.Graph().as_default():
      v = variable_v1.VariableV1([1.0, 2.0, 3.0])

      def _init_fn(sess):
        sess.run(v.initializer)

      sv = supervisor.Supervisor(logdir=logdir, init_op=None, init_fn=_init_fn)
      sess = sv.prepare_or_wait_for_session("")
      self.assertAllClose([1.0, 2.0, 3.0], sess.run(v))
      sv.stop()

  @test_util.run_v1_only("train.Supervisor is for v1 only")
  def testInitOpWithFeedDict(self):
    logdir = self._test_dir("feed_dict_init_op")
    with ops.Graph().as_default():
      p = array_ops.placeholder(dtypes.float32, shape=(3,))
      v = variable_v1.VariableV1(p, name="v")
      sv = supervisor.Supervisor(
          logdir=logdir,
          init_op=variables.global_variables_initializer(),
          init_feed_dict={p: [1.0, 2.0, 3.0]})
      sess = sv.prepare_or_wait_for_session("")
      self.assertAllClose([1.0, 2.0, 3.0], sess.run(v))
      sv.stop()

  @test_util.run_v1_only("train.Supervisor is for v1 only")
  def testReadyForLocalInitOp(self):
    server = server_lib.Server.create_local_server()
    logdir = self._test_dir("default_ready_for_local_init_op")

    uid = uuid.uuid4().hex

    def get_session(is_chief):
      g = ops.Graph()
      with g.as_default():
        with ops.device("/job:localhost"):
          v = variable_v1.VariableV1(
              1, name="default_ready_for_local_init_op_v_" + str(uid))
          vadd = v.assign_add(1)
          w = variable_v1.VariableV1(
              v,
              trainable=False,
              collections=[ops.GraphKeys.LOCAL_VARIABLES],
              name="default_ready_for_local_init_op_w_" + str(uid))
          ready_for_local_init_op = variables.report_uninitialized_variables(
              variables.global_variables())
      sv = supervisor.Supervisor(
          logdir=logdir,
          is_chief=is_chief,
          graph=g,
          recovery_wait_secs=1,
          init_op=v.initializer,
          ready_for_local_init_op=ready_for_local_init_op)
      sess = sv.prepare_or_wait_for_session(server.target)

      return sv, sess, v, vadd, w

    sv0, sess0, v0, _, w0 = get_session(True)
    sv1, sess1, _, vadd1, w1 = get_session(False)

    self.assertEqual(1, sess0.run(w0))
    self.assertEqual(2, sess1.run(vadd1))
    self.assertEqual(1, sess1.run(w1))
    self.assertEqual(2, sess0.run(v0))

    sv0.stop()
    sv1.stop()

  @test_util.run_v1_only("train.Supervisor is for v1 only")
  def testReadyForLocalInitOpRestoreFromCheckpoint(self):
    server = server_lib.Server.create_local_server()
    logdir = self._test_dir("ready_for_local_init_op_restore")

    uid = uuid.uuid4().hex

    # Create a checkpoint.
    with ops.Graph().as_default():
      v = variable_v1.VariableV1(
          10.0, name="ready_for_local_init_op_restore_v_" + str(uid))
      summary.scalar("ready_for_local_init_op_restore_v_" + str(uid), v)
      sv = supervisor.Supervisor(logdir=logdir)
      sv.prepare_or_wait_for_session(server.target)
      save_path = sv.save_path
      self._wait_for_glob(save_path, 3.0)
      self._wait_for_glob(
          os.path.join(logdir, "*events*"), 3.0, for_checkpoint=False)
      # Wait to make sure everything is written to file before stopping.
      time.sleep(1)
      sv.stop()

    def get_session(is_chief):
      g = ops.Graph()
      with g.as_default():
        with ops.device("/job:localhost"):
          v = variable_v1.VariableV1(
              1.0, name="ready_for_local_init_op_restore_v_" + str(uid))
          vadd = v.assign_add(1)
          w = variable_v1.VariableV1(
              v,
              trainable=False,
              collections=[ops.GraphKeys.LOCAL_VARIABLES],
              name="ready_for_local_init_op_restore_w_" + str(uid))
          ready_for_local_init_op = variables.report_uninitialized_variables(
              variables.global_variables())
      sv = supervisor.Supervisor(
          logdir=logdir,
          is_chief=is_chief,
          graph=g,
          recovery_wait_secs=1,
          ready_for_local_init_op=ready_for_local_init_op)
      sess = sv.prepare_or_wait_for_session(server.target)

      return sv, sess, v, vadd, w

    sv0, sess0, v0, _, w0 = get_session(True)
    sv1, sess1, _, vadd1, w1 = get_session(False)

    self.assertEqual(10, sess0.run(w0))
    self.assertEqual(11, sess1.run(vadd1))
    self.assertEqual(10, sess1.run(w1))
    self.assertEqual(11, sess0.run(v0))

    sv0.stop()
    sv1.stop()

  def testLocalInitOp(self):
    logdir = self._test_dir("default_local_init_op")
    with ops.Graph().as_default():
      # A local variable.
      v = variable_v1.VariableV1([1.0, 2.0, 3.0],
                                 trainable=False,
                                 collections=[ops.GraphKeys.LOCAL_VARIABLES])

      # An entity which is initialized through a TABLE_INITIALIZER.
      w = variable_v1.VariableV1([4, 5, 6], trainable=False, collections=[])
      ops.add_to_collection(ops.GraphKeys.TABLE_INITIALIZERS, w.initializer)

      # This shouldn't add a variable to the VARIABLES collection responsible
      # for variables that are saved/restored from checkpoints.
      self.assertEqual(len(variables.global_variables()), 0)

      # Suppress normal variable inits to make sure the local one is
      # initialized via local_init_op.
      sv = supervisor.Supervisor(logdir=logdir, init_op=None)
      sess = sv.prepare_or_wait_for_session("")
      self.assertAllClose([1.0, 2.0, 3.0], sess.run(v))
      self.assertAllClose([4, 5, 6], sess.run(w))
      sv.stop()

  def testLocalInitOpForNonChief(self):
    logdir = self._test_dir("default_local_init_op_non_chief")
    with ops.Graph().as_default():
      with ops.device("/job:localhost"):
        # A local variable.
        v = variable_v1.VariableV1([1.0, 2.0, 3.0],
                                   trainable=False,
                                   collections=[ops.GraphKeys.LOCAL_VARIABLES])
        # This shouldn't add a variable to the VARIABLES collection responsible
        # for variables that are saved/restored from checkpoints.
        self.assertEqual(len(variables.global_variables()), 0)

      # Suppress normal variable inits to make sure the local one is
      # initialized via local_init_op.
      sv = supervisor.Supervisor(logdir=logdir, init_op=None, is_chief=False)
      sess = sv.prepare_or_wait_for_session("")
      self.assertAllClose([1.0, 2.0, 3.0], sess.run(v))
      sv.stop()

  def testInitOpFails(self):
    server = server_lib.Server.create_local_server()
    logdir = self._test_dir("default_init_op_fails")
    with ops.Graph().as_default():
      v = variable_v1.VariableV1([1.0, 2.0, 3.0], name="v")
      variable_v1.VariableV1([4.0, 5.0, 6.0], name="w")
      # w will not be initialized.
      sv = supervisor.Supervisor(logdir=logdir, init_op=v.initializer)
      with self.assertRaisesRegex(RuntimeError, "Variables not initialized: w"):
        sv.prepare_or_wait_for_session(server.target)

  def testInitOpFailsForTransientVariable(self):
    server = server_lib.Server.create_local_server()
    logdir = self._test_dir("default_init_op_fails_for_local_variable")
    with ops.Graph().as_default():
      v = variable_v1.VariableV1([1.0, 2.0, 3.0],
                                 name="v",
                                 collections=[ops.GraphKeys.LOCAL_VARIABLES])
      variable_v1.VariableV1([1.0, 2.0, 3.0],
                             name="w",
                             collections=[ops.GraphKeys.LOCAL_VARIABLES])
      # w will not be initialized.
      sv = supervisor.Supervisor(logdir=logdir, local_init_op=v.initializer)
      with self.assertRaisesRegex(RuntimeError, "Variables not initialized: w"):
        sv.prepare_or_wait_for_session(server.target)

  @test_util.run_v1_only("train.Supervisor is for v1 only")
  def testSetupFail(self):
    logdir = self._test_dir("setup_fail")
    with ops.Graph().as_default():
      variable_v1.VariableV1([1.0, 2.0, 3.0], name="v")
      with self.assertRaisesRegex(ValueError, "must have their device set"):
        supervisor.Supervisor(logdir=logdir, is_chief=False)
    with ops.Graph().as_default(), ops.device("/job:ps"):
      variable_v1.VariableV1([1.0, 2.0, 3.0], name="v")
      supervisor.Supervisor(logdir=logdir, is_chief=False)

  @test_util.run_v1_only("train.Supervisor is for v1 only")
  def testDefaultGlobalStep(self):
    logdir = self._test_dir("default_global_step")
    with ops.Graph().as_default():
      variable_v1.VariableV1(287, name="global_step")
      sv = supervisor.Supervisor(logdir=logdir)
      sess = sv.prepare_or_wait_for_session("")
      self.assertEqual(287, sess.run(sv.global_step))
      sv.stop()

  @test_util.run_v1_only("train.Supervisor is for v1 only")
  def testRestoreFromMetaGraph(self):
    logdir = self._test_dir("restore_from_meta_graph")
    with ops.Graph().as_default():
      variable_v1.VariableV1(1, name="v0")
      sv = supervisor.Supervisor(logdir=logdir)
      sess = sv.prepare_or_wait_for_session("")
      filename = sv.saver.save(sess, sv.save_path)
      sv.stop()
    # Create a new Graph and Supervisor and recover.
    with ops.Graph().as_default():
      new_saver = saver_lib.import_meta_graph(".".join([filename, "meta"]))
      self.assertIsNotNone(new_saver)
      sv2 = supervisor.Supervisor(logdir=logdir, saver=new_saver)
      sess = sv2.prepare_or_wait_for_session("")
      self.assertEqual(1, sess.run("v0:0"))
      sv2.saver.save(sess, sv2.save_path)
      sv2.stop()

  # This test is based on the fact that the standard services start
  # right away and get to run once before sv.stop() returns.
  # We still sleep a bit to make the test robust.
  @test_util.run_v1_only("train.Supervisor is for v1 only")
  def testStandardServicesWithoutGlobalStep(self):
    logdir = self._test_dir("standard_services_without_global_step")
    # Create a checkpoint.
    with ops.Graph().as_default():
      v = variable_v1.VariableV1([1.0], name="foo")
      summary.scalar("v", v[0])
      sv = supervisor.Supervisor(logdir=logdir)
      meta_graph_def = meta_graph.create_meta_graph_def(
          saver_def=sv.saver.saver_def)
      sess = sv.prepare_or_wait_for_session("")
      save_path = sv.save_path
      self._wait_for_glob(save_path, 3.0)
      self._wait_for_glob(
          os.path.join(logdir, "*events*"), 3.0, for_checkpoint=False)
      # Wait to make sure everything is written to file before stopping.
      time.sleep(1)
      sv.stop()
    # There should be an event file with a version number.
    rr = _summary_iterator(logdir)
    ev = next(rr)
    self.assertEqual("brain.Event:2", ev.file_version)
    ev = next(rr)
    ev_graph = graph_pb2.GraphDef()
    ev_graph.ParseFromString(ev.graph_def)
    self.assertProtoEquals(sess.graph.as_graph_def(add_shapes=True), ev_graph)

    # Stored MetaGraphDef
    ev = next(rr)
    ev_meta_graph = meta_graph_pb2.MetaGraphDef()
    ev_meta_graph.ParseFromString(ev.meta_graph_def)
    self.assertProtoEquals(meta_graph_def, ev_meta_graph)
    self.assertProtoEquals(
        sess.graph.as_graph_def(add_shapes=True), ev_meta_graph.graph_def)

    ev = next(rr)
    self.assertProtoEquals("value { tag: 'v' simple_value: 1.0 }", ev.summary)

    ev = next(rr)
    self.assertEqual(event_pb2.SessionLog.STOP, ev.session_log.status)

    self.assertRaises(StopIteration, lambda: next(rr))
    # There should be a checkpoint file with the variable "foo"
    with ops.Graph().as_default(), self.cached_session() as sess:
      v = variable_v1.VariableV1([10.10], name="foo")
      sav = saver_lib.Saver([v])
      sav.restore(sess, save_path)
      self.assertEqual(1.0, self.evaluate(v)[0])

  # Same as testStandardServicesNoGlobalStep but with a global step.
  # We should get a summary about the step time.
  @test_util.run_v1_only("train.Supervisor is for v1 only")
  def testStandardServicesWithGlobalStep(self):
    logdir = self._test_dir("standard_services_with_global_step")
    # Create a checkpoint.
    with ops.Graph().as_default():
      v = variable_v1.VariableV1([123], name="global_step")
      sv = supervisor.Supervisor(logdir=logdir)
      meta_graph_def = meta_graph.create_meta_graph_def(
          saver_def=sv.saver.saver_def)
      sess = sv.prepare_or_wait_for_session("")
      # This is where the checkpoint will appear, with step number 123.
      save_path = "%s-123" % sv.save_path
      self._wait_for_glob(save_path, 3.0)
      self._wait_for_glob(
          os.path.join(logdir, "*events*"), 3.0, for_checkpoint=False)
      # Wait to make sure everything is written to file before stopping.
      time.sleep(1)
      sv.stop()
    # There should be an event file with a version number.
    rr = _summary_iterator(logdir)
    ev = next(rr)
    self.assertEqual("brain.Event:2", ev.file_version)
    ev = next(rr)
    ev_graph = graph_pb2.GraphDef()
    ev_graph.ParseFromString(ev.graph_def)
    self.assertProtoEquals(sess.graph.as_graph_def(add_shapes=True), ev_graph)
    ev = next(rr)
    ev_meta_graph = meta_graph_pb2.MetaGraphDef()
    ev_meta_graph.ParseFromString(ev.meta_graph_def)
    self.assertProtoEquals(meta_graph_def, ev_meta_graph)
    self.assertProtoEquals(
        sess.graph.as_graph_def(add_shapes=True), ev_meta_graph.graph_def)
    ev = next(rr)
    # It is actually undeterministic whether SessionLog.START gets written
    # before the summary or the checkpoint, but this works when run 10000 times.
    self.assertEqual(123, ev.step)
    self.assertEqual(event_pb2.SessionLog.START, ev.session_log.status)
    first = next(rr)
    second = next(rr)
    # It is undeterministic whether the value gets written before the checkpoint
    # since they are on separate threads, so we check for both conditions.
    if first.HasField("summary"):
      self.assertProtoEquals("""value { tag: 'global_step/sec'
                                        simple_value: 0.0 }""", first.summary)
      self.assertEqual(123, second.step)
      self.assertEqual(event_pb2.SessionLog.CHECKPOINT,
                       second.session_log.status)
    else:
      self.assertEqual(123, first.step)
      self.assertEqual(event_pb2.SessionLog.CHECKPOINT,
                       first.session_log.status)
      self.assertProtoEquals("""value { tag: 'global_step/sec'
                                        simple_value: 0.0 }""", second.summary)
    ev = next(rr)
    self.assertEqual(event_pb2.SessionLog.STOP, ev.session_log.status)
    self.assertRaises(StopIteration, lambda: next(rr))
    # There should be a checkpoint file with the variable "foo"
    with ops.Graph().as_default(), self.cached_session() as sess:
      v = variable_v1.VariableV1([-12], name="global_step")
      sav = saver_lib.Saver([v])
      sav.restore(sess, save_path)
      self.assertEqual(123, self.evaluate(v)[0])

  def testNoQueueRunners(self):
    with ops.Graph().as_default(), self.cached_session() as sess:
      sv = supervisor.Supervisor(logdir=self._test_dir("no_queue_runners"))
      self.assertEqual(0, len(sv.start_queue_runners(sess)))
      sv.stop()

  def testPrepareSessionAfterStopForChief(self):
    logdir = self._test_dir("prepare_after_stop_chief")
    with ops.Graph().as_default():
      sv = supervisor.Supervisor(logdir=logdir, is_chief=True)

      # Create a first session and then stop.
      sess = sv.prepare_or_wait_for_session("")
      sv.stop()
      sess.close()
      self.assertTrue(sv.should_stop())

      # Now create a second session and test that we don't stay stopped, until
      # we ask to stop again.
      sess2 = sv.prepare_or_wait_for_session("")
      self.assertFalse(sv.should_stop())
      sv.stop()
      sess2.close()
      self.assertTrue(sv.should_stop())

  def testPrepareSessionAfterStopForNonChief(self):
    logdir = self._test_dir("prepare_after_stop_nonchief")
    with ops.Graph().as_default():
      sv = supervisor.Supervisor(logdir=logdir, is_chief=False)

      # Create a first session and then stop.
      sess = sv.prepare_or_wait_for_session("")
      sv.stop()
      sess.close()
      self.assertTrue(sv.should_stop())

      # Now create a second session and test that we don't stay stopped, until
      # we ask to stop again.
      sess2 = sv.prepare_or_wait_for_session("")
      self.assertFalse(sv.should_stop())
      sv.stop()
      sess2.close()
      self.assertTrue(sv.should_stop())


if __name__ == "__main__":
  test.main()
