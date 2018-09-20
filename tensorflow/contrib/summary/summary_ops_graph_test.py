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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile
import time

import six

from tensorflow.contrib.summary import summary_test_util
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import summary_ops_v2 as summary_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.training import training_util

get_all = summary_test_util.get_all


class GraphFileTest(test_util.TensorFlowTestCase):

  def testSummaryOps(self):
    logdir = self.get_temp_dir()
    writer = summary_ops.create_file_writer(logdir, max_queue=0)
    with writer.as_default(), summary_ops.always_record_summaries():
      summary_ops.generic('tensor', 1, step=1)
      summary_ops.scalar('scalar', 2.0, step=1)
      summary_ops.histogram('histogram', [1.0], step=1)
      summary_ops.image('image', [[[[1.0]]]], step=1)
      summary_ops.audio('audio', [[1.0]], 1.0, 1, step=1)
    with self.test_session() as sess:
      sess.run(summary_ops.summary_writer_initializer_op())
      sess.run(summary_ops.all_summary_ops())
    # The working condition of the ops is tested in the C++ test so we just
    # test here that we're calling them correctly.
    self.assertTrue(gfile.Exists(logdir))

  def testSummaryName(self):
    logdir = self.get_temp_dir()
    writer = summary_ops.create_file_writer(logdir, max_queue=0)
    with writer.as_default(), summary_ops.always_record_summaries():
      summary_ops.scalar('scalar', 2.0, step=1)
    with self.test_session() as sess:
      sess.run(summary_ops.summary_writer_initializer_op())
      sess.run(summary_ops.all_summary_ops())
    events = summary_test_util.events_from_logdir(logdir)
    self.assertEqual(2, len(events))
    self.assertEqual('scalar', events[1].summary.value[0].tag)

  def testSummaryNameScope(self):
    logdir = self.get_temp_dir()
    writer = summary_ops.create_file_writer(logdir, max_queue=0)
    with writer.as_default(), summary_ops.always_record_summaries():
      with ops.name_scope('scope'):
        summary_ops.scalar('scalar', 2.0, step=1)
    with self.test_session() as sess:
      sess.run(summary_ops.summary_writer_initializer_op())
      sess.run(summary_ops.all_summary_ops())
    events = summary_test_util.events_from_logdir(logdir)
    self.assertEqual(2, len(events))
    self.assertEqual('scope/scalar', events[1].summary.value[0].tag)

  def testSummaryGlobalStep(self):
    training_util.get_or_create_global_step()
    logdir = self.get_temp_dir()
    writer = summary_ops.create_file_writer(logdir, max_queue=0)
    with writer.as_default(), summary_ops.always_record_summaries():
      summary_ops.scalar('scalar', 2.0)
    with self.test_session() as sess:
      sess.run(variables.global_variables_initializer())
      sess.run(summary_ops.summary_writer_initializer_op())
      step, _ = sess.run(
          [training_util.get_global_step(), summary_ops.all_summary_ops()])
    events = summary_test_util.events_from_logdir(logdir)
    self.assertEqual(2, len(events))
    self.assertEqual(step, events[1].step)

  def testMaxQueue(self):
    logdir = self.get_temp_dir()
    writer = summary_ops.create_file_writer(
        logdir, max_queue=1, flush_millis=999999)
    with writer.as_default(), summary_ops.always_record_summaries():
      summary_ops.scalar('scalar', 2.0, step=1)
    with self.test_session() as sess:
      sess.run(summary_ops.summary_writer_initializer_op())
      get_total = lambda: len(summary_test_util.events_from_logdir(logdir))
      # Note: First tf.Event is always file_version.
      self.assertEqual(1, get_total())
      sess.run(summary_ops.all_summary_ops())
      self.assertEqual(1, get_total())
      # Should flush after second summary since max_queue = 1
      sess.run(summary_ops.all_summary_ops())
      self.assertEqual(3, get_total())

  def testFlushFunction(self):
    logdir = self.get_temp_dir()
    writer = summary_ops.create_file_writer(
        logdir, max_queue=999999, flush_millis=999999)
    with writer.as_default(), summary_ops.always_record_summaries():
      summary_ops.scalar('scalar', 2.0, step=1)
      flush_op = summary_ops.flush()
    with self.test_session() as sess:
      sess.run(summary_ops.summary_writer_initializer_op())
      get_total = lambda: len(summary_test_util.events_from_logdir(logdir))
      # Note: First tf.Event is always file_version.
      self.assertEqual(1, get_total())
      sess.run(summary_ops.all_summary_ops())
      self.assertEqual(1, get_total())
      sess.run(flush_op)
      self.assertEqual(2, get_total())
      # Test "writer" parameter
      sess.run(summary_ops.all_summary_ops())
      sess.run(summary_ops.flush(writer=writer))
      self.assertEqual(3, get_total())
      sess.run(summary_ops.all_summary_ops())
      sess.run(summary_ops.flush(writer=writer._resource))  # pylint:disable=protected-access
      self.assertEqual(4, get_total())

  def testSharedName(self):
    logdir = self.get_temp_dir()
    with summary_ops.always_record_summaries():
      # Create with default shared name (should match logdir)
      writer1 = summary_ops.create_file_writer(logdir)
      with writer1.as_default():
        summary_ops.scalar('one', 1.0, step=1)
      # Create with explicit logdir shared name (should be same resource/file)
      shared_name = 'logdir:' + logdir
      writer2 = summary_ops.create_file_writer(logdir, name=shared_name)
      with writer2.as_default():
        summary_ops.scalar('two', 2.0, step=2)
      # Create with different shared name (should be separate resource/file)
      writer3 = summary_ops.create_file_writer(logdir, name='other')
      with writer3.as_default():
        summary_ops.scalar('three', 3.0, step=3)

    with self.test_session() as sess:
      # Run init ops across writers sequentially to avoid race condition.
      # TODO(nickfelt): fix race condition in resource manager lookup or create
      sess.run(writer1.init())
      sess.run(writer2.init())
      time.sleep(1.1)  # Ensure filename has a different timestamp
      sess.run(writer3.init())
      sess.run(summary_ops.all_summary_ops())
      sess.run([writer1.flush(), writer2.flush(), writer3.flush()])

    event_files = iter(sorted(gfile.Glob(os.path.join(logdir, '*tfevents*'))))

    # First file has tags "one" and "two"
    events = summary_test_util.events_from_file(next(event_files))
    self.assertEqual('brain.Event:2', events[0].file_version)
    tags = [e.summary.value[0].tag for e in events[1:]]
    self.assertItemsEqual(['one', 'two'], tags)

    # Second file has tag "three"
    events = summary_test_util.events_from_file(next(event_files))
    self.assertEqual('brain.Event:2', events[0].file_version)
    tags = [e.summary.value[0].tag for e in events[1:]]
    self.assertItemsEqual(['three'], tags)

    # No more files
    self.assertRaises(StopIteration, lambda: next(event_files))

  def testWriterInitAndClose(self):
    logdir = self.get_temp_dir()
    with summary_ops.always_record_summaries():
      writer = summary_ops.create_file_writer(
          logdir, max_queue=100, flush_millis=1000000)
      with writer.as_default():
        summary_ops.scalar('one', 1.0, step=1)
    with self.test_session() as sess:
      sess.run(summary_ops.summary_writer_initializer_op())
      get_total = lambda: len(summary_test_util.events_from_logdir(logdir))
      self.assertEqual(1, get_total())  # file_version Event
      # Running init() again while writer is open has no effect
      sess.run(writer.init())
      self.assertEqual(1, get_total())
      sess.run(summary_ops.all_summary_ops())
      self.assertEqual(1, get_total())
      # Running close() should do an implicit flush
      sess.run(writer.close())
      self.assertEqual(2, get_total())
      # Running init() on a closed writer should start a new file
      time.sleep(1.1)  # Ensure filename has a different timestamp
      sess.run(writer.init())
      sess.run(summary_ops.all_summary_ops())
      sess.run(writer.close())
      files = sorted(gfile.Glob(os.path.join(logdir, '*tfevents*')))
      self.assertEqual(2, len(files))
      self.assertEqual(2, len(summary_test_util.events_from_file(files[1])))

  def testWriterFlush(self):
    logdir = self.get_temp_dir()
    with summary_ops.always_record_summaries():
      writer = summary_ops.create_file_writer(
          logdir, max_queue=100, flush_millis=1000000)
      with writer.as_default():
        summary_ops.scalar('one', 1.0, step=1)
    with self.test_session() as sess:
      sess.run(summary_ops.summary_writer_initializer_op())
      get_total = lambda: len(summary_test_util.events_from_logdir(logdir))
      self.assertEqual(1, get_total())  # file_version Event
      sess.run(summary_ops.all_summary_ops())
      self.assertEqual(1, get_total())
      sess.run(writer.flush())
      self.assertEqual(2, get_total())


class GraphDbTest(summary_test_util.SummaryDbTest):

  def testGraphPassedToGraph_isForbiddenForThineOwnSafety(self):
    with self.assertRaises(TypeError):
      summary_ops.graph(ops.Graph())
    with self.assertRaises(TypeError):
      summary_ops.graph('')

  def testGraphSummary(self):
    training_util.get_or_create_global_step()
    name = 'hi'
    graph = graph_pb2.GraphDef(node=(node_def_pb2.NodeDef(name=name),))
    with self.test_session():
      with self.create_db_writer().as_default():
        summary_ops.initialize(graph=graph)
    six.assertCountEqual(self, [name],
                         get_all(self.db, 'SELECT node_name FROM Nodes'))

  def testScalarSummary(self):
    """Test record_summaries_every_n_global_steps and all_summaries()."""
    with ops.Graph().as_default(), self.test_session() as sess:
      global_step = training_util.get_or_create_global_step()
      global_step.initializer.run()
      with ops.device('/cpu:0'):
        step_increment = state_ops.assign_add(global_step, 1)
      sess.run(step_increment)  # Increment global step from 0 to 1

      logdir = tempfile.mkdtemp()
      with summary_ops.create_file_writer(logdir, max_queue=0,
                                          name='t2').as_default():
        with summary_ops.record_summaries_every_n_global_steps(2):
          summary_ops.initialize()
          summary_op = summary_ops.scalar('my_scalar', 2.0)

          # Neither of these should produce a summary because
          # global_step is 1 and "1 % 2 != 0"
          sess.run(summary_ops.all_summary_ops())
          sess.run(summary_op)
          events = summary_test_util.events_from_logdir(logdir)
          self.assertEqual(len(events), 1)

          # Increment global step from 1 to 2 and check that the summary
          # is now written
          sess.run(step_increment)
          sess.run(summary_ops.all_summary_ops())
          events = summary_test_util.events_from_logdir(logdir)
          self.assertEqual(len(events), 2)
          self.assertEqual(events[1].summary.value[0].tag, 'my_scalar')

  def testScalarSummaryNameScope(self):
    """Test record_summaries_every_n_global_steps and all_summaries()."""
    with ops.Graph().as_default(), self.test_session() as sess:
      global_step = training_util.get_or_create_global_step()
      global_step.initializer.run()
      with ops.device('/cpu:0'):
        step_increment = state_ops.assign_add(global_step, 1)
      sess.run(step_increment)  # Increment global step from 0 to 1

      logdir = tempfile.mkdtemp()
      with summary_ops.create_file_writer(logdir, max_queue=0,
                                          name='t2').as_default():
        with summary_ops.record_summaries_every_n_global_steps(2):
          summary_ops.initialize()
          with ops.name_scope('scope'):
            summary_op = summary_ops.scalar('my_scalar', 2.0)

          # Neither of these should produce a summary because
          # global_step is 1 and "1 % 2 != 0"
          sess.run(summary_ops.all_summary_ops())
          sess.run(summary_op)
          events = summary_test_util.events_from_logdir(logdir)
          self.assertEqual(len(events), 1)

          # Increment global step from 1 to 2 and check that the summary
          # is now written
          sess.run(step_increment)
          sess.run(summary_ops.all_summary_ops())
          events = summary_test_util.events_from_logdir(logdir)
          self.assertEqual(len(events), 2)
          self.assertEqual(events[1].summary.value[0].tag, 'scope/my_scalar')

  def testSummaryGraphModeCond(self):
    with ops.Graph().as_default(), self.test_session():
      training_util.get_or_create_global_step()
      logdir = tempfile.mkdtemp()
      with summary_ops.create_file_writer(
          logdir, max_queue=0,
          name='t2').as_default(), summary_ops.always_record_summaries():
        summary_ops.initialize()
        training_util.get_or_create_global_step().initializer.run()
        def f():
          summary_ops.scalar('scalar', 2.0)
          return constant_op.constant(True)
        pred = array_ops.placeholder(dtypes.bool)
        x = control_flow_ops.cond(pred, f,
                                  lambda: constant_op.constant(False))
        x.eval(feed_dict={pred: True})

      events = summary_test_util.events_from_logdir(logdir)
      self.assertEqual(len(events), 2)
      self.assertEqual(events[1].summary.value[0].tag, 'cond/scalar')

  def testSummaryGraphModeWhile(self):
    with ops.Graph().as_default(), self.test_session():
      training_util.get_or_create_global_step()
      logdir = tempfile.mkdtemp()
      with summary_ops.create_file_writer(
          logdir, max_queue=0,
          name='t2').as_default(), summary_ops.always_record_summaries():
        summary_ops.initialize()
        training_util.get_or_create_global_step().initializer.run()
        def body(unused_pred):
          summary_ops.scalar('scalar', 2.0)
          return constant_op.constant(False)
        def cond(pred):
          return pred
        pred = array_ops.placeholder(dtypes.bool)
        x = control_flow_ops.while_loop(cond, body, [pred])
        x.eval(feed_dict={pred: True})

      events = summary_test_util.events_from_logdir(logdir)
      self.assertEqual(len(events), 2)
      self.assertEqual(events[1].summary.value[0].tag, 'while/scalar')


if __name__ == '__main__':
  test.main()
