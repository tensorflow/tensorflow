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

import tempfile

import six

from tensorflow.contrib.summary import summary_ops
from tensorflow.contrib.summary import summary_test_util
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.platform import test
from tensorflow.python.training import training_util

get_all = summary_test_util.get_all


class DbTest(summary_test_util.SummaryDbTest):

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
