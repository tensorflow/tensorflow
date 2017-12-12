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
