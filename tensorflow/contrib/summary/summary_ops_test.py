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

import numpy as np
import six

from tensorflow.contrib.summary import summary_ops
from tensorflow.contrib.summary import summary_test_util
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.python.eager import function
from tensorflow.python.eager import test
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.platform import gfile
from tensorflow.python.training import training_util

get_all = summary_test_util.get_all
get_one = summary_test_util.get_one

_NUMPY_NUMERIC_TYPES = {
    types_pb2.DT_HALF: np.float16,
    types_pb2.DT_FLOAT: np.float32,
    types_pb2.DT_DOUBLE: np.float64,
    types_pb2.DT_INT8: np.int8,
    types_pb2.DT_INT16: np.int16,
    types_pb2.DT_INT32: np.int32,
    types_pb2.DT_INT64: np.int64,
    types_pb2.DT_UINT8: np.uint8,
    types_pb2.DT_UINT16: np.uint16,
    types_pb2.DT_UINT32: np.uint32,
    types_pb2.DT_UINT64: np.uint64,
    types_pb2.DT_COMPLEX64: np.complex64,
    types_pb2.DT_COMPLEX128: np.complex128,
    types_pb2.DT_BOOL: np.bool_,
}


class TargetTest(test_util.TensorFlowTestCase):

  def testInvalidDirectory(self):
    logdir = '/tmp/apath/that/doesnt/exist'
    self.assertFalse(gfile.Exists(logdir))
    with self.assertRaises(errors.NotFoundError):
      summary_ops.create_file_writer(logdir, max_queue=0, name='t0')

  def testShouldRecordSummary(self):
    self.assertFalse(summary_ops.should_record_summaries())
    with summary_ops.always_record_summaries():
      self.assertTrue(summary_ops.should_record_summaries())

  def testSummaryOps(self):
    training_util.get_or_create_global_step()
    logdir = tempfile.mkdtemp()
    with summary_ops.create_file_writer(
        logdir, max_queue=0,
        name='t0').as_default(), summary_ops.always_record_summaries():
      summary_ops.generic('tensor', 1, '')
      summary_ops.scalar('scalar', 2.0)
      summary_ops.histogram('histogram', [1.0])
      summary_ops.image('image', [[[[1.0]]]])
      summary_ops.audio('audio', [[1.0]], 1.0, 1)
      # The working condition of the ops is tested in the C++ test so we just
      # test here that we're calling them correctly.
      self.assertTrue(gfile.Exists(logdir))

  def testDefunSummarys(self):
    training_util.get_or_create_global_step()
    logdir = tempfile.mkdtemp()
    with summary_ops.create_file_writer(
        logdir, max_queue=0,
        name='t1').as_default(), summary_ops.always_record_summaries():

      @function.defun
      def write():
        summary_ops.scalar('scalar', 2.0)

      write()
      events = summary_test_util.events_from_logdir(logdir)
      self.assertEqual(len(events), 2)
      self.assertEqual(events[1].summary.value[0].simple_value, 2.0)

  def testSummaryName(self):
    training_util.get_or_create_global_step()
    logdir = tempfile.mkdtemp()
    with summary_ops.create_file_writer(
        logdir, max_queue=0,
        name='t2').as_default(), summary_ops.always_record_summaries():

      summary_ops.scalar('scalar', 2.0)

      events = summary_test_util.events_from_logdir(logdir)
      self.assertEqual(len(events), 2)
      self.assertEqual(events[1].summary.value[0].tag, 'scalar')

  def testSummaryGlobalStep(self):
    step = training_util.get_or_create_global_step()
    logdir = tempfile.mkdtemp()
    with summary_ops.create_file_writer(
        logdir, max_queue=0,
        name='t2').as_default(), summary_ops.always_record_summaries():

      summary_ops.scalar('scalar', 2.0, step=step)

      events = summary_test_util.events_from_logdir(logdir)
      self.assertEqual(len(events), 2)
      self.assertEqual(events[1].summary.value[0].tag, 'scalar')

  def testMaxQueue(self):
    logs = tempfile.mkdtemp()
    with summary_ops.create_file_writer(
        logs, max_queue=2, flush_millis=999999,
        name='lol').as_default(), summary_ops.always_record_summaries():
      get_total = lambda: len(summary_test_util.events_from_logdir(logs))
      # Note: First tf.Event is always file_version.
      self.assertEqual(1, get_total())
      summary_ops.scalar('scalar', 2.0, step=1)
      self.assertEqual(1, get_total())
      summary_ops.scalar('scalar', 2.0, step=2)
      self.assertEqual(3, get_total())

  def testFlush(self):
    logs = tempfile.mkdtemp()
    with summary_ops.create_file_writer(
        logs, max_queue=999999, flush_millis=999999,
        name='lol').as_default(), summary_ops.always_record_summaries():
      get_total = lambda: len(summary_test_util.events_from_logdir(logs))
      # Note: First tf.Event is always file_version.
      self.assertEqual(1, get_total())
      summary_ops.scalar('scalar', 2.0, step=1)
      summary_ops.scalar('scalar', 2.0, step=2)
      self.assertEqual(1, get_total())
      summary_ops.flush()
      self.assertEqual(3, get_total())


class DbTest(summary_test_util.SummaryDbTest):

  def testIntegerSummaries(self):
    step = training_util.create_global_step()
    writer = self.create_db_writer()

    def adder(x, y):
      state_ops.assign_add(step, 1)
      summary_ops.generic('x', x)
      summary_ops.generic('y', y)
      sum_ = x + y
      summary_ops.generic('sum', sum_)
      return sum_

    with summary_ops.always_record_summaries():
      with writer.as_default():
        self.assertEqual(5, adder(int64(2), int64(3)).numpy())

    six.assertCountEqual(
        self, [1, 1, 1],
        get_all(self.db, 'SELECT step FROM Tensors WHERE dtype IS NOT NULL'))
    six.assertCountEqual(self, ['x', 'y', 'sum'],
                         get_all(self.db, 'SELECT tag_name FROM Tags'))
    x_id = get_one(self.db, 'SELECT tag_id FROM Tags WHERE tag_name = "x"')
    y_id = get_one(self.db, 'SELECT tag_id FROM Tags WHERE tag_name = "y"')
    sum_id = get_one(self.db, 'SELECT tag_id FROM Tags WHERE tag_name = "sum"')

    with summary_ops.always_record_summaries():
      with writer.as_default():
        self.assertEqual(9, adder(int64(4), int64(5)).numpy())

    six.assertCountEqual(
        self, [1, 1, 1, 2, 2, 2],
        get_all(self.db, 'SELECT step FROM Tensors WHERE dtype IS NOT NULL'))
    six.assertCountEqual(self, [x_id, y_id, sum_id],
                         get_all(self.db, 'SELECT tag_id FROM Tags'))
    self.assertEqual(2, get_tensor(self.db, x_id, 1))
    self.assertEqual(3, get_tensor(self.db, y_id, 1))
    self.assertEqual(5, get_tensor(self.db, sum_id, 1))
    self.assertEqual(4, get_tensor(self.db, x_id, 2))
    self.assertEqual(5, get_tensor(self.db, y_id, 2))
    self.assertEqual(9, get_tensor(self.db, sum_id, 2))
    six.assertCountEqual(
        self, ['experiment'],
        get_all(self.db, 'SELECT experiment_name FROM Experiments'))
    six.assertCountEqual(self, ['run'],
                         get_all(self.db, 'SELECT run_name FROM Runs'))
    six.assertCountEqual(self, ['user'],
                         get_all(self.db, 'SELECT user_name FROM Users'))

  def testBadExperimentName(self):
    with self.assertRaises(ValueError):
      self.create_db_writer(experiment_name='\0')

  def testBadRunName(self):
    with self.assertRaises(ValueError):
      self.create_db_writer(run_name='\0')

  def testBadUserName(self):
    with self.assertRaises(ValueError):
      self.create_db_writer(user_name='-hi')
    with self.assertRaises(ValueError):
      self.create_db_writer(user_name='hi-')
    with self.assertRaises(ValueError):
      self.create_db_writer(user_name='@')

  def testGraphSummary(self):
    training_util.get_or_create_global_step()
    name = 'hi'
    graph = graph_pb2.GraphDef(node=(node_def_pb2.NodeDef(name=name),))
    with summary_ops.always_record_summaries():
      with self.create_db_writer().as_default():
        summary_ops.graph(graph)
    six.assertCountEqual(self, [name],
                         get_all(self.db, 'SELECT node_name FROM Nodes'))


def get_tensor(db, tag_id, step):
  cursor = db.execute(
      'SELECT dtype, shape, data FROM Tensors WHERE series = ? AND step = ?',
      (tag_id, step))
  dtype, shape, data = cursor.fetchone()
  assert dtype in _NUMPY_NUMERIC_TYPES
  buf = np.frombuffer(data, dtype=_NUMPY_NUMERIC_TYPES[dtype])
  if not shape:
    return buf[0]
  return buf.reshape([int(i) for i in shape.split(',')])


def int64(x):
  return array_ops.constant(x, dtypes.int64)


if __name__ == '__main__':
  test.main()
