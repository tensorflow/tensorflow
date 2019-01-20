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
"""Tests for V2 summary ops from summary_ops_v2."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.core.framework import summary_pb2
from tensorflow.core.util import event_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import test_util
from tensorflow.python.lib.io import tf_record
from tensorflow.python.ops import summary_ops_v2 as summary_ops
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test


class SummaryOpsTest(test_util.TensorFlowTestCase):

  def testWrite(self):
    logdir = self.get_temp_dir()
    with context.eager_mode():
      with summary_ops.create_file_writer(logdir).as_default():
        output = summary_ops.write('tag', 42, step=12)
        self.assertTrue(output.numpy())
    events = events_from_logdir(logdir)
    self.assertEqual(2, len(events))
    self.assertEqual(12, events[1].step)
    value = events[1].summary.value[0]
    self.assertEqual('tag', value.tag)
    self.assertEqual(42, to_numpy(value))

  def testWrite_fromFunction(self):
    logdir = self.get_temp_dir()
    @def_function.function
    def f():
      with summary_ops.create_file_writer(logdir).as_default():
        return summary_ops.write('tag', 42, step=12)
    with context.eager_mode():
      output = f()
      self.assertTrue(output.numpy())
    events = events_from_logdir(logdir)
    self.assertEqual(2, len(events))
    self.assertEqual(12, events[1].step)
    value = events[1].summary.value[0]
    self.assertEqual('tag', value.tag)
    self.assertEqual(42, to_numpy(value))

  def testWrite_metadata(self):
    logdir = self.get_temp_dir()
    metadata = summary_pb2.SummaryMetadata()
    metadata.plugin_data.plugin_name = 'foo'
    with context.eager_mode():
      with summary_ops.create_file_writer(logdir).as_default():
        summary_ops.write('obj', 0, 0, metadata=metadata)
        summary_ops.write('bytes', 0, 0, metadata=metadata.SerializeToString())
        m = constant_op.constant(metadata.SerializeToString())
        summary_ops.write('string_tensor', 0, 0, metadata=m)
    events = events_from_logdir(logdir)
    self.assertEqual(4, len(events))
    self.assertEqual(metadata, events[1].summary.value[0].metadata)
    self.assertEqual(metadata, events[2].summary.value[0].metadata)
    self.assertEqual(metadata, events[3].summary.value[0].metadata)

  def testWrite_name(self):
    @def_function.function
    def f():
      output = summary_ops.write('tag', 42, step=12, name='anonymous')
      self.assertTrue(output.name.startswith('anonymous'))
    f()

  def testWrite_ndarray(self):
    logdir = self.get_temp_dir()
    with context.eager_mode():
      with summary_ops.create_file_writer(logdir).as_default():
        summary_ops.write('tag', [[1, 2], [3, 4]], step=12)
    events = events_from_logdir(logdir)
    value = events[1].summary.value[0]
    self.assertAllEqual([[1, 2], [3, 4]], to_numpy(value))

  def testWrite_tensor(self):
    logdir = self.get_temp_dir()
    with context.eager_mode():
      t = constant_op.constant([[1, 2], [3, 4]])
      with summary_ops.create_file_writer(logdir).as_default():
        summary_ops.write('tag', t, step=12)
      expected = t.numpy()
    events = events_from_logdir(logdir)
    value = events[1].summary.value[0]
    self.assertAllEqual(expected, to_numpy(value))

  def testWrite_tensor_fromFunction(self):
    logdir = self.get_temp_dir()
    @def_function.function
    def f(t):
      with summary_ops.create_file_writer(logdir).as_default():
        summary_ops.write('tag', t, step=12)
    with context.eager_mode():
      t = constant_op.constant([[1, 2], [3, 4]])
      f(t)
      expected = t.numpy()
    events = events_from_logdir(logdir)
    value = events[1].summary.value[0]
    self.assertAllEqual(expected, to_numpy(value))

  def testWrite_stringTensor(self):
    logdir = self.get_temp_dir()
    with context.eager_mode():
      with summary_ops.create_file_writer(logdir).as_default():
        summary_ops.write('tag', [b'foo', b'bar'], step=12)
    events = events_from_logdir(logdir)
    value = events[1].summary.value[0]
    self.assertAllEqual([b'foo', b'bar'], to_numpy(value))

  @test_util.also_run_as_tf_function
  def testWrite_noDefaultWriter(self):
    with context.eager_mode():
      self.assertFalse(summary_ops.write('tag', 42, step=0))

  def testWrite_shouldRecordSummaries(self):
    logdir = self.get_temp_dir()
    with context.eager_mode():
      with summary_ops.create_file_writer(logdir).as_default():
        self.assertTrue(summary_ops.write('default_on', 1, step=0))
        with summary_ops.always_record_summaries():
          self.assertTrue(summary_ops.write('set_on', 1, step=0))
        with summary_ops.never_record_summaries():
          self.assertFalse(summary_ops.write('set_off', 1, step=0))
    events = events_from_logdir(logdir)
    self.assertEqual(3, len(events))
    self.assertEqual('default_on', events[1].summary.value[0].tag)
    self.assertEqual('set_on', events[2].summary.value[0].tag)

  def testWrite_shouldRecordSummaries_fromFunction(self):
    logdir = self.get_temp_dir()
    @def_function.function
    def f(tag_prefix):
      with summary_ops.create_file_writer(logdir).as_default():
        default_output = summary_ops.write(tag_prefix + '_default', 1, step=0)
        with summary_ops.always_record_summaries():
          on_output = summary_ops.write(tag_prefix + '_on', 1, step=0)
        with summary_ops.never_record_summaries():
          off_output = summary_ops.write(tag_prefix + '_off', 1, step=0)
        return [default_output, on_output, off_output]
    with context.eager_mode():
      self.assertAllEqual([True, True, False], f('default'))
      with summary_ops.always_record_summaries():
        self.assertAllEqual([True, True, False], f('on'))
      with summary_ops.never_record_summaries():
        self.assertAllEqual([False, True, False], f('off'))
    events = events_from_logdir(logdir)
    self.assertEqual(6, len(events))
    self.assertEqual('default_default', events[1].summary.value[0].tag)
    self.assertEqual('default_on', events[2].summary.value[0].tag)
    self.assertEqual('on_default', events[3].summary.value[0].tag)
    self.assertEqual('on_on', events[4].summary.value[0].tag)
    self.assertEqual('off_on', events[5].summary.value[0].tag)

  @test_util.also_run_as_tf_function
  def testSummaryScope(self):
    with summary_ops.summary_scope('foo') as (tag, scope):
      self.assertEqual('foo', tag)
      self.assertEqual('foo/', scope)
      with summary_ops.summary_scope('bar') as (tag, scope):
        self.assertEqual('foo/bar', tag)
        self.assertEqual('foo/bar/', scope)
      with summary_ops.summary_scope('with/slash') as (tag, scope):
        self.assertEqual('foo/with/slash', tag)
        self.assertEqual('foo/with/slash/', scope)
      with ops.name_scope(None):
        with summary_ops.summary_scope('unnested') as (tag, scope):
          self.assertEqual('unnested', tag)
          self.assertEqual('unnested/', scope)

  @test_util.also_run_as_tf_function
  def testSummaryScope_defaultName(self):
    with summary_ops.summary_scope(None) as (tag, scope):
      self.assertEqual('summary', tag)
      self.assertEqual('summary/', scope)
    with summary_ops.summary_scope(None, 'backup') as (tag, scope):
      self.assertEqual('backup', tag)
      self.assertEqual('backup/', scope)

  @test_util.also_run_as_tf_function
  def testSummaryScope_handlesCharactersIllegalForScope(self):
    with summary_ops.summary_scope('f?o?o') as (tag, scope):
      self.assertEqual('f?o?o', tag)
      self.assertEqual('foo/', scope)
    # If all characters aren't legal for a scope name, use default name.
    with summary_ops.summary_scope('???', 'backup') as (tag, scope):
      self.assertEqual('???', tag)
      self.assertEqual('backup/', scope)

  @test_util.also_run_as_tf_function
  def testSummaryScope_nameNotUniquifiedForTag(self):
    constant_op.constant(0, name='foo')
    with summary_ops.summary_scope('foo') as (tag, _):
      self.assertEqual('foo', tag)
    with summary_ops.summary_scope('foo') as (tag, _):
      self.assertEqual('foo', tag)
    with ops.name_scope('with'):
      constant_op.constant(0, name='slash')
    with summary_ops.summary_scope('with/slash') as (tag, _):
      self.assertEqual('with/slash', tag)


def events_from_file(filepath):
  """Returns all events in a single event file.

  Args:
    filepath: Path to the event file.

  Returns:
    A list of all tf.Event protos in the event file.
  """
  records = list(tf_record.tf_record_iterator(filepath))
  result = []
  for r in records:
    event = event_pb2.Event()
    event.ParseFromString(r)
    result.append(event)
  return result


def events_from_logdir(logdir):
  """Returns all events in the single eventfile in logdir.

  Args:
    logdir: The directory in which the single event file is sought.

  Returns:
    A list of all tf.Event protos from the single event file.

  Raises:
    AssertionError: If logdir does not contain exactly one file.
  """
  assert gfile.Exists(logdir)
  files = gfile.ListDirectory(logdir)
  assert len(files) == 1, 'Found not exactly one file in logdir: %s' % files
  return events_from_file(os.path.join(logdir, files[0]))


def to_numpy(summary_value):
  return tensor_util.MakeNdarray(summary_value.tensor)


if __name__ == '__main__':
  test.main()
