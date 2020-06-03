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
import unittest

import six

from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.core.framework import step_stats_pb2
from tensorflow.core.framework import summary_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.util import event_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import test_util
from tensorflow.python.keras.engine.sequential import Sequential
from tensorflow.python.keras.engine.training import Model
from tensorflow.python.keras.layers.core import Activation
from tensorflow.python.keras.layers.core import Dense
from tensorflow.python.lib.io import tf_record
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import summary_ops_v2 as summary_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging


class SummaryOpsCoreTest(test_util.TensorFlowTestCase):

  def testWrite(self):
    logdir = self.get_temp_dir()
    with context.eager_mode():
      with summary_ops.create_file_writer_v2(logdir).as_default():
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
    with context.eager_mode():
      writer = summary_ops.create_file_writer_v2(logdir)
      @def_function.function
      def f():
        with writer.as_default():
          return summary_ops.write('tag', 42, step=12)
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
      with summary_ops.create_file_writer_v2(logdir).as_default():
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
      with summary_ops.create_file_writer_v2(logdir).as_default():
        summary_ops.write('tag', [[1, 2], [3, 4]], step=12)
    events = events_from_logdir(logdir)
    value = events[1].summary.value[0]
    self.assertAllEqual([[1, 2], [3, 4]], to_numpy(value))

  def testWrite_tensor(self):
    logdir = self.get_temp_dir()
    with context.eager_mode():
      t = constant_op.constant([[1, 2], [3, 4]])
      with summary_ops.create_file_writer_v2(logdir).as_default():
        summary_ops.write('tag', t, step=12)
      expected = t.numpy()
    events = events_from_logdir(logdir)
    value = events[1].summary.value[0]
    self.assertAllEqual(expected, to_numpy(value))

  def testWrite_tensor_fromFunction(self):
    logdir = self.get_temp_dir()
    with context.eager_mode():
      writer = summary_ops.create_file_writer_v2(logdir)
      @def_function.function
      def f(t):
        with writer.as_default():
          summary_ops.write('tag', t, step=12)
      t = constant_op.constant([[1, 2], [3, 4]])
      f(t)
      expected = t.numpy()
    events = events_from_logdir(logdir)
    value = events[1].summary.value[0]
    self.assertAllEqual(expected, to_numpy(value))

  def testWrite_stringTensor(self):
    logdir = self.get_temp_dir()
    with context.eager_mode():
      with summary_ops.create_file_writer_v2(logdir).as_default():
        summary_ops.write('tag', [b'foo', b'bar'], step=12)
    events = events_from_logdir(logdir)
    value = events[1].summary.value[0]
    self.assertAllEqual([b'foo', b'bar'], to_numpy(value))

  @test_util.run_gpu_only
  def testWrite_gpuDeviceContext(self):
    logdir = self.get_temp_dir()
    with context.eager_mode():
      with summary_ops.create_file_writer(logdir).as_default():
        with ops.device('/GPU:0'):
          value = constant_op.constant(42.0)
          step = constant_op.constant(12, dtype=dtypes.int64)
          summary_ops.write('tag', value, step=step).numpy()
    empty_metadata = summary_pb2.SummaryMetadata()
    events = events_from_logdir(logdir)
    self.assertEqual(2, len(events))
    self.assertEqual(12, events[1].step)
    self.assertEqual(42, to_numpy(events[1].summary.value[0]))
    self.assertEqual(empty_metadata, events[1].summary.value[0].metadata)

  @test_util.also_run_as_tf_function
  def testWrite_noDefaultWriter(self):
    # Use assertAllEqual instead of assertFalse since it works in a defun.
    self.assertAllEqual(False, summary_ops.write('tag', 42, step=0))

  @test_util.also_run_as_tf_function
  def testWrite_noStep_okayIfAlsoNoDefaultWriter(self):
    # Use assertAllEqual instead of assertFalse since it works in a defun.
    self.assertAllEqual(False, summary_ops.write('tag', 42))

  @test_util.also_run_as_tf_function
  def testWrite_noStep(self):
    logdir = self.get_temp_dir()
    with summary_ops.create_file_writer(logdir).as_default():
      with self.assertRaisesRegex(ValueError, 'No step set'):
        summary_ops.write('tag', 42)

  def testWrite_usingDefaultStep(self):
    logdir = self.get_temp_dir()
    try:
      with context.eager_mode():
        with summary_ops.create_file_writer(logdir).as_default():
          summary_ops.set_step(1)
          summary_ops.write('tag', 1.0)
          summary_ops.set_step(2)
          summary_ops.write('tag', 1.0)
          mystep = variables.Variable(10, dtype=dtypes.int64)
          summary_ops.set_step(mystep)
          summary_ops.write('tag', 1.0)
          mystep.assign_add(1)
          summary_ops.write('tag', 1.0)
      events = events_from_logdir(logdir)
      self.assertEqual(5, len(events))
      self.assertEqual(1, events[1].step)
      self.assertEqual(2, events[2].step)
      self.assertEqual(10, events[3].step)
      self.assertEqual(11, events[4].step)
    finally:
      # Reset to default state for other tests.
      summary_ops.set_step(None)

  def testWrite_usingDefaultStepConstant_fromFunction(self):
    logdir = self.get_temp_dir()
    try:
      with context.eager_mode():
        writer = summary_ops.create_file_writer(logdir)
        @def_function.function
        def f():
          with writer.as_default():
            summary_ops.write('tag', 1.0)
        summary_ops.set_step(1)
        f()
        summary_ops.set_step(2)
        f()
      events = events_from_logdir(logdir)
      self.assertEqual(3, len(events))
      self.assertEqual(1, events[1].step)
      # The step value will still be 1 because the value was captured at the
      # time the function was first traced.
      self.assertEqual(1, events[2].step)
    finally:
      # Reset to default state for other tests.
      summary_ops.set_step(None)

  def testWrite_usingDefaultStepVariable_fromFunction(self):
    logdir = self.get_temp_dir()
    try:
      with context.eager_mode():
        writer = summary_ops.create_file_writer(logdir)
        @def_function.function
        def f():
          with writer.as_default():
            summary_ops.write('tag', 1.0)
        mystep = variables.Variable(0, dtype=dtypes.int64)
        summary_ops.set_step(mystep)
        f()
        mystep.assign_add(1)
        f()
        mystep.assign(10)
        f()
      events = events_from_logdir(logdir)
      self.assertEqual(4, len(events))
      self.assertEqual(0, events[1].step)
      self.assertEqual(1, events[2].step)
      self.assertEqual(10, events[3].step)
    finally:
      # Reset to default state for other tests.
      summary_ops.set_step(None)

  def testWrite_usingDefaultStepConstant_fromLegacyGraph(self):
    logdir = self.get_temp_dir()
    try:
      with context.graph_mode():
        writer = summary_ops.create_file_writer(logdir)
        summary_ops.set_step(1)
        with writer.as_default():
          write_op = summary_ops.write('tag', 1.0)
        summary_ops.set_step(2)
        with self.cached_session() as sess:
          sess.run(writer.init())
          sess.run(write_op)
          sess.run(write_op)
          sess.run(writer.flush())
      events = events_from_logdir(logdir)
      self.assertEqual(3, len(events))
      self.assertEqual(1, events[1].step)
      # The step value will still be 1 because the value was captured at the
      # time the graph was constructed.
      self.assertEqual(1, events[2].step)
    finally:
      # Reset to default state for other tests.
      summary_ops.set_step(None)

  def testWrite_usingDefaultStepVariable_fromLegacyGraph(self):
    logdir = self.get_temp_dir()
    try:
      with context.graph_mode():
        writer = summary_ops.create_file_writer(logdir)
        mystep = variables.Variable(0, dtype=dtypes.int64)
        summary_ops.set_step(mystep)
        with writer.as_default():
          write_op = summary_ops.write('tag', 1.0)
        first_assign_op = mystep.assign_add(1)
        second_assign_op = mystep.assign(10)
        with self.cached_session() as sess:
          sess.run(writer.init())
          sess.run(mystep.initializer)
          sess.run(write_op)
          sess.run(first_assign_op)
          sess.run(write_op)
          sess.run(second_assign_op)
          sess.run(write_op)
          sess.run(writer.flush())
      events = events_from_logdir(logdir)
      self.assertEqual(4, len(events))
      self.assertEqual(0, events[1].step)
      self.assertEqual(1, events[2].step)
      self.assertEqual(10, events[3].step)
    finally:
      # Reset to default state for other tests.
      summary_ops.set_step(None)

  def testWrite_usingDefaultStep_fromAsDefault(self):
    logdir = self.get_temp_dir()
    try:
      with context.eager_mode():
        writer = summary_ops.create_file_writer(logdir)
        with writer.as_default(step=1):
          summary_ops.write('tag', 1.0)
          with writer.as_default():
            summary_ops.write('tag', 1.0)
            with writer.as_default(step=2):
              summary_ops.write('tag', 1.0)
            summary_ops.write('tag', 1.0)
            summary_ops.set_step(3)
          summary_ops.write('tag', 1.0)
      events = events_from_logdir(logdir)
      self.assertListEqual([1, 1, 2, 1, 3], [e.step for e in events[1:]])
    finally:
      # Reset to default state for other tests.
      summary_ops.set_step(None)

  def testWrite_usingDefaultStepVariable_fromAsDefault(self):
    logdir = self.get_temp_dir()
    try:
      with context.eager_mode():
        writer = summary_ops.create_file_writer(logdir)
        mystep = variables.Variable(1, dtype=dtypes.int64)
        with writer.as_default(step=mystep):
          summary_ops.write('tag', 1.0)
          with writer.as_default():
            mystep.assign(2)
            summary_ops.write('tag', 1.0)
            with writer.as_default(step=3):
              summary_ops.write('tag', 1.0)
            summary_ops.write('tag', 1.0)
            mystep.assign(4)
          summary_ops.write('tag', 1.0)
      events = events_from_logdir(logdir)
      self.assertListEqual([1, 2, 3, 2, 4], [e.step for e in events[1:]])
    finally:
      # Reset to default state for other tests.
      summary_ops.set_step(None)

  def testWrite_usingDefaultStep_fromSetAsDefault(self):
    logdir = self.get_temp_dir()
    try:
      with context.eager_mode():
        writer = summary_ops.create_file_writer(logdir)
        mystep = variables.Variable(1, dtype=dtypes.int64)
        writer.set_as_default(step=mystep)
        summary_ops.write('tag', 1.0)
        mystep.assign(2)
        summary_ops.write('tag', 1.0)
        writer.set_as_default(step=3)
        summary_ops.write('tag', 1.0)
        writer.flush()
      events = events_from_logdir(logdir)
      self.assertListEqual([1, 2, 3], [e.step for e in events[1:]])
    finally:
      # Reset to default state for other tests.
      summary_ops.set_step(None)

  def testWrite_usingDefaultStepVariable_fromSetAsDefault(self):
    logdir = self.get_temp_dir()
    try:
      with context.eager_mode():
        writer = summary_ops.create_file_writer(logdir)
        writer.set_as_default(step=1)
        summary_ops.write('tag', 1.0)
        writer.set_as_default(step=2)
        summary_ops.write('tag', 1.0)
        writer.set_as_default()
        summary_ops.write('tag', 1.0)
        writer.flush()
      events = events_from_logdir(logdir)
      self.assertListEqual([1, 2, 2], [e.step for e in events[1:]])
    finally:
      # Reset to default state for other tests.
      summary_ops.set_step(None)

  def testWrite_recordIf_constant(self):
    logdir = self.get_temp_dir()
    with context.eager_mode():
      with summary_ops.create_file_writer_v2(logdir).as_default():
        self.assertTrue(summary_ops.write('default', 1, step=0))
        with summary_ops.record_if(True):
          self.assertTrue(summary_ops.write('set_on', 1, step=0))
        with summary_ops.record_if(False):
          self.assertFalse(summary_ops.write('set_off', 1, step=0))
    events = events_from_logdir(logdir)
    self.assertEqual(3, len(events))
    self.assertEqual('default', events[1].summary.value[0].tag)
    self.assertEqual('set_on', events[2].summary.value[0].tag)

  def testWrite_recordIf_constant_fromFunction(self):
    logdir = self.get_temp_dir()
    with context.eager_mode():
      writer = summary_ops.create_file_writer_v2(logdir)
      @def_function.function
      def f():
        with writer.as_default():
          # Use assertAllEqual instead of assertTrue since it works in a defun.
          self.assertAllEqual(summary_ops.write('default', 1, step=0), True)
          with summary_ops.record_if(True):
            self.assertAllEqual(summary_ops.write('set_on', 1, step=0), True)
          with summary_ops.record_if(False):
            self.assertAllEqual(summary_ops.write('set_off', 1, step=0), False)
      f()
    events = events_from_logdir(logdir)
    self.assertEqual(3, len(events))
    self.assertEqual('default', events[1].summary.value[0].tag)
    self.assertEqual('set_on', events[2].summary.value[0].tag)

  def testWrite_recordIf_callable(self):
    logdir = self.get_temp_dir()
    with context.eager_mode():
      step = variables.Variable(-1, dtype=dtypes.int64)
      def record_fn():
        step.assign_add(1)
        return int(step % 2) == 0
      with summary_ops.create_file_writer_v2(logdir).as_default():
        with summary_ops.record_if(record_fn):
          self.assertTrue(summary_ops.write('tag', 1, step=step))
          self.assertFalse(summary_ops.write('tag', 1, step=step))
          self.assertTrue(summary_ops.write('tag', 1, step=step))
          self.assertFalse(summary_ops.write('tag', 1, step=step))
          self.assertTrue(summary_ops.write('tag', 1, step=step))
    events = events_from_logdir(logdir)
    self.assertEqual(4, len(events))
    self.assertEqual(0, events[1].step)
    self.assertEqual(2, events[2].step)
    self.assertEqual(4, events[3].step)

  def testWrite_recordIf_callable_fromFunction(self):
    logdir = self.get_temp_dir()
    with context.eager_mode():
      writer = summary_ops.create_file_writer_v2(logdir)
      step = variables.Variable(-1, dtype=dtypes.int64)
      @def_function.function
      def record_fn():
        step.assign_add(1)
        return math_ops.equal(step % 2, 0)
      @def_function.function
      def f():
        with writer.as_default():
          with summary_ops.record_if(record_fn):
            return [
                summary_ops.write('tag', 1, step=step),
                summary_ops.write('tag', 1, step=step),
                summary_ops.write('tag', 1, step=step)]
      self.assertAllEqual(f(), [True, False, True])
      self.assertAllEqual(f(), [False, True, False])
    events = events_from_logdir(logdir)
    self.assertEqual(4, len(events))
    self.assertEqual(0, events[1].step)
    self.assertEqual(2, events[2].step)
    self.assertEqual(4, events[3].step)

  def testWrite_recordIf_tensorInput_fromFunction(self):
    logdir = self.get_temp_dir()
    with context.eager_mode():
      writer = summary_ops.create_file_writer_v2(logdir)
      @def_function.function(input_signature=[
          tensor_spec.TensorSpec(shape=[], dtype=dtypes.int64)])
      def f(step):
        with writer.as_default():
          with summary_ops.record_if(math_ops.equal(step % 2, 0)):
            return summary_ops.write('tag', 1, step=step)
      self.assertTrue(f(0))
      self.assertFalse(f(1))
      self.assertTrue(f(2))
      self.assertFalse(f(3))
      self.assertTrue(f(4))
    events = events_from_logdir(logdir)
    self.assertEqual(4, len(events))
    self.assertEqual(0, events[1].step)
    self.assertEqual(2, events[2].step)
    self.assertEqual(4, events[3].step)

  def testWriteRawPb(self):
    logdir = self.get_temp_dir()
    pb = summary_pb2.Summary()
    pb.value.add().simple_value = 42.0
    with context.eager_mode():
      with summary_ops.create_file_writer_v2(logdir).as_default():
        output = summary_ops.write_raw_pb(pb.SerializeToString(), step=12)
        self.assertTrue(output.numpy())
    events = events_from_logdir(logdir)
    self.assertEqual(2, len(events))
    self.assertEqual(12, events[1].step)
    self.assertProtoEquals(pb, events[1].summary)

  def testWriteRawPb_fromFunction(self):
    logdir = self.get_temp_dir()
    pb = summary_pb2.Summary()
    pb.value.add().simple_value = 42.0
    with context.eager_mode():
      writer = summary_ops.create_file_writer_v2(logdir)
      @def_function.function
      def f():
        with writer.as_default():
          return summary_ops.write_raw_pb(pb.SerializeToString(), step=12)
      output = f()
      self.assertTrue(output.numpy())
    events = events_from_logdir(logdir)
    self.assertEqual(2, len(events))
    self.assertEqual(12, events[1].step)
    self.assertProtoEquals(pb, events[1].summary)

  def testWriteRawPb_multipleValues(self):
    logdir = self.get_temp_dir()
    pb1 = summary_pb2.Summary()
    pb1.value.add().simple_value = 1.0
    pb1.value.add().simple_value = 2.0
    pb2 = summary_pb2.Summary()
    pb2.value.add().simple_value = 3.0
    pb3 = summary_pb2.Summary()
    pb3.value.add().simple_value = 4.0
    pb3.value.add().simple_value = 5.0
    pb3.value.add().simple_value = 6.0
    pbs = [pb.SerializeToString() for pb in (pb1, pb2, pb3)]
    with context.eager_mode():
      with summary_ops.create_file_writer_v2(logdir).as_default():
        output = summary_ops.write_raw_pb(pbs, step=12)
        self.assertTrue(output.numpy())
    events = events_from_logdir(logdir)
    self.assertEqual(2, len(events))
    self.assertEqual(12, events[1].step)
    expected_pb = summary_pb2.Summary()
    for i in range(6):
      expected_pb.value.add().simple_value = i + 1.0
    self.assertProtoEquals(expected_pb, events[1].summary)

  def testWriteRawPb_invalidValue(self):
    logdir = self.get_temp_dir()
    with context.eager_mode():
      with summary_ops.create_file_writer_v2(logdir).as_default():
        with self.assertRaisesRegex(
            errors.DataLossError,
            'Bad tf.compat.v1.Summary binary proto tensor string'):
          summary_ops.write_raw_pb('notaproto', step=12)

  @test_util.also_run_as_tf_function
  def testGetSetStep(self):
    try:
      self.assertIsNone(summary_ops.get_step())
      summary_ops.set_step(1)
      # Use assertAllEqual instead of assertEqual since it works in a defun.
      self.assertAllEqual(1, summary_ops.get_step())
      summary_ops.set_step(constant_op.constant(2))
      self.assertAllEqual(2, summary_ops.get_step())
    finally:
      # Reset to default state for other tests.
      summary_ops.set_step(None)

  def testGetSetStep_variable(self):
    with context.eager_mode():
      try:
        mystep = variables.Variable(0)
        summary_ops.set_step(mystep)
        self.assertAllEqual(0, summary_ops.get_step().read_value())
        mystep.assign_add(1)
        self.assertAllEqual(1, summary_ops.get_step().read_value())
        # Check that set_step() properly maintains reference to variable.
        del mystep
        self.assertAllEqual(1, summary_ops.get_step().read_value())
        summary_ops.get_step().assign_add(1)
        self.assertAllEqual(2, summary_ops.get_step().read_value())
      finally:
        # Reset to default state for other tests.
        summary_ops.set_step(None)

  def testGetSetStep_variable_fromFunction(self):
    with context.eager_mode():
      try:
        @def_function.function
        def set_step(step):
          summary_ops.set_step(step)
          return summary_ops.get_step()
        @def_function.function
        def get_and_increment():
          summary_ops.get_step().assign_add(1)
          return summary_ops.get_step()
        mystep = variables.Variable(0)
        self.assertAllEqual(0, set_step(mystep))
        self.assertAllEqual(0, summary_ops.get_step().read_value())
        self.assertAllEqual(1, get_and_increment())
        self.assertAllEqual(2, get_and_increment())
        # Check that set_step() properly maintains reference to variable.
        del mystep
        self.assertAllEqual(3, get_and_increment())
      finally:
        # Reset to default state for other tests.
        summary_ops.set_step(None)

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
      with ops.name_scope(None, skip_on_eager=False):
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
    with ops.name_scope('with', skip_on_eager=False):
      constant_op.constant(0, name='slash')
    with summary_ops.summary_scope('with/slash') as (tag, _):
      self.assertEqual('with/slash', tag)

  def testAllV2SummaryOps(self):
    logdir = self.get_temp_dir()
    def define_ops():
      result = []
      # TF 2.0 summary ops
      result.append(summary_ops.write('write', 1, step=0))
      result.append(summary_ops.write_raw_pb(b'', step=0, name='raw_pb'))
      # TF 1.x tf.contrib.summary ops
      result.append(summary_ops.generic('tensor', 1, step=1))
      result.append(summary_ops.scalar('scalar', 2.0, step=1))
      result.append(summary_ops.histogram('histogram', [1.0], step=1))
      result.append(summary_ops.image('image', [[[[1.0]]]], step=1))
      result.append(summary_ops.audio('audio', [[1.0]], 1.0, 1, step=1))
      return result
    with context.graph_mode():
      ops_without_writer = define_ops()
      with summary_ops.create_file_writer_v2(logdir).as_default():
        with summary_ops.record_if(True):
          ops_recording_on = define_ops()
        with summary_ops.record_if(False):
          ops_recording_off = define_ops()
      # We should be collecting all ops defined with a default writer present,
      # regardless of whether recording was set on or off, but not those defined
      # without a writer at all.
      del ops_without_writer
      expected_ops = ops_recording_on + ops_recording_off
      self.assertCountEqual(expected_ops, summary_ops.all_v2_summary_ops())


class SummaryWriterTest(test_util.TensorFlowTestCase):

  def testCreate_withInitAndClose(self):
    logdir = self.get_temp_dir()
    with context.eager_mode():
      writer = summary_ops.create_file_writer_v2(
          logdir, max_queue=1000, flush_millis=1000000)
      get_total = lambda: len(events_from_logdir(logdir))
      self.assertEqual(1, get_total())  # file_version Event
      # Calling init() again while writer is open has no effect
      writer.init()
      self.assertEqual(1, get_total())
      with writer.as_default():
        summary_ops.write('tag', 1, step=0)
        self.assertEqual(1, get_total())
        # Calling .close() should do an implicit flush
        writer.close()
        self.assertEqual(2, get_total())

  def testCreate_fromFunction(self):
    logdir = self.get_temp_dir()
    @def_function.function
    def f():
      # Returned SummaryWriter must be stored in a non-local variable so it
      # lives throughout the function execution.
      if not hasattr(f, 'writer'):
        f.writer = summary_ops.create_file_writer_v2(logdir)
    with context.eager_mode():
      f()
    event_files = gfile.Glob(os.path.join(logdir, '*'))
    self.assertEqual(1, len(event_files))

  def testCreate_graphTensorArgument_raisesError(self):
    logdir = self.get_temp_dir()
    with context.graph_mode():
      logdir_tensor = constant_op.constant(logdir)
    with context.eager_mode():
      with self.assertRaisesRegex(
          ValueError, 'Invalid graph Tensor argument.*logdir'):
        summary_ops.create_file_writer_v2(logdir_tensor)
    self.assertEmpty(gfile.Glob(os.path.join(logdir, '*')))

  def testCreate_fromFunction_graphTensorArgument_raisesError(self):
    logdir = self.get_temp_dir()
    @def_function.function
    def f():
      summary_ops.create_file_writer_v2(constant_op.constant(logdir))
    with context.eager_mode():
      with self.assertRaisesRegex(
          ValueError, 'Invalid graph Tensor argument.*logdir'):
        f()
    self.assertEmpty(gfile.Glob(os.path.join(logdir, '*')))

  def testCreate_fromFunction_unpersistedResource_raisesError(self):
    logdir = self.get_temp_dir()
    @def_function.function
    def f():
      with summary_ops.create_file_writer_v2(logdir).as_default():
        pass  # Calling .as_default() is enough to indicate use.
    with context.eager_mode():
      # TODO(nickfelt): change this to a better error
      with self.assertRaisesRegex(
          errors.NotFoundError, 'Resource.*does not exist'):
        f()
    # Even though we didn't use it, an event file will have been created.
    self.assertEqual(1, len(gfile.Glob(os.path.join(logdir, '*'))))

  def testCreate_immediateSetAsDefault_retainsReference(self):
    logdir = self.get_temp_dir()
    try:
      with context.eager_mode():
        summary_ops.create_file_writer_v2(logdir).set_as_default()
        summary_ops.flush()
    finally:
      # Ensure we clean up no matter how the test executes.
      summary_ops._summary_state.writer = None  # pylint: disable=protected-access

  def testCreate_immediateAsDefault_retainsReference(self):
    logdir = self.get_temp_dir()
    with context.eager_mode():
      with summary_ops.create_file_writer_v2(logdir).as_default():
        summary_ops.flush()

  def testNoSharing(self):
    # Two writers with the same logdir should not share state.
    logdir = self.get_temp_dir()
    with context.eager_mode():
      writer1 = summary_ops.create_file_writer_v2(logdir)
      with writer1.as_default():
        summary_ops.write('tag', 1, step=1)
      event_files = gfile.Glob(os.path.join(logdir, '*'))
      self.assertEqual(1, len(event_files))
      file1 = event_files[0]

      writer2 = summary_ops.create_file_writer_v2(logdir)
      with writer2.as_default():
        summary_ops.write('tag', 1, step=2)
      event_files = gfile.Glob(os.path.join(logdir, '*'))
      self.assertEqual(2, len(event_files))
      event_files.remove(file1)
      file2 = event_files[0]

      # Extra writes to ensure interleaved usage works.
      with writer1.as_default():
        summary_ops.write('tag', 1, step=1)
      with writer2.as_default():
        summary_ops.write('tag', 1, step=2)

    events = iter(events_from_file(file1))
    self.assertEqual('brain.Event:2', next(events).file_version)
    self.assertEqual(1, next(events).step)
    self.assertEqual(1, next(events).step)
    self.assertRaises(StopIteration, lambda: next(events))
    events = iter(events_from_file(file2))
    self.assertEqual('brain.Event:2', next(events).file_version)
    self.assertEqual(2, next(events).step)
    self.assertEqual(2, next(events).step)
    self.assertRaises(StopIteration, lambda: next(events))

  def testNoSharing_fromFunction(self):
    logdir = self.get_temp_dir()
    @def_function.function
    def f1():
      if not hasattr(f1, 'writer'):
        f1.writer = summary_ops.create_file_writer_v2(logdir)
      with f1.writer.as_default():
        summary_ops.write('tag', 1, step=1)
    @def_function.function
    def f2():
      if not hasattr(f2, 'writer'):
        f2.writer = summary_ops.create_file_writer_v2(logdir)
      with f2.writer.as_default():
        summary_ops.write('tag', 1, step=2)
    with context.eager_mode():
      f1()
      event_files = gfile.Glob(os.path.join(logdir, '*'))
      self.assertEqual(1, len(event_files))
      file1 = event_files[0]

      f2()
      event_files = gfile.Glob(os.path.join(logdir, '*'))
      self.assertEqual(2, len(event_files))
      event_files.remove(file1)
      file2 = event_files[0]

      # Extra writes to ensure interleaved usage works.
      f1()
      f2()

    events = iter(events_from_file(file1))
    self.assertEqual('brain.Event:2', next(events).file_version)
    self.assertEqual(1, next(events).step)
    self.assertEqual(1, next(events).step)
    self.assertRaises(StopIteration, lambda: next(events))
    events = iter(events_from_file(file2))
    self.assertEqual('brain.Event:2', next(events).file_version)
    self.assertEqual(2, next(events).step)
    self.assertEqual(2, next(events).step)
    self.assertRaises(StopIteration, lambda: next(events))

  def testMaxQueue(self):
    logdir = self.get_temp_dir()
    with context.eager_mode():
      with summary_ops.create_file_writer_v2(
          logdir, max_queue=1, flush_millis=999999).as_default():
        get_total = lambda: len(events_from_logdir(logdir))
        # Note: First tf.compat.v1.Event is always file_version.
        self.assertEqual(1, get_total())
        summary_ops.write('tag', 1, step=0)
        self.assertEqual(1, get_total())
        # Should flush after second summary since max_queue = 1
        summary_ops.write('tag', 1, step=0)
        self.assertEqual(3, get_total())

  def testWriterFlush(self):
    logdir = self.get_temp_dir()
    get_total = lambda: len(events_from_logdir(logdir))
    with context.eager_mode():
      writer = summary_ops.create_file_writer_v2(
          logdir, max_queue=1000, flush_millis=1000000)
      self.assertEqual(1, get_total())  # file_version Event
      with writer.as_default():
        summary_ops.write('tag', 1, step=0)
        self.assertEqual(1, get_total())
        writer.flush()
        self.assertEqual(2, get_total())
        summary_ops.write('tag', 1, step=0)
        self.assertEqual(2, get_total())
      # Exiting the "as_default()" should do an implicit flush
      self.assertEqual(3, get_total())

  def testFlushFunction(self):
    logdir = self.get_temp_dir()
    with context.eager_mode():
      writer = summary_ops.create_file_writer_v2(
          logdir, max_queue=999999, flush_millis=999999)
      with writer.as_default():
        get_total = lambda: len(events_from_logdir(logdir))
        # Note: First tf.compat.v1.Event is always file_version.
        self.assertEqual(1, get_total())
        summary_ops.write('tag', 1, step=0)
        summary_ops.write('tag', 1, step=0)
        self.assertEqual(1, get_total())
        summary_ops.flush()
        self.assertEqual(3, get_total())
        # Test "writer" parameter
        summary_ops.write('tag', 1, step=0)
        self.assertEqual(3, get_total())
        summary_ops.flush(writer=writer)
        self.assertEqual(4, get_total())
        summary_ops.write('tag', 1, step=0)
        self.assertEqual(4, get_total())
        summary_ops.flush(writer=writer._resource)  # pylint:disable=protected-access
        self.assertEqual(5, get_total())

  @test_util.assert_no_new_tensors
  def testNoMemoryLeak_graphMode(self):
    logdir = self.get_temp_dir()
    with context.graph_mode(), ops.Graph().as_default():
      summary_ops.create_file_writer_v2(logdir)

  @test_util.assert_no_new_pyobjects_executing_eagerly
  def testNoMemoryLeak_eagerMode(self):
    logdir = self.get_temp_dir()
    with summary_ops.create_file_writer_v2(logdir).as_default():
      summary_ops.write('tag', 1, step=0)

  def testClose_preventsLaterUse(self):
    logdir = self.get_temp_dir()
    with context.eager_mode():
      writer = summary_ops.create_file_writer_v2(logdir)
      writer.close()
      writer.close()  # redundant close() is a no-op
      writer.flush()  # redundant flush() is a no-op
      with self.assertRaisesRegex(RuntimeError, 'already closed'):
        writer.init()
      with self.assertRaisesRegex(RuntimeError, 'already closed'):
        with writer.as_default():
          self.fail('should not get here')
      with self.assertRaisesRegex(RuntimeError, 'already closed'):
        writer.set_as_default()

  def testClose_closesOpenFile(self):
    try:
      import psutil  # pylint: disable=g-import-not-at-top
    except ImportError:
      raise unittest.SkipTest('test requires psutil')
    proc = psutil.Process()
    get_open_filenames = lambda: set(info[0] for info in proc.open_files())
    logdir = self.get_temp_dir()
    with context.eager_mode():
      writer = summary_ops.create_file_writer_v2(logdir)
      files = gfile.Glob(os.path.join(logdir, '*'))
      self.assertEqual(1, len(files))
      eventfile = files[0]
      self.assertIn(eventfile, get_open_filenames())
      writer.close()
      self.assertNotIn(eventfile, get_open_filenames())

  def testDereference_closesOpenFile(self):
    try:
      import psutil  # pylint: disable=g-import-not-at-top
    except ImportError:
      raise unittest.SkipTest('test requires psutil')
    proc = psutil.Process()
    get_open_filenames = lambda: set(info[0] for info in proc.open_files())
    logdir = self.get_temp_dir()
    with context.eager_mode():
      writer = summary_ops.create_file_writer_v2(logdir)
      files = gfile.Glob(os.path.join(logdir, '*'))
      self.assertEqual(1, len(files))
      eventfile = files[0]
      self.assertIn(eventfile, get_open_filenames())
      del writer
      self.assertNotIn(eventfile, get_open_filenames())


class SummaryOpsTest(test_util.TensorFlowTestCase):

  def tearDown(self):
    summary_ops.trace_off()

  def run_metadata(self, *args, **kwargs):
    assert context.executing_eagerly()
    logdir = self.get_temp_dir()
    writer = summary_ops.create_file_writer(logdir)
    with writer.as_default():
      summary_ops.run_metadata(*args, **kwargs)
    writer.close()
    events = events_from_logdir(logdir)
    return events[1]

  def run_metadata_graphs(self, *args, **kwargs):
    assert context.executing_eagerly()
    logdir = self.get_temp_dir()
    writer = summary_ops.create_file_writer(logdir)
    with writer.as_default():
      summary_ops.run_metadata_graphs(*args, **kwargs)
    writer.close()
    events = events_from_logdir(logdir)
    return events[1]

  def create_run_metadata(self):
    step_stats = step_stats_pb2.StepStats(dev_stats=[
        step_stats_pb2.DeviceStepStats(
            device='cpu:0',
            node_stats=[step_stats_pb2.NodeExecStats(node_name='hello')])
    ])
    return config_pb2.RunMetadata(
        function_graphs=[
            config_pb2.RunMetadata.FunctionGraphs(
                pre_optimization_graph=graph_pb2.GraphDef(
                    node=[node_def_pb2.NodeDef(name='foo')]))
        ],
        step_stats=step_stats)

  def keras_model(self, *args, **kwargs):
    logdir = self.get_temp_dir()
    writer = summary_ops.create_file_writer(logdir)
    with writer.as_default():
      summary_ops.keras_model(*args, **kwargs)
    writer.close()
    events = events_from_logdir(logdir)
    # The first event contains no summary values. The written content goes to
    # the second event.
    return events[1]

  def run_trace(self, f, step=1):
    assert context.executing_eagerly()
    logdir = self.get_temp_dir()
    writer = summary_ops.create_file_writer(logdir)
    summary_ops.trace_on(graph=True, profiler=False)
    with writer.as_default():
      f()
      summary_ops.trace_export(name='foo', step=step)
    writer.close()
    events = events_from_logdir(logdir)
    return events[1]

  @test_util.run_v2_only
  def testRunMetadata_usesNameAsTag(self):
    meta = config_pb2.RunMetadata()

    with ops.name_scope('foo', skip_on_eager=False):
      event = self.run_metadata(name='my_name', data=meta, step=1)
      first_val = event.summary.value[0]

    self.assertEqual('foo/my_name', first_val.tag)

  @test_util.run_v2_only
  def testRunMetadata_summaryMetadata(self):
    expected_summary_metadata = """
      plugin_data {
        plugin_name: "graph_run_metadata"
        content: "1"
      }
    """
    meta = config_pb2.RunMetadata()
    event = self.run_metadata(name='my_name', data=meta, step=1)
    actual_summary_metadata = event.summary.value[0].metadata
    self.assertProtoEquals(expected_summary_metadata, actual_summary_metadata)

  @test_util.run_v2_only
  def testRunMetadata_wholeRunMetadata(self):
    expected_run_metadata = """
      step_stats {
        dev_stats {
          device: "cpu:0"
          node_stats {
            node_name: "hello"
          }
        }
      }
      function_graphs {
        pre_optimization_graph {
          node {
            name: "foo"
          }
        }
      }
    """
    meta = self.create_run_metadata()
    event = self.run_metadata(name='my_name', data=meta, step=1)
    first_val = event.summary.value[0]

    actual_run_metadata = config_pb2.RunMetadata.FromString(
        first_val.tensor.string_val[0])
    self.assertProtoEquals(expected_run_metadata, actual_run_metadata)

  @test_util.run_v2_only
  def testRunMetadata_usesDefaultStep(self):
    meta = config_pb2.RunMetadata()
    try:
      summary_ops.set_step(42)
      event = self.run_metadata(name='my_name', data=meta)
      self.assertEqual(42, event.step)
    finally:
      # Reset to default state for other tests.
      summary_ops.set_step(None)

  @test_util.run_v2_only
  def testRunMetadataGraph_usesNameAsTag(self):
    meta = config_pb2.RunMetadata()

    with ops.name_scope('foo', skip_on_eager=False):
      event = self.run_metadata_graphs(name='my_name', data=meta, step=1)
      first_val = event.summary.value[0]

    self.assertEqual('foo/my_name', first_val.tag)

  @test_util.run_v2_only
  def testRunMetadataGraph_summaryMetadata(self):
    expected_summary_metadata = """
      plugin_data {
        plugin_name: "graph_run_metadata_graph"
        content: "1"
      }
    """
    meta = config_pb2.RunMetadata()
    event = self.run_metadata_graphs(name='my_name', data=meta, step=1)
    actual_summary_metadata = event.summary.value[0].metadata
    self.assertProtoEquals(expected_summary_metadata, actual_summary_metadata)

  @test_util.run_v2_only
  def testRunMetadataGraph_runMetadataFragment(self):
    expected_run_metadata = """
      function_graphs {
        pre_optimization_graph {
          node {
            name: "foo"
          }
        }
      }
    """
    meta = self.create_run_metadata()

    event = self.run_metadata_graphs(name='my_name', data=meta, step=1)
    first_val = event.summary.value[0]

    actual_run_metadata = config_pb2.RunMetadata.FromString(
        first_val.tensor.string_val[0])
    self.assertProtoEquals(expected_run_metadata, actual_run_metadata)

  @test_util.run_v2_only
  def testRunMetadataGraph_usesDefaultStep(self):
    meta = config_pb2.RunMetadata()
    try:
      summary_ops.set_step(42)
      event = self.run_metadata_graphs(name='my_name', data=meta)
      self.assertEqual(42, event.step)
    finally:
      # Reset to default state for other tests.
      summary_ops.set_step(None)

  @test_util.run_v2_only
  def testKerasModel(self):
    model = Sequential(
        [Dense(10, input_shape=(100,)),
         Activation('relu', name='my_relu')])
    event = self.keras_model(name='my_name', data=model, step=1)
    first_val = event.summary.value[0]
    self.assertEqual(model.to_json(), first_val.tensor.string_val[0].decode())

  @test_util.run_v2_only
  def testKerasModel_usesDefaultStep(self):
    model = Sequential(
        [Dense(10, input_shape=(100,)),
         Activation('relu', name='my_relu')])
    try:
      summary_ops.set_step(42)
      event = self.keras_model(name='my_name', data=model)
      self.assertEqual(42, event.step)
    finally:
      # Reset to default state for other tests.
      summary_ops.set_step(None)

  @test_util.run_v2_only
  def testKerasModel_subclass(self):

    class SimpleSubclass(Model):

      def __init__(self):
        super(SimpleSubclass, self).__init__(name='subclass')
        self.dense = Dense(10, input_shape=(100,))
        self.activation = Activation('relu', name='my_relu')

      def call(self, inputs):
        x = self.dense(inputs)
        return self.activation(x)

    model = SimpleSubclass()
    with test.mock.patch.object(logging, 'warn') as mock_log:
      self.assertFalse(
          summary_ops.keras_model(name='my_name', data=model, step=1))
      self.assertRegexpMatches(
          str(mock_log.call_args), 'Model failed to serialize as JSON.')

  @test_util.run_v2_only
  def testKerasModel_otherExceptions(self):
    model = Sequential()

    with test.mock.patch.object(model, 'to_json') as mock_to_json:
      with test.mock.patch.object(logging, 'warn') as mock_log:
        mock_to_json.side_effect = Exception('oops')
        self.assertFalse(
            summary_ops.keras_model(name='my_name', data=model, step=1))
        self.assertRegexpMatches(
            str(mock_log.call_args),
            'Model failed to serialize as JSON. Ignoring... oops')

  @test_util.run_v2_only
  def testTrace(self):

    @def_function.function
    def f():
      x = constant_op.constant(2)
      y = constant_op.constant(3)
      return x**y

    event = self.run_trace(f)

    first_val = event.summary.value[0]
    actual_run_metadata = config_pb2.RunMetadata.FromString(
        first_val.tensor.string_val[0])

    # Content of function_graphs is large and, for instance, device can change.
    self.assertTrue(hasattr(actual_run_metadata, 'function_graphs'))

  @test_util.run_v2_only
  def testTrace_cannotEnableTraceInFunction(self):

    @def_function.function
    def f():
      summary_ops.trace_on(graph=True, profiler=False)
      x = constant_op.constant(2)
      y = constant_op.constant(3)
      return x**y

    with test.mock.patch.object(logging, 'warn') as mock_log:
      f()
      self.assertRegexpMatches(
          str(mock_log.call_args), 'Cannot enable trace inside a tf.function.')

  @test_util.run_v2_only
  def testTrace_cannotEnableTraceInGraphMode(self):
    with test.mock.patch.object(logging, 'warn') as mock_log:
      with context.graph_mode():
        summary_ops.trace_on(graph=True, profiler=False)
      self.assertRegexpMatches(
          str(mock_log.call_args), 'Must enable trace in eager mode.')

  @test_util.run_v2_only
  def testTrace_cannotExportTraceWithoutTrace(self):
    with six.assertRaisesRegex(self, ValueError,
                               'Must enable trace before export.'):
      summary_ops.trace_export(name='foo', step=1)

  @test_util.run_v2_only
  def testTrace_cannotExportTraceInFunction(self):
    summary_ops.trace_on(graph=True, profiler=False)

    @def_function.function
    def f():
      x = constant_op.constant(2)
      y = constant_op.constant(3)
      summary_ops.trace_export(name='foo', step=1)
      return x**y

    with test.mock.patch.object(logging, 'warn') as mock_log:
      f()
      self.assertRegexpMatches(
          str(mock_log.call_args),
          'Cannot export trace inside a tf.function.')

  @test_util.run_v2_only
  def testTrace_cannotExportTraceInGraphMode(self):
    with test.mock.patch.object(logging, 'warn') as mock_log:
      with context.graph_mode():
        summary_ops.trace_export(name='foo', step=1)
      self.assertRegexpMatches(
          str(mock_log.call_args),
          'Can only export trace while executing eagerly.')

  @test_util.run_v2_only
  def testTrace_usesDefaultStep(self):

    @def_function.function
    def f():
      x = constant_op.constant(2)
      y = constant_op.constant(3)
      return x**y

    try:
      summary_ops.set_step(42)
      event = self.run_trace(f, step=None)
      self.assertEqual(42, event.step)
    finally:
      # Reset to default state for other tests.
      summary_ops.set_step(None)


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
