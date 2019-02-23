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
"""Tests for the API surface of the V1 tf.summary ops.

These tests don't check the actual serialized proto summary value for the
more complex summaries (e.g. audio, image).  Those test live separately in
tensorflow/python/kernel_tests/summary_v1_*.py.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.core.framework import summary_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.summary import summary as summary_lib


class SummaryTest(test.TestCase):

  @test_util.run_deprecated_v1
  def testScalarSummary(self):
    with self.cached_session() as s:
      i = constant_op.constant(3)
      with ops.name_scope('outer'):
        im = summary_lib.scalar('inner', i)
      summary_str = s.run(im)
    summary = summary_pb2.Summary()
    summary.ParseFromString(summary_str)
    values = summary.value
    self.assertEqual(len(values), 1)
    self.assertEqual(values[0].tag, 'outer/inner')
    self.assertEqual(values[0].simple_value, 3.0)

  @test_util.run_deprecated_v1
  def testScalarSummaryWithFamily(self):
    with self.cached_session() as s:
      i = constant_op.constant(7)
      with ops.name_scope('outer'):
        im1 = summary_lib.scalar('inner', i, family='family')
        self.assertEquals(im1.op.name, 'outer/family/inner')
        im2 = summary_lib.scalar('inner', i, family='family')
        self.assertEquals(im2.op.name, 'outer/family/inner_1')
      sm1, sm2 = s.run([im1, im2])
    summary = summary_pb2.Summary()

    summary.ParseFromString(sm1)
    values = summary.value
    self.assertEqual(len(values), 1)
    self.assertEqual(values[0].tag, 'family/outer/family/inner')
    self.assertEqual(values[0].simple_value, 7.0)

    summary.ParseFromString(sm2)
    values = summary.value
    self.assertEqual(len(values), 1)
    self.assertEqual(values[0].tag, 'family/outer/family/inner_1')
    self.assertEqual(values[0].simple_value, 7.0)

  @test_util.run_deprecated_v1
  def testSummarizingVariable(self):
    with self.cached_session() as s:
      c = constant_op.constant(42.0)
      v = variables.Variable(c)
      ss = summary_lib.scalar('summary', v)
      init = variables.global_variables_initializer()
      s.run(init)
      summ_str = s.run(ss)
    summary = summary_pb2.Summary()
    summary.ParseFromString(summ_str)
    self.assertEqual(len(summary.value), 1)
    value = summary.value[0]
    self.assertEqual(value.tag, 'summary')
    self.assertEqual(value.simple_value, 42.0)

  @test_util.run_deprecated_v1
  def testImageSummary(self):
    with self.cached_session() as s:
      i = array_ops.ones((5, 4, 4, 3))
      with ops.name_scope('outer'):
        im = summary_lib.image('inner', i, max_outputs=3)
      summary_str = s.run(im)
    summary = summary_pb2.Summary()
    summary.ParseFromString(summary_str)
    values = summary.value
    self.assertEqual(len(values), 3)
    tags = sorted(v.tag for v in values)
    expected = sorted('outer/inner/image/{}'.format(i) for i in xrange(3))
    self.assertEqual(tags, expected)

  @test_util.run_deprecated_v1
  def testImageSummaryWithFamily(self):
    with self.cached_session() as s:
      i = array_ops.ones((5, 2, 3, 1))
      with ops.name_scope('outer'):
        im = summary_lib.image('inner', i, max_outputs=3, family='family')
        self.assertEquals(im.op.name, 'outer/family/inner')
      summary_str = s.run(im)
    summary = summary_pb2.Summary()
    summary.ParseFromString(summary_str)
    values = summary.value
    self.assertEqual(len(values), 3)
    tags = sorted(v.tag for v in values)
    expected = sorted('family/outer/family/inner/image/{}'.format(i)
                      for i in xrange(3))
    self.assertEqual(tags, expected)

  @test_util.run_deprecated_v1
  def testHistogramSummary(self):
    with self.cached_session() as s:
      i = array_ops.ones((5, 4, 4, 3))
      with ops.name_scope('outer'):
        summ_op = summary_lib.histogram('inner', i)
      summary_str = s.run(summ_op)
    summary = summary_pb2.Summary()
    summary.ParseFromString(summary_str)
    self.assertEqual(len(summary.value), 1)
    self.assertEqual(summary.value[0].tag, 'outer/inner')

  @test_util.run_deprecated_v1
  def testHistogramSummaryWithFamily(self):
    with self.cached_session() as s:
      i = array_ops.ones((5, 4, 4, 3))
      with ops.name_scope('outer'):
        summ_op = summary_lib.histogram('inner', i, family='family')
        self.assertEquals(summ_op.op.name, 'outer/family/inner')
      summary_str = s.run(summ_op)
    summary = summary_pb2.Summary()
    summary.ParseFromString(summary_str)
    self.assertEqual(len(summary.value), 1)
    self.assertEqual(summary.value[0].tag, 'family/outer/family/inner')

  def testHistogramSummaryTypes(self):
    for dtype in (dtypes.int8, dtypes.uint8, dtypes.int16, dtypes.int32,
                  dtypes.float32, dtypes.float64):
      const = constant_op.constant(10, dtype=dtype)
      summary_lib.histogram('h', const)

  @test_util.run_deprecated_v1
  def testAudioSummary(self):
    with self.cached_session() as s:
      i = array_ops.ones((5, 3, 4))
      with ops.name_scope('outer'):
        aud = summary_lib.audio('inner', i, 0.2, max_outputs=3)
      summary_str = s.run(aud)
    summary = summary_pb2.Summary()
    summary.ParseFromString(summary_str)
    values = summary.value
    self.assertEqual(len(values), 3)
    tags = sorted(v.tag for v in values)
    expected = sorted('outer/inner/audio/{}'.format(i) for i in xrange(3))
    self.assertEqual(tags, expected)

  @test_util.run_deprecated_v1
  def testAudioSummaryWithFamily(self):
    with self.cached_session() as s:
      i = array_ops.ones((5, 3, 4))
      with ops.name_scope('outer'):
        aud = summary_lib.audio('inner', i, 0.2, max_outputs=3, family='family')
        self.assertEquals(aud.op.name, 'outer/family/inner')
      summary_str = s.run(aud)
    summary = summary_pb2.Summary()
    summary.ParseFromString(summary_str)
    values = summary.value
    self.assertEqual(len(values), 3)
    tags = sorted(v.tag for v in values)
    expected = sorted('family/outer/family/inner/audio/{}'.format(i)
                      for i in xrange(3))
    self.assertEqual(tags, expected)

  @test_util.run_deprecated_v1
  def testTextSummary(self):
    with self.cached_session():
      with self.assertRaises(ValueError):
        num = array_ops.constant(1)
        summary_lib.text('foo', num)

      # The API accepts vectors.
      arr = array_ops.constant(['one', 'two', 'three'])
      summ = summary_lib.text('foo', arr)
      self.assertEqual(summ.op.type, 'TensorSummaryV2')

      # the API accepts scalars
      summ = summary_lib.text('foo', array_ops.constant('one'))
      self.assertEqual(summ.op.type, 'TensorSummaryV2')

  @test_util.run_deprecated_v1
  def testSummaryNameConversion(self):
    c = constant_op.constant(3)
    s = summary_lib.scalar('name with spaces', c)
    self.assertEqual(s.op.name, 'name_with_spaces')

    s2 = summary_lib.scalar('name with many $#illegal^: characters!', c)
    self.assertEqual(s2.op.name, 'name_with_many___illegal___characters_')

    s3 = summary_lib.scalar('/name/with/leading/slash', c)
    self.assertEqual(s3.op.name, 'name/with/leading/slash')

  @test_util.run_deprecated_v1
  def testSummaryWithFamilyMetaGraphExport(self):
    with ops.name_scope('outer'):
      i = constant_op.constant(11)
      summ = summary_lib.scalar('inner', i)
      self.assertEquals(summ.op.name, 'outer/inner')
      summ_f = summary_lib.scalar('inner', i, family='family')
      self.assertEquals(summ_f.op.name, 'outer/family/inner')

    metagraph_def, _ = meta_graph.export_scoped_meta_graph(export_scope='outer')

    with ops.Graph().as_default() as g:
      meta_graph.import_scoped_meta_graph(metagraph_def, graph=g,
                                          import_scope='new_outer')
      # The summaries should exist, but with outer scope renamed.
      new_summ = g.get_tensor_by_name('new_outer/inner:0')
      new_summ_f = g.get_tensor_by_name('new_outer/family/inner:0')

      # However, the tags are unaffected.
      with self.cached_session() as s:
        new_summ_str, new_summ_f_str = s.run([new_summ, new_summ_f])
        new_summ_pb = summary_pb2.Summary()
        new_summ_pb.ParseFromString(new_summ_str)
        self.assertEquals('outer/inner', new_summ_pb.value[0].tag)
        new_summ_f_pb = summary_pb2.Summary()
        new_summ_f_pb.ParseFromString(new_summ_f_str)
        self.assertEquals('family/outer/family/inner',
                          new_summ_f_pb.value[0].tag)


if __name__ == '__main__':
  test.main()
