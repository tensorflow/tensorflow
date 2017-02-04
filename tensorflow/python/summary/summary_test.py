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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin

from google.protobuf import json_format

from tensorflow.core.framework import summary_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.summary import summary as summary_lib


class ScalarSummaryTest(test.TestCase):

  def testScalarSummary(self):
    with self.test_session() as s:
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

  def testSummarizingVariable(self):
    with self.test_session() as s:
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

  def testImageSummary(self):
    with self.test_session() as s:
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

  def testHistogramSummary(self):
    with self.test_session() as s:
      i = array_ops.ones((5, 4, 4, 3))
      with ops.name_scope('outer'):
        summ_op = summary_lib.histogram('inner', i)
      summary_str = s.run(summ_op)
    summary = summary_pb2.Summary()
    summary.ParseFromString(summary_str)
    self.assertEqual(len(summary.value), 1)
    self.assertEqual(summary.value[0].tag, 'outer/inner')

  def testSummaryNameConversion(self):
    c = constant_op.constant(3)
    s = summary_lib.scalar('name with spaces', c)
    self.assertEqual(s.op.name, 'name_with_spaces')

    s2 = summary_lib.scalar('name with many $#illegal^: characters!', c)
    self.assertEqual(s2.op.name, 'name_with_many___illegal___characters_')

    s3 = summary_lib.scalar('/name/with/leading/slash', c)
    self.assertEqual(s3.op.name, 'name/with/leading/slash')


if __name__ == '__main__':
  test.main()
