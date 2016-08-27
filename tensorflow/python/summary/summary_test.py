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

import six
import tensorflow as tf

from tensorflow.core.framework import types_pb2


class ScalarSummaryTest(tf.test.TestCase):

  def testDtypeErrors(self):
    def _TryMakingScalarSummary(dtype):
      base = dtype.base_dtype
      if base == tf.bool:
        v = False
      elif base == tf.string:
        v = ''
      elif base.is_complex:
        v = complex(0, 0)
      else:
        v = base.min
      c = tf.constant(v, dtype)
      return tf.summary.scalar('name', c)

    for datatype_enum in types_pb2.DataType.values():
      if datatype_enum == types_pb2.DT_INVALID:
        continue
      dtype = tf.as_dtype(datatype_enum)
      if dtype.is_quantized:
        # Quantized ops are funky, and not expected to work.
        continue
      if dtype.is_integer or dtype.is_floating:
        _TryMakingScalarSummary(dtype)
        # No exception should be thrown
      else:
        with self.assertRaises(ValueError):
          _TryMakingScalarSummary(dtype)

  def testShapeErrors(self):
    c1 = tf.constant(0)
    c2 = tf.zeros(5)
    c3 = tf.zeros(5, 5)

    tf.summary.scalar('1', c1)
    with self.assertRaises(ValueError):
      tf.summary.scalar('2', c2)
    with self.assertRaises(ValueError):
      tf.summary.scalar('3', c3)

  def testLabelsAdded(self):
    c = tf.constant(0)

    no_labels = tf.summary.scalar('2', c)
    labels = tf.summary.scalar('1', c, labels=['foo'])

    def _GetLabels(n):
      return n.op.get_attr('labels')

    expected_label = six.b(tf.summary.SCALAR_SUMMARY_LABEL)
    self.assertEquals(_GetLabels(no_labels), [expected_label])
    self.assertEquals(_GetLabels(labels), [six.b('foo'), expected_label])

  def testTensorSummaryOpCreated(self):
    c = tf.constant(0)
    s = tf.summary.scalar('', c)
    self.assertEquals(s.op.type, 'TensorSummary')
    self.assertEquals(s.op.inputs[0], c)


if __name__ == '__main__':
  tf.test.main()
