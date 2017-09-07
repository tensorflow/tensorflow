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

from tensorflow.contrib.labeled_tensor.python.ops import core
from tensorflow.contrib.labeled_tensor.python.ops import io_ops
from tensorflow.contrib.labeled_tensor.python.ops import test_util
from tensorflow.core.example import example_pb2
from tensorflow.core.example import feature_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test as test_lib


class ParseBase(test_util.Base):

  def setUp(self):
    super(ParseBase, self).setUp()
    examples = [
        example_pb2.Example(features=feature_pb2.Features(feature={
            'a':
                feature_pb2.Feature(
                    int64_list=feature_pb2.Int64List(value=[1])),
            'b':
                feature_pb2.Feature(
                    int64_list=feature_pb2.Int64List(value=[2, 3, 4])),
        })),
        example_pb2.Example(features=feature_pb2.Features(feature={
            'a':
                feature_pb2.Feature(
                    int64_list=feature_pb2.Int64List(value=[5])),
            'b':
                feature_pb2.Feature(
                    int64_list=feature_pb2.Int64List(value=[6, 7, 8])),
        })),
    ]
    self.serialized = core.LabeledTensor(
        constant_op.constant([ex.SerializeToString() for ex in examples]),
        ['batch'])
    self.features = {
        'a': io_ops.FixedLenFeature([], dtypes.int64),
        'b': io_ops.FixedLenFeature([('x', 3)], dtypes.int64)
    }


class TestParseExample(ParseBase):

  def test(self):
    expected_a = core.LabeledTensor(constant_op.constant([1, 5]), ['batch'])
    expected_b = core.LabeledTensor(
        constant_op.constant([[2, 3, 4], [6, 7, 8]]), ['batch', 'x'])
    parsed = io_ops.parse_example(self.serialized, self.features)
    self.assertLabeledTensorsEqual(expected_a, parsed['a'])
    self.assertLabeledTensorsEqual(expected_b, parsed['b'])

  def test_placeholder(self):
    serialized = core.LabeledTensor(
        array_ops.placeholder(dtypes.string, [None]), ['batch'])
    # should not raise
    io_ops.parse_example(serialized, self.features)


class TestParseSingleExample(ParseBase):

  def test(self):
    expected_a = core.LabeledTensor(constant_op.constant(1), [])
    expected_b = core.LabeledTensor(constant_op.constant([2, 3, 4]), ['x'])
    parsed = io_ops.parse_single_example(self.serialized[0], self.features)
    self.assertLabeledTensorsEqual(expected_a, parsed['a'])
    self.assertLabeledTensorsEqual(expected_b, parsed['b'])

  def test_unknown_size(self):
    features = {'a': io_ops.FixedLenFeature([('x', None)], dtypes.int64)}
    serialized = array_ops.placeholder(dtypes.string, [])
    with self.assertRaisesRegexp(ValueError, 'unknown size'):
      io_ops.parse_single_example(serialized, features)


class PlaceholderTest(test_util.Base):

  def test_name(self):
    placeholder_lt = io_ops.placeholder(dtypes.float32, [])
    self.assertIn('lt_placeholder', placeholder_lt.name)

  def test(self):
    placeholder_lt = io_ops.placeholder(dtypes.float32,
                                        ['batch', ('x', ['a', 'b'])])
    self.assertEqual(placeholder_lt.dtype, dtypes.float32)
    self.assertEqual(placeholder_lt.axes,
                     core.Axes([('batch', None), ('x', ['a', 'b'])]))

  def test_feed(self):
    sess = session.Session()
    placeholder_lt = io_ops.placeholder(dtypes.float32, [])
    two_times = 2.0 * placeholder_lt
    result = sess.run(two_times, {placeholder_lt.tensor: 1})
    self.assertEqual(result, 2.0)


if __name__ == '__main__':
  test_lib.main()
