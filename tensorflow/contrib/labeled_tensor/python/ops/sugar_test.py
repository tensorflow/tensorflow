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

from six.moves import range  # pylint: disable=redefined-builtin

from tensorflow.contrib.labeled_tensor.python.ops import core
from tensorflow.contrib.labeled_tensor.python.ops import ops
from tensorflow.contrib.labeled_tensor.python.ops import sugar
from tensorflow.contrib.labeled_tensor.python.ops import test_util
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class Base(test_util.Base):

  def setUp(self):
    super(Base, self).setUp()

    self.small_lt = core.LabeledTensor(constant_op.constant([1]), [('x', 1)])


class ReshapeCoderTest(Base):

  def setUp(self):
    super(ReshapeCoderTest, self).setUp()

    self.batch_size = 8
    self.num_rows = 50
    self.num_columns = 100
    self.channels = ['red', 'green', 'blue']
    self.masks = [False, True]

    tensor = math_ops.range(0,
                            self.batch_size * self.num_rows * self.num_columns *
                            len(self.channels) * len(self.masks))
    tensor = array_ops.reshape(tensor, [
        self.batch_size, self.num_rows, self.num_columns, len(self.channels),
        len(self.masks)
    ])

    self.batch_axis = ('batch', range(self.batch_size))
    self.row_axis = ('row', range(self.num_rows))
    self.column_axis = ('column', range(self.num_columns))
    self.channel_axis = ('channel', self.channels)
    self.mask_axis = ('mask', self.masks)

    axes = [
        self.batch_axis, self.row_axis, self.column_axis, self.channel_axis,
        self.mask_axis
    ]
    self.masked_image_lt = core.LabeledTensor(tensor, axes)

  def test_name(self):
    rc = sugar.ReshapeCoder(['channel', 'mask'], ['depth'])
    encode_lt = rc.encode(self.masked_image_lt)
    decode_lt = rc.decode(encode_lt)
    self.assertIn('lt_reshape_encode', encode_lt.name)
    self.assertIn('lt_reshape_decode', decode_lt.name)

  def test_bijection_flat(self):
    rc = sugar.ReshapeCoder(['channel', 'mask'], ['depth'])

    encode_lt = rc.encode(self.masked_image_lt)
    golden_axes = core.Axes([
        self.batch_axis, self.row_axis, self.column_axis,
        ('depth', len(self.channels) * len(self.masks))
    ])
    self.assertEqual(encode_lt.axes, golden_axes)

    decode_lt = rc.decode(encode_lt)
    self.assertLabeledTensorsEqual(decode_lt, self.masked_image_lt)

  def test_bijection_with_labels(self):
    depth_axis = core.Axis('depth', range(len(self.channels) * len(self.masks)))
    rc = sugar.ReshapeCoder(['channel', 'mask'],
                            [depth_axis, ('other', ['label'])])

    encode_lt = rc.encode(self.masked_image_lt)
    golden_axes = core.Axes([
        self.batch_axis, self.row_axis, self.column_axis, depth_axis,
        ('other', ['label'])
    ])
    self.assertEqual(encode_lt.axes, golden_axes)

    decode_lt = rc.decode(encode_lt)
    self.assertLabeledTensorsEqual(decode_lt, self.masked_image_lt)

  def test_invalid_input(self):
    with self.assertRaises(ValueError):
      rc = sugar.ReshapeCoder(['channel', 'mask'], ['depth'])
      rc.decode(self.masked_image_lt)
    with self.assertRaises(ValueError):
      rc = sugar.ReshapeCoder(['channel', 'mask'], ['depth'])
      rc.encode(self.masked_image_lt)
      rc.encode(ops.select(self.masked_image_lt, {'channel': 'red'}))


if __name__ == '__main__':
  test.main()
