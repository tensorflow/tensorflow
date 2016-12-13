# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tf.layers.core."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.layers import utils
from tensorflow.python.platform import test


class ConvUtilsTest(test.TestCase):

  def testConvertDataFormat(self):
    self.assertEqual(utils.convert_data_format('channels_first', 4), 'NCHW')
    self.assertEqual(utils.convert_data_format('channels_first', 3), 'NCW')
    self.assertEqual(utils.convert_data_format('channels_last', 4), 'NHWC')
    self.assertEqual(utils.convert_data_format('channels_last', 3), 'NWC')
    self.assertEqual(utils.convert_data_format('channels_last', 5), 'NDHWC')

    with self.assertRaises(ValueError):
      utils.convert_data_format('invalid', 2)

  def testNormalizeTuple(self):
    self.assertEqual(utils.normalize_tuple(2, n=3, name='strides'), (2, 2, 2))
    self.assertEqual(
        utils.normalize_tuple((2, 1, 2), n=3, name='strides'), (2, 1, 2))

    with self.assertRaises(ValueError):
      utils.normalize_tuple((2, 1), n=3, name='strides')

    with self.assertRaises(ValueError):
      utils.normalize_tuple(None, n=3, name='strides')

  def testNormalizeDataFormat(self):
    self.assertEqual(
        utils.normalize_data_format('Channels_Last'), 'channels_last')
    self.assertEqual(
        utils.normalize_data_format('CHANNELS_FIRST'), 'channels_first')

    with self.assertRaises(ValueError):
      utils.normalize_data_format('invalid')

  def testNormalizePadding(self):
    self.assertEqual(utils.normalize_padding('SAME'), 'same')
    self.assertEqual(utils.normalize_padding('VALID'), 'valid')

    with self.assertRaises(ValueError):
      utils.normalize_padding('invalid')


if __name__ == '__main__':
  test.main()
