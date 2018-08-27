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
"""Tests for tf.layers.utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.layers import utils
from tensorflow.python.platform import test


class ConvUtilsTest(test.TestCase):

  def testConvertDataFormat(self):
    self.assertEqual('NCDHW', utils.convert_data_format('channels_first', 5))
    self.assertEqual('NCHW', utils.convert_data_format('channels_first', 4))
    self.assertEqual('NCW', utils.convert_data_format('channels_first', 3))
    self.assertEqual('NHWC', utils.convert_data_format('channels_last', 4))
    self.assertEqual('NWC', utils.convert_data_format('channels_last', 3))
    self.assertEqual('NDHWC', utils.convert_data_format('channels_last', 5))

    with self.assertRaises(ValueError):
      utils.convert_data_format('invalid', 2)

  def testNormalizeTuple(self):
    self.assertEqual((2, 2, 2), utils.normalize_tuple(2, n=3, name='strides'))
    self.assertEqual(
        (2, 1, 2), utils.normalize_tuple((2, 1, 2), n=3, name='strides'))

    with self.assertRaises(ValueError):
      utils.normalize_tuple((2, 1), n=3, name='strides')

    with self.assertRaises(ValueError):
      utils.normalize_tuple(None, n=3, name='strides')

  def testNormalizeDataFormat(self):
    self.assertEqual(
        'channels_last', utils.normalize_data_format('Channels_Last'))
    self.assertEqual(
        'channels_first', utils.normalize_data_format('CHANNELS_FIRST'))

    with self.assertRaises(ValueError):
      utils.normalize_data_format('invalid')

  def testNormalizePadding(self):
    self.assertEqual('same', utils.normalize_padding('SAME'))
    self.assertEqual('valid', utils.normalize_padding('VALID'))

    with self.assertRaises(ValueError):
      utils.normalize_padding('invalid')

  def testConvOutputLength(self):
    self.assertEqual(4, utils.conv_output_length(4, 2, 'same', 1, 1))
    self.assertEqual(2, utils.conv_output_length(4, 2, 'same', 2, 1))
    self.assertEqual(3, utils.conv_output_length(4, 2, 'valid', 1, 1))
    self.assertEqual(2, utils.conv_output_length(4, 2, 'valid', 2, 1))
    self.assertEqual(5, utils.conv_output_length(4, 2, 'full', 1, 1))
    self.assertEqual(3, utils.conv_output_length(4, 2, 'full', 2, 1))
    self.assertEqual(2, utils.conv_output_length(5, 2, 'valid', 2, 2))

  def testConvInputLength(self):
    self.assertEqual(3, utils.conv_input_length(4, 2, 'same', 1))
    self.assertEqual(2, utils.conv_input_length(2, 2, 'same', 2))
    self.assertEqual(4, utils.conv_input_length(3, 2, 'valid', 1))
    self.assertEqual(4, utils.conv_input_length(2, 2, 'valid', 2))
    self.assertEqual(3, utils.conv_input_length(4, 2, 'full', 1))
    self.assertEqual(4, utils.conv_input_length(3, 2, 'full', 2))

  def testDeconvOutputLength(self):
    self.assertEqual(4, utils.deconv_output_length(4, 2, 'same', 1))
    self.assertEqual(8, utils.deconv_output_length(4, 2, 'same', 2))
    self.assertEqual(5, utils.deconv_output_length(4, 2, 'valid', 1))
    self.assertEqual(8, utils.deconv_output_length(4, 2, 'valid', 2))
    self.assertEqual(3, utils.deconv_output_length(4, 2, 'full', 1))
    self.assertEqual(6, utils.deconv_output_length(4, 2, 'full', 2))


if __name__ == '__main__':
  test.main()
