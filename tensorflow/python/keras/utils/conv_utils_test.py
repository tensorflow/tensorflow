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
"""Tests for conv_utils."""

import itertools

from absl.testing import parameterized
import numpy as np

from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.platform import test


def _get_const_output_shape(input_shape, dim):
  return tuple([min(d, dim) for d in input_shape])


input_shapes = [
    (0,),
    (0, 0),
    (1,),
    (2,),
    (3,),
    (1, 0),
    (0, 3),
    (1, 1),
    (1, 2),
    (3, 1),
    (2, 2),
    (3, 3),
    (1, 0, 1),
    (5, 2, 3),
    (3, 5, 6, 7, 0),
    (3, 2, 2, 4, 4),
    (1, 2, 3, 4, 7, 2),
]


class TestBasicConvUtilsTest(test.TestCase):

  def test_convert_data_format(self):
    self.assertEqual('NCDHW', conv_utils.convert_data_format(
        'channels_first', 5))
    self.assertEqual('NCHW', conv_utils.convert_data_format(
        'channels_first', 4))
    self.assertEqual('NCW', conv_utils.convert_data_format('channels_first', 3))
    self.assertEqual('NHWC', conv_utils.convert_data_format('channels_last', 4))
    self.assertEqual('NWC', conv_utils.convert_data_format('channels_last', 3))
    self.assertEqual('NDHWC', conv_utils.convert_data_format(
        'channels_last', 5))

    with self.assertRaises(ValueError):
      conv_utils.convert_data_format('invalid', 2)

  def test_normalize_tuple(self):
    self.assertEqual((2, 2, 2),
                     conv_utils.normalize_tuple(2, n=3, name='strides'))
    self.assertEqual((2, 1, 2),
                     conv_utils.normalize_tuple((2, 1, 2), n=3, name='strides'))

    with self.assertRaises(ValueError):
      conv_utils.normalize_tuple((2, 1), n=3, name='strides')

    with self.assertRaises(ValueError):
      conv_utils.normalize_tuple(None, n=3, name='strides')

  def test_normalize_data_format(self):
    self.assertEqual('channels_last',
                     conv_utils.normalize_data_format('Channels_Last'))
    self.assertEqual('channels_first',
                     conv_utils.normalize_data_format('CHANNELS_FIRST'))

    with self.assertRaises(ValueError):
      conv_utils.normalize_data_format('invalid')

  def test_normalize_padding(self):
    self.assertEqual('same', conv_utils.normalize_padding('SAME'))
    self.assertEqual('valid', conv_utils.normalize_padding('VALID'))

    with self.assertRaises(ValueError):
      conv_utils.normalize_padding('invalid')

  def test_conv_output_length(self):
    self.assertEqual(4, conv_utils.conv_output_length(4, 2, 'same', 1, 1))
    self.assertEqual(2, conv_utils.conv_output_length(4, 2, 'same', 2, 1))
    self.assertEqual(3, conv_utils.conv_output_length(4, 2, 'valid', 1, 1))
    self.assertEqual(2, conv_utils.conv_output_length(4, 2, 'valid', 2, 1))
    self.assertEqual(5, conv_utils.conv_output_length(4, 2, 'full', 1, 1))
    self.assertEqual(3, conv_utils.conv_output_length(4, 2, 'full', 2, 1))
    self.assertEqual(2, conv_utils.conv_output_length(5, 2, 'valid', 2, 2))

  def test_conv_input_length(self):
    self.assertEqual(3, conv_utils.conv_input_length(4, 2, 'same', 1))
    self.assertEqual(2, conv_utils.conv_input_length(2, 2, 'same', 2))
    self.assertEqual(4, conv_utils.conv_input_length(3, 2, 'valid', 1))
    self.assertEqual(4, conv_utils.conv_input_length(2, 2, 'valid', 2))
    self.assertEqual(3, conv_utils.conv_input_length(4, 2, 'full', 1))
    self.assertEqual(4, conv_utils.conv_input_length(3, 2, 'full', 2))

  def test_deconv_output_length(self):
    self.assertEqual(4, conv_utils.deconv_output_length(4, 2, 'same', stride=1))
    self.assertEqual(8, conv_utils.deconv_output_length(4, 2, 'same', stride=2))
    self.assertEqual(5, conv_utils.deconv_output_length(
        4, 2, 'valid', stride=1))
    self.assertEqual(8, conv_utils.deconv_output_length(
        4, 2, 'valid', stride=2))
    self.assertEqual(3, conv_utils.deconv_output_length(4, 2, 'full', stride=1))
    self.assertEqual(6, conv_utils.deconv_output_length(4, 2, 'full', stride=2))
    self.assertEqual(
        5,
        conv_utils.deconv_output_length(
            4, 2, 'same', output_padding=2, stride=1))
    self.assertEqual(
        7,
        conv_utils.deconv_output_length(
            4, 2, 'same', output_padding=1, stride=2))
    self.assertEqual(
        7,
        conv_utils.deconv_output_length(
            4, 2, 'valid', output_padding=2, stride=1))
    self.assertEqual(
        9,
        conv_utils.deconv_output_length(
            4, 2, 'valid', output_padding=1, stride=2))
    self.assertEqual(
        5,
        conv_utils.deconv_output_length(
            4, 2, 'full', output_padding=2, stride=1))
    self.assertEqual(
        7,
        conv_utils.deconv_output_length(
            4, 2, 'full', output_padding=1, stride=2))
    self.assertEqual(
        5,
        conv_utils.deconv_output_length(
            4, 2, 'same', output_padding=1, stride=1, dilation=2))
    self.assertEqual(
        12,
        conv_utils.deconv_output_length(
            4, 2, 'valid', output_padding=2, stride=2, dilation=3))
    self.assertEqual(
        6,
        conv_utils.deconv_output_length(
            4, 2, 'full', output_padding=2, stride=2, dilation=3))


@parameterized.parameters(input_shapes)
class TestConvUtils(test.TestCase, parameterized.TestCase):

  def test_conv_kernel_mask_fc(self, *input_shape):
    padding = 'valid'
    kernel_shape = input_shape
    ndims = len(input_shape)
    strides = (1,) * ndims
    output_shape = _get_const_output_shape(input_shape, dim=1)
    mask = np.ones(input_shape + output_shape, np.bool_)
    self.assertAllEqual(
        mask,
        conv_utils.conv_kernel_mask(
            input_shape,
            kernel_shape,
            strides,
            padding
        )
    )

  def test_conv_kernel_mask_diag(self, *input_shape):
    ndims = len(input_shape)
    kernel_shape = (1,) * ndims
    strides = (1,) * ndims

    for padding in ['valid', 'same']:
      mask = np.identity(int(np.prod(input_shape)), np.bool_)
      mask = np.reshape(mask, input_shape * 2)
      self.assertAllEqual(
          mask,
          conv_utils.conv_kernel_mask(
              input_shape,
              kernel_shape,
              strides,
              padding
          )
      )

  def test_conv_kernel_mask_full_stride(self, *input_shape):
    padding = 'valid'
    ndims = len(input_shape)
    kernel_shape = (1,) * ndims
    strides = tuple([max(d, 1) for d in input_shape])
    output_shape = _get_const_output_shape(input_shape, dim=1)

    mask = np.zeros(input_shape + output_shape, np.bool_)
    if all(d > 0 for d in mask.shape):  # pylint: disable=not-an-iterable
      mask[(0,) * len(output_shape)] = True

    self.assertAllEqual(
        mask,
        conv_utils.conv_kernel_mask(
            input_shape,
            kernel_shape,
            strides,
            padding
        )
    )

  def test_conv_kernel_mask_almost_full_stride(self, *input_shape):
    padding = 'valid'
    ndims = len(input_shape)
    kernel_shape = (1,) * ndims
    strides = tuple([max(d - 1, 1) for d in input_shape])
    output_shape = _get_const_output_shape(input_shape, dim=2)

    mask = np.zeros(input_shape + output_shape, np.bool_)
    if all(d > 0 for d in mask.shape):  # pylint: disable=not-an-iterable
      for in_position in itertools.product(*[[0, d - 1] for d in input_shape]):
        out_position = tuple([min(p, 1) for p in in_position])
        mask[in_position + out_position] = True

    self.assertAllEqual(
        mask,
        conv_utils.conv_kernel_mask(
            input_shape,
            kernel_shape,
            strides,
            padding
        )
    )

  def test_conv_kernel_mask_rect_kernel(self, *input_shape):
    padding = 'valid'
    ndims = len(input_shape)
    strides = (1,) * ndims

    for d in range(ndims):
      kernel_shape = [1] * ndims
      kernel_shape[d] = input_shape[d]

      output_shape = list(input_shape)
      output_shape[d] = min(1, input_shape[d])

      mask = np.identity(int(np.prod(input_shape)), np.bool_)
      mask = np.reshape(mask, input_shape * 2)

      for p in itertools.product(*[range(input_shape[dim])
                                   for dim in range(ndims)]):
        p = list(p)
        p[d] = slice(None)
        mask[p * 2] = True

      mask = np.take(mask, range(0, min(1, input_shape[d])), ndims + d)

      self.assertAllEqual(
          mask,
          conv_utils.conv_kernel_mask(
              input_shape,
              kernel_shape,
              strides,
              padding
          )
      )

  def test_conv_kernel_mask_wrong_padding(self, *input_shape):
    ndims = len(input_shape)
    kernel_shape = (1,) * ndims
    strides = (1,) * ndims

    conv_utils.conv_kernel_mask(
        input_shape,
        kernel_shape,
        strides,
        'valid'
    )

    conv_utils.conv_kernel_mask(
        input_shape,
        kernel_shape,
        strides,
        'same'
    )

    self.assertRaises(NotImplementedError,
                      conv_utils.conv_kernel_mask,
                      input_shape, kernel_shape, strides, 'full')

  def test_conv_kernel_mask_wrong_dims(self, *input_shape):
    kernel_shape = 1
    strides = 1

    conv_utils.conv_kernel_mask(
        input_shape,
        kernel_shape,
        strides,
        'valid'
    )

    ndims = len(input_shape)

    kernel_shape = (2,) * (ndims + 1)
    self.assertRaises(ValueError,
                      conv_utils.conv_kernel_mask,
                      input_shape, kernel_shape, strides, 'same')

    strides = (1,) * ndims
    self.assertRaises(ValueError,
                      conv_utils.conv_kernel_mask,
                      input_shape, kernel_shape, strides, 'valid')

    kernel_shape = (1,) * ndims
    strides = (2,) * (ndims - 1)
    self.assertRaises(ValueError,
                      conv_utils.conv_kernel_mask,
                      input_shape, kernel_shape, strides, 'valid')

    strides = (2,) * ndims
    conv_utils.conv_kernel_mask(
        input_shape,
        kernel_shape,
        strides,
        'valid'
    )


if __name__ == '__main__':
  test.main()
