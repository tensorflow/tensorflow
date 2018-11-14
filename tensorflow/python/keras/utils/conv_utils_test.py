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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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


@parameterized.parameters(input_shapes)
class TestConvUtils(test.TestCase, parameterized.TestCase):

  def test_conv_kernel_mask_fc(self, *input_shape):
    padding = 'valid'
    kernel_shape = input_shape
    ndims = len(input_shape)
    strides = (1,) * ndims
    output_shape = _get_const_output_shape(input_shape, dim=1)
    mask = np.ones(input_shape + output_shape, np.bool)
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
      mask = np.identity(int(np.prod(input_shape)), np.bool)
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

    mask = np.zeros(input_shape + output_shape, np.bool)
    if all(d > 0 for d in mask.shape):
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

    mask = np.zeros(input_shape + output_shape, np.bool)
    if all(d > 0 for d in mask.shape):
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

      mask = np.identity(int(np.prod(input_shape)), np.bool)
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
