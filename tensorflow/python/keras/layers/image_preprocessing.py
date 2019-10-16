# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Keras image preprocessing layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import image_ops_impl as image_ops
from tensorflow.python.ops import math_ops

ResizeMethod = image_ops.ResizeMethod

_RESIZE_METHODS = {
    'bilinear': ResizeMethod.BILINEAR,
    'nearest': ResizeMethod.NEAREST_NEIGHBOR,
    'bicubic': ResizeMethod.BICUBIC,
    'area': ResizeMethod.AREA,
    'lanczos3': ResizeMethod.LANCZOS3,
    'lanczos5': ResizeMethod.LANCZOS5,
    'gaussian': ResizeMethod.GAUSSIAN,
    'mitchellcubic': ResizeMethod.MITCHELLCUBIC
}


class Resizing(Layer):
  """Image resizing layer.

  Resize the batched image input to target height and width. The input should
  be a 4-D tensor in the format of NHWC.

  Attributes:
    height: Integer, the height of the output shape.
    width: Integer, the width of the output shape.
    interpolation: String, the interpolation method. Defaults to `bilinear`.
      Supports `bilinear`, `nearest`, `bicubic`, `area`, `lanczos3`, `lanczos5`,
      `gaussian`, `mitchellcubic`
  """

  def __init__(self, height, width, interpolation='bilinear', **kwargs):
    self.target_height = height
    self.target_width = width
    self.interpolation = interpolation
    self._interpolation_method = get_interpolation(interpolation)
    self.input_spec = InputSpec(ndim=4)
    super(Resizing, self).__init__(**kwargs)

  def build(self, input_shape):
    channel_axis = 3
    channel_dim = int(input_shape[channel_axis])
    self.input_spec = InputSpec(ndim=4, axes={channel_axis: channel_dim})
    self.built = True

  def call(self, inputs):
    outputs = image_ops.resize_images_v2(
        images=inputs,
        size=[self.target_height, self.target_width],
        method=self._interpolation_method)
    return outputs

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    return tensor_shape.TensorShape(
        [input_shape[0], self.target_height, self.target_width, input_shape[3]])

  def get_config(self):
    config = {
        'height': self.target_height,
        'width': self.target_width,
        'interpolation': self.interpolation,
    }
    base_config = super(Resizing, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class CenterCrop(Layer):
  """Crop the central portion of the images to target height and width.

  Input shape:
    4D tensor with shape:
    `(samples, height, width, channels)`, data_format='channels_last'.

  Output shape:
    4D tensor with shape:
    `(samples, target_height, target_width, channels)`.

  If the input height/width is even and the target height/width is odd (or
  inversely), the input image is left-padded by 1 pixel.

  Attributes:
    height: Integer, the height of the output shape.
    width: Integer, the width of the output shape.
  """

  def __init__(self, height, width, **kwargs):
    self.target_height = height
    self.target_width = width
    self.input_spec = InputSpec(ndim=4)
    super(CenterCrop, self).__init__(**kwargs)

  def build(self, input_shape):
    channel_axis = 3
    channel_dim = int(input_shape[channel_axis])
    self.input_spec = InputSpec(ndim=4, axes={channel_axis: channel_dim})
    self.built = True

  def call(self, inputs):
    inputs_shape = array_ops.shape(inputs)
    h_axis, w_axis = 1, 2
    img_hd = inputs_shape[h_axis]
    img_wd = inputs_shape[w_axis]
    img_hd_diff = img_hd - self.target_height
    img_wd_diff = img_wd - self.target_width
    checks = []
    checks.append(
        check_ops.assert_non_negative(
            img_hd_diff,
            message='The crop height {} should not be greater than input '
            'height.'.format(self.target_height)))
    checks.append(
        check_ops.assert_non_negative(
            img_wd_diff,
            message='The crop width {} should not be greater than input '
            'width.'.format(self.target_width)))
    with ops.control_dependencies(checks):
      bbox_h_start = math_ops.cast(img_hd_diff / 2, dtypes.int32)
      bbox_w_start = math_ops.cast(img_wd_diff / 2, dtypes.int32)
      bbox_begin = array_ops.stack([0, bbox_h_start, bbox_w_start, 0])
      bbox_size = array_ops.stack(
          [-1, self.target_height, self.target_width, -1])
      outputs = array_ops.slice(inputs, bbox_begin, bbox_size)
      return outputs

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    return tensor_shape.TensorShape(
        [input_shape[0], self.target_height, self.target_width, input_shape[3]])

  def get_config(self):
    config = {
        'height': self.target_height,
        'width': self.target_width,
    }
    base_config = super(CenterCrop, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


def get_interpolation(interpolation):
  interpolation = interpolation.lower()
  if interpolation not in _RESIZE_METHODS:
    raise NotImplementedError(
        'Value not recognized for `interpolation`: {}. Supported values '
        'are: {}'.format(interpolation, _RESIZE_METHODS.keys()))
  return _RESIZE_METHODS[interpolation]
