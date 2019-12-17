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
# pylint: disable=g-classes-have-attributes
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import image_ops_impl as image_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import stateful_random_ops
from tensorflow.python.ops import stateless_random_ops

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

  Arguments:
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

  Arguments:
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


class RandomCrop(Layer):
  """Randomly crop the images to target height and width.

  This layer will crop all the images in the same batch to the same cropping
  location.
  By default, random croppping is only applied during training. At inference
  time, the images will be first rescaled to preserve the shorter side, and
  center cropped. If you need to apply random cropping at inference time,
  set `training` to True when calling the layer.

  Input shape:
    4D tensor with shape:
    `(samples, height, width, channels)`, data_format='channels_last'.

  Output shape:
    4D tensor with shape:
    `(samples, target_height, target_width, channels)`.

  Arguments:
    height: Integer, the height of the output shape.
    width: Integer, the width of the output shape.
    seed: Integer. Used to create a random seed.
  """

  def __init__(self, height, width, seed=None, **kwargs):
    self.height = height
    self.width = width
    self.seed = seed
    self._rng = make_generator(self.seed)
    self.input_spec = InputSpec(ndim=4)
    super(RandomCrop, self).__init__(**kwargs)

  def call(self, inputs, training=None):
    if training is None:
      training = K.learning_phase()

    def random_cropped_inputs():
      """Cropped inputs with stateless random ops."""
      input_shape = array_ops.shape(inputs)
      crop_size = array_ops.stack(
          [input_shape[0], self.height, self.width, input_shape[3]])
      check = control_flow_ops.Assert(
          math_ops.reduce_all(input_shape >= crop_size), [
              'Need value.shape >= size, got input shape', input_shape,
              ' but height is ', self.height, ' and weight is ', self.width
          ])
      input_shape = control_flow_ops.with_dependencies([check], input_shape)
      limit = input_shape - crop_size + 1
      offset = stateless_random_ops.stateless_random_uniform(
          array_ops.shape(input_shape),
          dtype=crop_size.dtype,
          maxval=crop_size.dtype.max,
          seed=self._rng.make_seeds()[:, 0]) % limit
      return array_ops.slice(inputs, offset, crop_size)

    # TODO(b/143885775): Share logic with Resize and CenterCrop.
    def resize_and_center_cropped_inputs():
      """Deterministically resize to shorter side and center crop."""
      input_shape = array_ops.shape(inputs)
      input_height_t = input_shape[1]
      input_width_t = input_shape[2]
      ratio_cond = (input_height_t / input_width_t > 1.)
      # pylint: disable=g-long-lambda
      resized_height = tf_utils.smart_cond(
          ratio_cond,
          lambda: math_ops.cast(self.width * input_height_t / input_width_t,
                                input_height_t.dtype), lambda: self.height)
      resized_width = tf_utils.smart_cond(
          ratio_cond, lambda: self.width,
          lambda: math_ops.cast(self.height * input_width_t / input_height_t,
                                input_width_t.dtype))
      # pylint: enable=g-long-lambda
      resized_inputs = image_ops.resize_images_v2(
          images=inputs, size=array_ops.stack([resized_height, resized_width]))

      img_hd_diff = resized_height - self.height
      img_wd_diff = resized_width - self.width
      bbox_h_start = math_ops.cast(img_hd_diff / 2, dtypes.int32)
      bbox_w_start = math_ops.cast(img_wd_diff / 2, dtypes.int32)
      bbox_begin = array_ops.stack([0, bbox_h_start, bbox_w_start, 0])
      bbox_size = array_ops.stack([-1, self.height, self.width, -1])
      outputs = array_ops.slice(resized_inputs, bbox_begin, bbox_size)
      return outputs

    output = tf_utils.smart_cond(training, random_cropped_inputs,
                                 resize_and_center_cropped_inputs)
    original_shape = inputs.shape.as_list()
    batch_size, num_channels = original_shape[0], original_shape[3]
    output_shape = [batch_size] + [self.height, self.width] + [num_channels]
    output.set_shape(output_shape)
    return output

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    return tensor_shape.TensorShape(
        [input_shape[0], self.height, self.width, input_shape[3]])

  def get_config(self):
    config = {
        'height': self.height,
        'width': self.width,
        'seed': self.seed,
    }
    base_config = super(RandomCrop, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class Rescaling(Layer):
  """Multiply inputs by `scale`.

  For instance, to rescale an input in the `[0, 255]` range
  to be in the `[0, 1]` range, you would pass `scale=1./255`.

  The rescaling is applied both during training and inference.

  Input shape:
    Arbitrary.

  Output shape:
    Same as input.

  Arguments:
    scale: Float, the scale to apply to the inputs.
  """

  def __init__(self, scale, **kwargs):
    self.scale = scale
    super(Rescaling, self).__init__(**kwargs)

  def call(self, inputs):
    dtype = self._compute_dtype
    return math_ops.cast(inputs, dtype) * math_ops.cast(self.scale, dtype)

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    config = {
        'scale': self.scale,
    }
    base_config = super(Rescaling, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class RandomFlip(Layer):
  """Randomly flip each image horizontally and vertically.

  This layer will by default flip the images horizontally and then vertically
  during training time.
  `RandomFlip(horizontal=True)` will only flip the input horizontally.
  `RandomFlip(vertical=True)` will only flip the input vertically.
  During inference time, the output will be identical to input. Call the layer
  with `training=True` to flip the input.

  Input shape:
    4D tensor with shape:
    `(samples, height, width, channels)`, data_format='channels_last'.

  Output shape:
    4D tensor with shape:
    `(samples, height, width, channels)`, data_format='channels_last'.

  Attributes:
    horizontal: Bool, whether to randomly flip horizontally.
    width: Bool, whether to randomly flip vertically.
    seed: Integer. Used to create a random seed.
  """

  def __init__(self, horizontal=None, vertical=None, seed=None, **kwargs):
    # If both arguments are None, set both to True.
    if horizontal is None and vertical is None:
      self.horizontal = True
      self.vertical = True
    else:
      self.horizontal = horizontal or False
      self.vertical = vertical or False
    self.seed = seed
    self._rng = make_generator(self.seed)
    self.input_spec = InputSpec(ndim=4)
    super(RandomFlip, self).__init__(**kwargs)

  def call(self, inputs, training=None):
    if training is None:
      training = K.learning_phase()

    def random_flipped_inputs():
      flipped_outputs = inputs
      if self.horizontal:
        flipped_outputs = image_ops.random_flip_up_down(flipped_outputs,
                                                        self.seed)
      if self.vertical:
        flipped_outputs = image_ops.random_flip_left_right(
            flipped_outputs, self.seed)
      return flipped_outputs

    output = tf_utils.smart_cond(training, random_flipped_inputs,
                                 lambda: inputs)
    output.set_shape(inputs.shape)
    return output

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    config = {
        'horizontal': self.horizontal,
        'vertical': self.vertical,
        'seed': self.seed,
    }
    base_config = super(RandomFlip, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


def make_generator(seed=None):
  if seed:
    return stateful_random_ops.Generator.from_seed(seed)
  else:
    return stateful_random_ops.Generator.from_non_deterministic_state()


def get_interpolation(interpolation):
  interpolation = interpolation.lower()
  if interpolation not in _RESIZE_METHODS:
    raise NotImplementedError(
        'Value not recognized for `interpolation`: {}. Supported values '
        'are: {}'.format(interpolation, _RESIZE_METHODS.keys()))
  return _RESIZE_METHODS[interpolation]
