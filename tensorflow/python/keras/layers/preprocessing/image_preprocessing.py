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

import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import stateful_random_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.util.tf_export import keras_export

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

# Visually the data format should be NWHC instead of NHWC.
H_AXIS = 2
W_AXIS = 1


def check_fill_mode_and_interpolation(fill_mode, interpolation):
  if fill_mode not in {'reflect', 'wrap', 'constant'}:
    raise NotImplementedError(
        'Unknown `fill_mode` {}. Only `reflect`, `wrap` and '
        '`constant` are supported.'.format(fill_mode))
  if interpolation not in {'nearest', 'bilinear'}:
    raise NotImplementedError('Unknown `interpolation` {}. Only `nearest` and '
                              '`bilinear` are supported.'.format(interpolation))


@keras_export('keras.layers.experimental.preprocessing.Resizing')
class Resizing(Layer):
  """Image resizing layer.

  Resize the batched image input to target height and width. The input should
  be a 4-D tensor, channels_last format.

  Arguments:
    height: Integer, the height of the output shape.
    width: Integer, the width of the output shape.
    interpolation: String, the interpolation method. Defaults to `bilinear`.
      Supports `bilinear`, `nearest`, `bicubic`, `area`, `lanczos3`, `lanczos5`,
      `gaussian`, `mitchellcubic`
    name: A string, the name of the layer.
  """

  def __init__(self,
               height,
               width,
               interpolation='bilinear',
               name=None,
               **kwargs):
    self.target_height = height
    self.target_width = width
    self.interpolation = interpolation
    self._interpolation_method = get_interpolation(interpolation)
    self.input_spec = InputSpec(ndim=4)
    super(Resizing, self).__init__(name=name, **kwargs)

  def call(self, inputs):
    outputs = image_ops.resize_images_v2(
        images=inputs,
        size=[self.target_width, self.target_height],
        method=self._interpolation_method)
    return outputs

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    return tensor_shape.TensorShape(
        [input_shape[0], self.target_width, self.target_height, input_shape[3]])

  def get_config(self):
    config = {
        'height': self.target_height,
        'width': self.target_width,
        'interpolation': self.interpolation,
    }
    base_config = super(Resizing, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@keras_export('keras.layers.experimental.preprocessing.CenterCrop')
class CenterCrop(Layer):
  """Crop the central portion of the images to target height and width.

  Input shape:
    4D tensor with shape:
    `(samples, width, height, channels)`, data_format='channels_last'.

  Output shape:
    4D tensor with shape:
    `(samples, target_width, target_height, channels)`.

  If the input height/width is even and the target height/width is odd (or
  inversely), the input image is left-padded by 1 pixel.

  Arguments:
    height: Integer, the height of the output shape.
    width: Integer, the width of the output shape.
    name: A string, the name of the layer.
  """

  def __init__(self, height, width, name=None, **kwargs):
    self.target_height = height
    self.target_width = width
    self.input_spec = InputSpec(ndim=4)
    super(CenterCrop, self).__init__(name=name, **kwargs)

  def call(self, inputs):
    inputs_shape = array_ops.shape(inputs)
    img_hd = inputs_shape[H_AXIS]
    img_wd = inputs_shape[W_AXIS]
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
      bbox_begin = array_ops.stack([0, bbox_w_start, bbox_h_start, 0])
      bbox_size = array_ops.stack(
          [-1, self.target_width, self.target_height, -1])
      outputs = array_ops.slice(inputs, bbox_begin, bbox_size)
      return outputs

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    return tensor_shape.TensorShape(
        [input_shape[0], self.target_width, self.target_height, input_shape[3]])

  def get_config(self):
    config = {
        'height': self.target_height,
        'width': self.target_width,
    }
    base_config = super(CenterCrop, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@keras_export('keras.layers.experimental.preprocessing.RandomCrop')
class RandomCrop(Layer):
  """Randomly crop the images to target height and width.

  This layer will crop all the images in the same batch to the same cropping
  location.
  By default, random cropping is only applied during training. At inference
  time, the images will be first rescaled to preserve the shorter side, and
  center cropped. If you need to apply random cropping at inference time,
  set `training` to True when calling the layer.

  Input shape:
    4D tensor with shape:
    `(samples, width, height, channels)`, data_format='channels_last'.

  Output shape:
    4D tensor with shape:
    `(samples, target_width, target_height, channels)`.

  Arguments:
    height: Integer, the height of the output shape.
    width: Integer, the width of the output shape.
    seed: Integer. Used to create a random seed.
    name: A string, the name of the layer.
  """

  def __init__(self, height, width, seed=None, name=None, **kwargs):
    self.height = height
    self.width = width
    self.seed = seed
    self._rng = make_generator(self.seed)
    self.input_spec = InputSpec(ndim=4)
    super(RandomCrop, self).__init__(name=name, **kwargs)

  def call(self, inputs, training=True):
    if training is None:
      training = K.learning_phase()

    def random_cropped_inputs():
      """Cropped inputs with stateless random ops."""
      input_shape = array_ops.shape(inputs)
      crop_size = array_ops.stack(
          [input_shape[0], self.width, self.height, input_shape[3]])
      check = control_flow_ops.Assert(
          math_ops.reduce_all(input_shape >= crop_size),
          [self.width, self.height])
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
      input_height_t = input_shape[H_AXIS]
      input_width_t = input_shape[W_AXIS]
      ratio_cond = (input_height_t / input_width_t > (self.height / self.width))
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
          images=inputs, size=array_ops.stack([resized_width, resized_height]))

      img_hd_diff = resized_height - self.height
      img_wd_diff = resized_width - self.width
      bbox_h_start = math_ops.cast(img_hd_diff / 2, dtypes.int32)
      bbox_w_start = math_ops.cast(img_wd_diff / 2, dtypes.int32)
      bbox_begin = array_ops.stack([0, bbox_w_start, bbox_h_start, 0])
      bbox_size = array_ops.stack([-1, self.width, self.height, -1])
      outputs = array_ops.slice(resized_inputs, bbox_begin, bbox_size)
      return outputs

    output = tf_utils.smart_cond(training, random_cropped_inputs,
                                 resize_and_center_cropped_inputs)
    original_shape = inputs.shape.as_list()
    batch_size, num_channels = original_shape[0], original_shape[3]
    output_shape = [batch_size] + [self.width, self.height] + [num_channels]
    output.set_shape(output_shape)
    return output

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    return tensor_shape.TensorShape(
        [input_shape[0], self.width, self.height, input_shape[3]])

  def get_config(self):
    config = {
        'height': self.height,
        'width': self.width,
        'seed': self.seed,
    }
    base_config = super(RandomCrop, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@keras_export('keras.layers.experimental.preprocessing.Rescaling')
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
    name: A string, the name of the layer.
  """

  def __init__(self, scale, name=None, **kwargs):
    self.scale = scale
    super(Rescaling, self).__init__(name=name, **kwargs)

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


HORIZONTAL = 'horizontal'
VERTICAL = 'vertical'
HORIZONTAL_AND_VERTICAL = 'horizontal_and_vertical'


@keras_export('keras.layers.experimental.preprocessing.RandomFlip')
class RandomFlip(Layer):
  """Randomly flip each image horizontally and vertically.

  This layer will flip the images based on the `mode` attribute.
  During inference time, the output will be identical to input. Call the layer
  with `training=True` to flip the input.

  Input shape:
    4D tensor with shape:
    `(samples, width, height, channels)`, data_format='channels_last'.

  Output shape:
    4D tensor with shape:
    `(samples, width, height, channels)`, data_format='channels_last'.

  Attributes:
    mode: String indicating which flip mode to use. Can be "horizontal",
      "vertical", or "horizontal_and_vertical". Defaults to
      "horizontal_and_vertical". "horizontal" is a left-right flip and
      "vertical" is a top-bottom flip.
    seed: Integer. Used to create a random seed.
    name: A string, the name of the layer.
  """

  def __init__(self,
               mode=HORIZONTAL_AND_VERTICAL,
               seed=None,
               name=None,
               **kwargs):
    super(RandomFlip, self).__init__(name=name, **kwargs)
    self.mode = mode
    if mode == HORIZONTAL:
      self.horizontal = True
      self.vertical = False
    elif mode == VERTICAL:
      self.horizontal = False
      self.vertical = True
    elif mode == HORIZONTAL_AND_VERTICAL:
      self.horizontal = True
      self.vertical = True
    else:
      raise ValueError('RandomFlip layer {name} received an unknown mode '
                       'argument {arg}'.format(name=name, arg=mode))
    self.seed = seed
    self._rng = make_generator(self.seed)
    self.input_spec = InputSpec(ndim=4)

  def call(self, inputs, training=True):
    if training is None:
      training = K.learning_phase()

    def random_flipped_inputs():
      flipped_outputs = inputs
      if self.horizontal:
        flipped_outputs = image_ops.random_flip_left_right(flipped_outputs,
                                                           self.seed)
      if self.vertical:
        flipped_outputs = image_ops.random_flip_up_down(
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
        'mode': self.mode,
        'seed': self.seed,
    }
    base_config = super(RandomFlip, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


# TODO(tanzheny): Add examples, here and everywhere.
@keras_export('keras.layers.experimental.preprocessing.RandomTranslation')
class RandomTranslation(Layer):
  """Randomly translate each image during training.

  Arguments:
    height_factor: a float represented as fraction of value, or a tuple
      of size 2 representing lower and upper bound for shifting vertically.
      A negative value means shifting image up, while a positive value
      means shifting image down. When represented as a single positive float,
      this value is used for both the upper and lower bound. For instance,
      `height_factor=(-0.2, 0.3)` results in an output shifted by a random
      amount in the range [-20%, +30%].
      `height_factor=0.2` results in an output height shifted by a random
      amount in the range [-20%, +20%].
    width_factor: a float represented as fraction of value, or a tuple
      of size 2 representing lower and upper bound for shifting horizontally.
      A negative value means shifting image left, while a positive value
      means shifting image right. When represented as a single positive float,
      this value is used for both the upper and lower bound. For instance,
      `width_factor=(-0.2, 0.3)` results in an output shifted left by 20%, and
      shifted right by 30%.
      `width_factor=0.2` results in an output height shifted left or right
      by 20%.
    fill_mode: Points outside the boundaries of the input are filled according
      to the given mode (one of `{'constant', 'reflect', 'wrap'}`).
      - *reflect*: `(d c b a | a b c d | d c b a)`
        The input is extended by reflecting about the edge of the last pixel.
      - *constant*: `(k k k k | a b c d | k k k k)`
        The input is extended by filling all values beyond the edge with the
        same constant value k = 0.
      - *wrap*: `(a b c d | a b c d | a b c d)`
        The input is extended by wrapping around to the opposite edge.
    interpolation: Interpolation mode. Supported values: "nearest", "bilinear".
    seed: Integer. Used to create a random seed.
    name: A string, the name of the layer.

  Input shape:
    4D tensor with shape: `(samples, width, height, channels)`,
      data_format='channels_last'.

  Output shape:
    4D tensor with shape: `(samples, width, height, channels)`,
      data_format='channels_last'.

  Raise:
    ValueError: if either bound is not between [0, 1], or upper bound is
      less than lower bound.
  """

  def __init__(self,
               height_factor,
               width_factor,
               fill_mode='reflect',
               interpolation='bilinear',
               seed=None,
               name=None,
               **kwargs):
    self.height_factor = height_factor
    if isinstance(height_factor, (tuple, list)):
      self.height_lower = height_factor[0]
      self.height_upper = height_factor[1]
    else:
      self.height_lower = -height_factor
      self.height_upper = height_factor
    if self.height_upper < self.height_lower:
      raise ValueError('`height_factor` cannot have upper bound less than '
                       'lower bound, got {}'.format(height_factor))
    if abs(self.height_lower) > 1. or abs(self.height_upper) > 1.:
      raise ValueError('`height_factor` must have values between [-1, 1], '
                       'got {}'.format(height_factor))

    self.width_factor = width_factor
    if isinstance(width_factor, (tuple, list)):
      self.width_lower = width_factor[0]
      self.width_upper = width_factor[1]
    else:
      self.width_lower = -width_factor
      self.width_upper = width_factor
    if self.width_upper < self.width_lower:
      raise ValueError('`width_factor` cannot have upper bound less than '
                       'lower bound, got {}'.format(width_factor))
    if abs(self.width_lower) > 1. or abs(self.width_upper) > 1.:
      raise ValueError('`width_factor` must have values between [-1, 1], '
                       'got {}'.format(width_factor))

    check_fill_mode_and_interpolation(fill_mode, interpolation)

    self.fill_mode = fill_mode
    self.interpolation = interpolation
    self.seed = seed
    self._rng = make_generator(self.seed)
    self.input_spec = InputSpec(ndim=4)
    super(RandomTranslation, self).__init__(name=name, **kwargs)

  def call(self, inputs, training=True):
    if training is None:
      training = K.learning_phase()

    def random_translated_inputs():
      """Translated inputs with random ops."""
      inputs_shape = array_ops.shape(inputs)
      batch_size = inputs_shape[0]
      h_axis, w_axis = H_AXIS, W_AXIS
      img_hd = math_ops.cast(inputs_shape[h_axis], dtypes.float32)
      img_wd = math_ops.cast(inputs_shape[w_axis], dtypes.float32)
      height_translate = self._rng.uniform(
          shape=[batch_size, 1],
          minval=self.height_lower,
          maxval=self.height_upper,
          dtype=dtypes.float32)
      height_translate = height_translate * img_hd
      width_translate = self._rng.uniform(
          shape=[batch_size, 1],
          minval=self.width_lower,
          maxval=self.width_upper,
          dtype=dtypes.float32)
      width_translate = width_translate * img_wd
      translations = math_ops.cast(
          array_ops.concat([width_translate, height_translate], axis=1),
          dtype=dtypes.float32)
      return transform(
          inputs,
          get_translation_matrix(translations),
          interpolation=self.interpolation,
          fill_mode=self.fill_mode)

    output = tf_utils.smart_cond(training, random_translated_inputs,
                                 lambda: inputs)
    output.set_shape(inputs.shape)
    return output

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    config = {
        'height_factor': self.height_factor,
        'width_factor': self.width_factor,
        'fill_mode': self.fill_mode,
        'interpolation': self.interpolation,
        'seed': self.seed,
    }
    base_config = super(RandomTranslation, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


def get_translation_matrix(translations, name=None):
  """Returns projective transform(s) for the given translation(s).

  Args:
    translations: A matrix of 2-element lists representing [dx, dy] to translate
      for each image (for a batch of images).
    name: The name of the op.

  Returns:
    A tensor of shape (num_images, 8) projective transforms which can be given
      to `transform`.
  """
  with ops.name_scope(name, 'translation_matrix'):
    num_translations = array_ops.shape(translations)[0]
    # The translation matrix looks like:
    #     [[1 0 -dx]
    #      [0 1 -dy]
    #      [0 0 1]]
    # where the last entry is implicit.
    # Translation matrices are always float32.
    return array_ops.concat(
        values=[
            array_ops.ones((num_translations, 1), dtypes.float32),
            array_ops.zeros((num_translations, 1), dtypes.float32),
            -translations[:, 0, None],
            array_ops.zeros((num_translations, 1), dtypes.float32),
            array_ops.ones((num_translations, 1), dtypes.float32),
            -translations[:, 1, None],
            array_ops.zeros((num_translations, 2), dtypes.float32),
        ],
        axis=1)


def transform(images,
              transforms,
              fill_mode='reflect',
              interpolation='bilinear',
              output_shape=None,
              name=None):
  """Applies the given transform(s) to the image(s).

  Args:
    images: A tensor of shape (num_images, num_rows, num_columns, num_channels),
      (num_rows, num_columns, num_channels) (HWC), or (num_rows,
      num_columns) (HW). The rank must be statically known (the shape is not
      `TensorShape(None)`.
    transforms: Projective transform matrix/matrices. A vector of length 8 or
      tensor of size N x 8. If one row of transforms is [a0, a1, a2, b0, b1, b2,
      c0, c1], then it maps the *output* point `(x, y)` to a transformed *input*
      point `(x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k)`, where
      `k = c0 x + c1 y + 1`. The transforms are *inverted* compared to the
      transform mapping input points to output points. Note that gradients are
      not backpropagated into transformation parameters.
    fill_mode: Points outside the boundaries of the input are filled according
      to the given mode (one of `{'constant', 'reflect', 'wrap'}`).
    interpolation: Interpolation mode. Supported values: "nearest", "bilinear".
    output_shape: Output dimesion after the transform, [height, width]. If None,
      output is the same size as input image.
    name: The name of the op.

  ## Fill mode.
  Behavior for each valid value is as follows:

  reflect (d c b a | a b c d | d c b a)
  The input is extended by reflecting about the edge of the last pixel.

  constant (k k k k | a b c d | k k k k)
  The input is extended by filling all values beyond the edge with the same
  constant value k = 0.

  wrap (a b c d | a b c d | a b c d)
  The input is extended by wrapping around to the opposite edge.

  Input shape:
    4D tensor with shape: `(samples, height, width, channels)`,
      data_format='channels_last'.
  Output shape:
    4D tensor with shape: `(samples, height, width, channels)`,
      data_format='channels_last'.

  Returns:
    Image(s) with the same type and shape as `images`, with the given
    transform(s) applied. Transformed coordinates outside of the input image
    will be filled with zeros.

  Raises:
    TypeError: If `image` is an invalid type.
    ValueError: If output shape is not 1-D int32 Tensor.
  """
  with ops.name_scope(name, 'transform'):
    if output_shape is None:
      output_shape = array_ops.shape(images)[1:3]
      if not context.executing_eagerly():
        output_shape_value = tensor_util.constant_value(output_shape)
        if output_shape_value is not None:
          output_shape = output_shape_value

    output_shape = ops.convert_to_tensor_v2(
        output_shape, dtypes.int32, name='output_shape')

    if not output_shape.get_shape().is_compatible_with([2]):
      raise ValueError('output_shape must be a 1-D Tensor of 2 elements: '
                       'new_height, new_width, instead got '
                       '{}'.format(output_shape))

    return image_ops.image_projective_transform_v2(
        images,
        output_shape=output_shape,
        transforms=transforms,
        fill_mode=fill_mode.upper(),
        interpolation=interpolation.upper())


def get_rotation_matrix(angles, image_height, image_width, name=None):
  """Returns projective transform(s) for the given angle(s).

  Args:
    angles: A scalar angle to rotate all images by, or (for batches of images) a
      vector with an angle to rotate each image in the batch. The rank must be
      statically known (the shape is not `TensorShape(None)`).
    image_height: Height of the image(s) to be transformed.
    image_width: Width of the image(s) to be transformed.
    name: The name of the op.

  Returns:
    A tensor of shape (num_images, 8). Projective transforms which can be given
      to operation `image_projective_transform_v2`. If one row of transforms is
       [a0, a1, a2, b0, b1, b2, c0, c1], then it maps the *output* point
       `(x, y)` to a transformed *input* point
       `(x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k)`,
       where `k = c0 x + c1 y + 1`.
  """
  with ops.name_scope(name, 'rotation_matrix'):
    x_offset = ((image_width - 1) - (math_ops.cos(angles) *
                                     (image_width - 1) - math_ops.sin(angles) *
                                     (image_height - 1))) / 2.0
    y_offset = ((image_height - 1) - (math_ops.sin(angles) *
                                      (image_width - 1) + math_ops.cos(angles) *
                                      (image_height - 1))) / 2.0
    num_angles = array_ops.shape(angles)[0]
    return array_ops.concat(
        values=[
            math_ops.cos(angles)[:, None],
            -math_ops.sin(angles)[:, None],
            x_offset[:, None],
            math_ops.sin(angles)[:, None],
            math_ops.cos(angles)[:, None],
            y_offset[:, None],
            array_ops.zeros((num_angles, 2), dtypes.float32),
        ],
        axis=1)


@keras_export('keras.layers.experimental.preprocessing.RandomRotation')
class RandomRotation(Layer):
  """Randomly rotate each image.

  By default, random rotations are only applied during training.
  At inference time, the layer does nothing. If you need to apply random
  rotations at inference time, set `training` to True when calling the layer.

  Input shape:
    4D tensor with shape:
    `(samples, height, width, channels)`, data_format='channels_last'.

  Output shape:
    4D tensor with shape:
    `(samples, height, width, channels)`, data_format='channels_last'.

  Attributes:
    factor: a float represented as fraction of 2pi, or a tuple of size
      2 representing lower and upper bound for rotating clockwise and
      counter-clockwise. A positive values means rotating counter clock-wise,
      while a negative value means clock-wise. When represented as a single
      float, this value is used for both the upper and lower bound. For
      instance, `factor=(-0.2, 0.3)` results in an output
      rotation by a random amount in the range `[-20% * 2pi, 30% * 2pi]`.
      `factor=0.2` results in an output rotating by a random amount in the range
      `[-20% * 2pi, 20% * 2pi]`.
    fill_mode: Points outside the boundaries of the input are filled according
      to the given mode (one of `{'constant', 'reflect', 'wrap'}`).
      - *reflect*: `(d c b a | a b c d | d c b a)`
        The input is extended by reflecting about the edge of the last pixel.
      - *constant*: `(k k k k | a b c d | k k k k)`
        The input is extended by filling all values beyond the edge with the
        same constant value k = 0.
      - *wrap*: `(a b c d | a b c d | a b c d)`
    interpolation: Interpolation mode. Supported values: "nearest", "bilinear".
    seed: Integer. Used to create a random seed.
    name: A string, the name of the layer.

  Input shape:
    4D tensor with shape: `(samples, height, width, channels)`,
      data_format='channels_last'.
  Output shape:
    4D tensor with shape: `(samples, height, width, channels)`,
      data_format='channels_last'.

  Raise:
    ValueError: if either bound is not between [0, 1], or upper bound is
      less than lower bound.
  """

  def __init__(self,
               factor,
               fill_mode='reflect',
               interpolation='bilinear',
               seed=None,
               name=None,
               **kwargs):
    self.factor = factor
    if isinstance(factor, (tuple, list)):
      self.lower = factor[0]
      self.upper = factor[1]
    else:
      self.lower = -factor
      self.upper = factor
    if self.upper < self.lower:
      raise ValueError('Factor cannot have negative values, '
                       'got {}'.format(factor))
    check_fill_mode_and_interpolation(fill_mode, interpolation)
    self.fill_mode = fill_mode
    self.interpolation = interpolation
    self.seed = seed
    self._rng = make_generator(self.seed)
    self.input_spec = InputSpec(ndim=4)
    super(RandomRotation, self).__init__(name=name, **kwargs)

  def call(self, inputs, training=True):
    if training is None:
      training = K.learning_phase()

    def random_rotated_inputs():
      """Rotated inputs with random ops."""
      inputs_shape = array_ops.shape(inputs)
      batch_size = inputs_shape[0]
      img_hd = math_ops.cast(inputs_shape[H_AXIS], dtypes.float32)
      img_wd = math_ops.cast(inputs_shape[W_AXIS], dtypes.float32)
      min_angle = self.lower * 2. * np.pi
      max_angle = self.upper * 2. * np.pi
      angles = self._rng.uniform(
          shape=[batch_size], minval=min_angle, maxval=max_angle)
      return transform(
          inputs,
          get_rotation_matrix(angles, img_hd, img_wd),
          fill_mode=self.fill_mode,
          interpolation=self.interpolation)

    output = tf_utils.smart_cond(training, random_rotated_inputs,
                                 lambda: inputs)
    output.set_shape(inputs.shape)
    return output

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    config = {
        'factor': self.factor,
        'fill_mode': self.fill_mode,
        'interpolation': self.interpolation,
        'seed': self.seed,
    }
    base_config = super(RandomRotation, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class RandomZoom(Layer):
  """Randomly zoom each image during training.

  Arguments:
    height_factor: a float represented as fraction of value, or a tuple
      of size 2 representing lower and upper bound for zooming vertically.
      When represented as a single float, this value is used for both the
      upper and lower bound. A positive value means zooming out, while a
      negative value means zooming in.
      For instance, `height_factor=(0.2, 0.3)` result in an output zoomed out
      by a random amount in the range [+20%, +30%].
      `height_factor=(-0.3, -0.2)` result in an output zoomed in by a random
      amount in the range [+20%, +30%].
    width_factor: a float represented as fraction of value, or a tuple
      of size 2 representing lower and upper bound for zooming horizontally.
      When represented as a single float, this value is used for both the
      upper and lower bound.
      For instance, `width_factor=(0.2, 0.3)` result in an output zooming out
      between 20% to 30%.
      `width_factor=(-0.3, -0.2)` result in an output zooming in between 20%
      to 30%.
    fill_mode: Points outside the boundaries of the input are filled according
      to the given mode (one of `{'constant', 'reflect', 'wrap'}`).
      - *reflect*: `(d c b a | a b c d | d c b a)`
        The input is extended by reflecting about the edge of the last pixel.
      - *constant*: `(k k k k | a b c d | k k k k)`
        The input is extended by filling all values beyond the edge with the
        same constant value k = 0.
      - *wrap*: `(a b c d | a b c d | a b c d)`
    interpolation: Interpolation mode. Supported values: "nearest", "bilinear".
    seed: Integer. Used to create a random seed.
    name: A string, the name of the layer.

  Input shape:
    4D tensor with shape:
    `(samples, height, width, channels)`, data_format='channels_last'.

  Output shape:
    4D tensor with shape:
    `(samples, height, width, channels)`, data_format='channels_last'.

  Raise:
    ValueError: if lower bound is not between [0, 1], or upper bound is
      negative.
  """

  def __init__(self,
               height_factor,
               width_factor,
               fill_mode='reflect',
               interpolation='bilinear',
               seed=None,
               name=None,
               **kwargs):
    self.height_factor = height_factor
    if isinstance(height_factor, (tuple, list)):
      self.height_lower = height_factor[0]
      self.height_upper = height_factor[1]
    else:
      self.height_lower = -height_factor
      self.height_upper = height_factor

    if abs(self.height_lower) > 1. or abs(self.height_upper) > 1.:
      raise ValueError('`height_factor` must have values between [-1, 1], '
                       'got {}'.format(height_factor))

    self.width_factor = width_factor
    if isinstance(width_factor, (tuple, list)):
      self.width_lower = width_factor[0]
      self.width_upper = width_factor[1]
    else:
      self.width_lower = -width_factor
      self.width_upper = width_factor

    if self.width_lower < -1. or self.width_upper < -1.:
      raise ValueError('`width_factor` must have values larger than -1, '
                       'got {}'.format(width_factor))

    check_fill_mode_and_interpolation(fill_mode, interpolation)

    self.fill_mode = fill_mode
    self.interpolation = interpolation
    self.seed = seed
    self._rng = make_generator(self.seed)
    self.input_spec = InputSpec(ndim=4)
    super(RandomZoom, self).__init__(name=name, **kwargs)

  def call(self, inputs, training=True):
    if training is None:
      training = K.learning_phase()

    def random_zoomed_inputs():
      """Zoomed inputs with random ops."""
      inputs_shape = array_ops.shape(inputs)
      batch_size = inputs_shape[0]
      img_hd = math_ops.cast(inputs_shape[H_AXIS], dtypes.float32)
      img_wd = math_ops.cast(inputs_shape[W_AXIS], dtypes.float32)
      height_zoom = self._rng.uniform(
          shape=[batch_size, 1],
          minval=1. + self.height_lower,
          maxval=1. + self.height_upper)
      width_zoom = self._rng.uniform(
          shape=[batch_size, 1],
          minval=1. + self.width_lower,
          maxval=1. + self.width_upper)
      zooms = math_ops.cast(
          array_ops.concat([width_zoom, height_zoom], axis=1),
          dtype=dtypes.float32)
      return transform(
          inputs, get_zoom_matrix(zooms, img_hd, img_wd),
          fill_mode=self.fill_mode,
          interpolation=self.interpolation)

    output = tf_utils.smart_cond(training, random_zoomed_inputs,
                                 lambda: inputs)
    output.set_shape(inputs.shape)
    return output

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    config = {
        'height_factor': self.height_factor,
        'width_factor': self.width_factor,
        'fill_mode': self.fill_mode,
        'interpolation': self.interpolation,
        'seed': self.seed,
    }
    base_config = super(RandomZoom, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


def get_zoom_matrix(zooms, image_height, image_width, name=None):
  """Returns projective transform(s) for the given zoom(s).

  Args:
    zooms: A matrix of 2-element lists representing [zx, zy] to zoom
      for each image (for a batch of images).
    image_height: Height of the image(s) to be transformed.
    image_width: Width of the image(s) to be transformed.
    name: The name of the op.

  Returns:
    A tensor of shape (num_images, 8). Projective transforms which can be given
      to operation `image_projective_transform_v2`. If one row of transforms is
       [a0, a1, a2, b0, b1, b2, c0, c1], then it maps the *output* point
       `(x, y)` to a transformed *input* point
       `(x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k)`,
       where `k = c0 x + c1 y + 1`.
  """
  with ops.name_scope(name, 'zoom_matrix'):
    num_zooms = array_ops.shape(zooms)[0]
    # The zoom matrix looks like:
    #     [[zx 0 0]
    #      [0 zy 0]
    #      [0 0 1]]
    # where the last entry is implicit.
    # Zoom matrices are always float32.
    x_offset = ((image_width - 1.) / 2.0) * (1.0 - zooms[:, 0, None])
    y_offset = ((image_height - 1.) / 2.0) * (1.0 - zooms[:, 1, None])
    return array_ops.concat(
        values=[
            zooms[:, 0, None],
            array_ops.zeros((num_zooms, 1), dtypes.float32),
            x_offset,
            array_ops.zeros((num_zooms, 1), dtypes.float32),
            zooms[:, 1, None],
            y_offset,
            array_ops.zeros((num_zooms, 2), dtypes.float32),
        ],
        axis=1)


@keras_export('keras.layers.experimental.preprocessing.RandomContrast')
class RandomContrast(Layer):
  """Adjust the contrast of an image or images by a random factor.

  Contrast is adjusted independently for each channel of each image during
  training.

  For each channel, this layer computes the mean of the image pixels in the
  channel and then adjusts each component `x` of each pixel to
  `(x - mean) * contrast_factor + mean`.

  Input shape:
    4D tensor with shape:
    `(samples, height, width, channels)`, data_format='channels_last'.

  Output shape:
    4D tensor with shape:
    `(samples, height, width, channels)`, data_format='channels_last'.

  Attributes:
    factor: a positive float represented as fraction of value, or a tuple of
      size 2 representing lower and upper bound. When represented as a single
      float, lower = upper. The contrast factor will be randomly picked between
      [1.0 - lower, 1.0 + upper].
    seed: Integer. Used to create a random seed.
    name: A string, the name of the layer.

  Raise:
    ValueError: if lower bound is not between [0, 1], or upper bound is
      negative.
  """

  def __init__(self, factor, seed=None, name=None, **kwargs):
    self.factor = factor
    if isinstance(factor, (tuple, list)):
      self.lower = factor[0]
      self.upper = factor[1]
    else:
      self.lower = self.upper = factor
    if self.lower < 0. or self.upper < 0. or self.lower > 1.:
      raise ValueError('Factor cannot have negative values, '
                       'got {}'.format(factor))
    self.seed = seed
    self.input_spec = InputSpec(ndim=4)
    super(RandomContrast, self).__init__(name=name, **kwargs)

  def call(self, inputs, training=True):
    if training is None:
      training = K.learning_phase()

    def random_contrasted_inputs():
      return image_ops.random_contrast(inputs, 1. - self.lower, 1. + self.upper,
                                       self.seed)

    output = tf_utils.smart_cond(training, random_contrasted_inputs,
                                 lambda: inputs)
    output.set_shape(inputs.shape)
    return output

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    config = {
        'factor': self.factor,
        'seed': self.seed,
    }
    base_config = super(RandomContrast, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@keras_export('keras.layers.experimental.preprocessing.RandomHeight')
class RandomHeight(Layer):
  """Randomly vary the height of a batch of images during training.

  Adjusts the height of a batch of images by a random factor. The input
  should be a 4-D tensor in the "channels_last" image data format.

  By default, this layer is inactive during inference.

  Arguments:
    factor: A positive float (fraction of original height), or a tuple of size 2
      representing lower and upper bound for resizing vertically. When
      represented as a single float, this value is used for both the upper and
      lower bound. For instance, `factor=(0.2, 0.3)` results in an output with
      height changed by a random amount in the range `[20%, 30%]`.
      `factor=(-0.2, 0.3)` results in an output with height changed by a random
      amount in the range `[-20%, +30%]. `factor=0.2` results in an output with
      height changed by a random amount in the range `[-20%, +20%]`.
    interpolation: String, the interpolation method. Defaults to `bilinear`.
      Supports `bilinear`, `nearest`, `bicubic`, `area`, `lanczos3`, `lanczos5`,
      `gaussian`, `mitchellcubic`
    seed: Integer. Used to create a random seed.
    name: A string, the name of the layer.

  Input shape:
    4D tensor with shape: `(samples, width, height, channels)`
      (data_format='channels_last').
  Output shape:
    4D tensor with shape: `(samples, width, random_height, channels)`.
  """

  def __init__(self,
               factor,
               interpolation='bilinear',
               seed=None,
               name=None,
               **kwargs):
    self.factor = factor
    if isinstance(factor, (tuple, list)):
      self.height_lower = factor[0]
      self.height_upper = factor[1]
    else:
      self.height_lower = -factor
      self.height_upper = factor

    if self.height_upper < self.height_lower:
      raise ValueError('`factor` cannot have upper bound less than '
                       'lower bound, got {}'.format(factor))
    if self.height_lower < -1. or self.height_upper < -1.:
      raise ValueError('`factor` must have values larger than -1, '
                       'got {}'.format(factor))
    self.interpolation = interpolation
    self._interpolation_method = get_interpolation(interpolation)
    self.input_spec = InputSpec(ndim=4)
    self.seed = seed
    self._rng = make_generator(self.seed)
    super(RandomHeight, self).__init__(name=name, **kwargs)

  def call(self, inputs, training=True):
    if training is None:
      training = K.learning_phase()

    def random_height_inputs():
      """Inputs height-adjusted with random ops."""
      inputs_shape = array_ops.shape(inputs)
      img_hd = math_ops.cast(inputs_shape[H_AXIS], dtypes.float32)
      img_wd = inputs_shape[W_AXIS]
      height_factor = self._rng.uniform(
          shape=[],
          minval=(1.0 + self.height_lower),
          maxval=(1.0 + self.height_upper))
      adjusted_height = math_ops.cast(height_factor * img_hd, dtypes.int32)
      adjusted_size = array_ops.stack([img_wd, adjusted_height])
      output = image_ops.resize_images_v2(
          images=inputs, size=adjusted_size, method=self._interpolation_method)
      original_shape = inputs.shape.as_list()
      output_shape = original_shape[0:2] + [None] + [original_shape[3]]
      output.set_shape(output_shape)
      return output

    return tf_utils.smart_cond(training, random_height_inputs, lambda: inputs)

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    return tensor_shape.TensorShape(
        [input_shape[0], input_shape[1], None, input_shape[3]])

  def get_config(self):
    config = {
        'factor': self.factor,
        'interpolation': self.interpolation,
        'seed': self.seed,
    }
    base_config = super(RandomHeight, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@keras_export('keras.layers.experimental.preprocessing.RandomWidth')
class RandomWidth(Layer):
  """Randomly vary the width of a batch of images during training.

  Adjusts the width of a batch of images by a random factor. The input
  should be a 4-D tensor in the "channels_last" image data format.

  By default, this layer is inactive during inference.

  Arguments:
    factor: A positive float (fraction of original height), or a tuple of size 2
      representing lower and upper bound for resizing vertically. When
      represented as a single float, this value is used for both the upper and
      lower bound. For instance, `factor=(0.2, 0.3)` results in an output with
      width changed by a random amount in the range `[20%, 30%]`.
      `factor=(-0.2, 0.3)` results in an output with width changed by a random
      amount in the range `[-20%, +30%]. `factor=0.2` results in an output with
      width changed by a random amount in the range `[-20%, +20%]`.
    interpolation: String, the interpolation method. Defaults to `bilinear`.
      Supports `bilinear`, `nearest`, `bicubic`, `area`, `lanczos3`, `lanczos5`,
      `gaussian`, `mitchellcubic`
    seed: Integer. Used to create a random seed.
    name: A string, the name of the layer.

  Input shape:
    4D tensor with shape:
    `(samples, width, height, channels)` (data_format='channels_last').

  Output shape:
    4D tensor with shape:
    `(samples, random_width, height, channels)`.
  """

  def __init__(self,
               factor,
               interpolation='bilinear',
               seed=None,
               name=None,
               **kwargs):
    self.factor = factor
    if isinstance(factor, (tuple, list)):
      self.width_lower = factor[0]
      self.width_upper = factor[1]
    else:
      self.width_lower = -factor
      self.width_upper = factor
    if self.width_upper < self.width_lower:
      raise ValueError('`factor` cannot have upper bound less than '
                       'lower bound, got {}'.format(factor))
    if self.width_lower < -1. or self.width_upper < -1.:
      raise ValueError('`factor` must have values larger than -1, '
                       'got {}'.format(factor))
    self.interpolation = interpolation
    self._interpolation_method = get_interpolation(interpolation)
    self.input_spec = InputSpec(ndim=4)
    self.seed = seed
    self._rng = make_generator(self.seed)
    super(RandomWidth, self).__init__(name=name, **kwargs)

  def call(self, inputs, training=True):
    if training is None:
      training = K.learning_phase()

    def random_width_inputs():
      """Inputs width-adjusted with random ops."""
      inputs_shape = array_ops.shape(inputs)
      img_hd = inputs_shape[H_AXIS]
      img_wd = math_ops.cast(inputs_shape[W_AXIS], dtypes.float32)
      width_factor = self._rng.uniform(
          shape=[],
          minval=(1.0 + self.width_lower),
          maxval=(1.0 + self.width_upper))
      adjusted_width = math_ops.cast(width_factor * img_wd, dtypes.int32)
      adjusted_size = array_ops.stack([adjusted_width, img_hd])
      output = image_ops.resize_images_v2(
          images=inputs, size=adjusted_size, method=self._interpolation_method)
      original_shape = inputs.shape.as_list()
      output_shape = [original_shape[0]] + [None] + original_shape[2:4]
      output.set_shape(output_shape)
      return output

    return tf_utils.smart_cond(training, random_width_inputs, lambda: inputs)

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    return tensor_shape.TensorShape(
        [input_shape[0], None, input_shape[2], input_shape[3]])

  def get_config(self):
    config = {
        'factor': self.factor,
        'interpolation': self.interpolation,
        'seed': self.seed,
    }
    base_config = super(RandomWidth, self).get_config()
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
