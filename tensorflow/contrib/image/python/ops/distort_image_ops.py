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
"""Python layer for distort_image_ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.util import loader
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import resource_loader

_distort_image_ops = loader.load_op_library(
    resource_loader.get_path_to_datafile('_distort_image_ops.so'))


# pylint: disable=invalid-name
def random_hsv_in_yiq(image,
                      max_delta_hue=0,
                      lower_saturation=1,
                      upper_saturation=1,
                      lower_value=1,
                      upper_value=1,
                      seed=None):
  """Adjust hue, saturation, value of an RGB image randomly in YIQ color space.

  Equivalent to `adjust_yiq_hsv()` but uses a `delta_h` randomly
  picked in the interval `[-max_delta_hue, max_delta_hue]`, a `scale_saturation`
  randomly picked in the interval `[lower_saturation, upper_saturation]`, and
  a `scale_value` randomly picked in the interval
  `[lower_saturation, upper_saturation]`.

  Args:
    image: RGB image or images. Size of the last dimension must be 3.
    max_delta_hue: float. Maximum value for the random delta_hue. Passing 0
                   disables adjusting hue.
    lower_saturation: float.  Lower bound for the random scale_saturation.
    upper_saturation: float.  Upper bound for the random scale_saturation.
    lower_value: float.  Lower bound for the random scale_value.
    upper_value: float.  Upper bound for the random scale_value.
    seed: An operation-specific seed. It will be used in conjunction
      with the graph-level seed to determine the real seeds that will be
      used in this operation. Please see the documentation of
      set_random_seed for its interaction with the graph-level random seed.

  Returns:
    3-D float tensor of shape `[height, width, channels]`.

  Raises:
    ValueError: if `max_delta`, `lower_saturation`, `upper_saturation`,
               `lower_value`, or `upper_Value` is invalid.
  """
  if max_delta_hue < 0:
    raise ValueError('max_delta must be non-negative.')

  if lower_saturation < 0:
    raise ValueError('lower_saturation must be non-negative.')

  if lower_value < 0:
    raise ValueError('lower_value must be non-negative.')

  if lower_saturation > upper_saturation:
    raise ValueError('lower_saturation must be < upper_saturation.')

  if lower_value > upper_value:
    raise ValueError('lower_value must be < upper_value.')

  if max_delta_hue == 0:
    delta_hue = 0
  else:
    delta_hue = random_ops.random_uniform(
        [], -max_delta_hue, max_delta_hue, seed=seed)
  if lower_saturation == upper_saturation:
    scale_saturation = lower_saturation
  else:
    scale_saturation = random_ops.random_uniform(
        [], lower_saturation, upper_saturation, seed=seed)
  if lower_value == upper_value:
    scale_value = lower_value
  else:
    scale_value = random_ops.random_uniform(
        [], lower_value, upper_value, seed=seed)
  return adjust_hsv_in_yiq(image, delta_hue, scale_saturation, scale_value)


def adjust_hsv_in_yiq(image,
                      delta_hue=0,
                      scale_saturation=1,
                      scale_value=1,
                      name=None):
  """Adjust hue, saturation, value of an RGB image in YIQ color space.

  This is a convenience method that converts an RGB image to float
  representation, converts it to YIQ, rotates the color around the Y channel by
  delta_hue in radians, scales the chrominance channels (I, Q) by
  scale_saturation, scales all channels (Y, I, Q) by scale_value,
  converts back to RGB, and then back to the original data type.

  `image` is an RGB image.  The image hue is adjusted by converting the
  image to YIQ, rotating around the luminance channel (Y) by
  `delta_hue` in radians, multiplying the chrominance channels (I, Q)  by
  `scale_saturation`, and multiplying all channels (Y, I, Q)  by
  `scale_value`.  The image is then converted back to RGB.

  Args:
    image: RGB image or images. Size of the last dimension must be 3.
    delta_hue: float, the hue rotation amount, in radians.
    scale_saturation: float, factor to multiply the saturation by.
    scale_value: float, factor to multiply the value by.
    name: A name for this operation (optional).

  Returns:
    Adjusted image(s), same shape and DType as `image`.
  """
  with ops.name_scope(name, 'adjust_hsv_in_yiq', [image]) as name:
    image = ops.convert_to_tensor(image, name='image')
    # Remember original dtype to so we can convert back if needed
    orig_dtype = image.dtype
    flt_image = image_ops.convert_image_dtype(image, dtypes.float32)

    rgb_altered = _distort_image_ops.adjust_hsv_in_yiq(
        flt_image, delta_hue, scale_saturation, scale_value)

    return image_ops.convert_image_dtype(rgb_altered, orig_dtype)
