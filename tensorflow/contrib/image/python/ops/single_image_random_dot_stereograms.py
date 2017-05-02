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
"""Python layer for image_ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.util import loader
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import resource_loader

_sirds_ops = loader.load_op_library(
    resource_loader.get_path_to_datafile("_single_image_random_dot_stereograms.so"))

def single_image_random_dot_stereograms(depth_values,
                                        hidden_surface_removal=None,
                                        convergence_dots_size=None,
                                        dots_per_inch=None,
                                        eye_separation=None, mu=None,
                                        normalize=None, normalize_max=None,
                                        normalize_min=None,
                                        boarder_level=None,
                                        number_colors=None,
                                        generation_mode=None,
                                        output_image_shape=None,
                                        output_data_window=None):
  r"""Output a RandomDotStereogram Tensor of shape "output_image_shape" for export via encode_PNG or encode_JPG OP.

  Based upon:
  'http://www.learningace.com/doc/4331582/b6ab058d1e206d68ab60e4e1ead2fe6e/sirds-paper'

  Example use which outputs a SIRDS image as picture_out.png:
  img=[[1,2,3,3,2,1],
       [1,2,3,4,5,2],
       [1,2,3,4,5,3],
       [1,2,3,4,5,4],
       [6,5,4,4,5,5]]

  session = tf.InteractiveSession()

  sirds = single_image_random_dot_stereograms(img,convergence_dots_size=8,number_colors=256,normalize=True)

  out = sirds.eval()

  png = tf.image.encode_png(out).eval()

  with open('picture_out.png', 'wb') as f:
      f.write(png)

  Args:
    depth_values: A `Tensor`. Must be one of the following types: `float64`, `float32`, `int64`, `int32`.
      Z values of data to encode into "output_data_window" window, lower further away {0.0 floor(far), 1.0 ceiling(near) after normalization}, must be rank 2
    hidden_surface_removal: An optional `bool`. Defaults to `True`.
      Activate hidden surface removal (True)
    convergence_dots_size: An optional `int`. Defaults to `8`.
      Black dot size in pixels to help view converge image, drawn on bottom of image (8 pixels)
    dots_per_inch: An optional `int`. Defaults to `72`.
      Output device in dots/inch (72 default)
    eye_separation: An optional `float`. Defaults to `2.5`.
      Separation between eyes in inches (2.5 inchs)
    mu: An optional `float`. Defaults to `0.3333`.
      Depth of field, Fraction of viewing distance (1/3 = .3333)
    normalize: An optional `bool`. Defaults to `True`.
      Normalize input data to [0.0, 1.0] (True)
    normalize_max: An optional `float`. Defaults to `-100`.
      Fix MAX value for Normalization (0.0) - if < MIN, autoscale
    normalize_min: An optional `float`. Defaults to `100`.
      Fix MIN value for Normalization (0.0) - if > MAX, autoscale
    boarder_level: An optional `float`. Defaults to `0`.
      Value of board in depth 0.0 {far} to 1.0 {near} (0.0)
    number_colors: An optional `int`. Defaults to `256`.
      2 (Black & White),256 (grayscale), and Numbers > 256 (Full Color) are all that are supported currently
    generation_mode: An optional `string`. Defaults to `"SIRDS"`.
      Mode for Stereogram
      SIRDS - 2 color stereogram (Default)
    output_image_shape: An optional `tf.TensorShape` or list of `ints`. Defaults to `[1024, 768, 1]`.
      Output size of returned image in X,Y, Channels 1-grayscale, 3 color (1024, 768, 1), channels will be updated to 3 if number_colors > 256
    output_data_window: An optional `tf.TensorShape` or list of `ints`. Defaults to `[1022, 757]`.
      Size of "DATA" window, must be equal to or smaller than output_image_shape, will be centered
      and use convergence_dots_size for best fit to avoid overlap if possible
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `uint8`.
    returns a Tensor of size output_image_shape with depth_values encoded into image
  """
  result = _sirds_ops.single_image_random_dot_stereograms( depth_values=depth_values,
                                hidden_surface_removal=hidden_surface_removal,
                                convergence_dots_size=convergence_dots_size,
                                dots_per_inch=dots_per_inch,
                                eye_separation=eye_separation, mu=mu,
                                normalize=normalize,
                                normalize_max=normalize_max,
                                normalize_min=normalize_min,
                                boarder_level=boarder_level,
                                number_colors=number_colors,
                                generation_mode=generation_mode,
                                output_image_shape=output_image_shape,
                                output_data_window=output_data_window)
  return result


ops.NotDifferentiable("SingleImageRandomDotStereograms")
