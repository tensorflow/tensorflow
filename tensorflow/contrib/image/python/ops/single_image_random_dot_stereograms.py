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

from tensorflow.contrib.image.ops import gen_single_image_random_dot_stereograms_ops
from tensorflow.contrib.util import loader
from tensorflow.python.framework import ops
from tensorflow.python.platform import resource_loader

_sirds_ops = loader.load_op_library(
    resource_loader.get_path_to_datafile(
        "_single_image_random_dot_stereograms.so"))


def single_image_random_dot_stereograms(depth_values,
                                        hidden_surface_removal=None,
                                        convergence_dots_size=None,
                                        dots_per_inch=None,
                                        eye_separation=None,
                                        mu=None,
                                        normalize=None,
                                        normalize_max=None,
                                        normalize_min=None,
                                        border_level=None,
                                        number_colors=None,
                                        output_image_shape=None,
                                        output_data_window=None):
  """Output a RandomDotStereogram Tensor for export via encode_PNG/JPG OP.

  Given the 2-D tensor 'depth_values' with encoded Z values, this operation
  will encode 3-D data into a 2-D image.  The output of this Op is suitable
  for the encode_PNG/JPG ops.  Be careful with image compression as this may
  corrupt the encode 3-D data witin the image.

  Based upon [this
  paper](http://www.learningace.com/doc/4331582/b6ab058d1e206d68ab60e4e1ead2fe6e/sirds-paper).

  This outputs a SIRDS image as picture_out.png:

  ```python
  img=[[1,2,3,3,2,1],
       [1,2,3,4,5,2],
       [1,2,3,4,5,3],
       [1,2,3,4,5,4],
       [6,5,4,4,5,5]]
  session = tf.InteractiveSession()
  sirds = single_image_random_dot_stereograms(
      img,
      convergence_dots_size=8,
      number_colors=256,normalize=True)

  out = sirds.eval()
  png = tf.image.encode_png(out).eval()
  with open('picture_out.png', 'wb') as f:
    f.write(png)
  ```

  Args:
    depth_values: A `Tensor`. Must be one of the following types:
      `float64`, `float32`, `int64`, `int32`.  Z values of data to encode
      into 'output_data_window' window, lower further away {0.0 floor(far),
      1.0 ceiling(near) after norm}, must be 2-D tensor
    hidden_surface_removal: An optional `bool`. Defaults to `True`.
      Activate hidden surface removal
    convergence_dots_size: An optional `int`. Defaults to `8`.
      Black dot size in pixels to help view converge image, drawn on bottom
      of the image
    dots_per_inch: An optional `int`. Defaults to `72`.
      Output device in dots/inch
    eye_separation: An optional `float`. Defaults to `2.5`.
      Separation between eyes in inches
    mu: An optional `float`. Defaults to `0.3333`.
      Depth of field, Fraction of viewing distance (eg. 1/3 = 0.3333)
    normalize: An optional `bool`. Defaults to `True`.
      Normalize input data to [0.0, 1.0]
    normalize_max: An optional `float`. Defaults to `-100`.
      Fix MAX value for Normalization (0.0) - if < MIN, autoscale
    normalize_min: An optional `float`. Defaults to `100`.
      Fix MIN value for Normalization (0.0) - if > MAX, autoscale
    border_level: An optional `float`. Defaults to `0`.
      Value of bord in depth 0.0 {far} to 1.0 {near}
    number_colors: An optional `int`. Defaults to `256`. 2 (Black &
      White), 256 (grayscale), and Numbers > 256 (Full Color) are
      supported
    output_image_shape: An optional `tf.TensorShape` or list of `ints`.
      Defaults to shape `[1024, 768, 1]`. Defines output shape of returned
      image in '[X,Y, Channels]' 1-grayscale, 3 color; channels will be
      updated to 3 if number_colors > 256
    output_data_window: An optional `tf.TensorShape` or list of `ints`.
      Defaults to `[1022, 757]`. Size of "DATA" window, must be equal to or
      smaller than `output_image_shape`, will be centered and use
      `convergence_dots_size` for best fit to avoid overlap if possible

  Returns:
    A `Tensor` of type `uint8` of shape 'output_image_shape' with encoded
    'depth_values'
  """

  result = gen_single_image_random_dot_stereograms_ops.single_image_random_dot_stereograms(  # pylint: disable=line-too-long
      depth_values=depth_values,
      hidden_surface_removal=hidden_surface_removal,
      convergence_dots_size=convergence_dots_size,
      dots_per_inch=dots_per_inch,
      eye_separation=eye_separation,
      mu=mu,
      normalize=normalize,
      normalize_max=normalize_max,
      normalize_min=normalize_min,
      border_level=border_level,
      number_colors=number_colors,
      output_image_shape=output_image_shape,
      output_data_window=output_data_window)
  return result


ops.NotDifferentiable("SingleImageRandomDotStereograms")
