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

# pylint: disable=g-short-docstring-punctuation
"""## Encoding and Decoding

TensorFlow provides Ops to decode and encode JPEG and PNG formats.  Encoded
images are represented by scalar string Tensors, decoded images by 3-D uint8
tensors of shape `[height, width, channels]`. (PNG also supports uint16.)

The encode and decode Ops apply to one image at a time.  Their input and output
are all of variable size.  If you need fixed size images, pass the output of
the decode Ops to one of the cropping and resizing Ops.

Note: The PNG encode and decode Ops support RGBA, but the conversions Ops
presently only support RGB, HSV, and GrayScale. Presently, the alpha channel has
to be stripped from the image and re-attached using slicing ops.

@@decode_gif

@@decode_jpeg
@@encode_jpeg

@@decode_png
@@encode_png

@@decode_image

## Resizing

The resizing Ops accept input images as tensors of several types.  They always
output resized images as float32 tensors.

The convenience function [`resize_images()`](#resize_images) supports both 4-D
and 3-D tensors as input and output.  4-D tensors are for batches of images,
3-D tensors for individual images.

Other resizing Ops only support 4-D batches of images as input:
[`resize_area`](#resize_area), [`resize_bicubic`](#resize_bicubic),
[`resize_bilinear`](#resize_bilinear),
[`resize_nearest_neighbor`](#resize_nearest_neighbor).

Example:

```python
# Decode a JPG image and resize it to 299 by 299 using default method.
image = tf.image.decode_jpeg(...)
resized_image = tf.image.resize_images(image, [299, 299])
```

@@resize_images

@@resize_area
@@resize_bicubic
@@resize_bilinear
@@resize_nearest_neighbor

## Cropping

@@resize_image_with_crop_or_pad

@@central_crop
@@pad_to_bounding_box
@@crop_to_bounding_box
@@extract_glimpse

@@crop_and_resize

## Flipping, Rotating and Transposing

@@flip_up_down
@@random_flip_up_down

@@flip_left_right
@@random_flip_left_right

@@transpose_image

@@rot90

## Converting Between Colorspaces.

Image ops work either on individual images or on batches of images, depending on
the shape of their input Tensor.

If 3-D, the shape is `[height, width, channels]`, and the Tensor represents one
image. If 4-D, the shape is `[batch_size, height, width, channels]`, and the
Tensor represents `batch_size` images.

Currently, `channels` can usefully be 1, 2, 3, or 4. Single-channel images are
grayscale, images with 3 channels are encoded as either RGB or HSV. Images
with 2 or 4 channels include an alpha channel, which has to be stripped from the
image before passing the image to most image processing functions (and can be
re-attached later).

Internally, images are either stored in as one `float32` per channel per pixel
(implicitly, values are assumed to lie in `[0,1)`) or one `uint8` per channel
per pixel (values are assumed to lie in `[0,255]`).

TensorFlow can convert between images in RGB or HSV. The conversion functions
work only on float images, so you need to convert images in other formats using
[`convert_image_dtype`](#convert-image-dtype).

Example:

```python
# Decode an image and convert it to HSV.
rgb_image = tf.image.decode_png(...,  channels=3)
rgb_image_float = tf.image.convert_image_dtype(rgb_image, tf.float32)
hsv_image = tf.image.rgb_to_hsv(rgb_image)
```

@@rgb_to_grayscale
@@grayscale_to_rgb

@@hsv_to_rgb
@@rgb_to_hsv

@@convert_image_dtype

## Image Adjustments

TensorFlow provides functions to adjust images in various ways: brightness,
contrast, hue, and saturation.  Each adjustment can be done with predefined
parameters or with random parameters picked from predefined intervals. Random
adjustments are often useful to expand a training set and reduce overfitting.

If several adjustments are chained it is advisable to minimize the number of
redundant conversions by first converting the images to the most natural data
type and representation (RGB or HSV).

@@adjust_brightness
@@random_brightness

@@adjust_contrast
@@random_contrast

@@adjust_hue
@@random_hue

@@adjust_gamma

@@adjust_saturation
@@random_saturation

@@per_image_standardization

## Working with Bounding Boxes

@@draw_bounding_boxes
@@non_max_suppression
@@sample_distorted_bounding_box
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

<<<<<<< HEAD
=======
import os

from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gen_string_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables

>>>>>>> Add decode_image Op

# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_image_ops import *
from tensorflow.python.ops.image_ops_impl import *
# pylint: enable=wildcard-import

# TODO(drpng): remove these once internal use has discontinued.
# pylint: disable=unused-import
from tensorflow.python.ops.image_ops_impl import _Check3DImage
from tensorflow.python.ops.image_ops_impl import _ImageDimensions
# pylint: enable=unused-import

from tensorflow.python.util.all_util import remove_undocumented

_allowed_symbols = [
    # ResizeMethod is not documented, but is documented in functions
    # that use it.
    'ResizeMethod',
]

<<<<<<< HEAD
remove_undocumented(__name__, _allowed_symbols)
=======
def decode_image(contents, channels=None, ratio=None, fancy_upscaling=None,
                 try_recover_truncated=None, acceptable_fraction=None,
                 name=None):
  """Convenience function for `decode_gif`, `decode_jpeg`, and `decode_png`.
  Detects whether an image is a GIF, JPEG, or PNG, and performs the appropriate 
  operation to convert the input bytes `string` into a `Tensor` of type `uint8`.

  Note: `decode_gif` returns a 4-D array `[num_frames, height, width, 3]`, as 
  opposed to `decode_jpeg` and `decode_png`, which return 3-D arrays 
  `[height, width, num_channels]`. Make sure to take this into account when 
  constructing your graph if you are intermixing GIF files with JPEG and/or PNG 
  files.

  Args:
    contents: 0-D `string`. The encoded image bytes.
    channels: Number of color channels for the decoded image.
    ratio: Downscaling ratio (only used when decoding JPEG images)
    fancy_upscaling: If true use a slower but nicer upscaling of the chroma 
      planes (yuv420/422 JPEG images only).
    try_recover_truncated: If true, try to recover an image from truncated input
      (only used when decoding JPEG images).
    acceptable_fraction: The minimum required fraction of lines before a 
      truncated input is accepted (only used when decoding JPEG images).
  
  Returns:
    `Tensor` with type `uint8`. Shape `[height, width, num_channels]` for JPEG 
      and PNG images. Shape `[num_frames, height, width, 3]` for GIF images.
  """
  with ops.name_scope(name, 'decode_image') as scope:
    def _gif():
      return gen_image_ops.decode_gif(contents)

    def _jpeg():
      return gen_image_ops.decode_jpeg(contents, channels, ratio, 
                                       fancy_upscaling, try_recover_truncated, 
                                       acceptable_fraction)
    def _png():
      return gen_image_ops.decode_png(contents, channels, dtypes.uint8)

    is_gif = math_ops.equal(gen_string_ops.substr(contents, 0, 4),
                            b'\x47\x49\x46\x38')
    is_jpeg = math_ops.equal(gen_string_ops.substr(contents, 0, 4), 
                            b'\xff\xd8\xff\xe0')
    is_png = math_ops.equal(gen_string_ops.substr(contents, 0, 8), 
                            b'\211PNG\r\n\032\n')
    is_decodable = math_ops.logical_or(is_gif, is_jpeg)
    is_decodable = math_ops.logical_or(is_decodable, is_png)
    assert_decodable = control_flow_ops.Assert(is_decodable, 
                                               [b'Unable to decode bytes as a '
                                                b'PNG or JPEG. Is the file '
                                                b'encoded properly?'])
    # Leaving default case to be decode_png
    cases = [(is_gif, _gif),
             (is_jpeg, _jpeg),
            ]
    with ops.control_dependencies([assert_decodable]):
      return control_flow_ops.case(cases, _png, exclusive=True, 
                                   name=scope)


__all__ = make_all(__name__)
# ResizeMethod is not documented, but is documented in functions that use it.
__all__.append('ResizeMethod')
# TODO(skye): per_image_whitening() will be removed once all callers switch to
# per_image_standardization()
__all__.append('per_image_whitening')
>>>>>>> Add decode_image Op
