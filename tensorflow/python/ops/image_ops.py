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
"""Image processing and decoding ops.

See the @{$python/image} guide.

@@decode_bmp
@@decode_gif
@@decode_jpeg
@@decode_and_crop_jpeg
@@encode_jpeg
@@extract_jpeg_shape
@@decode_png
@@encode_png
@@decode_image
@@resize_images
@@resize_area
@@resize_bicubic
@@resize_bilinear
@@resize_nearest_neighbor
@@resize_image_with_crop_or_pad
@@central_crop
@@pad_to_bounding_box
@@crop_to_bounding_box
@@extract_glimpse
@@crop_and_resize
@@flip_up_down
@@random_flip_up_down
@@flip_left_right
@@random_flip_left_right
@@transpose_image
@@rot90

@@rgb_to_grayscale
@@grayscale_to_rgb
@@hsv_to_rgb
@@rgb_to_hsv
@@rgb_to_yiq
@@yiq_to_rgb
@@rgb_to_yuv
@@yuv_to_rgb
@@convert_image_dtype
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
@@draw_bounding_boxes
@@non_max_suppression
@@sample_distorted_bounding_box
@@total_variation
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


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

remove_undocumented(__name__, _allowed_symbols)
