# Lint as: python2, python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
r"""This tool converts an image file into a CSV data array.

Designed to help create test inputs that can be shared between Python and
on-device test cases to investigate accuracy issues.

Example usage:

python convert_image_to_csv.py some_image.jpg --width=16 --height=20 \
  --want_grayscale
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.python.client import session
from tensorflow.python.framework import ops
from tensorflow.python.framework.errors_impl import NotFoundError
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.platform import app


def get_image(width, height, want_grayscale, filepath):
  """Returns an image loaded into an np.ndarray with dims [height, width, (3 or 1)].

  Args:
    width: Width to rescale the image to.
    height: Height to rescale the image to.
    want_grayscale: Whether the result should be converted to grayscale.
    filepath: Path of the image file..

  Returns:
    np.ndarray of shape (height, width, channels) where channels is 1 if
      want_grayscale is true, otherwise 3.
  """
  with ops.Graph().as_default():
    with session.Session():
      file_data = io_ops.read_file(filepath)
      channels = 1 if want_grayscale else 3
      image_tensor = image_ops.decode_image(file_data,
                                            channels=channels).eval()
      resized_tensor = image_ops.resize_images_v2(
          image_tensor, (height, width)).eval()
  return resized_tensor


def array_to_int_csv(array_data):
  """Converts all elements in a numerical array to a comma-separated string.

  Args:
    array_data: Numerical array to convert.

  Returns:
    String containing array values as integers, separated by commas.
  """
  flattened_array = array_data.flatten()
  array_as_strings = [item.astype(int).astype(str) for item in flattened_array]
  return ','.join(array_as_strings)


def run_main(_):
  """Application run loop."""
  parser = argparse.ArgumentParser(
      description='Loads JPEG or PNG input files, resizes them, optionally'
      ' converts to grayscale, and writes out as comma-separated variables,'
      ' one image per row.')
  parser.add_argument(
      'image_file_names',
      type=str,
      nargs='+',
      help='List of paths to the input images.')
  parser.add_argument(
      '--width', type=int, default=96, help='Width to scale images to.')
  parser.add_argument(
      '--height', type=int, default=96, help='Height to scale images to.')
  parser.add_argument(
      '--want_grayscale',
      action='store_true',
      help='Whether to convert the image to monochrome.')
  args = parser.parse_args()

  for image_file_name in args.image_file_names:
    try:
      image_data = get_image(args.width, args.height, args.want_grayscale,
                             image_file_name)
      print(array_to_int_csv(image_data))
    except NotFoundError:
      sys.stderr.write('Image file not found at {0}\n'.format(image_file_name))
      sys.exit(1)


def main():
  app.run(main=run_main, argv=sys.argv[:1])


if __name__ == '__main__':
  main()
