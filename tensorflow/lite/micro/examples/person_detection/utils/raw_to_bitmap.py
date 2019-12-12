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
"""Convert raw bytes to a bitmap.

Converts a raw image dumped to a file into a bitmap.  The file must contain
complete bitmap images in 324 x 244 resolution, formatted as follows:

+++ frame +++
<byte number> <16 one-byte values separated by spaces>
--- frame ---

For example, the first line might look like:
0x00000000 C5 C3 CE D1 D9 DA D6 E3 E2 EB E9 EB DB E4 F5 FF

The bitmaps are automatically saved to the same directory as the log file, and
are displayed by the script.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import os.path
import re
import numpy as np

_DICT_RESOLUTIONS = {
    'QVGA': (324, 244, 1),
    'GRAY': (96, 96, 1),
    'RGB': (96, 96, 3),
}

_VERSION = 0
_SUBVERSION = 1


def check_file_existence(x):
  if not os.path.isfile(x):
    # Argparse uses the ArgumentTypeError to give a rejection message like:
    # error: argument input: x does not exist
    raise argparse.ArgumentTypeError('{0} does not exist'.format(x))
  return x


def show_and_save_bitmaps(input_file, bitmap_list, channels):
  """Display and save a list of bitmaps.

  Args:
    input_file: input file name
    bitmap_list: list of numpy arrays to represent bitmap images
    channels: color channel count
  """
  try:
    from PIL import Image  # pylint: disable=g-import-not-at-top
  except ImportError:
    raise NotImplementedError('Image display and save not implemented.')

  for idx, bitmap in enumerate(bitmap_list):
    path = os.path.dirname(os.path.abspath(input_file))
    basename = os.path.split(os.path.splitext(input_file)[0])[-1]
    outputfile = os.path.join(path, basename + '_' + str(idx) + '.bmp')

    if channels == 3:
      img = Image.fromarray(bitmap, 'RGB')
    else:
      img = Image.fromarray(bitmap, 'L')

    img.save(outputfile)
    img.show()


def reshape_bitmaps(frame_list, width, height, channels):
  """Reshape flat integer arrays.

  Args:
    frame_list: list of 1-D arrays to represent raw image data
    width: image width in pixels
    height: image height in pixels
    channels: color channel count

  Returns:
    list of numpy arrays to represent bitmap images
  """

  bitmap_list = []
  for frame in frame_list:
    shape = (height, width, channels) if channels > 1 else (height, width)
    bitmap = np.reshape(frame, shape)
    bitmap = np.flip(bitmap, 0)
    bitmap_list.append(bitmap)
  return bitmap_list


def parse_file(inputfile, width, height, channels):
  """Convert log file to array of pixels.

  Args:
    inputfile: log file to parse
    width: image width in pixels
    height: image height in pixels
    channels: color channel count

  Returns:
    list 1-D arrays to represent raw image data.
  """

  data = None
  bytes_written = 0
  frame_start = False
  frame_stop = False
  frame_list = list()

  # collect all pixel data into an int array
  for line in inputfile:
    if line == '+++ frame +++\n':
      frame_start = True
      data = np.zeros(height * width * channels, dtype=np.uint8)
      bytes_written = 0
      continue
    elif line == '--- frame ---\n':
      frame_stop = True

    if frame_start and not frame_stop:
      linelist = re.findall(r"[\w']+", line)

      if len(linelist) != 17:
        # drop this frame
        frame_start = False
        continue

      for item in linelist[1:]:
        data[bytes_written] = int(item, base=16)
        bytes_written += 1

    elif frame_start and frame_stop:
      if bytes_written == height * width * channels:
        frame_list.append(data)
        frame_start = False
        frame_stop = False

  return frame_list


def main():
  parser = argparse.ArgumentParser(
      description='This program converts raw data from HM01B0 to a bmp file.')

  parser.add_argument(
      '-i',
      '--input',
      dest='inputfile',
      required=True,
      help='input file',
      metavar='FILE',
      type=check_file_existence)

  parser.add_argument(
      '-r',
      '--resolution',
      dest='resolution',
      required=False,
      help='Resolution',
      choices=['QVGA', 'RGB', 'GRAY'],
      default='QVGA',
  )

  parser.add_argument(
      '-v',
      '--version',
      help='Program version',
      action='version',
      version='%(prog)s {ver}'.format(ver='v%d.%d' % (_VERSION, _SUBVERSION)))

  args = parser.parse_args()

  (width, height,
   channels) = _DICT_RESOLUTIONS.get(args.resolution,
                                     ('Resolution not supported', 0, 0, 0))
  frame_list = parse_file(open(args.inputfile), width, height, channels)
  bitmap_list = reshape_bitmaps(frame_list, width, height, channels)
  show_and_save_bitmaps(args.inputfile, bitmap_list, channels)


if __name__ == '__main__':
  main()
