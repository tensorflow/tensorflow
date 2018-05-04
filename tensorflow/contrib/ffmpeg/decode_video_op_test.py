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
# =============================================================================
"""Tests for third_party.tensorflow.contrib.ffmpeg.decode_video_op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import six  # pylint: disable=unused-import

from tensorflow.contrib import ffmpeg
from tensorflow.python.ops import image_ops
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import test


class DecodeVideoOpTest(test.TestCase):

  def _loadFileAndTest(self, filename, width, height, frames, bmp_filename,
                       index):
    """Loads an video file and validates the output tensor.

    Args:
      filename: The filename of the input file.
      width: The width of the video.
      height: The height of the video.
      frames: The frames of the video.
      bmp_filename: The filename for the bmp file.
      index: Index location inside the video.
    """
    with self.test_session():
      path = os.path.join(resource_loader.get_data_files_path(), 'testdata',
                          filename)
      with open(path, 'rb') as f:
        contents = f.read()

      bmp_path = os.path.join(resource_loader.get_data_files_path(), 'testdata',
                              bmp_filename)
      with open(bmp_path, 'rb') as f:
        bmp_contents = f.read()

      image_op = image_ops.decode_bmp(bmp_contents)
      image = image_op.eval()
      self.assertEqual(image.shape, (height, width, 3))
      video_op = ffmpeg.decode_video(contents)
      video = video_op.eval()
      self.assertEqual(video.shape, (frames, height, width, 3))
      self.assertAllEqual(video[index, :, :, :], image)

  def testMp4(self):
    self._loadFileAndTest('small.mp4', 560, 320, 166, 'small_100.bmp', 99)


if __name__ == '__main__':
  test.main()
