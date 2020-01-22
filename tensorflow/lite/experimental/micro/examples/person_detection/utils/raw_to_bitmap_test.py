# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for raw to bitmap converter utility."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io

import numpy as np

from tensorflow.lite.experimental.micro.examples.person_detection.utils.raw_to_bitmap import parse_file
from tensorflow.lite.experimental.micro.examples.person_detection.utils.raw_to_bitmap import reshape_bitmaps
from tensorflow.python.platform import test

_RGB_RAW = u"""
+++ frame +++
0x0000 0x00 0x00 0x00 0x01 0x01 0x01 0x02 0x02 0x02 0x03 0x03 0x03 0x04 0x04 0x04 0x05
0x0010 0x05 0x05 0x06 0x06 0x06 0x07 0x07 0x07 0x08 0x08 0x08 0x09 0x09 0x09 0x0a 0x0a
0x0020 0x0a 0x0b 0x0b 0x0b 0x0c 0x0c 0x0c 0x0d 0x0d 0x0d 0x0e 0x0e 0x0e 0x0f 0x0f 0x0f
--- frame ---
"""

_RGB_FLAT = np.array([[
    0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8,
    8, 8, 9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12, 13, 13, 13, 14, 14, 14,
    15, 15, 15
]])

_RGB_RESHAPED = np.array(
    [[[[12, 12, 12], [13, 13, 13], [14, 14, 14], [15, 15, 15]],
      [[8, 8, 8], [9, 9, 9], [10, 10, 10], [11, 11, 11]],
      [[4, 4, 4], [5, 5, 5], [6, 6, 6], [7, 7, 7]],
      [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]]]])

_GRAYSCALE_RAW = u"""
+++ frame +++
0x0000 0x00 0x01 0x02 0x03 0x04 0x05 0x06 0x07 0x08 0x09 0x0a 0x0b 0x0c 0x0d 0x0e 0x0f
--- frame ---
"""

_GRAYSCALE_FLAT = np.array(
    [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]])

_GRAYSCALE_RESHAPED = np.array([[[12, 13, 14, 15],
                                 [8, 9, 10, 11],
                                 [4, 5, 6, 7],
                                 [0, 1, 2, 3]]])


_GRAYSCALE_RAW_MULTI = u"""
+++ frame +++
0x0000 0x00 0x01 0x02 0x03 0x04 0x05 0x06 0x07 0x08 0x09 0x0a 0x0b 0x0c 0x0d 0x0e 0x0f
--- frame ---
+++ frame +++
0x0000 0x10 0x11 0x12 0x13 0x14 0x15 0x16 0x17 0x18 0x19 0x1a 0x1b 0x1c 0x1d 0x1e 0x1f
--- frame ---
+++ frame +++
0x0000 0x20 0x21 0x22 0x23 0x24 0x25 0x26 0x27 0x28 0x29 0x2a 0x2b 0x2c 0x2d 0x2e 0x2f
--- frame ---
+++ frame +++
0x0000 0x30 0x31 0x32 0x33 0x34 0x35 0x36 0x37 0x38 0x39 0x3a 0x3b 0x3c 0x3d 0x3e 0x3f
--- frame ---
"""

_GRAYSCALE_FLAT_MULTI = [
    np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
    np.array([16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]),
    np.array([32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]),
    np.array([48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63])]

_GRAYSCALE_RESHAPED_MULTI = [
    np.array([[12, 13, 14, 15],
              [8, 9, 10, 11],
              [4, 5, 6, 7],
              [0, 1, 2, 3]]),
    np.array([[28, 29, 30, 31],
              [24, 25, 26, 27],
              [20, 21, 22, 23],
              [16, 17, 18, 19]]),
    np.array([[44, 45, 46, 47],
              [40, 41, 42, 43],
              [36, 37, 38, 39],
              [32, 33, 34, 35]]),
    np.array([[60, 61, 62, 63],
              [56, 57, 58, 59],
              [52, 53, 54, 55],
              [48, 49, 50, 51]])]


class RawToBitmapTest(test.TestCase):

  def testParseRgb(self):
    frame_list = parse_file(io.StringIO(_RGB_RAW), 4, 4, 3)
    self.assertTrue(np.array_equal(_RGB_FLAT, frame_list))

  def testParseGrayscale(self):
    frame_list = parse_file(io.StringIO(_GRAYSCALE_RAW), 4, 4, 1)
    self.assertTrue(np.array_equal(_GRAYSCALE_FLAT, frame_list))

  def testReshapeRgb(self):
    reshaped = reshape_bitmaps(_RGB_FLAT, 4, 4, 3)
    self.assertTrue(np.array_equal(_RGB_RESHAPED, reshaped))

  def testReshapeGrayscale(self):
    reshaped = reshape_bitmaps(_GRAYSCALE_FLAT, 4, 4, 1)
    self.assertTrue(np.array_equal(_GRAYSCALE_RESHAPED, reshaped))

  def testMultipleGrayscale(self):
    frame_list = parse_file(io.StringIO(_GRAYSCALE_RAW_MULTI), 4, 4, 1)
    self.assertTrue(np.array_equal(_GRAYSCALE_FLAT_MULTI, frame_list))
    reshaped = reshape_bitmaps(frame_list, 4, 4, 1)
    self.assertTrue(np.array_equal(_GRAYSCALE_RESHAPED_MULTI, reshaped))


if __name__ == '__main__':
  test.main()
