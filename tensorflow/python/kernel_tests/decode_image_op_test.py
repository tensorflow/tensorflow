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
"""Tests for decode_image."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import numpy as np
import tensorflow as tf


class DecodeImageOpTest(tf.test.TestCase):

  def testGif(self):
    # Read some real GIFs
    path = os.path.join('tensorflow', 'core', 'lib', 'gif', 'testdata', 
                        'scan.gif')
    WIDTH = 20
    HEIGHT = 40
    STRIDE = 5
    shape = (12, HEIGHT, WIDTH, 3)

    with self.test_session(use_gpu=True) as sess:
      gif0 = tf.read_file(path)
      image0 = tf.image.decode_image(gif0)
      image1 = tf.image.decode_gif(gif0)
      gif0, image0, image1 = sess.run([gif0, image0, image1])

      self.assertEqual(image0.shape, shape)
      self.assertAllEqual(image0, image1)

      for frame_idx, frame in enumerate(image0):
        gt = np.zeros(shape[1:], dtype=np.uint8)
        start = frame_idx * STRIDE
        end = (frame_idx + 1) * STRIDE
        if end <= WIDTH:
          gt[:, start:end, :] = 255
        else:
          start -= WIDTH
          end -= WIDTH
          gt[start:end, :, :] = 255

        self.assertAllClose(frame, gt)

        bad_channels = tf.image.decode_image(gif0, channels=1)
        with self.assertRaises(tf.errors.InvalidArgumentError):
          bad_channels.eval()


  def testJpeg(self):
    # Read a real jpeg and verify shape
    path = os.path.join('tensorflow', 'core', 'lib', 'jpeg', 'testdata',
                        'jpeg_merge_test1.jpg')
    with self.test_session(use_gpu=True) as sess:
      jpeg0 = tf.read_file(path)
      image0 = tf.image.decode_image(jpeg0)
      image1 = tf.image.decode_jpeg(jpeg0)
      jpeg0, image0, image1 = sess.run([jpeg0, image0, image1])
      self.assertEqual(len(jpeg0), 3771)
      self.assertEqual(image0.shape, (256, 128, 3))
      self.assertAllEqual(image0, image1)

  def testPng(self):
    # Read some real PNGs, converting to different channel numbers
    prefix = ['tensorflow', 'core', 'lib', 'png', 'testdata']
    inputs = [(1, 'lena_gray.png')]
    for channels_in, filename in inputs:
      for channels in 0, 1, 3:
        with self.test_session(use_gpu=True) as sess:
          path = prefix + [filename]
          png0 = tf.read_file(os.path.join(*path))
          image0 = tf.image.decode_image(png0, channels=channels)
          image1 = tf.image.decode_png(png0, channels=channels)
          png0, image0, image1 = sess.run([png0, image0, image1])
          self.assertEqual(image0.shape, (26, 51, channels or channels_in))
          self.assertAllEqual(image0, image1)

  def testInvalidBytes(self):
    image_bytes = b'ThisIsNotAnImage!'
    decode = tf.image.decode_image(image_bytes)
    with self.test_session():
      with self.assertRaises(tf.errors.InvalidArgumentError):
        decode.eval()

  def testInvalidChannels(self):
    image_bytes = b'unused'
    with self.assertRaises(ValueError):
      decode = tf.image.decode_image(image_bytes, channels=4)


if __name__ == "__main__":
  tf.test.main()
