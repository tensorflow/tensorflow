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
"""Tests for connected component analysis."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np

from tensorflow.contrib.image.python.ops import image_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest

# Image for testing connected_components, with a single, winding component.
SNAKE = np.asarray(
    [[0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 1, 1, 1, 1, 0, 0, 0, 0],
     [0, 0, 0, 0, 1, 1, 1, 1, 0],
     [0, 0, 0, 0, 0, 0, 0, 1, 0],
     [0, 1, 1, 1, 1, 1, 1, 1, 0],
     [0, 1, 0, 0, 0, 0, 0, 0, 0],
     [0, 1, 0, 1, 1, 1, 1, 1, 0],
     [0, 1, 0, 0, 0, 0, 0, 1, 0],
     [0, 1, 1, 1, 1, 1, 1, 1, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0]])  # pyformat: disable


class SegmentationTest(test_util.TensorFlowTestCase):

  def testDisconnected(self):
    arr = math_ops.cast(
        [[1, 0, 0, 1, 0, 0, 0, 0, 1],
         [0, 1, 0, 0, 0, 1, 0, 1, 0],
         [1, 0, 1, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0, 0, 0]],
        dtypes.bool)  # pyformat: disable
    expected = (
        [[1, 0, 0, 2, 0, 0, 0, 0, 3],
         [0, 4, 0, 0, 0, 5, 0, 6, 0],
         [7, 0, 8, 0, 0, 0, 9, 0, 0],
         [0, 0, 0, 0, 10, 0, 0, 0, 0],
         [0, 0, 11, 0, 0, 0, 0, 0, 0]])  # pyformat: disable
    with self.test_session():
      self.assertAllEqual(image_ops.connected_components(arr).eval(), expected)

  def testSimple(self):
    arr = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
    with self.test_session():
      # Single component with id 1.
      self.assertAllEqual(
          image_ops.connected_components(math_ops.cast(
              arr, dtypes.bool)).eval(), arr)

  def testSnake(self):
    with self.test_session():
      # Single component with id 1.
      self.assertAllEqual(
          image_ops.connected_components(math_ops.cast(
              SNAKE, dtypes.bool)).eval(), SNAKE)

  def testSnake_disconnected(self):
    for i in range(SNAKE.shape[0]):
      for j in range(SNAKE.shape[1]):
        with self.test_session():
          # If we disconnect any part of the snake except for the endpoints,
          # there will be 2 components.
          if SNAKE[i, j] and (i, j) not in [(1, 1), (6, 3)]:
            disconnected_snake = SNAKE.copy()
            disconnected_snake[i, j] = 0
            components = image_ops.connected_components(
                math_ops.cast(disconnected_snake, dtypes.bool)).eval()
            self.assertEqual(components.max(), 2, 'disconnect (%d, %d)' % (i,
                                                                           j))
            bins = np.bincount(components.ravel())
            # Nonzero number of pixels labeled 0, 1, or 2.
            self.assertGreater(bins[0], 0)
            self.assertGreater(bins[1], 0)
            self.assertGreater(bins[2], 0)

  def testMultipleImages(self):
    images = [[[1, 1, 1, 1],
               [1, 0, 0, 1],
               [1, 0, 0, 1],
               [1, 1, 1, 1]],
              [[1, 0, 0, 1],
               [0, 0, 0, 0],
               [0, 0, 0, 0],
               [1, 0, 0, 1]],
              [[1, 1, 0, 1],
               [0, 1, 1, 0],
               [1, 0, 1, 0],
               [0, 0, 1, 1]]]  # pyformat: disable
    expected = [[[1, 1, 1, 1],
                 [1, 0, 0, 1],
                 [1, 0, 0, 1],
                 [1, 1, 1, 1]],
                [[2, 0, 0, 3],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0],
                 [4, 0, 0, 5]],
                [[6, 6, 0, 7],
                 [0, 6, 6, 0],
                 [8, 0, 6, 0],
                 [0, 0, 6, 6]]]  # pyformat: disable
    with self.test_session():
      self.assertAllEqual(
          image_ops.connected_components(math_ops.cast(
              images, dtypes.bool)).eval(), expected)

  def testZeros(self):
    with self.test_session():
      self.assertAllEqual(
          image_ops.connected_components(
              array_ops.zeros((100, 20, 50), dtypes.bool)).eval(),
          np.zeros((100, 20, 50)))

  def testOnes(self):
    with self.test_session():
      self.assertAllEqual(
          image_ops.connected_components(
              array_ops.ones((100, 20, 50), dtypes.bool)).eval(),
          np.tile(np.arange(100)[:, None, None] + 1, [1, 20, 50]))

  def testOnes_small(self):
    with self.test_session():
      self.assertAllEqual(
          image_ops.connected_components(array_ops.ones((3, 5),
                                                        dtypes.bool)).eval(),
          np.ones((3, 5)))

  def testRandom_scipy(self):
    np.random.seed(42)
    images = np.random.randint(0, 2, size=(10, 100, 200)).astype(np.bool)
    expected = connected_components_reference_implementation(images)
    if expected is None:
      return
    with self.test_session():
      self.assertAllEqual(
          image_ops.connected_components(images).eval(), expected)


def connected_components_reference_implementation(images):
  try:
    # pylint: disable=g-import-not-at-top
    from scipy.ndimage import measurements
  except ImportError:
    logging.exception('Skipping test method because scipy could not be loaded')
    return
  image_or_images = np.asarray(images)
  if len(image_or_images.shape) == 2:
    images = image_or_images[None, :, :]
  elif len(image_or_images.shape) == 3:
    images = image_or_images
  components = np.asarray([measurements.label(image)[0] for image in images])
  # Get the count of nonzero ids for each image, and offset each image's nonzero
  # ids using the cumulative sum.
  num_ids_per_image = components.reshape(
      [-1, components.shape[1] * components.shape[2]]).max(axis=-1)
  positive_id_start_per_image = np.cumsum(num_ids_per_image)
  for i in range(components.shape[0]):
    new_id_start = positive_id_start_per_image[i - 1] if i > 0 else 0
    components[i, components[i] > 0] += new_id_start
  if len(image_or_images.shape) == 2:
    return components[0, :, :]
  else:
    return components


if __name__ == '__main__':
  googletest.main()
