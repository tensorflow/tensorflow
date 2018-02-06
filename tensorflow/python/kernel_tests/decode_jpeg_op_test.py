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
"""Tests for DecodeJpegOp."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

from six.moves import xrange
from tensorflow.python.client import session
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test

prefix_path = 'third_party/tensorflow/core/lib/jpeg/testdata'


class DecodeJpegBenchmark(test.Benchmark):
  """Evaluate tensorflow DecodeJpegOp performance."""

  def _evalDecodeJpeg(self,
                      image_name,
                      parallelism,
                      num_iters,
                      crop_during_decode=None,
                      crop_window=None,
                      tile=None):
    """Evaluate DecodeJpegOp for the given image.

    TODO(tanmingxing): add decoding+cropping as well.

    Args:
      image_name: a string of image file name (without suffix).
      parallelism: the number of concurrent decode_jpeg ops to be run.
      num_iters: number of iterations for evaluation.
      crop_during_decode: If true, use fused DecodeAndCropJpeg instead of
          separate decode and crop ops. It is ignored if crop_window is None.
      crop_window: if not None, crop the decoded image. Depending on
          crop_during_decode, cropping could happen during or after decoding.
      tile: if not None, tile the image to composite a larger fake image.

    Returns:
      The duration of the run in seconds.
    """
    ops.reset_default_graph()

    image_file_path = os.path.join(prefix_path, image_name)

    if tile is None:
      image_content = variable_scope.get_variable(
          'image_%s' % image_name,
          initializer=io_ops.read_file(image_file_path))
    else:
      single_image = image_ops.decode_jpeg(
          io_ops.read_file(image_file_path), channels=3, name='single_image')
      # Tile the image to composite a new larger image.
      tiled_image = array_ops.tile(single_image, tile)
      image_content = variable_scope.get_variable(
          'tiled_image_%s' % image_name,
          initializer=image_ops.encode_jpeg(tiled_image))

    with session.Session() as sess:
      sess.run(variables.global_variables_initializer())
      images = []
      for _ in xrange(parallelism):
        if crop_window is None:
          # No crop.
          image = image_ops.decode_jpeg(image_content, channels=3)
        elif crop_during_decode:
          # combined decode and crop.
          image = image_ops.decode_and_crop_jpeg(
              image_content, crop_window, channels=3)
        else:
          # separate decode and crop.
          image = image_ops.decode_jpeg(image_content, channels=3)
          image = image_ops.crop_to_bounding_box(
              image,
              offset_height=crop_window[0],
              offset_width=crop_window[1],
              target_height=crop_window[2],
              target_width=crop_window[3])

        images.append(image)
      r = control_flow_ops.group(*images)

      for _ in xrange(3):
        # Skip warm up time.
        sess.run(r)

      start_time = time.time()
      for _ in xrange(num_iters):
        sess.run(r)
    return time.time() - start_time

  def benchmarkDecodeJpegSmall(self):
    """Evaluate single DecodeImageOp for small size image."""
    num_iters = 10
    crop_window = [10, 10, 50, 50]
    for parallelism in [1, 100]:
      duration_decode = self._evalDecodeJpeg('small.jpg', parallelism,
                                             num_iters)
      duration_decode_crop = self._evalDecodeJpeg('small.jpg', parallelism,
                                                  num_iters, False, crop_window)
      duration_decode_after_crop = self._evalDecodeJpeg(
          'small.jpg', parallelism, num_iters, True, crop_window)
      self.report_benchmark(
          name='decode_jpeg_small_p%d' % (parallelism),
          iters=num_iters,
          wall_time=duration_decode)
      self.report_benchmark(
          name='decode_crop_jpeg_small_p%d' % (parallelism),
          iters=num_iters,
          wall_time=duration_decode_crop)
      self.report_benchmark(
          name='decode_after_crop_jpeg_small_p%d' % (parallelism),
          iters=num_iters,
          wall_time=duration_decode_after_crop)

  def benchmarkDecodeJpegMedium(self):
    """Evaluate single DecodeImageOp for medium size image."""
    num_iters = 10
    crop_window = [10, 10, 50, 50]
    for parallelism in [1, 100]:
      duration_decode = self._evalDecodeJpeg('medium.jpg', parallelism,
                                             num_iters)
      duration_decode_crop = self._evalDecodeJpeg('medium.jpg', parallelism,
                                                  num_iters, False, crop_window)
      duration_decode_after_crop = self._evalDecodeJpeg(
          'medium.jpg', parallelism, num_iters, True, crop_window)
      self.report_benchmark(
          name='decode_jpeg_medium_p%d' % (parallelism),
          iters=num_iters,
          wall_time=duration_decode)
      self.report_benchmark(
          name='decode_crop_jpeg_medium_p%d' % (parallelism),
          iters=num_iters,
          wall_time=duration_decode_crop)
      self.report_benchmark(
          name='decode_after_crop_jpeg_medium_p%d' % (parallelism),
          iters=num_iters,
          wall_time=duration_decode_after_crop)

  def benchmarkDecodeJpegLarge(self):
    """Evaluate single DecodeImageOp for large size image."""
    num_iters = 10
    crop_window = [10, 10, 50, 50]
    tile = [4, 4, 1]
    for parallelism in [1, 100]:
      # Tile the medium size image to composite a larger fake image.
      duration_decode = self._evalDecodeJpeg('medium.jpg', parallelism,
                                             num_iters, tile)
      duration_decode_crop = self._evalDecodeJpeg(
          'medium.jpg', parallelism, num_iters, False, crop_window, tile)
      duration_decode_after_crop = self._evalDecodeJpeg(
          'medium.jpg', parallelism, num_iters, True, crop_window, tile)
      self.report_benchmark(
          name='decode_jpeg_large_p%d' % (parallelism),
          iters=num_iters,
          wall_time=duration_decode)
      self.report_benchmark(
          name='decode_crop_jpeg_large_p%d' % (parallelism),
          iters=num_iters,
          wall_time=duration_decode_crop)
      self.report_benchmark(
          name='decode_after_crop_jpeg_large_p%d' % (parallelism),
          iters=num_iters,
          wall_time=duration_decode_after_crop)


if __name__ == '__main__':
  test.main()
