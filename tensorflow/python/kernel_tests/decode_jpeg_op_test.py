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

  def _evalDecodeJpeg(self, image_name, parallelism, num_iters, tile=None):
    """Evaluate DecodeJpegOp for the given image.

    TODO(tanmingxing): add decoding+cropping as well.

    Args:
      image_name: a string of image file name (without suffix).
      parallelism: the number of concurrent decode_jpeg ops to be run.
      num_iters: number of iterations for evaluation.
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
      for i in xrange(parallelism):
        images.append(
            image_ops.decode_jpeg(
                image_content, channels=3, name='image_%d' % (i)))

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
    parallelism = 1
    num_iters = 10
    for parallelism in [1, 10, 100]:
      duration = self._evalDecodeJpeg('small.jpg', parallelism, num_iters)
      self.report_benchmark(
          name='decode_jpeg_small_p%d' % (parallelism),
          iters=num_iters,
          wall_time=duration)

  def benchmarkDecodeJpegMedium(self):
    """Evaluate single DecodeImageOp for medium size image."""
    parallelism = 1
    num_iters = 10
    for parallelism in [1, 10, 100]:
      duration = self._evalDecodeJpeg('medium.jpg', parallelism, num_iters)
      self.report_benchmark(
          name='decode_jpeg_medium_p%d' % (parallelism),
          iters=num_iters,
          wall_time=duration)

  def benchmarkDecodeJpegLarge(self):
    """Evaluate single DecodeImageOp for large size image."""
    parallelism = 1
    num_iters = 10
    for parallelism in [1, 10, 100]:
      # Tile the medium size image to composite a larger fake image.
      duration = self._evalDecodeJpeg(
          'medium.jpg', parallelism, num_iters, tile=[4, 4, 1])
      self.report_benchmark(
          name='decode_jpeg_large_p%d' % (parallelism),
          iters=num_iters,
          wall_time=duration)


if __name__ == '__main__':
  test.main()
