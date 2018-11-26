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
"""Tests for dense_image_warp."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np

from tensorflow.contrib.image.python.ops import dense_image_warp

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest

from tensorflow.python.training import adam


class DenseImageWarpTest(test_util.TensorFlowTestCase):

  def setUp(self):
    np.random.seed(0)

  def test_interpolate_small_grid_ij(self):
    grid = constant_op.constant(
        [[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]], shape=[1, 3, 3, 1])
    query_points = constant_op.constant(
        [[0., 0.], [1., 0.], [2., 0.5], [1.5, 1.5]], shape=[1, 4, 2])
    expected_results = np.reshape(np.array([0., 3., 6.5, 6.]), [1, 4, 1])

    interp = dense_image_warp._interpolate_bilinear(grid, query_points)

    with self.cached_session() as sess:
      predicted = sess.run(interp)
      self.assertAllClose(expected_results, predicted)

  def test_interpolate_small_grid_xy(self):
    grid = constant_op.constant(
        [[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]], shape=[1, 3, 3, 1])
    query_points = constant_op.constant(
        [[0., 0.], [0., 1.], [0.5, 2.0], [1.5, 1.5]], shape=[1, 4, 2])
    expected_results = np.reshape(np.array([0., 3., 6.5, 6.]), [1, 4, 1])

    interp = dense_image_warp._interpolate_bilinear(
        grid, query_points, indexing='xy')

    with self.cached_session() as sess:
      predicted = sess.run(interp)
      self.assertAllClose(expected_results, predicted)

  def test_interpolate_small_grid_batched(self):
    grid = constant_op.constant(
        [[[0., 1.], [3., 4.]], [[5., 6.], [7., 8.]]], shape=[2, 2, 2, 1])
    query_points = constant_op.constant([[[0., 0.], [1., 0.], [0.5, 0.5]],
                                         [[0.5, 0.], [1., 0.], [1., 1.]]])
    expected_results = np.reshape(
        np.array([[0., 3., 2.], [6., 7., 8.]]), [2, 3, 1])

    interp = dense_image_warp._interpolate_bilinear(grid, query_points)

    with self.cached_session() as sess:
      predicted = sess.run(interp)
      self.assertAllClose(expected_results, predicted)

  def get_image_and_flow_placeholders(self, shape, image_type, flow_type):
    batch_size, height, width, numchannels = shape
    image_shape = [batch_size, height, width, numchannels]
    flow_shape = [batch_size, height, width, 2]

    tf_type = {
        'float16': dtypes.half,
        'float32': dtypes.float32,
        'float64': dtypes.float64
    }

    image = array_ops.placeholder(dtype=tf_type[image_type], shape=image_shape)

    flows = array_ops.placeholder(dtype=tf_type[flow_type], shape=flow_shape)
    return image, flows

  def get_random_image_and_flows(self, shape, image_type, flow_type):
    batch_size, height, width, numchannels = shape
    image_shape = [batch_size, height, width, numchannels]
    image = np.random.normal(size=image_shape)
    flow_shape = [batch_size, height, width, 2]
    flows = np.random.normal(size=flow_shape) * 3
    return image.astype(image_type), flows.astype(flow_type)

  def assert_correct_interpolation_value(self,
                                         image,
                                         flows,
                                         pred_interpolation,
                                         batch_index,
                                         y_index,
                                         x_index,
                                         low_precision=False):
    """Assert that the tf interpolation matches hand-computed value."""

    height = image.shape[1]
    width = image.shape[2]
    displacement = flows[batch_index, y_index, x_index, :]
    float_y = y_index - displacement[0]
    float_x = x_index - displacement[1]
    floor_y = max(min(height - 2, math.floor(float_y)), 0)
    floor_x = max(min(width - 2, math.floor(float_x)), 0)
    ceil_y = floor_y + 1
    ceil_x = floor_x + 1

    alpha_y = min(max(0.0, float_y - floor_y), 1.0)
    alpha_x = min(max(0.0, float_x - floor_x), 1.0)

    floor_y = int(floor_y)
    floor_x = int(floor_x)
    ceil_y = int(ceil_y)
    ceil_x = int(ceil_x)

    top_left = image[batch_index, floor_y, floor_x, :]
    top_right = image[batch_index, floor_y, ceil_x, :]
    bottom_left = image[batch_index, ceil_y, floor_x, :]
    bottom_right = image[batch_index, ceil_y, ceil_x, :]

    interp_top = alpha_x * (top_right - top_left) + top_left
    interp_bottom = alpha_x * (bottom_right - bottom_left) + bottom_left
    interp = alpha_y * (interp_bottom - interp_top) + interp_top
    atol = 1e-6
    rtol = 1e-6
    if low_precision:
      atol = 1e-2
      rtol = 1e-3
    self.assertAllClose(
        interp,
        pred_interpolation[batch_index, y_index, x_index, :],
        atol=atol,
        rtol=rtol)

  def check_zero_flow_correctness(self, shape, image_type, flow_type):
    """Assert using zero flows doesn't change the input image."""

    image, flows = self.get_image_and_flow_placeholders(shape, image_type,
                                                        flow_type)
    interp = dense_image_warp.dense_image_warp(image, flows)

    with self.cached_session() as sess:
      rand_image, rand_flows = self.get_random_image_and_flows(
          shape, image_type, flow_type)
      rand_flows *= 0

      predicted_interpolation = sess.run(
          interp, feed_dict={
              image: rand_image,
              flows: rand_flows
          })
      self.assertAllClose(rand_image, predicted_interpolation)

  def test_zero_flows(self):
    """Apply check_zero_flow_correctness() for a few sizes and types."""

    shapes_to_try = [[3, 4, 5, 6], [1, 2, 2, 1]]
    for shape in shapes_to_try:
      self.check_zero_flow_correctness(
          shape, image_type='float32', flow_type='float32')

  def check_interpolation_correctness(self,
                                      shape,
                                      image_type,
                                      flow_type,
                                      num_probes=5):
    """Interpolate, and then assert correctness for a few query locations."""

    image, flows = self.get_image_and_flow_placeholders(shape, image_type,
                                                        flow_type)
    interp = dense_image_warp.dense_image_warp(image, flows)
    low_precision = image_type == 'float16' or flow_type == 'float16'
    with self.cached_session() as sess:
      rand_image, rand_flows = self.get_random_image_and_flows(
          shape, image_type, flow_type)

      pred_interpolation = sess.run(
          interp, feed_dict={
              image: rand_image,
              flows: rand_flows
          })

      for _ in range(num_probes):
        batch_index = np.random.randint(0, shape[0])
        y_index = np.random.randint(0, shape[1])
        x_index = np.random.randint(0, shape[2])

        self.assert_correct_interpolation_value(
            rand_image,
            rand_flows,
            pred_interpolation,
            batch_index,
            y_index,
            x_index,
            low_precision=low_precision)

  def test_interpolation(self):
    """Apply check_interpolation_correctness() for a few sizes and types."""

    shapes_to_try = [[3, 4, 5, 6], [1, 5, 5, 3], [1, 2, 2, 1]]
    for im_type in ['float32', 'float64', 'float16']:
      for flow_type in ['float32', 'float64', 'float16']:
        for shape in shapes_to_try:
          self.check_interpolation_correctness(shape, im_type, flow_type)

  def test_gradients_exist(self):
    """Check that backprop can run.

    The correctness of the gradients is assumed, since the forward propagation
    is tested to be correct and we only use built-in tf ops.
    However, we perform a simple test to make sure that backprop can actually
    run. We treat the flows as a tf.Variable and optimize them to minimize
    the difference between the interpolated image and the input image.
    """

    batch_size, height, width, numchannels = [4, 5, 6, 7]
    image_shape = [batch_size, height, width, numchannels]
    image = random_ops.random_normal(image_shape)
    flow_shape = [batch_size, height, width, 2]
    init_flows = np.float32(np.random.normal(size=flow_shape) * 0.25)
    flows = variables.Variable(init_flows)

    interp = dense_image_warp.dense_image_warp(image, flows)
    loss = math_ops.reduce_mean(math_ops.square(interp - image))

    optimizer = adam.AdamOptimizer(1.0)
    grad = gradients.gradients(loss, [flows])
    opt_func = optimizer.apply_gradients(zip(grad, [flows]))
    init_op = variables.global_variables_initializer()

    with self.cached_session() as sess:
      sess.run(init_op)
      for _ in range(10):
        sess.run(opt_func)

  def test_size_exception(self):
    """Make sure it throws an exception for images that are too small."""

    shape = [1, 2, 1, 1]
    msg = 'Should have raised an exception for invalid image size'
    with self.assertRaises(errors.InvalidArgumentError, msg=msg):
      self.check_interpolation_correctness(shape, 'float32', 'float32')


if __name__ == '__main__':
  googletest.main()
