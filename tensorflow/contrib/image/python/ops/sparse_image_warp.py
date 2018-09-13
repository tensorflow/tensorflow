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
"""Image warping using sparse flow defined at control points."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.image.python.ops import dense_image_warp
from tensorflow.contrib.image.python.ops import interpolate_spline

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops


def _get_grid_locations(image_height, image_width):
  """Wrapper for np.meshgrid."""

  y_range = np.linspace(0, image_height - 1, image_height)
  x_range = np.linspace(0, image_width - 1, image_width)
  y_grid, x_grid = np.meshgrid(y_range, x_range, indexing='ij')
  return np.stack((y_grid, x_grid), -1)


def _expand_to_minibatch(np_array, batch_size):
  """Tile arbitrarily-sized np_array to include new batch dimension."""
  tiles = [batch_size] + [1] * np_array.ndim
  return np.tile(np.expand_dims(np_array, 0), tiles)


def _get_boundary_locations(image_height, image_width, num_points_per_edge):
  """Compute evenly-spaced indices along edge of image."""
  y_range = np.linspace(0, image_height - 1, num_points_per_edge + 2)
  x_range = np.linspace(0, image_width - 1, num_points_per_edge + 2)
  ys, xs = np.meshgrid(y_range, x_range, indexing='ij')
  is_boundary = np.logical_or(
      np.logical_or(xs == 0, xs == image_width - 1),
      np.logical_or(ys == 0, ys == image_height - 1))
  return np.stack([ys[is_boundary], xs[is_boundary]], axis=-1)


def _add_zero_flow_controls_at_boundary(control_point_locations,
                                        control_point_flows, image_height,
                                        image_width, boundary_points_per_edge):
  """Add control points for zero-flow boundary conditions.

   Augment the set of control points with extra points on the
   boundary of the image that have zero flow.

  Args:
    control_point_locations: input control points
    control_point_flows: their flows
    image_height: image height
    image_width: image width
    boundary_points_per_edge: number of points to add in the middle of each
                           edge (not including the corners).
                           The total number of points added is
                           4 + 4*(boundary_points_per_edge).

  Returns:
    merged_control_point_locations: augmented set of control point locations
    merged_control_point_flows: augmented set of control point flows
  """

  batch_size = control_point_locations.get_shape()[0].value

  boundary_point_locations = _get_boundary_locations(image_height, image_width,
                                                     boundary_points_per_edge)

  boundary_point_flows = np.zeros([boundary_point_locations.shape[0], 2])

  type_to_use = control_point_locations.dtype
  boundary_point_locations = constant_op.constant(
      _expand_to_minibatch(boundary_point_locations, batch_size),
      dtype=type_to_use)

  boundary_point_flows = constant_op.constant(
      _expand_to_minibatch(boundary_point_flows, batch_size), dtype=type_to_use)

  merged_control_point_locations = array_ops.concat(
      [control_point_locations, boundary_point_locations], 1)

  merged_control_point_flows = array_ops.concat(
      [control_point_flows, boundary_point_flows], 1)

  return merged_control_point_locations, merged_control_point_flows


def sparse_image_warp(image,
                      source_control_point_locations,
                      dest_control_point_locations,
                      interpolation_order=2,
                      regularization_weight=0.0,
                      num_boundary_points=0,
                      name='sparse_image_warp'):
  """Image warping using correspondences between sparse control points.

  Apply a non-linear warp to the image, where the warp is specified by
  the source and destination locations of a (potentially small) number of
  control points. First, we use a polyharmonic spline
  (`tf.contrib.image.interpolate_spline`) to interpolate the displacements
  between the corresponding control points to a dense flow field.
  Then, we warp the image using this dense flow field
  (`tf.contrib.image.dense_image_warp`).

  Let t index our control points. For regularization_weight=0, we have:
  warped_image[b, dest_control_point_locations[b, t, 0],
                  dest_control_point_locations[b, t, 1], :] =
  image[b, source_control_point_locations[b, t, 0],
           source_control_point_locations[b, t, 1], :].

  For regularization_weight > 0, this condition is met approximately, since
  regularized interpolation trades off smoothness of the interpolant vs.
  reconstruction of the interpolant at the control points.
  See `tf.contrib.image.interpolate_spline` for further documentation of the
  interpolation_order and regularization_weight arguments.


  Args:
    image: `[batch, height, width, channels]` float `Tensor`
    source_control_point_locations: `[batch, num_control_points, 2]` float
      `Tensor`
    dest_control_point_locations: `[batch, num_control_points, 2]` float
      `Tensor`
    interpolation_order: polynomial order used by the spline interpolation
    regularization_weight: weight on smoothness regularizer in interpolation
    num_boundary_points: How many zero-flow boundary points to include at
      each image edge.Usage:
        num_boundary_points=0: don't add zero-flow points
        num_boundary_points=1: 4 corners of the image
        num_boundary_points=2: 4 corners and one in the middle of each edge
          (8 points total)
        num_boundary_points=n: 4 corners and n-1 along each edge
    name: A name for the operation (optional).

    Note that image and offsets can be of type tf.half, tf.float32, or
    tf.float64, and do not necessarily have to be the same type.

  Returns:
    warped_image: `[batch, height, width, channels]` float `Tensor` with same
      type as input image.
    flow_field: `[batch, height, width, 2]` float `Tensor` containing the dense
      flow field produced by the interpolation.
  """

  image = ops.convert_to_tensor(image)
  source_control_point_locations = ops.convert_to_tensor(
      source_control_point_locations)
  dest_control_point_locations = ops.convert_to_tensor(
      dest_control_point_locations)

  control_point_flows = (
      dest_control_point_locations - source_control_point_locations)

  clamp_boundaries = num_boundary_points > 0
  boundary_points_per_edge = num_boundary_points - 1

  with ops.name_scope(name):

    batch_size, image_height, image_width, _ = image.get_shape().as_list()

    # This generates the dense locations where the interpolant
    # will be evaluated.
    grid_locations = _get_grid_locations(image_height, image_width)

    flattened_grid_locations = np.reshape(grid_locations,
                                          [image_height * image_width, 2])

    flattened_grid_locations = constant_op.constant(
        _expand_to_minibatch(flattened_grid_locations, batch_size), image.dtype)

    if clamp_boundaries:
      (dest_control_point_locations,
       control_point_flows) = _add_zero_flow_controls_at_boundary(
           dest_control_point_locations, control_point_flows, image_height,
           image_width, boundary_points_per_edge)

    flattened_flows = interpolate_spline.interpolate_spline(
        dest_control_point_locations, control_point_flows,
        flattened_grid_locations, interpolation_order, regularization_weight)

    dense_flows = array_ops.reshape(flattened_flows,
                                    [batch_size, image_height, image_width, 2])

    warped_image = dense_image_warp.dense_image_warp(image, dense_flows)

    return warped_image, dense_flows
