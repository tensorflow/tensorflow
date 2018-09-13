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
"""Tests for interpolate_spline."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy import interpolate as sc_interpolate

from tensorflow.contrib.image.python.ops import interpolate_spline

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest

from tensorflow.python.training import momentum


class _InterpolationProblem(object):
  """Abstract class for interpolation problem descriptions."""

  def get_problem(self, optimizable=False, extrapolate=True, dtype='float32'):
    """Make data for an interpolation problem where all x vectors are n-d.

    Args:
      optimizable: If True, then make train_points a tf.Variable.
      extrapolate: If False, then clamp the query_points values to be within
      the max and min of train_points.
      dtype: The data type to use.

    Returns:
      query_points, query_values, train_points, train_values: training and
      test tensors for interpolation problem
    """

    # The values generated here depend on a seed of 0.
    np.random.seed(0)

    batch_size = 1
    num_training_points = 10
    num_query_points = 4

    init_points = np.random.uniform(
        size=[batch_size, num_training_points, self.DATA_DIM])

    init_points = init_points.astype(dtype)
    train_points = (
        variables.Variable(init_points)
        if optimizable else constant_op.constant(init_points))
    train_values = self.tf_function(train_points)

    query_points_np = np.random.uniform(
        size=[batch_size, num_query_points, self.DATA_DIM])
    query_points_np = query_points_np.astype(dtype)
    if not extrapolate:
      query_points_np = np.clip(query_points_np, np.min(init_points),
                                np.max(init_points))

    query_points = constant_op.constant(query_points_np)
    query_values = self.np_function(query_points_np)

    return query_points, query_values, train_points, train_values


class _QuadraticPlusSinProblem1D(_InterpolationProblem):
  """1D interpolation problem used for regression testing."""
  DATA_DIM = 1
  HARDCODED_QUERY_VALUES = {
      (1.0, 0.0): [6.2647187603, -7.84362604077, -5.63690142322, 1.42928896387],
      (1.0,
       0.01): [6.77688289946, -8.02163669853, -5.79491157027, 1.4063285693],
      (2.0,
       0.0): [8.67110264937, -8.41281390883, -5.80190044693, 1.50155606059],
      (2.0,
       0.01): [6.70797816797, -7.49709587663, -5.28965776238, 1.52284731741],
      (3.0,
       0.0): [9.37691802935, -8.50390141515, -5.80786417426, 1.63467762122],
      (3.0,
       0.01): [4.47106304758, -5.71266128361, -3.92529303296, 1.86755293857],
      (4.0,
       0.0): [9.58172461111, -8.51432104771, -5.80967675388, 1.63361164256],
      (4.0, 0.01): [
          -3.87902711352, -0.0253462273846, 1.79857618022, -0.769339675725
      ]
  }

  def np_function(self, x):
    """Takes np array, evaluates the test function, and returns np array."""
    return np.sum(
        np.power((x - 0.5), 3) - 0.25 * x + 10 * np.sin(x * 10),
        axis=2,
        keepdims=True)

  def tf_function(self, x):
    """Takes tf tensor, evaluates the test function,  and returns tf tensor."""
    return math_ops.reduce_mean(
        math_ops.pow((x - 0.5), 3) - 0.25 * x + 10 * math_ops.sin(x * 10),
        2,
        keepdims=True)


class _QuadraticPlusSinProblemND(_InterpolationProblem):
  """3D interpolation problem used for regression testing."""

  DATA_DIM = 3
  HARDCODED_QUERY_VALUES = {
      (1.0, 0.0): [1.06609663962, 1.28894849357, 1.10882405595, 1.63966936885],
      (1.0, 0.01): [1.03123780748, 1.2952930985, 1.10366822954, 1.65265118569],
      (2.0, 0.0): [0.627787735064, 1.43802857251, 1.00194632358, 1.91667538215],
      (2.0, 0.01): [0.730159985046, 1.41702471595, 1.0065827217, 1.85758519312],
      (3.0, 0.0): [0.350460417862, 1.67223539464, 1.00475331246, 2.31580322491],
      (3.0,
       0.01): [0.624557250556, 1.63138876667, 0.976588193162, 2.12511237866],
      (4.0,
       0.0): [0.898129669986, 1.24434133638, -0.938056116931, 1.59910338833],
      (4.0,
       0.01): [0.0930360338179, -3.38791305538, -1.00969032567, 0.745535080382],
  }

  def np_function(self, x):
    """Takes np array, evaluates the test function, and returns np array."""
    return np.sum(
        np.square(x - 0.5) + 0.25 * x + 1 * np.sin(x * 15),
        axis=2,
        keepdims=True)

  def tf_function(self, x):
    """Takes tf tensor, evaluates the test function,  and returns tf tensor."""
    return math_ops.reduce_sum(
        math_ops.square(x - 0.5) + 0.25 * x + 1 * math_ops.sin(x * 15),
        2,
        keepdims=True)


class InterpolateSplineTest(test_util.TensorFlowTestCase):

  def test_1d_linear_interpolation(self):
    """For 1d linear interpolation, we can compare directly to scipy."""

    tp = _QuadraticPlusSinProblem1D()
    (query_points, _, train_points, train_values) = tp.get_problem(
        extrapolate=False, dtype='float64')
    interpolation_order = 1

    with ops.name_scope('interpolator'):
      interpolator = interpolate_spline.interpolate_spline(
          train_points, train_values, query_points, interpolation_order)
      with self.cached_session() as sess:
        fetches = [query_points, train_points, train_values, interpolator]
        query_points_, train_points_, train_values_, interp_ = sess.run(fetches)

        # Just look at the first element of the minibatch.
        # Also, trim the final singleton dimension.
        interp_ = interp_[0, :, 0]
        query_points_ = query_points_[0, :, 0]
        train_points_ = train_points_[0, :, 0]
        train_values_ = train_values_[0, :, 0]

        # Compute scipy interpolation.
        scipy_interp_function = sc_interpolate.interp1d(
            train_points_, train_values_, kind='linear')

        scipy_interpolation = scipy_interp_function(query_points_)
        scipy_interpolation_on_train = scipy_interp_function(train_points_)

        # Even with float64 precision, the interpolants disagree with scipy a
        # bit due to the fact that we add the EPSILON to prevent sqrt(0), etc.
        tol = 1e-3

        self.assertAllClose(
            train_values_, scipy_interpolation_on_train, atol=tol, rtol=tol)
        self.assertAllClose(interp_, scipy_interpolation, atol=tol, rtol=tol)

  def test_1d_interpolation(self):
    """Regression test for interpolation with 1-D points."""

    tp = _QuadraticPlusSinProblem1D()
    (query_points, _, train_points,
     train_values) = tp.get_problem(dtype='float64')

    for order in (1, 2, 3):
      for reg_weight in (0, 0.01):
        interpolator = interpolate_spline.interpolate_spline(
            train_points, train_values, query_points, order, reg_weight)

        target_interpolation = tp.HARDCODED_QUERY_VALUES[(order, reg_weight)]
        target_interpolation = np.array(target_interpolation)
        with self.cached_session() as sess:
          interp_val = sess.run(interpolator)
          self.assertAllClose(interp_val[0, :, 0], target_interpolation)

  def test_nd_linear_interpolation(self):
    """Regression test for interpolation with N-D points."""

    tp = _QuadraticPlusSinProblemND()
    (query_points, _, train_points,
     train_values) = tp.get_problem(dtype='float64')

    for order in (1, 2, 3):
      for reg_weight in (0, 0.01):
        interpolator = interpolate_spline.interpolate_spline(
            train_points, train_values, query_points, order, reg_weight)

        target_interpolation = tp.HARDCODED_QUERY_VALUES[(order, reg_weight)]
        target_interpolation = np.array(target_interpolation)
        with self.cached_session() as sess:
          interp_val = sess.run(interpolator)
          self.assertAllClose(interp_val[0, :, 0], target_interpolation)

  def test_nd_linear_interpolation_unspecified_shape(self):
    """Ensure that interpolation supports dynamic batch_size and num_points."""

    tp = _QuadraticPlusSinProblemND()
    (query_points, _, train_points,
     train_values) = tp.get_problem(dtype='float64')

    # Construct placeholders such that the batch size, number of train points,
    # and number of query points are not known at graph construction time.
    feature_dim = query_points.shape[-1]
    value_dim = train_values.shape[-1]
    train_points_ph = array_ops.placeholder(
        dtype=train_points.dtype, shape=[None, None, feature_dim])
    train_values_ph = array_ops.placeholder(
        dtype=train_values.dtype, shape=[None, None, value_dim])
    query_points_ph = array_ops.placeholder(
        dtype=query_points.dtype, shape=[None, None, feature_dim])

    order = 1
    reg_weight = 0.01

    interpolator = interpolate_spline.interpolate_spline(
        train_points_ph, train_values_ph, query_points_ph, order, reg_weight)

    target_interpolation = tp.HARDCODED_QUERY_VALUES[(order, reg_weight)]
    target_interpolation = np.array(target_interpolation)
    with self.cached_session() as sess:

      (train_points_value, train_values_value, query_points_value) = sess.run(
          [train_points, train_values, query_points])

      interp_val = sess.run(
          interpolator,
          feed_dict={
              train_points_ph: train_points_value,
              train_values_ph: train_values_value,
              query_points_ph: query_points_value
          })
      self.assertAllClose(interp_val[0, :, 0], target_interpolation)

  def test_fully_unspecified_shape(self):
    """Ensure that erreor is thrown when input/output dim unspecified."""

    tp = _QuadraticPlusSinProblemND()
    (query_points, _, train_points,
     train_values) = tp.get_problem(dtype='float64')

    # Construct placeholders such that the batch size, number of train points,
    # and number of query points are not known at graph construction time.
    feature_dim = query_points.shape[-1]
    value_dim = train_values.shape[-1]
    train_points_ph = array_ops.placeholder(
        dtype=train_points.dtype, shape=[None, None, feature_dim])
    train_points_ph_invalid = array_ops.placeholder(
        dtype=train_points.dtype, shape=[None, None, None])
    train_values_ph = array_ops.placeholder(
        dtype=train_values.dtype, shape=[None, None, value_dim])
    train_values_ph_invalid = array_ops.placeholder(
        dtype=train_values.dtype, shape=[None, None, None])
    query_points_ph = array_ops.placeholder(
        dtype=query_points.dtype, shape=[None, None, feature_dim])

    order = 1
    reg_weight = 0.01

    with self.assertRaises(ValueError):
      _ = interpolate_spline.interpolate_spline(
          train_points_ph_invalid, train_values_ph, query_points_ph, order,
          reg_weight)

    with self.assertRaises(ValueError):
      _ = interpolate_spline.interpolate_spline(
          train_points_ph, train_values_ph_invalid, query_points_ph, order,
          reg_weight)

  def test_interpolation_gradient(self):
    """Make sure that backprop can run. Correctness of gradients is assumed.

    Here, we create a use a small 'training' set and a more densely-sampled
    set of query points, for which we know the true value in advance. The goal
    is to choose x locations for the training data such that interpolating using
    this training data yields the best reconstruction for the function
    values at the query points. The training data locations are optimized
    iteratively using gradient descent.
    """
    tp = _QuadraticPlusSinProblemND()
    (query_points, query_values, train_points,
     train_values) = tp.get_problem(optimizable=True)

    regularization = 0.001
    for interpolation_order in (1, 2, 3, 4):
      interpolator = interpolate_spline.interpolate_spline(
          train_points, train_values, query_points, interpolation_order,
          regularization)

      loss = math_ops.reduce_mean(math_ops.square(query_values - interpolator))

      optimizer = momentum.MomentumOptimizer(0.001, 0.9)
      grad = gradients.gradients(loss, [train_points])
      grad, _ = clip_ops.clip_by_global_norm(grad, 1.0)
      opt_func = optimizer.apply_gradients(zip(grad, [train_points]))
      init_op = variables.global_variables_initializer()

      with self.cached_session() as sess:
        sess.run(init_op)
        for _ in range(100):
          sess.run([loss, opt_func])


if __name__ == '__main__':
  googletest.main()
