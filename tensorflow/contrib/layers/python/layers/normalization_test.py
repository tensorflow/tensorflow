# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for contrib.layers.python.layers.normalization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.framework.python.ops import variables as contrib_variables
from tensorflow.contrib.layers.python.layers import normalization
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class InstanceNormTest(test.TestCase):

  def testUnknownShape(self):
    inputs = array_ops.placeholder(dtypes.float32)
    with self.assertRaisesRegexp(ValueError, 'undefined rank'):
      normalization.instance_norm(inputs)

  def testBadDataFormat(self):
    inputs = array_ops.placeholder(dtypes.float32, shape=(2, 5, 5))
    with self.assertRaisesRegexp(ValueError,
                                 'data_format has to be either NCHW or NHWC.'):
      normalization.instance_norm(inputs, data_format='NHCW')

  def testParamsShapeNotFullyDefinedNCHW(self):
    inputs = array_ops.placeholder(dtypes.float32, shape=(3, None, 4))
    with self.assertRaisesRegexp(ValueError, 'undefined channels dimension'):
      normalization.instance_norm(inputs, data_format='NCHW')

  def testParamsShapeNotFullyDefinedNHWC(self):
    inputs = array_ops.placeholder(dtypes.float32, shape=(3, 4, None))
    with self.assertRaisesRegexp(ValueError, 'undefined channels dimension'):
      normalization.instance_norm(inputs, data_format='NHWC')

  def testCreateOp(self):
    height, width = 3, 3
    images = random_ops.random_uniform((5, height, width, 3), seed=1)
    output = normalization.instance_norm(images)
    print('name: ', output.op.name)
    self.assertStartsWith(
        output.op.name, 'InstanceNorm/instancenorm')
    self.assertListEqual([5, height, width, 3], output.shape.as_list())

  def testCreateOpFloat64(self):
    height, width = 3, 3
    images = random_ops.random_uniform(
        (5, height, width, 3), dtype=dtypes.float64, seed=1)
    output = normalization.instance_norm(images)
    self.assertStartsWith(
        output.op.name, 'InstanceNorm/instancenorm')
    self.assertListEqual([5, height, width, 3], output.shape.as_list())

  def testCreateOpNoScaleCenter(self):
    height, width = 3, 3
    images = random_ops.random_uniform(
        (5, height, width, 3), dtype=dtypes.float64, seed=1)
    output = normalization.instance_norm(images, center=False, scale=False)
    self.assertStartsWith(
        output.op.name, 'InstanceNorm/instancenorm')
    self.assertListEqual([5, height, width, 3], output.shape.as_list())
    self.assertEqual(0, len(contrib_variables.get_variables_by_name('beta')))
    self.assertEqual(0, len(contrib_variables.get_variables_by_name('gamma')))

  def testCreateVariables(self):
    height, width = 3, 3
    images = random_ops.random_uniform((5, height, width, 3), seed=1)
    normalization.instance_norm(images, center=True, scale=True)
    beta = contrib_variables.get_variables_by_name('beta')[0]
    gamma = contrib_variables.get_variables_by_name('gamma')[0]
    self.assertEqual('InstanceNorm/beta', beta.op.name)
    self.assertEqual('InstanceNorm/gamma', gamma.op.name)

  def testReuseVariables(self):
    height, width = 3, 3
    images = random_ops.random_uniform((5, height, width, 3), seed=1)
    normalization.instance_norm(images, scale=True, scope='IN')
    normalization.instance_norm(images, scale=True, scope='IN', reuse=True)
    beta = contrib_variables.get_variables_by_name('beta')
    gamma = contrib_variables.get_variables_by_name('gamma')
    self.assertEqual(1, len(beta))
    self.assertEqual(1, len(gamma))

  def testValueCorrectWithReuseVars(self):
    height, width = 3, 3
    image_shape = (10, height, width, 3)
    images = random_ops.random_uniform(image_shape, seed=1)
    output_train = normalization.instance_norm(images, scope='IN')
    output_eval = normalization.instance_norm(images, scope='IN', reuse=True)
    with self.test_session() as sess:
      sess.run(variables.global_variables_initializer())
      # output_train and output_eval should be the same.
      train_np, eval_np = sess.run([output_train, output_eval])
      self.assertAllClose(train_np, eval_np)

  def doOutputTest(self, input_shape, data_format, tol=1e-3):
    axis = -1 if data_format == 'NHWC' else 1
    for mu in (0.0, 1e2):
      for sigma in (1.0, 0.1):
        # Determine shape of Tensor after normalization.
        reduced_shape = (input_shape[0], input_shape[axis])
        expected_mean = np.zeros(reduced_shape)
        expected_var = np.ones(reduced_shape)

        # Determine axes that will be normalized.
        reduced_axes = list(range(len(input_shape)))
        del reduced_axes[axis]
        del reduced_axes[0]
        reduced_axes = tuple(reduced_axes)

        inputs = random_ops.random_uniform(input_shape, seed=0) * sigma + mu
        output_op = normalization.instance_norm(
            inputs, center=False, scale=False, data_format=data_format)
        with self.test_session() as sess:
          sess.run(variables.global_variables_initializer())
          outputs = sess.run(output_op)
          # Make sure that there are no NaNs
          self.assertFalse(np.isnan(outputs).any())
          mean = np.mean(outputs, axis=reduced_axes)
          var = np.var(outputs, axis=reduced_axes)
          # The mean and variance of each example should be close to 0 and 1
          # respectively.
          self.assertAllClose(expected_mean, mean, rtol=tol, atol=tol)
          self.assertAllClose(expected_var, var, rtol=tol, atol=tol)

  def testOutputSmallInput4DNHWC(self):
    self.doOutputTest((10, 10, 10, 30), 'NHWC', tol=1e-2)

  def testOutputSmallInput4DNCHW(self):
    self.doOutputTest((10, 10, 10, 30), 'NCHW', tol=1e-2)

  def testOutputBigInput4DNHWC(self):
    self.doOutputTest((1, 100, 100, 1), 'NHWC', tol=1e-3)

  def testOutputBigInput4DNCHW(self):
    self.doOutputTest((1, 100, 100, 1), 'NCHW', tol=1e-3)

  def testOutputSmallInput5DNHWC(self):
    self.doOutputTest((10, 10, 10, 10, 30), 'NHWC', tol=1e-2)

  def testOutputSmallInput5DNCHW(self):
    self.doOutputTest((10, 10, 10, 10, 30), 'NCHW', tol=1e-2)

  def testOutputBigInput5DNHWC(self):
    self.doOutputTest((1, 100, 100, 1, 1), 'NHWC', tol=1e-3)

  def testOutputBigInput5DNCHW(self):
    self.doOutputTest((1, 100, 100, 1, 1), 'NCHW', tol=1e-3)


class GroupNormTest(test.TestCase):

  def testInvalidGroupSize(self):
    inputs = array_ops.placeholder(dtypes.float32, shape=(5, 2, 10, 10))
    with self.assertRaisesRegexp(ValueError,
                                 'Invalid groups 10 for 2 channels.'):
      normalization.group_norm(inputs, groups=10,
                               reduction_axes=[-2, -1], channels_axis=-3)

  def testBadCommensurateGroup(self):
    inputs = array_ops.placeholder(dtypes.float32, shape=(5, 4, 10, 10))
    with self.assertRaisesRegexp(ValueError,
                                 '4 channels is not commensurate with '
                                 '3 groups.'):
      normalization.group_norm(inputs, groups=3,
                               reduction_axes=[-2, -1], channels_axis=-3)

  def testAxisIsBad(self):
    inputs = array_ops.placeholder(dtypes.float32, shape=(1, 2, 4, 5))
    with self.assertRaisesRegexp(ValueError,
                                 'Axis is out of bounds.'):
      normalization.group_norm(inputs, channels_axis=5)
    with self.assertRaisesRegexp(ValueError,
                                 'Axis is out of bounds.'):
      normalization.group_norm(inputs, reduction_axes=[1, 5])

  def testNotMutuallyExclusiveAxis(self):
    inputs = array_ops.placeholder(dtypes.float32, shape=(10, 32, 32, 32))
    # Specify axis with negative values.
    with self.assertRaisesRegexp(ValueError, 'mutually exclusive'):
      normalization.group_norm(inputs, channels_axis=-2, reduction_axes=[-2])
    # Specify axis with positive values.
    with self.assertRaisesRegexp(ValueError, 'mutually exclusive'):
      normalization.group_norm(inputs, channels_axis=1, reduction_axes=[1, 3])
    # Specify axis with mixed positive and negative values.
    with self.assertRaisesRegexp(ValueError, 'mutually exclusive'):
      normalization.group_norm(inputs, channels_axis=-2, reduction_axes=[2])

  def testUnknownShape(self):
    inputs = array_ops.placeholder(dtypes.float32)
    with self.assertRaisesRegexp(ValueError, 'undefined rank'):
      normalization.group_norm(inputs)

  def testParamsShapeNotFullyDefinedReductionAxes(self):
    inputs = array_ops.placeholder(dtypes.float32, shape=(1, 32, None, 4))
    with self.assertRaisesRegexp(ValueError, 'undefined dimensions'):
      normalization.group_norm(inputs)

  def testParamsShapeNotFullyDefinedChannelsAxis(self):
    inputs = array_ops.placeholder(dtypes.float32, shape=(1, 3, 4, None))
    with self.assertRaisesRegexp(ValueError, 'undefined channel dimension'):
      normalization.group_norm(inputs, channels_axis=-1,
                               reduction_axes=[-3, -2])

  def testCreateOp(self):
    height, width, groups = 3, 3, 4
    images = random_ops.random_uniform((5, height, width, 2*groups), seed=1)
    output = normalization.group_norm(images, groups=groups, channels_axis=-1,
                                      reduction_axes=[-3, -2])
    print('name: ', output.op.name)
    self.assertListEqual([5, height, width, 2*groups], output.shape.as_list())

  def testCreateOpFloat64(self):
    height, width, groups = 3, 3, 5
    images = random_ops.random_uniform(
        (5, height, width, 4*groups), dtype=dtypes.float64, seed=1)
    output = normalization.group_norm(images, groups=groups)
    self.assertEqual(dtypes.float64, output.dtype)
    self.assertListEqual([5, height, width, 4*groups], output.shape.as_list())

  def testCreateOpNoScaleCenter(self):
    height, width, groups = 3, 3, 7
    images = random_ops.random_uniform(
        (5, height, width, 3*groups), dtype=dtypes.float32, seed=1)
    output = normalization.group_norm(images, groups=groups, center=False,
                                      scale=False)
    self.assertListEqual([5, height, width, 3*groups], output.shape.as_list())
    self.assertEqual(0, len(contrib_variables.get_variables_by_name('beta')))
    self.assertEqual(0, len(contrib_variables.get_variables_by_name('gamma')))

  def testCreateVariables_NHWC(self):
    height, width = 3, 3
    images = random_ops.random_uniform((5, height, width, 8), seed=1)
    normalization.group_norm(images, groups=4,
                             channels_axis=-1, reduction_axes=(-3, -2),
                             center=True, scale=True)
    beta = contrib_variables.get_variables_by_name('beta')[0]
    gamma = contrib_variables.get_variables_by_name('gamma')[0]
    self.assertEqual('GroupNorm/beta', beta.op.name)
    self.assertEqual('GroupNorm/gamma', gamma.op.name)

  def testCreateVariables_NCHW(self):
    height, width, groups = 3, 3, 4
    images = random_ops.random_uniform((5, 2*groups, height, width), seed=1)
    normalization.group_norm(images, groups=4,
                             channels_axis=-3, reduction_axes=(-2, -1),
                             center=True, scale=True)
    beta = contrib_variables.get_variables_by_name('beta')[0]
    gamma = contrib_variables.get_variables_by_name('gamma')[0]
    self.assertEqual('GroupNorm/beta', beta.op.name)
    self.assertEqual('GroupNorm/gamma', gamma.op.name)

  def testReuseVariables(self):
    height, width = 3, 3
    images = random_ops.random_uniform((5, height, width, 4), seed=1)
    normalization.group_norm(images, groups=2, scale=True, scope='IN')
    normalization.group_norm(images, groups=2, scale=True, scope='IN',
                             reuse=True)
    beta = contrib_variables.get_variables_by_name('beta')
    gamma = contrib_variables.get_variables_by_name('gamma')
    self.assertEqual(1, len(beta))
    self.assertEqual(1, len(gamma))

  def testValueCorrectWithReuseVars(self):
    height, width = 3, 3
    image_shape = (10, height, width, 4)
    images = random_ops.random_uniform(image_shape, seed=1)
    output_train = normalization.group_norm(images, groups=2, scope='IN')
    output_eval = normalization.group_norm(images, groups=2, scope='IN',
                                           reuse=True)
    with self.test_session() as sess:
      sess.run(variables.global_variables_initializer())
      # output_train and output_eval should be the same.
      train_np, eval_np = sess.run([output_train, output_eval])
      self.assertAllClose(train_np, eval_np)

  def doOutputTest(self, input_shape, channels_axis=None, reduction_axes=None,
                   groups=2, tol=1e-2):
    # Select the axis for the channel and the dimensions along which statistics
    # are accumulated.
    if channels_axis < 0:
      channels_axis += len(input_shape)
    reduced_axes = [channels_axis + 1]
    for a in reduction_axes:
      if a < 0:
        a += len(input_shape)
      if a < channels_axis:
        reduced_axes.append(a)
      else:
        reduced_axes.append(a+1)
    reduced_axes = tuple(reduced_axes)

    # Calculate the final shape for the output Tensor.
    axes_before_channels = input_shape[:channels_axis]
    axes_after_channels = input_shape[channels_axis+1:]
    channels = input_shape[channels_axis]
    outputs_shape = (axes_before_channels + [groups, channels // groups] +
                     axes_after_channels)

    # Calculate the final shape for the output statistics.
    reduced_shape = []
    for i, a in enumerate(outputs_shape):
      if i not in reduced_axes:
        reduced_shape.append(a)

    for mu in (0.0, 1e2):
      for sigma in (1.0, 0.1):
        # Determine shape of Tensor after normalization.
        expected_mean = np.zeros(reduced_shape)
        expected_var = np.ones(reduced_shape)

        inputs = random_ops.random_uniform(input_shape, seed=0) * sigma + mu
        output_op = normalization.group_norm(
            inputs, groups=groups, center=False, scale=False,
            channels_axis=channels_axis,
            reduction_axes=reduction_axes)
        with self.test_session() as sess:
          sess.run(variables.global_variables_initializer())
          outputs = sess.run(output_op)
          # Make sure that there are no NaNs
          self.assertFalse(np.isnan(outputs).any())

          outputs = np.reshape(outputs, outputs_shape)
          mean = np.mean(outputs, axis=reduced_axes)
          var = np.var(outputs, axis=reduced_axes)
          # The mean and variance of each example should be close to 0 and 1
          # respectively.
          self.assertAllClose(expected_mean, mean, rtol=tol, atol=tol)
          self.assertAllClose(expected_var, var, rtol=tol, atol=tol)

  def testOutputSmallInput4D_NHWC(self):
    input_shape = [10, 10, 10, 30]
    # Specify axes with positive values.
    self.doOutputTest(input_shape, channels_axis=3, reduction_axes=[1, 2])
    # Specify axes with negative values.
    self.doOutputTest(input_shape, channels_axis=-1, reduction_axes=[-3, -2])

  def testOutputSmallInput3D_NHWC(self):
    input_shape = [10, 10, 30]
    # Specify axes with positive values.
    self.doOutputTest(input_shape, channels_axis=2, reduction_axes=[0, 1])
    # Specify axes with negative values.
    self.doOutputTest(input_shape, channels_axis=-1, reduction_axes=[-3, -2])

  def testOutputSmallInput4D_NCHW(self):
    input_shape = [10, 10, 10, 30]
    # Specify axes with positive values.
    self.doOutputTest(input_shape, channels_axis=1, reduction_axes=[2, 3])
    # Specify axes with negative values.
    self.doOutputTest(input_shape, channels_axis=-3, reduction_axes=[-2, -1])

  def testOutputSmallInput3D_NCHW(self):
    input_shape = [10, 10, 30]
    # Specify axes with positive values.
    self.doOutputTest(input_shape, channels_axis=0, reduction_axes=[1, 2])
    # Specify axes with negative values.
    self.doOutputTest(input_shape, channels_axis=-3, reduction_axes=[-2, -1])

  def testOutputBigInput4D_NHWC(self):
    self.doOutputTest([5, 100, 100, 1], channels_axis=3, reduction_axes=[1, 2],
                      groups=1)

  def testOutputBigInput4D_NCHW(self):
    self.doOutputTest([1, 100, 100, 4], channels_axis=1, reduction_axes=[2, 3],
                      groups=4)

  def testOutputSmallInput2D_NC(self):
    self.doOutputTest([10, 7*100], channels_axis=1, reduction_axes=[], groups=7)

  def testOutputSmallInput5D_NCXXX(self):
    self.doOutputTest([10, 10, 20, 40, 5],
                      channels_axis=1,
                      reduction_axes=[2, 3, 4],
                      groups=5)

if __name__ == '__main__':
  test.main()
