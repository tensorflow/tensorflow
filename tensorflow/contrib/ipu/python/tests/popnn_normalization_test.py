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
from tensorflow.contrib.ipu.python import popnn_normalization
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test

# This implementation is based on:
# tensorflow/contrib/layers/python/layers/normalization_test.py

class PopnnGroupNormTest(test.TestCase):

  def testInvalidGroupSize(self):
    inputs = array_ops.placeholder(dtypes.float32, shape=(5, 2, 10, 10))
    with self.assertRaisesRegexp(ValueError,
                                 'Invalid groups 10 for 2 channels.'):
      popnn_normalization.group_norm(inputs, groups=10,
                               reduction_axes=[-2, -1], channels_axis=-3)

  def testBadCommensurateGroup(self):
    inputs = array_ops.placeholder(dtypes.float32, shape=(5, 4, 10, 10))
    with self.assertRaisesRegexp(ValueError,
                                 '4 channels is not commensurate with '
                                 '3 groups.'):
      popnn_normalization.group_norm(inputs, groups=3,
                               reduction_axes=[-2, -1], channels_axis=-3)

  def testAxisIsBad(self):
    inputs = array_ops.placeholder(dtypes.float32, shape=(1, 2, 4, 5))
    with self.assertRaisesRegexp(ValueError,
                                 'Axis is out of bounds.'):
      popnn_normalization.group_norm(inputs, channels_axis=5)
    with self.assertRaisesRegexp(ValueError,
                                 'Axis is out of bounds.'):
      popnn_normalization.group_norm(inputs, reduction_axes=[1, 5])

  def testNotMutuallyExclusiveAxis(self):
    inputs = array_ops.placeholder(dtypes.float32, shape=(10, 32, 32, 32))
    # Specify axis with negative values.
    with self.assertRaisesRegexp(ValueError, 'mutually exclusive'):
      popnn_normalization.group_norm(inputs, channels_axis=-1, reduction_axes=[-1])
    # Specify axis with positive values.
    with self.assertRaisesRegexp(ValueError, 'mutually exclusive'):
      popnn_normalization.group_norm(inputs, channels_axis=1, reduction_axes=[1, 3])
    # Specify axis with mixed positive and negative values.
    with self.assertRaisesRegexp(ValueError, 'mutually exclusive'):
      popnn_normalization.group_norm(inputs, channels_axis=-1, reduction_axes=[3])

  def testUnknownShape(self):
    inputs = array_ops.placeholder(dtypes.float32)
    with self.assertRaisesRegexp(ValueError, 'undefined rank'):
      popnn_normalization.group_norm(inputs)

  def testParamsShapeNotFullyDefinedReductionAxes(self):
    inputs = array_ops.placeholder(dtypes.float32, shape=(1, 32, None, 4))
    with self.assertRaisesRegexp(ValueError, 'undefined dimensions'):
      popnn_normalization.group_norm(inputs)

  def testParamsShapeNotFullyDefinedChannelsAxis(self):
    inputs = array_ops.placeholder(dtypes.float32, shape=(1, 3, 4, None))
    with self.assertRaisesRegexp(ValueError, 'undefined channel dimension'):
      popnn_normalization.group_norm(inputs, channels_axis=-1,
                               reduction_axes=[-3, -2])

  def testCreateOp(self):
    height, width, groups = 3, 3, 4
    images = random_ops.random_uniform((5, height, width, 2*groups), seed=1)
    output = popnn_normalization.group_norm(images, groups=groups, channels_axis=-1,
                                      reduction_axes=[-3, -2])
    print('name: ', output.op.name)
    self.assertListEqual([5, height, width, 2*groups], output.shape.as_list())

  def testCreateOpNoScaleCenter(self):
    height, width, groups = 3, 3, 7
    images = random_ops.random_uniform(
        (5, height, width, 3*groups), dtype=dtypes.float32, seed=1)
    output = popnn_normalization.group_norm(images, groups=groups, center=False,
                                      scale=False)
    self.assertListEqual([5, height, width, 3*groups], output.shape.as_list())
    self.assertEqual(0, len(contrib_variables.get_variables_by_name('beta')))
    self.assertEqual(0, len(contrib_variables.get_variables_by_name('gamma')))

  def testCreateVariables_NHWC(self):
    height, width = 3, 3
    images = random_ops.random_uniform((5, height, width, 8), seed=1)
    popnn_normalization.group_norm(images, groups=4,
                             channels_axis=-1, reduction_axes=(-3, -2),
                             center=True, scale=True)
    beta = contrib_variables.get_variables_by_name('beta')[0]
    gamma = contrib_variables.get_variables_by_name('gamma')[0]
    self.assertEqual('GroupNorm/beta', beta.op.name)
    self.assertEqual('GroupNorm/gamma', gamma.op.name)

  def testCreateVariables_NCHW(self):
    height, width, groups = 3, 3, 4
    images = random_ops.random_uniform((5, 2*groups, height, width), seed=1)
    popnn_normalization.group_norm(images, groups=4,
                             channels_axis=-3, reduction_axes=(-2, -1),
                             center=True, scale=True)
    beta = contrib_variables.get_variables_by_name('beta')[0]
    gamma = contrib_variables.get_variables_by_name('gamma')[0]
    self.assertEqual('GroupNorm/beta', beta.op.name)
    self.assertEqual('GroupNorm/gamma', gamma.op.name)

  def testReuseVariables(self):
    height, width = 3, 3
    images = random_ops.random_uniform((5, height, width, 4), seed=1)
    popnn_normalization.group_norm(images, groups=2, scale=True, scope='IN')
    popnn_normalization.group_norm(images, groups=2, scale=True, scope='IN',
                             reuse=True)
    beta = contrib_variables.get_variables_by_name('beta')
    gamma = contrib_variables.get_variables_by_name('gamma')
    self.assertEqual(1, len(beta))
    self.assertEqual(1, len(gamma))

  def testValueCorrectWithReuseVars(self):
    height, width = 3, 3
    image_shape = (10, height, width, 4)
    images = random_ops.random_uniform(image_shape, seed=1)
    output_train = popnn_normalization.group_norm(images, groups=2, scope='IN')
    output_eval = popnn_normalization.group_norm(images, groups=2, scope='IN',
                                           reuse=True)
    with self.cached_session() as sess:
      sess.run(variables.global_variables_initializer())
      # output_train and output_eval should be the same.
      train_np, eval_np = sess.run([output_train, output_eval])
      self.assertAllClose(train_np, eval_np)

  def doOutputTest(self,
                   input_shape,
                   channels_axis=None,
                   reduction_axes=None,
                   groups=2,
                   tol=1e-1):
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
    channels = input_shape[channels_axis]
    group_size = channels // groups
    # Calculate the final shape for the output Tensor.
    axes_before_channels = input_shape[:channels_axis]
    axes_after_channels = input_shape[channels_axis+1:]
    outputs_shape = (axes_before_channels + [groups, group_size] +
                     axes_after_channels)

    # Calculate the final shape for the output statistics.
    reduced_shape = []
    for i, a in enumerate(outputs_shape):
      if i not in reduced_axes:
        reduced_shape.append(a)

    mu = 1.0
    sigma = 1.0
    # Determine shape of Tensor after normalization.
    expected_mean = np.zeros(reduced_shape)
    expected_var = np.ones(reduced_shape)

    inputs = random_ops.random_normal(input_shape, seed=0) * sigma + mu
    output_op = popnn_normalization.group_norm(
        inputs,
        groups=groups,
        center=False,
        scale=False,
        channels_axis=channels_axis,
        reduction_axes=reduction_axes,
        training=True)
    with self.cached_session() as sess:
      sess.run(variables.global_variables_initializer())
      outputs = sess.run(output_op)

      # Make sure that there are no NaNs
      self.assertFalse(np.isnan(outputs).any())

      # Implementation detail - in Poplibs group norm, the groups are not
      # contiguous, but strided - we replicate that here
      # Move the channels to the first dimension for inputs, gamma and beta
      outputs = np.swapaxes(outputs, 0, channels_axis)
      reshuffled_outputs = np.empty(outputs.shape, outputs.dtype)
      for from_idx in range(channels):
        to_idx = (from_idx % groups) * group_size + from_idx // groups
        reshuffled_outputs[to_idx] = outputs[from_idx]
      outputs = np.swapaxes(reshuffled_outputs, 0, channels_axis)

      outputs = np.reshape(outputs, outputs_shape)
      mean = np.mean(outputs, axis=reduced_axes, dtype=np.float32)
      var = np.var(outputs, axis=reduced_axes, dtype=np.float32)
      # The mean and variance of each example should be close to 0 and 1
      # respectively.
      self.assertAllClose(expected_mean, mean, rtol=tol, atol=tol)
      self.assertAllClose(expected_var, var, rtol=tol, atol=tol)

  def testOutput4D_NHWC(self):
    input_shape = [10, 10, 10, 30]
    # Specify axes with positive values.
    self.doOutputTest(input_shape, channels_axis=3, reduction_axes=[1, 2])
    # Specify axes with negative values.
    self.doOutputTest(input_shape, channels_axis=-1, reduction_axes=[-3, -2])

  def testOutput3D_NHWC(self):
    input_shape = [10, 10, 30]
    # Specify axes with positive values.
    self.doOutputTest(input_shape, channels_axis=2, reduction_axes=[0, 1])
    # Specify axes with negative values.
    self.doOutputTest(input_shape, channels_axis=-1, reduction_axes=[-3, -2])

  def testOutput4D_NCHW(self):
    input_shape = [10, 10, 10, 30]
    # Specify axes with positive values.
    self.doOutputTest(input_shape, channels_axis=1, reduction_axes=[2, 3])
    # Specify axes with negative values.
    self.doOutputTest(input_shape, channels_axis=-3, reduction_axes=[-2, -1])

  def testOutput3D_NCHW(self):
    input_shape = [10, 10, 30]
    # Specify axes with positive values.
    self.doOutputTest(input_shape, channels_axis=1, reduction_axes=[0, 2])
    # Specify axes with negative values.
    self.doOutputTest(input_shape, channels_axis=-2, reduction_axes=[-3, -1])

  def testOutput2D_NC(self):
    self.doOutputTest(
        [10, 7 * 100], channels_axis=1, reduction_axes=[], groups=7)

  def testOutput5D_NCXXX(self):
    self.doOutputTest(
        [4, 4, 4, 10, 4],
        channels_axis=1,
        reduction_axes=[2, 3, 4],
        groups=2)


if __name__ == '__main__':
  test.main()
