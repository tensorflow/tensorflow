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

if __name__ == '__main__':
  test.main()
