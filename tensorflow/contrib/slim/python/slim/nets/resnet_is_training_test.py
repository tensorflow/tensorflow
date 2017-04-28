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
"""Specifying is_training in resnet_arg_scope is being deprecated.

Test that everything behaves as expected in the meantime.

Note: This test modifies the layers.batch_norm function.
Other tests that use layers.batch_norm may not work if added to this file.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.slim.python.slim.nets import resnet_utils
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
from tensorflow.contrib.slim.python.slim.nets import resnet_v2
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


def create_test_input(batch, height, width, channels):
  """Create test input tensor."""
  if None in [batch, height, width, channels]:
    return array_ops.placeholder(dtypes.float32, (batch, height, width,
                                                  channels))
  else:
    return math_ops.to_float(
        np.tile(
            np.reshape(
                np.reshape(np.arange(height), [height, 1]) +
                np.reshape(np.arange(width), [1, width]),
                [1, height, width, 1]),
            [batch, 1, 1, channels]))


class ResnetIsTrainingTest(test.TestCase):

  def _testDeprecatingIsTraining(self, network_fn):
    batch_norm_fn = layers.batch_norm

    @add_arg_scope
    def batch_norm_expect_is_training(*args, **kwargs):
      assert kwargs['is_training']
      return batch_norm_fn(*args, **kwargs)

    @add_arg_scope
    def batch_norm_expect_is_not_training(*args, **kwargs):
      assert not kwargs['is_training']
      return batch_norm_fn(*args, **kwargs)

    global_pool = True
    num_classes = 10
    inputs = create_test_input(2, 224, 224, 3)

    # Default argument for resnet_arg_scope
    layers.batch_norm = batch_norm_expect_is_training
    with arg_scope(resnet_utils.resnet_arg_scope()):
      network_fn(inputs, num_classes, global_pool=global_pool, scope='resnet1')

    layers.batch_norm = batch_norm_expect_is_training
    with arg_scope(resnet_utils.resnet_arg_scope()):
      network_fn(
          inputs,
          num_classes,
          is_training=True,
          global_pool=global_pool,
          scope='resnet2')

    layers.batch_norm = batch_norm_expect_is_not_training
    with arg_scope(resnet_utils.resnet_arg_scope()):
      network_fn(
          inputs,
          num_classes,
          is_training=False,
          global_pool=global_pool,
          scope='resnet3')

    # resnet_arg_scope with is_training set to True (deprecated)
    layers.batch_norm = batch_norm_expect_is_training
    with arg_scope(resnet_utils.resnet_arg_scope(is_training=True)):
      network_fn(inputs, num_classes, global_pool=global_pool, scope='resnet4')

    layers.batch_norm = batch_norm_expect_is_training
    with arg_scope(resnet_utils.resnet_arg_scope(is_training=True)):
      network_fn(
          inputs,
          num_classes,
          is_training=True,
          global_pool=global_pool,
          scope='resnet5')

    layers.batch_norm = batch_norm_expect_is_not_training
    with arg_scope(resnet_utils.resnet_arg_scope(is_training=True)):
      network_fn(
          inputs,
          num_classes,
          is_training=False,
          global_pool=global_pool,
          scope='resnet6')

    # resnet_arg_scope with is_training set to False (deprecated)
    layers.batch_norm = batch_norm_expect_is_not_training
    with arg_scope(resnet_utils.resnet_arg_scope(is_training=False)):
      network_fn(inputs, num_classes, global_pool=global_pool, scope='resnet7')

    layers.batch_norm = batch_norm_expect_is_training
    with arg_scope(resnet_utils.resnet_arg_scope(is_training=False)):
      network_fn(
          inputs,
          num_classes,
          is_training=True,
          global_pool=global_pool,
          scope='resnet8')

    layers.batch_norm = batch_norm_expect_is_not_training
    with arg_scope(resnet_utils.resnet_arg_scope(is_training=False)):
      network_fn(
          inputs,
          num_classes,
          is_training=False,
          global_pool=global_pool,
          scope='resnet9')

    layers.batch_norm = batch_norm_fn

  def testDeprecatingIsTrainingResnetV1(self):
    self._testDeprecatingIsTraining(resnet_v1.resnet_v1_50)

  def testDeprecatingIsTrainingResnetV2(self):
    self._testDeprecatingIsTraining(resnet_v2.resnet_v2_50)


if __name__ == '__main__':
  test.main()
