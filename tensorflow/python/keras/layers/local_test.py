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
"""Tests for locally-connected layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python import keras
from tensorflow.python.framework import test_util as tf_test_util
from tensorflow.python.keras import testing_utils
from tensorflow.python.platform import test


class LocallyConnectedLayersTest(test.TestCase):

  @tf_test_util.run_in_graph_and_eager_modes()
  def test_locallyconnected_1d(self):
    num_samples = 2
    num_steps = 8
    input_dim = 5
    filter_length = 3
    filters = 4

    for padding in ['valid']:
      for strides in [1]:
        if padding == 'same' and strides != 1:
          continue
        for data_format in ['channels_first', 'channels_last']:
          testing_utils.layer_test(
              keras.layers.LocallyConnected1D,
              kwargs={
                  'filters': filters,
                  'kernel_size': filter_length,
                  'padding': padding,
                  'strides': strides,
                  'data_format': data_format
              },
              input_shape=(num_samples, num_steps, input_dim))

  def test_locallyconnected_1d_regularization(self):
    num_samples = 2
    num_steps = 8
    input_dim = 5
    filter_length = 3
    filters = 4
    for data_format in ['channels_first', 'channels_last']:
      kwargs = {
          'filters': filters,
          'kernel_size': filter_length,
          'kernel_regularizer': 'l2',
          'bias_regularizer': 'l2',
          'activity_regularizer': 'l2',
          'data_format': data_format
      }

      with self.test_session():
        layer = keras.layers.LocallyConnected1D(**kwargs)
        layer.build((num_samples, num_steps, input_dim))
        self.assertEqual(len(layer.losses), 2)
        layer(
            keras.backend.variable(np.ones((num_samples,
                                            num_steps,
                                            input_dim))))
        self.assertEqual(len(layer.losses), 3)

      k_constraint = keras.constraints.max_norm(0.01)
      b_constraint = keras.constraints.max_norm(0.01)
      kwargs = {
          'filters': filters,
          'kernel_size': filter_length,
          'kernel_constraint': k_constraint,
          'bias_constraint': b_constraint,
      }
      with self.test_session():
        layer = keras.layers.LocallyConnected1D(**kwargs)
        layer.build((num_samples, num_steps, input_dim))
        self.assertEqual(layer.kernel.constraint, k_constraint)
        self.assertEqual(layer.bias.constraint, b_constraint)

  @tf_test_util.run_in_graph_and_eager_modes()
  def test_locallyconnected_2d(self):
    num_samples = 8
    filters = 3
    stack_size = 4
    num_row = 6
    num_col = 10

    for padding in ['valid']:
      for strides in [(1, 1), (2, 2)]:
        if padding == 'same' and strides != (1, 1):
          continue

        testing_utils.layer_test(
            keras.layers.LocallyConnected2D,
            kwargs={
                'filters': filters,
                'kernel_size': 3,
                'padding': padding,
                'kernel_regularizer': 'l2',
                'bias_regularizer': 'l2',
                'strides': strides,
                'data_format': 'channels_last'
            },
            input_shape=(num_samples, num_row, num_col, stack_size))

  @tf_test_util.run_in_graph_and_eager_modes()
  def test_locallyconnected_2d_channels_first(self):
    num_samples = 8
    filters = 3
    stack_size = 4
    num_row = 6
    num_col = 10

    testing_utils.layer_test(
        keras.layers.LocallyConnected2D,
        kwargs={
            'filters': filters,
            'kernel_size': 3,
            'data_format': 'channels_first'
        },
        input_shape=(num_samples, num_row, num_col, stack_size))

  def test_locallyconnected_2d_regularization(self):
    num_samples = 8
    filters = 3
    stack_size = 4
    num_row = 6
    num_col = 10
    kwargs = {
        'filters': filters,
        'kernel_size': 3,
        'kernel_regularizer': 'l2',
        'bias_regularizer': 'l2',
        'activity_regularizer': 'l2',
    }
    with self.test_session():
      layer = keras.layers.LocallyConnected2D(**kwargs)
      layer.build((num_samples, num_row, num_col, stack_size))
      self.assertEqual(len(layer.losses), 2)
      layer(
          keras.backend.variable(
              np.ones((num_samples, num_row, num_col, stack_size))))
      self.assertEqual(len(layer.losses), 3)

    k_constraint = keras.constraints.max_norm(0.01)
    b_constraint = keras.constraints.max_norm(0.01)
    kwargs = {
        'filters': filters,
        'kernel_size': 3,
        'kernel_constraint': k_constraint,
        'bias_constraint': b_constraint,
    }
    with self.test_session():
      layer = keras.layers.LocallyConnected2D(**kwargs)
      layer.build((num_samples, num_row, num_col, stack_size))
      self.assertEqual(layer.kernel.constraint, k_constraint)
      self.assertEqual(layer.bias.constraint, b_constraint)


if __name__ == '__main__':
  test.main()
