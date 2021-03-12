# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Distribution tests for keras.layers.preprocessing.normalization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python import keras
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import combinations as ds_combinations
from tensorflow.python.eager import context
from tensorflow.python.framework import test_combinations as combinations
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras.distribute.strategy_combinations import all_strategies
from tensorflow.python.keras.layers.preprocessing import normalization
from tensorflow.python.keras.layers.preprocessing import normalization_v1
from tensorflow.python.keras.layers.preprocessing import preprocessing_test_utils
from tensorflow.python.platform import test


def get_layer_class():
  if context.executing_eagerly():
    return normalization.Normalization
  else:
    return normalization_v1.Normalization


def _get_layer_computation_test_cases():
  test_cases = ({
      "adapt_data": np.array([[1.], [2.], [3.], [4.], [5.]], dtype=np.float32),
      "axis": -1,
      "test_data": np.array([[1.], [2.], [3.]], np.float32),
      "expected": np.array([[-1.414214], [-.707107], [0]], np.float32),
      "testcase_name": "2d_single_element"
  }, {
      "adapt_data": np.array([[1.], [2.], [3.], [4.], [5.]], dtype=np.float32),
      "axis": None,
      "test_data": np.array([[1.], [2.], [3.]], np.float32),
      "expected": np.array([[-1.414214], [-.707107], [0]], np.float32),
      "testcase_name": "2d_single_element_none_axis"
  }, {
      "adapt_data": np.array([[1., 2., 3., 4., 5.]], dtype=np.float32),
      "axis": None,
      "test_data": np.array([[1.], [2.], [3.]], np.float32),
      "expected": np.array([[-1.414214], [-.707107], [0]], np.float32),
      "testcase_name": "2d_single_element_none_axis_flat_data"
  }, {
      "adapt_data":
          np.array([[[1., 2., 3.], [2., 3., 4.]], [[3., 4., 5.], [4., 5., 6.]]],
                   np.float32),
      "axis":
          1,
      "test_data":
          np.array([[[1., 2., 3.], [2., 3., 4.]], [[3., 4., 5.], [4., 5., 6.]]],
                   np.float32),
      "expected":
          np.array([[[-1.549193, -0.774597, 0.], [-1.549193, -0.774597, 0.]],
                    [[0., 0.774597, 1.549193], [0., 0.774597, 1.549193]]],
                   np.float32),
      "testcase_name":
          "3d_internal_axis"
  }, {
      "adapt_data":
          np.array(
              [[[1., 0., 3.], [2., 3., 4.]], [[3., -1., 5.], [4., 5., 8.]]],
              np.float32),
      "axis": (1, 2),
      "test_data":
          np.array(
              [[[3., 1., -1.], [2., 5., 4.]], [[3., 0., 5.], [2., 5., 8.]]],
              np.float32),
      "expected":
          np.array(
              [[[1., 3., -5.], [-1., 1., -1.]], [[1., 1., 1.], [-1., 1., 1.]]],
              np.float32),
      "testcase_name":
          "3d_multiple_axis"
  })

  crossed_test_cases = []
  # Cross above test cases with use_dataset in (True, False)
  for use_dataset in (True, False):
    for case in test_cases:
      case = case.copy()
      if use_dataset:
        case["testcase_name"] = case["testcase_name"] + "_with_dataset"
      case["use_dataset"] = use_dataset
      crossed_test_cases.append(case)

  return crossed_test_cases


@ds_combinations.generate(
    combinations.times(
        combinations.combine(
            distribution=all_strategies,
            mode=["eager", "graph"]), _get_layer_computation_test_cases()))
class NormalizationTest(keras_parameterized.TestCase,
                        preprocessing_test_utils.PreprocessingLayerTest):

  def test_layer_computation(self, distribution, adapt_data, axis, test_data,
                             use_dataset, expected):
    input_shape = tuple([None for _ in range(test_data.ndim - 1)])
    if use_dataset:
      # Keras APIs expect batched datasets
      adapt_data = dataset_ops.Dataset.from_tensor_slices(adapt_data).batch(
          test_data.shape[0] // 2)
      test_data = dataset_ops.Dataset.from_tensor_slices(test_data).batch(
          test_data.shape[0] // 2)

    with distribution.scope():
      input_data = keras.Input(shape=input_shape)
      layer = get_layer_class()(axis=axis)
      layer.adapt(adapt_data)
      output = layer(input_data)
      model = keras.Model(input_data, output)
      output_data = model.predict(test_data)
    self.assertAllClose(expected, output_data)


if __name__ == "__main__":
  test.main()
