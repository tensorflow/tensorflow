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
"""Tests for keras.layers.preprocessing.normalization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

import numpy as np

from tensorflow.python import keras
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import context
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.layers.preprocessing import normalization
from tensorflow.python.keras.layers.preprocessing import normalization_v1
from tensorflow.python.keras.layers.preprocessing import preprocessing_test_utils
from tensorflow.python.keras.utils.generic_utils import CustomObjectScope
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
      "adapt_data": np.array([[1], [2], [3], [4], [5]], dtype=np.int32),
      "axis": -1,
      "test_data": np.array([[1], [2], [3]], np.int32),
      "expected": np.array([[-1.414214], [-.707107], [0]], np.float32),
      "testcase_name": "2d_int_data"
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


@keras_parameterized.run_all_keras_modes
class NormalizationTest(keras_parameterized.TestCase,
                        preprocessing_test_utils.PreprocessingLayerTest):

  def test_layer_api_compatibility(self):
    cls = get_layer_class()
    with CustomObjectScope({"Normalization": cls}):
      output_data = testing_utils.layer_test(
          cls,
          kwargs={"axis": -1},
          input_shape=(None, 3),
          input_data=np.array([[3, 1, 2], [6, 5, 4]], dtype=np.float32),
          validate_training=False,
          adapt_data=np.array([[1, 2, 1], [2, 3, 4], [1, 2, 1], [2, 3, 4]]))
    expected = np.array([[3., -3., -0.33333333], [9., 5., 1.]])
    self.assertAllClose(expected, output_data)

  def test_combiner_api_compatibility(self):
    data = np.array([[1], [2], [3], [4], [5]])
    combiner = normalization._NormalizingCombiner(axis=-1)
    expected = {
        "count": np.array(5.0),
        "variance": np.array([2.]),
        "mean": np.array([3.])
    }
    expected_accumulator = combiner._create_accumulator(expected["count"],
                                                        expected["mean"],
                                                        expected["variance"])
    self.validate_accumulator_serialize_and_deserialize(combiner, data,
                                                        expected_accumulator)
    self.validate_accumulator_uniqueness(combiner, data)
    self.validate_accumulator_extract(combiner, data, expected)
    self.validate_accumulator_extract_and_restore(combiner, data,
                                                  expected)
  @parameterized.named_parameters(
      {
          "data": np.array([[1], [2], [3], [4], [5]]),
          "axis": -1,
          "expected": {
              "count": np.array(5.0),
              "variance": np.array([2.]),
              "mean": np.array([3.])
          },
          "testcase_name": "2d_single_element"
      }, {
          "data": np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]),
          "axis": -1,
          "expected": {
              "count": np.array(5.0),
              "mean": np.array([3., 4.]),
              "variance": np.array([2., 2.])
          },
          "testcase_name": "2d_multi_element"
      }, {
          "data": np.array([[[1, 2]], [[2, 3]], [[3, 4]], [[4, 5]], [[5, 6]]]),
          "axis": 2,
          "expected": {
              "count": np.array(5.0),
              "mean": np.array([3., 4.]),
              "variance": np.array([2., 2.])
          },
          "testcase_name": "3d_multi_element"
      }, {
          "data": np.array([[[1, 2]], [[2, 3]], [[3, 4]], [[4, 5]], [[5, 6]]]),
          "axis": (1, 2),
          "expected": {
              "count": np.array(5.0),
              "mean": np.array([[3., 4.]]),
              "variance": np.array([[2., 2.]])
          },
          "testcase_name": "3d_multi_element_multi_axis"
      }, {
          "data":
              np.array([[[1, 2], [2, 3]], [[3, 4], [4, 5]], [[1, 2], [2, 3]],
                        [[3, 4], [4, 5]]]),
          "axis":
              1,
          "expected": {
              "count": np.array(8.0),
              "mean": np.array([2.5, 3.5]),
              "variance": np.array([1.25, 1.25])
          },
          "testcase_name":
              "3d_multi_element_internal_axis"
      })
  def test_combiner_computation_multi_value_axis(self, data, axis, expected):
    combiner = normalization._NormalizingCombiner(axis=axis)
    expected_accumulator = combiner._create_accumulator(**expected)
    self.validate_accumulator_computation(combiner, data, expected_accumulator)

  @parameterized.named_parameters(*_get_layer_computation_test_cases())
  def test_layer_computation(self, adapt_data, axis, test_data, use_dataset,
                             expected):
    input_shape = tuple([None for _ in range(test_data.ndim - 1)])
    if use_dataset:
      # Keras APIs expect batched datasets
      adapt_data = dataset_ops.Dataset.from_tensor_slices(adapt_data).batch(
          test_data.shape[0] // 2)
      test_data = dataset_ops.Dataset.from_tensor_slices(test_data).batch(
          test_data.shape[0] // 2)

    cls = get_layer_class()
    layer = cls(axis=axis)
    layer.adapt(adapt_data)

    input_data = keras.Input(shape=input_shape)
    output = layer(input_data)
    model = keras.Model(input_data, output)
    model._run_eagerly = testing_utils.should_run_eagerly()
    output_data = model.predict(test_data)
    self.assertAllClose(expected, output_data)

  def test_mean_setting_continued_adapt_failure(self):

    if not context.executing_eagerly():
      self.skipTest("'assign' doesn't work in V1, so don't test in V1.")

    cls = get_layer_class()
    layer = cls(axis=-1)
    layer.build((2,))
    layer.mean.assign([1.3, 2.0])
    with self.assertRaisesRegex(RuntimeError, "without also setting 'count'"):
      layer.adapt(np.array([[1, 2]]), reset_state=False)

  def test_var_setting_continued_adapt_failure(self):

    if not context.executing_eagerly():
      self.skipTest("'assign' doesn't work in V1, so don't test in V1.")

    cls = get_layer_class()
    layer = cls(axis=-1)
    layer.build((2,))
    layer.variance.assign([1.3, 2.0])
    with self.assertRaisesRegex(RuntimeError, "without also setting 'count'"):
      layer.adapt(np.array([[1, 2]]), reset_state=False)

  def test_weight_setting_continued_adapt_failure(self):
    cls = get_layer_class()
    layer = cls(axis=-1)
    layer.build((2,))
    layer.set_weights([np.array([1.3, 2.0]), np.array([0.0, 1.0]), np.array(0)])
    with self.assertRaisesRegex(RuntimeError, "without also setting 'count'"):
      layer.adapt(np.array([[1, 2]]), reset_state=False)

  def test_weight_setting_no_count_continued_adapt_failure(self):
    cls = get_layer_class()
    layer = cls(axis=-1)
    layer.build((2,))
    layer.set_weights([np.array([1.3, 2.0]), np.array([0.0, 1.0])])
    with self.assertRaisesRegex(RuntimeError, "without also setting 'count'"):
      layer.adapt(np.array([[1, 2]]), reset_state=False)


if __name__ == "__main__":
  test.main()
