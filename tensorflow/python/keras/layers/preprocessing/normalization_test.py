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
from tensorflow.python.framework import constant_op
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.layers.preprocessing import normalization
from tensorflow.python.keras.layers.preprocessing import normalization_v1
from tensorflow.python.keras.layers.preprocessing import preprocessing_test_utils
from tensorflow.python.keras.utils.generic_utils import CustomObjectScope
from tensorflow.python.ops import variables
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
  }, {
      "adapt_data":
          np.zeros((3, 4)),
      "axis": -1,
      "test_data":
          np.zeros((3, 4)),
      "expected":
          np.zeros((3, 4)),
      "testcase_name":
          "zero_variance"
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

  @parameterized.named_parameters(*_get_layer_computation_test_cases())
  def test_layer_computation(self, adapt_data, axis, test_data, use_dataset,
                             expected):
    input_shape = tuple([test_data.shape[i] for i in range(1, test_data.ndim)])
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

    weights = layer.get_weights()
    mean = weights[0]
    var = weights[1]

    direct_set_layer = cls(axis=axis, mean=mean, variance=var)
    input_data = keras.Input(shape=input_shape)
    output = direct_set_layer(input_data)
    model = keras.Model(input_data, output)
    model._run_eagerly = testing_utils.should_run_eagerly()
    output_data = model.predict(test_data)
    self.assertAllClose(expected, output_data)

  def test_broadcasting_during_direct_setting(self):
    cls = get_layer_class()
    layer = cls(axis=-1, mean=[1.0], variance=[2.0])
    layer.build((None, 2))
    weights = layer.get_weights()
    self.assertAllClose([1.0, 1.0], weights[0])
    self.assertAllClose([2.0, 2.0], weights[1])

  def test_broadcasting_during_direct_setting_with_tensors(self):
    cls = get_layer_class()
    layer = cls(
        axis=-1,
        mean=constant_op.constant([1.0]),
        variance=constant_op.constant([2.0]))
    layer.build((None, 2))
    weights = layer.get_weights()
    self.assertAllClose([1.0, 1.0], weights[0])
    self.assertAllClose([2.0, 2.0], weights[1])

  def test_broadcasting_during_direct_setting_with_variables_fails(self):
    cls = get_layer_class()
    with self.assertRaisesRegex(ValueError, "passing a Variable"):
      _ = cls(
          axis=-1,
          mean=variables.Variable([1.0]),
          variance=variables.Variable([2.0]))

  def test_1d_data(self):
    data = [0, 2, 0, 2]
    cls = get_layer_class()
    layer = cls(axis=-1)
    layer.adapt(data)
    output = layer(data)
    self.assertListEqual(output.shape.as_list(), [4, 1])
    if context.executing_eagerly():
      self.assertAllClose(output.numpy(), [[-1], [1], [-1], [1]])

  def test_0d_data(self):
    if not context.executing_eagerly():
      self.skipTest("Only supported in TF2.")

    data = [0, 2, 0, 2]
    cls = get_layer_class()
    layer = cls(axis=-1)
    layer.adapt(data)
    output = layer(0.)
    self.assertListEqual(output.shape.as_list(), [1, 1])
    self.assertAllClose(output.numpy(), [[-1]])

  @parameterized.parameters(
      {"axis": 0},
      {"axis": (-1, 0)},
  )
  def test_zeros_fail_init(self, axis):
    cls = get_layer_class()
    with self.assertRaisesRegex(ValueError,
                                "The argument 'axis' may not be 0."):
      cls(axis=axis)

  @parameterized.parameters(
      # Out of bounds
      {"axis": 3},
      {"axis": -3},
      # In a tuple
      {"axis": (1, 3)},
      {"axis": (1, -3)},
  )
  def test_bad_axis_fail_build(self, axis):
    cls = get_layer_class()
    layer = cls(axis=axis)
    with self.assertRaisesRegex(ValueError, r"in the range"):
      layer.build([None, 2, 3])

  @parameterized.parameters(
      # Results should be identical no matter how the axes are specified (3d).
      {"axis": (1, 2)},
      {"axis": (2, 1)},
      {"axis": (1, -1)},
      {"axis": (-1, 1)},
  )
  def test_axis_permutations(self, axis):
    cls = get_layer_class()
    layer = cls(axis=axis)
    # data.shape = [2, 2, 3]
    data = np.array([[[0., 1., 2.], [0., 2., 6.]],
                     [[2., 3., 4.], [3., 6., 10.]]])
    expect = np.array([[[-1., -1., -1.], [-1., -1., -1.]],
                       [[1., 1., 1.], [1., 1., 1.]]])
    layer.adapt(data)
    self.assertAllClose(expect, layer(data))

  def test_model_summary_after_layer_adapt(self):
    data = np.array([[[0., 1., 2.], [0., 2., 6.]],
                     [[2., 3., 4.], [3., 6., 10.]]])
    cls = get_layer_class()
    layer = cls(axis=-1)
    layer.adapt(data)
    model = keras.Sequential(
        [layer,
         keras.layers.Dense(64, activation="relu"),
         keras.layers.Dense(1)])
    model.summary()

  def test_merge_state(self):
    if not context.executing_eagerly():
      self.skipTest("`merge_state` only supported in TF2")

    cls = get_layer_class()
    data = np.random.rand(30, 10, 2)
    ds = dataset_ops.Dataset.from_tensor_slices(data).batch(2)
    norm = cls(axis=(1, 2))
    norm.adapt(ds)

    partial_ds_1 = ds.shard(3, 0)
    partial_ds_2 = ds.shard(3, 1)
    partial_ds_3 = ds.shard(3, 2)

    norm_1 = cls(axis=(1, 2))
    norm_2 = cls(axis=(1, 2))
    norm_3 = cls(axis=(1, 2))

    norm_1.adapt(partial_ds_1)
    norm_2.adapt(partial_ds_2)
    norm_3.adapt(partial_ds_3)

    norm_1.merge_state([norm_2, norm_3])
    merged_norm = norm_1

    self.assertAllClose(norm(data), merged_norm(data))


if __name__ == "__main__":
  test.main()
