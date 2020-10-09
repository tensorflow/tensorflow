# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for keras.layers.preprocessing.reduction."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensorflow.python import keras

from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras.layers.preprocessing import reduction
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import test


@keras_parameterized.run_all_keras_modes
class ReductionTest(keras_parameterized.TestCase):

  @parameterized.named_parameters(
      {
          "testcase_name": "max",
          "reduction_str": "max",
          "expected_output": [[3.0, 3.0], [3.0, 2.0]]
      }, {
          "testcase_name": "mean",
          "reduction_str": "mean",
          "expected_output": [[2.0, 2.0], [2.0, 1.5]]
      }, {
          "testcase_name": "min",
          "reduction_str": "min",
          "expected_output": [[1.0, 1.0], [1.0, 1.0]]
      }, {
          "testcase_name": "prod",
          "reduction_str": "prod",
          "expected_output": [[6.0, 6.0], [3.0, 2.0]]
      }, {
          "testcase_name": "sum",
          "reduction_str": "sum",
          "expected_output": [[6.0, 6.0], [4.0, 3.0]]
      })
  def test_unweighted_ragged_reduction(self, reduction_str, expected_output):
    data = ragged_factory_ops.constant([[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
                                        [[3.0, 1.0], [1.0, 2.0]]])
    input_tensor = keras.Input(shape=(None, None), ragged=True)

    output_tensor = reduction.Reduction(reduction=reduction_str)(input_tensor)
    model = keras.Model(input_tensor, output_tensor)

    output = model.predict(data)

    self.assertAllClose(expected_output, output)

  @parameterized.named_parameters(
      {
          "testcase_name": "max",
          "reduction_str": "max",
          "expected_output": [[4.0, 4.0], [1.5, 6.0]]
      }, {
          "testcase_name": "mean",
          "reduction_str": "mean",
          "expected_output": [[2.0, 2.0], [1.666667, 1.75]]
      }, {
          "testcase_name": "min",
          "reduction_str": "min",
          "expected_output": [[1.0, 1.0], [1.0, 1.0]]
      }, {
          "testcase_name": "prod",
          "reduction_str": "prod",
          "expected_output": [[12.0, 12.0], [1.5, 6.0]]
      }, {
          "testcase_name": "sum",
          "reduction_str": "sum",
          "expected_output": [[8.0, 8.0], [2.5, 7.0]]
      }, {
          "testcase_name": "sqrtn",
          "reduction_str": "sqrtn",
          "expected_output": [[3.265986, 3.265986], [2.236067, 2.213594]]
      })
  def test_weighted_ragged_reduction(self, reduction_str, expected_output):
    data = ragged_factory_ops.constant([[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
                                        [[3.0, 1.0], [1.0, 2.0]]])
    input_tensor = keras.Input(shape=(None, None), ragged=True)

    weights = ragged_factory_ops.constant([[[1.0, 1.0], [2.0, 2.0], [1.0, 1.0]],
                                           [[0.5, 1.0], [1.0, 3.0]]])
    weight_input_tensor = keras.Input(shape=(None, None), ragged=True)

    output_tensor = reduction.Reduction(reduction=reduction_str)(
        input_tensor, weights=weight_input_tensor)
    model = keras.Model([input_tensor, weight_input_tensor], output_tensor)

    output = model.predict([data, weights])
    self.assertAllClose(expected_output, output)

  def test_weighted_ragged_reduction_with_different_dimensionality(self):
    data = ragged_factory_ops.constant([[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
                                        [[3.0, 1.0], [1.0, 2.0]]])
    input_tensor = keras.Input(shape=(None, None), ragged=True)

    weights = ragged_factory_ops.constant([[1.0, 2.0, 1.0], [1.0, 1.0]])
    weight_input_tensor = keras.Input(shape=(None,), ragged=True)

    output_tensor = reduction.Reduction(reduction="mean")(
        input_tensor, weights=weight_input_tensor)
    model = keras.Model([input_tensor, weight_input_tensor], output_tensor)

    output = model.predict([data, weights])
    expected_output = [[2.0, 2.0], [2.0, 1.5]]
    self.assertAllClose(expected_output, output)

  @parameterized.named_parameters(
      {
          "testcase_name": "max",
          "reduction_str": "max",
          "expected_output": [[3.0, 3.0], [3.0, 2.0]]
      }, {
          "testcase_name": "mean",
          "reduction_str": "mean",
          "expected_output": [[2.0, 2.0], [1.333333, 1.0]]
      }, {
          "testcase_name": "min",
          "reduction_str": "min",
          "expected_output": [[1.0, 1.0], [0.0, 0.0]]
      }, {
          "testcase_name": "prod",
          "reduction_str": "prod",
          "expected_output": [[6.0, 6.0], [0.0, 0.0]]
      }, {
          "testcase_name": "sum",
          "reduction_str": "sum",
          "expected_output": [[6.0, 6.0], [4.0, 3.0]]
      })
  def test_unweighted_dense_reduction(self, reduction_str, expected_output):
    data = np.array([[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
                     [[3.0, 1.0], [1.0, 2.0], [0.0, 0.0]]])
    input_tensor = keras.Input(shape=(None, None))

    output_tensor = reduction.Reduction(reduction=reduction_str)(input_tensor)
    model = keras.Model(input_tensor, output_tensor)

    output = model.predict(data)

    self.assertAllClose(expected_output, output)

  @parameterized.named_parameters(
      {
          "testcase_name": "max",
          "reduction_str": "max",
          "expected_output": [[4.0, 4.0], [1.5, 6.0]]
      }, {
          "testcase_name": "mean",
          "reduction_str": "mean",
          "expected_output": [[2.0, 2.0], [1.666667, 1.75]]
      }, {
          "testcase_name": "min",
          "reduction_str": "min",
          "expected_output": [[1.0, 1.0], [0.0, 0.0]]
      }, {
          "testcase_name": "prod",
          "reduction_str": "prod",
          "expected_output": [[12.0, 12.0], [0.0, 0.0]]
      }, {
          "testcase_name": "sum",
          "reduction_str": "sum",
          "expected_output": [[8.0, 8.0], [2.5, 7.0]]
      }, {
          "testcase_name": "sqrtn",
          "reduction_str": "sqrtn",
          "expected_output": [[3.265986, 3.265986], [2.236067, 2.213594]]
      })
  def test_weighted_dense_reduction(self, reduction_str, expected_output):
    data = np.array([[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
                     [[3.0, 1.0], [1.0, 2.0], [0.0, 0.0]]])
    input_tensor = keras.Input(shape=(None, None))

    weights = np.array([[[1.0, 1.0], [2.0, 2.0], [1.0, 1.0]],
                        [[0.5, 1.0], [1.0, 3.0], [0.0, 0.0]]])
    weight_input_tensor = keras.Input(shape=(None, None))

    output_tensor = reduction.Reduction(reduction=reduction_str)(
        input_tensor, weights=weight_input_tensor)
    model = keras.Model([input_tensor, weight_input_tensor], output_tensor)

    output = model.predict([data, weights])

    self.assertAllClose(expected_output, output)

  def test_weighted_dense_reduction_with_different_dimensionality(self):
    data = np.array([[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
                     [[3.0, 1.0], [1.0, 2.0], [0.0, 0.0]]])
    input_tensor = keras.Input(shape=(None, None))

    weights = np.array([[1.0, 2.0, 1.0], [1.0, 1.0, 0.0]])
    weight_input_tensor = keras.Input(shape=(None,))

    output_tensor = reduction.Reduction(reduction="mean")(
        input_tensor, weights=weight_input_tensor)
    model = keras.Model([input_tensor, weight_input_tensor], output_tensor)

    output = model.predict([data, weights])
    expected_output = [[2.0, 2.0], [2.0, 1.5]]
    self.assertAllClose(expected_output, output)

  def test_sqrtn_fails_on_unweighted_ragged(self):
    input_tensor = keras.Input(shape=(None, None), ragged=True)
    with self.assertRaisesRegex(ValueError, ".*sqrtn.*"):
      _ = reduction.Reduction(reduction="sqrtn")(input_tensor)

  def test_sqrtn_fails_on_unweighted_dense(self):
    input_tensor = keras.Input(shape=(None, None))
    with self.assertRaisesRegex(ValueError, ".*sqrtn.*"):
      _ = reduction.Reduction(reduction="sqrtn")(input_tensor)

if __name__ == "__main__":
  test.main()
