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
"""Tests for Keras-based einsum dense layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

import numpy as np

from tensorflow.python import keras

from tensorflow.python.keras import keras_parameterized  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.layers import einsum_dense
from tensorflow.python.platform import test


@keras_parameterized.run_all_keras_modes
@parameterized.named_parameters(
    {
        "testcase_name": "_1d_end_weight",
        "equation": "ab,b->a",
        "bias_axes": None,
        "input_shape": (None, 32),
        "output_shape": [],
        "expected_weight_shape": [32],
        "expected_bias_shape": None,
        "expected_output_shape": (None,)
    }, {
        "testcase_name": "_2d_middle_weight",
        "equation": "ab,bc->ac",
        "bias_axes": None,
        "input_shape": (None, 32),
        "output_shape": (64),
        "expected_weight_shape": [32, 64],
        "expected_bias_shape": None,
        "expected_output_shape": (None, 64)
    }, {
        "testcase_name": "_3d_bert",
        "equation": "abc,cde->abde",
        "bias_axes": None,
        "input_shape": (None, 1, 2),
        "output_shape": (1, 3, 4),
        "expected_weight_shape": [2, 3, 4],
        "expected_bias_shape": None,
        "expected_output_shape": (None, 1, 3, 4)
    }, {
        "testcase_name": "_3d_3_bias",
        "equation": "abc,cde->abde",
        "bias_axes": "e",
        "input_shape": (None, 1, 2),
        "output_shape": (1, 3, 4),
        "expected_weight_shape": [2, 3, 4],
        "expected_bias_shape": [4],
        "expected_output_shape": (None, 1, 3, 4)
    }, {
        "testcase_name": "_3d_2_bias",
        "equation": "abc,cde->abde",
        "bias_axes": "d",
        "input_shape": (None, 1, 2),
        "output_shape": (1, 3, 4),
        "expected_weight_shape": [2, 3, 4],
        "expected_bias_shape": [3, 1],
        "expected_output_shape": (None, 1, 3, 4)
    }, {
        "testcase_name": "_3d_1_3_bias",
        "equation": "abc,cde->abde",
        "bias_axes": "be",
        "input_shape": (None, 7, 2),
        "output_shape": (7, 3, 4),
        "expected_weight_shape": [2, 3, 4],
        "expected_bias_shape": [7, 1, 4],
        "expected_output_shape": (None, 7, 3, 4)
    }, {
        "testcase_name": "_3d_bert_projection",
        "equation": "BFNH,NHD->BFD",
        "bias_axes": None,
        "input_shape": (None, 1, 2, 3),
        "output_shape": (1, 4),
        "expected_weight_shape": [2, 3, 4],
        "expected_bias_shape": None,
        "expected_output_shape": (None, 1, 4)
    }, {
        "testcase_name": "_2d_bert",
        "equation": "abc,cd->abd",
        "bias_axes": None,
        "input_shape": (None, 1, 2),
        "output_shape": (1, 4),
        "expected_weight_shape": [2, 4],
        "expected_bias_shape": None,
        "expected_output_shape": (None, 1, 4)
    }, {
        "testcase_name": "_embedding_1d",
        "equation": "i,d->id",
        "bias_axes": None,
        "input_shape": (None,),
        "output_shape": (2),
        "expected_weight_shape": [2],
        "expected_bias_shape": None,
        "expected_output_shape": (None, 2)
    }, {
        "testcase_name": "_xlnet_lm",
        "equation": "ibd,nd->ibn",
        "bias_axes": None,
        "input_shape": (None, None, 1),
        "output_shape": (None, 2),
        "expected_weight_shape": [2, 1],
        "expected_bias_shape": None,
        "expected_output_shape": (None, None, 2)
    }, {
        "testcase_name": "_2d_precast",
        "equation": "...b,bc->...c",
        "bias_axes": None,
        "input_shape": (None, 32),
        "output_shape": (64),
        "expected_weight_shape": [32, 64],
        "expected_bias_shape": None,
        "expected_output_shape": (None, 64)
    }, {
        "testcase_name": "_2d_precast_multiple_elided_dims",
        "equation": "...b,bc->...c",
        "bias_axes": None,
        "input_shape": (None, None, 32),
        "output_shape": (64),
        "expected_weight_shape": [32, 64],
        "expected_bias_shape": None,
        "expected_output_shape": (None, None, 64)
    }, {
        "testcase_name": "_3d_precast",
        "equation": "...c,cde->...de",
        "bias_axes": None,
        "input_shape": (None, 1, 2),
        "output_shape": (3, 4),
        "expected_weight_shape": [2, 3, 4],
        "expected_bias_shape": None,
        "expected_output_shape": (None, 1, 3, 4)
    }, {
        "testcase_name": "_3d_precast_3_bias",
        "equation": "...c,cde->...de",
        "bias_axes": "e",
        "input_shape": (None, 1, 2),
        "output_shape": (3, 4),
        "expected_weight_shape": [2, 3, 4],
        "expected_bias_shape": [4],
        "expected_output_shape": (None, 1, 3, 4)
    }, {
        "testcase_name": "_3d_precast_2_bias",
        "equation": "...c,cde->...de",
        "bias_axes": "d",
        "input_shape": (None, 1, 2),
        "output_shape": (3, 4),
        "expected_weight_shape": [2, 3, 4],
        "expected_bias_shape": [3, 1],
        "expected_output_shape": (None, 1, 3, 4)
    }, {
        "testcase_name": "_3d_precast_2_3_bias",
        "equation": "...c,cde->...de",
        "bias_axes": "de",
        "input_shape": (None, 1, 2),
        "output_shape": (3, 4),
        "expected_weight_shape": [2, 3, 4],
        "expected_bias_shape": [3, 4],
        "expected_output_shape": (None, 1, 3, 4)
    }, {
        "testcase_name": "_2d_postcast",
        "equation": "bc...,cd->bd...",
        "bias_axes": None,
        "input_shape": (None, 1, 2, 3),
        "output_shape": (4),
        "expected_weight_shape": [1, 4],
        "expected_bias_shape": None,
        "expected_output_shape": (None, 4, 2, 3)
    }, {
        "testcase_name": "_3d_postcast",
        "equation": "bc...,cde->bde...",
        "bias_axes": None,
        "input_shape": (None, 1, 2),
        "output_shape": (3, 4),
        "expected_weight_shape": [1, 3, 4],
        "expected_bias_shape": None,
        "expected_output_shape": (None, 3, 4, 2)
    }, {
        "testcase_name": "_3d_postcast_1_bias",
        "equation": "bc...,cde->bde...",
        "bias_axes": "d",
        "input_shape": (None, 1, 2),
        "output_shape": (3, 4),
        "expected_weight_shape": [1, 3, 4],
        "expected_bias_shape": [3, 1, 1],
        "expected_output_shape": (None, 3, 4, 2)
    }, {
        "testcase_name": "_3d_postcast_2_bias",
        "equation": "bc...,cde->bde...",
        "bias_axes": "e",
        "input_shape": (None, 1, 2),
        "output_shape": (3, 4),
        "expected_weight_shape": [1, 3, 4],
        "expected_bias_shape": [4, 1],
        "expected_output_shape": (None, 3, 4, 2)
    }, {
        "testcase_name": "_3d_postcast_1_2_bias",
        "equation": "bc...,cde->bde...",
        "bias_axes": "de",
        "input_shape": (None, 1, 2),
        "output_shape": (3, 4),
        "expected_weight_shape": [1, 3, 4],
        "expected_bias_shape": [3, 4, 1],
        "expected_output_shape": (None, 3, 4, 2)
    })
class TestEinsumDenseLayer(keras_parameterized.TestCase):

  def test_weight_shapes(self, equation, bias_axes, input_shape, output_shape,
                         expected_weight_shape, expected_bias_shape,
                         expected_output_shape):
    del expected_output_shape  # Not used in this test.

    weight_shape, bias_shape, _ = einsum_dense._analyze_einsum_string(
        equation, bias_axes, input_shape, output_shape)

    self.assertAllEqual(expected_weight_shape, weight_shape)
    self.assertAllEqual(expected_bias_shape, bias_shape)

  def test_layer_creation(self, equation, bias_axes, input_shape, output_shape,
                          expected_weight_shape, expected_bias_shape,
                          expected_output_shape):
    # Keras elides the 0-dimension of the input shape when constructing inputs.
    non_batch_input_shape = list(input_shape)[1:]

    input_tensor = keras.Input(shape=non_batch_input_shape)
    layer = einsum_dense.EinsumDense(
        equation=equation, output_shape=output_shape, bias_axes=bias_axes)
    output_tensor = layer(input_tensor)

    self.assertAllEqual(expected_weight_shape, layer.kernel.shape.as_list())
    if expected_bias_shape is None:
      self.assertIsNone(layer.bias)
    else:
      self.assertAllEqual(expected_bias_shape, layer.bias.shape.as_list())
    self.assertAllEqual(expected_output_shape, output_tensor.shape.as_list())


@keras_parameterized.run_all_keras_modes
class TestEinsumLayerAPI(keras_parameterized.TestCase):

  def test_layer_api(self):
    input_data = np.array([[1.0, 2.0], [3.0, 4.0]])
    kwargs = {
        "equation": "...b,bc->...c",
        "bias_axes": "c",
        "output_shape": 4,
        "bias_initializer": keras.initializers.constant(0.03),
        "kernel_initializer": keras.initializers.constant(0.5),
        "dtype": input_data.dtype
    }
    expected_output = np.array([[1.53, 1.53, 1.53, 1.53],
                                [3.53, 3.53, 3.53, 3.53]])

    output_data = testing_utils.layer_test(
        einsum_dense.EinsumDense,
        kwargs=kwargs,
        input_shape=(None, 2),
        input_data=input_data)

    self.assertAllClose(expected_output, output_data)

  def test_unspecified_bias_dim_fails(self):
    input_tensor = keras.Input(shape=(32,))
    layer = einsum_dense.EinsumDense(
        equation="ab,bc->ac", output_shape=64, bias_axes="y")
    with self.assertRaisesRegex(
        ValueError, ".*is not a part of the output specification.*"):
      _ = layer(input_tensor)

  def test_incompatible_input_output_shape_fails(self):
    input_tensor = keras.Input(shape=(32, 64))
    layer = einsum_dense.EinsumDense(
        equation="abc,cd->abd", output_shape=(10, 96))
    with self.assertRaisesRegex(
        ValueError, ".*Input shape and output shape do not match at shared "
        "dimension 'b'.*"):
      _ = layer(input_tensor)

  def test_unspecified_output_dim_fails(self):
    input_tensor = keras.Input(shape=(32,))
    layer = einsum_dense.EinsumDense(equation="ab,bc->cd", output_shape=64)
    with self.assertRaisesRegex(
        ValueError, ".*Dimension 'd' was specified in the output 'cd' but has "
        "no corresponding dim.*"):
      _ = layer(input_tensor)

  def test_unspecified_weight_dim_fails(self):
    input_tensor = keras.Input(shape=(32,))
    layer = einsum_dense.EinsumDense(equation="ab,zd->ad", output_shape=64)
    with self.assertRaisesRegex(ValueError,
                                ".*Weight dimension 'z' did not have a match "):
      _ = layer(input_tensor)


if __name__ == "__main__":
  test.main()
