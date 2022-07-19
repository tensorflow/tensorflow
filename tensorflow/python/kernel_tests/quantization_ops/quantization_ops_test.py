# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tf.quantize ops."""
import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import googletest


class FakeQuantWithMinMaxVarsOpTest(test_util.TensorFlowTestCase):

  @test_util.run_in_graph_and_eager_modes
  def test_invalid_inputs(self):
    inputs = constant_op.constant(
        value=[[1.0], [2.0], [4.0]], dtype=dtypes.float32)

    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                "must be rank 0"):
      self.evaluate(
          array_ops.fake_quant_with_min_max_vars(
              inputs=inputs, min=0.0, max=[[1.0], [2.0], [4.0]]))

    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                "must be rank 0"):
      self.evaluate(
          array_ops.fake_quant_with_min_max_vars(
              inputs=inputs, min=[[1.0], [2.0], [4.0]], max=1.0))


class FakeQuantWithMinMaxVarsPerChannelOpTest(test_util.TensorFlowTestCase):

  @test_util.run_in_graph_and_eager_modes
  def test_invalid_inputs(self):
    inputs = constant_op.constant(
        value=[[1.0], [2.0], [4.0]], dtype=dtypes.float32)

    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                "must be rank 1"):
      self.evaluate(
          array_ops.fake_quant_with_min_max_vars_per_channel(
              inputs=inputs, min=[[0.0]], max=[1.0]))

    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                "Dimensions must be equal|incorrect size"):
      self.evaluate(
          array_ops.fake_quant_with_min_max_vars_per_channel(
              inputs=inputs, min=[0.0, 0.1], max=[1.0]))

    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                "must be rank 1"):
      self.evaluate(
          array_ops.fake_quant_with_min_max_vars_per_channel(
              inputs=inputs, min=[1.0], max=[[1.0]]))

    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                "Dimensions must be equal|incorrect size"):
      self.evaluate(
          array_ops.fake_quant_with_min_max_vars_per_channel(
              inputs=inputs, min=[0.0], max=[1.0, 1.1]))


class QuantizedBiasedAddTest(test_util.TensorFlowTestCase):

  @test_util.run_in_graph_and_eager_modes
  def test_invalid_inputs(self):
    inputs = constant_op.constant(
        np.int8(0), shape=[3, 3, 3, 3], dtype=dtypes.qint8)
    bias = constant_op.constant(np.int8(0), shape=[3], dtype=dtypes.qint8)

    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                "must be rank 0"):
      self.evaluate(
          nn_ops.quantized_bias_add(
              input=inputs,
              bias=bias,
              min_input=[],
              max_input=1.0,
              min_bias=0.0,
              max_bias=1.0,
              out_type=dtypes.qint32))

    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                "must be rank 0"):
      self.evaluate(
          nn_ops.quantized_bias_add(
              input=inputs,
              bias=bias,
              min_input=0.0,
              max_input=[],
              min_bias=0.0,
              max_bias=1.0,
              out_type=dtypes.qint32))

    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                "must be rank 0"):
      self.evaluate(
          nn_ops.quantized_bias_add(
              input=inputs,
              bias=bias,
              min_input=0.0,
              max_input=1.0,
              min_bias=[],
              max_bias=1.0,
              out_type=dtypes.qint32))

    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                "must be rank 0"):
      self.evaluate(
          nn_ops.quantized_bias_add(
              input=inputs,
              bias=bias,
              min_input=0.0,
              max_input=1.0,
              min_bias=0.0,
              max_bias=[],
              out_type=dtypes.qint32))


class QuantizedInstanceNormOpTest(test_util.TensorFlowTestCase):

  @test_util.run_in_graph_and_eager_modes
  def test_invalid_inputs(self):
    inputs = constant_op.constant(
        np.uint8(0), shape=[3, 3, 3, 3], dtype=dtypes.quint8)

    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                "must be rank 0"):
      self.evaluate(
          array_ops.quantized_instance_norm(
              x=inputs, x_min=0.0, x_max=[[1.0], [2.0], [4.0]]))

    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                "must be rank 0"):
      self.evaluate(
          array_ops.quantized_instance_norm(
              x=inputs, x_min=[[1.0], [2.0], [4.0]], x_max=1.0))


class RequantizeOpTest(test_util.TensorFlowTestCase):

  @test_util.run_in_graph_and_eager_modes
  def test_invalid_inputs(self):
    inputs = constant_op.constant(
        np.int32(0), shape=[3, 3, 3, 3], dtype=dtypes.qint32)

    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                "must be rank 0"):
      self.evaluate(
          math_ops.requantize(
              input=inputs,
              input_min=[],
              input_max=1.0,
              requested_output_min=0.0,
              requested_output_max=1.0,
              out_type=dtypes.qint8))

    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                "must be rank 0"):
      self.evaluate(
          math_ops.requantize(
              input=inputs,
              input_min=0.0,
              input_max=[],
              requested_output_min=0.0,
              requested_output_max=1.0,
              out_type=dtypes.qint8))

    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                "must be rank 0"):
      self.evaluate(
          math_ops.requantize(
              input=inputs,
              input_min=0.0,
              input_max=1.0,
              requested_output_min=[],
              requested_output_max=1.0,
              out_type=dtypes.qint8))

    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                "must be rank 0"):
      self.evaluate(
          math_ops.requantize(
              input=inputs,
              input_min=0.0,
              input_max=1.0,
              requested_output_min=0.0,
              requested_output_max=[],
              out_type=dtypes.qint8))


class QuantizedAddOpTest(test_util.TensorFlowTestCase):

  @test_util.run_in_graph_and_eager_modes
  def test_invalid_inputs(self):
    x = constant_op.constant(
        np.int8(0), shape=[3, 3, 3, 3], dtype=dtypes.quint8)
    y = constant_op.constant(np.int8(0), shape=[3], dtype=dtypes.quint8)

    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                "must be rank 0"):
      self.evaluate(
          math_ops.quantized_add(
              x=x,
              y=y,
              min_x=[],
              max_x=1.0,
              min_y=0.0,
              max_y=1.0,
              Toutput=dtypes.qint32))


class QuantizedReluOpTest(test_util.TensorFlowTestCase):

  @test_util.run_in_graph_and_eager_modes
  def test_invalid_inputs(self):
    inputs = constant_op.constant(
        np.int8(0), shape=[3, 3, 3, 3], dtype=dtypes.quint8)

    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                "must be rank 0"):
      self.evaluate(
          nn_ops.quantized_relu(
              features=inputs,
              min_features=[],
              max_features=127.0,
              out_type=dtypes.quint8))


class QuantizedRelu6OpTest(test_util.TensorFlowTestCase):

  @test_util.run_in_graph_and_eager_modes
  def test_invalid_inputs(self):
    inputs = constant_op.constant(
        np.int8(0), shape=[3, 3, 3, 3], dtype=dtypes.quint8)

    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                "must be rank 0"):
      self.evaluate(
          nn_ops.quantized_relu6(
              features=inputs,
              min_features=[],
              max_features=127.0,
              out_type=dtypes.quint8))


if __name__ == "__main__":
  googletest.main()
