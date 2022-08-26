# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for quantize_model."""
import itertools
import os
from typing import List, Mapping, Optional
import warnings

from absl.testing import parameterized
import numpy as np
import tensorflow  # pylint: disable=unused-import

from tensorflow.compiler.mlir.quantization.tensorflow import quantization_options_pb2 as quant_opts_pb2
from tensorflow.compiler.mlir.quantization.tensorflow.python import quantize_model
from tensorflow.compiler.mlir.quantization.tensorflow.python import representative_dataset as repr_dataset
from tensorflow.compiler.mlir.quantization.tensorflow.python.integration_test import quantize_model_test_base
from tensorflow.python.client import session
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model import loader_impl as saved_model_loader
from tensorflow.python.saved_model import save as saved_model_save
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils_impl
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.types import core

# Type aliases for quantization method protobuf enums.
_Method = quant_opts_pb2.QuantizationMethod.Method
_ExperimentalMethod = quant_opts_pb2.QuantizationMethod.ExperimentalMethod


def parameter_combinations(test_parameters):
  """Generate all combinations of test parameters."""
  real_parameters = []
  for parameters in test_parameters:
    keys = parameters.keys()
    for curr in itertools.product(*parameters.values()):
      real_parameters.append(dict(zip(keys, curr)))
  return real_parameters


class MultipleSignatureModel(module.Module):
  """A model with 2 signatures.

  Used to test where the quantizer has to handle multiple signatures.
  """

  @def_function.function(input_signature=[
      tensor_spec.TensorSpec(shape=[1, 4], dtype=dtypes.float32)
  ])
  def matmul(self, matmul_input: core.Tensor) -> Mapping[str, core.Tensor]:
    """Performs a matrix multiplication.

    Args:
      matmul_input: Input tensor to matmul with the filter.

    Returns:
      A map of: output key -> output result.
    """
    filters = random_ops.random_uniform(shape=(4, 3), minval=-1.0, maxval=1.0)
    out = math_ops.matmul(matmul_input, filters)

    return {'output': out}

  @def_function.function(input_signature=[
      tensor_spec.TensorSpec(shape=(1, 3, 4, 3), dtype=dtypes.float32)
  ])
  def conv(self, conv_input: core.Tensor) -> Mapping[str, core.Tensor]:
    """Performs a 2D convolution operation.

    Args:
      conv_input: Input tensor to perform convolution on.

    Returns:
      A map of: output key -> output result.
    """
    filters = np.random.uniform(
        low=-10, high=10, size=(2, 3, 3, 2)).astype('f4')
    out = nn_ops.conv2d(
        conv_input,
        filters,
        strides=[1, 1, 2, 1],
        dilations=[1, 1, 1, 1],
        padding='SAME',
        data_format='NHWC')

    return {'output': out}


@test_util.run_all_in_graph_and_eager_modes
class QuantizationMethodTest(quantize_model_test_base.QuantizedModelTest):
  """Test cases regarding the use of QuantizationMethod proto.

  Run all tests cases in both the graph mode (default in TF1) and the eager mode
  (default in TF2) to ensure support for when TF2 is disabled.
  """

  class SimpleModel(module.Module):

    @def_function.function(input_signature=[
        tensor_spec.TensorSpec(shape=[1, 4], dtype=dtypes.float32)
    ])
    def __call__(self, input_tensor: core.Tensor) -> Mapping[str, core.Tensor]:
      """Performs a matrix multiplication.

      Args:
        input_tensor: Input tensor to matmul with the filter.

      Returns:
        A map of: output key -> output result.
      """
      filters = np.random.uniform(low=-1.0, high=1.0, size=(4, 3)).astype('f4')

      out = math_ops.matmul(input_tensor, filters)
      return {'output': out}

  def _simple_model_data_gen(self) -> repr_dataset.RepresentativeDataset:
    """Creates an interable of representative samples.

    Yields:
      Representative samples, which is basically a mapping of: input key ->
      input value.
    """
    for _ in range(8):
      yield {
          'input_tensor':
              ops.convert_to_tensor(
                  np.random.uniform(low=0, high=150, size=(1, 4)).astype('f4')),
      }

  def test_static_range_quantization_by_default(self):
    model = self.SimpleModel()

    input_saved_model_path = self.create_tempdir('input').full_path
    saved_model_save.save(model, input_saved_model_path)

    # Use default QuantizationOptions.
    converted_model = quantize_model.quantize(
        input_saved_model_path,
        representative_dataset=self._simple_model_data_gen())

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(converted_model.signatures._signatures.keys(),
                          {'serving_default'})

    # Indirectly prove that it is performing a static-range quantization
    # by checking that it complains about representative_dataset when it is
    # not provided.
    with self.assertRaisesRegex(ValueError, 'representative_dataset'):
      quantize_model.quantize(input_saved_model_path)

  def test_method_unspecified_raises_value_error(self):
    model = self.SimpleModel()

    input_saved_model_path = self.create_tempdir('input').full_path
    saved_model_save.save(model, input_saved_model_path)

    options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            method=_Method.METHOD_UNSPECIFIED))

    with self.assertRaises(ValueError):
      quantize_model.quantize(
          input_saved_model_path, quantization_options=options)

  def test_invalid_method_raises_value_error(self):
    model = self.SimpleModel()

    input_saved_model_path = self.create_tempdir('input').full_path
    saved_model_save.save(model, input_saved_model_path)

    # Set an invalid value of -1 to QuantizationMethod.method.
    options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(method=-1))

    with self.assertRaises(ValueError):
      quantize_model.quantize(
          input_saved_model_path, quantization_options=options)


class StaticRangeQuantizationTest(quantize_model_test_base.QuantizedModelTest):

  def _any_warning_contains(
      self, substring: str,
      warnings_list: List[warnings.WarningMessage]) -> bool:
    """Returns True if any of the warnings contains a given substring.

    Args:
      substring: A piece of string to check whether it exists in the warning
        message.
      warnings_list: A list of `WarningMessage`s.

    Returns:
      True if and only if the substring exists in any of the warnings in
      `warnings_list`.
    """
    return any(
        map(lambda warning: substring in str(warning.message), warnings_list))

  @parameterized.parameters(
      parameter_combinations([{
          'activation_fn': [None, nn_ops.relu, nn_ops.relu6],
          'has_bias': [True, False],
          'target_opset': [quant_opts_pb2.XLA],
      }]))
  @test_util.run_in_graph_and_eager_modes
  def test_qat_conv_model(self, activation_fn: Optional[ops.Operation],
                          has_bias: bool, target_opset: quant_opts_pb2.OpSet):

    class ConvModel(module.Module):

      def __init__(self):
        self.filter_value = np.random.uniform(
            low=-0.5, high=0.5, size=(2, 3, 3, 2)).astype('f4')

      @def_function.function(input_signature=[
          tensor_spec.TensorSpec(
              name='input', shape=[1, 3, 4, 3], dtype=dtypes.float32),
      ])
      def conv(self, input_tensor: core.Tensor) -> Mapping[str, core.Tensor]:
        """Performs a 2D convolution operation.

        Args:
          input_tensor: Input tensor to perform convolution on.

        Returns:
          A map of: output key -> output result.
        """
        q_input = array_ops.fake_quant_with_min_max_args(
            input_tensor, min=-0.1, max=0.2, num_bits=8, narrow_range=False)
        filter_tensor = ops.convert_to_tensor(self.filter_value)
        bias = array_ops.constant([0.1, 0.2], dtype=dtypes.float32)
        out = nn_ops.conv2d(
            q_input,
            filter_tensor,
            strides=[1, 1, 2, 1],
            dilations=[1, 1, 1, 1],
            padding='SAME',
            data_format='NHWC')
        if has_bias:
          out = nn_ops.bias_add(out, bias, data_format='NHWC')
        if activation_fn is not None:
          out = activation_fn(out)
        q_out = array_ops.fake_quant_with_min_max_args(
            out, min=-0.3, max=0.4, num_bits=8, narrow_range=False)
        return {'output': q_out}

    model = ConvModel()
    input_saved_model_path = self.create_tempdir('input').full_path
    saved_model_save.save(model, input_saved_model_path)

    signature_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    tags = [tag_constants.SERVING]

    # Check the converted model with TF opset as the baseline.
    output_directory = self.create_tempdir().full_path
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            experimental_method=_ExperimentalMethod.STATIC_RANGE),
        op_set=quant_opts_pb2.TF)

    converted_model = quantize_model.quantize(input_saved_model_path,
                                              [signature_key], tags,
                                              output_directory,
                                              quantization_options)
    self.assertIsNotNone(converted_model)
    self.assertCountEqual(converted_model.signatures._signatures.keys(),
                          {signature_key})

    input_data = np.random.uniform(
        low=-0.1, high=0.2, size=(1, 3, 4, 3)).astype('f4')
    expected_outputs = model.conv(input_data)
    got_outputs = converted_model.signatures[signature_key](
        input=ops.convert_to_tensor(input_data))
    # TODO(b/215633216): Check if the accuracy is acceptable.
    self.assertAllClose(expected_outputs, got_outputs, atol=0.01)

    output_loader = saved_model_loader.SavedModelLoader(output_directory)
    output_meta_graphdef = output_loader.get_meta_graph_def_from_tags(tags)
    self.assertTrue(
        self._contains_quantized_function_call(output_meta_graphdef))

    # Check the converted model in the target opset.
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            experimental_method=_ExperimentalMethod.STATIC_RANGE),
        op_set=target_opset)

    output_directory = self.create_tempdir().full_path
    converted_model = quantize_model.quantize(input_saved_model_path,
                                              [signature_key], tags,
                                              output_directory,
                                              quantization_options)

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(converted_model.signatures._signatures.keys(),
                          {signature_key})
    loader = saved_model_loader.SavedModelLoader(output_directory)
    meta_graphdef = loader.get_meta_graph_def_from_tags(tags)
    if target_opset == quant_opts_pb2.XLA:
      self.assertTrue(self._contains_op(meta_graphdef, 'XlaConvV2'))

    new_outputs = converted_model.signatures[signature_key](
        input=ops.convert_to_tensor(input_data))
    # The difference between TF and XLA path is expected to be small (smaller
    # or equal to 1 in the quantized domain).
    self.assertAllClose(new_outputs, got_outputs, atol=0.00275)

  # Run this test only with the eager mode.
  @test_util.run_v2_only
  def test_ptq_model_with_variable(self):

    class ConvModelWithVariable(module.Module):
      """A simple model that performs a single convolution to the input tensor.

      It keeps the filter as a tf.Variable.
      """

      def __init__(self) -> None:
        """Initializes the filter variable."""
        self.filters = variables.Variable(
            random_ops.random_uniform(
                shape=(2, 3, 3, 2), minval=-1., maxval=1.))

      @def_function.function(input_signature=[
          tensor_spec.TensorSpec(
              name='input', shape=(1, 3, 4, 3), dtype=dtypes.float32),
      ])
      def __call__(self, x: core.Tensor) -> Mapping[str, core.Tensor]:
        """Performs a 2D convolution operation.

        Args:
          x: Input tensor to perform convolution on.

        Returns:
          A map of: output key -> output result.
        """
        out = nn_ops.conv2d(
            x,
            self.filters,
            strides=[1, 1, 2, 1],
            dilations=[1, 1, 1, 1],
            padding='SAME',
            data_format='NHWC')
        return {'output': out}

    def gen_data() -> repr_dataset.RepresentativeDataset:
      """Creates an interable of representative samples.

      Yields:
        Representative samples, which is basically a mapping of: input key ->
        input value.
      """
      for _ in range(8):
        yield {
            'input':
                random_ops.random_uniform(
                    shape=(1, 3, 4, 3), minval=0, maxval=150)
        }

    model = ConvModelWithVariable()
    input_saved_model_path = self.create_tempdir('input').full_path
    saved_model_save.save(model, input_saved_model_path)

    signature_keys = [signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    tags = {tag_constants.SERVING}
    output_directory = self.create_tempdir().full_path

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            experimental_method=_ExperimentalMethod.STATIC_RANGE))

    converted_model = quantize_model.quantize(
        input_saved_model_path,
        signature_keys,
        tags,
        output_directory,
        quantization_options,
        representative_dataset=gen_data())

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(converted_model.signatures._signatures.keys(),
                          signature_keys)

    output_loader = saved_model_loader.SavedModelLoader(output_directory)
    output_meta_graphdef = output_loader.get_meta_graph_def_from_tags(tags)
    self.assertTrue(
        self._contains_quantized_function_call(output_meta_graphdef))

  @parameterized.named_parameters(
      ('none', None, False, False, quant_opts_pb2.TF, False),
      ('relu', nn_ops.relu, False, False, quant_opts_pb2.TF, False),
      ('relu6', nn_ops.relu6, False, False, quant_opts_pb2.TF, False),
      ('bn', None, False, True, quant_opts_pb2.TF, False),
      ('bn_and_relu', nn_ops.relu, False, True, quant_opts_pb2.TF, False),
      ('with_bias', None, True, False, quant_opts_pb2.TF, False),
      ('with_bias_and_bn', None, True, True, quant_opts_pb2.TF, False),
      ('with_bias_and_bn_and_relu', nn_ops.relu, True, True, quant_opts_pb2.TF,
       False),
      ('with_bias_and_relu', nn_ops.relu, True, False, quant_opts_pb2.TF,
       False),
      ('with_bias_and_relu6', nn_ops.relu6, True, False, quant_opts_pb2.TF,
       False),
      ('with_bias_and_bn_to_xla', None, True, True, quant_opts_pb2.XLA, False),
      ('with_bias_and_relu6_to_xla', nn_ops.relu6, True, False,
       quant_opts_pb2.XLA, False),
      ('with_bias_and_bn_to_xla_dynamic', None, True, True, quant_opts_pb2.XLA,
       True),
      ('with_bias_and_relu6_to_xla_dynamic', nn_ops.relu6, True, False,
       quant_opts_pb2.XLA, True),
  )
  @test_util.run_in_graph_and_eager_modes
  def test_conv_ptq_model(self, activation_fn: Optional[ops.Operation],
                          has_bias: bool, has_bn: bool,
                          target_opset: quant_opts_pb2.OpSet,
                          input_shape_dynamic: bool):
    input_shape = [None, None, None, 3] if input_shape_dynamic else [1, 3, 4, 3]

    class ConvModel(module.Module):

      @def_function.function(input_signature=[
          tensor_spec.TensorSpec(shape=input_shape, dtype=dtypes.float32)
      ])
      def conv(self, input_tensor: core.Tensor) -> Mapping[str, core.Tensor]:
        """Performs a 2D convolution operation.

        Args:
          input_tensor: Input tensor to perform convolution on.

        Returns:
          A map of: output key -> output result.
        """
        filters = np.random.uniform(
            low=-10, high=10, size=(2, 3, 3, 2)).astype('f4')
        bias = np.random.uniform(low=0, high=10, size=(2)).astype('f4')
        scale, offset = [1.0, 1.0], [0.5, 0.5]
        mean, variance = scale, offset
        out = nn_ops.conv2d(
            input_tensor,
            filters,
            strides=[1, 1, 2, 1],
            dilations=[1, 1, 1, 1],
            padding='SAME',
            data_format='NHWC')
        if has_bias:
          out = nn_ops.bias_add(out, bias)
        if has_bn:
          # Fusing is supported for non-training case.
          out, _, _, _, _, _ = nn_ops.fused_batch_norm_v3(
              out, scale, offset, mean, variance, is_training=False)
        if activation_fn is not None:
          out = activation_fn(out)
        return {'output': out}

    model = ConvModel()
    input_saved_model_path = self.create_tempdir('input').full_path
    saved_model_save.save(model, input_saved_model_path)

    def data_gen() -> repr_dataset.RepresentativeDataset:
      for _ in range(8):
        yield {
            'input_tensor':
                ops.convert_to_tensor(
                    np.random.uniform(low=0, high=150,
                                      size=(1, 3, 4, 3)).astype('f4')),
        }

    tags = [tag_constants.SERVING]
    output_directory = self.create_tempdir().full_path

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            experimental_method=_ExperimentalMethod.STATIC_RANGE),
        op_set=target_opset)

    converted_model = quantize_model.quantize(
        input_saved_model_path, ['serving_default'],
        tags,
        output_directory,
        quantization_options,
        representative_dataset=data_gen())
    self.assertIsNotNone(converted_model)
    self.assertCountEqual(converted_model.signatures._signatures.keys(),
                          {'serving_default'})

    output_loader = saved_model_loader.SavedModelLoader(output_directory)
    output_meta_graphdef = output_loader.get_meta_graph_def_from_tags(tags)
    if target_opset == quant_opts_pb2.XLA:
      self.assertTrue(self._contains_op(output_meta_graphdef, 'XlaConvV2'))
    else:
      self.assertTrue(
          self._contains_quantized_function_call(output_meta_graphdef))
    self.assertFalse(
        self._contains_op(output_meta_graphdef, 'FusedBatchNormV3'))

  @parameterized.named_parameters(
      ('none', None, False, False, quant_opts_pb2.TF, False),
      ('relu', nn_ops.relu, False, False, quant_opts_pb2.TF, False),
      ('relu6', nn_ops.relu6, False, False, quant_opts_pb2.TF, False),
      ('bn', None, False, True, quant_opts_pb2.TF, False),
      ('bn_and_relu', nn_ops.relu, False, True, quant_opts_pb2.TF, False),
      ('with_bias', None, True, False, quant_opts_pb2.TF, False),
      ('with_bias_and_bn', None, True, True, quant_opts_pb2.TF, False),
      ('with_bias_and_bn_and_relu', nn_ops.relu, True, True, quant_opts_pb2.TF,
       False),
      ('with_bias_and_relu', nn_ops.relu, True, False, quant_opts_pb2.TF,
       False),
      ('with_bias_and_relu6', nn_ops.relu6, True, False, quant_opts_pb2.TF,
       False),
      ('with_bias_and_bn_to_xla', None, True, True, quant_opts_pb2.XLA, False),
      ('with_bias_and_relu6_to_xla', nn_ops.relu6, True, False,
       quant_opts_pb2.XLA, False),
      ('with_bias_and_bn_to_xla_dynamic', None, True, True, quant_opts_pb2.XLA,
       True),
      ('with_bias_and_relu6_to_xla_dynamic', nn_ops.relu6, True, False,
       quant_opts_pb2.XLA, True),
  )
  @test_util.run_in_graph_and_eager_modes
  def test_depthwise_conv_ptq_model(self,
                                    activation_fn: Optional[ops.Operation],
                                    has_bias: bool, has_bn: bool,
                                    target_opset: quant_opts_pb2.OpSet,
                                    input_shape_dynamic: bool):
    input_shape = [None, None, None, 3] if input_shape_dynamic else [1, 3, 4, 3]

    class DepthwiseConvModel(module.Module):

      @def_function.function(input_signature=[
          tensor_spec.TensorSpec(shape=input_shape, dtype=dtypes.float32)
      ])
      def conv(self, input_tensor: core.Tensor) -> Mapping[str, core.Tensor]:
        """Performs a 2D convolution operation.

        Args:
          input_tensor: Input tensor to perform convolution on.

        Returns:
          A map of: output key -> output result.
        """
        filters = np.random.uniform(
            low=-10, high=10, size=(2, 3, 3, 1)).astype('f4')
        bias = np.random.uniform(low=0, high=10, size=(3)).astype('f4')
        scale, offset = [1.0, 1.0, 1.0], [0.5, 0.5, 0.5]
        mean, variance = scale, offset
        out = nn_ops.depthwise_conv2d_native(
            input_tensor,
            filters,
            strides=[1, 2, 2, 1],
            dilations=[1, 1, 1, 1],
            padding='SAME',
            data_format='NHWC')
        if has_bias:
          out = nn_ops.bias_add(out, bias)
        if has_bn:
          # Fusing is supported for non-training case.
          out, _, _, _, _, _ = nn_ops.fused_batch_norm_v3(
              out, scale, offset, mean, variance, is_training=False)
        if activation_fn is not None:
          out = activation_fn(out)
        return {'output': out}

    model = DepthwiseConvModel()
    input_saved_model_path = self.create_tempdir('input').full_path
    saved_model_save.save(model, input_saved_model_path)

    def data_gen() -> repr_dataset.RepresentativeDataset:
      for _ in range(8):
        yield {
            'input_tensor':
                ops.convert_to_tensor(
                    np.random.uniform(low=0, high=150,
                                      size=(1, 3, 4, 3)).astype('f4')),
        }

    tags = [tag_constants.SERVING]
    output_directory = self.create_tempdir().full_path

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            experimental_method=_ExperimentalMethod.STATIC_RANGE),
        op_set=target_opset)

    converted_model = quantize_model.quantize(
        input_saved_model_path, ['serving_default'],
        tags,
        output_directory,
        quantization_options,
        representative_dataset=data_gen())
    self.assertIsNotNone(converted_model)
    self.assertCountEqual(converted_model.signatures._signatures.keys(),
                          {'serving_default'})

    output_loader = saved_model_loader.SavedModelLoader(output_directory)
    output_meta_graphdef = output_loader.get_meta_graph_def_from_tags(tags)
    if target_opset == quant_opts_pb2.XLA:
      self.assertTrue(self._contains_op(output_meta_graphdef, 'XlaConvV2'))
    else:
      self.assertTrue(
          self._contains_quantized_function_call(output_meta_graphdef))
    self.assertFalse(
        self._contains_op(output_meta_graphdef, 'FusedBatchNormV3'))

  @parameterized.named_parameters(
      ('none', None, False),
      ('relu', nn_ops.relu, False),
      ('relu6', nn_ops.relu6, False),
      ('with_bias', None, True),
      ('with_bias_and_relu', nn_ops.relu, True),
      ('with_bias_and_relu6', nn_ops.relu6, True),
  )
  @test_util.run_in_graph_and_eager_modes
  def test_matmul_ptq_model(self, activation_fn: Optional[ops.Operation],
                            has_bias: bool):
    model = self._create_matmul_model(has_bias, activation_fn)
    input_saved_model_path = self.create_tempdir('input').full_path
    saved_model_save.save(model, input_saved_model_path)

    def data_gen() -> repr_dataset.RepresentativeDataset:
      for _ in range(8):
        yield {
            'input_tensor':
                ops.convert_to_tensor(
                    np.random.uniform(low=0, high=5,
                                      size=(1, 1024)).astype('f4')),
        }

    tags = [tag_constants.SERVING]
    output_directory = self.create_tempdir().full_path

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            experimental_method=_ExperimentalMethod.STATIC_RANGE))

    converted_model = quantize_model.quantize(
        input_saved_model_path, ['serving_default'],
        tags,
        output_directory,
        quantization_options,
        representative_dataset=data_gen())
    self.assertIsNotNone(converted_model)
    self.assertCountEqual(converted_model.signatures._signatures.keys(),
                          {'serving_default'})

    output_loader = saved_model_loader.SavedModelLoader(output_directory)
    output_meta_graphdef = output_loader.get_meta_graph_def_from_tags(tags)
    self.assertTrue(
        self._contains_quantized_function_call(output_meta_graphdef))

  @parameterized.named_parameters(
      ('use_constant', False),
      ('use_variable', True),
  )
  @test_util.run_v2_only
  def test_gather_model(self, use_variable):

    model = self._create_gather_model(use_variable)

    input_saved_model_path = self.create_tempdir('input').full_path
    saved_model_save.save(model, input_saved_model_path)

    tags = [tag_constants.SERVING]
    output_directory = self.create_tempdir().full_path

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            experimental_method=_ExperimentalMethod.STATIC_RANGE))

    data_gen = self._create_data_generator(
        input_key='input_tensor',
        shape=[6],
        minval=0,
        maxval=10,
        dtype=dtypes.int64)

    converted_model = quantize_model.quantize(
        input_saved_model_path, ['serving_default'],
        tags,
        output_directory,
        quantization_options,
        representative_dataset=data_gen)

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(converted_model.signatures._signatures.keys(),
                          {'serving_default'})

    output_loader = saved_model_loader.SavedModelLoader(output_directory)
    output_meta_graphdef = output_loader.get_meta_graph_def_from_tags(tags)
    # Currently gather is not supported.
    self.assertFalse(
        self._contains_quantized_function_call(output_meta_graphdef))

  @test_util.run_in_graph_and_eager_modes
  def test_model_ptq_use_representative_samples_list(self):
    model = self._create_matmul_model()
    input_savedmodel_dir = self.create_tempdir('input').full_path
    saved_model_save.save(model, input_savedmodel_dir)

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            experimental_method=_ExperimentalMethod.STATIC_RANGE))
    output_savedmodel_dir = self.create_tempdir().full_path
    tags = {tag_constants.SERVING}

    representative_dataset: repr_dataset.RepresentativeDataset = [{
        'input_tensor': random_ops.random_uniform(shape=(1, 1024)),
    } for _ in range(8)]

    converted_model = quantize_model.quantize(
        input_savedmodel_dir, ['serving_default'],
        output_directory=output_savedmodel_dir,
        quantization_options=quantization_options,
        representative_dataset=representative_dataset)

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(converted_model.signatures._signatures.keys(),
                          {'serving_default'})
    output_loader = saved_model_loader.SavedModelLoader(output_savedmodel_dir)
    output_meta_graphdef = output_loader.get_meta_graph_def_from_tags(tags)
    self.assertTrue(
        self._contains_quantized_function_call(output_meta_graphdef))

  @test_util.run_in_graph_and_eager_modes
  def test_model_ptq_use_ndarray_representative_dataset(self):
    model = self._create_matmul_model()
    input_savedmodel_dir = self.create_tempdir('input').full_path
    saved_model_save.save(model, input_savedmodel_dir)

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            experimental_method=_ExperimentalMethod.STATIC_RANGE))
    output_savedmodel_dir = self.create_tempdir().full_path
    tags = {tag_constants.SERVING}

    # Use np.ndarrays instead of tf.Tensors for the representative dataset.
    representative_dataset = [{
        'input_tensor': np.random.uniform(size=(1, 1024)).astype(np.float32),
    } for _ in range(4)]

    converted_model = quantize_model.quantize(
        input_savedmodel_dir, ['serving_default'],
        tags=tags,
        output_directory=output_savedmodel_dir,
        quantization_options=quantization_options,
        representative_dataset=representative_dataset)

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(converted_model.signatures._signatures.keys(),
                          {'serving_default'})
    output_loader = saved_model_loader.SavedModelLoader(output_savedmodel_dir)
    output_meta_graphdef = output_loader.get_meta_graph_def_from_tags(tags)
    self.assertTrue(
        self._contains_quantized_function_call(output_meta_graphdef))

  @test_util.run_in_graph_and_eager_modes
  def test_model_ptq_use_python_list_representative_dataset(self):
    model = self._create_matmul_model()
    input_savedmodel_dir = self.create_tempdir('input').full_path
    saved_model_save.save(model, input_savedmodel_dir)

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            experimental_method=_ExperimentalMethod.STATIC_RANGE))
    output_savedmodel_dir = self.create_tempdir().full_path
    tags = {tag_constants.SERVING}

    # Use plain python lists as representative samples.
    representative_dataset = [{
        'input_tensor': [[i * 0.1 for i in range(1024)]],
    } for _ in range(4)]

    converted_model = quantize_model.quantize(
        input_savedmodel_dir, ['serving_default'],
        tags=tags,
        output_directory=output_savedmodel_dir,
        quantization_options=quantization_options,
        representative_dataset=representative_dataset)

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(converted_model.signatures._signatures.keys(),
                          {'serving_default'})
    output_loader = saved_model_loader.SavedModelLoader(output_savedmodel_dir)
    output_meta_graphdef = output_loader.get_meta_graph_def_from_tags(tags)
    self.assertTrue(
        self._contains_quantized_function_call(output_meta_graphdef))

  @test_util.run_in_graph_and_eager_modes
  def test_model_ptq_call_twice(self):
    model = self._create_matmul_model()
    input_savedmodel_dir = self.create_tempdir('input').full_path
    saved_model_save.save(model, input_savedmodel_dir)

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            experimental_method=_ExperimentalMethod.STATIC_RANGE))
    output_savedmodel_dir_1 = self.create_tempdir().full_path
    tags = {tag_constants.SERVING}
    signature_def_keys = [signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

    representative_dataset: repr_dataset.RepresentativeDataset = [{
        'input_tensor': random_ops.random_uniform(shape=(1, 1024)),
    } for _ in range(8)]

    # Test the first run.
    converted_model_1 = quantize_model.quantize(
        input_savedmodel_dir,
        signature_def_keys,
        output_directory=output_savedmodel_dir_1,
        quantization_options=quantization_options,
        representative_dataset=representative_dataset)

    self.assertIsNotNone(converted_model_1)
    self.assertCountEqual(converted_model_1.signatures._signatures.keys(),
                          signature_def_keys)
    output_loader = saved_model_loader.SavedModelLoader(output_savedmodel_dir_1)
    output_meta_graphdef = output_loader.get_meta_graph_def_from_tags(tags)
    self.assertTrue(
        self._contains_quantized_function_call(output_meta_graphdef))

    # Test the second run on the same model.
    output_savedmodel_dir_2 = self.create_tempdir().full_path
    converted_model_2 = quantize_model.quantize(
        input_savedmodel_dir,
        signature_def_keys,
        output_directory=output_savedmodel_dir_2,
        quantization_options=quantization_options,
        representative_dataset=representative_dataset)

    self.assertIsNotNone(converted_model_2)
    self.assertCountEqual(converted_model_2.signatures._signatures.keys(),
                          signature_def_keys)
    output_loader = saved_model_loader.SavedModelLoader(output_savedmodel_dir_2)
    output_meta_graphdef = output_loader.get_meta_graph_def_from_tags(tags)
    self.assertTrue(
        self._contains_quantized_function_call(output_meta_graphdef))

  # tf.data.Dataset is as an Iterable (thus can be used as representative
  # dataset) only in TF2 (eager mode).
  @test_util.run_v2_only
  def test_model_ptq_use_tf_dataset_for_representative_dataset(self):
    model = self._create_matmul_model()
    input_savedmodel_dir = self.create_tempdir('input').full_path
    saved_model_save.save(model, input_savedmodel_dir)

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            experimental_method=_ExperimentalMethod.STATIC_RANGE))
    output_savedmodel_dir = self.create_tempdir().full_path
    tags = {tag_constants.SERVING}

    representative_samples = [{
        'input_tensor': random_ops.random_uniform(shape=(1, 1024)),
    } for _ in range(8)]

    # Construct a tf.data.Dataset from the representative samples.
    representative_dataset = dataset_ops.DatasetV2.from_generator(
        lambda: representative_samples,
        output_signature={
            'input_tensor':
                tensor_spec.TensorSpec(shape=(1, 1024), dtype=dtypes.float32),
        })

    converted_model = quantize_model.quantize(
        input_savedmodel_dir, ['serving_default'],
        output_directory=output_savedmodel_dir,
        quantization_options=quantization_options,
        representative_dataset=representative_dataset)

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(converted_model.signatures._signatures.keys(),
                          {'serving_default'})
    output_loader = saved_model_loader.SavedModelLoader(output_savedmodel_dir)
    output_meta_graphdef = output_loader.get_meta_graph_def_from_tags(tags)
    self.assertTrue(
        self._contains_quantized_function_call(output_meta_graphdef))

  @test_util.run_in_graph_and_eager_modes
  def test_model_ptq_no_representative_sample_shows_warnings(self):
    model = self._create_matmul_model()
    input_savedmodel_dir = self.create_tempdir('input').full_path
    output_savedmodel_dir = self.create_tempdir().full_path
    saved_model_save.save(model, input_savedmodel_dir)

    tags = [tag_constants.SERVING]
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            experimental_method=_ExperimentalMethod.STATIC_RANGE))

    with warnings.catch_warnings(record=True) as warnings_list:
      converted_model = quantize_model.quantize(
          input_savedmodel_dir,
          ['serving_default'],
          tags,
          output_savedmodel_dir,
          quantization_options,
          # Put no sample into the representative dataset to make calibration
          # impossible.
          representative_dataset=[])

      self.assertNotEmpty(warnings_list)

      # Warning message should contain the function name.
      self.assertTrue(self._any_warning_contains('matmul', warnings_list))
      self.assertTrue(
          self._any_warning_contains('does not have min or max values',
                                     warnings_list))

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(converted_model.signatures._signatures.keys(),
                          {'serving_default'})
    output_loader = saved_model_loader.SavedModelLoader(output_savedmodel_dir)
    output_meta_graphdef = output_loader.get_meta_graph_def_from_tags(tags)
    # Model is not quantized because there was no sample data for calibration.
    self.assertFalse(
        self._contains_quantized_function_call(output_meta_graphdef))

  @test_util.run_in_graph_and_eager_modes
  def test_model_ptq_with_uncalibrated_subgraph(self):

    class IfModel(module.Module):
      """A model that contains a branching op."""

      @def_function.function(input_signature=[
          tensor_spec.TensorSpec(shape=[1, 4], dtype=dtypes.float32)
      ])
      def model_fn(self, x: core.Tensor) -> Mapping[str, core.Tensor]:
        """Runs the input tensor to a branched operations.

        The graph is branched by a condition whether the sum of elements of `x`
        is greater than 10.

        Args:
          x: Input tensor.

        Returns:
          A map of: output key -> output result.
        """
        if math_ops.reduce_sum(x) > 10.0:
          filters = np.random.uniform(
              low=-1.0, high=1.0, size=(4, 3)).astype('f4')
          bias = np.random.uniform(low=-1.0, high=1.0, size=(3,)).astype('f4')
          out = math_ops.matmul(x, filters)
          out = nn_ops.bias_add(out, bias)
          return {'output': out}

        filters = np.random.uniform(
            low=-1.0, high=1.0, size=(4, 3)).astype('f4')
        bias = np.random.uniform(low=-1.0, high=1.0, size=(3,)).astype('f4')
        out = math_ops.matmul(x, filters)
        out = nn_ops.bias_add(out, bias)
        return {'output': out}

    model = IfModel()
    input_saved_model_path = self.create_tempdir('input').full_path
    saved_model_save.save(model, input_saved_model_path)

    def data_gen() -> repr_dataset.RepresentativeDataset:
      for _ in range(8):
        yield {
            'x':
                ops.convert_to_tensor(
                    np.random.uniform(low=0.0, high=1.0,
                                      size=(1, 4)).astype('f4')),
        }

    tags = [tag_constants.SERVING]
    output_directory = self.create_tempdir().full_path

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            experimental_method=_ExperimentalMethod.STATIC_RANGE))

    with warnings.catch_warnings(record=True) as warnings_list:
      converted_model = quantize_model.quantize(
          input_saved_model_path, ['serving_default'],
          tags,
          output_directory,
          quantization_options,
          representative_dataset=data_gen())

      self.assertNotEmpty(warnings_list)

      # Warning message should contain the function name. The uncalibrated path
      # is when the condition is true, so 'cond_true' function must be part of
      # the warning message.
      self.assertTrue(self._any_warning_contains('cond_true', warnings_list))
      self.assertFalse(self._any_warning_contains('cond_false', warnings_list))
      self.assertTrue(
          self._any_warning_contains('does not have min or max values',
                                     warnings_list))

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(converted_model.signatures._signatures.keys(),
                          {'serving_default'})
    output_loader = saved_model_loader.SavedModelLoader(output_directory)
    output_meta_graphdef = output_loader.get_meta_graph_def_from_tags(tags)
    self.assertTrue(
        self._contains_quantized_function_call(output_meta_graphdef))

  # Run this test only with the eager mode.
  @test_util.run_v2_only
  def test_ptq_model_with_multiple_signatures(self):
    # Create and save a model having 2 signatures.
    model = MultipleSignatureModel()

    signatures = {
        'sig1':
            model.matmul.get_concrete_function(
                tensor_spec.TensorSpec(shape=(1, 4), dtype=dtypes.float32)),
        'sig2':
            model.conv.get_concrete_function(
                tensor_spec.TensorSpec(
                    shape=(1, 3, 4, 3), dtype=dtypes.float32)),
    }
    input_saved_model_path = self.create_tempdir('input').full_path
    saved_model_save.save(model, input_saved_model_path, signatures=signatures)

    output_directory = self.create_tempdir().full_path
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            experimental_method=_ExperimentalMethod.STATIC_RANGE))

    def data_gen_sig1() -> repr_dataset.RepresentativeDataset:
      """Generates tuple-style samples for signature 'sig1'.

      The first element of the tuple identifies the signature key the input data
      is for.

      Yields:
        Representative sample for 'sig1'.
      """
      for _ in range(4):
        yield {'matmul_input': random_ops.random_uniform(shape=(1, 4))}

    def data_gen_sig2() -> repr_dataset.RepresentativeDataset:
      """Generates tuple-style samples for signature 'sig2'.

      The first element of the tuple identifies the signature key the input data
      is for.

      Yields:
        Representative sample for 'sig2'.
      """
      for _ in range(4):
        yield {'conv_input': random_ops.random_uniform(shape=(1, 3, 4, 3))}

    tags = {tag_constants.SERVING}
    converted_model = quantize_model.quantize(
        input_saved_model_path,
        signature_keys=['sig1', 'sig2'],
        tags=tags,
        output_directory=output_directory,
        quantization_options=quantization_options,
        representative_dataset={
            'sig1': data_gen_sig1(),
            'sig2': data_gen_sig2(),
        })
    self.assertIsNotNone(converted_model)
    self.assertCountEqual(converted_model.signatures._signatures.keys(),
                          {'sig1', 'sig2'})

    output_loader = saved_model_loader.SavedModelLoader(output_directory)
    output_meta_graphdef = output_loader.get_meta_graph_def_from_tags(tags)
    self.assertTrue(
        self._contains_quantized_function_call(output_meta_graphdef))

  # Run this test only with the eager mode.
  @test_util.run_v2_only
  def test_ptq_multiple_signatures_invalid_dataset_raises_value_error(self):
    # Create and save a model having 2 signatures.
    model = MultipleSignatureModel()

    signatures = {
        'sig1':
            model.matmul.get_concrete_function(
                tensor_spec.TensorSpec(shape=(1, 4), dtype=dtypes.float32)),
        'sig2':
            model.conv.get_concrete_function(
                tensor_spec.TensorSpec(
                    shape=(1, 3, 4, 3), dtype=dtypes.float32)),
    }
    input_saved_model_path = self.create_tempdir('input').full_path
    saved_model_save.save(model, input_saved_model_path, signatures=signatures)

    output_directory = self.create_tempdir().full_path
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            experimental_method=_ExperimentalMethod.STATIC_RANGE))

    # Use a dict-style samples instead of tuple-style samples. This is invalid
    # because for a model multiple signatures one must use tuple-style samples.
    invalid_dataset: repr_dataset.RepresentativeDataset = [{
        'matmul_input': random_ops.random_uniform(shape=(1, 4))
    } for _ in range(8)]

    with self.assertRaisesRegex(ValueError, 'Invalid representative dataset.'):
      quantize_model.quantize(
          input_saved_model_path,
          signature_keys=['sig1', 'sig2'],
          tags={tag_constants.SERVING},
          output_directory=output_directory,
          quantization_options=quantization_options,
          representative_dataset=invalid_dataset)

  @test_util.run_in_graph_and_eager_modes
  def test_ptq_model_with_tf1_saved_model_with_variable_for_conv2d(self):
    input_saved_model_path = self.create_tempdir('input').full_path
    signature_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    tags = {tag_constants.SERVING}

    input_placeholder = self._create_and_save_tf1_conv_model(
        input_saved_model_path,
        signature_key,
        tags,
        input_key='x',
        output_key='output',
        use_variable=True)

    signature_keys = [signature_key]
    output_directory = self.create_tempdir().full_path

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            experimental_method=_ExperimentalMethod.STATIC_RANGE))

    data_gen = self._create_data_generator(
        input_key='x', shape=input_placeholder.shape)

    converted_model = quantize_model.quantize(
        input_saved_model_path,
        signature_keys,
        tags,
        output_directory,
        quantization_options,
        representative_dataset=data_gen)

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(converted_model.signatures._signatures.keys(),
                          signature_keys)

    output_loader = saved_model_loader.SavedModelLoader(output_directory)
    output_meta_graphdef = output_loader.get_meta_graph_def_from_tags(tags)
    self.assertTrue(
        self._contains_quantized_function_call(output_meta_graphdef))

  @parameterized.named_parameters(
      ('use_constant', False),
      ('use_variable', True),
  )
  @test_util.run_in_graph_and_eager_modes
  def test_ptq_model_with_tf1_saved_model_with_variable_for_gather(
      self, use_variable):
    input_saved_model_path = self.create_tempdir('input').full_path
    signature_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    tags = {tag_constants.SERVING}

    input_placeholder = self._create_and_save_tf1_gather_model(
        input_saved_model_path,
        signature_key,
        tags,
        input_key='x',
        output_key='output',
        use_variable=use_variable)

    signature_keys = [signature_key]
    output_directory = self.create_tempdir().full_path

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            experimental_method=_ExperimentalMethod.STATIC_RANGE))

    data_gen = self._create_data_generator(
        input_key='x',
        shape=input_placeholder.shape,
        minval=0,
        maxval=10,
        dtype=dtypes.int64)

    converted_model = quantize_model.quantize(
        input_saved_model_path,
        signature_keys,
        tags,
        output_directory,
        quantization_options,
        representative_dataset=data_gen)

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(converted_model.signatures._signatures.keys(),
                          signature_keys)

    output_loader = saved_model_loader.SavedModelLoader(output_directory)
    output_meta_graphdef = output_loader.get_meta_graph_def_from_tags(tags)
    # Quantization is not currently supported for gather.
    self.assertFalse(
        self._contains_quantized_function_call(output_meta_graphdef))

  @test_util.run_in_graph_and_eager_modes
  def test_ptq_model_with_tf1_saved_model(self):
    input_saved_model_path = self.create_tempdir('input').full_path
    tags = {tag_constants.SERVING}
    signature_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY

    input_placeholder = self._create_and_save_tf1_conv_model(
        input_saved_model_path,
        signature_key,
        tags,
        input_key='p',
        output_key='output',
        use_variable=False)

    signature_keys = [signature_key]
    output_directory = self.create_tempdir().full_path

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            experimental_method=_ExperimentalMethod.STATIC_RANGE))

    data_gen = self._create_data_generator(
        input_key='p', shape=input_placeholder.shape)

    converted_model = quantize_model.quantize(
        input_saved_model_path,
        signature_keys,
        tags,
        output_directory,
        quantization_options,
        representative_dataset=data_gen)

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(converted_model.signatures._signatures.keys(),
                          signature_keys)

    output_loader = saved_model_loader.SavedModelLoader(output_directory)
    output_meta_graphdef = output_loader.get_meta_graph_def_from_tags(tags)
    self.assertTrue(
        self._contains_quantized_function_call(output_meta_graphdef))

  @test_util.run_in_graph_and_eager_modes
  def test_ptq_model_with_tf1_saved_model_multiple_signatures(self):
    input_saved_model_path = self.create_tempdir('input').full_path
    tags = {tag_constants.SERVING}

    # Create two models and add them to a same SavedModel under different
    # signature keys.
    with ops.Graph().as_default(), session.Session() as sess:
      in_placeholder_1, output_tensor_1 = self._create_simple_tf1_conv_model()
      sig_def_1 = signature_def_utils_impl.predict_signature_def(
          inputs={'x1': in_placeholder_1}, outputs={'output1': output_tensor_1})

      in_placeholder_2, output_tensor_2 = self._create_simple_tf1_conv_model()
      sig_def_2 = signature_def_utils_impl.predict_signature_def(
          inputs={'x2': in_placeholder_2}, outputs={'output2': output_tensor_2})

      v1_builder = builder.SavedModelBuilder(input_saved_model_path)
      v1_builder.add_meta_graph_and_variables(
          sess, tags, signature_def_map={
              'sig1': sig_def_1,
              'sig2': sig_def_2,
          })

      v1_builder.save()

    output_directory = self.create_tempdir().full_path
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            experimental_method=_ExperimentalMethod.STATIC_RANGE))

    def data_gen_sig1() -> repr_dataset.RepresentativeDataset:
      """Generates tuple-style samples.

      The first element of the tuple identifies the signature key the input data
      is for.

      Yields:
        Representative samples for signature 'sig1'.
      """
      for _ in range(4):
        yield {'x1': random_ops.random_uniform(shape=in_placeholder_1.shape)}

    def data_gen_sig2() -> repr_dataset.RepresentativeDataset:
      """Generates tuple-style samples.

      The first element of the tuple identifies the signature key the input data
      is for.

      Yields:
        Representative samples for signature 'sig2'.
      """
      for _ in range(4):
        yield {'x2': random_ops.random_uniform(shape=in_placeholder_2.shape)}

    converted_model = quantize_model.quantize(
        input_saved_model_path,
        signature_keys=['sig1', 'sig2'],
        tags=tags,
        output_directory=output_directory,
        quantization_options=quantization_options,
        representative_dataset={
            'sig1': data_gen_sig1(),
            'sig2': data_gen_sig2(),
        })

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(converted_model.signatures._signatures.keys(),
                          {'sig1', 'sig2'})

    output_loader = saved_model_loader.SavedModelLoader(output_directory)
    output_meta_graphdef = output_loader.get_meta_graph_def_from_tags(tags)
    self.assertTrue(
        self._contains_quantized_function_call(output_meta_graphdef))

  @test_util.run_in_graph_and_eager_modes
  def test_ptq_model_with_tf1_saved_model_invalid_input_key_raises_value_error(
      self):
    input_saved_model_path = self.create_tempdir('input').full_path
    signature_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    tags = {tag_constants.SERVING}

    input_placeholder = self._create_and_save_tf1_conv_model(
        input_saved_model_path,
        signature_key,
        tags,
        input_key='x',
        output_key='output',
        use_variable=False)

    signature_keys = [signature_key]
    output_directory = self.create_tempdir().full_path

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            experimental_method=_ExperimentalMethod.STATIC_RANGE))

    # Representative generator function that yields with an invalid input key.
    invalid_data_gen = self._create_data_generator(
        input_key='invalid_input_key', shape=input_placeholder.shape)

    with self.assertRaisesRegex(
        ValueError,
        'Failed to run graph for post-training quantization calibration'):
      quantize_model.quantize(
          input_saved_model_path,
          signature_keys,
          tags,
          output_directory,
          quantization_options,
          representative_dataset=invalid_data_gen)

  @test_util.run_in_graph_and_eager_modes
  def test_ptq_model_with_non_default_tags(self):
    input_saved_model_path = self.create_tempdir('input').full_path
    signature_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    # Use a different set of tags other than {"serve"}.
    tags = {tag_constants.TRAINING, tag_constants.GPU}

    # Non-default tags are usually used when saving multiple metagraphs in TF1.
    input_placeholder = self._create_and_save_tf1_conv_model(
        input_saved_model_path,
        signature_key,
        tags,
        input_key='input',
        output_key='output',
        use_variable=True)

    signature_keys = [signature_key]
    output_directory = self.create_tempdir().full_path

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            experimental_method=_ExperimentalMethod.STATIC_RANGE))

    data_gen = self._create_data_generator(
        input_key='input', shape=input_placeholder.shape)

    converted_model = quantize_model.quantize(
        input_saved_model_path,
        signature_keys,
        tags,
        output_directory,
        quantization_options,
        representative_dataset=data_gen)

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(converted_model.signatures._signatures.keys(),
                          signature_keys)

    output_loader = saved_model_loader.SavedModelLoader(output_directory)
    output_meta_graphdef = output_loader.get_meta_graph_def_from_tags(tags)
    self.assertTrue(
        self._contains_quantized_function_call(output_meta_graphdef))

  @test_util.run_in_graph_and_eager_modes
  def test_ptq_model_with_wrong_tags_raises_error(self):
    input_saved_model_path = self.create_tempdir('input').full_path
    signature_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    save_tags = {tag_constants.TRAINING, tag_constants.GPU}

    input_placeholder = self._create_and_save_tf1_conv_model(
        input_saved_model_path,
        signature_key,
        save_tags,
        input_key='input',
        output_key='output',
        use_variable=True)

    signature_keys = [signature_key]
    output_directory = self.create_tempdir().full_path

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            experimental_method=_ExperimentalMethod.STATIC_RANGE))

    # Try to use a different set of tags to quantize.
    tags = {tag_constants.SERVING}
    data_gen = self._create_data_generator(
        input_key='input', shape=input_placeholder.shape)
    with self.assertRaisesRegex(RuntimeError,
                                'Failed to retrieve MetaGraphDef'):
      quantize_model.quantize(
          input_saved_model_path,
          signature_keys,
          tags,
          output_directory,
          quantization_options,
          representative_dataset=data_gen)


class DynamicRangeQuantizationTest(quantize_model_test_base.QuantizedModelTest):
  """Test cases for dynamic range quantization.

  Run all tests cases in both the graph mode (default in TF1) and the eager mode
  (default in TF2) to ensure support for when TF2 is disabled.
  """

  @test_util.run_in_graph_and_eager_modes
  def test_matmul_model(self):
    model = self._create_matmul_model()
    input_saved_model_path = self.create_tempdir('input').full_path
    saved_model_save.save(model, input_saved_model_path)

    tags = [tag_constants.SERVING]
    output_directory = self.create_tempdir().full_path

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            experimental_method=_ExperimentalMethod.DYNAMIC_RANGE),
        op_set=quant_opts_pb2.OpSet.UNIFORM_QUANTIZED)

    converted_model = quantize_model.quantize(input_saved_model_path,
                                              ['serving_default'], tags,
                                              output_directory,
                                              quantization_options)
    self.assertIsNotNone(converted_model)
    self.assertCountEqual(converted_model.signatures._signatures.keys(),
                          {'serving_default'})

    output_loader = saved_model_loader.SavedModelLoader(output_directory)
    output_meta_graphdef = output_loader.get_meta_graph_def_from_tags(tags)
    self.assertTrue(
        self._contains_quantized_function_call(output_meta_graphdef))
    self.assertTrue(
        self._contains_op(output_meta_graphdef, 'UniformQuantizedDotHybrid'))
    self.assertFalse(self._contains_op(output_meta_graphdef, 'MatMul'))

  @test_util.run_in_graph_and_eager_modes
  def test_conv_model(self):
    model = self._create_conv2d_model()
    input_saved_model_path = self.create_tempdir('input').full_path
    saved_model_save.save(model, input_saved_model_path)

    tags = [tag_constants.SERVING]
    output_directory = self.create_tempdir().full_path

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            experimental_method=_ExperimentalMethod.DYNAMIC_RANGE),
        op_set=quant_opts_pb2.OpSet.UNIFORM_QUANTIZED)

    converted_model = quantize_model.quantize(input_saved_model_path,
                                              ['serving_default'], tags,
                                              output_directory,
                                              quantization_options)

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(converted_model.signatures._signatures.keys(),
                          {'serving_default'})

    output_loader = saved_model_loader.SavedModelLoader(output_directory)
    output_meta_graphdef = output_loader.get_meta_graph_def_from_tags(tags)
    # Currently conv is not supported.
    self.assertFalse(
        self._contains_quantized_function_call(output_meta_graphdef))

  @parameterized.named_parameters(
      ('use_constant', False),
      ('use_variable', True),
  )
  @test_util.run_v2_only
  def test_gather_model(self, use_variable):
    model = self._create_gather_model(use_variable)
    input_saved_model_path = self.create_tempdir('input').full_path
    saved_model_save.save(model, input_saved_model_path)

    tags = [tag_constants.SERVING]
    output_directory = self.create_tempdir().full_path

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            experimental_method=_ExperimentalMethod.DYNAMIC_RANGE),
        op_set=quant_opts_pb2.OpSet.UNIFORM_QUANTIZED)

    converted_model = quantize_model.quantize(input_saved_model_path,
                                              ['serving_default'], tags,
                                              output_directory,
                                              quantization_options)

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(converted_model.signatures._signatures.keys(),
                          {'serving_default'})

    output_loader = saved_model_loader.SavedModelLoader(output_directory)
    output_meta_graphdef = output_loader.get_meta_graph_def_from_tags(tags)
    # Currently gather is not supported.
    self.assertFalse(
        self._contains_quantized_function_call(output_meta_graphdef))

  @test_util.run_in_graph_and_eager_modes
  def test_conv_model_with_wrong_tags_raises_error(self):
    input_saved_model_path = self.create_tempdir('input').full_path
    signature_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    save_tags = {tag_constants.TRAINING, tag_constants.GPU}

    input_placeholder = self._create_and_save_tf1_conv_model(
        input_saved_model_path,
        signature_key,
        save_tags,
        input_key='input',
        output_key='output',
        use_variable=True)

    signature_keys = [signature_key]
    output_directory = self.create_tempdir().full_path

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            experimental_method=_ExperimentalMethod.DYNAMIC_RANGE),
        op_set=quant_opts_pb2.OpSet.UNIFORM_QUANTIZED)

    # Try to use a different set of tags to quantize.
    tags = {tag_constants.SERVING}
    data_gen = self._create_data_generator(
        input_key='input', shape=input_placeholder.shape)
    with self.assertRaisesRegex(ValueError, 'Failed to import SavedModel'):
      quantize_model.quantize(
          input_saved_model_path,
          signature_keys,
          tags,
          output_directory,
          quantization_options,
          representative_dataset=data_gen)

  @parameterized.named_parameters(
      ('quantize', True, 0),
      ('not_quantize', False, 10000),
  )
  @test_util.run_in_graph_and_eager_modes
  def test_minimum_elements_for_weights(self, quantize, num_elements):
    model = self._create_matmul_model()
    input_saved_model_path = self.create_tempdir('input').full_path
    saved_model_save.save(model, input_saved_model_path)

    tags = [tag_constants.SERVING]
    output_directory = self.create_tempdir().full_path

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            experimental_method=_ExperimentalMethod.DYNAMIC_RANGE),
        op_set=quant_opts_pb2.OpSet.UNIFORM_QUANTIZED)
    quantization_options.min_num_elements_for_weights = num_elements

    converted_model = quantize_model.quantize(input_saved_model_path,
                                              ['serving_default'], tags,
                                              output_directory,
                                              quantization_options)

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(converted_model.signatures._signatures.keys(),
                          {'serving_default'})

    output_loader = saved_model_loader.SavedModelLoader(output_directory)
    output_meta_graphdef = output_loader.get_meta_graph_def_from_tags(tags)
    if quantize:
      self.assertTrue(
          self._contains_quantized_function_call(output_meta_graphdef))
    else:
      self.assertFalse(
          self._contains_quantized_function_call(output_meta_graphdef))

  @parameterized.named_parameters(
      ('use_constant', False),
      ('use_variable', True),
  )
  @test_util.run_in_graph_and_eager_modes
  def test_gather_model_tf1(self, use_variable):
    input_saved_model_path = self.create_tempdir('input').full_path
    signature_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    tags = {tag_constants.SERVING}

    _ = self._create_and_save_tf1_gather_model(
        input_saved_model_path,
        signature_key,
        tags,
        input_key='x',
        output_key='output',
        use_variable=use_variable)

    signature_keys = [signature_key]
    output_directory = self.create_tempdir().full_path

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            experimental_method=_ExperimentalMethod.DYNAMIC_RANGE),
        op_set=quant_opts_pb2.OpSet.UNIFORM_QUANTIZED)

    converted_model = quantize_model.quantize(input_saved_model_path,
                                              signature_keys, tags,
                                              output_directory,
                                              quantization_options)

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(converted_model.signatures._signatures.keys(),
                          signature_keys)

    output_loader = saved_model_loader.SavedModelLoader(output_directory)
    output_meta_graphdef = output_loader.get_meta_graph_def_from_tags(tags)
    # Quantization is not currently supported for gather.
    self.assertFalse(
        self._contains_quantized_function_call(output_meta_graphdef))

  @test_util.run_in_graph_and_eager_modes
  def test_non_empty_directory_raises_file_exists_error(self):
    model = self._create_matmul_model()

    input_saved_model_path = self.create_tempdir('input').full_path
    saved_model_save.save(model, input_saved_model_path)

    tags = [tag_constants.SERVING]

    # Create a file inside the output directory.
    output_directory = self.create_tempdir().full_path
    file_io.write_string_to_file(
        filename=os.path.join(output_directory, 'dummy_file.txt'),
        file_content='Test content')

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            experimental_method=_ExperimentalMethod.DYNAMIC_RANGE))

    with self.assertRaisesRegex(FileExistsError,
                                'Output directory already exists'):
      quantize_model.quantize(input_saved_model_path, ['serving_default'], tags,
                              output_directory, quantization_options)

  @test_util.run_in_graph_and_eager_modes
  def test_non_empty_directory_overwritten(self):
    model = self._create_matmul_model()

    input_saved_model_path = self.create_tempdir('input').full_path
    saved_model_save.save(model, input_saved_model_path)

    tags = [tag_constants.SERVING]

    # Create a file inside the output directory.
    output_directory = self.create_tempdir().full_path
    file_io.write_string_to_file(
        filename=os.path.join(output_directory, 'dummy_file.txt'),
        file_content='Test content')

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            experimental_method=_ExperimentalMethod.DYNAMIC_RANGE))

    converted_model = quantize_model.quantize(
        input_saved_model_path, ['serving_default'],
        tags,
        output_directory,
        quantization_options,
        overwrite_output_directory=True)

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(converted_model.signatures._signatures.keys(),
                          {'serving_default'})

    output_loader = saved_model_loader.SavedModelLoader(output_directory)
    output_meta_graphdef = output_loader.get_meta_graph_def_from_tags(tags)
    self.assertTrue(
        self._contains_quantized_function_call(output_meta_graphdef))


if __name__ == '__main__':
  test.main()
