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
from typing import List, Mapping, Optional, Sequence, Tuple, Union

from absl.testing import parameterized
import numpy as np
import tensorflow  # pylint: disable=unused-import

from tensorflow.compiler.mlir.quantization.tensorflow import quantization_options_pb2 as quant_opts_pb2
from tensorflow.compiler.mlir.quantization.tensorflow.python import quantize_model
from tensorflow.compiler.mlir.quantization.tensorflow.python import representative_dataset as repr_dataset
from tensorflow.compiler.mlir.quantization.tensorflow.python import save_model
from tensorflow.compiler.mlir.quantization.tensorflow.python.integration_test import quantize_model_test_base
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.python.client import session
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
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
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model import loader_impl as saved_model_loader
from tensorflow.python.saved_model import save as saved_model_save
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils_impl
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.training import checkpoint_utils
from tensorflow.python.types import core

# Type aliases for quantization method protobuf enums.
_Method = quant_opts_pb2.QuantizationMethod.Method
_ExperimentalMethod = quant_opts_pb2.QuantizationMethod.ExperimentalMethod


def _is_variable(node_def: node_def_pb2.NodeDef) -> bool:
  """Determines whether `node_def` is a variable node.

  Args:
    node_def: `NodeDef` to test whether it is a variable or not.

  Returns:
    Returns True if it is a variable.
  """
  return node_def.op == 'VarHandleOp'


def _find_variables(
    graph_def: graph_pb2.GraphDef) -> Mapping[str, node_def_pb2.NodeDef]:
  """Finds all variables within `graph_def`.

  This function makes sense for TF 1 graphs only, as it depends on
  `shared_name`.

  Args:
    graph_def: `GraphDef` to find variables from.

  Returns:
    A mapping of `shared_name` -> `NodeDef` corresponding to a variable op.
  """
  variable_nodes = {}

  for var_node in filter(_is_variable, graph_def.node):
    shared_name = str(var_node.attr['shared_name'].s, encoding='utf-8')
    variable_nodes[shared_name] = var_node

  for func in graph_def.library.function:
    for var_node in filter(_is_variable, func.node_def):
      variable_nodes[shared_name] = var_node

  return variable_nodes


def parameter_combinations(test_parameters):
  """Generate all combinations of test parameters."""
  real_parameters = []
  for parameters in test_parameters:
    keys = parameters.keys()
    for curr in itertools.product(*parameters.values()):
      real_parameters.append(dict(zip(keys, curr)))
  return real_parameters


def get_dir_size(path='.'):
  """Get the total size of files and sub-directories under the path."""
  total = 0
  for root, dirs, files in os.walk(path):
    for filename in files:
      total += os.path.getsize(os.path.join(root, filename))
    for dirname in dirs:
      total += get_dir_size(os.path.join(root, dirname))
  return total


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
class QuantizationOptionsTest(quantize_model_test_base.QuantizedModelTest):
  """Test cases regarding the use of QuantizationOptions proto.

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

  def test_per_channel_for_non_uniform_opset_raises_value_error(self):
    model = self.SimpleModel()

    input_saved_model_path = self.create_tempdir('input').full_path
    saved_model_save.save(model, input_saved_model_path)

    options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            experimental_method=_ExperimentalMethod.DYNAMIC_RANGE),
        op_set=quant_opts_pb2.TF,
        enable_per_channel_quantization=True)

    with self.assertRaises(ValueError):
      quantize_model.quantize(
          input_saved_model_path, quantization_options=options)


class TensorNamePreservationTest(quantize_model_test_base.QuantizedModelTest):

  def test_preserving_input_output_tensor_names(self):

    class MultiSignatureModel(module.Module):

      @def_function.function(input_signature=[
          tensor_spec.TensorSpec(
              name='input', shape=[32], dtype=dtypes.float32),
      ])
      def multiple_output_ops(
          self, input_tensor: core.Tensor) -> Mapping[str, core.Tensor]:
        k = array_ops.constant(4, dtype=dtypes.int32)
        values, indices = nn_ops.top_k(input_tensor, k, name='TopK')
        adj_values = values + 2
        return {'indices': indices, 'adj_values': adj_values, 'values': values}

      @def_function.function(input_signature=[
          tensor_spec.TensorSpec(
              name='input', shape=[32], dtype=dtypes.float32),
      ])
      def duplicate_outputs(
          self, input_tensor: core.Tensor) -> Mapping[str, core.Tensor]:
        q_input = array_ops.fake_quant_with_min_max_args(
            input_tensor, min=-0.1, max=0.2, num_bits=8, narrow_range=False)
        adj_values = q_input + 2
        return {'adj_values_1': adj_values, 'adj_values_2': adj_values}

      @def_function.function(input_signature=[
          tensor_spec.TensorSpec(
              name='input', shape=[32], dtype=dtypes.float32),
      ])
      def return_higher_index_only(
          self, input_tensor: core.Tensor) -> Mapping[str, core.Tensor]:
        k = array_ops.constant(4, dtype=dtypes.int32)
        values, indices = nn_ops.top_k(input_tensor, k, name='TopK')
        adj_values = values + 2
        return {'indices': indices, 'adj_values': adj_values}

    model = MultiSignatureModel()
    signatures = {
        'multiple_output_ops': model.multiple_output_ops,
        'duplicate_outputs': model.duplicate_outputs,
        'return_higher_index_only': model.return_higher_index_only,
    }
    input_saved_model_path = self.create_tempdir('input').full_path
    saved_model_save.save(model, input_saved_model_path, signatures=signatures)

    tags = {tag_constants.SERVING}
    original_signature_map = save_model.get_signatures_from_saved_model(
        input_saved_model_path, signature_keys=signatures.keys(), tags=tags)

    output_directory = self.create_tempdir('output').full_path
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            experimental_method=_ExperimentalMethod.STATIC_RANGE),
        op_set=quant_opts_pb2.TF)
    quantize_model.quantize(input_saved_model_path, signatures.keys(), tags,
                            output_directory, quantization_options)
    converted_signature_map = save_model.get_signatures_from_saved_model(
        output_directory, signature_keys=signatures.keys(), tags=tags)

    # The original and converted model should have the same signature map.
    self.assertAllInSet(
        list(original_signature_map.keys()), set(signatures.keys()))
    self.assertDictEqual(original_signature_map, converted_signature_map)

  def test_duplicated_tensor_name(self):
    with session.Session(graph=ops.Graph()) as sess:
      input_tensor = array_ops.placeholder(
          dtypes.float32, shape=[], name='input')
      q_input = array_ops.fake_quant_with_min_max_args(
          input_tensor, min=-0.1, max=0.2, num_bits=8, narrow_range=False)
      sqrt = math_ops.sqrt(q_input, name='sqrt')
      identity = array_ops.identity(sqrt, name='output')

      input_map = {'input': input_tensor}
      output_map = {'sqrt': identity}
      signature = signature_def_utils_impl.predict_signature_def(
          inputs=input_map, outputs=output_map)
      signature_map = {'main': signature}

      tags = {tag_constants.SERVING}
      saved_model_path = self.create_tempdir('input').full_path
      v1_builder = builder.SavedModelBuilder(saved_model_path)
      v1_builder.add_meta_graph_and_variables(
          sess, tags, signature_def_map=signature_map)
      v1_builder.save()

    output_directory = self.create_tempdir('output').full_path
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            experimental_method=_ExperimentalMethod.STATIC_RANGE),
        op_set=quant_opts_pb2.TF)
    quantize_model.quantize(saved_model_path, signature_map.keys(), tags,
                            output_directory, quantization_options)
    converted_signature_map = save_model.get_signatures_from_saved_model(
        output_directory, signature_keys=signature_map.keys(), tags=tags)
    # The original and converted model should have the same signature map.
    self.assertDictEqual(signature_map, converted_signature_map)

_TensorShape = Sequence[Union[int, None]]


class StaticRangeQuantizationTest(quantize_model_test_base.QuantizedModelTest):

  def _any_warning_contains(self, substring: str,
                            warnings_list: List['LogRecord']) -> bool:
    """Returns True if any of the warnings contains a given substring.

    Args:
      substring: A piece of string to check whether it exists in the warning
        message.
      warnings_list: A list of `absl.logging.LogRecord`s.

    Returns:
      True if and only if the substring exists in any of the warnings in
      `warnings_list`.
    """
    return any(
        map(lambda warning: substring in str(warning.message), warnings_list))

  @parameterized.parameters(
      parameter_combinations([{
          'shapes': [([3, 3], [3, 3]), ([3, None], [None, 3]),
                     ([None, None], [None, None]), ([4, 3, 3], [4, 3, 3]),
                     ([4, 3, None], [4, None, 3]),
                     ([None, None, None], [None, None, None])],
          'activation_fn': [None, nn_ops.relu, nn_ops.relu6],
          'has_bias': [True, False],
      }]))
  @test_util.run_in_graph_and_eager_modes
  def test_qat_matmul_model(
      self, shapes: Sequence[Tuple[_TensorShape, _TensorShape]],
      activation_fn: Optional[ops.Operation], has_bias: bool):

    n = 5
    x_shape = [v if v is not None else n for v in shapes[0]]
    kernel_shape = [v if v is not None else n for v in shapes[1]]

    class MatmulModel(module.Module):

      def __init__(self, bias: Optional[core.Tensor]):
        self._bias = bias
        self._kernel = np.random.uniform(size=kernel_shape).astype('f4')
        self._min = (-0.8, -0.8, -0.9)
        self._max = (0.9, 0.9, 1.0)

      @def_function.function(input_signature=[
          tensor_spec.TensorSpec(
              name='x', shape=shapes[0], dtype=dtypes.float32)
      ])
      def matmul(self, x: core.Tensor) -> Mapping[str, core.Tensor]:
        """A tensor-by-tensor matmul function with fake-quants.

        Args:
          x: Tensor for matmul

        Returns:
          The matmul result with fake-quant operation attached.
        """
        x = array_ops.fake_quant_with_min_max_vars(
            x,
            min=ops.convert_to_tensor(self._min[0]),
            max=ops.convert_to_tensor(self._max[0]),
            num_bits=8,
            narrow_range=False)
        kernel = array_ops.fake_quant_with_min_max_vars(
            ops.convert_to_tensor(self._kernel),
            min=ops.convert_to_tensor(self._min[1]),
            max=ops.convert_to_tensor(self._max[1]),
            num_bits=8,
            narrow_range=False)

        out = math_ops.matmul(x, kernel)
        if self._bias is not None:
          out = nn_ops.bias_add(out, self._bias)
        if activation_fn is not None:
          out = activation_fn(out)
        out = array_ops.fake_quant_with_min_max_vars(
            out,
            min=ops.convert_to_tensor(self._min[2]),
            max=ops.convert_to_tensor(self._max[2]),
            num_bits=8,
            narrow_range=False)
        return {'output': out}

    np.random.seed(1234)
    bias = None
    if has_bias:
      bias_shape = shapes[1][-1]
      if bias_shape is not None:
        bias = array_ops.constant(
            np.random.uniform(size=[shapes[1][-1]]), dtype=dtypes.float32)
    model = MatmulModel(bias)
    input_saved_model_path = self.create_tempdir('input').full_path
    saved_model_save.save(model, input_saved_model_path)

    signature_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    tags = {tag_constants.SERVING}

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

    x = array_ops.constant(
        np.random.uniform(size=x_shape), dtype=dtypes.float32)

    expected_outputs = model.matmul(x)
    got_outputs = converted_model.signatures[signature_key](x=x)
    self.assertAllClose(expected_outputs, got_outputs, atol=1e-1)

    output_loader = saved_model_loader.SavedModelLoader(output_directory)
    output_meta_graphdef = output_loader.get_meta_graph_def_from_tags(tags)
    self.assertTrue(
        self._contains_quantized_function_call(output_meta_graphdef))

    # Check the converted model in the XLA opset.
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            experimental_method=_ExperimentalMethod.STATIC_RANGE),
        op_set=quant_opts_pb2.XLA)

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
    self.assertTrue(self._contains_op(meta_graphdef, 'XlaDotV2'))

    new_outputs = converted_model.signatures[signature_key](x=x)

    # The difference between TF and XLA path is expected to be small (smaller
    # or equal to 1 in the quantized domain).
    self.assertAllClose(new_outputs, expected_outputs, atol=1e-1)

  @parameterized.parameters(
      parameter_combinations([{
          'activation_fn': [None, nn_ops.relu, nn_ops.relu6],
          'has_bias': [True, False],
          'has_batch_norm': [True, False],
          'target_opset': [quant_opts_pb2.XLA],
      }]))
  @test_util.run_in_graph_and_eager_modes
  def test_qat_conv_model(self, activation_fn: Optional[ops.Operation],
                          has_bias: bool, has_batch_norm: bool,
                          target_opset: quant_opts_pb2.OpSet):

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
        filter_min = array_ops.identity(
            array_ops.constant([-0.5, -0.5], dtype=dtypes.float32))
        filter_max = array_ops.identity(
            array_ops.constant([0.5, 0.5], dtype=dtypes.float32))
        q_filter = array_ops.fake_quant_with_min_max_vars_per_channel(
            filter_tensor,
            filter_min,
            filter_max,
            num_bits=8,
            narrow_range=True)
        bias = array_ops.constant([0.1, 0.2], dtype=dtypes.float32)
        scale, offset = [1.0] * 2, [0.5] * 2
        mean, variance = scale, offset
        out = nn_ops.conv2d(
            q_input,
            q_filter,
            strides=[1, 1, 2, 1],
            dilations=[1, 1, 1, 1],
            padding='SAME',
            data_format='NHWC')
        if has_bias:
          out = nn_ops.bias_add(out, bias, data_format='NHWC')
        if activation_fn is not None:
          # The accuracy is not good when having FusedBatchNorm without
          # activation in this test.
          if has_batch_norm:
            # Fusing is supported for non-training case.
            out, _, _, _, _, _ = nn_ops.fused_batch_norm_v3(
                out, scale, offset, mean, variance, is_training=False)
          out = activation_fn(out)
        out_min = array_ops.constant([-0.18, -0.32], dtype=dtypes.float32)
        out_max = array_ops.constant([0.5, 0.5], dtype=dtypes.float32)
        q_out = array_ops.fake_quant_with_min_max_vars_per_channel(
            out, min=out_min, max=out_max, num_bits=8, narrow_range=True)
        return {'output': q_out}

    np.random.seed(1234)
    model = ConvModel()
    input_saved_model_path = self.create_tempdir('input').full_path
    saved_model_save.save(model, input_saved_model_path)

    signature_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    tags = {tag_constants.SERVING}

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
    self.assertAllClose(expected_outputs, got_outputs, atol=0.00323)

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
    self.assertAllClose(new_outputs, got_outputs, atol=0.00154)

  # TODO(b/244276332): Allow table initialization in TF2 eager mode.
  @test_util.deprecated_graph_mode_only
  def test_qat_vocab_table_lookup_model(self):
    tags = {tag_constants.SERVING}
    signature_def_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    input_model_dir = self.create_tempdir('input').full_path

    # Create and save a simple model that involves a hash table.
    inputs, outputs = self._create_and_save_vocab_table_lookup_qat_model_tf1(
        input_model_dir, tags, signature_def_key)

    # Make sure that the desired input key and output key is present.
    self.assertIn('input_vocabs', inputs.keys())
    self.assertIn('lookup', outputs.keys())

    # Representative dataset is composed of a set of vocabs for table lookup.
    repr_ds = [{
        'input_vocabs': np.array([b'hello', b'model', b'quantization'])
    } for _ in range(4)]

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            experimental_method=_ExperimentalMethod.STATIC_RANGE))

    signature_def_keys = [signature_def_key]
    output_model_dir = self.create_tempdir('output').full_path

    quantize_model.quantize(
        input_model_dir,
        signature_def_keys,
        tags,
        output_model_dir,
        quantization_options,
        representative_dataset=repr_ds)

    # Tests table lookup to make sure the table has been initialized
    # successfully.
    with session.Session(graph=ops.Graph()) as sess:
      output_meta_graph_def = saved_model_loader.load(
          sess, tags=tags, export_dir=output_model_dir)

      # The graph should contain a quantized function call (it contains a
      # single f32 matmul node).
      self.assertTrue(
          self._contains_quantized_function_call(output_meta_graph_def))
      self.assertCountEqual(output_meta_graph_def.signature_def.keys(),
                            signature_def_keys)

      signature_def = output_meta_graph_def.signature_def[signature_def_key]

      input_tensor_name = signature_def.inputs['input_vocabs'].name
      input_tensor = sess.graph.get_tensor_by_name(input_tensor_name)

      lookup_tensor_name = signature_def.outputs['lookup'].name
      lookup_tensor = sess.graph.get_tensor_by_name(lookup_tensor_name)

      lookup_val = sess.run(
          lookup_tensor,
          feed_dict={
              input_tensor: np.array([b'model', b'quantization', b'hello'])
          })

      self.assertAllClose(lookup_val, [1., 2., 0.])

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

  # TODO(b/263830952): Use dictionaries instead of tuples for parameters.
  @parameterized.named_parameters(
      ('none', None, False, False, quant_opts_pb2.TF, False, False),
      ('relu', nn_ops.relu, False, False, quant_opts_pb2.TF, False, False),
      ('relu6', nn_ops.relu6, False, False, quant_opts_pb2.TF, False, False),
      ('bn', None, False, True, quant_opts_pb2.TF, False, False),
      ('bn_and_relu', nn_ops.relu, False, True, quant_opts_pb2.TF, False,
       False),
      ('with_bias', None, True, False, quant_opts_pb2.TF, False, False),
      ('with_bias_and_bn', None, True, True, quant_opts_pb2.TF, False, False),
      ('with_bias_and_bn_and_relu', nn_ops.relu, True, True, quant_opts_pb2.TF,
       False, False),
      ('with_bias_and_relu', nn_ops.relu, True, False, quant_opts_pb2.TF, False,
       False),
      ('with_bias_and_relu6', nn_ops.relu6, True, False, quant_opts_pb2.TF,
       False, False),
      ('with_bias_and_bn_to_xla', None, True, True, quant_opts_pb2.XLA, False,
       False),
      ('with_bias_and_relu6_to_xla', nn_ops.relu6, True, False,
       quant_opts_pb2.XLA, False, False),
      ('with_bias_and_bn_to_xla_dynamic', None, True, True, quant_opts_pb2.XLA,
       True, False),
      ('with_bias_and_relu6_to_xla_dynamic', nn_ops.relu6, True, False,
       quant_opts_pb2.XLA, True, False),
      ('none_to_uq', None, False, False, quant_opts_pb2.UNIFORM_QUANTIZED,
       False, False),
      ('none_to_uq_per_channel', None, False, False,
       quant_opts_pb2.UNIFORM_QUANTIZED, False, True),
      ('relu_to_uq', nn_ops.relu, False, False,
       quant_opts_pb2.UNIFORM_QUANTIZED, False, False),
      ('with_bias_to_uq', None, True, False, quant_opts_pb2.UNIFORM_QUANTIZED,
       False, False),
      ('with_bias_and_relu_to_uq', nn_ops.relu, True, False,
       quant_opts_pb2.UNIFORM_QUANTIZED, False, False),
      ('with_bias_and_relu6_to_uq', nn_ops.relu6, True, False,
       quant_opts_pb2.UNIFORM_QUANTIZED, False, False),
  )
  @test_util.run_in_graph_and_eager_modes
  def test_conv_ptq_model(self, activation_fn: Optional[ops.Operation],
                          has_bias: bool, has_batch_norm: bool,
                          target_opset: quant_opts_pb2.OpSet,
                          input_shape_dynamic: bool,
                          enable_per_channel_quantization: bool):
    input_shape = [None, None, None, 3] if input_shape_dynamic else [1, 3, 4, 3]
    filter_shape = [2, 3, 3, 2]

    np.random.seed(1234)
    model = self._create_conv2d_model(input_shape, filter_shape, has_bias,
                                      has_batch_norm, activation_fn)
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

    tags = {tag_constants.SERVING}
    output_directory = self.create_tempdir().full_path

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            experimental_method=_ExperimentalMethod.STATIC_RANGE),
        op_set=target_opset,
        enable_per_channel_quantization=enable_per_channel_quantization)

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
    elif target_opset == quant_opts_pb2.UNIFORM_QUANTIZED:
      # _contains_op for UQ ops also checks _contains_quantized_function_call.
      self.assertTrue(
          self._contains_op(output_meta_graphdef,
                            'UniformQuantizedConvolution'))
      if enable_per_channel_quantization:
        quantized_axis_attr = attr_value_pb2.AttrValue(i=3)
        self.assertTrue(
            self._contains_op(output_meta_graphdef,
                              'UniformQuantizedConvolution',
                              'rhs_quantization_axis', quantized_axis_attr))
    else:
      self.assertTrue(
          self._contains_quantized_function_call(output_meta_graphdef))
    self.assertFalse(
        self._contains_op(output_meta_graphdef, 'FusedBatchNormV3'))

  # TODO(b/263830952): Use dictionaries instead of tuples for parameters.
  @parameterized.named_parameters(
      ('none', None, False, False, quant_opts_pb2.TF, False, False),
      ('relu', nn_ops.relu, False, False, quant_opts_pb2.TF, False, False),
      ('relu6', nn_ops.relu6, False, False, quant_opts_pb2.TF, False, False),
      ('bn', None, False, True, quant_opts_pb2.TF, False, False),
      ('bn_and_relu', nn_ops.relu, False, True, quant_opts_pb2.TF, False,
       False),
      ('with_bias', None, True, False, quant_opts_pb2.TF, False, False),
      ('with_bias_and_bn', None, True, True, quant_opts_pb2.TF, False, False),
      ('with_bias_and_bn_and_relu', nn_ops.relu, True, True, quant_opts_pb2.TF,
       False, False),
      ('with_bias_and_relu', nn_ops.relu, True, False, quant_opts_pb2.TF, False,
       False),
      ('with_bias_and_relu6', nn_ops.relu6, True, False, quant_opts_pb2.TF,
       False, False),
      ('with_bias_and_bn_to_xla', None, True, True, quant_opts_pb2.XLA, False,
       False),
      ('with_bias_and_relu6_to_xla', nn_ops.relu6, True, False,
       quant_opts_pb2.XLA, False, False),
      ('with_bias_and_bn_to_xla_dynamic', None, True, True, quant_opts_pb2.XLA,
       True, False),
      ('with_bias_and_relu6_to_xla_dynamic', nn_ops.relu6, True, False,
       quant_opts_pb2.XLA, True, False),
      ('none_to_uq', None, False, False, quant_opts_pb2.UNIFORM_QUANTIZED,
       False, False),
      ('none_to_uq_per_channel', None, False, False,
       quant_opts_pb2.UNIFORM_QUANTIZED, False, True),
      ('relu_to_uq', nn_ops.relu, False, False,
       quant_opts_pb2.UNIFORM_QUANTIZED, False, False),
      ('with_bias_to_uq', None, True, False, quant_opts_pb2.UNIFORM_QUANTIZED,
       False, False),
      ('with_bias_and_relu_to_uq', nn_ops.relu, True, False,
       quant_opts_pb2.UNIFORM_QUANTIZED, False, False),
      ('with_bias_and_relu6_to_uq', nn_ops.relu6, True, False,
       quant_opts_pb2.UNIFORM_QUANTIZED, False, False),
  )
  @test_util.run_in_graph_and_eager_modes
  def test_depthwise_conv_ptq_model(self,
                                    activation_fn: Optional[ops.Operation],
                                    has_bias: bool, has_batch_norm: bool,
                                    target_opset: quant_opts_pb2.OpSet,
                                    input_shape_dynamic: bool,
                                    enable_per_channel_quantization: bool):
    input_shape = ([None, None, None, 3]
                   if input_shape_dynamic else [1, 3, 4, 3])
    filter_shape = [2, 3, 3, 1]
    model = self._create_depthwise_conv2d_model(input_shape, filter_shape,
                                                has_bias, has_batch_norm,
                                                activation_fn)
    np.random.seed(1234)
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

    tags = {tag_constants.SERVING}
    output_directory = self.create_tempdir().full_path

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            experimental_method=_ExperimentalMethod.STATIC_RANGE),
        op_set=target_opset,
        enable_per_channel_quantization=enable_per_channel_quantization)

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
      self.assertTrue(
          self._contains_op(output_meta_graphdef, 'DepthwiseConv2dNative'))
    elif target_opset == quant_opts_pb2.UNIFORM_QUANTIZED:
      # _contains_op for UQ ops also checks _contains_quantized_function_call.
      self.assertTrue(
          self._contains_op(output_meta_graphdef,
                            'UniformQuantizedConvolution'))
      if enable_per_channel_quantization:
        quantized_axis_attr = attr_value_pb2.AttrValue(i=3)
        self.assertTrue(
            self._contains_op(output_meta_graphdef,
                              'UniformQuantizedConvolution',
                              'rhs_quantization_axis', quantized_axis_attr))
    else:
      self.assertTrue(
          self._contains_quantized_function_call(output_meta_graphdef))
    self.assertFalse(
        self._contains_op(output_meta_graphdef, 'FusedBatchNormV3'))

  @parameterized.parameters(*parameter_combinations([
      {
          'activation_fn': [None, nn_ops.relu, nn_ops.relu6],
          'has_bias': [True, False],
          'batch_sizes': [([], []), ([2, 3], [2, 3])],
          'target_opset': [quant_opts_pb2.XLA],
      },
      # Test broadcastable batch sizes.
      {
          'activation_fn': [None],
          'has_bias': [True],
          'batch_sizes': [([2], []), ([], [2]), ([1], [2]), ([None], [])],
          'target_opset': [quant_opts_pb2.XLA],
      }
  ]))
  @test_util.run_in_graph_and_eager_modes
  def test_matmul_ptq_model(self, activation_fn: Optional[ops.Operation],
                            has_bias: bool, batch_sizes: Sequence[int],
                            target_opset: quant_opts_pb2.OpSet):
    np.random.seed(1234)
    lhs_batch_size, rhs_batch_size = batch_sizes
    input_shape = (*lhs_batch_size, 1, 1024)
    filter_shape = (*rhs_batch_size, 1024, 3)
    static_input_shape = [dim if dim is not None else 2 for dim in input_shape]
    input_saved_model_path = self.create_tempdir('input').full_path
    model = self._create_matmul_model(input_shape, filter_shape,
                                      input_saved_model_path, has_bias,
                                      activation_fn)

    def data_gen() -> repr_dataset.RepresentativeDataset:
      for _ in range(500):
        yield {
            'input_tensor':
                ops.convert_to_tensor(
                    np.random.uniform(
                        low=0.0, high=1.0,
                        size=static_input_shape).astype('f4')),
        }

    tags = {tag_constants.SERVING}
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

    input_data = ops.convert_to_tensor(
        np.random.uniform(low=0.0, high=1.0,
                          size=static_input_shape).astype('f4'))
    expected_outputs = model.matmul(input_data)
    got_outputs = converted_model.signatures['serving_default'](
        input_tensor=ops.convert_to_tensor(input_data))
    self.assertAllClose(expected_outputs, got_outputs, atol=0.1674)

    # Check the converted model in the target opset.
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            experimental_method=_ExperimentalMethod.STATIC_RANGE),
        op_set=target_opset)

    output_directory = self.create_tempdir().full_path
    converted_model = quantize_model.quantize(
        input_saved_model_path, ['serving_default'],
        tags,
        output_directory,
        quantization_options,
        representative_dataset=data_gen())

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(converted_model.signatures._signatures.keys(),
                          {'serving_default'})
    loader = saved_model_loader.SavedModelLoader(output_directory)
    output_meta_graphdef = loader.get_meta_graph_def_from_tags(tags)
    if target_opset == quant_opts_pb2.XLA:
      self.assertTrue(self._contains_op(output_meta_graphdef, 'XlaDotV2'))

    new_outputs = converted_model.signatures['serving_default'](
        input_tensor=ops.convert_to_tensor(input_data))
    # The difference between TF and target path is expected to be small.
    self.assertAllClose(new_outputs, got_outputs, atol=0.1202)
    self.assertAllClose(new_outputs, expected_outputs, atol=0.1023)

  @parameterized.parameters(
      ('abc,cde->abde', (2, 2, 64), (64, 3, 3), (3, 3), quant_opts_pb2.XLA),
      ('abc,dce->adbe', (2, 2, 64), (3, 64, 3), (2, 3), quant_opts_pb2.XLA))
  def test_einsum_ptq_model(self, equation: str, input_shape: Sequence[int],
                            weight_shape: Sequence[int],
                            bias_shape: Sequence[int],
                            target_opset: quant_opts_pb2.OpSet):
    input_saved_model_path = self.create_tempdir('input').full_path
    model = self._create_einsum_model(
        input_saved_model_path,
        equation,
        input_shape,
        weight_shape,
        bias_shape,
        activation_fn=nn_ops.relu)

    def data_gen() -> repr_dataset.RepresentativeDataset:
      for _ in range(200):
        yield {
            'input_tensor':
                ops.convert_to_tensor(
                    np.random.uniform(low=0.0, high=1.0,
                                      size=input_shape).astype('f4')),
        }

    tags = {tag_constants.SERVING}
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

    input_data = ops.convert_to_tensor(
        np.random.uniform(low=0.0, high=1.0, size=input_shape).astype('f4'))
    expected_outputs = model.einsum(input_data)
    got_outputs = converted_model.signatures['serving_default'](
        input_tensor=ops.convert_to_tensor(input_data))
    self.assertAllClose(expected_outputs, got_outputs, atol=0.0608)

    # Check the converted model in the target opset.
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            experimental_method=_ExperimentalMethod.STATIC_RANGE),
        op_set=target_opset)

    output_directory = self.create_tempdir().full_path
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
      self.assertTrue(self._contains_op(output_meta_graphdef, 'XlaDotV2'))

    new_outputs = converted_model.signatures['serving_default'](
        input_tensor=ops.convert_to_tensor(input_data))
    # The difference between TF and target path is expected to be small.
    self.assertAllClose(new_outputs, got_outputs, atol=0.0666)
    self.assertAllClose(new_outputs, expected_outputs, atol=0.057)

  @test_util.deprecated_graph_mode_only
  def test_matmul_ptq_model_with_unfreeze_constants(self):
    input_saved_model_path = self.create_tempdir('input').full_path

    # Uses large weight to exceed the constant size threshold of 64KiB
    # (specified by `kDefaultConstantSizeThresholdInBytes`) for unfreezing.
    self._create_matmul_model(
        input_shape=(1, 20),
        weight_shape=(20, 4096),
        saved_model_path=input_saved_model_path)

    repr_ds = self._create_data_generator(
        input_key='input_tensor', shape=(1, 20), num_examples=2)

    tags = {tag_constants.SERVING}
    output_directory = self.create_tempdir().full_path

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            experimental_method=_ExperimentalMethod.STATIC_RANGE),
        freeze_all_variables=quant_opts_pb2.FreezeAllVariables(enabled=False),
    )

    converted_model = quantize_model.quantize(
        input_saved_model_path, ['serving_default'],
        tags,
        output_directory,
        quantization_options,
        representative_dataset=repr_ds)

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(converted_model.signatures._signatures.keys(),
                          {'serving_default'})

    # Confirms that quantization is applied to the model.
    output_loader = saved_model_loader.SavedModelLoader(output_directory)
    output_meta_graphdef = output_loader.get_meta_graph_def_from_tags(tags)
    self.assertTrue(
        self._contains_quantized_function_call(output_meta_graphdef))

    # Tests that there are variables in the model.
    variable_node_defs = _find_variables(output_meta_graphdef.graph_def)
    self.assertLen(variable_node_defs, 1)

    # Reads the variables from the checkpoint file and matches with the
    # variables found in the graph.
    checkpoint_path = os.path.join(output_directory, 'variables', 'variables')
    var_name_and_shapes = checkpoint_utils.list_variables(checkpoint_path)

    # Checks that each variable's name and shape match.
    self.assertEqual(len(variable_node_defs), len(var_name_and_shapes))
    for var_name, shape in var_name_and_shapes:
      self.assertIn(var_name, variable_node_defs)
      self.assertEqual(
          shape,
          tensor_shape.TensorShape(
              variable_node_defs[var_name].attr['shape'].shape))

  @parameterized.named_parameters(
      ('use_constant', False),
      ('use_variable', True),
  )
  @test_util.run_v2_only
  def test_gather_model(self, use_variable):

    model = self._create_gather_model(use_variable)

    input_saved_model_path = self.create_tempdir('input').full_path
    saved_model_save.save(model, input_saved_model_path)

    tags = {tag_constants.SERVING}
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
    input_savedmodel_dir = self.create_tempdir('input').full_path
    self._create_matmul_model(
        input_shape=(1, 1024),
        weight_shape=(1024, 3),
        saved_model_path=input_savedmodel_dir)

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
    input_savedmodel_dir = self.create_tempdir('input').full_path
    self._create_matmul_model(
        input_shape=(1, 1024),
        weight_shape=(1024, 3),
        saved_model_path=input_savedmodel_dir)

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
    input_savedmodel_dir = self.create_tempdir('input').full_path
    self._create_matmul_model(
        input_shape=(1, 1024),
        weight_shape=(1024, 3),
        saved_model_path=input_savedmodel_dir)

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
    input_savedmodel_dir = self.create_tempdir('input').full_path
    self._create_matmul_model(
        input_shape=(1, 1024),
        weight_shape=(1024, 3),
        saved_model_path=input_savedmodel_dir)

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
    input_savedmodel_dir = self.create_tempdir('input').full_path
    self._create_matmul_model(
        input_shape=(1, 1024),
        weight_shape=(1024, 3),
        saved_model_path=input_savedmodel_dir)

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
    input_savedmodel_dir = self.create_tempdir('input').full_path
    self._create_matmul_model(
        input_shape=(1, 1024),
        weight_shape=(1024, 3),
        saved_model_path=input_savedmodel_dir)
    output_savedmodel_dir = self.create_tempdir().full_path

    tags = {tag_constants.SERVING}
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            experimental_method=_ExperimentalMethod.STATIC_RANGE))

    with self.assertLogs(level='WARN') as warning_logs:
      # Save the logger verbosity.
      prev_log_level = logging.get_verbosity()
      logging.set_verbosity(logging.WARN)

      try:
        converted_model = quantize_model.quantize(
            input_savedmodel_dir,
            ['serving_default'],
            tags,
            output_savedmodel_dir,
            quantization_options,
            # Put no sample into the representative dataset to make calibration
            # impossible.
            representative_dataset=[])
      finally:
        # Restore the logger verbosity.
        logging.set_verbosity(prev_log_level)

      self.assertNotEmpty(warning_logs.records)

      # Warning message should contain the function name.
      self.assertTrue(
          self._any_warning_contains('matmul', warning_logs.records))
      self.assertTrue(
          self._any_warning_contains('does not have min or max values',
                                     warning_logs.records))

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

    tags = {tag_constants.SERVING}
    output_directory = self.create_tempdir().full_path

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            experimental_method=_ExperimentalMethod.STATIC_RANGE))

    with self.assertLogs(level='WARN') as warning_logs:
      # Save the logger verbosity.
      log_level = logging.get_verbosity()
      logging.set_verbosity(logging.WARN)

      try:
        converted_model = quantize_model.quantize(
            input_saved_model_path, ['serving_default'],
            tags,
            output_directory,
            quantization_options,
            representative_dataset=data_gen())
      finally:
        # Restore the logger verbosity.
        logging.set_verbosity(log_level)

      self.assertNotEmpty(warning_logs.records)

      # Warning message should contain the function name. The uncalibrated path
      # is when the condition is true, so 'cond_true' function must be part of
      # the warning message.
      self.assertTrue(
          self._any_warning_contains('cond_true', warning_logs.records))
      self.assertFalse(
          self._any_warning_contains('cond_false', warning_logs.records))
      self.assertTrue(
          self._any_warning_contains('does not have min or max values',
                                     warning_logs.records))

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

  @test_util.deprecated_graph_mode_only
  def test_ptq_model_with_variable_tf1_saved_model_unfreeze_constants(self):
    input_saved_model_path = self.create_tempdir('input').full_path
    signature_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    tags = {tag_constants.SERVING}

    input_placeholder = self._create_and_save_tf1_conv_model(
        input_saved_model_path,
        signature_key,
        tags,
        input_key='x',
        output_key='output',
        input_shape=(1, 16, 16, 8),
        # Uses large filter to exceed the constant size threshold of 64KiB
        # (specified by `kDefaultConstantSizeThresholdInBytes`) for unfreezing.
        filter_shape=(256, 8, 8, 16),
        use_variable=True)

    signature_keys = [signature_key]
    output_directory = self.create_tempdir().full_path

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            experimental_method=_ExperimentalMethod.STATIC_RANGE),
        freeze_all_variables=quant_opts_pb2.FreezeAllVariables(enabled=False),
    )

    repr_ds = self._create_data_generator(
        input_key='x', shape=input_placeholder.shape, num_examples=2)

    converted_model = quantize_model.quantize(
        input_saved_model_path,
        signature_keys,
        tags,
        output_directory,
        quantization_options,
        representative_dataset=repr_ds)
    self.assertIsNotNone(converted_model)
    self.assertCountEqual(converted_model.signatures._signatures.keys(),
                          {'serving_default'})

    # Checks that quantization is applied.
    output_loader = saved_model_loader.SavedModelLoader(output_directory)
    output_meta_graphdef = output_loader.get_meta_graph_def_from_tags(tags)
    self.assertTrue(
        self._contains_quantized_function_call(output_meta_graphdef))

    # Tests that there are variables in the model.
    variable_node_defs = _find_variables(output_meta_graphdef.graph_def)
    self.assertLen(variable_node_defs, 1)

    # Reads the variables from the checkpoint file and matches with the
    # variables found in the graph.
    checkpoint_path = os.path.join(output_directory, 'variables', 'variables')
    var_name_and_shapes = checkpoint_utils.list_variables(checkpoint_path)

    # Checks that each variable's name and shape match.
    self.assertEqual(len(variable_node_defs), len(var_name_and_shapes))
    for var_name, shape in var_name_and_shapes:
      self.assertIn(var_name, variable_node_defs)
      self.assertEqual(
          shape,
          tensor_shape.TensorShape(
              variable_node_defs[var_name].attr['shape'].shape))

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
    with self.assertRaisesRegex(
        RuntimeError,
        "MetaGraphDef associated with tags {'serve'} could not be found"):
      quantize_model.quantize(
          input_saved_model_path,
          signature_keys,
          tags,
          output_directory,
          quantization_options,
          representative_dataset=data_gen)

  # TODO(b/244276332): Allow table initialization in TF2 eager mode.
  @test_util.deprecated_graph_mode_only
  def test_ptq_vocab_table_lookup_model(self):
    tags = {tag_constants.SERVING}
    signature_def_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    input_model_dir = self.create_tempdir('input').full_path

    # Create and save a simple model that involves a hash table.
    inputs, outputs = self._create_and_save_vocab_table_lookup_model_tf1(
        input_model_dir, tags, signature_def_key)

    # Make sure that the desired input key and output key is present.
    self.assertIn('input_vocabs', inputs.keys())
    self.assertIn('lookup', outputs.keys())

    # Representative dataset is composed of a set of vocabs for table lookup.
    repr_ds = [{
        'input_vocabs': np.array([b'hello', b'model', b'quantization'])
    } for _ in range(4)]

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            experimental_method=_ExperimentalMethod.STATIC_RANGE))

    signature_def_keys = [signature_def_key]
    output_model_dir = self.create_tempdir('output').full_path

    quantize_model.quantize(
        input_model_dir,
        signature_def_keys,
        tags,
        output_model_dir,
        quantization_options,
        representative_dataset=repr_ds)

    # Tests table lookup to make sure the table has been initialized
    # successfully.
    with session.Session(graph=ops.Graph()) as sess:
      output_meta_graph_def = saved_model_loader.load(
          sess, tags=tags, export_dir=output_model_dir)

      # The graph should contain a quantized function call (it contains a
      # single f32 matmul node).
      self.assertTrue(
          self._contains_quantized_function_call(output_meta_graph_def))
      self.assertCountEqual(output_meta_graph_def.signature_def.keys(),
                            signature_def_keys)

      signature_def = output_meta_graph_def.signature_def[signature_def_key]

      input_tensor_name = signature_def.inputs['input_vocabs'].name
      input_tensor = sess.graph.get_tensor_by_name(input_tensor_name)

      lookup_tensor_name = signature_def.outputs['lookup'].name
      lookup_tensor = sess.graph.get_tensor_by_name(lookup_tensor_name)

      lookup_val = sess.run(
          lookup_tensor,
          feed_dict={
              input_tensor: np.array([b'model', b'quantization', b'hello'])
          })

      self.assertAllClose(lookup_val, [1., 2., 0.])

  @parameterized.named_parameters(
      ('none', None, False, False, quant_opts_pb2.TF, False, 'SAME'),
      ('relu', nn_ops.relu, False, False, quant_opts_pb2.TF, False, 'SAME'),
      ('relu6', nn_ops.relu6, False, False, quant_opts_pb2.TF, False, 'SAME'),
      ('with_bias', None, True, False, quant_opts_pb2.TF, False, 'SAME'),
      ('with_bias_and_relu', nn_ops.relu, True, False, quant_opts_pb2.TF, False,
       'SAME'),
      ('with_bias_and_relu6', nn_ops.relu6, True, False, quant_opts_pb2.TF,
       False, 'SAME'),
      ('none_to_xla', None, False, False, quant_opts_pb2.XLA, False, 'SAME'),
      ('with_bias_and_relu6_to_xla', nn_ops.relu6, True, False,
       quant_opts_pb2.XLA, False, 'SAME'),
      ('with_bias_to_xla_dynamic', None, True, False, quant_opts_pb2.XLA, True,
       'SAME'),
      ('none_to_xla_padding_valid', None, False, False, quant_opts_pb2.XLA,
       False, 'VALID'),
      ('with_bias_and_relu6_to_xla_padding_valid', nn_ops.relu6, True, False,
       quant_opts_pb2.XLA, False, 'VALID'),
      ('with_bias_to_xla_dynamic_padding_valid', None, True, False,
       quant_opts_pb2.XLA, True, 'VALID'),
  )
  def test_conv3d_ptq_model(self, activation_fn: Optional[ops.Operation],
                            has_bias: bool, has_batch_norm: bool,
                            target_opset: quant_opts_pb2.OpSet,
                            input_shape_dynamic: bool, padding: str):
    input_shape = [1, 3, 4, 3, 3]
    if input_shape_dynamic:
      input_shape = [None, None, None, None, 3]

    class ConvModel(module.Module):

      def __init__(self):
        self.filters = np.random.uniform(
            low=-0.5, high=0.5, size=(2, 3, 3, 3, 2)).astype('f4')
        self.bias = np.random.uniform(low=0.0, high=0.2, size=(2)).astype('f4')

      @def_function.function(input_signature=[
          tensor_spec.TensorSpec(shape=input_shape, dtype=dtypes.float32)
      ])
      def conv3d(self, input_tensor: core.Tensor) -> Mapping[str, core.Tensor]:
        """Performs a 3D convolution operation.

        Args:
          input_tensor: Input tensor to perform convolution on.

        Returns:
          A map of: output key -> output result.
        """
        out = nn_ops.conv3d(
            input_tensor,
            self.filters,
            strides=[1, 1, 2, 1, 1],
            dilations=[1, 1, 1, 1, 1],
            padding=padding,
            data_format='NDHWC')
        if has_bias:
          out = nn_ops.bias_add(out, self.bias)
        if activation_fn is not None:
          out = activation_fn(out)
        return {'output': out}

    np.random.seed(1234)
    model = ConvModel()
    input_saved_model_path = self.create_tempdir('input').full_path
    saved_model_save.save(model, input_saved_model_path)

    repr_ds = []
    for _ in range(500):
      repr_ds.append({
          'input_tensor':
              ops.convert_to_tensor(
                  np.random.uniform(low=-0.1, high=0.2,
                                    size=(1, 3, 4, 3, 3)).astype('f4')),
      })

    signature_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    tags = {tag_constants.SERVING}

    # Check the converted model with TF opset as the baseline.
    output_directory = self.create_tempdir().full_path
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            experimental_method=_ExperimentalMethod.STATIC_RANGE),
        op_set=quant_opts_pb2.TF)

    converted_model = quantize_model.quantize(
        input_saved_model_path, [signature_key],
        tags,
        output_directory,
        quantization_options,
        representative_dataset=repr_ds)
    self.assertIsNotNone(converted_model)
    self.assertCountEqual(converted_model.signatures._signatures.keys(),
                          {signature_key})

    input_data = np.random.uniform(
        low=-0.1, high=0.2, size=(1, 3, 4, 3, 3)).astype('f4')
    expected_outputs = model.conv3d(input_data)
    got_outputs = converted_model.signatures[signature_key](
        input_tensor=ops.convert_to_tensor(input_data))
    self.assertAllClose(expected_outputs, got_outputs, atol=0.00494)

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
    converted_model = quantize_model.quantize(
        input_saved_model_path, [signature_key],
        tags,
        output_directory,
        quantization_options,
        representative_dataset=repr_ds)

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(converted_model.signatures._signatures.keys(),
                          {signature_key})
    loader = saved_model_loader.SavedModelLoader(output_directory)
    meta_graphdef = loader.get_meta_graph_def_from_tags(tags)
    if target_opset == quant_opts_pb2.XLA:
      self.assertTrue(self._contains_op(meta_graphdef, 'XlaConvV2'))

    new_outputs = converted_model.signatures[signature_key](
        input_tensor=ops.convert_to_tensor(input_data))
    # The quantized model in XLA opset is expected to have similar fidelity
    # compared to the quantized model in TF opset.
    self.assertAllClose(new_outputs, got_outputs, atol=0.00306)
    self.assertAllClose(new_outputs, expected_outputs, atol=0.00494)


class DynamicRangeQuantizationTest(quantize_model_test_base.QuantizedModelTest):
  """Test cases for dynamic range quantization.

  Tries to run all tests cases in both the graph mode (default in TF1) and the
  eager mode (default in TF2) to ensure support for when TF2 is disabled.
  """

  @parameterized.named_parameters(
      ('to_tf_per_tensor', quant_opts_pb2.TF, False),
      ('to_xla_per_tensor', quant_opts_pb2.XLA, False),
      ('to_uniform_quantized_per_tensor', quant_opts_pb2.UNIFORM_QUANTIZED,
       False),
      ('to_uniform_quantized_per_channel', quant_opts_pb2.UNIFORM_QUANTIZED,
       True),
  )
  @test_util.run_in_graph_and_eager_modes
  def test_matmul_model(self, target_opset: quant_opts_pb2.OpSet,
                        enable_per_channel_quantization: bool):

    input_saved_model_path = self.create_tempdir('input').full_path
    self._create_matmul_model(
        input_shape=(1, 1024),
        weight_shape=(1024, 3),
        saved_model_path=input_saved_model_path)

    tags = {tag_constants.SERVING}
    output_directory = self.create_tempdir().full_path

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            experimental_method=_ExperimentalMethod.DYNAMIC_RANGE),
        op_set=target_opset,
        enable_per_channel_quantization=enable_per_channel_quantization)

    converted_model = quantize_model.quantize(input_saved_model_path,
                                              ['serving_default'], tags,
                                              output_directory,
                                              quantization_options)
    self.assertIsNotNone(converted_model)
    self.assertCountEqual(converted_model.signatures._signatures.keys(),
                          {'serving_default'})

    output_loader = saved_model_loader.SavedModelLoader(output_directory)
    output_meta_graphdef = output_loader.get_meta_graph_def_from_tags(tags)

    if target_opset == quant_opts_pb2.UNIFORM_QUANTIZED:
      # _contains_op for UQ ops also checks _contains_quantized_function_call.
      self.assertTrue(
          self._contains_op(output_meta_graphdef, 'UniformQuantizedDotHybrid'))
      self.assertFalse(self._contains_op(output_meta_graphdef, 'MatMul'))
      if enable_per_channel_quantization:
        quantized_axis_attr = attr_value_pb2.AttrValue(i=-1)
        self.assertTrue(
            self._contains_op(output_meta_graphdef, 'UniformQuantizedDotHybrid',
                              'rhs_quantization_axis', quantized_axis_attr))
    elif target_opset == quant_opts_pb2.XLA:
      self.assertTrue(self._contains_op(output_meta_graphdef, 'XlaDotV2'))
      self.assertFalse(self._contains_op(output_meta_graphdef, 'MatMul'))
    else:
      self.assertTrue(
          self._contains_quantized_function_call(output_meta_graphdef))
      self.assertTrue(self._contains_op(output_meta_graphdef, 'MatMul'))

  @parameterized.named_parameters(
      ('to_tf_per_tensor', quant_opts_pb2.TF, False),
      ('to_xla_per_tensor', quant_opts_pb2.XLA, False),
      ('to_uniform_quantized_per_tensor', quant_opts_pb2.UNIFORM_QUANTIZED,
       False),
      ('to_uniform_quantized_per_channel', quant_opts_pb2.UNIFORM_QUANTIZED,
       True),
  )
  @test_util.run_in_graph_and_eager_modes
  def test_conv_model(self, target_opset: quant_opts_pb2.OpSet,
                      enable_per_channel_quantization: bool):
    filter_shape = (2, 3, 512, 2)

    model = self._create_conv2d_model(
        input_shape=(1, 3, 4, 512),
        filter_shape=filter_shape,
        has_bias=True,
        has_batch_norm=True,
        activation_fn=nn_ops.relu6)

    input_saved_model_path = self.create_tempdir('input').full_path
    saved_model_save.save(model, input_saved_model_path)

    tags = {tag_constants.SERVING}
    output_directory = self.create_tempdir().full_path

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            experimental_method=_ExperimentalMethod.DYNAMIC_RANGE),
        op_set=target_opset,
        enable_per_channel_quantization=enable_per_channel_quantization)

    converted_model = quantize_model.quantize(input_saved_model_path,
                                              ['serving_default'], tags,
                                              output_directory,
                                              quantization_options)

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(converted_model.signatures._signatures.keys(),
                          {'serving_default'})

    output_loader = saved_model_loader.SavedModelLoader(output_directory)
    output_meta_graphdef = output_loader.get_meta_graph_def_from_tags(tags)

    if enable_per_channel_quantization:
      quantized_axis = 3
      quantized_axis_attr = attr_value_pb2.AttrValue(i=quantized_axis)
      quantized_dim_size_attr = attr_value_pb2.AttrValue(
          list=attr_value_pb2.AttrValue.ListValue(shape=[
              tensor_shape_pb2.TensorShapeProto(dim=[
                  tensor_shape_pb2.TensorShapeProto.Dim(
                      size=filter_shape[quantized_axis])
              ])
          ]))

    if target_opset == quant_opts_pb2.UNIFORM_QUANTIZED:
      # _contains_op for UQ ops also checks _contains_quantized_function_call.
      self.assertTrue(
          self._contains_op(output_meta_graphdef,
                            'UniformQuantizedConvolutionHybrid'))
      self.assertFalse(self._contains_op(output_meta_graphdef, 'Conv2D'))
      if enable_per_channel_quantization:
        self.assertTrue(
            self._contains_op(output_meta_graphdef,
                              'UniformQuantizedConvolutionHybrid',
                              'rhs_quantization_axis', quantized_axis_attr))
        self.assertTrue(
            self._contains_op(output_meta_graphdef, 'Const', '_output_shapes',
                              quantized_dim_size_attr))
    elif target_opset == quant_opts_pb2.XLA:
      self.assertTrue(self._contains_op(output_meta_graphdef, 'XlaConvV2'))
      self.assertFalse(self._contains_op(output_meta_graphdef, 'Conv2D'))
    else:
      self.assertTrue(
          self._contains_quantized_function_call(output_meta_graphdef))
      self.assertTrue(self._contains_op(output_meta_graphdef, 'Conv2D'))

  @parameterized.named_parameters(
      ('to_tf_per_tensor', quant_opts_pb2.TF, False),
      ('to_xla_per_tensor', quant_opts_pb2.XLA, False),
      ('to_uniform_quantized_per_tensor', quant_opts_pb2.UNIFORM_QUANTIZED,
       False),
      ('to_uniform_quantized_per_channel', quant_opts_pb2.UNIFORM_QUANTIZED,
       True),
  )
  @test_util.run_in_graph_and_eager_modes
  def test_depthwise_conv_model(
      self,
      target_opset: quant_opts_pb2.OpSet,
      enable_per_channel_quantization: bool,
  ):

    filter_shape = (2, 3, 1024, 2)
    strides = (1, 2, 2, 1)

    model = self._create_depthwise_conv2d_model(
        input_shape=(1, 3, 4, 1024), filter_shape=filter_shape, strides=strides)

    input_saved_model_path = self.create_tempdir('input').full_path
    saved_model_save.save(model, input_saved_model_path)

    tags = [tag_constants.SERVING]
    output_directory = self.create_tempdir().full_path

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            experimental_method=_ExperimentalMethod.DYNAMIC_RANGE),
        op_set=target_opset,
        enable_per_channel_quantization=enable_per_channel_quantization)

    converted_model = quantize_model.quantize(input_saved_model_path,
                                              ['serving_default'], tags,
                                              output_directory,
                                              quantization_options)

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(converted_model.signatures._signatures.keys(),
                          {'serving_default'})

    output_loader = saved_model_loader.SavedModelLoader(output_directory)
    output_meta_graphdef = output_loader.get_meta_graph_def_from_tags(tags)

    # Uniform Quantized op takes only the first and the second values for
    # strides.
    strides_to_check = (
        strides[1], strides[2]
    ) if target_opset == quant_opts_pb2.UNIFORM_QUANTIZED else strides
    strides_attr = attr_value_pb2.AttrValue(
        list=attr_value_pb2.AttrValue.ListValue(i=strides_to_check))

    if enable_per_channel_quantization:
      quantized_axis_attr = attr_value_pb2.AttrValue(i=3)
      quantized_dim_size_attr = attr_value_pb2.AttrValue(
          list=attr_value_pb2.AttrValue.ListValue(shape=[
              tensor_shape_pb2.TensorShapeProto(dim=[
                  tensor_shape_pb2.TensorShapeProto.Dim(size=filter_shape[2] *
                                                        filter_shape[3])
              ])
          ]))

    if target_opset == quant_opts_pb2.UNIFORM_QUANTIZED:
      # _contains_op for UQ ops also checks _contains_quantized_function_call.
      self.assertTrue(
          self._contains_op(output_meta_graphdef,
                            'UniformQuantizedConvolutionHybrid',
                            'window_strides', strides_attr))
      self.assertFalse(
          self._contains_op(output_meta_graphdef, 'DepthwiseConv2dNative'))
      if enable_per_channel_quantization:
        self.assertTrue(
            self._contains_op(output_meta_graphdef,
                              'UniformQuantizedConvolutionHybrid',
                              'rhs_quantization_axis', quantized_axis_attr))
        self.assertTrue(
            self._contains_op(output_meta_graphdef, 'Const', '_output_shapes',
                              quantized_dim_size_attr))
    elif target_opset == quant_opts_pb2.XLA:
      self.assertTrue(self._contains_op(output_meta_graphdef, 'XlaConvV2'))
      self.assertFalse(
          self._contains_op(output_meta_graphdef, 'DepthwiseConv2dNative'))
    else:
      self.assertTrue(
          self._contains_quantized_function_call(output_meta_graphdef))
      self.assertTrue(
          self._contains_op(output_meta_graphdef, 'DepthwiseConv2dNative',
                            'strides', strides_attr))

  @parameterized.named_parameters(
      ('use_constant', False),
      ('use_variable', True),
  )
  @test_util.run_v2_only
  def test_gather_model(self, use_variable):
    model = self._create_gather_model(use_variable)
    input_saved_model_path = self.create_tempdir('input').full_path
    saved_model_save.save(model, input_saved_model_path)

    tags = {tag_constants.SERVING}
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
    input_saved_model_path = self.create_tempdir('input').full_path
    self._create_matmul_model(
        input_shape=(1, 1024),
        weight_shape=(1024, 3),
        saved_model_path=input_saved_model_path)

    tags = {tag_constants.SERVING}
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
    input_saved_model_path = self.create_tempdir('input').full_path
    self._create_matmul_model(
        input_shape=(1, 1024),
        weight_shape=(1024, 3),
        saved_model_path=input_saved_model_path)
    tags = {tag_constants.SERVING}

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
    input_saved_model_path = self.create_tempdir('input').full_path
    self._create_matmul_model(
        input_shape=(1, 1024),
        weight_shape=(1024, 3),
        saved_model_path=input_saved_model_path)
    tags = {tag_constants.SERVING}

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

  # TODO(b/244276332): Allow table initialization in TF2 eager mode.
  @test_util.deprecated_graph_mode_only
  def test_table_initialized_when_model_has_table_tf1(self):
    tags = {tag_constants.SERVING}
    signature_def_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    input_model_dir = self.create_tempdir('input').full_path

    # Create and save a simple model that involves a hash table.
    inputs, outputs = self._create_and_save_vocab_table_lookup_model_tf1(
        input_model_dir, tags, signature_def_key)

    # Make sure that the desired input key and output key is present.
    self.assertIn('input_vocabs', inputs.keys())
    self.assertIn('lookup', outputs.keys())

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            experimental_method=_ExperimentalMethod.DYNAMIC_RANGE))

    signature_def_keys = [signature_def_key]
    output_model_dir = self.create_tempdir('output').full_path

    quantize_model.quantize(input_model_dir, signature_def_keys, tags,
                            output_model_dir, quantization_options)

    # Tests table lookup to make sure the table has been initialized
    # successfully.
    with session.Session(graph=ops.Graph()) as sess:
      output_meta_graph_def = saved_model_loader.load(
          sess, tags=tags, export_dir=output_model_dir)

      self.assertCountEqual(output_meta_graph_def.signature_def.keys(),
                            signature_def_keys)

      signature_def = output_meta_graph_def.signature_def[signature_def_key]

      input_tensor_name = signature_def.inputs['input_vocabs'].name
      input_tensor = sess.graph.get_tensor_by_name(input_tensor_name)

      lookup_tensor_name = signature_def.outputs['lookup'].name
      lookup_tensor = sess.graph.get_tensor_by_name(lookup_tensor_name)

      lookup_val = sess.run(
          lookup_tensor,
          feed_dict={
              input_tensor: np.array([b'model', b'quantization', b'hello'])
          })

      self.assertAllClose(lookup_val, [1., 2., 0.])


class WeightOnlyQuantizationTest(quantize_model_test_base.QuantizedModelTest):
  """Test cases for weight-only quantization.

  Run all tests cases in both the graph mode (default in TF1) and the eager mode
  (default in TF2) to ensure support for when TF2 is disabled.
  """

  @parameterized.named_parameters(
      ('to_tf_per_tensor', quant_opts_pb2.TF, False),
      ('to_xla_per_tensor', quant_opts_pb2.XLA, False))
  @test_util.run_in_graph_and_eager_modes
  def test_matmul_model(self, target_opset: quant_opts_pb2.OpSet,
                        enable_per_channel_quantization: bool):

    input_shape = (1, 512)
    input_saved_model_path = self.create_tempdir('input').full_path

    self._create_matmul_model(
        input_shape=input_shape,
        weight_shape=(512, 2),
        saved_model_path=input_saved_model_path)

    tags = {tag_constants.SERVING}
    output_directory = self.create_tempdir().full_path
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            experimental_method=_ExperimentalMethod.WEIGHT_ONLY),
        op_set=target_opset,
        enable_per_channel_quantization=enable_per_channel_quantization)

    converted_model = quantize_model.quantize(input_saved_model_path,
                                              ['serving_default'], tags,
                                              output_directory,
                                              quantization_options)
    self.assertIsNotNone(converted_model)
    self.assertCountEqual(converted_model.signatures._signatures.keys(),
                          {'serving_default'})

    output_loader = saved_model_loader.SavedModelLoader(output_directory)
    output_meta_graphdef = output_loader.get_meta_graph_def_from_tags(tags)

    self.assertTrue(self._contains_op(output_meta_graphdef, 'MatMul'))
    # Due to other meta data, the compression is not exactly 1/4.
    self.assertLessEqual(
        get_dir_size(output_directory) / get_dir_size(input_saved_model_path),
        1 / 3)

  @parameterized.named_parameters(
      ('to_tf_per_tensor', quant_opts_pb2.TF, False),
      ('to_xla_per_tensor', quant_opts_pb2.XLA, False))
  @test_util.run_in_graph_and_eager_modes
  def test_conv_model(self, target_opset: quant_opts_pb2.OpSet,
                      enable_per_channel_quantization: bool):

    input_saved_model_path = self.create_tempdir('input').full_path

    model = self._create_conv2d_model(
        input_shape=(1, 3, 4, 512),
        filter_shape=(2, 3, 512, 2),
        has_bias=False,
        has_batch_norm=False,
        activation_fn=nn_ops.relu6)
    saved_model_save.save(model, input_saved_model_path)

    tags = {tag_constants.SERVING}
    output_directory = self.create_tempdir().full_path

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            experimental_method=_ExperimentalMethod.WEIGHT_ONLY),
        op_set=target_opset,
        enable_per_channel_quantization=enable_per_channel_quantization)

    converted_model = quantize_model.quantize(input_saved_model_path,
                                              ['serving_default'], tags,
                                              output_directory,
                                              quantization_options)

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(converted_model.signatures._signatures.keys(),
                          {'serving_default'})

    output_loader = saved_model_loader.SavedModelLoader(output_directory)
    output_meta_graphdef = output_loader.get_meta_graph_def_from_tags(tags)

    self.assertTrue(self._contains_op(output_meta_graphdef, 'Conv2D'))
    # Due to other meta data, the compression is not exactly 1/4.
    self.assertLessEqual(
        get_dir_size(output_directory) / get_dir_size(input_saved_model_path),
        1 / 3)

  @parameterized.named_parameters(
      ('to_tf_per_tensor', quant_opts_pb2.TF, False),
      ('to_xla_per_tensor', quant_opts_pb2.XLA, False))
  @test_util.run_in_graph_and_eager_modes
  def test_depthwise_conv_model(
      self,
      target_opset: quant_opts_pb2.OpSet,
      enable_per_channel_quantization: bool,
  ):

    filter_shape = (2, 3, 512, 2)
    strides = (1, 2, 2, 1)

    model = self._create_depthwise_conv2d_model(
        input_shape=(1, 3, 4, 512), filter_shape=filter_shape, strides=strides)

    input_saved_model_path = self.create_tempdir('input').full_path
    saved_model_save.save(model, input_saved_model_path)

    tags = {tag_constants.SERVING}
    output_directory = self.create_tempdir().full_path

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            experimental_method=_ExperimentalMethod.WEIGHT_ONLY),
        op_set=target_opset,
        enable_per_channel_quantization=enable_per_channel_quantization)

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
        self._contains_op(output_meta_graphdef, 'DepthwiseConv2dNative'))
    # Due to other meta data, the compression is not exactly 1/4.
    self.assertLessEqual(
        get_dir_size(output_directory) / get_dir_size(input_saved_model_path),
        1 / 3)


if __name__ == '__main__':
  test.main()
