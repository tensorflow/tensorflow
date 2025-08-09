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
# TODO(b/264234648): Refactor and cleanup this file.
import itertools
import os
from typing import Mapping, Optional, Sequence, Tuple, Union

from absl.testing import parameterized
import numpy as np
import tensorflow  # pylint: disable=unused-import

from tensorflow.compiler.mlir.quantization.common.python import testing
from tensorflow.compiler.mlir.quantization.stablehlo import quantization_config_pb2 as stablehlo_quant_config_pb2
from tensorflow.compiler.mlir.quantization.tensorflow import quantization_options_pb2 as quant_opts_pb2
from tensorflow.compiler.mlir.quantization.tensorflow.python import quantize_model
from tensorflow.compiler.mlir.quantization.tensorflow.python import representative_dataset as repr_dataset
from tensorflow.compiler.mlir.quantization.tensorflow.python import save_model
from tensorflow.compiler.mlir.quantization.tensorflow.python.integration_test import quantize_model_test_base
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.core.framework import tensor_pb2
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.python.client import session
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.lib.io import tf_record
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
from tensorflow.python.saved_model import save_options
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils_impl
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.training import checkpoint_utils
from tensorflow.python.types import core


# Type aliases for quantization method protobuf enums.
_PresetMethod = quant_opts_pb2.QuantizationMethod.PresetMethod
_CalibrationMethod = (
    stablehlo_quant_config_pb2.CalibrationOptions.CalibrationMethod
)

_QuantizationComponent = (
    quant_opts_pb2.QuantizationComponentSpec.QuantizationComponent
)
_TensorType = quant_opts_pb2.QuantizationComponentSpec.TensorType

_TensorShape = Sequence[Union[int, None]]

_PER_CHANNEL_QUANTIZED_OPS = (
    'UniformQuantizedConvolution',
    'UniformQuantizedConvolutionHybrid',
    'UniformQuantizedDotHybrid',
)

_DebuggerConfig = stablehlo_quant_config_pb2.DebuggerConfig

# Lists of ops whose channel dimension should be changed if per_channel
# quantization is enabled. Respectively refers to (scale, zero_point).
_SUFFIXES = ('/filter1', '/filter2')
_PER_CHANNEL_OP_NAMES = (
    f'{op}{suffix}'
    for op, suffix in itertools.product(_PER_CHANNEL_QUANTIZED_OPS, _SUFFIXES)
)


def _is_variable(node_def: node_def_pb2.NodeDef) -> bool:
  """Determines whether `node_def` is a variable node.

  Args:
    node_def: `NodeDef` to test whether it is a variable or not.

  Returns:
    Returns True if it is a variable.
  """
  return node_def.op == 'VarHandleOp'


def _find_variables(
    graph_def: graph_pb2.GraphDef,
) -> Mapping[str, node_def_pb2.NodeDef]:
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


class MultipleSignatureModel(module.Module):
  """A model with 2 signatures.

  Used to test where the quantizer has to handle multiple signatures.
  """

  def __init__(self):
    self.matmul_filters = random_ops.random_uniform(
        shape=(4, 3), minval=-1.0, maxval=1.0
    )
    self.conv_filters = np.random.uniform(
        low=-10, high=10, size=(2, 3, 3, 2)
    ).astype('f4')

  @def_function.function(
      input_signature=[
          tensor_spec.TensorSpec(shape=[1, 4], dtype=dtypes.float32)
      ]
  )
  def matmul(self, matmul_input: core.Tensor) -> Mapping[str, core.Tensor]:
    """Performs a matrix multiplication.

    Args:
      matmul_input: Input tensor to matmul with the filter.

    Returns:
      A map of: output key -> output result.
    """
    out = math_ops.matmul(matmul_input, self.matmul_filters)

    return {'output': out}

  @def_function.function(
      input_signature=[
          tensor_spec.TensorSpec(shape=(1, 3, 4, 3), dtype=dtypes.float32)
      ]
  )
  def conv(self, conv_input: core.Tensor) -> Mapping[str, core.Tensor]:
    """Performs a 2D convolution operation.

    Args:
      conv_input: Input tensor to perform convolution on.

    Returns:
      A map of: output key -> output result.
    """
    out = nn_ops.conv2d(
        conv_input,
        self.conv_filters,
        strides=[1, 1, 2, 1],
        dilations=[1, 1, 1, 1],
        padding='SAME',
        data_format='NHWC',
    )

    return {'output': out}


# TODO(b/280208261): Add unit tests for comparing unquantized and
# quantized results
@test_util.run_all_in_graph_and_eager_modes
class QuantizationOptionsTest(quantize_model_test_base.QuantizedModelTest):
  """Test cases regarding the use of QuantizationOptions proto.

  Run all tests cases in both the graph mode (default in TF1) and the eager mode
  (default in TF2) to ensure support for when TF2 is disabled.
  """

  class SimpleModel(module.Module):

    def __init__(self):
      self.filters = np.random.uniform(low=-1.0, high=1.0, size=(4, 3)).astype(
          'f4'
      )

    @def_function.function(
        input_signature=[
            tensor_spec.TensorSpec(shape=[1, 4], dtype=dtypes.float32)
        ]
    )
    def __call__(self, input_tensor: core.Tensor) -> Mapping[str, core.Tensor]:
      """Performs a matrix multiplication.

      Args:
        input_tensor: Input tensor to matmul with the filter.

      Returns:
        A map of: output key -> output result.
      """

      out = math_ops.matmul(input_tensor, self.filters)
      return {'output': out}

  def _simple_model_data_gen(self) -> repr_dataset.RepresentativeDataset:
    """Creates an interable of representative samples.

    Yields:
      Representative samples, which is basically a mapping of: input key ->
      input value.
    """
    for _ in range(8):
      yield {
          'input_tensor': ops.convert_to_tensor(
              np.random.uniform(low=0, high=150, size=(1, 4)).astype('f4')
          ),
      }

  def test_static_range_quantization_by_default(self):
    model = self.SimpleModel()

    saved_model_save.save(model, self._input_saved_model_path)

    # Use default QuantizationOptions.
    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        representative_dataset=self._simple_model_data_gen(),
    )

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )

    # Indirectly prove that it is performing a static-range quantization
    # by checking that it complains about representative_dataset when it is
    # not provided.
    with self.assertRaisesRegex(ValueError, 'representative_dataset'):
      quantize_model.quantize(self._input_saved_model_path)

  def test_method_unspecified_raises_value_error(self):
    model = self.SimpleModel()

    saved_model_save.save(model, self._input_saved_model_path)

    options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_UNSPECIFIED
        )
    )

    with self.assertRaises(ValueError):
      quantize_model.quantize(
          self._input_saved_model_path, quantization_options=options
      )

  def test_predefined_method_component_spec(self):
    options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        )
    )
    quantize_model._populate_quantization_component_spec(
        options.quantization_method
    )

    # Quantize activation, weight and bias for static range quantization.
    self.assertLen(options.quantization_method.quantization_component_specs, 3)

  def test_invalid_spec_raise_value_error(self):
    options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            quantization_component_specs=[
                quant_opts_pb2.QuantizationComponentSpec(
                    quantization_component=(
                        _QuantizationComponent.COMPONENT_ACTIVATION
                    ),
                    tensor_type=_TensorType.TENSORTYPE_INT_4,
                )
            ]
        )
    )

    with self.assertRaises(ValueError):
      # Activation 4bit is not a valid configuration.
      quantize_model._populate_quantization_component_spec(
          options.quantization_method
      )

  def test_invalid_method_raises_value_error(self):
    model = self.SimpleModel()

    saved_model_save.save(model, self._input_saved_model_path)

    # Set an invalid value of -1 to QuantizationMethod.preset_method.
    options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(preset_method=-1)
    )

    with self.assertRaises(ValueError):
      quantize_model.quantize(
          self._input_saved_model_path, quantization_options=options
      )

  def test_drq_per_channel_for_non_uniform_opset_raises_value_error(
      self,
  ):
    model = self.SimpleModel()

    saved_model_save.save(model, self._input_saved_model_path)

    options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_DYNAMIC_RANGE_INT8
        ),
        op_set=quant_opts_pb2.TF,
        enable_per_channel_quantization=True,
    )

    with self.assertRaises(ValueError):
      quantize_model.quantize(
          self._input_saved_model_path, quantization_options=options
      )

  def test_force_graph_mode_calibration(self):
    model = self.SimpleModel()

    saved_model_save.save(model, self._input_saved_model_path)

    options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        op_set=quant_opts_pb2.TF,
        force_graph_mode_calibration=True,
    )

    with self.assertLogs(level='INFO') as info_logs:
      # Save the logger verbosity.
      prev_log_level = logging.get_verbosity()
      logging.set_verbosity(logging.INFO)

      try:
        quantize_model.quantize(
            self._input_saved_model_path,
            quantization_options=options,
            representative_dataset=self._simple_model_data_gen(),
        )
      finally:
        # Restore the logger verbosity.
        logging.set_verbosity(prev_log_level)

      self.assertNotEmpty(info_logs.records)
      self.assertTrue(
          self._any_log_contains(
              'Calibration step is executed in graph mode.',
              info_logs.records,
          )
      )


class TensorNamePreservationTest(quantize_model_test_base.QuantizedModelTest):

  def test_preserving_input_output_tensor_names(self):
    class MultiSignatureModel(module.Module):

      @def_function.function(
          input_signature=[
              tensor_spec.TensorSpec(
                  name='input', shape=[32], dtype=dtypes.float32
              ),
          ]
      )
      def multiple_output_ops(
          self, input_tensor: core.Tensor
      ) -> Mapping[str, core.Tensor]:
        k = array_ops.constant(4, dtype=dtypes.int32)
        values, indices = nn_ops.top_k(input_tensor, k, name='TopK')
        adj_values = values + 2
        return {'indices': indices, 'adj_values': adj_values, 'values': values}

      @def_function.function(
          input_signature=[
              tensor_spec.TensorSpec(
                  name='input', shape=[32], dtype=dtypes.float32
              ),
          ]
      )
      def duplicate_outputs(
          self, input_tensor: core.Tensor
      ) -> Mapping[str, core.Tensor]:
        q_input = array_ops.fake_quant_with_min_max_args(
            input_tensor, min=-0.1, max=0.2, num_bits=8, narrow_range=False
        )
        adj_values = q_input + 2
        return {'adj_values_1': adj_values, 'adj_values_2': adj_values}

      @def_function.function(
          input_signature=[
              tensor_spec.TensorSpec(
                  name='input', shape=[32], dtype=dtypes.float32
              ),
          ]
      )
      def return_higher_index_only(
          self, input_tensor: core.Tensor
      ) -> Mapping[str, core.Tensor]:
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
    saved_model_save.save(
        model, self._input_saved_model_path, signatures=signatures
    )

    tags = {tag_constants.SERVING}
    original_signature_map = save_model.get_signatures_from_saved_model(
        self._input_saved_model_path,
        signature_keys=signatures.keys(),
        tags=tags,
    )

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=signatures.keys(),
        op_set=quant_opts_pb2.TF,
    )
    quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
    )
    converted_signature_map = save_model.get_signatures_from_saved_model(
        self._output_saved_model_path,
        signature_keys=signatures.keys(),
        tags=tags,
    )

    # The original and converted model should have the same signature map.
    self.assertAllInSet(
        list(original_signature_map.keys()), set(signatures.keys())
    )
    self.assertDictEqual(original_signature_map, converted_signature_map)

  def test_duplicated_tensor_name(self):
    with session.Session(graph=ops.Graph()) as sess:
      input_tensor = array_ops.placeholder(
          dtypes.float32, shape=[], name='input'
      )
      q_input = array_ops.fake_quant_with_min_max_args(
          input_tensor, min=-0.1, max=0.2, num_bits=8, narrow_range=False
      )
      sqrt = math_ops.sqrt(q_input, name='sqrt')
      identity = array_ops.identity(sqrt, name='output')

      input_map = {'input': input_tensor}
      output_map = {'sqrt': identity}
      signature = signature_def_utils_impl.predict_signature_def(
          inputs=input_map, outputs=output_map
      )
      signature_map = {'main': signature}

      tags = {tag_constants.SERVING}
      v1_builder = builder.SavedModelBuilder(self._input_saved_model_path)
      v1_builder.add_meta_graph_and_variables(
          sess, tags, signature_def_map=signature_map
      )
      v1_builder.save()

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=signature_map.keys(),
        op_set=quant_opts_pb2.TF,
    )
    quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
    )
    converted_signature_map = save_model.get_signatures_from_saved_model(
        self._output_saved_model_path,
        signature_keys=signature_map.keys(),
        tags=tags,
    )
    # The original and converted model should have the same signature map.
    self.assertDictEqual(signature_map, converted_signature_map)


class StaticRangeQuantizationTest(quantize_model_test_base.QuantizedModelTest):

  @parameterized.parameters(
      testing.parameter_combinations([{
          'shapes': [
              ([3, 3], [3, 3]),
              ([3, None], [None, 3]),
              ([None, None], [None, None]),
              ([4, 3, 3], [4, 3, 3]),
              ([4, 3, None], [4, None, 3]),
              ([None, None, None], [None, None, None]),
          ],
          'activation_fn': [None, nn_ops.relu, nn_ops.relu6],
          'has_bias': [True, False],
          'use_kernel': [True, False],
      }])
  )
  @test_util.run_in_graph_and_eager_modes
  def test_qat_matmul_model(
      self,
      shapes: Sequence[Tuple[_TensorShape, _TensorShape]],
      activation_fn: Optional[ops.Operation],
      has_bias: bool,
      use_kernel: bool,
  ):
    n = 5
    x_shape = [v if v is not None else n for v in shapes[0]]
    y_shape = [v if v is not None else n for v in shapes[1]]

    class MatmulModel(module.Module):

      def __init__(self, bias: Optional[core.Tensor]):
        self._bias = bias
        self._kernel = np.random.uniform(size=y_shape).astype('f4')
        self._min = (-0.8, -0.8, -0.9)
        self._max = (0.9, 0.9, 1.0)

      @def_function.function(
          input_signature=[
              tensor_spec.TensorSpec(
                  name='x', shape=shapes[0], dtype=dtypes.float32
              )
          ]
      )
      def matmul_with_kernel(self, x: core.Tensor) -> Mapping[str, core.Tensor]:
        return self._matmul(x, self._kernel)

      @def_function.function(
          input_signature=[
              tensor_spec.TensorSpec(
                  name='x', shape=shapes[0], dtype=dtypes.float32
              ),
              tensor_spec.TensorSpec(
                  name='y', shape=shapes[1], dtype=dtypes.float32
              ),
          ]
      )
      def matmul_without_kernel(
          self, x: core.Tensor, y: core.Tensor
      ) -> Mapping[str, core.Tensor]:
        return self._matmul(x, y)

      def _matmul(self, x, y):
        x = array_ops.fake_quant_with_min_max_vars(
            x,
            min=ops.convert_to_tensor(self._min[0]),
            max=ops.convert_to_tensor(self._max[0]),
            num_bits=8,
            narrow_range=False,
        )
        y = array_ops.fake_quant_with_min_max_vars(
            y,
            min=ops.convert_to_tensor(self._min[1]),
            max=ops.convert_to_tensor(self._max[1]),
            num_bits=8,
            narrow_range=False,
        )

        out = math_ops.matmul(x, y)
        if self._bias is not None:
          out = nn_ops.bias_add(out, self._bias)
        if activation_fn is not None:
          out = activation_fn(out)
        out = array_ops.fake_quant_with_min_max_vars(
            out,
            min=ops.convert_to_tensor(self._min[2]),
            max=ops.convert_to_tensor(self._max[2]),
            num_bits=8,
            narrow_range=False,
        )
        return {'output': out}

    bias = None
    if has_bias:
      bias_shape = shapes[1][-1]
      if bias_shape is not None:
        bias = array_ops.constant(
            np.random.uniform(size=[shapes[1][-1]]), dtype=dtypes.float32
        )
    model = MatmulModel(bias)
    x = array_ops.constant(
        np.random.uniform(size=x_shape), dtype=dtypes.float32
    )
    y = array_ops.constant(
        np.random.uniform(size=y_shape), dtype=dtypes.float32
    )
    if use_kernel:
      model.matmul = model.matmul_with_kernel
      model_inputs = {'x': x}
    else:
      model.matmul = model.matmul_without_kernel
      model_inputs = {'x': x, 'y': y}

    saved_model_save.save(
        model, self._input_saved_model_path, signatures=model.matmul
    )

    signature_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    tags = {tag_constants.SERVING}

    # Check the converted model with TF opset as the baseline.
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=[signature_key],
        op_set=quant_opts_pb2.TF,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
    )
    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {signature_key}
    )

    expected_outputs = model.matmul(**model_inputs)
    got_outputs = converted_model.signatures[signature_key](**model_inputs)
    self.assertAllClose(expected_outputs, got_outputs, atol=1e-1)

    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def
    self.assertTrue(self._contains_quantized_function_call(output_graphdef))

    # Check the converted model in the XLA opset.
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=[signature_key],
        op_set=quant_opts_pb2.XLA,
        enable_two_input_tensors=not use_kernel,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path_2,
        quantization_options,
    )

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {signature_key}
    )
    loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path_2
    )
    graphdef = loader.get_meta_graph_def_from_tags(tags).graph_def
    self.assertTrue(self._contains_op(graphdef, 'XlaDotV2'))

    new_outputs = converted_model.signatures[signature_key](**model_inputs)

    # The difference between TF and XLA path is expected to be small (smaller
    # or equal to 1 in the quantized domain).
    self.assertAllClose(new_outputs, expected_outputs, atol=1e-1)

  @parameterized.parameters(
      testing.parameter_combinations([{
          'activation_fn': [None, nn_ops.relu, nn_ops.relu6],
          'has_bias': [True, False],
          'has_batch_norm': [True, False],
          'target_opset': [quant_opts_pb2.XLA],
      }])
  )
  @test_util.run_in_graph_and_eager_modes
  def test_qat_conv_model(
      self,
      activation_fn: Optional[ops.Operation],
      has_bias: bool,
      has_batch_norm: bool,
      target_opset: quant_opts_pb2.OpSet,
  ):
    class ConvModel(module.Module):

      def __init__(self):
        self.filter_value = np.random.uniform(
            low=-0.5, high=0.5, size=(2, 3, 3, 2)
        ).astype('f4')

      @def_function.function(
          input_signature=[
              tensor_spec.TensorSpec(
                  name='input', shape=[1, 3, 4, 3], dtype=dtypes.float32
              ),
          ]
      )
      def conv(self, input_tensor: core.Tensor) -> Mapping[str, core.Tensor]:
        """Performs a 2D convolution operation.

        Args:
          input_tensor: Input tensor to perform convolution on.

        Returns:
          A map of: output key -> output result.
        """
        q_input = array_ops.fake_quant_with_min_max_args(
            input_tensor, min=-0.1, max=0.2, num_bits=8, narrow_range=False
        )
        filter_tensor = ops.convert_to_tensor(self.filter_value)
        filter_min = array_ops.identity(
            array_ops.constant([-0.5, -0.5], dtype=dtypes.float32)
        )
        filter_max = array_ops.identity(
            array_ops.constant([0.5, 0.5], dtype=dtypes.float32)
        )
        q_filter = array_ops.fake_quant_with_min_max_vars_per_channel(
            filter_tensor, filter_min, filter_max, num_bits=8, narrow_range=True
        )
        bias = array_ops.constant([0.1, 0.2], dtype=dtypes.float32)
        scale, offset = [1.0] * 2, [0.5] * 2
        mean, variance = scale, offset
        out = nn_ops.conv2d(
            q_input,
            q_filter,
            strides=[1, 1, 2, 1],
            dilations=[1, 1, 1, 1],
            padding='SAME',
            data_format='NHWC',
            name='sample/conv2d',
        )
        if has_bias:
          out = nn_ops.bias_add(out, bias, data_format='NHWC')
        if activation_fn is not None:
          # The accuracy is not good when having FusedBatchNorm without
          # activation in this test.
          if has_batch_norm:
            # Fusing is supported for non-training case.
            out, _, _, _, _, _ = nn_ops.fused_batch_norm_v3(
                out, scale, offset, mean, variance, is_training=False
            )
          out = activation_fn(out)
        out_min = array_ops.constant([-0.18, -0.32], dtype=dtypes.float32)
        out_max = array_ops.constant([0.5, 0.5], dtype=dtypes.float32)
        q_out = array_ops.fake_quant_with_min_max_vars_per_channel(
            out, min=out_min, max=out_max, num_bits=8, narrow_range=True
        )
        return {'output': q_out}

    model = ConvModel()
    saved_model_save.save(model, self._input_saved_model_path)

    signature_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    tags = {tag_constants.SERVING}

    # Check the converted model with TF opset as the baseline.
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=[signature_key],
        op_set=quant_opts_pb2.TF,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
    )
    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {signature_key}
    )

    input_data = np.random.uniform(
        low=-0.1, high=0.2, size=(1, 3, 4, 3)
    ).astype('f4')
    expected_outputs = model.conv(input_data)
    got_outputs = converted_model.signatures[signature_key](
        input=ops.convert_to_tensor(input_data)
    )
    self.assertAllClose(expected_outputs, got_outputs, atol=0.00323)

    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def
    self.assertTrue(self._contains_quantized_function_call(output_graphdef))

    # Check the converted model in the target opset.
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=[signature_key],
        op_set=target_opset,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path_2,
        quantization_options,
    )

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {signature_key}
    )
    loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path_2
    )
    graphdef = loader.get_meta_graph_def_from_tags(tags).graph_def
    if target_opset == quant_opts_pb2.XLA:
      self.assertTrue(
          self._contains_op(graphdef, 'XlaConvV2', node_name='sample/conv2d.*')
      )

    new_outputs = converted_model.signatures[signature_key](
        input=ops.convert_to_tensor(input_data)
    )
    # The difference between TF and XLA path is expected to be small (smaller
    # or equal to 1 in the quantized domain).
    self.assertAllClose(new_outputs, got_outputs, atol=0.00154)

  # Currently, only some specific forms of equantions are supported for
  # batchmatmul conversion.
  @parameterized.parameters(
      testing.parameter_combinations([{
          'equation': ('abc,cd->abd', 'abcd,cde->abe'),
          'shape_unknown': (True, False),
          'activation_fn': (None, nn_ops.relu, nn_ops.relu6),
          'has_bias': (True, False),
          'use_kernel': (True, False),
      }])
  )
  @test_util.run_in_graph_and_eager_modes
  def test_qat_einsum_model_with_batchmatmul_conversion(
      self,
      equation: str,
      shape_unknown: bool,
      activation_fn: Optional[ops.Operation],
      has_bias: bool,
      use_kernel: bool,
  ):
    x_shape, y_shape, bias_shape, x_signature, y_signature = (
        self._prepare_sample_einsum_datashapes(
            equation, shape_unknown, has_bias and not shape_unknown
        )
    )
    model = self._create_einsum_model(
        equation,
        y_shape,
        x_signature,
        y_signature,
        bias_shape,
        activation_fn,
        is_qat_model=True,
    )
    x = array_ops.constant(
        np.random.uniform(size=x_shape), dtype=dtypes.float32
    )
    y = array_ops.constant(
        np.random.uniform(size=y_shape), dtype=dtypes.float32
    )
    if use_kernel:
      model.einsum = model.einsum_with_kernel
      model_inputs = {'x': x}
    else:
      model.einsum = model.einsum_without_kernel
      model_inputs = {'x': x, 'y': y}

    saved_model_save.save(
        model, self._input_saved_model_path, signatures=model.einsum
    )

    signature_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    tags = {tag_constants.SERVING}

    # Check the converted model with TF opset as the baseline.
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=[signature_key],
        op_set=quant_opts_pb2.TF,
        enable_two_input_tensors=not use_kernel,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
    )
    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {signature_key}
    )

    expected_outputs = model.einsum(**model_inputs)
    got_outputs = converted_model.signatures[signature_key](**model_inputs)
    self.assertAllClose(expected_outputs, got_outputs, atol=1e-1)

    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def
    self.assertTrue(self._contains_quantized_function_call(output_graphdef))

    # Check the converted model in the XLA opset.
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=[signature_key],
        op_set=quant_opts_pb2.XLA,
        enable_two_input_tensors=not use_kernel,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path_2,
        quantization_options,
    )

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {signature_key}
    )
    loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path_2
    )
    graphdef = loader.get_meta_graph_def_from_tags(tags).graph_def
    self.assertTrue(self._contains_op(graphdef, 'XlaDotV2'))

    new_outputs = converted_model.signatures[signature_key](**model_inputs)

    # The difference between TF and XLA path is expected to be small (smaller
    # or equal to 1 in the quantized domain).
    self.assertAllClose(new_outputs, expected_outputs, atol=1e-1)

  # Equations only supported for XLA operations.
  @parameterized.parameters(
      testing.parameter_combinations([{
          'equation': ('abc,acd->abd', 'abcd,aecd->acbe'),
          'shape_unknown': (True, False),
          'activation_fn': (None, nn_ops.relu, nn_ops.relu6),
          'has_bias': (True, False),
          'use_kernel': (True, False),
      }])
  )
  @test_util.run_in_graph_and_eager_modes
  def test_qat_einsum_model_with_xla(
      self,
      equation: str,
      shape_unknown: bool,
      activation_fn: Optional[ops.Operation],
      has_bias: bool,
      use_kernel: bool,
  ):
    x_shape, y_shape, bias_shape, x_signature, y_signature = (
        self._prepare_sample_einsum_datashapes(
            equation, shape_unknown, has_bias and not shape_unknown
        )
    )
    model = self._create_einsum_model(
        equation,
        y_shape,
        x_signature,
        y_signature,
        bias_shape,
        activation_fn,
        is_qat_model=True,
    )

    x = array_ops.constant(
        np.random.uniform(size=x_shape), dtype=dtypes.float32
    )
    y = array_ops.constant(
        np.random.uniform(size=y_shape), dtype=dtypes.float32
    )
    if use_kernel:
      model.einsum = model.einsum_with_kernel
      model_inputs = {'x': x}
    else:
      model.einsum = model.einsum_without_kernel
      model_inputs = {'x': x, 'y': y}

    saved_model_save.save(
        model, self._input_saved_model_path, signatures=model.einsum
    )

    signature_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    tags = {tag_constants.SERVING}

    # Check the converted model in the XLA opset.
    expected_outputs = model.einsum(**model_inputs)
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=[signature_key],
        op_set=quant_opts_pb2.XLA,
        enable_two_input_tensors=not use_kernel,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
    )

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {signature_key}
    )
    loader = saved_model_loader.SavedModelLoader(self._output_saved_model_path)
    graphdef = loader.get_meta_graph_def_from_tags(tags).graph_def
    self.assertTrue(self._contains_op(graphdef, 'XlaDotV2'))

    outputs = converted_model.signatures[signature_key](**model_inputs)

    self.assertAllClose(outputs, expected_outputs, atol=1e-1)

  # Equations NOT supported for XLA operations.
  @parameterized.parameters(
      testing.parameter_combinations([{
          'equation': ('aecd,abcd->acbe', 'abc,acd->adb'),
          'use_kernel': (True, False),
      }])
  )
  @test_util.run_in_graph_and_eager_modes
  def test_qat_einsum_model_not_supported_with_xla(
      self,
      equation: str,
      use_kernel: bool,
  ):
    _, y_shape, _, x_signature, y_signature = (
        self._prepare_sample_einsum_datashapes(equation)
    )

    model = self._create_einsum_model(
        equation,
        y_shape,
        x_signature,
        y_signature,
        bias_shape=None,
        activation_fn=None,
        is_qat_model=True,
    )

    if use_kernel:
      model.einsum = model.einsum_with_kernel
    else:
      model.einsum = model.einsum_without_kernel

    saved_model_save.save(
        model, self._input_saved_model_path, signatures=model.einsum
    )

    signature_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    tags = {tag_constants.SERVING}

    # Check the converted model does NOT have XLA opset.
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=[signature_key],
        op_set=quant_opts_pb2.XLA,
        enable_two_input_tensors=not use_kernel,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path_2,
        quantization_options,
    )

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {signature_key}
    )
    loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path_2
    )
    graphdef = loader.get_meta_graph_def_from_tags(tags).graph_def
    self.assertFalse(self._contains_op(graphdef, 'XlaDotV2'))

  @test_util.run_in_graph_and_eager_modes
  def test_qat_gather_and_conv_model(
      self,
  ):
    input_type = dtypes.int32
    model = self._create_simple_gather_and_conv_model(
        input_type,
        filter_shape=(2, 3, 3, 1024),
        is_qat_model=True,
    )

    saved_model_save.save(model, self._input_saved_model_path)

    tags = {tag_constants.SERVING}

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=quant_opts_pb2.XLA,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
    )
    self.assertIsNotNone(converted_model)
    # Due to other meta data, the compression is not exactly 1/4.
    self.assertLess(
        testing.get_size_ratio(
            self._output_saved_model_path, self._input_saved_model_path
        ),
        0.5,
    )

  def test_qat_vocab_table_lookup_model(self):
    tags = {tag_constants.SERVING}
    signature_def_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY

    # Create and save a simple model that involves a hash table.
    inputs, outputs = self._create_and_save_vocab_table_lookup_qat_model_tf1(
        self._input_saved_model_path, tags, signature_def_key
    )

    # Make sure that the desired input key and output key is present.
    self.assertIn('input_vocabs', inputs.keys())
    self.assertIn('lookup', outputs.keys())

    # Representative dataset is composed of a set of vocabs for table lookup.
    repr_ds = [
        {'input_vocabs': np.array([b'hello', b'model', b'quantization'])}
        for _ in range(4)
    ]

    signature_def_keys = [signature_def_key]
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=signature_def_keys,
        op_set=quant_opts_pb2.TF,
    )

    quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
        representative_dataset=repr_ds,
    )

    # Tests table lookup to make sure the table has been initialized
    # successfully.
    with session.Session(graph=ops.Graph()) as sess:
      output_meta_graph_def = saved_model_loader.load(
          sess, tags=tags, export_dir=self._output_saved_model_path
      )

      # The graph should contain a quantized function call (it contains a
      # single f32 matmul node).
      self.assertTrue(
          self._contains_quantized_function_call(
              output_meta_graph_def.graph_def
          )
      )
      self.assertCountEqual(
          output_meta_graph_def.signature_def.keys(), signature_def_keys
      )

      signature_def = output_meta_graph_def.signature_def[signature_def_key]

      input_tensor_name = signature_def.inputs['input_vocabs'].name
      input_tensor = sess.graph.get_tensor_by_name(input_tensor_name)

      lookup_tensor_name = signature_def.outputs['lookup'].name
      lookup_tensor = sess.graph.get_tensor_by_name(lookup_tensor_name)

      lookup_val = sess.run(
          lookup_tensor,
          feed_dict={
              input_tensor: np.array([b'model', b'quantization', b'hello'])
          },
      )

      self.assertAllClose(lookup_val, [1.0, 2.0, 0.0])

  def test_qat_file_init_hash_table_lookup_model_tf1(self):
    tags = {tag_constants.SERVING}
    signature_def_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY

    # Create and save a simple model that involves a hash table.
    inputs, outputs = self._create_and_save_file_init_hash_table_qat_model_tf1(
        self._input_saved_model_path, tags, signature_def_key
    )

    # Make sure that the desired input key and output key is present.
    self.assertIn('input_vocabs', inputs.keys())
    self.assertIn('lookup', outputs.keys())

    # Representative dataset is composed of a set of vocabs for table lookup.
    repr_ds = [
        {'input_vocabs': np.array([b'static', b'range', b'quantization'])}
        for _ in range(4)
    ]

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=[signature_def_key],
        op_set=quant_opts_pb2.TF,
    )
    signature_def_keys = [signature_def_key]

    quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
        representative_dataset=repr_ds,
    )

    # Tests table lookup to make sure the table has been initialized
    # successfully.
    with session.Session(graph=ops.Graph()) as sess:
      output_meta_graph_def = saved_model_loader.load(
          sess, tags=tags, export_dir=self._output_saved_model_path
      )

      # The graph should contain a quantized function call (it contains a
      # single f32 matmul node).
      self.assertTrue(
          self._contains_quantized_function_call(
              output_meta_graph_def.graph_def
          )
      )
      self.assertCountEqual(
          output_meta_graph_def.signature_def.keys(), signature_def_keys
      )

      signature_def = output_meta_graph_def.signature_def[signature_def_key]
      input_tensor_name = signature_def.inputs['input_vocabs'].name
      input_tensor = sess.graph.get_tensor_by_name(input_tensor_name)
      lookup_tensor_name = signature_def.outputs['lookup'].name
      lookup_tensor = sess.graph.get_tensor_by_name(lookup_tensor_name)

      lookup_val = sess.run(
          lookup_tensor,
          feed_dict={
              input_tensor: np.array([b'dynamic', b'quantization', b'range'])
          },
      )

      # "dynamic" is not in the table: -1 (default value)
      self.assertAllClose(lookup_val, [-1.0, 2.0, 1.0])

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
                shape=(2, 3, 3, 2), minval=-1.0, maxval=1.0
            )
        )

      @def_function.function(
          input_signature=[
              tensor_spec.TensorSpec(
                  name='input', shape=(1, 3, 4, 3), dtype=dtypes.float32
              ),
          ]
      )
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
            data_format='NHWC',
        )
        return {'output': out}

    def gen_data() -> repr_dataset.RepresentativeDataset:
      """Creates an interable of representative samples.

      Yields:
        Representative samples, which is basically a mapping of: input key ->
        input value.
      """
      for _ in range(8):
        yield {
            'input': random_ops.random_uniform(
                shape=(1, 3, 4, 3), minval=0, maxval=150
            )
        }

    model = ConvModelWithVariable()
    saved_model_save.save(model, self._input_saved_model_path)

    signature_keys = [signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    tags = {tag_constants.SERVING}

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=signature_keys,
        op_set=quant_opts_pb2.TF,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
        representative_dataset=gen_data(),
    )

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), signature_keys
    )

    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def
    self.assertTrue(self._contains_quantized_function_call(output_graphdef))

  # Check only the most simple case and the most complicated cases.
  @parameterized.named_parameters(
      {
          'testcase_name': 'none',
          'activation_fn': None,
          'has_bias': False,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.TF,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': False,
      },
      {
          'testcase_name': 'dilation',
          'activation_fn': None,
          'has_bias': False,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.TF,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': False,
          'dilations': [1, 2, 2, 1],
      },
      {
          'testcase_name': 'relu',
          'activation_fn': nn_ops.relu,
          'has_bias': False,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.TF,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': False,
      },
      {
          'testcase_name': 'dilation_relu',
          'activation_fn': nn_ops.relu,
          'has_bias': False,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.TF,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': False,
          'dilations': [1, 2, 2, 1],
      },
      {
          'testcase_name': 'relu6',
          'activation_fn': nn_ops.relu6,
          'has_bias': False,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.TF,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': False,
      },
      {
          'testcase_name': 'dilation_relu6',
          'activation_fn': nn_ops.relu6,
          'has_bias': False,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.TF,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': False,
          'dilations': [1, 2, 2, 1],
      },
      {
          'testcase_name': 'bn',
          'activation_fn': None,
          'has_bias': False,
          'has_batch_norm': True,
          'target_opset': quant_opts_pb2.TF,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': False,
      },
      {
          'testcase_name': 'dilation_bn',
          'activation_fn': None,
          'has_bias': False,
          'has_batch_norm': True,
          'target_opset': quant_opts_pb2.TF,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': False,
          'dilations': [1, 2, 2, 1],
      },
      {
          'testcase_name': 'with_bias',
          'activation_fn': None,
          'has_bias': True,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.TF,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': False,
      },
      {
          'testcase_name': 'dilation_with_bias',
          'activation_fn': None,
          'has_bias': True,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.TF,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': False,
          'dilations': [1, 2, 2, 1],
      },
      {
          'testcase_name': 'with_bias_and_relu6',
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.TF,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': False,
      },
      {
          'testcase_name': 'dilation_with_bias_and_relu6',
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.TF,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': False,
          'dilations': [1, 2, 2, 1],
      },
      {
          'testcase_name': 'with_bias_and_bn_and_relu6',
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': True,
          'target_opset': quant_opts_pb2.TF,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': False,
      },
      {
          'testcase_name': 'dilation_with_bias_and_bn_and_relu6',
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': True,
          'target_opset': quant_opts_pb2.TF,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': False,
          'dilations': [1, 2, 2, 1],
      },
      {
          'testcase_name': 'with_bias_and_relu6_to_xla_per_tensor',
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.XLA,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': False,
      },
      {
          'testcase_name': 'with_bias_and_relu6_to_xla_per_channel',
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.XLA,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': True,
      },
      {
          'testcase_name': 'dilation_with_bias_and_relu6_to_xla',
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.XLA,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': False,
          'dilations': [1, 2, 2, 1],
      },
      {
          'testcase_name': 'dilation_with_bias_and_relu6_to_xla_per_channel',
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.XLA,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': True,
          'dilations': [1, 2, 2, 1],
      },
      {
          'testcase_name': 'with_bias_and_bn_and_relu6_to_xla',
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': True,
          'target_opset': quant_opts_pb2.XLA,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': False,
      },
      {
          'testcase_name': 'with_bias_and_bn_and_relu6_to_xla_per_channel',
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': True,
          'target_opset': quant_opts_pb2.XLA,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': True,
      },
      {
          'testcase_name': 'dilation_with_bias_and_bn_and_relu6_to_xla',
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': True,
          'target_opset': quant_opts_pb2.XLA,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': False,
          'dilations': [1, 2, 2, 1],
      },
      {
          'testcase_name': (
              'dilation_with_bias_and_bn_and_relu6_to_xla_per_channel'
          ),
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': True,
          'target_opset': quant_opts_pb2.XLA,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': True,
          'dilations': [1, 2, 2, 1],
      },
      {
          'testcase_name': 'with_bias_and_relu6_to_xla_dynamic',
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.XLA,
          'input_shape_dynamic': True,
          'enable_per_channel_quantization': False,
      },
      {
          'testcase_name': 'with_bias_and_relu6_to_xla_dynamic_per_channel',
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.XLA,
          'input_shape_dynamic': True,
          'enable_per_channel_quantization': True,
      },
      {
          'testcase_name': 'dilation_with_bias_and_relu6_to_xla_dynamic',
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.XLA,
          'input_shape_dynamic': True,
          'enable_per_channel_quantization': False,
          'dilations': [1, 2, 2, 1],
      },
      {
          'testcase_name': (
              'dilation_with_bias_and_relu6_to_xla_dynamic_per_channel'
          ),
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.XLA,
          'input_shape_dynamic': True,
          'enable_per_channel_quantization': True,
          'dilations': [1, 2, 2, 1],
      },
      {
          'testcase_name': 'with_bias_and_bn_and_relu6_to_xla_dynamic',
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': True,
          'target_opset': quant_opts_pb2.XLA,
          'input_shape_dynamic': True,
          'enable_per_channel_quantization': False,
      },
      {
          'testcase_name': (
              'with_bias_and_bn_and_relu6_to_xla_dynamic_per_channel'
          ),
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': True,
          'target_opset': quant_opts_pb2.XLA,
          'input_shape_dynamic': True,
          'enable_per_channel_quantization': True,
      },
      {
          'testcase_name': 'dilation_with_bias_and_bn_and_relu6_to_xla_dynamic',
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': True,
          'target_opset': quant_opts_pb2.XLA,
          'input_shape_dynamic': True,
          'enable_per_channel_quantization': False,
          'dilations': [1, 2, 2, 1],
      },
      {
          'testcase_name': (
              'dilation_with_bias_and_bn_and_relu6_to_xla_dynamic_per_channel'
          ),
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': True,
          'target_opset': quant_opts_pb2.XLA,
          'input_shape_dynamic': True,
          'enable_per_channel_quantization': True,
          'dilations': [1, 2, 2, 1],
      },
      {
          'testcase_name': 'with_bias_and_relu6_to_uq',
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.UNIFORM_QUANTIZED,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': False,
      },
      {
          'testcase_name': 'dilation_with_bias_and_relu6_to_uq',
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.UNIFORM_QUANTIZED,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': False,
          'dilations': [1, 2, 2, 1],
      },
      {
          'testcase_name': 'with_bias_and_bn_and_relu6_to_uq',
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': True,
          'target_opset': quant_opts_pb2.UNIFORM_QUANTIZED,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': False,
      },
      {
          'testcase_name': 'dilation_with_bias_and_bn_and_relu6_to_uq',
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': True,
          'target_opset': quant_opts_pb2.UNIFORM_QUANTIZED,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': False,
          'dilations': [1, 2, 2, 1],
      },
      {
          'testcase_name': 'with_bias_and_relu6_to_uq_per_channel',
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.UNIFORM_QUANTIZED,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': True,
      },
      {
          'testcase_name': 'dilation_with_bias_and_relu6_to_uq_per_channel',
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.UNIFORM_QUANTIZED,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': True,
          'dilations': [1, 2, 2, 1],
      },
      {
          'testcase_name': 'with_bias_and_bn_and_relu6_to_uq_per_channel',
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': True,
          'target_opset': quant_opts_pb2.UNIFORM_QUANTIZED,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': True,
      },
      {
          'testcase_name': (
              'dilation_with_bias_and_bn_and_relu6_to_uq_per_channel'
          ),
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': True,
          'target_opset': quant_opts_pb2.UNIFORM_QUANTIZED,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': True,
          'dilations': [1, 2, 2, 1],
      },
      {
          'testcase_name': 'with_bias_and_relu6_to_stablehlo_per_channel',
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.STABLEHLO,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': True,
      },
  )
  @test_util.run_in_graph_and_eager_modes
  def test_conv_ptq_model(
      self,
      activation_fn: Optional[ops.Operation],
      has_bias: bool,
      has_batch_norm: bool,
      target_opset: quant_opts_pb2.OpSet,
      input_shape_dynamic: bool,
      enable_per_channel_quantization: bool,
      dilations: Sequence[int] = None,
  ):
    input_shape = [None, None, None, 3] if input_shape_dynamic else [1, 3, 4, 3]
    filter_shape = [2, 3, 3, 2]
    strides = [1, 1, 1, 1]

    model = self._create_conv2d_model(
        input_shape,
        filter_shape,
        has_bias,
        has_batch_norm,
        activation_fn,
        strides,
        dilations,
    )
    saved_model_save.save(model, self._input_saved_model_path)

    # Generate model input data.
    rng = np.random.default_rng(seed=1234)
    static_input_shape = [dim if dim is not None else 2 for dim in input_shape]
    input_data = ops.convert_to_tensor(
        rng.uniform(low=0.0, high=1.0, size=static_input_shape).astype(
            np.float32
        )
    )

    def data_gen() -> repr_dataset.RepresentativeDataset:
      for _ in range(500):
        yield {
            'input_tensor': rng.uniform(
                low=0.0, high=1.0, size=static_input_shape
            ).astype(np.float32)
        }

    tags = {tag_constants.SERVING}

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=target_opset,
        enable_per_channel_quantization=enable_per_channel_quantization,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
        representative_dataset=data_gen(),
    )
    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )

    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def

    # The difference between float model and target path quantized model is
    # expected to be small.
    # The atol value is arbitrary.
    if not enable_per_channel_quantization:
      expected_outputs = model.conv(input_data)
      target_outputs = converted_model.signatures['serving_default'](
          input_tensor=ops.convert_to_tensor(input_data)
      )
      self.assertAllClose(target_outputs, expected_outputs, atol=0.06)

    if target_opset == quant_opts_pb2.XLA:
      self.assertTrue(self._contains_op(output_graphdef, 'XlaConvV2'))
      if enable_per_channel_quantization:
        per_channel_size_attr = attr_value_pb2.AttrValue(
            list=attr_value_pb2.AttrValue.ListValue(
                shape=[
                    tensor_shape_pb2.TensorShapeProto(
                        dim=[
                            tensor_shape_pb2.TensorShapeProto.Dim(
                                size=filter_shape[-1]
                            )
                        ]
                    )
                ]
            )
        )
        self.assertTrue(
            self._contains_op(
                output_graphdef,
                'Const',
                '_output_shapes',
                per_channel_size_attr,
            )
        )
    elif target_opset == quant_opts_pb2.UNIFORM_QUANTIZED:
      self.assertTrue(
          self._contains_op(output_graphdef, 'UniformQuantizedConvolution')
      )
      if enable_per_channel_quantization:
        quantized_axis = 3
        quantized_dim_size_attr = attr_value_pb2.AttrValue(
            list=attr_value_pb2.AttrValue.ListValue(
                shape=[
                    tensor_shape_pb2.TensorShapeProto(
                        dim=[
                            tensor_shape_pb2.TensorShapeProto.Dim(
                                size=filter_shape[quantized_axis]
                            )
                        ]
                    )
                ]
            )
        )
      else:
        quantized_axis = -1
        # Empty dimension. Per-tensor quantization has singular channel.
        quantized_dim_size_attr = attr_value_pb2.AttrValue(
            list=attr_value_pb2.AttrValue.ListValue(
                shape=[tensor_shape_pb2.TensorShapeProto()]
            )
        )
      quantized_axis_attr = attr_value_pb2.AttrValue(i=quantized_axis)
      self.assertEqual(
          self._count_ops(
              output_graphdef,
              _PER_CHANNEL_QUANTIZED_OPS,
              'rhs_quantization_axis',
              quantized_axis_attr,
          ),
          self._count_ops(output_graphdef, _PER_CHANNEL_QUANTIZED_OPS),
      )
      self.assertEqual(
          self._count_ops(
              output_graphdef,
              _PER_CHANNEL_OP_NAMES,
              '_output_shapes',
              quantized_dim_size_attr,
              get_op_name=True,
          ),
          self._count_ops(
              output_graphdef,
              _PER_CHANNEL_OP_NAMES,
              get_op_name=True,
          ),
      )
      self.assertFalse(self._contains_op(output_graphdef, 'Conv2D'))
    elif target_opset == quant_opts_pb2.STABLEHLO:
      # This is to verify the invocation of StableHLO quantizer works. More
      # thorough functional tests are in StableHLO quantizer directory.
      self.assertTrue(self._contains_op(output_graphdef, 'XlaCallModule'))
    else:
      self.assertTrue(self._contains_quantized_function_call(output_graphdef))
    self.assertFalse(self._contains_op(output_graphdef, 'FusedBatchNormV3'))

  @parameterized.named_parameters(
      ('to_tf_with_int32_input_type', dtypes.int32, quant_opts_pb2.TF),
      ('to_xla_with_int32_input_type', dtypes.int32, quant_opts_pb2.XLA),
      ('to_xla_with_int64_input_type', dtypes.int64, quant_opts_pb2.XLA),
      (
          'to_uq_with_int32_input_type',
          dtypes.int32,
          quant_opts_pb2.UNIFORM_QUANTIZED,
      ),
  )
  @test_util.run_v2_only
  def test_gather_and_conv_model(
      self, input_type: dtypes, target_opset: quant_opts_pb2.OpSet
  ):
    model = self._create_simple_gather_and_conv_model(
        input_type, filter_shape=(2, 3, 3, 1024)
    )
    saved_model_save.save(model, self._input_saved_model_path)

    data_gen = self._create_data_generator(
        input_key='input_tensor',
        shape=[500],
        minval=0,
        maxval=64,
        dtype=input_type,
    )

    tags = {tag_constants.SERVING}

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=target_opset,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
        representative_dataset=data_gen,
    )
    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )

    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def

    if target_opset == quant_opts_pb2.UNIFORM_QUANTIZED:
      self.assertGreater(
          testing.get_size_ratio(
              self._output_saved_model_path, self._input_saved_model_path
          ),
          0.68,
      )
      self.assertTrue(
          self._contains_op(output_graphdef, 'UniformQuantizedConvolution')
      )
    else:
      # Due to other meta data, the compression is not exactly 1/4.
      self.assertLess(
          testing.get_size_ratio(
              self._output_saved_model_path, self._input_saved_model_path
          ),
          1 / 3,
      )
      if target_opset == quant_opts_pb2.XLA:
        self.assertTrue(self._contains_op(output_graphdef, 'XlaConvV2'))
      else:
        self.assertTrue(self._contains_quantized_function_call(output_graphdef))

    test_data = np.random.uniform(low=0, high=64, size=(32)).astype(
        input_type.as_numpy_dtype
    )
    original_outputs = model.model(test_data)['output']
    quantized_output = converted_model.signatures['serving_default'](
        input_tensor=ops.convert_to_tensor(test_data)
    )['output']
    self.assertAllClose(original_outputs, quantized_output, atol=442.7)

  @test_util.run_v2_only
  def test_while_op_model(
      self,
  ):
    input_shape = (1, 5, 5, 32)
    model = self._create_while_model(input_shape)
    saved_model_save.save(model, self._input_saved_model_path)

    tags = {tag_constants.SERVING}
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=quant_opts_pb2.XLA,
    )

    def data_gen() -> repr_dataset.RepresentativeDataset:
      for _ in range(2):
        yield {
            'input_tensor': ops.convert_to_tensor(
                np.random.uniform(low=0, high=150, size=input_shape).astype(
                    'f4'
                )
            ),
        }

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
        representative_dataset=data_gen(),
    )
    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )

    loader = saved_model_loader.SavedModelLoader(self._output_saved_model_path)
    output_graphdef = loader.get_meta_graph_def_from_tags(tags).graph_def

    # Convolution ouside the while op is quantized.
    self.assertTrue(
        self._contains_op(
            output_graphdef,
            op_name='XlaConvV2',
            attr_name='RhsT',
            attr_val=attr_value_pb2.AttrValue(type=types_pb2.DT_INT8),
        )
    )
    # TODO: b/294783597 - [Converter][TF-Quantizer] Support quantization for the
    # ops in the while op body for both SRQ and WO
    # Convolution inside the while op is not quantized.
    self.assertTrue(
        self._contains_op(
            output_graphdef,
            op_name='Conv2D',
            attr_name='T',
            attr_val=attr_value_pb2.AttrValue(type=types_pb2.DT_FLOAT),
        )
    )

  # Check only the most simple case and the most complicated cases.
  @parameterized.named_parameters(
      {
          'testcase_name': 'none',
          'activation_fn': None,
          'has_bias': False,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.TF,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': False,
      },
      {
          'testcase_name': 'relu',
          'activation_fn': nn_ops.relu,
          'has_bias': False,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.TF,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': False,
      },
      {
          'testcase_name': 'relu6',
          'activation_fn': nn_ops.relu6,
          'has_bias': False,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.TF,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': False,
      },
      {
          'testcase_name': 'bn',
          'activation_fn': None,
          'has_bias': False,
          'has_batch_norm': True,
          'target_opset': quant_opts_pb2.TF,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': False,
      },
      {
          'testcase_name': 'with_bias',
          'activation_fn': None,
          'has_bias': True,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.TF,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': False,
      },
      {
          'testcase_name': 'with_bias_and_relu6',
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.TF,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': False,
      },
      {
          'testcase_name': 'with_bias_and_bn_and_relu6',
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': True,
          'target_opset': quant_opts_pb2.TF,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': False,
      },
      {
          'testcase_name': 'with_bias_and_relu6_to_xla',
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.XLA,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': False,
      },
      {
          'testcase_name': 'with_bias_and_relu6_to_xla_per_channel',
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.XLA,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': True,
      },
      {
          'testcase_name': 'with_bias_and_bn_and_relu6_to_xla',
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': True,
          'target_opset': quant_opts_pb2.XLA,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': False,
      },
      {
          'testcase_name': 'with_bias_and_bn_and_relu6_to_xla_per_channel',
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': True,
          'target_opset': quant_opts_pb2.XLA,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': True,
      },
      {
          'testcase_name': 'with_bias_and_relu6_to_xla_dynamic',
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.XLA,
          'input_shape_dynamic': True,
          'enable_per_channel_quantization': False,
      },
      {
          'testcase_name': 'with_bias_and_relu6_to_xla_dynamic_per_channel',
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.XLA,
          'input_shape_dynamic': True,
          'enable_per_channel_quantization': True,
      },
      {
          'testcase_name': 'with_bias_and_bn_and_relu6_to_xla_dynamic',
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': True,
          'target_opset': quant_opts_pb2.XLA,
          'input_shape_dynamic': True,
          'enable_per_channel_quantization': False,
      },
      {
          'testcase_name': (
              'with_bias_and_bn_and_relu6_to_xla_dynamic_per_channel'
          ),
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': True,
          'target_opset': quant_opts_pb2.XLA,
          'input_shape_dynamic': True,
          'enable_per_channel_quantization': True,
      },
      {
          'testcase_name': 'with_bias_and_relu6_to_uq',
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.UNIFORM_QUANTIZED,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': False,
      },
      {
          'testcase_name': 'with_bias_and_bn_and_relu6_to_uq',
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': True,
          'target_opset': quant_opts_pb2.UNIFORM_QUANTIZED,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': False,
      },
      {
          'testcase_name': 'with_bias_and_relu6_to_uq_per_channel',
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.UNIFORM_QUANTIZED,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': True,
      },
      {
          'testcase_name': 'with_bias_and_bn_and_relu6_to_uq_per_channel',
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': True,
          'target_opset': quant_opts_pb2.UNIFORM_QUANTIZED,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': True,
      },
  )
  @test_util.run_in_graph_and_eager_modes
  def test_depthwise_conv_ptq_model(
      self,
      activation_fn: Optional[ops.Operation],
      has_bias: bool,
      has_batch_norm: bool,
      target_opset: quant_opts_pb2.OpSet,
      input_shape_dynamic: bool,
      enable_per_channel_quantization: bool,
  ):
    input_shape = [None, None, None, 3] if input_shape_dynamic else [1, 3, 4, 3]
    filter_shape = [2, 3, 3, 1]
    model = self._create_depthwise_conv2d_model(
        input_shape, filter_shape, has_bias, has_batch_norm, activation_fn
    )
    saved_model_save.save(model, self._input_saved_model_path)

    def data_gen() -> repr_dataset.RepresentativeDataset:
      for _ in range(8):
        yield {
            'input_tensor': ops.convert_to_tensor(
                np.random.uniform(low=0, high=150, size=(1, 3, 4, 3)).astype(
                    'f4'
                )
            ),
        }

    tags = {tag_constants.SERVING}

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=target_opset,
        enable_per_channel_quantization=enable_per_channel_quantization,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
        representative_dataset=data_gen(),
    )
    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )

    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def
    if target_opset == quant_opts_pb2.XLA:
      self.assertTrue(
          self._contains_op(output_graphdef, 'DepthwiseConv2dNative')
      )
      if enable_per_channel_quantization:
        per_channel_size_attr = attr_value_pb2.AttrValue(
            list=attr_value_pb2.AttrValue.ListValue(
                shape=[
                    tensor_shape_pb2.TensorShapeProto(
                        dim=[
                            tensor_shape_pb2.TensorShapeProto.Dim(
                                size=filter_shape[-1] * filter_shape[2]
                            )
                        ]
                    )
                ]
            )
        )
        self.assertTrue(
            self._contains_op(
                output_graphdef,
                'Const',
                '_output_shapes',
                per_channel_size_attr,
            )
        )
    elif target_opset == quant_opts_pb2.UNIFORM_QUANTIZED:
      self.assertTrue(
          self._contains_op(output_graphdef, 'UniformQuantizedConvolution')
      )
      if enable_per_channel_quantization:
        quantized_axis = 3
        quantized_dim_size_attr = attr_value_pb2.AttrValue(
            list=attr_value_pb2.AttrValue.ListValue(
                shape=[
                    tensor_shape_pb2.TensorShapeProto(
                        dim=[
                            tensor_shape_pb2.TensorShapeProto.Dim(
                                # Depthwise conv is reshaped to [H,W,1,CxM].
                                size=filter_shape[quantized_axis]
                                * filter_shape[2]
                            )
                        ]
                    )
                ]
            )
        )
      else:
        quantized_axis = -1
        # Empty dimension. Per-tensor quantization has singular channel.
        quantized_dim_size_attr = attr_value_pb2.AttrValue(
            list=attr_value_pb2.AttrValue.ListValue(
                shape=[tensor_shape_pb2.TensorShapeProto()]
            )
        )
      quantized_axis_attr = attr_value_pb2.AttrValue(i=quantized_axis)
      self.assertEqual(
          self._count_ops(
              output_graphdef,
              _PER_CHANNEL_QUANTIZED_OPS,
              'rhs_quantization_axis',
              quantized_axis_attr,
          ),
          self._count_ops(output_graphdef, _PER_CHANNEL_QUANTIZED_OPS),
      )
      self.assertEqual(
          self._count_ops(
              output_graphdef,
              _PER_CHANNEL_OP_NAMES,
              '_output_shapes',
              quantized_dim_size_attr,
              get_op_name=True,
          ),
          self._count_ops(
              output_graphdef,
              _PER_CHANNEL_OP_NAMES,
              get_op_name=True,
          ),
      )
      self.assertFalse(self._contains_op(output_graphdef, 'Conv2D'))
    else:
      self.assertTrue(self._contains_quantized_function_call(output_graphdef))
    self.assertFalse(self._contains_op(output_graphdef, 'FusedBatchNormV3'))

  @parameterized.parameters(
      *testing.parameter_combinations([
          {
              'activation_fn': [None, nn_ops.relu, nn_ops.relu6],
              'has_bias': [True, False],
              'batch_sizes': [([], []), ([10], [10]), ([2, 3], [2, 3])],
              'target_opset': [quant_opts_pb2.XLA],
          },
          # Test broadcastable batch sizes.
          {
              'activation_fn': [None],
              'has_bias': [True],
              'batch_sizes': [
                  ([2], []),
                  ([], [2]),
                  ([1], [2]),
                  ([None], []),
              ],
              'target_opset': [quant_opts_pb2.XLA],
          },
          {
              'activation_fn': [None, nn_ops.relu, nn_ops.relu6],
              'has_bias': [True, False],
              'batch_sizes': [([], []), ([10], [10]), ([2, 3], [2, 3])],
              'target_opset': [quant_opts_pb2.UNIFORM_QUANTIZED],
          },
      ])
  )
  @test_util.run_in_graph_and_eager_modes
  def test_matmul_ptq_model(
      self,
      activation_fn: Optional[ops.Operation],
      has_bias: bool,
      batch_sizes: Sequence[int],
      target_opset: quant_opts_pb2.OpSet,
  ):
    lhs_batch_size, rhs_batch_size = batch_sizes
    input_shape = (*lhs_batch_size, 1, 1024)
    filter_shape = (*rhs_batch_size, 1024, 3)
    static_input_shape = [dim if dim is not None else 2 for dim in input_shape]
    model = self._create_matmul_model(
        input_shape,
        filter_shape,
        self._input_saved_model_path,
        has_bias,
        activation_fn,
    )
    rng = np.random.default_rng(seed=1234)

    def data_gen() -> repr_dataset.RepresentativeDataset:
      for _ in range(500):
        yield {
            'input_tensor': rng.uniform(
                low=0.0, high=1.0, size=static_input_shape
            ).astype(np.float32)
        }

    tags = {tag_constants.SERVING}

    input_data = ops.convert_to_tensor(
        rng.uniform(low=0.0, high=1.0, size=static_input_shape).astype(
            np.float32
        )
    )
    expected_outputs = model.matmul(input_data)

    # Check the converted model in the target opset.
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=target_opset,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
        representative_dataset=data_gen(),
    )

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )
    loader = saved_model_loader.SavedModelLoader(self._output_saved_model_path)
    output_graphdef = loader.get_meta_graph_def_from_tags(tags).graph_def
    if target_opset == quant_opts_pb2.XLA:
      self.assertTrue(
          self._contains_op(
              output_graphdef, 'XlaDotV2', node_name='sample/matmul.*'
          )
      )
    elif target_opset == quant_opts_pb2.UNIFORM_QUANTIZED:
      self.assertTrue(
          self._contains_op(
              output_graphdef,
              'UniformQuantizedDot',
              node_name='sample/matmul.*',
          )
      )

    new_outputs = converted_model.signatures['serving_default'](
        input_tensor=ops.convert_to_tensor(input_data)
    )
    # The difference between TF and target path is expected to be small.
    # The atol value is arbitrary.
    # Currently, Uniform Quantized Opset are producing non-optimal graphs:
    # unnecessary requantization followed by dequantization, so the error will
    # be higher.
    if target_opset == quant_opts_pb2.UNIFORM_QUANTIZED:
      self.assertAllClose(new_outputs, expected_outputs, atol=0.25)
    else:
      self.assertAllClose(new_outputs, expected_outputs, atol=0.13)

  @parameterized.named_parameters(
      {
          'testcase_name': 'with_biasadd',
          'input_shape': (32, 16),
          'filter_shape': (16, 8),
          'bias_size': 4,
          'use_biasadd': True,
          'activation_fn': nn_ops.relu,
      },
      {
          'testcase_name': 'with_addv2',
          'input_shape': (32, 16),
          'filter_shape': (16, 8),
          'bias_size': 4,
          'use_biasadd': False,
          'activation_fn': nn_ops.relu,
      },
  )
  def test_matmul_with_reshape_and_bias_ptq_model(
      self, input_shape, filter_shape, bias_size, activation_fn, use_biasadd
  ):
    model = self._create_matmul_model(
        input_shape,
        filter_shape,
        self._input_saved_model_path,
        True,
        activation_fn,
        bias_size,
        use_biasadd,
    )

    rng = np.random.default_rng(seed=1234)

    def data_gen() -> repr_dataset.RepresentativeDataset:
      for _ in range(5):
        yield {
            'input_tensor': rng.uniform(
                low=0.0, high=1.0, size=input_shape
            ).astype(np.float32)
        }

    tags = {tag_constants.SERVING}

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=quant_opts_pb2.OpSet.XLA,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
        representative_dataset=data_gen(),
    )

    input_data = ops.convert_to_tensor(
        rng.uniform(low=0.0, high=1.0, size=input_shape).astype(np.float32)
    )
    expected_outputs = model.matmul(input_data)

    got_outputs = converted_model.signatures['serving_default'](
        input_tensor=ops.convert_to_tensor(input_data)
    )

    self.assertAllClose(expected_outputs, got_outputs, atol=0.05)

  @parameterized.parameters(
      ('abc,cde->abde', quant_opts_pb2.XLA),
      ('abc,dce->abde', quant_opts_pb2.XLA),
  )
  def test_einsum_ptq_model(
      self,
      equation: str,
      target_opset: quant_opts_pb2.OpSet,
  ):
    _, y_shape, bias_shape, x_signature, y_signature = (
        self._prepare_sample_einsum_datashapes(equation, use_bias=True)
    )

    model = self._create_einsum_model(
        equation,
        y_shape,
        x_signature,
        y_signature,
        bias_shape,
        activation_fn=nn_ops.relu,
    )

    signatures = {
        'serving_default': model.einsum_with_kernel.get_concrete_function(),
    }

    saved_model_save.save(model, self._input_saved_model_path, signatures)

    def data_gen() -> repr_dataset.RepresentativeDataset:
      for _ in range(4):
        yield {
            'x': ops.convert_to_tensor(
                np.random.uniform(low=0.0, high=1.0, size=x_signature).astype(
                    'f4'
                )
            ),
        }

    tags = {tag_constants.SERVING}

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=quant_opts_pb2.TF,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
        representative_dataset=data_gen(),
    )
    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )

    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def
    self.assertTrue(self._contains_quantized_function_call(output_graphdef))

    input_data = ops.convert_to_tensor(
        np.random.uniform(low=0.0, high=1.0, size=x_signature).astype('f4')
    )
    expected_outputs = model.einsum_with_kernel(input_data)
    got_outputs = converted_model.signatures['serving_default'](
        x=ops.convert_to_tensor(input_data)
    )
    self.assertAllClose(expected_outputs, got_outputs, atol=0.097)

    # Check the converted model in the target opset.
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=target_opset,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path_2,
        quantization_options,
        representative_dataset=data_gen(),
    )

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )
    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path_2
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def
    if target_opset == quant_opts_pb2.XLA:
      self.assertTrue(self._contains_op(output_graphdef, 'XlaDotV2'))

    new_outputs = converted_model.signatures['serving_default'](
        x=ops.convert_to_tensor(input_data)
    )
    # The difference between TF and target path is expected to be small.
    self.assertAllClose(new_outputs, got_outputs, atol=0.097)
    self.assertAllClose(new_outputs, expected_outputs, atol=0.057)

  def test_reuse_calibration_data(self):
    model = self._create_simple_gather_and_conv_model(
        dtypes.int32, filter_shape=(2, 3, 3, 1024)
    )
    saved_model_save.save(model, self._input_saved_model_path)

    data_gen = self._create_data_generator(
        input_key='input_tensor',
        shape=[50],
        minval=0,
        maxval=64,
        dtype=dtypes.int32,
    )

    tags = {tag_constants.SERVING}

    calibration_data_dir = self.create_tempdir('calibration_data').full_path
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=quant_opts_pb2.XLA,
        force_graph_mode_calibration=True,
        calibration_options=stablehlo_quant_config_pb2.CalibrationOptions(
            calibration_method=_CalibrationMethod.CALIBRATION_METHOD_MIN_MAX,
            calibration_data_dir=calibration_data_dir,
        ),
    )

    # Run quantization the first time, calibration is expected to be run.
    with self.assertLogs(level='INFO') as info_logs:
      # Save the logger verbosity.
      prev_log_level = logging.get_verbosity()
      logging.set_verbosity(logging.INFO)
      try:
        converted_model1 = quantize_model.quantize(
            self._input_saved_model_path,
            self._output_saved_model_path,
            quantization_options,
            representative_dataset=data_gen,
        )
      finally:
        # Restore the logger verbosity.
        logging.set_verbosity(prev_log_level)

      self.assertNotEmpty(info_logs.records)
      self.assertTrue(
          self._any_log_contains(
              'Calibration step is executed in graph mode.',
              info_logs.records,
          )
      )
      self.assertIsNotNone(converted_model1)
      self.assertCountEqual(
          converted_model1.signatures._signatures.keys(), {'serving_default'}
      )

      output_loader = saved_model_loader.SavedModelLoader(
          self._output_saved_model_path
      )
      output_graphdef = output_loader.get_meta_graph_def_from_tags(
          tags
      ).graph_def
      self.assertTrue(self._contains_op(output_graphdef, 'XlaConvV2'))

    # Run quantization the first time, calibration is expected to be skipped.
    with self.assertLogs(level='INFO') as info_logs:
      # Save the logger verbosity.
      prev_log_level = logging.get_verbosity()
      logging.set_verbosity(logging.INFO)
      try:
        converted_model2 = quantize_model.quantize(
            self._input_saved_model_path,
            self._output_saved_model_path,
            quantization_options,
            representative_dataset=data_gen,
            overwrite_output_directory=True,
        )
      finally:
        # Restore the logger verbosity.
        logging.set_verbosity(prev_log_level)

      self.assertNotEmpty(info_logs.records)
      self.assertFalse(
          self._any_log_contains(
              'Calibration step is executed in graph mode.',
              info_logs.records,
          )
      )
      self.assertIsNotNone(converted_model2)
      self.assertCountEqual(
          converted_model2.signatures._signatures.keys(), {'serving_default'}
      )

      # Expect two models to produce the same results.
      test_data = ops.convert_to_tensor(
          np.random.uniform(low=0, high=64, size=(32)).astype(
              dtypes.int32.as_numpy_dtype
          )
      )
      new_outputs_1 = converted_model1.signatures['serving_default'](
          input_tensor=test_data
      )['output']
      new_outputs_2 = converted_model2.signatures['serving_default'](
          input_tensor=test_data
      )['output']
      self.assertAllClose(new_outputs_1, new_outputs_2)

  @test_util.run_in_graph_and_eager_modes
  def test_function_alias_preserved(self):
    model = self._create_conv2d_model(
        input_shape=(1, 3, 4, 3), filter_shape=(2, 3, 3, 2)
    )

    signatures = {
        'serving_default': model.conv.get_concrete_function(),
    }
    save_opts = save_options.SaveOptions(
        function_aliases={'conv_func': model.conv}
    )

    saved_model_save.save(
        model, self._input_saved_model_path, signatures, save_opts
    )

    def data_gen() -> repr_dataset.RepresentativeDataset:
      rng = np.random.default_rng(seed=123)
      for _ in range(2):
        yield {
            'input_tensor': rng.uniform(
                low=0, high=150, size=(1, 3, 4, 3)
            ).astype(np.float32),
        }

    tags = {tag_constants.SERVING}

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=quant_opts_pb2.OpSet.XLA,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
        representative_dataset=data_gen(),
    )

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )

    # Test whether the aliased function exists.
    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )

    # Confirm that the function alias is preserved.
    meta_graph_def = output_loader.get_meta_graph_def_from_tags(tags)
    function_aliases = meta_graph_def.meta_info_def.function_aliases
    self.assertNotEmpty(function_aliases)
    self.assertCountEqual(function_aliases.values(), {'conv_func'})

    # Test that the aliased function contains a quantized op.
    for func_name, alias in function_aliases.items():
      if alias == 'conv_func':
        for func in meta_graph_def.graph_def.library.function:
          if func.signature.name == func_name:
            self.assertTrue(
                self._contains_op_with_name_and_attribute(
                    func.node_def,
                    op_name='XlaConvV2',
                    attr_name='',
                    attr_val=None,
                )
            )

  @test_util.run_in_graph_and_eager_modes
  def test_function_alias_preserved_in_qat(self):
    _, y_shape, _, x_signature, y_signature = (
        self._prepare_sample_einsum_datashapes('ab,bc->ac')
    )
    model = self._create_einsum_model(
        'ab,bc->ac', y_shape, x_signature, y_signature, is_qat_model=True
    )

    signatures = {
        'serving_default': model.einsum_with_kernel.get_concrete_function(),
    }
    save_opts = save_options.SaveOptions(
        function_aliases={'einsum_with_kernel': model.einsum_with_kernel}
    )

    saved_model_save.save(
        model, self._input_saved_model_path, signatures, save_opts
    )

    tags = {tag_constants.SERVING}

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=quant_opts_pb2.OpSet.XLA,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
    )

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )

    # Test whether the aliased function exists.
    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )

    # Confirm that the function alias is preserved.
    meta_graph_def = output_loader.get_meta_graph_def_from_tags(tags)
    function_aliases = meta_graph_def.meta_info_def.function_aliases
    self.assertNotEmpty(function_aliases)
    self.assertCountEqual(function_aliases.values(), {'einsum_with_kernel'})

    # Test that the aliased function contains a quantized op.
    for func_name, alias in function_aliases.items():
      if alias == 'einsum_with_kernel':
        for func in meta_graph_def.graph_def.library.function:
          if func.signature.name == func_name:
            self.assertTrue(
                self._contains_op_with_name_and_attribute(
                    func.node_def,
                    op_name='XlaDotV2',
                    attr_name='',
                    attr_val=None,
                )
            )

  def test_matmul_ptq_model_with_unfreeze_constants(self):
    # Uses large weight to exceed the constant size threshold of 64KiB
    # (specified by `kDefaultConstantSizeThresholdInBytes`) for unfreezing.
    self._create_matmul_model(
        input_shape=(1, 20),
        weight_shape=(20, 4096),
        saved_model_path=self._input_saved_model_path,
    )

    repr_ds = self._create_data_generator(
        input_key='input_tensor', shape=(1, 20), num_examples=2
    )

    tags = {tag_constants.SERVING}

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=quant_opts_pb2.TF,
        freeze_all_variables=False,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
        representative_dataset=repr_ds,
    )

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )

    # Test that the quantized model successfully loads without error.
    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    with session.Session(graph=ops.Graph()) as sess:
      output_meta_graph_def = output_loader.load(sess, tags)

    # Confirms that quantization is applied to the model.
    output_graphdef = output_meta_graph_def.graph_def
    self.assertTrue(self._contains_quantized_function_call(output_graphdef))

    # Tests that there are variables in the model.
    variable_node_defs = _find_variables(output_graphdef)
    self.assertLen(variable_node_defs, 1)

    # Reads the variables from the checkpoint file and matches with the
    # variables found in the graph.
    checkpoint_path = os.path.join(
        self._output_saved_model_path, 'variables', 'variables'
    )
    var_name_and_shapes = checkpoint_utils.list_variables(checkpoint_path)

    # Checks that each variable's name and shape match.
    self.assertEqual(len(variable_node_defs), len(var_name_and_shapes))
    for var_name, shape in var_name_and_shapes:
      self.assertIn(var_name, variable_node_defs)
      self.assertEqual(
          shape,
          tensor_shape.TensorShape(
              variable_node_defs[var_name].attr['shape'].shape
          ),
      )

  @parameterized.named_parameters(
      ('use_constant_with_int32_input', dtypes.int32, False, True),
      ('use_variable_with_int32_input', dtypes.int32, True, True),
      ('use_constant_with_int64_input', dtypes.int64, False, True),
      ('use_variable_with_int64_input', dtypes.int64, True, True),
      ('small_gather_use_constant', dtypes.int32, False, False),
      ('small_gather_use_variable', dtypes.int32, True, False),
  )
  @test_util.run_v2_only
  def test_gather_model(
      self, input_type, use_variable, expect_quantized_gather
  ):
    model = self._create_gather_model(input_type, use_variable)

    saved_model_save.save(model, self._input_saved_model_path)

    tags = {tag_constants.SERVING}

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=quant_opts_pb2.XLA,
        # Gather op is opt-outed if the size is smaller than the threshold.
        min_num_elements_for_weights=1024 if expect_quantized_gather else 8192,
    )

    data_gen = self._create_data_generator(
        input_key='input_tensor',
        shape=[6],
        minval=0,
        maxval=10,
        dtype=input_type,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
        representative_dataset=data_gen,
    )

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )

    if expect_quantized_gather:
      # Due to other meta data, the compression is not exactly 1/4.
      self.assertLess(
          testing.get_size_ratio(
              self._output_saved_model_path, self._input_saved_model_path
          ),
          1 / 3,
      )
    else:
      self.assertGreater(
          testing.get_size_ratio(
              self._output_saved_model_path, self._input_saved_model_path
          ),
          2 / 3,
      )

  @test_util.run_in_graph_and_eager_modes
  def test_model_ptq_use_representative_samples_list(self):
    self._create_matmul_model(
        input_shape=(1, 1024),
        weight_shape=(1024, 3),
        saved_model_path=self._input_saved_model_path,
    )

    tags = {tag_constants.SERVING}
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=quant_opts_pb2.TF,
    )

    representative_dataset: repr_dataset.RepresentativeDataset = [
        {
            'input_tensor': random_ops.random_uniform(shape=(1, 1024)),
        }
        for _ in range(8)
    ]

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options=quantization_options,
        representative_dataset=representative_dataset,
    )

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )
    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def
    self.assertTrue(self._contains_quantized_function_call(output_graphdef))

  @test_util.run_in_graph_and_eager_modes
  def test_model_ptq_use_ndarray_representative_dataset(self):
    self._create_matmul_model(
        input_shape=(1, 1024),
        weight_shape=(1024, 3),
        saved_model_path=self._input_saved_model_path,
    )

    tags = {tag_constants.SERVING}
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=quant_opts_pb2.TF,
    )

    # Use np.ndarrays instead of tf.Tensors for the representative dataset.
    rng = np.random.default_rng(seed=1234)
    representative_dataset = [
        {'input_tensor': rng.uniform(size=(1, 1024)).astype(np.float32)}
        for _ in range(4)
    ]

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options=quantization_options,
        representative_dataset=representative_dataset,
    )

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )
    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def
    self.assertTrue(self._contains_quantized_function_call(output_graphdef))

  @test_util.run_in_graph_and_eager_modes
  def test_model_ptq_use_python_list_representative_dataset(self):
    self._create_matmul_model(
        input_shape=(1, 1024),
        weight_shape=(1024, 3),
        saved_model_path=self._input_saved_model_path,
    )

    tags = {tag_constants.SERVING}
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=quant_opts_pb2.TF,
    )

    # Use plain python lists as representative samples.
    representative_dataset = [
        {
            'input_tensor': [[i * 0.1 for i in range(1024)]],
        }
        for _ in range(4)
    ]

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options=quantization_options,
        representative_dataset=representative_dataset,
    )

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )
    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def
    self.assertTrue(self._contains_quantized_function_call(output_graphdef))

  @test_util.run_in_graph_and_eager_modes
  def test_model_ptq_use_representative_samples_file(self):
    self._create_matmul_model(
        input_shape=(1, 1024),
        weight_shape=(1024, 3),
        saved_model_path=self._input_saved_model_path,
    )

    rng = np.random.default_rng(seed=1234)
    representative_dataset: repr_dataset.RepresentativeDataset = [
        {'input_tensor': rng.uniform(size=(1, 1024)).astype(np.float32)}
        for _ in range(4)
    ]
    dataset_file_map = repr_dataset.TfRecordRepresentativeDatasetSaver(
        {'serving_default': os.path.join(self._input_saved_model_path, 'repr')}
    ).save({'serving_default': representative_dataset})

    tags = {tag_constants.SERVING}
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=quant_opts_pb2.TF,
        representative_datasets=dataset_file_map,
    )

    with self.assertRaisesRegex(
        ValueError,
        'Do not specify both the `representative_dataset` argument and the'
        ' `representative_datasets` field in `QuantizationOptions`',
    ):
      quantize_model.quantize(
          self._input_saved_model_path,
          self._output_saved_model_path,
          quantization_options=quantization_options,
          representative_dataset=representative_dataset,
      )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options=quantization_options,
    )
    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )
    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def
    self.assertTrue(self._contains_quantized_function_call(output_graphdef))

  @test_util.run_in_graph_and_eager_modes
  def test_model_ptq_call_twice(self):
    self._create_matmul_model(
        input_shape=(1, 1024),
        weight_shape=(1024, 3),
        saved_model_path=self._input_saved_model_path,
    )

    tags = {tag_constants.SERVING}
    signature_def_keys = [signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=signature_def_keys,
        op_set=quant_opts_pb2.TF,
    )

    representative_dataset: repr_dataset.RepresentativeDataset = [
        {
            'input_tensor': random_ops.random_uniform(shape=(1, 1024)),
        }
        for _ in range(8)
    ]

    # Test the first run.
    converted_model_1 = quantize_model.quantize(
        self._input_saved_model_path,
        output_directory=self._output_saved_model_path,
        quantization_options=quantization_options,
        representative_dataset=representative_dataset,
    )

    self.assertIsNotNone(converted_model_1)
    self.assertCountEqual(
        converted_model_1.signatures._signatures.keys(), signature_def_keys
    )
    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def
    self.assertTrue(self._contains_quantized_function_call(output_graphdef))

    # Test the second run on the same model.
    converted_model_2 = quantize_model.quantize(
        self._input_saved_model_path,
        output_directory=self._output_saved_model_path_2,
        quantization_options=quantization_options,
        representative_dataset=representative_dataset,
    )

    self.assertIsNotNone(converted_model_2)
    self.assertCountEqual(
        converted_model_2.signatures._signatures.keys(), signature_def_keys
    )
    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path_2
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def
    self.assertTrue(self._contains_quantized_function_call(output_graphdef))

  def test_model_ptq_preserving_assets_extra(self):
    self._create_matmul_model(
        input_shape=(1, 1024),
        weight_shape=(1024, 3),
        saved_model_path=self._input_saved_model_path,
    )
    asset_filename = 'assets.extra/tf_serving_warmup_requests'
    file_io.create_dir_v2(
        os.path.join(self._input_saved_model_path, 'assets.extra')
    )
    file_io.write_string_to_file(
        filename=os.path.join(self._input_saved_model_path, asset_filename),
        file_content='Test content',
    )

    tags = {tag_constants.SERVING}
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=quant_opts_pb2.TF,
    )

    # Use plain python lists as representative samples.
    representative_dataset = [
        {
            'input_tensor': [[i * 0.1 for i in range(1024)]],
        }
        for _ in range(4)
    ]

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options=quantization_options,
        representative_dataset=representative_dataset,
    )
    self.assertIsNotNone(converted_model)
    # Check if the assets.extra file exists in the output model.
    self.assertTrue(
        file_io.file_exists_v2(
            os.path.join(self._output_saved_model_path, asset_filename)
        )
    )

  # tf.data.Dataset is as an Iterable (thus can be used as representative
  # dataset) only in TF2 (eager mode).
  @test_util.run_v2_only
  def test_model_ptq_use_tf_dataset_for_representative_dataset(self):
    self._create_matmul_model(
        input_shape=(1, 1024),
        weight_shape=(1024, 3),
        saved_model_path=self._input_saved_model_path,
    )

    tags = {tag_constants.SERVING}
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=quant_opts_pb2.TF,
    )

    representative_samples = [
        {
            'input_tensor': random_ops.random_uniform(shape=(1, 1024)),
        }
        for _ in range(8)
    ]

    # Construct a tf.data.Dataset from the representative samples.
    representative_dataset = dataset_ops.DatasetV2.from_generator(
        lambda: representative_samples,
        output_signature={
            'input_tensor': tensor_spec.TensorSpec(
                shape=(1, 1024), dtype=dtypes.float32
            ),
        },
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options=quantization_options,
        representative_dataset=representative_dataset,
    )

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )
    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def
    self.assertTrue(self._contains_quantized_function_call(output_graphdef))

  @test_util.run_in_graph_and_eager_modes
  def test_model_ptq_no_representative_sample_not_quantized(self):
    self._create_matmul_model(
        input_shape=(1, 1024),
        weight_shape=(1024, 3),
        saved_model_path=self._input_saved_model_path,
    )

    tags = {tag_constants.SERVING}
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
        # Put no sample into the representative dataset to make calibration
        # impossible.
        representative_dataset=[],
    )

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )
    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def
    # Model is not quantized because there was no sample data for calibration.
    self.assertFalse(self._contains_quantized_function_call(output_graphdef))

  @test_util.run_in_graph_and_eager_modes
  def test_model_ptq_with_uncalibrated_subgraph(self):
    class IfModel(module.Module):
      """A model that contains a branching op."""

      def __init__(self):
        self.filters_0 = np.random.uniform(
            low=-1.0, high=1.0, size=(4, 3)
        ).astype('f4')
        self.bias_0 = np.random.uniform(low=-1.0, high=1.0, size=(3,)).astype(
            'f4'
        )

        self.filters_1 = np.random.uniform(
            low=-1.0, high=1.0, size=(4, 3)
        ).astype('f4')
        self.bias_1 = np.random.uniform(low=-1.0, high=1.0, size=(3,)).astype(
            'f4'
        )

      @def_function.function(
          input_signature=[
              tensor_spec.TensorSpec(shape=[1, 4], dtype=dtypes.float32)
          ]
      )
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
          out = math_ops.matmul(x, self.filters_0)
          out = nn_ops.bias_add(out, self.bias_0)
          return {'output': out}

        out = math_ops.matmul(x, self.filters_1)
        out = nn_ops.bias_add(out, self.bias_1)
        return {'output': out}

    model = IfModel()
    saved_model_save.save(model, self._input_saved_model_path)

    def data_gen() -> repr_dataset.RepresentativeDataset:
      for _ in range(8):
        yield {
            'x': ops.convert_to_tensor(
                np.random.uniform(low=0.0, high=1.0, size=(1, 4)).astype('f4')
            ),
        }

    tags = {tag_constants.SERVING}

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=quant_opts_pb2.TF,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
        representative_dataset=data_gen(),
    )

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )
    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def
    self.assertTrue(self._contains_quantized_function_call(output_graphdef))

    # Tests that the false branch contains a quantized function call whereas the
    # true branch doesn't.
    def _is_quantized_function_call_node(
        node_def: node_def_pb2.NodeDef,
    ) -> bool:
      return node_def.op == 'PartitionedCall' and node_def.attr[
          'f'
      ].func.name.startswith('quantized_')

    for func in output_graphdef.library.function:
      if func.signature.name.startswith('cond_false'):
        self.assertTrue(
            any(map(_is_quantized_function_call_node, func.node_def))
        )
      elif func.signature.name.startswith('cond_true'):
        self.assertFalse(
            any(map(_is_quantized_function_call_node, func.node_def))
        )

  # Run this test only with the eager mode.
  @test_util.run_v2_only
  def test_ptq_model_with_multiple_signatures(self):
    # Create and save a model having 2 signatures.
    model = MultipleSignatureModel()

    signatures = {
        'sig1': model.matmul.get_concrete_function(
            tensor_spec.TensorSpec(shape=(1, 4), dtype=dtypes.float32)
        ),
        'sig2': model.conv.get_concrete_function(
            tensor_spec.TensorSpec(shape=(1, 3, 4, 3), dtype=dtypes.float32)
        ),
    }
    saved_model_save.save(
        model, self._input_saved_model_path, signatures=signatures
    )

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

    dataset_file_map = repr_dataset.TfRecordRepresentativeDatasetSaver({
        'sig1': os.path.join(self._input_saved_model_path, 'sig1_repr'),
        'sig2': os.path.join(self._input_saved_model_path, 'sig2_repr'),
    }).save({
        'sig1': data_gen_sig1(),
        'sig2': data_gen_sig2(),
    })
    tags = {tag_constants.SERVING}
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=['sig1', 'sig2'],
        op_set=quant_opts_pb2.TF,
        representative_datasets=dataset_file_map,
    )
    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        output_directory=self._output_saved_model_path,
        quantization_options=quantization_options,
    )
    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'sig1', 'sig2'}
    )

    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def
    self.assertTrue(self._contains_quantized_function_call(output_graphdef))

  # Run this test only with the eager mode.
  @test_util.run_v2_only
  def test_ptq_multiple_signatures_invalid_dataset_raises_value_error(self):
    # Create and save a model having 2 signatures.
    model = MultipleSignatureModel()

    signatures = {
        'sig1': model.matmul.get_concrete_function(
            tensor_spec.TensorSpec(shape=(1, 4), dtype=dtypes.float32)
        ),
        'sig2': model.conv.get_concrete_function(
            tensor_spec.TensorSpec(shape=(1, 3, 4, 3), dtype=dtypes.float32)
        ),
    }
    saved_model_save.save(
        model, self._input_saved_model_path, signatures=signatures
    )

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags={tag_constants.SERVING},
        signature_keys=['sig1', 'sig2'],
    )

    # Use a dict-style samples instead of tuple-style samples. This is invalid
    # because for a model multiple signatures one must use tuple-style samples.
    invalid_dataset: repr_dataset.RepresentativeDataset = [
        {'matmul_input': random_ops.random_uniform(shape=(1, 4))}
        for _ in range(8)
    ]

    with self.assertRaisesRegex(
        Exception, 'Representative dataset is not a mapping'
    ):
      quantize_model.quantize(
          self._input_saved_model_path,
          output_directory=self._output_saved_model_path,
          quantization_options=quantization_options,
          representative_dataset=invalid_dataset,
      )

  @test_util.run_in_graph_and_eager_modes
  def test_ptq_model_with_tf1_saved_model_with_variable_for_conv2d(self):
    signature_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    tags = {tag_constants.SERVING}

    input_placeholder = self._create_and_save_tf1_conv_model(
        self._input_saved_model_path,
        signature_key,
        tags,
        input_key='x',
        output_key='output',
        use_variable=True,
    )

    signature_keys = [signature_key]

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=signature_keys,
        op_set=quant_opts_pb2.TF,
    )

    data_gen = self._create_data_generator(
        input_key='x', shape=input_placeholder.shape
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
        representative_dataset=data_gen,
    )

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), signature_keys
    )

    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def
    self.assertTrue(self._contains_quantized_function_call(output_graphdef))

  @parameterized.named_parameters(
      ('use_constant_with_int32_input', dtypes.int32, False),
      ('use_variable_with_int32_input', dtypes.int32, True),
      ('use_constant_with_int64_input', dtypes.int64, False),
      ('use_variable_with_int64_input', dtypes.int64, True),
  )
  @test_util.run_in_graph_and_eager_modes
  def test_ptq_model_with_tf1_saved_model_with_variable_for_gather(
      self, input_type, use_variable
  ):
    signature_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    tags = {tag_constants.SERVING}

    input_placeholder = self._create_and_save_tf1_gather_model(
        self._input_saved_model_path,
        signature_key,
        tags,
        input_key='x',
        output_key='output',
        input_type=input_type,
        use_variable=use_variable,
    )

    signature_keys = [signature_key]

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=signature_keys,
        op_set=quant_opts_pb2.TF,
    )

    data_gen = self._create_data_generator(
        input_key='x',
        shape=input_placeholder.shape,
        minval=0,
        maxval=10,
        dtype=input_type,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
        representative_dataset=data_gen,
    )

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), signature_keys
    )

    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def
    self.assertTrue(self._contains_quantized_function_call(output_graphdef))

  def test_ptq_model_with_variable_tf1_saved_model_unfreeze_constants(self):
    signature_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    tags = {tag_constants.SERVING}

    input_placeholder = self._create_and_save_tf1_conv_model(
        self._input_saved_model_path,
        signature_key,
        tags,
        input_key='x',
        output_key='output',
        input_shape=(1, 16, 16, 8),
        # Uses large filter to exceed the constant size threshold of 64KiB
        # (specified by `kDefaultConstantSizeThresholdInBytes`) for unfreezing.
        filter_shape=(256, 8, 8, 16),
        use_variable=True,
    )

    signature_keys = [signature_key]

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=signature_keys,
        op_set=quant_opts_pb2.TF,
        freeze_all_variables=False,
    )

    repr_ds = self._create_data_generator(
        input_key='x', shape=input_placeholder.shape, num_examples=2
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
        representative_dataset=repr_ds,
    )
    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )

    # Confirm that the quantized model loads successfully.
    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )

    with session.Session(graph=ops.Graph()) as sess:
      output_meta_graph_def = output_loader.load(sess, tags)

    # Checks that quantization is applied.
    output_graphdef = output_meta_graph_def.graph_def
    self.assertTrue(self._contains_quantized_function_call(output_graphdef))

    # Tests that there are variables in the model.
    variable_node_defs = _find_variables(output_graphdef)
    self.assertLen(variable_node_defs, 1)

    # Reads the variables from the checkpoint file and matches with the
    # variables found in the graph.
    checkpoint_path = os.path.join(
        self._output_saved_model_path, 'variables', 'variables'
    )
    var_name_and_shapes = checkpoint_utils.list_variables(checkpoint_path)

    # Checks that each variable's name and shape match.
    self.assertEqual(len(variable_node_defs), len(var_name_and_shapes))
    for var_name, shape in var_name_and_shapes:
      self.assertIn(var_name, variable_node_defs)
      self.assertEqual(
          shape,
          tensor_shape.TensorShape(
              variable_node_defs[var_name].attr['shape'].shape
          ),
      )

  @test_util.run_in_graph_and_eager_modes
  def test_ptq_model_with_tf1_saved_model(self):
    tags = {tag_constants.SERVING}
    signature_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY

    input_placeholder = self._create_and_save_tf1_conv_model(
        self._input_saved_model_path,
        signature_key,
        tags,
        input_key='p',
        output_key='output',
        use_variable=False,
    )

    signature_keys = [signature_key]

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=signature_keys,
        op_set=quant_opts_pb2.TF,
    )

    data_gen = self._create_data_generator(
        input_key='p', shape=input_placeholder.shape
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
        representative_dataset=data_gen,
    )

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), signature_keys
    )

    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def
    self.assertTrue(self._contains_quantized_function_call(output_graphdef))

  @test_util.run_in_graph_and_eager_modes
  def test_ptq_model_with_tf1_saved_model_multiple_signatures(self):
    tags = {tag_constants.SERVING}

    # Create two models and add them to a same SavedModel under different
    # signature keys.
    with ops.Graph().as_default(), session.Session() as sess:
      in_placeholder_1, output_tensor_1 = self._create_simple_tf1_conv_model()
      sig_def_1 = signature_def_utils_impl.predict_signature_def(
          inputs={'x1': in_placeholder_1}, outputs={'output1': output_tensor_1}
      )

      in_placeholder_2, output_tensor_2 = self._create_simple_tf1_conv_model()
      sig_def_2 = signature_def_utils_impl.predict_signature_def(
          inputs={'x2': in_placeholder_2}, outputs={'output2': output_tensor_2}
      )

      v1_builder = builder.SavedModelBuilder(self._input_saved_model_path)
      v1_builder.add_meta_graph_and_variables(
          sess,
          tags,
          signature_def_map={
              'sig1': sig_def_1,
              'sig2': sig_def_2,
          },
      )

      v1_builder.save()

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=['sig1', 'sig2'],
        op_set=quant_opts_pb2.TF,
    )

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
        self._input_saved_model_path,
        output_directory=self._output_saved_model_path,
        quantization_options=quantization_options,
        representative_dataset={
            'sig1': data_gen_sig1(),
            'sig2': data_gen_sig2(),
        },
    )

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'sig1', 'sig2'}
    )

    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def
    self.assertTrue(self._contains_quantized_function_call(output_graphdef))

  @test_util.run_in_graph_and_eager_modes
  def test_ptq_model_with_tf1_saved_model_invalid_input_key_raises_value_error(
      self,
  ):
    signature_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    tags = {tag_constants.SERVING}

    input_placeholder = self._create_and_save_tf1_conv_model(
        self._input_saved_model_path,
        signature_key,
        tags,
        input_key='x',
        output_key='output',
        use_variable=False,
    )

    signature_keys = [signature_key]

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=signature_keys,
    )

    # Representative generator function that yields with an invalid input key.
    invalid_data_gen = self._create_data_generator(
        input_key='invalid_input_key', shape=input_placeholder.shape
    )

    with self.assertRaisesRegex(
        Exception,
        'Invalid input keys for representative sample.',
    ):
      quantize_model.quantize(
          self._input_saved_model_path,
          self._output_saved_model_path,
          quantization_options,
          representative_dataset=invalid_data_gen,
      )

  @test_util.run_in_graph_and_eager_modes
  def test_ptq_model_with_non_default_tags(self):
    signature_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    # Use a different set of tags other than {"serve"}.
    tags = {tag_constants.TRAINING, tag_constants.GPU}

    # Non-default tags are usually used when saving multiple metagraphs in TF1.
    input_placeholder = self._create_and_save_tf1_conv_model(
        self._input_saved_model_path,
        signature_key,
        tags,
        input_key='input',
        output_key='output',
        use_variable=True,
    )

    signature_keys = [signature_key]

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=signature_keys,
        op_set=quant_opts_pb2.TF,
    )

    data_gen = self._create_data_generator(
        input_key='input', shape=input_placeholder.shape
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
        representative_dataset=data_gen,
    )

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), signature_keys
    )

    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def
    self.assertTrue(self._contains_quantized_function_call(output_graphdef))

  @test_util.run_in_graph_and_eager_modes
  def test_ptq_model_with_wrong_tags_raises_error(self):
    signature_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    save_tags = {tag_constants.TRAINING, tag_constants.GPU}

    input_placeholder = self._create_and_save_tf1_conv_model(
        self._input_saved_model_path,
        signature_key,
        save_tags,
        input_key='input',
        output_key='output',
        use_variable=True,
    )

    # Try to use a different set of tags to quantize.
    tags = {tag_constants.SERVING}
    signature_keys = [signature_key]
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=signature_keys,
    )

    data_gen = self._create_data_generator(
        input_key='input', shape=input_placeholder.shape
    )
    with self.assertRaisesRegex(
        RuntimeError,
        "MetaGraphDef associated with tags {'serve'} could not be found",
    ):
      quantize_model.quantize(
          self._input_saved_model_path,
          self._output_saved_model_path,
          quantization_options,
          representative_dataset=data_gen,
      )

  def test_ptq_vocab_table_lookup_model(self):
    tags = {tag_constants.SERVING}
    signature_def_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY

    # Create and save a simple model that involves a hash table.
    inputs, outputs = self._create_and_save_vocab_table_lookup_model_tf1(
        self._input_saved_model_path, tags, signature_def_key
    )

    # Make sure that the desired input key and output key is present.
    self.assertIn('input_vocabs', inputs.keys())
    self.assertIn('lookup', outputs.keys())

    # Representative dataset is composed of a set of vocabs for table lookup.
    repr_ds = [
        {'input_vocabs': np.array([b'hello', b'model', b'quantization'])}
        for _ in range(4)
    ]

    signature_def_keys = [signature_def_key]
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=signature_def_keys,
        op_set=quant_opts_pb2.TF,
    )

    quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
        representative_dataset=repr_ds,
    )

    # Tests table lookup to make sure the table has been initialized
    # successfully.
    with session.Session(graph=ops.Graph()) as sess:
      output_meta_graph_def = saved_model_loader.load(
          sess, tags=tags, export_dir=self._output_saved_model_path
      )

      # The graph should contain a quantized function call (it contains a
      # single f32 matmul node).
      self.assertTrue(
          self._contains_quantized_function_call(
              output_meta_graph_def.graph_def
          )
      )
      self.assertCountEqual(
          output_meta_graph_def.signature_def.keys(), signature_def_keys
      )

      signature_def = output_meta_graph_def.signature_def[signature_def_key]

      input_tensor_name = signature_def.inputs['input_vocabs'].name
      input_tensor = sess.graph.get_tensor_by_name(input_tensor_name)

      lookup_tensor_name = signature_def.outputs['lookup'].name
      lookup_tensor = sess.graph.get_tensor_by_name(lookup_tensor_name)

      lookup_val = sess.run(
          lookup_tensor,
          feed_dict={
              input_tensor: np.array([b'model', b'quantization', b'hello'])
          },
      )

      self.assertAllClose(lookup_val, [1.0, 2.0, 0.0])

  def test_ptq_file_init_hash_table_lookup_model(self):
    tags = {tag_constants.SERVING}
    signature_def_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY

    # Create and save a simple model that involves a hash table.
    inputs, outputs = self._create_and_save_file_init_hash_table_model_tf1(
        self._input_saved_model_path, tags, signature_def_key
    )

    # Make sure that the desired input key and output key is present.
    self.assertIn('input_vocabs', inputs.keys())
    self.assertIn('lookup', outputs.keys())

    # Representative dataset is composed of a set of vocabs for table lookup.
    repr_ds = [
        {'input_vocabs': np.array([b'static', b'range', b'quantization'])}
        for _ in range(4)
    ]

    signature_def_keys = [signature_def_key]
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=signature_def_keys,
        op_set=quant_opts_pb2.TF,
    )

    quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
        representative_dataset=repr_ds,
    )

    # Tests table lookup to make sure the table has been initialized
    # successfully.
    with session.Session(graph=ops.Graph()) as sess:
      output_meta_graph_def = saved_model_loader.load(
          sess, tags=tags, export_dir=self._output_saved_model_path
      )

      # The graph should contain a quantized function call (it contains a
      # single f32 matmul node).
      self.assertTrue(
          self._contains_quantized_function_call(
              output_meta_graph_def.graph_def
          )
      )
      self.assertCountEqual(
          output_meta_graph_def.signature_def.keys(), signature_def_keys
      )

      signature_def = output_meta_graph_def.signature_def[signature_def_key]

      input_tensor_name = signature_def.inputs['input_vocabs'].name
      input_tensor = sess.graph.get_tensor_by_name(input_tensor_name)

      lookup_tensor_name = signature_def.outputs['lookup'].name
      lookup_tensor = sess.graph.get_tensor_by_name(lookup_tensor_name)

      lookup_val = sess.run(
          lookup_tensor,
          feed_dict={
              input_tensor: np.array([b'dynamic', b'quantization', b'range'])
          },
      )

      # "dynamic" is not in the table: -1 (default value)
      self.assertAllClose(lookup_val, [-1.0, 2.0, 1.0])

  @parameterized.named_parameters(
      ('none', None, False, False, quant_opts_pb2.TF, False, 'SAME'),
      ('relu', nn_ops.relu, False, False, quant_opts_pb2.TF, False, 'SAME'),
      ('relu6', nn_ops.relu6, False, False, quant_opts_pb2.TF, False, 'SAME'),
      ('with_bias', None, True, False, quant_opts_pb2.TF, False, 'SAME'),
      (
          'with_bias_and_relu',
          nn_ops.relu,
          True,
          False,
          quant_opts_pb2.TF,
          False,
          'SAME',
      ),
      (
          'with_bias_and_relu6',
          nn_ops.relu6,
          True,
          False,
          quant_opts_pb2.TF,
          False,
          'SAME',
      ),
      ('none_to_xla', None, False, False, quant_opts_pb2.XLA, False, 'SAME'),
      (
          'with_bias_and_relu6_to_xla',
          nn_ops.relu6,
          True,
          False,
          quant_opts_pb2.XLA,
          False,
          'SAME',
      ),
      (
          'with_bias_to_xla_dynamic',
          None,
          True,
          False,
          quant_opts_pb2.XLA,
          True,
          'SAME',
      ),
      (
          'none_to_xla_padding_valid',
          None,
          False,
          False,
          quant_opts_pb2.XLA,
          False,
          'VALID',
      ),
      (
          'with_bias_and_relu6_to_xla_padding_valid',
          nn_ops.relu6,
          True,
          False,
          quant_opts_pb2.XLA,
          False,
          'VALID',
      ),
      (
          'with_bias_to_xla_dynamic_padding_valid',
          None,
          True,
          False,
          quant_opts_pb2.XLA,
          True,
          'VALID',
      ),
  )
  def test_conv3d_ptq_model(
      self,
      activation_fn: Optional[ops.Operation],
      has_bias: bool,
      has_batch_norm: bool,
      target_opset: quant_opts_pb2.OpSet,
      input_shape_dynamic: bool,
      padding: str,
  ):
    input_shape = [1, 3, 4, 3, 3]
    if input_shape_dynamic:
      input_shape = [None, None, None, None, 3]

    class ConvModel(module.Module):

      def __init__(self):
        self.filters = np.random.uniform(
            low=-0.5, high=0.5, size=(2, 3, 3, 3, 2)
        ).astype('f4')
        self.bias = np.random.uniform(low=0.0, high=0.2, size=(2)).astype('f4')

      @def_function.function(
          input_signature=[
              tensor_spec.TensorSpec(shape=input_shape, dtype=dtypes.float32)
          ]
      )
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
            data_format='NDHWC',
        )
        if has_bias:
          out = nn_ops.bias_add(out, self.bias)
        if activation_fn is not None:
          out = activation_fn(out)
        return {'output': out}

    model = ConvModel()
    saved_model_save.save(model, self._input_saved_model_path)

    repr_ds = []
    for _ in range(500):
      repr_ds.append({
          'input_tensor': ops.convert_to_tensor(
              np.random.uniform(
                  low=-0.1, high=0.2, size=(1, 3, 4, 3, 3)
              ).astype('f4')
          ),
      })

    signature_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    tags = {tag_constants.SERVING}

    # Check the converted model with TF opset as the baseline.
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=[signature_key],
        op_set=quant_opts_pb2.TF,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
        representative_dataset=repr_ds,
    )
    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {signature_key}
    )

    input_data = np.random.uniform(
        low=-0.1, high=0.2, size=(1, 3, 4, 3, 3)
    ).astype('f4')
    expected_outputs = model.conv3d(input_data)
    got_outputs = converted_model.signatures[signature_key](
        input_tensor=ops.convert_to_tensor(input_data)
    )
    self.assertAllClose(expected_outputs, got_outputs, atol=0.00494)

    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def
    self.assertTrue(self._contains_quantized_function_call(output_graphdef))

    # Check the converted model in the target opset.
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=[signature_key],
        op_set=target_opset,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path_2,
        quantization_options,
        representative_dataset=repr_ds,
    )

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {signature_key}
    )
    loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path_2
    )
    graphdef = loader.get_meta_graph_def_from_tags(tags).graph_def
    if target_opset == quant_opts_pb2.XLA:
      self.assertTrue(self._contains_op(graphdef, 'XlaConvV2'))

    new_outputs = converted_model.signatures[signature_key](
        input_tensor=ops.convert_to_tensor(input_data)
    )
    # The quantized model in XLA opset is expected to have similar fidelity
    # compared to the quantized model in TF opset.
    self.assertAllClose(new_outputs, got_outputs, atol=0.00306)
    self.assertAllClose(new_outputs, expected_outputs, atol=0.00494)

  # Tests the case of having a signature key of `main` because it is a
  # special name in the TF quantizer's MLIR pipeline that should be treated
  # with care.
  @test_util.run_in_graph_and_eager_modes
  def test_ptq_model_with_signature_key_main(self):
    signature_key = 'main'
    tags = {tag_constants.SERVING}

    input_placeholder = self._create_and_save_tf1_conv_model(
        self._input_saved_model_path,
        signature_key,
        tags,
        input_key='x',
        output_key='output',
        use_variable=True,
    )

    signature_keys = [signature_key]
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=signature_keys,
        op_set=quant_opts_pb2.TF,
    )

    data_gen = self._create_data_generator(
        input_key='x', shape=input_placeholder.shape
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
        representative_dataset=data_gen,
    )

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), signature_keys
    )

    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def
    self.assertTrue(self._contains_quantized_function_call(output_graphdef))

    # Makes sure that the original function identified by the signature key
    # `main` is renamed to `main_0` (see `InsertMainFunctionPass` for details).
    self.assertTrue(
        any(
            map(
                lambda func: func.signature.name == 'main_0',
                output_graphdef.library.function,
            )
        )
    )


class DynamicRangeQuantizationTest(quantize_model_test_base.QuantizedModelTest):
  """Test cases for dynamic range quantization.

  Tries to run all tests cases in both the graph mode (default in TF1) and the
  eager mode (default in TF2) to ensure support for when TF2 is disabled.
  """

  @parameterized.parameters(
      (True, quant_opts_pb2.XLA),
      (False, quant_opts_pb2.XLA),
      (True, quant_opts_pb2.UNIFORM_QUANTIZED),
      (False, quant_opts_pb2.UNIFORM_QUANTIZED),
  )
  @test_util.run_in_graph_and_eager_modes
  def test_einsum_model(
      self,
      constant_y_operand: bool,
      target_opset: quant_opts_pb2.OpSet,
  ):
    equation = 'abc,cde->abde'
    _, y_shape, bias_shape, x_signature, y_signature = (
        self._prepare_sample_einsum_datashapes(equation, use_bias=True)
    )

    model = self._create_einsum_model(
        equation,
        y_shape,
        x_signature,
        y_signature,
        bias_shape,
        activation_fn=nn_ops.relu,
    )

    if constant_y_operand:
      signatures = {
          'serving_default': model.einsum_with_kernel.get_concrete_function(),
      }
    else:
      signatures = {
          'serving_default': (
              model.einsum_without_kernel.get_concrete_function()
          ),
      }

    saved_model_save.save(model, self._input_saved_model_path, signatures)

    tags = {tag_constants.SERVING}
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_DYNAMIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=target_opset,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
    )
    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )

    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def

    # TODO(b/286489783): Support Einsum
    if target_opset == quant_opts_pb2.UNIFORM_QUANTIZED:
      self.assertFalse(self._contains_op(output_graphdef, 'XlaDotV2'))
      self.assertTrue(self._contains_op(output_graphdef, 'BatchMatMulV2'))
    else:
      self.assertFalse(self._contains_op(output_graphdef, 'XlaDotV2'))
      self.assertTrue(self._contains_op(output_graphdef, 'Einsum'))

  @parameterized.named_parameters(
      ('to_tf_per_tensor', quant_opts_pb2.TF, False),
      ('to_xla_per_tensor', quant_opts_pb2.XLA, False),
      (
          'to_uniform_quantized_per_tensor',
          quant_opts_pb2.UNIFORM_QUANTIZED,
          False,
      ),
      (
          'to_uniform_quantized_per_channel',
          quant_opts_pb2.UNIFORM_QUANTIZED,
          True,
      ),
  )
  @test_util.run_in_graph_and_eager_modes
  def test_matmul_model(
      self,
      target_opset: quant_opts_pb2.OpSet,
      enable_per_channel_quantization: bool,
  ):
    self._create_matmul_model(
        input_shape=(1, 1024),
        weight_shape=(1024, 3),
        saved_model_path=self._input_saved_model_path,
    )

    tags = {tag_constants.SERVING}

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_DYNAMIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=target_opset,
        enable_per_channel_quantization=enable_per_channel_quantization,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
    )
    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )

    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def

    if target_opset == quant_opts_pb2.UNIFORM_QUANTIZED:
      self.assertTrue(
          self._contains_op(output_graphdef, 'UniformQuantizedDotHybrid')
      )
      self.assertFalse(self._contains_op(output_graphdef, 'MatMul'))
      if enable_per_channel_quantization:
        quantized_axis_attr = attr_value_pb2.AttrValue(i=-1)
        self.assertTrue(
            self._contains_op(
                output_graphdef,
                'UniformQuantizedDotHybrid',
                'rhs_quantization_axis',
                quantized_axis_attr,
            )
        )
    elif target_opset == quant_opts_pb2.XLA:
      self.assertTrue(self._contains_op(output_graphdef, 'XlaDotV2'))
      self.assertFalse(self._contains_op(output_graphdef, 'MatMul'))
    else:
      self.assertTrue(self._contains_quantized_function_call(output_graphdef))
      self.assertTrue(self._contains_op(output_graphdef, 'MatMul'))

  @parameterized.named_parameters(
      ('to_tf_per_tensor', quant_opts_pb2.TF, False),
      ('to_xla_per_tensor', quant_opts_pb2.XLA, False),
      (
          'to_uniform_quantized_per_tensor',
          quant_opts_pb2.UNIFORM_QUANTIZED,
          False,
      ),
      (
          'to_uniform_quantized_per_channel',
          quant_opts_pb2.UNIFORM_QUANTIZED,
          True,
      ),
  )
  @test_util.run_in_graph_and_eager_modes
  def test_conv_model(
      self,
      target_opset: quant_opts_pb2.OpSet,
      enable_per_channel_quantization: bool,
  ):
    filter_shape = (2, 3, 512, 2)

    model = self._create_conv2d_model(
        input_shape=(1, 3, 4, 512),
        filter_shape=filter_shape,
        has_bias=True,
        has_batch_norm=True,
        activation_fn=nn_ops.relu6,
    )

    saved_model_save.save(model, self._input_saved_model_path)

    tags = {tag_constants.SERVING}

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_DYNAMIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=target_opset,
        enable_per_channel_quantization=enable_per_channel_quantization,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
    )

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )

    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def

    if enable_per_channel_quantization:
      quantized_axis = 3
      quantized_axis_attr = attr_value_pb2.AttrValue(i=quantized_axis)
      quantized_dim_size_attr = attr_value_pb2.AttrValue(
          list=attr_value_pb2.AttrValue.ListValue(
              shape=[
                  tensor_shape_pb2.TensorShapeProto(
                      dim=[
                          tensor_shape_pb2.TensorShapeProto.Dim(
                              size=filter_shape[quantized_axis]
                          )
                      ]
                  )
              ]
          )
      )

    if target_opset == quant_opts_pb2.UNIFORM_QUANTIZED:
      self.assertTrue(
          self._contains_op(
              output_graphdef, 'UniformQuantizedConvolutionHybrid'
          )
      )
      self.assertFalse(self._contains_op(output_graphdef, 'Conv2D'))
      if enable_per_channel_quantization:
        self.assertTrue(
            self._contains_op(
                output_graphdef,
                'UniformQuantizedConvolutionHybrid',
                'rhs_quantization_axis',
                quantized_axis_attr,
            )
        )
        self.assertTrue(
            self._contains_op(
                output_graphdef,
                'Const',
                '_output_shapes',
                quantized_dim_size_attr,
            )
        )
    elif target_opset == quant_opts_pb2.XLA:
      self.assertTrue(self._contains_op(output_graphdef, 'XlaConvV2'))
      self.assertFalse(self._contains_op(output_graphdef, 'Conv2D'))
    else:
      self.assertTrue(self._contains_quantized_function_call(output_graphdef))
      self.assertTrue(self._contains_op(output_graphdef, 'Conv2D'))

  @parameterized.named_parameters(
      ('to_tf_per_tensor', quant_opts_pb2.TF, False),
      ('to_xla_per_tensor', quant_opts_pb2.XLA, False),
      (
          'to_uniform_quantized_per_tensor',
          quant_opts_pb2.UNIFORM_QUANTIZED,
          False,
      ),
      (
          'to_uniform_quantized_per_channel',
          quant_opts_pb2.UNIFORM_QUANTIZED,
          True,
      ),
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
        input_shape=(1, 3, 4, 1024), filter_shape=filter_shape, strides=strides
    )

    saved_model_save.save(model, self._input_saved_model_path)

    tags = [tag_constants.SERVING]

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_DYNAMIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=target_opset,
        enable_per_channel_quantization=enable_per_channel_quantization,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
    )

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )

    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def

    # Uniform Quantized op takes only the first and the second values for
    # strides.
    strides_to_check = (
        (strides[1], strides[2])
        if target_opset == quant_opts_pb2.UNIFORM_QUANTIZED
        else strides
    )
    strides_attr = attr_value_pb2.AttrValue(
        list=attr_value_pb2.AttrValue.ListValue(i=strides_to_check)
    )

    if enable_per_channel_quantization:
      quantized_axis_attr = attr_value_pb2.AttrValue(i=3)
      quantized_dim_size_attr = attr_value_pb2.AttrValue(
          list=attr_value_pb2.AttrValue.ListValue(
              shape=[
                  tensor_shape_pb2.TensorShapeProto(
                      dim=[
                          tensor_shape_pb2.TensorShapeProto.Dim(
                              size=filter_shape[2] * filter_shape[3]
                          )
                      ]
                  )
              ]
          )
      )

    if target_opset == quant_opts_pb2.UNIFORM_QUANTIZED:
      self.assertTrue(
          self._contains_op(
              output_graphdef,
              'UniformQuantizedConvolutionHybrid',
              'window_strides',
              strides_attr,
          )
      )
      self.assertFalse(
          self._contains_op(output_graphdef, 'DepthwiseConv2dNative')
      )
      if enable_per_channel_quantization:
        self.assertTrue(
            self._contains_op(
                output_graphdef,
                'UniformQuantizedConvolutionHybrid',
                'rhs_quantization_axis',
                quantized_axis_attr,
            )
        )
        self.assertTrue(
            self._contains_op(
                output_graphdef,
                'Const',
                '_output_shapes',
                quantized_dim_size_attr,
            )
        )
    elif target_opset == quant_opts_pb2.XLA:
      self.assertTrue(self._contains_op(output_graphdef, 'XlaConvV2'))
      self.assertFalse(
          self._contains_op(output_graphdef, 'DepthwiseConv2dNative')
      )
    else:
      self.assertTrue(self._contains_quantized_function_call(output_graphdef))
      self.assertTrue(
          self._contains_op(
              output_graphdef, 'DepthwiseConv2dNative', 'strides', strides_attr
          )
      )

  @parameterized.named_parameters(
      ('to_tf_use_constant', quant_opts_pb2.TF, False),
      ('to_xla_use_constant', quant_opts_pb2.XLA, False),
      (
          'to_uniform_quantized_use_constant',
          quant_opts_pb2.UNIFORM_QUANTIZED,
          False,
      ),
      ('to_tf_use_variable', quant_opts_pb2.TF, True),
      ('to_xla_use_variable', quant_opts_pb2.XLA, True),
      (
          'to_uniform_quantized_use_variable',
          quant_opts_pb2.UNIFORM_QUANTIZED,
          True,
      ),
  )
  @test_util.run_v2_only
  def test_gather_model(
      self, target_opset: quant_opts_pb2.OpSet, use_variable: bool
  ):
    input_type = dtypes.int64
    model = self._create_gather_model(input_type, use_variable)
    saved_model_save.save(model, self._input_saved_model_path)

    tags = {tag_constants.SERVING}

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_DYNAMIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=target_opset,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
    )

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )

    if target_opset == quant_opts_pb2.UNIFORM_QUANTIZED:
      self.assertGreater(
          testing.get_size_ratio(
              self._output_saved_model_path, self._input_saved_model_path
          ),
          0.65,
      )
    else:
      # Due to other meta data, the compression is not exactly 1/4.
      self.assertLess(
          testing.get_size_ratio(
              self._output_saved_model_path, self._input_saved_model_path
          ),
          1 / 3,
      )

  @parameterized.named_parameters(
      ('to_tf_with_int32_input_type', dtypes.int32, quant_opts_pb2.TF),
      ('to_xla_with_int32_input_type', dtypes.int32, quant_opts_pb2.XLA),
      ('to_xla_with_int64_input_type', dtypes.int64, quant_opts_pb2.XLA),
      (
          'to_uq_with_int32_input_type',
          dtypes.int32,
          quant_opts_pb2.UNIFORM_QUANTIZED,
      ),
  )
  @test_util.run_v2_only
  def test_gather_and_conv_model(
      self, input_type: dtypes, target_opset: quant_opts_pb2.OpSet
  ):
    model = self._create_simple_gather_and_conv_model(
        input_type, filter_shape=(2, 3, 3, 1024)
    )
    saved_model_save.save(model, self._input_saved_model_path)

    tags = {tag_constants.SERVING}

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_DYNAMIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=target_opset,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
    )
    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )

    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def

    if target_opset == quant_opts_pb2.UNIFORM_QUANTIZED:
      self.assertGreater(
          testing.get_size_ratio(
              self._output_saved_model_path, self._input_saved_model_path
          ),
          0.65,
      )
      self.assertTrue(
          self._contains_op(
              output_graphdef, 'UniformQuantizedConvolutionHybrid'
          )
      )
    else:
      # Due to other meta data, the compression is not exactly 1/4.
      self.assertLess(
          testing.get_size_ratio(
              self._output_saved_model_path, self._input_saved_model_path
          ),
          1 / 3,
      )
      if target_opset == quant_opts_pb2.XLA:
        self.assertTrue(self._contains_op(output_graphdef, 'XlaConvV2'))
      else:
        self.assertTrue(self._contains_quantized_function_call(output_graphdef))

  @test_util.run_in_graph_and_eager_modes
  def test_conv_model_with_wrong_tags_raises_error(self):
    signature_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    save_tags = {tag_constants.TRAINING, tag_constants.GPU}

    input_placeholder = self._create_and_save_tf1_conv_model(
        self._input_saved_model_path,
        signature_key,
        save_tags,
        input_key='input',
        output_key='output',
        use_variable=True,
    )

    tags = {tag_constants.SERVING}
    signature_keys = [signature_key]
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_DYNAMIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=signature_keys,
        op_set=quant_opts_pb2.OpSet.UNIFORM_QUANTIZED,
    )

    # Try to use a different set of tags to quantize.
    data_gen = self._create_data_generator(
        input_key='input', shape=input_placeholder.shape
    )

    # StatusNotOk error. `Exception` is used here because importing
    # `StatusNotOk` may break the open-sourced version of TensorFlow.
    with self.assertRaisesRegex(
        Exception,
        'could not be found in SavedModel, with available tags',
    ) as raises:
      quantize_model.quantize(
          self._input_saved_model_path,
          self._output_saved_model_path,
          quantization_options,
          representative_dataset=data_gen,
      )

    self.assertEqual(raises.exception.__class__.__name__, 'RuntimeError')

  @parameterized.named_parameters(
      ('quantize', True, 0),
      ('not_quantize', False, 10000),
  )
  @test_util.run_in_graph_and_eager_modes
  def test_minimum_elements_for_weights(self, quantize, num_elements):
    self._create_matmul_model(
        input_shape=(1, 1024),
        weight_shape=(1024, 3),
        saved_model_path=self._input_saved_model_path,
    )

    tags = {tag_constants.SERVING}
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_DYNAMIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=quant_opts_pb2.OpSet.UNIFORM_QUANTIZED,
    )
    quantization_options.min_num_elements_for_weights = num_elements

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
    )

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )

    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def
    type_attr = attr_value_pb2.AttrValue(type=types_pb2.DT_QINT8)
    if quantize:
      self.assertTrue(
          self._contains_op(output_graphdef, 'Const', 'dtype', type_attr)
      )
    else:
      self.assertFalse(
          self._contains_op(output_graphdef, 'Const', 'dtype', type_attr)
      )

  @parameterized.named_parameters(
      ('to_tf_use_constant', quant_opts_pb2.TF, False),
      ('to_xla_use_constant', quant_opts_pb2.XLA, False),
      (
          'to_uniform_quantized_use_constant',
          quant_opts_pb2.UNIFORM_QUANTIZED,
          False,
      ),
      ('to_tf_use_variable', quant_opts_pb2.TF, True),
      ('to_xla_use_variable', quant_opts_pb2.XLA, True),
      (
          'to_uniform_quantized_use_variable',
          quant_opts_pb2.UNIFORM_QUANTIZED,
          True,
      ),
  )
  @test_util.run_in_graph_and_eager_modes
  def test_gather_model_tf1(
      self, target_opset: quant_opts_pb2.OpSet, use_variable: bool
  ):
    signature_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    tags = {tag_constants.SERVING}

    _ = self._create_and_save_tf1_gather_model(
        self._input_saved_model_path,
        signature_key,
        tags,
        input_key='x',
        output_key='output',
        input_type=dtypes.int32,
        use_variable=use_variable,
    )

    signature_keys = [signature_key]
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_DYNAMIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=signature_keys,
        op_set=target_opset,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
    )

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), signature_keys
    )

    if target_opset == quant_opts_pb2.UNIFORM_QUANTIZED:
      threshold = 0.45 if use_variable else 0.7
      self.assertGreater(
          testing.get_size_ratio(
              self._output_saved_model_path, self._input_saved_model_path
          ),
          threshold,
      )

    else:
      threshold = 0.19 if use_variable else 0.42
      self.assertLess(
          testing.get_size_ratio(
              self._output_saved_model_path, self._input_saved_model_path
          ),
          threshold,
      )

  @test_util.run_in_graph_and_eager_modes
  def test_non_empty_directory_raises_file_exists_error(self):
    self._create_matmul_model(
        input_shape=(1, 1024),
        weight_shape=(1024, 3),
        saved_model_path=self._input_saved_model_path,
    )

    # Create a file inside the output directory.
    file_io.write_string_to_file(
        filename=os.path.join(self._output_saved_model_path, 'dummy_file.txt'),
        file_content='Test content',
    )

    tags = {tag_constants.SERVING}
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_DYNAMIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
    )

    with self.assertRaisesRegex(
        FileExistsError, 'Output directory already exists'
    ):
      quantize_model.quantize(
          self._input_saved_model_path,
          self._output_saved_model_path,
          quantization_options,
      )

  @test_util.run_in_graph_and_eager_modes
  def test_non_empty_directory_overwritten(self):
    self._create_matmul_model(
        input_shape=(1, 1024),
        weight_shape=(1024, 3),
        saved_model_path=self._input_saved_model_path,
    )

    # Create a file inside the output directory.
    file_io.write_string_to_file(
        filename=os.path.join(self._output_saved_model_path, 'dummy_file.txt'),
        file_content='Test content',
    )

    tags = {tag_constants.SERVING}
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_DYNAMIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=quant_opts_pb2.TF,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
        overwrite_output_directory=True,
    )

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )

    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def
    self.assertTrue(self._contains_quantized_function_call(output_graphdef))

  def test_table_initialized_when_model_has_table_tf1(self):
    tags = {tag_constants.SERVING}
    signature_def_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY

    # Create and save a simple model that involves a hash table.
    inputs, outputs = self._create_and_save_vocab_table_lookup_model_tf1(
        self._input_saved_model_path, tags, signature_def_key
    )

    # Make sure that the desired input key and output key is present.
    self.assertIn('input_vocabs', inputs.keys())
    self.assertIn('lookup', outputs.keys())

    signature_def_keys = [signature_def_key]
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_DYNAMIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=signature_def_keys,
    )

    quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
    )

    # Tests table lookup to make sure the table has been initialized
    # successfully.
    with session.Session(graph=ops.Graph()) as sess:
      output_meta_graph_def = saved_model_loader.load(
          sess, tags=tags, export_dir=self._output_saved_model_path
      )

      self.assertCountEqual(
          output_meta_graph_def.signature_def.keys(), signature_def_keys
      )

      signature_def = output_meta_graph_def.signature_def[signature_def_key]

      input_tensor_name = signature_def.inputs['input_vocabs'].name
      input_tensor = sess.graph.get_tensor_by_name(input_tensor_name)

      lookup_tensor_name = signature_def.outputs['lookup'].name
      lookup_tensor = sess.graph.get_tensor_by_name(lookup_tensor_name)

      lookup_val = sess.run(
          lookup_tensor,
          feed_dict={
              input_tensor: np.array([b'model', b'quantization', b'hello'])
          },
      )

      self.assertAllClose(lookup_val, [1.0, 2.0, 0.0])

  def test_file_init_hash_table_lookup_model(self):
    tags = {tag_constants.SERVING}
    signature_def_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY

    # Create and save a simple model that involves a hash table.
    inputs, outputs = self._create_and_save_file_init_hash_table_model_tf1(
        self._input_saved_model_path, tags, signature_def_key
    )
    # Make sure that the desired input key and output key is present.
    self.assertIn('input_vocabs', inputs.keys())
    self.assertIn('lookup', outputs.keys())

    signature_def_keys = [signature_def_key]
    quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options=quant_opts_pb2.QuantizationOptions(
            quantization_method=quant_opts_pb2.QuantizationMethod(
                preset_method=_PresetMethod.METHOD_DYNAMIC_RANGE_INT8
            ),
            tags=tags,
            signature_keys=signature_def_keys,
        ),
    )

    # Tests table lookup to make sure the table has been initialized
    # successfully.
    with session.Session(graph=ops.Graph()) as sess:
      output_meta_graph_def = saved_model_loader.load(
          sess, tags=tags, export_dir=self._output_saved_model_path
      )

      self.assertCountEqual(
          output_meta_graph_def.signature_def.keys(), signature_def_keys
      )

      signature_def = output_meta_graph_def.signature_def[signature_def_key]

      input_tensor_name = signature_def.inputs['input_vocabs'].name
      input_tensor = sess.graph.get_tensor_by_name(input_tensor_name)

      lookup_tensor_name = signature_def.outputs['lookup'].name
      lookup_tensor = sess.graph.get_tensor_by_name(lookup_tensor_name)

      lookup_val = sess.run(
          lookup_tensor,
          feed_dict={
              input_tensor: np.array([b'dynamic', b'quantization', b'range'])
          },
      )

      # "dynamic" is not in the table: -1 (default value)
      self.assertAllClose(lookup_val, [-1.0, 2.0, 1.0])


class WeightOnlyQuantizationTest(quantize_model_test_base.QuantizedModelTest):
  """Test cases for weight-only quantization.

  Run all tests cases in both the graph mode (default in TF1) and the eager mode
  (default in TF2) to ensure support for when TF2 is disabled.
  """

  @test_util.run_in_graph_and_eager_modes
  def test_einsum_model(
      self,
  ):
    equation = 'abc,cde->abde'
    _, y_shape, bias_shape, x_signature, y_signature = (
        self._prepare_sample_einsum_datashapes(equation, use_bias=True)
    )

    model = self._create_einsum_model(
        equation,
        y_shape,
        x_signature,
        y_signature,
        bias_shape,
        activation_fn=nn_ops.relu,
    )

    # Use constant y operand.
    signatures = {
        'serving_default': model.einsum_with_kernel.get_concrete_function(),
    }

    saved_model_save.save(model, self._input_saved_model_path, signatures)

    tags = {tag_constants.SERVING}
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_WEIGHT_ONLY_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=quant_opts_pb2.XLA,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
    )
    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )

    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def

    self.assertTrue(
        self._contains_op(
            output_graphdef,
            op_name='Const',
            attr_name='dtype',
            attr_val=attr_value_pb2.AttrValue(type=types_pb2.DT_INT8),
        )
    )
    # Due to other meta data, the compression is not exactly 1/4.
    self.assertLess(
        testing.get_size_ratio(
            self._output_saved_model_path, self._input_saved_model_path
        ),
        0.5,
    )

  @parameterized.named_parameters(
      ('to_xla_per_tensor', quant_opts_pb2.XLA, False),
      ('stablehlo_per_channel', quant_opts_pb2.STABLEHLO, True),
  )
  @test_util.run_in_graph_and_eager_modes
  def test_matmul_model(
      self,
      target_opset: quant_opts_pb2.OpSet,
      enable_per_channel_quantization: bool,
  ):
    input_shape = (1, 512)

    self._create_matmul_model(
        input_shape=input_shape,
        weight_shape=(512, 2),
        saved_model_path=self._input_saved_model_path,
    )

    tags = {tag_constants.SERVING}
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_WEIGHT_ONLY_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=target_opset,
        enable_per_channel_quantization=enable_per_channel_quantization,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
    )
    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )

    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def

    if target_opset == quant_opts_pb2.XLA:
      self.assertTrue(self._contains_op(output_graphdef, 'XlaDotV2'))
    elif target_opset == quant_opts_pb2.STABLEHLO:
      # This is to verify the invocation of StableHLO quantizer works. More
      # thorough functional tests are in StableHLO quantizer directory.
      self.assertTrue(self._contains_op(output_graphdef, 'XlaCallModule'))

    # Due to other meta data, the compression is not exactly 1/4.
    self.assertLess(
        testing.get_size_ratio(
            self._output_saved_model_path, self._input_saved_model_path
        ),
        0.3,
    )

  @parameterized.named_parameters(
      ('to_xla_per_tensor', quant_opts_pb2.XLA, False),
      ('stablehlo_per_channel', quant_opts_pb2.STABLEHLO, True),
      # TODO: b/289761265 - [Converter Component][TF-Quantizer] Improve Weight-
      # only Quantization
      # Enable this back once new weight-only quantizer is supported for per-
      # channel quantization.
      # ('to_xla_per_channel', quant_opts_pb2.XLA, True),
  )
  @test_util.run_in_graph_and_eager_modes
  def test_conv_model(
      self,
      target_opset: quant_opts_pb2.OpSet,
      enable_per_channel_quantization: bool,
  ):
    input_shape = (1, 3, 4, 512)
    filter_shape = (2, 3, 512, 2)
    model = self._create_conv2d_model(
        input_shape=input_shape,
        filter_shape=filter_shape,
        has_bias=False,
        has_batch_norm=False,
        activation_fn=nn_ops.relu6,
    )
    saved_model_save.save(model, self._input_saved_model_path)

    tags = {tag_constants.SERVING}

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_WEIGHT_ONLY_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=target_opset,
        enable_per_channel_quantization=enable_per_channel_quantization,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
    )

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )

    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def

    # Due to other meta data, the compression is not exactly 1/4.
    self.assertLess(
        testing.get_size_ratio(
            self._output_saved_model_path, self._input_saved_model_path
        ),
        0.31,
    )

    if enable_per_channel_quantization and target_opset == quant_opts_pb2.XLA:
      per_channel_size_attr = attr_value_pb2.AttrValue(
          list=attr_value_pb2.AttrValue.ListValue(
              shape=[
                  tensor_shape_pb2.TensorShapeProto(
                      dim=[
                          tensor_shape_pb2.TensorShapeProto.Dim(
                              size=filter_shape[-1]
                          )
                      ]
                  )
              ]
          )
      )
      self.assertTrue(
          self._contains_op(
              output_graphdef, 'Const', '_output_shapes', per_channel_size_attr
          )
      )
    if target_opset == quant_opts_pb2.XLA:
      self.assertTrue(self._contains_op(output_graphdef, 'XlaConvV2'))
    elif target_opset == quant_opts_pb2.STABLEHLO:
      # This is to verify the invocation of StableHLO quantizer works. More
      # thorough functional tests are in StableHLO quantizer directory.
      self.assertTrue(self._contains_op(output_graphdef, 'XlaCallModule'))

    input_tensor = array_ops.constant(
        np.random.uniform(low=0, high=0.1, size=input_shape),
        dtype=dtypes.float32,
    )
    original_output = model.conv(input_tensor)
    quantized_output = converted_model.signatures['serving_default'](
        input_tensor
    )

    threshold = 0.015 if enable_per_channel_quantization else 0.02
    self.assertAllClose(original_output, quantized_output, atol=threshold)

  @parameterized.named_parameters(
      ('to_xla_per_tensor', quant_opts_pb2.XLA, False),
      # TODO: b/289761265 - [Converter Component][TF-Quantizer] Improve Weight-
      # only Quantization
      # Enable this back once new weight-only quantizer is supported for per-
      # channel quantization.
      # ('to_xla_per_channel', quant_opts_pb2.XLA, True),
  )
  @test_util.run_in_graph_and_eager_modes
  def test_depthwise_conv2d_model(
      self,
      target_opset: quant_opts_pb2.OpSet,
      enable_per_channel_quantization: bool,
  ):
    input_shape = (1, 3, 4, 512)
    filter_shape = (2, 3, 512, 2)
    strides = (1, 2, 2, 1)

    model = self._create_depthwise_conv2d_model(
        input_shape=input_shape, filter_shape=filter_shape, strides=strides
    )

    saved_model_save.save(model, self._input_saved_model_path)

    tags = {tag_constants.SERVING}

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_WEIGHT_ONLY_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=target_opset,
        enable_per_channel_quantization=enable_per_channel_quantization,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
    )

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )

    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def

    # Due to other meta data, the compression is not exactly 1/4.
    size_threshold = 0.5 if enable_per_channel_quantization else 0.33
    self.assertLess(
        testing.get_size_ratio(
            self._output_saved_model_path, self._input_saved_model_path
        ),
        size_threshold,
    )

    if enable_per_channel_quantization:
      per_channel_size_attr = attr_value_pb2.AttrValue(
          list=attr_value_pb2.AttrValue.ListValue(
              shape=[
                  tensor_shape_pb2.TensorShapeProto(
                      dim=[
                          tensor_shape_pb2.TensorShapeProto.Dim(
                              size=filter_shape[2] * filter_shape[3]
                          ),
                      ]
                  )
              ]
          )
      )
      self.assertTrue(
          self._contains_op(
              output_graphdef, 'Const', '_output_shapes', per_channel_size_attr
          )
      )

    input_tensor = array_ops.constant(
        np.random.uniform(low=-0.1, high=0.1, size=input_shape),
        dtype=dtypes.float32,
    )
    original_output = model.depthwise_conv(input_tensor)
    quantized_output = converted_model.signatures['serving_default'](
        input_tensor
    )

    threshold = 0.68 if enable_per_channel_quantization else 1.3
    self.assertAllClose(original_output, quantized_output, atol=threshold)

  @parameterized.named_parameters(
      ('to_tf_use_constant', quant_opts_pb2.TF, False),
      ('to_xla_use_constant', quant_opts_pb2.XLA, False),
      (
          'to_uniform_quantized_use_constant',
          quant_opts_pb2.UNIFORM_QUANTIZED,
          False,
      ),
      ('to_tf_use_variable', quant_opts_pb2.TF, True),
      ('to_xla_use_variable', quant_opts_pb2.XLA, True),
      (
          'to_uniform_quantized_use_variable',
          quant_opts_pb2.UNIFORM_QUANTIZED,
          True,
      ),
  )
  @test_util.run_v2_only
  def test_gather_model(
      self, target_opset: quant_opts_pb2.OpSet, use_variable: bool
  ):
    input_type = dtypes.int64
    model = self._create_gather_model(input_type, use_variable)
    input_saved_model_path = self.create_tempdir('input').full_path
    saved_model_save.save(model, input_saved_model_path)

    tags = {tag_constants.SERVING}
    output_directory = self.create_tempdir().full_path

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_WEIGHT_ONLY_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=target_opset,
    )

    if target_opset != quant_opts_pb2.XLA:
      # Uniform quantized opset is not supported for weight-only
      with self.assertRaisesRegex(
          ValueError, 'TF/Uniform quantized opset does not support weight-only.'
      ):
        converted_model = quantize_model.quantize(
            input_saved_model_path,
            output_directory,
            quantization_options,
        )
      return

    else:
      converted_model = quantize_model.quantize(
          input_saved_model_path,
          output_directory,
          quantization_options,
      )

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )
    # Due to other meta data, the compression is not exactly 1/4.
    self.assertLess(
        testing.get_size_ratio(
            self._output_saved_model_path, self._input_saved_model_path
        ),
        0.3,
    )

  @parameterized.named_parameters(
      ('to_tf_with_int32_input_type', dtypes.int32, quant_opts_pb2.TF),
      ('to_xla_with_int32_input_type', dtypes.int32, quant_opts_pb2.XLA),
      ('to_xla_with_int64_input_type', dtypes.int64, quant_opts_pb2.XLA),
      (
          'to_uq_with_int32_input_type',
          dtypes.int32,
          quant_opts_pb2.UNIFORM_QUANTIZED,
      ),
  )
  @test_util.run_v2_only
  def test_gather_and_conv_model(
      self, input_type: dtypes, target_opset: quant_opts_pb2.OpSet
  ):
    model = self._create_simple_gather_and_conv_model(
        input_type, filter_shape=(2, 3, 3, 1024)
    )
    saved_model_save.save(model, self._input_saved_model_path)

    tags = {tag_constants.SERVING}

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_WEIGHT_ONLY_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=target_opset,
    )

    if target_opset != quant_opts_pb2.XLA:
      # Uniform quantized opset is not supported for weight-only
      with self.assertRaisesRegex(
          ValueError, 'TF/Uniform quantized opset does not support weight-only.'
      ):
        converted_model = quantize_model.quantize(
            self._input_saved_model_path,
            self._output_saved_model_path,
            quantization_options,
        )
      return
    else:
      converted_model = quantize_model.quantize(
          self._input_saved_model_path,
          self._output_saved_model_path,
          quantization_options,
      )
    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )

    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def
    self.assertTrue(self._contains_op(output_graphdef, 'XlaConvV2'))
    # Due to other meta data, the compression is not exactly 1/4.
    self.assertLess(
        testing.get_size_ratio(
            self._output_saved_model_path, self._input_saved_model_path
        ),
        1 / 3,
    )

  @test_util.run_in_graph_and_eager_modes
  def test_function_alias_preserved(self):
    # Prepare test model
    function_alias = 'conv_func'
    tags = {tag_constants.SERVING}
    input_type, filter_shape = dtypes.int64, (2, 3, 3, 2)
    model = self._create_simple_gather_and_conv_model(input_type, filter_shape)
    save_opts = save_options.SaveOptions(
        function_aliases={function_alias: model.model}
    )
    signatures = {
        'serving_default': model.model.get_concrete_function(),
    }
    saved_model_save.save(
        model, self._input_saved_model_path, signatures, save_opts
    )

    # Quantize the model
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=quant_opts_pb2.QuantizationMethod.PresetMethod.METHOD_STATIC_RANGE_WEIGHT_ONLY_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=quant_opts_pb2.XLA,
        min_num_elements_for_weights=1,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
    )
    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )

    # Check if function alias is preserved
    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    meta_graph_def = output_loader.get_meta_graph_def_from_tags(tags)
    function_aliases = meta_graph_def.meta_info_def.function_aliases
    self.assertNotEmpty(function_aliases)
    self.assertCountEqual(function_aliases.values(), {function_alias})

    # Test that the aliased function contains a quantized op.
    for func_name, alias in function_aliases.items():
      if alias == function_alias:
        for func in meta_graph_def.graph_def.library.function:
          if func.signature.name == func_name:
            self.assertTrue(
                self._contains_op_with_name_and_attribute(
                    func.node_def,
                    op_name='Const',
                    attr_name='dtype',
                    attr_val=attr_value_pb2.AttrValue(type=types_pb2.DT_INT8),
                )
            )

  @test_util.run_v2_only
  def test_while_op_model(
      self,
  ):
    model = self._create_while_model()
    saved_model_save.save(model, self._input_saved_model_path)

    tags = {tag_constants.SERVING}
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_WEIGHT_ONLY_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=quant_opts_pb2.XLA,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
    )
    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )

    loader = saved_model_loader.SavedModelLoader(self._output_saved_model_path)
    output_graphdef = loader.get_meta_graph_def_from_tags(tags).graph_def

    # Convolution ouside the while op is quantized.
    self.assertTrue(
        self._contains_op(
            output_graphdef,
            op_name='XlaConvV2',
            attr_name='RhsT',
            attr_val=attr_value_pb2.AttrValue(type=types_pb2.DT_INT8),
        )
    )
    # TODO: b/294783597 - [Converter][TF-Quantizer] Support quantization for the
    # ops in the while op body for both SRQ and WO
    # Convolution inside the while op is not quantized.
    self.assertTrue(
        self._contains_op(
            output_graphdef,
            op_name='Conv2D',
            attr_name='T',
            attr_val=attr_value_pb2.AttrValue(type=types_pb2.DT_FLOAT),
        )
    )


class DebuggerTest(quantize_model_test_base.QuantizedModelTest):

  def _run_model_in_sess(self, model_dir, tags, signature_key, sample_inputs):
    with tensorflow.compat.v1.Session(graph=tensorflow.Graph()) as sess:
      meta_graph = saved_model_loader.load(sess, tags, export_dir=model_dir)
      signature_def = meta_graph.signature_def[signature_key]

      # DumpTensorOp only works in graph mode.
      # Execute the model using session to run DumpTensorOp.
      output_tensor_names = [
          output_tensor_info.name
          for output_tensor_info in signature_def.outputs.values()
      ]

      output_values = []
      for sample_input in sample_inputs:
        feed_dict = {}
        for input_key, input_value in sample_input.items():
          input_tensor_name = signature_def.inputs[input_key].name
          feed_dict[input_tensor_name] = input_value

        # Obtain the output of the model.
        output_values.append(
            sess.run(output_tensor_names, feed_dict=feed_dict)[0]
        )
    return output_values

  def _read_tensor_array_file(self, file_path):
    tensor_protos = []
    for raw_record in tf_record.tf_record_iterator(file_path, options='ZLIB'):
      tensor_protos.append(
          tensorflow.make_ndarray(tensor_pb2.TensorProto.FromString(raw_record))
      )
    return np.array(tensor_protos)

  @parameterized.named_parameters(
      {
          'testcase_name': 'none',
          'activation_fn': None,
          'has_bias': False,
      },
      {
          'testcase_name': 'relu',
          'activation_fn': nn_ops.relu,
          'has_bias': False,
      },
      {
          'testcase_name': 'with_bias',
          'activation_fn': None,
          'has_bias': True,
      },
      {
          'testcase_name': 'with_bias_and_relu',
          'activation_fn': nn_ops.relu,
          'has_bias': True,
      },
  )
  def test_conv2d_ptq_model_whole_model_verify(self, activation_fn, has_bias):
    input_shape = [None, None, None, 3]
    filter_shape = [2, 3, 3, 2]

    model = self._create_conv2d_model(
        input_shape,
        filter_shape,
        activation_fn=activation_fn,
        has_bias=has_bias,
    )
    saved_model_save.save(model, self._input_saved_model_path)

    def data_gen() -> repr_dataset.RepresentativeDataset:
      for _ in range(8):
        yield {
            'input_tensor': ops.convert_to_tensor(
                np.random.uniform(low=0, high=150, size=(1, 3, 4, 3)).astype(
                    'f4'
                )
            ),
        }

    tags = {tag_constants.SERVING}

    unquantized_dump_model_path = self.create_tempdir().full_path
    log_dir_path = self.create_tempdir().full_path

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        op_set=quant_opts_pb2.XLA,
        debugger_config=_DebuggerConfig(
            debugger_type=_DebuggerConfig.DebuggerType.DEBUGGER_TYPE_WHOLE_MODEL,
            unquantized_dump_model_path=unquantized_dump_model_path,
            log_dir_path=log_dir_path,
        ),
        tags=tags,
        signature_keys=['serving_default'],
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
        representative_dataset=data_gen(),
    )

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )

    sample_inputs = [
        {'input_tensor': np.random.uniform(low=0, high=1, size=(16, 3, 4, 3))},
        {'input_tensor': np.random.uniform(low=0, high=1, size=(16, 3, 4, 3))},
    ]

    # Check if output of the model and value saved by DumpTensorOp matches.
    # Verify for both unquantized model and quantized model.
    for model_path, file_name in [
        [unquantized_dump_model_path, 'unquantized_tensor_data.pb'],
        [self._output_saved_model_path, 'quantized_tensor_data.pb'],
    ]:
      output_values = self._run_model_in_sess(
          model_path, tags, 'serving_default', sample_inputs
      )

      # Find the dump file and parse it.
      folder = os.path.join(log_dir_path, os.listdir(log_dir_path)[0])
      dump_file_path = os.path.join(log_dir_path, folder, file_name)
      dump_file_numpy = self._read_tensor_array_file(dump_file_path)

      # Since the model only has one conv2d and its output is directly used as
      # the output of the model, output of the model and conv2d's dump value
      # should be the same.
      self.assertAllClose(output_values, dump_file_numpy)

      # Verify if quant_unit.pb file was created correctly.
      quant_unit_file_path = os.path.join(log_dir_path, folder, 'quant_unit.pb')

      quant_unit = (
          quant_opts_pb2.UnitWiseQuantizationSpec.QuantizationUnit.FromString(
              open(quant_unit_file_path, 'rb').read()
          )
      )

      self.assertEqual(quant_unit.node_name, 'Conv2D')
      self.assertRegex(quant_unit.func_name, r'^__inference_conv_\d+')

  @parameterized.parameters(
      testing.parameter_combinations([{
          'activation_fn': [None, nn_ops.relu, nn_ops.relu6],
          'has_bias': [True, False],
          'debugger_type': [
              _DebuggerConfig.DEBUGGER_TYPE_INT_PER_LAYER,
              _DebuggerConfig.DEBUGGER_TYPE_FLOAT_PER_LAYER,
          ],
          'target_opset': [quant_opts_pb2.XLA, quant_opts_pb2.STABLEHLO],
      }])
  )
  def test_conv2d_ptq_model_per_layer_verify(
      self,
      activation_fn: Optional[ops.Operation],
      has_bias: bool,
      debugger_type: _DebuggerConfig.DebuggerType,
      target_opset: quant_opts_pb2.OpSet,
  ):
    # TODO: b/326114903 - Support dynamic input dimensions after 0th rank in
    # op_set=STABLEHLO.
    input_shape_dynamic = target_opset != quant_opts_pb2.STABLEHLO
    concrete_input_shape = [None, 3, 4, 3]
    input_shape = (
        [None, None, None, 3] if input_shape_dynamic else concrete_input_shape
    )
    filter_shape = [2, 3, 3, 2]

    model = self._create_conv2d_model(
        input_shape,
        filter_shape,
        activation_fn=activation_fn,
        has_bias=has_bias,
    )
    saved_model_save.save(model, self._input_saved_model_path)

    def data_gen() -> repr_dataset.RepresentativeDataset:
      data_input_size = [1] + concrete_input_shape[1:]
      for _ in range(8):
        yield {
            'input_tensor': ops.convert_to_tensor(
                np.random.uniform(low=0, high=150, size=data_input_size).astype(
                    'f4'
                )
            ),
        }

    tags = {tag_constants.SERVING}

    log_dir_path = self.create_tempdir().full_path

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        op_set=target_opset,
        debugger_config=_DebuggerConfig(
            debugger_type=debugger_type,
            log_dir_path=log_dir_path,
        ),
        tags=tags,
        signature_keys=['serving_default'],
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
        representative_dataset=data_gen(),
    )

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )

    sample_input_size = [16] + concrete_input_shape[1:]
    sample_inputs = [
        {
            'input_tensor': np.random.uniform(
                low=0, high=1, size=sample_input_size
            )
        },
        {
            'input_tensor': np.random.uniform(
                low=0, high=1, size=sample_input_size
            )
        },
    ]

    output_value_from_original_model = self._run_model_in_sess(
        self._input_saved_model_path, tags, 'serving_default', sample_inputs
    )
    output_value_from_debugging_model = self._run_model_in_sess(
        self._output_saved_model_path, tags, 'serving_default', sample_inputs
    )

    # Find the both quantized and unquantized dump file.
    folder = os.path.join(log_dir_path, os.listdir(log_dir_path)[0])
    unquantized_dump_file_path = os.path.join(
        log_dir_path, folder, 'unquantized_tensor_data.pb'
    )
    quantized_dump_file_path = os.path.join(
        log_dir_path, folder, 'quantized_tensor_data.pb'
    )

    unquantized_dump_file_numpy = self._read_tensor_array_file(
        unquantized_dump_file_path
    )
    quantized_dump_file_numpy = self._read_tensor_array_file(
        quantized_dump_file_path
    )

    # Since the model only has one conv2d and its output is directly used as
    # the output of the model, output of the model and conv2d's dump value
    # should be the same.
    self.assertAllClose(
        output_value_from_original_model, unquantized_dump_file_numpy
    )
    # The output_value_from_debugging_model of DEBUGGER_TYPE_INT_PER_LAYER is
    # a quantized value, while for DEBUGGER_TYPE_FLOAT_PER_LAYER, it's an
    # unquantized value. Therefore there are different verifications for the
    # output value.
    if debugger_type == _DebuggerConfig.DEBUGGER_TYPE_INT_PER_LAYER:
      self.assertAllClose(
          output_value_from_debugging_model, quantized_dump_file_numpy
      )
    else:  # debugger_type == _DebuggerConfig.DEBUGGER_TYPE_FLOAT_PER_LAYER:
      self.assertAllClose(
          output_value_from_debugging_model, output_value_from_original_model
      )

    # Verify if quant_unit.pb file was created correctly.
    quant_unit_file_path = os.path.join(log_dir_path, folder, 'quant_unit.pb')
    quant_unit = (
        quant_opts_pb2.UnitWiseQuantizationSpec.QuantizationUnit.FromString(
            open(quant_unit_file_path, 'rb').read()
        )
    )

    if target_opset == quant_opts_pb2.XLA:
      self.assertEqual(quant_unit.node_name, 'Conv2D')
      self.assertRegex(quant_unit.func_name, r'^__inference_conv_\d+')
    elif target_opset == quant_opts_pb2.STABLEHLO:
      self.assertEqual(quant_unit.node_name, '_empty_node')
      self.assertRegex(
          quant_unit.func_name, r'^composite_conv_([a-zA-Z_0-9]+_)*fn_\d+'
      )
    else:
      assert False, f'Please add assertion for the op_set: {target_opset}.'


@test_util.run_all_in_graph_and_eager_modes
class CalibrationOptionsTest(quantize_model_test_base.QuantizedModelTest):
  """Test cases regarding the use of CalibrationOptions proto.

  Run all tests cases in both the graph mode (default in TF1) and the eager mode
  (default in TF2) to ensure support for when TF2 is disabled.
  """

  @parameterized.parameters(
      testing.parameter_combinations([{
          'target_opset': [
              quant_opts_pb2.TF,
              quant_opts_pb2.XLA,
              quant_opts_pb2.UNIFORM_QUANTIZED,
          ],
          'calibration_options': [
              stablehlo_quant_config_pb2.CalibrationOptions(
                  calibration_method=_CalibrationMethod.CALIBRATION_METHOD_MIN_MAX
              ),
              stablehlo_quant_config_pb2.CalibrationOptions(
                  calibration_method=_CalibrationMethod.CALIBRATION_METHOD_AVERAGE_MIN_MAX
              ),
              stablehlo_quant_config_pb2.CalibrationOptions(
                  calibration_method=_CalibrationMethod.CALIBRATION_METHOD_HISTOGRAM_PERCENTILE,
                  calibration_parameters=stablehlo_quant_config_pb2.CalibrationOptions.CalibrationParameters(
                      num_bins=32,
                  ),
              ),
              stablehlo_quant_config_pb2.CalibrationOptions(
                  calibration_method=_CalibrationMethod.CALIBRATION_METHOD_HISTOGRAM_MSE_BRUTEFORCE,
                  calibration_parameters=stablehlo_quant_config_pb2.CalibrationOptions.CalibrationParameters(
                      num_bins=32,
                  ),
              ),
              stablehlo_quant_config_pb2.CalibrationOptions(
                  calibration_method=_CalibrationMethod.CALIBRATION_METHOD_HISTOGRAM_MSE_MAX_FREQUENCY,
                  calibration_parameters=stablehlo_quant_config_pb2.CalibrationOptions.CalibrationParameters(
                      num_bins=32,
                  ),
              ),
              stablehlo_quant_config_pb2.CalibrationOptions(
                  calibration_method=_CalibrationMethod.CALIBRATION_METHOD_HISTOGRAM_MSE_SYMMETRIC,
                  calibration_parameters=stablehlo_quant_config_pb2.CalibrationOptions.CalibrationParameters(
                      num_bins=32,
                  ),
              ),
          ],
      }])
  )
  @test_util.run_in_graph_and_eager_modes
  def test_conv_ptq_model_by_calibration_options(
      self,
      target_opset: quant_opts_pb2.OpSet,
      calibration_options: stablehlo_quant_config_pb2.CalibrationOptions,
  ):
    has_bias = True
    has_batch_norm = True
    activation_fn = nn_ops.relu6
    enable_per_channel_quantization = False

    input_shape = [1, 3, 4, 3]
    filter_shape = [2, 3, 3, 2]

    model = self._create_conv2d_model(
        input_shape, filter_shape, has_bias, has_batch_norm, activation_fn
    )
    saved_model_save.save(model, self._input_saved_model_path)

    def data_gen() -> repr_dataset.RepresentativeDataset:
      for _ in range(10):
        yield {
            'input_tensor': ops.convert_to_tensor(
                np.random.uniform(low=0, high=10, size=(1, 3, 4, 3)).astype(
                    'f4'
                )
            ),
        }

    tags = {tag_constants.SERVING}

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=target_opset,
        enable_per_channel_quantization=enable_per_channel_quantization,
        calibration_options=calibration_options,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
        representative_dataset=data_gen(),
        overwrite_output_directory=True,
    )
    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )

    sample_input = ops.convert_to_tensor(
        np.random.uniform(low=0, high=10, size=(1, 3, 4, 3)).astype('f4')
    )
    expected_outputs = model.conv(sample_input)
    got_outputs = converted_model.signatures['serving_default'](sample_input)
    self.assertAllClose(expected_outputs, got_outputs, atol=1e-1)

    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def
    if target_opset == quant_opts_pb2.XLA:
      self.assertTrue(self._contains_op(output_graphdef, 'XlaConvV2'))
    elif target_opset == quant_opts_pb2.UNIFORM_QUANTIZED:
      self.assertTrue(
          self._contains_op(output_graphdef, 'UniformQuantizedConvolution')
      )
      if enable_per_channel_quantization:
        quantized_axis = 3
        quantized_dim_size_attr = attr_value_pb2.AttrValue(
            list=attr_value_pb2.AttrValue.ListValue(
                shape=[
                    tensor_shape_pb2.TensorShapeProto(
                        dim=[
                            tensor_shape_pb2.TensorShapeProto.Dim(
                                size=filter_shape[quantized_axis]
                            )
                        ]
                    )
                ]
            )
        )
      else:
        quantized_axis = -1
        # Empty dimension. Per-tensor quantization has singular channel.
        quantized_dim_size_attr = attr_value_pb2.AttrValue(
            list=attr_value_pb2.AttrValue.ListValue(
                shape=[tensor_shape_pb2.TensorShapeProto()]
            )
        )
      quantized_axis_attr = attr_value_pb2.AttrValue(i=quantized_axis)
      self.assertEqual(
          self._count_ops(
              output_graphdef,
              _PER_CHANNEL_QUANTIZED_OPS,
              'rhs_quantization_axis',
              quantized_axis_attr,
          ),
          self._count_ops(output_graphdef, _PER_CHANNEL_QUANTIZED_OPS),
      )
      self.assertEqual(
          self._count_ops(
              output_graphdef,
              _PER_CHANNEL_OP_NAMES,
              '_output_shapes',
              quantized_dim_size_attr,
              get_op_name=True,
          ),
          self._count_ops(
              output_graphdef,
              _PER_CHANNEL_OP_NAMES,
              get_op_name=True,
          ),
      )
      self.assertFalse(self._contains_op(output_graphdef, 'Conv2D'))
    else:
      self.assertTrue(self._contains_quantized_function_call(output_graphdef))
    self.assertFalse(self._contains_op(output_graphdef, 'FusedBatchNormV3'))

  @parameterized.named_parameters(
      {
          'testcase_name': 'with_calibration_method_unspecified',
          'calibration_options': stablehlo_quant_config_pb2.CalibrationOptions(
              calibration_method=_CalibrationMethod.CALIBRATION_METHOD_UNSPECIFIED
          ),
          'default_calibration_options': stablehlo_quant_config_pb2.CalibrationOptions(
              calibration_method=_CalibrationMethod.CALIBRATION_METHOD_MIN_MAX
          ),
      },
      {
          'testcase_name': 'with_histogram_percentile',
          'calibration_options': stablehlo_quant_config_pb2.CalibrationOptions(
              calibration_method=_CalibrationMethod.CALIBRATION_METHOD_HISTOGRAM_PERCENTILE
          ),
          'default_calibration_options': stablehlo_quant_config_pb2.CalibrationOptions(
              calibration_method=_CalibrationMethod.CALIBRATION_METHOD_HISTOGRAM_PERCENTILE,
              calibration_parameters=stablehlo_quant_config_pb2.CalibrationOptions.CalibrationParameters(
                  num_bins=512,
                  min_percentile=0.001,
                  max_percentile=99.999,
              ),
          ),
      },
      {
          'testcase_name': 'with_histogram_mse_bruteforce',
          'calibration_options': stablehlo_quant_config_pb2.CalibrationOptions(
              calibration_method=_CalibrationMethod.CALIBRATION_METHOD_HISTOGRAM_MSE_BRUTEFORCE
          ),
          'default_calibration_options': stablehlo_quant_config_pb2.CalibrationOptions(
              calibration_method=_CalibrationMethod.CALIBRATION_METHOD_HISTOGRAM_MSE_BRUTEFORCE,
              calibration_parameters=stablehlo_quant_config_pb2.CalibrationOptions.CalibrationParameters(
                  num_bins=512
              ),
          ),
      },
      {
          'testcase_name': 'with_histogram_mse_max_frequency',
          'calibration_options': stablehlo_quant_config_pb2.CalibrationOptions(
              calibration_method=_CalibrationMethod.CALIBRATION_METHOD_HISTOGRAM_MSE_MAX_FREQUENCY
          ),
          'default_calibration_options': stablehlo_quant_config_pb2.CalibrationOptions(
              calibration_method=_CalibrationMethod.CALIBRATION_METHOD_HISTOGRAM_MSE_MAX_FREQUENCY,
              calibration_parameters=stablehlo_quant_config_pb2.CalibrationOptions.CalibrationParameters(
                  num_bins=512
              ),
          ),
      },
      {
          'testcase_name': 'with_histogram_mse_symmetric',
          'calibration_options': stablehlo_quant_config_pb2.CalibrationOptions(
              calibration_method=_CalibrationMethod.CALIBRATION_METHOD_HISTOGRAM_MSE_SYMMETRIC
          ),
          'default_calibration_options': stablehlo_quant_config_pb2.CalibrationOptions(
              calibration_method=_CalibrationMethod.CALIBRATION_METHOD_HISTOGRAM_MSE_SYMMETRIC,
              calibration_parameters=stablehlo_quant_config_pb2.CalibrationOptions.CalibrationParameters(
                  num_bins=512
              ),
          ),
      },
  )
  @test_util.run_in_graph_and_eager_modes
  def test_default_calibration_options(
      self,
      calibration_options: stablehlo_quant_config_pb2.CalibrationOptions,
      default_calibration_options: stablehlo_quant_config_pb2.CalibrationOptions,
  ):
    quant_opts = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        calibration_options=calibration_options,
    )
    quantize_model._populate_quantization_component_spec(
        quant_opts.quantization_method
    )
    quantize_model._populate_calibration_options(quant_opts)

    self.assertEqual(
        quant_opts.calibration_options.calibration_method,
        default_calibration_options.calibration_method,
    )
    self.assertEqual(
        quant_opts.calibration_options.calibration_parameters.num_bins,
        default_calibration_options.calibration_parameters.num_bins,
    )
    self.assertEqual(
        quant_opts.calibration_options.calibration_parameters.min_percentile,
        default_calibration_options.calibration_parameters.min_percentile,
    )
    self.assertEqual(
        quant_opts.calibration_options.calibration_parameters.max_percentile,
        default_calibration_options.calibration_parameters.max_percentile,
    )

  @parameterized.named_parameters(
      {
          'testcase_name': 'none',
          'activation_fn': None,
          'has_bias': False,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.TF,
      },
      {
          'testcase_name': 'relu',
          'activation_fn': nn_ops.relu,
          'has_bias': False,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.TF,
      },
      {
          'testcase_name': 'bn',
          'activation_fn': None,
          'has_bias': False,
          'has_batch_norm': True,
          'target_opset': quant_opts_pb2.TF,
      },
      {
          'testcase_name': 'with_bias',
          'activation_fn': None,
          'has_bias': True,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.TF,
      },
      {
          'testcase_name': 'with_bias_and_relu',
          'activation_fn': nn_ops.relu,
          'has_bias': True,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.TF,
      },
      {
          'testcase_name': 'with_bias_and_bn_and_relu',
          'activation_fn': nn_ops.relu,
          'has_bias': True,
          'has_batch_norm': True,
          'target_opset': quant_opts_pb2.TF,
      },
      {
          'testcase_name': 'with_bias_and_relu_to_xla',
          'activation_fn': nn_ops.relu,
          'has_bias': True,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.XLA,
      },
      {
          'testcase_name': 'with_bias_and_bn_and_relu_to_xla',
          'activation_fn': nn_ops.relu,
          'has_bias': True,
          'has_batch_norm': True,
          'target_opset': quant_opts_pb2.XLA,
      },
  )
  @test_util.run_in_graph_and_eager_modes
  def test_conv_ptq_with_outlier_representative_data(
      self,
      activation_fn: Optional[ops.Operation],
      has_bias: bool,
      has_batch_norm: bool,
      target_opset: quant_opts_pb2.OpSet,
  ):
    input_shape = [1, 3, 4, 3]
    filter_shape = [2, 3, 3, 2]

    model = self._create_conv2d_model(
        input_shape, filter_shape, has_bias, has_batch_norm, activation_fn
    )
    saved_model_save.save(model, self._input_saved_model_path)

    def data_gen() -> repr_dataset.RepresentativeDataset:
      outlier = np.random.uniform(low=0, high=10, size=(1, 3, 4, 3)).astype(
          'f4'
      )
      outlier[0][0][0][0:2] = [-1000, 1000]
      yield {'input_tensor': ops.convert_to_tensor(outlier)}
      for _ in range(10):
        yield {
            'input_tensor': ops.convert_to_tensor(
                np.random.uniform(low=0, high=10, size=(1, 3, 4, 3)).astype(
                    'f4'
                )
            ),
        }

    tags = {tag_constants.SERVING}

    quantization_options_min_max = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=target_opset,
        enable_per_channel_quantization=False,
        calibration_options=stablehlo_quant_config_pb2.CalibrationOptions(
            calibration_method=_CalibrationMethod.CALIBRATION_METHOD_MIN_MAX,
        ),
    )

    converted_model_min_max = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options_min_max,
        representative_dataset=data_gen(),
        overwrite_output_directory=True,
    )

    self.assertIsNotNone(converted_model_min_max)
    self.assertCountEqual(
        converted_model_min_max.signatures._signatures.keys(),
        {'serving_default'},
    )

    quantization_options_average_min_max = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=target_opset,
        enable_per_channel_quantization=False,
        calibration_options=stablehlo_quant_config_pb2.CalibrationOptions(
            calibration_method=_CalibrationMethod.CALIBRATION_METHOD_AVERAGE_MIN_MAX,
        ),
    )

    converted_model_average_min_max = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options_average_min_max,
        representative_dataset=data_gen(),
        overwrite_output_directory=True,
    )

    self.assertIsNotNone(converted_model_average_min_max)
    self.assertCountEqual(
        converted_model_average_min_max.signatures._signatures.keys(),
        {'serving_default'},
    )

    sample_input = ops.convert_to_tensor(
        np.random.uniform(low=0, high=10, size=(1, 3, 4, 3)).astype('f4')
    )

    original_output = model.conv(sample_input)['output']
    min_max_output = converted_model_min_max.signatures['serving_default'](
        input_tensor=sample_input
    )['output']
    average_min_max_output = converted_model_average_min_max.signatures[
        'serving_default'
    ](input_tensor=sample_input)['output']

    def get_mean_square_error(x, y):
      ret = tensorflow.reduce_mean(tensorflow.square(tensorflow.subtract(x, y)))
      try:
        ret = ret.numpy()
      except AttributeError:
        ret = ret.eval()
      return ret

    min_max_mse = get_mean_square_error(original_output, min_max_output)
    average_min_max_mse = get_mean_square_error(
        original_output, average_min_max_output
    )

    self.assertLess(average_min_max_mse, min_max_mse)


class SelectiveQuantizationTest(quantize_model_test_base.QuantizedModelTest):
  """Test cases regarding selective quantization."""

  def test_unitwise_spec_with_no_units(self):
    _, y_shape, bias_shape, x_signature, y_signature = (
        self._prepare_sample_einsum_datashapes('abc,acd->abd')
    )
    model = self._create_einsum_model(
        'abc,acd->abd',
        y_shape,
        x_signature,
        y_signature,
        bias_shape,
        is_qat_model=True,
    )
    saved_model_save.save(
        model, self._input_saved_model_path, signatures=model.einsum_with_kernel
    )

    signature_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    tags = {tag_constants.SERVING}
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=[signature_key],
        op_set=quant_opts_pb2.XLA,
        unit_wise_quantization_specs=[
            quant_opts_pb2.UnitWiseQuantizationSpec(
                unit=[],
                quantization_method=quant_opts_pb2.QuantizationMethod(
                    preset_method=quant_opts_pb2.QuantizationMethod.METHOD_NO_QUANTIZE
                ),
            )
        ],
    )

    with self.assertRaisesRegex(
        ValueError, 'UnitWiseQuantizationSpec must contain at least one unit.'
    ):
      quantize_model.quantize(
          self._input_saved_model_path,
          self._output_saved_model_path,
          quantization_options,
      )

  def test_unitwise_spec_missing_unit_info(self):
    _, y_shape, bias_shape, x_signature, y_signature = (
        self._prepare_sample_einsum_datashapes('abc,acd->abd')
    )
    model = self._create_einsum_model(
        'abc,acd->abd',
        y_shape,
        x_signature,
        y_signature,
        bias_shape,
        is_qat_model=True,
    )
    saved_model_save.save(
        model, self._input_saved_model_path, signatures=model.einsum_with_kernel
    )

    signature_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    tags = {tag_constants.SERVING}
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=[signature_key],
        op_set=quant_opts_pb2.XLA,
        unit_wise_quantization_specs=[
            quant_opts_pb2.UnitWiseQuantizationSpec(
                unit=[
                    quant_opts_pb2.UnitWiseQuantizationSpec.QuantizationUnit(),
                ],
                quantization_method=quant_opts_pb2.QuantizationMethod(
                    preset_method=quant_opts_pb2.QuantizationMethod.METHOD_NO_QUANTIZE
                ),
            )
        ],
    )

    with self.assertRaisesRegex(
        ValueError, 'Either `op_type` or `node_name` must be specified.'
    ):
      quantize_model.quantize(
          self._input_saved_model_path,
          self._output_saved_model_path,
          quantization_options,
      )

  def test_unitwise_spec_unsupported_method(self):
    _, y_shape, bias_shape, x_signature, y_signature = (
        self._prepare_sample_einsum_datashapes('abc,acd->abd')
    )
    model = self._create_einsum_model(
        'abc,acd->abd',
        y_shape,
        x_signature,
        y_signature,
        bias_shape,
        is_qat_model=True,
    )
    saved_model_save.save(
        model, self._input_saved_model_path, signatures=model.einsum_with_kernel
    )

    signature_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    tags = {tag_constants.SERVING}
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=[signature_key],
        op_set=quant_opts_pb2.XLA,
        unit_wise_quantization_specs=[
            quant_opts_pb2.UnitWiseQuantizationSpec(
                unit=[
                    quant_opts_pb2.UnitWiseQuantizationSpec.QuantizationUnit(
                        op_type='Conv2D',
                    ),
                ],
                quantization_method=quant_opts_pb2.QuantizationMethod(
                    preset_method=quant_opts_pb2.QuantizationMethod.METHOD_STATIC_RANGE_WEIGHT_ONLY_INT8
                ),
            )
        ],
    )

    with self.assertRaisesRegex(
        ValueError,
        'Currently unit-wise quantization spec only supports NO_QUANTIZE and'
        ' same quantization method as the top-level',
    ):
      quantize_model.quantize(
          self._input_saved_model_path,
          self._output_saved_model_path,
          quantization_options,
      )

  def test_selective_quantization_qat(self):
    _, y_shape, bias_shape, x_signature, y_signature = (
        self._prepare_sample_einsum_datashapes('abc,acd->abd')
    )
    model = self._create_einsum_model(
        'abc,acd->abd',
        y_shape,
        x_signature,
        y_signature,
        bias_shape,
        is_qat_model=True,
    )
    saved_model_save.save(
        model, self._input_saved_model_path, signatures=model.einsum_with_kernel
    )

    signature_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    tags = {tag_constants.SERVING}
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=[signature_key],
        op_set=quant_opts_pb2.XLA,
        unit_wise_quantization_specs=[
            quant_opts_pb2.UnitWiseQuantizationSpec(
                unit=[
                    quant_opts_pb2.UnitWiseQuantizationSpec.QuantizationUnit(
                        op_type='Einsum',
                    ),
                ],
                quantization_method=quant_opts_pb2.QuantizationMethod(
                    preset_method=quant_opts_pb2.QuantizationMethod.METHOD_NO_QUANTIZE
                ),
            )
        ],
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
    )
    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {signature_key}
    )
    loader = saved_model_loader.SavedModelLoader(self._output_saved_model_path)
    graphdef = loader.get_meta_graph_def_from_tags(tags).graph_def
    # The Einsum ops shouldn't be quantized.
    self.assertTrue(self._contains_op(graphdef, 'Einsum'))
    self.assertFalse(self._contains_op(graphdef, 'XlaDotV2'))

  def test_selective_quantization_ptq(self):
    x_shape, y_shape, bias_shape, x_signature, y_signature = (
        self._prepare_sample_einsum_datashapes('abc,acd->abd')
    )
    model = self._create_einsum_model(
        'abc,acd->abd',
        y_shape,
        x_signature,
        y_signature,
        bias_shape,
        is_qat_model=False,
    )
    saved_model_save.save(
        model, self._input_saved_model_path, signatures=model.einsum_with_kernel
    )

    signature_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    tags = {tag_constants.SERVING}
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=[signature_key],
        op_set=quant_opts_pb2.XLA,
        unit_wise_quantization_specs=[
            quant_opts_pb2.UnitWiseQuantizationSpec(
                unit=[
                    quant_opts_pb2.UnitWiseQuantizationSpec.QuantizationUnit(
                        op_type='Einsum',
                    ),
                ],
                quantization_method=quant_opts_pb2.QuantizationMethod(
                    preset_method=quant_opts_pb2.QuantizationMethod.METHOD_NO_QUANTIZE
                ),
            )
        ],
    )

    rng = np.random.default_rng(seed=1234)

    def data_gen() -> repr_dataset.RepresentativeDataset:
      for _ in range(10):
        yield {
            'x': rng.uniform(low=0.0, high=1.0, size=x_shape).astype(np.float32)
        }

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
        representative_dataset=data_gen(),
    )
    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {signature_key}
    )
    loader = saved_model_loader.SavedModelLoader(self._output_saved_model_path)
    graphdef = loader.get_meta_graph_def_from_tags(tags).graph_def
    # The Einsum ops shouldn't be quantized.
    self.assertTrue(self._contains_op(graphdef, 'Einsum'))
    self.assertFalse(self._contains_op(graphdef, 'XlaDotV2'))

  def test_selective_quantization_on_gather(
      self,
  ):
    input_type = dtypes.int32
    model = self._create_simple_gather_and_conv_model(
        input_type,
        filter_shape=(2, 3, 3, 1024),
        is_qat_model=True,
    )

    saved_model_save.save(model, self._input_saved_model_path)

    tags = {tag_constants.SERVING}
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=quant_opts_pb2.XLA,
        unit_wise_quantization_specs=[
            quant_opts_pb2.UnitWiseQuantizationSpec(
                unit=[
                    quant_opts_pb2.UnitWiseQuantizationSpec.QuantizationUnit(
                        op_type='GatherV2',
                    ),
                ],
                quantization_method=quant_opts_pb2.QuantizationMethod(
                    preset_method=quant_opts_pb2.QuantizationMethod.METHOD_NO_QUANTIZE
                ),
            )
        ],
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
    )
    self.assertIsNotNone(converted_model)
    loader = saved_model_loader.SavedModelLoader(self._output_saved_model_path)
    graphdef = loader.get_meta_graph_def_from_tags(tags).graph_def
    # The Conv2D op shouldn't be quantized as it has no FakeQuant on input.
    self.assertTrue(self._contains_op(graphdef, 'Conv2D'))
    # If the Gather op is quantized, input_model_size / output_model_size > 2.
    self.assertLess(
        testing.get_size_ratio(
            self._input_saved_model_path, self._output_saved_model_path
        ),
        1.15,
    )


if __name__ == '__main__':
  test.main()
