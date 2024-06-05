# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Base test class for quantize_model Tests."""

from typing import List, Mapping, Optional, Sequence, Tuple

from absl.testing import parameterized
from mlir import ir
from mlir.dialects import stablehlo as stablehlo_dialect
import numpy as np
import tensorflow  # pylint: disable=unused-import

from tensorflow.compiler.mlir.stablehlo import stablehlo
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import loader_impl
from tensorflow.python.saved_model import save as saved_model_save
from tensorflow.python.types import core


FUNC_ALIAS = 'some_alias'


class QuantizedModelTest(test.TestCase, parameterized.TestCase):
  """Base test class for StableHLO quant tests."""

  def setUp(self) -> None:
    super().setUp()

    # Many test cases for quantization involve creating and saving the input
    # model and saving the output quantized model. These two member
    # attributes can be used to specify the paths for such models,
    # respectively. These paths will be cleaned up after each test case.
    self._input_saved_model_path = self.create_tempdir('input').full_path
    self._output_saved_model_path = self.create_tempdir('output').full_path
    # Extra output path occasionally used for comparing two different
    # quantized models.
    self._output_saved_model_path_2 = self.create_tempdir('output2').full_path

  def _extract_first_xla_call_module_op(
      self, output_saved_model_path: str
  ) -> str:
    """Extracts the first XlaCallModule op from output saved model to string."""
    root = load.load(output_saved_model_path)
    tf_graph_def = root.signatures['serving_default'].graph.as_graph_def()
    for function in tf_graph_def.library.function:
      for node_def in function.node_def:
        if node_def.op == 'XlaCallModule':
          with ir.Context() as context:
            stablehlo_dialect.register_dialect(context)
            # Serialization in VHLO dialect.
            serialized = node_def.attr.get('module').s
            # MLIR bytecode matching StableHLO version.
            mlir_bytecode = stablehlo.deserialize_portable_artifact(serialized)
            stablehlo_module = ir.Module.parse(mlir_bytecode, context=context)
            return str(stablehlo_module)
    raise ValueError('No XlaCallModule found in saved model.')

  def _get_num_xla_call_module_op(self, output_saved_model_path: str) -> int:
    """Gets the number of XlaCallModule ops in the output saved model."""
    root = load.load(output_saved_model_path)
    tf_graph_def = root.signatures['serving_default'].graph.as_graph_def()
    count = 0
    for node_def in tf_graph_def.node:
      if node_def.op == 'XlaCallModule':
        count += 1
    for function in tf_graph_def.library.function:
      for node_def in function.node_def:
        if node_def.op == 'XlaCallModule':
          count += 1
    return count

  def _get_function_aliases(
      self, output_saved_model_path: str, tags: List[str]
  ) -> dict[str, str]:
    """Gets the function aliases in the output saved model."""
    loader = loader_impl.SavedModelLoader(output_saved_model_path)
    return loader.get_meta_graph_def_from_tags(
        tags
    ).meta_info_def.function_aliases

  def _create_matmul_model(
      self,
      input_shape: Sequence[int],
      weight_shape: Sequence[int],
      saved_model_path: str,
      bias_fn: Optional[ops.Operation] = None,
      activation_fn: Optional[ops.Operation] = None,
  ) -> module.Module:
    class MatmulModel(module.Module):
      """A simple model with a single matmul.

      Bias and activation function are optional.
      """

      def __init__(
          self,
          weight_shape: Sequence[int],
      ) -> None:
        """Initializes a MatmulModel.

        Args:
          weight_shape: Shape of the weight tensor.
        """
        self.filters = np.random.uniform(low=-1.0, high=1.0, size=weight_shape)

        if bias_fn is not None:
          self.bias = np.random.uniform(
              low=-1.0, high=1.0, size=weight_shape[-1]
          )

      def has_reshape(self) -> bool:
        return self.bias_fn() and self.bias_size != self.filters.shape[-1]

      @def_function.function
      def matmul(self, input_tensor: core.Tensor) -> Mapping[str, core.Tensor]:
        """Performs a matrix multiplication.

        Depending on self.bias_fn and self.activation_fn, it may add a bias
        term or go through the activaction function.

        Args:
          input_tensor: Input tensor to matmul with the filter.

        Returns:
          A map of: output key -> output result.
        """
        out = math_ops.matmul(input_tensor, self.filters, name='sample/matmul')
        if bias_fn is not None:
          out = bias_fn(out, self.bias)
        if activation_fn is not None:
          out = activation_fn(out)
        return {'output': out}

    model = MatmulModel(weight_shape)
    saved_model_save.save(
        model,
        saved_model_path,
        signatures=model.matmul.get_concrete_function(
            tensor_spec.TensorSpec(
                shape=input_shape, dtype=dtypes.float32, name='input_tensor'
            )
        ),
    )
    return model

  def _any_log_contains(
      self, substring: str, log_record_list: List['logging.LogRecord']
  ) -> bool:
    """Returns True if any of the log contains a given substring.

    Args:
      substring: A piece of string to check whether it exists in the log
        message.
      log_record_list: A list of `absl.logging.LogRecord`s.

    Returns:
      True if and only if the substring exists in any of the log in
      `log_record_list`.
    """
    return any(
        map(
            lambda log_record: substring in str(log_record.message),
            log_record_list,
        )
    )

  def _create_matmul_and_same_scale_model(
      self,
      input_shape: Sequence[int],
      weight_shape: Sequence[int],
      saved_model_path: str,
      same_scale_op: str,
  ) -> module.Module:
    class MatmulAndSameScaleModel(module.Module):
      """A simple model with a same-scale op.

      Op name in StableHLO dialect is given as a string.
      """

      def __init__(
          self,
          weight_shape: Sequence[int],
          same_scale_op: str,
      ) -> None:
        """Initializes a MatmulModel.

        Args:
          weight_shape: Shape of the weight tensor.
          same_scale_op: Name of the same-scale op to be tested. Raises error
            when an unknown name is given.
        """
        self.filters = np.random.uniform(low=-1.0, high=1.0, size=weight_shape)
        self.same_scale_op = same_scale_op

      @def_function.function
      def matmul_and_same_scale(
          self, input_tensor: core.Tensor
      ) -> Mapping[str, core.Tensor]:
        """Performs a matrix multiplication.

        Args:
          input_tensor: Input tensor to matmul with the filter.

        Returns:
          A map of: output key -> output result.
        """
        out = math_ops.matmul(input_tensor, self.filters, name='sample/matmul')

        if self.same_scale_op == 'concatenate':
          ones = array_ops.ones_like(out)
          out = array_ops.concat([out, ones], 0)
        elif self.same_scale_op == 'gather':
          out = array_ops.gather(out, indices=[0], axis=0)
        elif self.same_scale_op == 'max_pool':
          out = nn_ops.max_pool(out, ksize=3, strides=1, padding='SAME')
        elif self.same_scale_op == 'pad':
          paddings = array_ops.ones(
              (array_ops.rank(out), 2), dtype=dtypes.int32
          )
          out = array_ops.pad(out, paddings, 'CONSTANT')
        elif self.same_scale_op == 'reshape':
          out = array_ops.reshape(out, [-1])
        elif self.same_scale_op == 'select':
          rng = np.random.default_rng(seed=1234)
          condition = ops.convert_to_tensor(
              rng.uniform(low=0.0, high=1.0, size=out.shape) < 0.5
          )
          ones = array_ops.ones_like(out)
          out = math_ops.select(condition, out, ones)
        elif self.same_scale_op == 'slice':
          begin = array_ops.zeros((array_ops.rank(out)), dtype=dtypes.int32)
          size = array_ops.ones((array_ops.rank(out)), dtype=dtypes.int32)
          out = array_ops.slice(out, begin, size)
        elif self.same_scale_op == 'transpose':
          out = array_ops.transpose(out)
        else:
          raise NotImplementedError(
              '{} is not implemented for integration test.'.format(
                  self.same_scale_op
              )
          )

        return {'output': out}

    model = MatmulAndSameScaleModel(weight_shape, same_scale_op)
    saved_model_save.save(
        model,
        saved_model_path,
        signatures=model.matmul_and_same_scale.get_concrete_function(
            tensor_spec.TensorSpec(
                shape=input_shape, dtype=dtypes.float32, name='input_tensor'
            )
        ),
    )
    return model

  def _create_conv2d_model(
      self,
      input_shape: Sequence[int],
      filter_shape: Sequence[int],
      saved_model_path: str,
      bias_fn: Optional[ops.Operation] = None,
      activation_fn: Optional[ops.Operation] = None,
      has_batch_norm: bool = False,
      strides: Sequence[int] = (1, 1, 1, 1),
      dilations: Sequence[int] = (1, 1, 1, 1),
      padding: str = 'SAME',
      has_func_alias: bool = False,
  ) -> module.Module:
    class ConvModel(module.Module):
      """A simple model with a single conv2d, bias and relu."""

      def __init__(self):
        self.out_channel_size = filter_shape[-1]

        # This ensures filters will have different value range per out channel
        self.filters = np.stack(
            [
                np.random.uniform(
                    low=-(i + 1), high=(i + 1), size=filter_shape[:-1]
                ).astype('f4')
                for i in range(self.out_channel_size)
            ],
            axis=-1,
        )

        self.bias = np.random.uniform(
            low=0, high=10, size=(self.out_channel_size)
        ).astype('f4')

      @def_function.function
      def conv2d(self, input_tensor: core.Tensor) -> Mapping[str, core.Tensor]:
        """Performs a 2D convolution operation.

        Args:
          input_tensor: Input tensor to perform convolution on.

        Returns:
          A map of: output key -> output result.
        """
        scale = [1.0] * self.out_channel_size
        offset = [0.5] * self.out_channel_size
        mean, variance = scale, offset
        out = nn_ops.conv2d(
            input_tensor,
            self.filters,
            strides=strides,
            dilations=dilations,
            padding=padding,
            data_format='NHWC',
            name='sample/conv',
        )
        if bias_fn is not None:
          out = nn_ops.bias_add(out, self.bias)
        if has_batch_norm:
          # Fusing is supported for non-training case.
          out, _, _, _, _, _ = nn_ops.fused_batch_norm_v3(
              out, scale, offset, mean, variance, is_training=False
          )
        if activation_fn is not None:
          out = activation_fn(out)
        return {'output': out}

    model = ConvModel()
    save_options = None
    if has_func_alias:
      save_options = tensorflow.saved_model.SaveOptions(
          function_aliases={FUNC_ALIAS: model.conv2d}
      )
    saved_model_save.save(
        model,
        saved_model_path,
        signatures=model.conv2d.get_concrete_function(
            tensor_spec.TensorSpec(
                shape=input_shape, dtype=dtypes.float32, name='input_tensor'
            )
        ),
        options=save_options,
    )
    return model

  def _create_gather_model(self, input_type, use_variable) -> module.Module:
    class GatherModel(module.Module):
      """A simple model with a single gather."""

      def __init__(self, use_variable):
        """Initializes a GatherModel.

        Args:
          use_variable: If True, creates a variable for weight.
        """
        super().__init__()
        w_val = np.random.randn(128, 32).astype('f4')
        if use_variable:
          self.w = variables.Variable(w_val)
        else:
          self.w = w_val

      @def_function.function(
          input_signature=[
              tensor_spec.TensorSpec(
                  shape=[6], dtype=input_type, name='input_tensor'
              )
          ]
      )
      def __call__(
          self, input_tensor: core.Tensor
      ) -> Mapping[str, core.Tensor]:
        """Performs a gather operation."""
        out = array_ops.gather_v2(self.w, input_tensor)
        return {'output': out}

    return GatherModel(use_variable)

  def _create_add_model(
      self,
      shape: Sequence[int],
      saved_model_path: str,
  ) -> module.Module:
    class AddModel(module.Module):
      """A simple model with a single add."""

      def __init__(self):
        pass

      @def_function.function
      def add(self, input_tensor: core.Tensor) -> Mapping[str, core.Tensor]:
        """Performs an add operation.

        Args:
          input_tensor: Input tensor to perform add on.

        Returns:
          A map of: output key -> output result.
        """
        out = math_ops.add(input_tensor, input_tensor)
        return {'output': out}

    model = AddModel()
    saved_model_save.save(
        model,
        saved_model_path,
        signatures=model.add.get_concrete_function(
            tensor_spec.TensorSpec(
                shape=shape, dtype=dtypes.float32, name='input_tensor'
            )
        ),
    )
    return model

  # Prepares sample einsum input data shapes.
  # This function returns:
  # 1. Shape for input 1
  # 2. Shape for input 2
  # 3. Shape for bias
  # 4. Signature for input 1 (Could contain None dimension)
  # 5. Signature for input 2 (Could contain None dimension)
  def _prepare_sample_einsum_datashapes(
      self,
      equation: str,
      generate_unknown_shape_signature: bool = False,
      use_bias: bool = False,
  ) -> Tuple[
      List[Optional[int]],
      List[Optional[int]],
      Optional[List[Optional[int]]],
      List[Optional[int]],
      List[Optional[int]],
  ]:
    # 1. Parse equation.
    comma_pos = equation.find(',')
    arrow_pos = equation.find('->')
    x_labels = equation[0:comma_pos]
    y_labels = equation[comma_pos + 1 : arrow_pos]
    out_labels = equation[arrow_pos + 1 :]

    # 2. Create sample shapes.
    label_to_size = {'a': 4, 'b': 32, 'c': 64, 'd': 128, 'e': 8}
    x_shape = [label_to_size.get(x_label) for x_label in x_labels]
    y_shape = [label_to_size.get(y_label) for y_label in y_labels]
    bias_shape = None
    if use_bias:
      bias_shape = [label_to_size.get(out_label) for out_label in out_labels]
      bias_shape = bias_shape[-1:]
    contracting_dims = set()

    x_signature = list(x_shape)
    y_signature = list(y_shape)
    if generate_unknown_shape_signature:
      for c in x_labels:
        if c in y_labels:
          contracting_dims.add(c)
      x_signature = [
          None if c not in contracting_dims else x_shape[cidx]
          for cidx, c in enumerate(x_labels)
      ]
      y_signature = [
          None if c not in contracting_dims else y_shape[cidx]
          for cidx, c in enumerate(y_labels)
      ]
    return x_shape, y_shape, bias_shape, x_signature, y_signature

  def _create_einsum_model(
      self,
      saved_model_path: str,
      equation: str,
      y_shape: Sequence[int],
      x_signature: Sequence[Optional[int]],
      y_signature: Sequence[Optional[int]],
      bias_shape: Optional[Sequence[int]] = None,
  ) -> module.Module:
    class EinsumModel(module.Module):
      """Einsum class."""

      def __init__(self):
        self._bias = None
        if bias_shape is not None:
          self._bias = array_ops.constant(
              np.random.uniform(size=bias_shape), dtype=dtypes.float32
          )

        self._kernel = np.random.uniform(size=y_shape).astype('f4')
        self._min = (-0.8, -0.8, -0.9)
        self._max = (0.9, 0.9, 1.0)

      @def_function.function(
          input_signature=[
              tensor_spec.TensorSpec(
                  name='x', shape=x_signature, dtype=dtypes.float32
              )
          ]
      )
      def einsum_with_kernel(self, x: core.Tensor) -> Mapping[str, core.Tensor]:
        return self._einsum(x, self._kernel)

      @def_function.function(
          input_signature=[
              tensor_spec.TensorSpec(
                  name='x', shape=x_signature, dtype=dtypes.float32
              ),
              tensor_spec.TensorSpec(
                  name='y', shape=y_signature, dtype=dtypes.float32
              ),
          ]
      )
      def einsum_without_kernel(
          self, x: core.Tensor, y: core.Tensor
      ) -> Mapping[str, core.Tensor]:
        return self._einsum(x, y)

      def _einsum(self, x, y):

        out = tensorflow.einsum(equation, x, y)
        if self._bias is not None:
          out = nn_ops.bias_add(out, self._bias)
        return {'output': out}

    model = EinsumModel()
    signatures = {
        'serving_default': model.einsum_with_kernel.get_concrete_function(
            tensor_spec.TensorSpec(
                name='x', shape=x_signature, dtype=dtypes.float32
            )
        ),
    }
    saved_model_save.save(model, saved_model_path, signatures=signatures)
    return model
