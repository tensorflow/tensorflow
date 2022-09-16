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
"""Base test class for quantize_model Tests."""
from typing import Iterable, Mapping, Sequence, Set, Tuple, Optional

from absl.testing import parameterized
import numpy as np
import tensorflow  # pylint: disable=unused-import

from tensorflow.compiler.mlir.quantization.tensorflow.python import representative_dataset as repr_dataset
from tensorflow.core.framework import function_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.client import session
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model import signature_def_utils_impl
from tensorflow.python.trackable import autotrackable
from tensorflow.python.types import core


class QuantizedModelTest(test.TestCase, parameterized.TestCase):
  """Base test class for TF-quant tests."""

  def _is_quantized_function(self, func: function_pb2.FunctionDef) -> bool:
    """Determine whether a FunctionDef is quantized.

    Args:
      func: A FunctionDef object.

    Returns:
      True iff `func` is quantized.
    """
    return func.signature.name.startswith('quantized_')

  def _is_composite_function(self, func: function_pb2.FunctionDef) -> bool:
    """Determine whether a FunctionDef is composite function.

    Args:
      func: A FunctionDef object.

    Returns:
      True iff `func` is composte function.
    """
    return func.signature.name.startswith('composite_')

  def _contains_op_with_name(self, nodes: Iterable[node_def_pb2.NodeDef],
                             op_name: str) -> bool:
    """Determine whether there is a node whose operation name matches `op_name`.

    Args:
      nodes: Iterable of NodeDefs.
      op_name: Name of the op to match.

    Returns:
      True iff there exists a node whose name matches `op_name`.
    """
    return any(node.op == op_name for node in nodes)

  def _contains_quantized_function_call(
      self, meta_graphdef: meta_graph_pb2.MetaGraphDef) -> bool:
    """Determines if the graph def has quantized function call.

    Args:
      meta_graphdef: A MetaGraphDef object.

    Returns:
      True if and only if the graph def contains a quantized function call.
    """
    return any(
        map(self._is_quantized_function,
            meta_graphdef.graph_def.library.function))

  def _contains_composite_function_call(
      self, meta_graphdef: meta_graph_pb2.MetaGraphDef) -> bool:
    """Determines if the graph def has composite function call.

    Args:
      meta_graphdef: A MetaGraphDef object.

    Returns:
      True if and only if the graph def contains a composite function call.
    """
    return any(
        map(self._is_composite_function,
            meta_graphdef.graph_def.library.function))

  def _contains_op(self, meta_graphdef: meta_graph_pb2.MetaGraphDef,
                   op_name: str) -> bool:
    """Determines if the graph def contains the given op.

    Args:
      meta_graphdef: A MetaGraphDef object.
      op_name: Name of the operation to find within the graph.

    Returns:
      True if and only if the graph def contains an op named `op_name`.
    """
    # Check the main graph
    if self._contains_op_with_name(
        nodes=meta_graphdef.graph_def.node, op_name=op_name):
      return True

    # Check the graph genederated from user defined functions
    return any(
        self._contains_op_with_name(nodes=func.node_def, op_name=op_name)
        for func in meta_graphdef.graph_def.library.function)

  def _create_simple_tf1_conv_model(self,
                                    use_variable_for_filter=False
                                   ) -> Tuple[core.Tensor, core.Tensor]:
    """Creates a basic convolution model.

    This is intended to be used for TF1 (graph mode) tests.

    Args:
      use_variable_for_filter: Setting this to `True` makes the filter for the
        conv operation a `tf.Variable`.

    Returns:
      in_placeholder: Input tensor placeholder.
      output_tensor: The resulting tensor of the convolution operation.
    """
    in_placeholder = array_ops.placeholder(dtypes.float32, shape=[1, 3, 4, 3])

    filters = random_ops.random_uniform(
        shape=(2, 3, 3, 2), minval=-1., maxval=1.)
    if use_variable_for_filter:
      filters = variables.Variable(filters)

    output_tensor = nn_ops.conv2d(
        in_placeholder,
        filters,
        strides=[1, 1, 2, 1],
        dilations=[1, 1, 1, 1],
        padding='SAME',
        data_format='NHWC')

    return in_placeholder, output_tensor

  def _create_simple_tf1_gather_model(self,
                                      use_variable_for_filter=False
                                     ) -> Tuple[core.Tensor, core.Tensor]:
    """Creates a basic gather model.

    This is intended to be used for TF1 (graph mode) tests.

    Args:
      use_variable_for_filter: Setting this to `True` makes the filter for the
        gather operation a `tf.Variable`.

    Returns:
      in_placeholder: Input tensor placeholder.
      output_tensor: The resulting tensor of the gather operation.
    """
    in_placeholder = array_ops.placeholder(dtypes.int64, shape=(6))

    filters = random_ops.random_uniform(shape=(64, 512), minval=-1., maxval=1.)
    if use_variable_for_filter:
      filters = variables.Variable(filters)

    output_tensor = array_ops.gather_v2(filters, in_placeholder)

    return in_placeholder, output_tensor

  def _create_data_generator(
      self,
      input_key: str,
      shape: Sequence[int],
      minval=-1.,
      maxval=1.,
      dtype=dtypes.float32,
      num_examples=8) -> repr_dataset.RepresentativeDataset:
    """Creates a data generator to be used as representative dataset.

    Supports generating random value input tensors mapped by the `input_key`.

    Args:
      input_key: The string key that identifies the created tensor as an input.
      shape: Shape of the tensor data.
      minval: The lower bound of the generated input
      maxval: The upper bound of the generated input
      dtype: The type of the generated input - usually dtypes.float32 for float
        and dtypes.int64 for int
      num_examples: Number of examples in the representative dataset.

    Yields:
      data_gen: A `quantize_model._RepresentativeSample` filled with random
        values.
    """
    for _ in range(num_examples):
      yield {input_key: random_ops.random_uniform(shape, minval, maxval, dtype)}

  def _save_tf1_model(self, sess: session.Session, saved_model_path: str,
                      signature_key: str, tags: Set[str],
                      inputs: Mapping[str, core.Tensor],
                      outputs: Mapping[str, core.Tensor]) -> None:
    """Saves a TF1 model.

    Args:
      sess: Current tf.Session object.
      saved_model_path: Directory to save the model.
      signature_key: The key to the SignatureDef that inputs & outputs
        correspond to.
      tags: Set of tags associated with the model.
      inputs: Input name -> input tensor mapping.
      outputs: Output name -> output tensor mapping.
    """
    v1_builder = builder.SavedModelBuilder(saved_model_path)
    sig_def = signature_def_utils_impl.predict_signature_def(
        inputs=inputs, outputs=outputs)

    v1_builder.add_meta_graph_and_variables(
        sess, tags, signature_def_map={signature_key: sig_def})
    v1_builder.save()

  def _create_and_save_tf1_gather_model(self,
                                        saved_model_path: str,
                                        signature_key: str,
                                        tags: Set[str],
                                        input_key: str,
                                        output_key: str,
                                        use_variable=False) -> core.Tensor:
    """Creates and saves a simple gather model.

    This is intended to be used for TF1 (graph mode) tests.

    Args:
      saved_model_path: Directory to save the model.
      signature_key: The key to the SignatureDef that inputs & outputs
        correspond to.
      tags: Set of tags associated with the model.
      input_key: The key to the input tensor.
      output_key: The key to the output tensor.
      use_variable: Setting this to `True` makes the filter for the gather
        operation a `tf.Variable`.

    Returns:
      in_placeholder: The placeholder tensor used as an input to the model.
    """
    with ops.Graph().as_default(), session.Session() as sess:
      in_placeholder, output_tensor = self._create_simple_tf1_gather_model(
          use_variable_for_filter=use_variable)

      if use_variable:
        sess.run(variables.global_variables_initializer())

      self._save_tf1_model(
          sess,
          saved_model_path,
          signature_key,
          tags,
          inputs={input_key: in_placeholder},
          outputs={output_key: output_tensor})

      return in_placeholder

  def _create_gather_model(self, use_variable):

    class GatherModel(autotrackable.AutoTrackable):
      """A simple model with a single gather."""

      def __init__(self, use_variable):
        """Initializes a GatherModel.

        Args:
          use_variable: If True, creates a variable for weight.
        """
        super(GatherModel, self).__init__()
        w_val = np.random.randint(
            low=0, high=100, size=(64, 512), dtype=np.int64)
        if use_variable:
          self.w = variables.Variable(w_val)
        else:
          self.w = w_val

      @def_function.function(input_signature=[
          tensor_spec.TensorSpec(
              shape=[6], dtype=dtypes.int64, name='input_tensor')
      ])
      def __call__(self,
                   input_tensor: core.Tensor) -> Mapping[str, core.Tensor]:
        """Performs a gather operation."""
        out = array_ops.gather_v2(self.w, input_tensor)
        return {'output': out}

    return GatherModel(use_variable)

  def _create_conv2d_model(self):

    class ConvModel(module.Module):
      """A simple model with a single conv2d, bias and relu."""

      @def_function.function(input_signature=[
          tensor_spec.TensorSpec(shape=[1, 3, 4, 512], dtype=dtypes.float32)
      ])
      def conv(self, input_tensor: core.Tensor) -> Mapping[str, core.Tensor]:
        """Performs a 2D convolution operation.

        Args:
          input_tensor: Input tensor to perform convolution on.

        Returns:
          A map of: output key -> output result.
        """
        filters = np.random.uniform(
            low=-10, high=10, size=(2, 3, 512, 2)).astype('f4')
        bias = np.random.uniform(low=0, high=10, size=(2)).astype('f4')
        out = nn_ops.conv2d(
            input_tensor,
            filters,
            strides=[1, 1, 2, 1],
            dilations=[1, 1, 1, 1],
            padding='SAME',
            data_format='NHWC')
        out = nn_ops.bias_add(out, bias, data_format='NHWC')
        out = nn_ops.relu6(out)
        return {'output': out}

    return ConvModel()

  def _create_matmul_model(self,
                           has_bias: bool = False,
                           activation_fn: Optional[ops.Operation] = None) ->...:

    class MatmulModel(module.Module):
      """A simple model with a single matmul.

      Bias and activation function are optional.
      """

      def __init__(self,
                   has_bias: bool = False,
                   activation_fn: Optional[ops.Operation] = None) -> None:
        """Initializes a MatmulModel.

        Args:
          has_bias: If True, creates and adds a bias term.
          activation_fn: The activation function to be used. No activation
            function if None.
        """
        self.has_bias = has_bias
        self.activation_fn = activation_fn
        self.filters = np.random.uniform(low=-1.0, high=1.0, size=(1024, 3))
        self.bias = np.random.uniform(low=-1.0, high=1.0, size=(3,))

      @def_function.function(input_signature=[
          tensor_spec.TensorSpec(
              shape=(1, 1024), dtype=dtypes.float32, name='input_tensor')
      ])
      def matmul(self, input_tensor: core.Tensor) -> Mapping[str, core.Tensor]:
        """Performs a matrix multiplication.

        Depending on self.has_bias and self.activation_fn, it may add a bias
        term or
        go through the activaction function.

        Args:
          input_tensor: Input tensor to matmul with the filter.

        Returns:
          A map of: output key -> output result.
        """
        out = math_ops.matmul(input_tensor, self.filters)

        if self.has_bias:
          out = nn_ops.bias_add(out, self.bias)

        if self.activation_fn is not None:
          out = self.activation_fn(out)

        return {'output': out}

    return MatmulModel(has_bias, activation_fn)

  def _create_and_save_tf1_conv_model(self,
                                      saved_model_path: str,
                                      signature_key: str,
                                      tags: Set[str],
                                      input_key: str,
                                      output_key: str,
                                      use_variable=False) -> core.Tensor:
    """Creates and saves a simple convolution model.

    This is intended to be used for TF1 (graph mode) tests.

    Args:
      saved_model_path: Directory to save the model.
      signature_key: The key to the SignatureDef that inputs & outputs
        correspond to.
      tags: Set of tags associated with the model.
      input_key: The key to the input tensor.
      output_key: The key to the output tensor.
      use_variable: Setting this to `True` makes the filter for the conv
        operation a `tf.Variable`.

    Returns:
      in_placeholder: The placeholder tensor used as an input to the model.
    """
    with ops.Graph().as_default(), session.Session() as sess:
      in_placeholder, output_tensor = self._create_simple_tf1_conv_model(
          use_variable_for_filter=use_variable)

      if use_variable:
        sess.run(variables.global_variables_initializer())

      self._save_tf1_model(
          sess,
          saved_model_path,
          signature_key,
          tags,
          inputs={input_key: in_placeholder},
          outputs={output_key: output_tensor})

    return in_placeholder
