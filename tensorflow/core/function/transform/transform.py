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
"""High level TF Function transformation API."""

from typing import Any, Callable, Iterator, Optional, Union

from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import function_pb2
from tensorflow.core.function.capture import restore_captures
from tensorflow.core.function.runtime_client import runtime_client
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as function_lib
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import function_def_to_graph as function_def_lib
from tensorflow.python.framework import ops
from tensorflow.python.ops import custom_gradient as custom_gradient_lib
from tensorflow.python.ops import default_gradient
from tensorflow.python.ops import handle_data_util
from tensorflow.python.platform import tf_logging
from tensorflow.python.util import compat

_TensorType = Union[ops.EagerTensor, ops.Tensor]
_FunctionDefTransformerType = Callable[[function_pb2.FunctionDef], None]


def transform_function(
    f: def_function.Function,
    inputs: Optional[list[Any]] = None,
    kw_inputs: Optional[dict[str, Any]] = None,
    transform_fn: Optional[Union[_FunctionDefTransformerType,
                                 list[_FunctionDefTransformerType]]] = None,
    mlir_pipeline: Optional[Union[str, list[str]]] = None,
    nested_fn_transforms: Optional[dict[
        str, Optional[Union[_FunctionDefTransformerType,
                            list[_FunctionDefTransformerType]]]]] = None,
    nested_mlir_transforms: Optional[dict[str,
                                          Optional[Union[str,
                                                         list[str]]]]] = None,
) -> function_lib.ConcreteFunction:
  """Applies a transformation to a tf.function to produce a new callable.

  When `transform_fn` is specified, the underlying `FunctionDef` is modified
  according to the `transform_fn`.

  When `mlir_pipeline` is specified, the underlying `FunctionDef` is converted
  to an MLIR representation and transformed based on the rules of the
  `mlir_pipeline`.

  If both are provided, `mlir_pipeline` is applied followed by `transform_fn`.

  Optionally, `transform_fn` could be a list of transformation functions and
  `mlir_pipeline` could be a a list of MLIR transformations. The transformations
  will be applied in order of the list. For each nested `FunctionDef`, MLIR
  transformations will be applied before Python function based transformations.


  Example:
  ```python
  def edit_fn(fndef):
    for node in fndef.node_def:
      if node.name == "x_plus_y":
        node.name = "x_times_y"
        node.op = "Mul"
      for idx, inp in enumerate(node.input):
        if inp == "x_plus_y:z:0":
          node.input[idx] = "x_times_y:z:0"

  @tf.function(input_signature=[
      tf.TensorSpec((), dtype=tf.float32),
      tf.TensorSpec((), dtype=tf.float32)
  ])
  def add(x, y):
    return tf.add(x, y, name="x_plus_y")

  multiply = transform_function(add, transform_fn=edit_fn)
  assert multiply(1.0, 2.0) == 2.0
  ```

  Args:
    f: The target tf.function.
    inputs: The inputs or input_signature of the tf.function. This does not need
      to be specified if the `input_signature` was specified in the tf.function
      decorator.
    kw_inputs: The keyword inputs of the tf.function. This does not need to be
      specified if the `input_signature` was specified in the tf.function
      decorator.
    transform_fn: A single transformation function or a list of transformation
      functions to apply on the `FunctionDef`.
    mlir_pipeline: A single MLIR pass or a list of MLIR passes to transform the
      `FunctionDef`.
    nested_fn_transforms: A dict of Python function based transformations to
      apply on functions in the library of `f`. The keys are the names of the
      library functions being targeted for transformation.
    nested_mlir_transforms: A dict of MLIR pass based transformations to apply
      on functions in the library of `f`. The keys are the names of the library
      functions being targeted for transformation.

  Returns:
    The transformed function.
  """
  # Early exit if no transformations need to be applied.
  if transform_fn is None and mlir_pipeline is None:
    return f

  if transform_fn is None:
    transform_fns = []
  elif isinstance(transform_fn, list):
    transform_fns = transform_fn
  else:
    transform_fns = [transform_fn]

  if mlir_pipeline is None:
    mlir_pipelines = []
  elif isinstance(mlir_pipeline, list):
    mlir_pipelines = mlir_pipeline
  else:
    mlir_pipelines = [mlir_pipeline]

  nested_fn_transforms = (
      nested_fn_transforms if nested_fn_transforms is not None else {})
  nested_mlir_transforms = (
      nested_mlir_transforms if nested_mlir_transforms is not None else {})

  # Extract the `ConcreteFunction` from the `tf.function.`
  if inputs is not None or kw_inputs is not None:
    inputs = [] if inputs is None else inputs
    kw_inputs = {} if kw_inputs is None else kw_inputs
    cf = f.get_concrete_function(*inputs, **kw_inputs)
  else:
    cf = f.get_concrete_function()

  # Promote all library functions to the parent scope so that any replicated
  # functions can also re-use them.
  graph = ops.get_default_graph()
  for edf in cf.graph._functions.values():  # pylint: disable=protected-access
    edf.add_to_graph(graph, overwrite=False)

  # Initialize the `runtime_client`.
  eager_ctx = runtime_client.GlobalPythonEagerContext()
  rt = runtime_client.Runtime(eager_ctx)

  # Apply the MLIR passes if provided.
  for mlir_pipeline in mlir_pipelines:
    rt.TransformFunction(cf.function_def.signature.name, mlir_pipeline)

  # Get the most up-to-date FunctionDef for the tf.function. This should only
  # be read after applying any specified mlir_pipelines as they directly
  # transform the FunctionDef in the runtime.
  fndef = rt.GetFunctionProto(cf.function_def.signature.name)

  # Apply any transformations if provided.
  for transform_fn in transform_fns:
    transform_fn(fndef)

  # Apply a transform to any of the nested _EagerDefinedFunctions(EDF) if
  # `nested_fn_transforms` or `nested_mlir_transforms` is provided.
  if nested_fn_transforms or nested_mlir_transforms:
    nested_functions = cf.graph._functions  # pylint: disable=protected-access

    # Store the new transformed functions.
    transformed_nested_functions = {}

    # Store a mapping between the old nested function names and the new
    # transformed function names.
    nested_transforms_map = {}

    # Transform every nested function specified in `nested_fn_transforms` and
    # `nested_mlir_transforms`.
    for edf_name in nested_mlir_transforms.keys() | nested_fn_transforms.keys():
      if edf_name in nested_functions:
        edf_transform_fn = nested_fn_transforms.get(edf_name, [])
        edf_mlir_pipeline = nested_mlir_transforms.get(edf_name, [])
        transformed_edf = transform_eager_defined_function(
            rt, nested_functions[edf_name], edf_transform_fn, edf_mlir_pipeline)
        transformed_edf.add_to_graph(graph, overwrite=True)
        transformed_edf_name = compat.as_str(transformed_edf.name)
        transformed_nested_functions[transformed_edf_name] = transformed_edf
        nested_transforms_map[edf_name] = transformed_edf_name

    # Update the `FunctionDef` to map to the newly created EDFs.
    for node in fndef.node_def:
      for attr_value in node.attr.values():
        if attr_value.HasField("func"):
          attr_value.func.name = nested_transforms_map[attr_value.func.name]

  # Register the updated fndef with the runtime.
  rt.CreateFunction(fndef)

  # Create a new FuncGraph from the modified FunctionDef.
  structured_input_signature = cf.structured_input_signature
  structured_outputs_signature = (
      func_graph_module.convert_structure_to_signature(cf.structured_outputs))
  with graph.as_default():
    func_graph = function_def_lib.function_def_to_graph(
        fndef,
        structured_input_signature=structured_input_signature,
        structured_outputs=structured_outputs_signature,
        propagate_device_spec=True)

  # Set handle data.
  for i, output in enumerate(cf.outputs):
    func_graph_output = func_graph.outputs[i]
    if isinstance(output, ops.Tensor) and isinstance(func_graph_output,
                                                     ops.Tensor):
      func_graph_output.set_shape(output.shape)
      handle_data_util.copy_handle_data(output, func_graph_output)

  # We delete the `_input_shapes` attribute to avoid any intermediate
  # ShapeInference information from being carried over as the user's
  # transformations can invalidate them.
  if "_input_shapes" in fndef.attr:
    del fndef.attr["_input_shapes"]

  # Replicate custom gradients to the new Graph.
  with ops.init_scope():
    _replicate_gradient_functions(cf._func_graph, func_graph)  # pylint: disable=protected-access

  # pylint: disable=protected-access
  # Get the new ConcreteFunction.
  updated_cf = function_lib.ConcreteFunction(
      func_graph, attrs=fndef.attr, spec=cf._function_spec)

  # Set arg_keywords and positional_args
  updated_cf._arg_keywords = cf._arg_keywords
  updated_cf._num_positional_args = cf._num_positional_args
  restore_captures.restore_captures(updated_cf, cf.captured_inputs)
  # pylint: enable=protected-access

  # Register the ConcreteFunction with the python Graph.
  if nested_fn_transforms or nested_mlir_transforms:
    for transformed_edf in transformed_nested_functions.values():
      transformed_edf.add_to_graph(updated_cf.graph, overwrite=True)
  updated_cf.add_to_graph(graph, overwrite=True)

  return updated_cf


def transform_eager_defined_function(
    rt: runtime_client.Runtime,
    f: function_lib._EagerDefinedFunction,
    transform_fn: Union[_FunctionDefTransformerType,
                        list[_FunctionDefTransformerType]],
    mlir_pipeline: Union[str, list[str]],
) -> function_lib._EagerDefinedFunction:
  """Applies transforms on an _EagerDefinedFunction."""
  transform_fns = (
      transform_fn if isinstance(transform_fn, list) else [transform_fn])
  mlir_pipelines = (
      mlir_pipeline if isinstance(mlir_pipeline, list) else [mlir_pipeline])
  # First apply the MLIR based transformation.
  for mlir_pipeline in mlir_pipelines:
    rt.TransformFunction(f.cached_definition.signature.name, mlir_pipeline)

  # Get the `FunctionDef` after MLIR transformation.
  fndef = rt.GetFunctionProto(f.cached_definition.signature.name)

  # Apply the Python function based transformation.
  for transform_fn in transform_fns:
    transform_fn(fndef)
  rt.CreateFunction(fndef)

  # Generate a new `FuncGraph`
  graph = ops.get_default_graph()
  with graph.as_default():
    func_graph = function_def_lib.function_def_to_graph(
        fndef,
        structured_input_signature=f.graph.structured_input_signature,
        structured_outputs=f.graph.structured_outputs,
        propagate_device_spec=True)

  # pylint: disable=protected-access
  # Ref: third_party/tensorflow/python/ops/control_flow_util_v2.py
  # Generate a new `_EagerDefinedFunction`.
  edf = function_lib._EagerDefinedFunction(
      fndef.signature.name,
      func_graph,
      func_graph.inputs,
      func_graph.outputs,
      fndef.attr,
  )
  # pylint: enable=protected-access

  return edf


def _replicate_gradient_functions(
    original_graph: func_graph_module.FuncGraph,
    replicated_graph: func_graph_module.FuncGraph) -> None:
  """Copies over any custom_gradients defined within the original Graph."""
  seen_ops = set()
  for gradient_op_type, op in _ops_with_custom_gradients(
      replicated_graph.get_operations()):
    # Soft-cache processed ops so we do not repeat the computation.
    if gradient_op_type in seen_ops:
      continue
    seen_ops.add(gradient_op_type)

    # Lookup the custom_gradient implementation if it exists. Currently all
    # custom_gradients are stored as python functions in a gradient_registry.
    # The gradient_registry returns a LookupError when a lookup fails.
    try:
      custom_gradient = ops.gradient_registry.lookup(gradient_op_type)
    except LookupError:
      continue

    # Convert the custom_gradient to a `ConcreteFunction`. This is done so we
    # can replicate the custom gradient and update any python captures.
    try:
      grad_fn = def_function.function(custom_gradient).get_concrete_function(
          None, *op.inputs)
    except Exception:  # pylint: disable=broad-except
      # TODO(xjun): Figure out why tracing of custom_gradient will fail.
      tf_logging.exception(
          f"Error when tracing gradients for {replicated_graph}.")
      continue

    # Re-bind all captures to values within the replicated graph.
    remapped_captures = []
    for capture in grad_fn.captured_inputs:
      outer_graph, outer_capture = _get_outer_most_capture(
          original_graph, capture)

      # We only need to re-bind captures originating from the `original_graph`.
      if outer_graph is not original_graph:
        continue

      if outer_capture.graph is not outer_graph:
        raise ValueError(
            f"Cannot replicate graph: {original_graph}. It utilizes a "
            f"`tf.custom_gradient` for op: {op} which has a "
            f"non-replicable capture: {capture}. Consider re-factoring your "
            f"custom_gradient to avoid the capture.")

      remapped_captures.append(
          replicated_graph.get_tensor_by_name(outer_capture.name))
    restore_captures.restore_captures(grad_fn, remapped_captures)
    new_gradient_op_type = custom_gradient_lib.generate_name()
    op._set_attr(  # pylint: disable=protected-access
        "_gradient_op_type",
        attr_value_pb2.AttrValue(s=compat.as_bytes(new_gradient_op_type)))
    ops.RegisterGradient(new_gradient_op_type)(_gen_gradient_func(grad_fn))


def _gen_gradient_func(func):
  """Wraps a ConcreteFunction to be compatible with the gradient registry."""

  def gradient_func(unused_op, *result_grads):
    result_grads = [
        x if x is not None else default_gradient.zeros_like(t)
        for (x, t) in zip(result_grads, func.graph.inputs)
    ]
    return func(*result_grads)

  return gradient_func


def _get_outer_most_capture(
    original_graph: func_graph_module.FuncGraph,
    capture: _TensorType) -> tuple[func_graph_module.FuncGraph, _TensorType]:
  """Tries to find the original captured tensor."""
  outer_graph = original_graph
  while outer_graph is not None and not isinstance(capture, ops.EagerTensor):
    if capture.graph is not outer_graph:
      outer_graph = outer_graph.outer_graph
    else:
      try:
        capture_index = outer_graph.internal_captures.index(capture)
      except ValueError:
        # Capture is a tensor inside the function and is not captured from
        # another external function
        break
      capture = outer_graph.external_captures[capture_index]
      outer_graph = outer_graph.outer_graph

  return outer_graph, capture


def _ops_with_custom_gradients(
    operations: list[ops.Operation]) -> Iterator[tuple[str, ops.Operation]]:
  """Returns an iterator over ops having custom_gradients."""
  for op in operations:
    try:
      gradient_op_type = op.get_attr("_gradient_op_type")
    except ValueError:
      continue
    yield gradient_op_type, op
