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
"""Implementation for AtomicFunction."""

import dataclasses
from typing import Any

from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.function.polymorphism import function_type as function_type_lib
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.eager import context
from tensorflow.python.eager import record
from tensorflow.python.eager.polymorphic_function import attributes as attributes_lib
from tensorflow.python.framework import auto_control_deps_utils as acd
from tensorflow.python.framework import error_interpolation
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import handle_data_util
from tensorflow.python.util import compat
from tensorflow.python.util import function_utils


class _InterpolateFunctionError(object):
  """Context Manager that interpolates the exception from 'top_level_func'."""

  __slots__ = ["_func"]

  def __init__(self, top_level_func):
    self._func = top_level_func

  def __enter__(self):
    pass

  def __exit__(self, typ, exc, tb):
    if not exc or not isinstance(exc, errors.OpError):
      return False
    message = compat.as_text(exc.message)
    _, func_tags, _ = error_interpolation.parse_message(message)
    g = None
    for func_tag in func_tags:
      # TODO(mdan): Tests should cover this.
      if func_tag.name == compat.as_str(self._func.name):
        g = self._func.graph
      elif g:
        next_func = g._get_function(func_tag.name)  # pylint: disable=protected-access
        if next_func is not None and isinstance(next_func, AtomicFunction):
          g = next_func.graph
    if g:
      exc._message = error_interpolation.interpolate(message, g)  # pylint: disable=protected-access
    return False


# TODO(b/232961485): Remove after quarantined `add_function_callback` removed.
function_callbacks = set()


# TODO(fmuham): Lower to FunctionRecord or remove otherwise.
@dataclasses.dataclass(frozen=True)
class GraphArtifacts:
  inputs: Any
  outputs: Any
  num_outputs: Any
  output_types: Any
  output_shapes: Any
  control_captures: Any
  func_graph_outputs: Any
  graph: Any
  stateful_ops: Any

# Maps the scope_id and name in runtime to the number of AtomicFunctions.
RUNTIME_FUNCTION_REFS = {}


class AtomicFunction:
  """A Python callable for functions in the TF Runtime.

  Supports tf.function features such as structured value inputs and outputs,
  captures and control dependencies.

  Lowest level abstraction in the Python tf.function implementation.
  """
  __slots__ = [
      "_name",
      "_bound_context",
      "_function_type",
      "_graph_artifacts",
      "_cached_definition",
  ]

  def __init__(self, name, bound_context, function_type, graph_artifacts):
    self._name = compat.as_bytes(name)
    self._bound_context = bound_context
    self._function_type = function_type
    self._graph_artifacts = graph_artifacts
    self._cached_definition = None

    ref_key = (self._bound_context.function_scope_id, self.name)
    if ref_key not in RUNTIME_FUNCTION_REFS:
      RUNTIME_FUNCTION_REFS[ref_key] = 1
    else:
      RUNTIME_FUNCTION_REFS[ref_key] += 1

  @property
  def _c_func(self):
    return context.get_c_function(self.name)

  @property
  def function_type(self):
    return self._function_type

  # TODO(fmuham): Remove this property.
  @property
  def graph(self):
    return self._graph_artifacts.graph

  # TODO(fmuham): Remove this property.
  @property
  def stateful_ops(self):
    return self._graph_artifacts.stateful_ops

  @property
  def definition(self):
    """Current FunctionDef in the Runtime."""
    return self._bound_context.get_function_def(self.name)

  # TODO(fmuham): Move caching to dependent code and remove method.
  @property
  def cached_definition(self):
    """Cached FunctionDef (not guaranteed to be fresh)."""
    if self._cached_definition is None:
      self._cached_definition = self.definition

    return self._cached_definition

  @property
  def name(self):
    """Name represented in UTF-8 encoded bytes."""
    return self._name

  @property
  def graph_call_attrs(self):
    """Returns a dictionary of attributes needed to add a call in graph."""
    attrs = {
        "is_stateful": len(self.stateful_ops) > 0,  # pylint: disable=g-explicit-length-test
        "tout": self._graph_artifacts.output_types,
        "xla_compile_attr": self.cached_definition.attr.get(
            attributes_lib.XLA_COMPILE, None
        ),
    }
    attrs.update(self._bound_context.function_call_options.as_attrs())
    return attrs

  def __call__(self, *args):
    """Calls this function with `args` as inputs.

    `ConcreteFunction` execution respects device annotations only if the
    function won't be compiled with xla.

    Args:
      *args: arguments to call this function with.

    Returns:
      The outputs of the function call.

    Raises:
      ValueError: if the number of arguments is incorrect.
      FunctionAlreadyGarbageCollectedError: if the function is no longer
        available to be called because it has been garbage collected.
    """
    if len(args) != len(self.cached_definition.signature.input_arg):
      raise ValueError(
          "Signature specifies"
          f" {len(list(self.cached_definition.signature.input_arg))} arguments,"
          f" got: {len(args)}."
      )

    with _InterpolateFunctionError(self):
      with ops.control_dependencies(self._graph_artifacts.control_captures):
        # The caller must use record_operation to record this operation in the
        # eager case, so we enforce the same requirement for the non-eager
        # case by explicitly pausing recording. We don't have a gradient
        # registered for PartitionedCall, so recording this operation confuses
        # forwardprop code (GradientTape manages to ignore it).
        with record.stop_recording():
          if self._bound_context.executing_eagerly():
            outputs = self._bound_context.call_function(
                self.name,
                list(args),
                self._graph_artifacts.num_outputs,
            )
          else:
            outputs = make_call_op_in_graph(self, list(args))

    for i, func_graph_output in enumerate(
        self._graph_artifacts.func_graph_outputs
    ):
      handle_data_util.copy_handle_data(func_graph_output, outputs[i])

    # TODO(fmuham): Use FunctionType cast here for all cases.
    if not self._bound_context.executing_eagerly():
      for i, shape in enumerate(self._graph_artifacts.output_shapes):
        outputs[i].set_shape(shape)

    return outputs

  def __del__(self):
    key = (self._bound_context.function_scope_id, self.name)
    RUNTIME_FUNCTION_REFS[key] -= 1
    if RUNTIME_FUNCTION_REFS[key] < 0:
      raise RuntimeError(
          f"AtomicFunction Refcounting for {self.name} is invalid."
      )

    if RUNTIME_FUNCTION_REFS[key] == 0:
      try:
        self._bound_context.remove_function(self.name)
        RUNTIME_FUNCTION_REFS.pop(key)
      except TypeError:
        # Suppress some exceptions, mainly for the case when we're running on
        # module deletion. Things that can go wrong include the context module
        # already being unloaded, self._handle._handle_data no longer being
        # valid, and so on. Printing warnings in these cases is silly
        # (exceptions raised from __del__ are printed as warnings to stderr).
        pass  # 'NoneType' object is not callable when the handle has been
        # partially unloaded.
      except AttributeError:
        pass  # 'NoneType' object has no attribute 'eager_mode' when context has
        # been unloaded. Will catch other module unloads as well.


def _set_read_only_resource_inputs_attr(op, func_graph):
  """Sets the list of resource inputs which are read-only.

  This is used by AutomaticControlDependencies.

  Args:
    op: PartitionedCall Operation.
    func_graph: FuncGraph.
  """
  read_only_indices = acd.get_read_only_resource_input_indices_graph(func_graph)
  ops.set_int_list_attr(op, acd.READ_ONLY_RESOURCE_INPUTS_ATTR,
                        read_only_indices)


def partitioned_call_op(
    name,
    args,
    is_stateful,
    tout,
    config=None,
    executor_type=None,
    xla_compile_attr=None,
):
  """Generates a function call op respecting device annotations.

  Args:
    name: Name of the function to call.
    args: The arguments of the function, including captured inputs.
    is_stateful: If the function is stateful.
    tout: a list containing the output dtypes enums
    config: (Optional) A `tensorflow::ConfigProto` proto, serialized. If `None`,
      all optimizations are disabled. Currently only handled for eager defined
      functions.
    executor_type: (Optional) A string for the name of the executor to be used
      in the function call. If not set, or set to an empty string, the default
      tensorflow executor will be used.
    xla_compile_attr: (Optional) value of the XLA compilation attribute.

  Returns:
    Returns the operation.
  """
  if config is None:
    config = function_utils.get_disabled_rewriter_config()

  if executor_type is None:
    executor_type = ""

  # The generated binding returns an empty list for functions that don't
  # return any Tensors, hence the need to use `create_op` directly.
  args = [ops.convert_to_tensor(x) for x in args]
  tin_attr = attr_value_pb2.AttrValue(
      list=attr_value_pb2.AttrValue.ListValue(
          type=[x.dtype.as_datatype_enum for x in args]))
  tout_attr = attr_value_pb2.AttrValue(
      list=attr_value_pb2.AttrValue.ListValue(type=tout))
  func_attr = attr_value_pb2.AttrValue(
      func=attr_value_pb2.NameAttrList(name=name))
  executor_type_attr = attr_value_pb2.AttrValue(
      s=compat.as_bytes(executor_type))

  # When running in graph mode, the graph and function graphs are optimized
  # (i.e. run through grappler) per the session options, so we can disable any
  # eager-specific rewriting.
  config_proto = attr_value_pb2.AttrValue(s=config)

  op_name = "StatefulPartitionedCall" if is_stateful else "PartitionedCall"

  # Propagate the attribute indicating the need to compile from function to the
  # call itself.
  op_attrs = {
      "Tin": tin_attr,
      "Tout": tout_attr,
      "f": func_attr,
      "config_proto": config_proto,
      "executor_type": executor_type_attr,
  }
  if xla_compile_attr is not None:
    op_attrs[attributes_lib.XLA_COMPILE] = xla_compile_attr

  op = ops.get_default_graph().create_op(
      op_name, args, tout, name=op_name, attrs=op_attrs
  )
  return op


def make_call_op_in_graph(atomic, tensor_inputs):
  """Adds an AtomicFunction to graph."""
  graph = ops.get_default_graph()
  graph._add_function_recursive(atomic)  # pylint: disable=protected-access

  function_call_attrs = atomic.graph_call_attrs
  op = partitioned_call_op(
      name=atomic.name,
      args=tensor_inputs,
      is_stateful=function_call_attrs["is_stateful"],
      tout=function_call_attrs["tout"],
      config=function_call_attrs["config_proto"],
      executor_type=function_call_attrs["executor_type"],
      xla_compile_attr=function_call_attrs["xla_compile_attr"],
  )
  _set_read_only_resource_inputs_attr(op, atomic.graph)
  if hasattr(atomic.graph, "collective_manager_ids_used"):
    ops.set_int_list_attr(
        op,
        acd.COLLECTIVE_MANAGER_IDS,
        atomic.graph.collective_manager_ids_used,
    )
  return op.outputs if op.outputs else op

# List of AtomicFunction -> AtomicFunction transformation functions.
FUNCTION_TRANSFORMS = []


def from_func_graph(name, graph, inputs, outputs, attrs):
  """Initializes an AtomicFunction from FuncGraph with transforms."""

  atomic = from_func_graph_no_transforms(name, graph, inputs, outputs, attrs)
  for transform in FUNCTION_TRANSFORMS:
    atomic = transform(atomic)
    if not isinstance(atomic, AtomicFunction):
      raise TypeError(
          f"Transformation {transform} did not return an AtomicFunction."
      )

  return atomic


def from_func_graph_no_transforms(
    name, graph, inputs, outputs, attrs, overwrite=False
):
  """Initializes an AtomicFunction from FuncGraph.

  Args:
    name: str, the name for the created function.
    graph: Graph, the graph containing the operations in the function
    inputs: the tensors in the graph to be used as inputs to the function
    outputs: the tensors in the graph which will be outputs from the function
    attrs: dict mapping names of attributes to their AttrValue values
    overwrite: overwrites function definition in the current context if needed

  Returns:
    An AtomicFunction instance.
  """
  input_ops = set(arg.op for arg in inputs)
  operations = [op for op in graph.get_operations() if op not in input_ops]

  graph_output_names = graph._output_names  # pylint: disable=protected-access
  if graph_output_names is not None and all(
      ops.tensor_id(t) in graph_output_names for t in outputs
  ):
    output_names = [
        compat.as_bytes(graph_output_names[ops.tensor_id(t)]) for t in outputs
    ]
    if len(set(output_names)) != len(output_names):
      # There are duplicate names for some reason, probably an invalid
      # signature. Revert to auto-naming.
      output_names = []
  else:
    output_names = []
  with graph._c_graph.get() as c_graph:  # pylint: disable=protected-access
    fn = pywrap_tf_session.TF_GraphToFunction_wrapper(
        c_graph,
        compat.as_str(name),
        False,
        [o._c_op for o in operations],  # pylint: disable=protected-access
        [t._as_tf_output() for t in inputs],  # pylint: disable=protected-access
        [t._as_tf_output() for t in outputs],  # pylint: disable=protected-access
        output_names,
        [o._c_op for o in graph.control_outputs],  # pylint: disable=protected-access
        [],  # control_output_names
        None,
        compat.as_str(""),
    )

  for attr_name, attr_value in attrs.items():
    serialized = attr_value.SerializeToString()
    pywrap_tf_session.TF_FunctionSetAttrValueProto(
        fn, compat.as_str(attr_name), serialized
    )

  name = compat.as_bytes(name)
  bound_context = context.context()

  if overwrite and bound_context.has_function(name):
    bound_context.remove_function(name)

  bound_context.add_c_function(fn)
  pywrap_tf_session.TF_DeleteFunction(fn)

  signature = bound_context.get_function_def(name).signature
  graph_artifacts = GraphArtifacts(
      inputs=inputs,
      outputs=outputs,
      num_outputs=len(signature.output_arg),
      output_types=[o.type for o in signature.output_arg],
      output_shapes=[o.shape for o in outputs],
      control_captures=graph.function_captures.control,
      func_graph_outputs=list(outputs),
      graph=graph,
      stateful_ops=tuple(op for op in operations if op._is_stateful),  # pylint: disable=protected-access
  )

  if graph.structured_input_signature is not None:
    input_signature = graph.structured_input_signature
  else:
    input_signature = (
        tuple(tensor_spec.TensorSpec.from_tensor(i) for i in inputs),
        {},
    )

  if graph.structured_outputs is not None:
    output_signature = graph.structured_outputs
  else:
    output_signature = tuple(
        tensor_spec.TensorSpec.from_tensor(o) for o in outputs
    )

  function_type = function_type_lib.from_structured_signature(
      input_signature,
      output_signature,
      graph.function_captures.capture_types,
  )

  return AtomicFunction(name, bound_context, function_type, graph_artifacts)
