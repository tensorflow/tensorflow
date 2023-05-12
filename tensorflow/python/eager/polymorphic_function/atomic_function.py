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
import traceback
from typing import Any, List

from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.function import trace_type
from tensorflow.core.function.polymorphism import function_type as function_type_lib
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.eager import context
from tensorflow.python.eager import record
from tensorflow.python.eager.polymorphic_function import attributes as attributes_lib
from tensorflow.python.framework import auto_control_deps_utils as acd
from tensorflow.python.framework import error_interpolation
from tensorflow.python.framework import errors
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import function_def_to_graph
from tensorflow.python.framework import ops
from tensorflow.python.ops import handle_data_util
from tensorflow.python.util import compat
from tensorflow.python.util import function_utils


# TODO(fmuham): Should be lowered to FunctionDef/FunctionRecord.
@dataclasses.dataclass(frozen=True)
class CallOptions:
  """Specifies additional configuration for an AtomicFunction call."""

  # Used by ACD to identify the CollectiveManager this function is scoped in.
  collective_manager_ids_used: List[int] = dataclasses.field(
      default_factory=list
  )

  # Used by ACD to list Ops/Tensors/Callables that must be called in advance.
  control_captures: List[Any] = dataclasses.field(default_factory=list)

  # Determines what kind of partitoned call is used for this function.
  is_stateful: bool = False


# Maps the (scope_id, name) in runtime to associated AtomicFunctions.
RUNTIME_FUNCTION_REFS = {}


class AtomicFunction:
  """A Python callable for functions in the TF Runtime.

  Provides core functionality for tf.function including:
    - automatic lifecycle management of runtime functions
    - structured inputs (including captures) and structured outputs
    - calls from both eager and graph mode
    - dependency tracking of children functions
    - runtime error interpolation to identify user code stack traces
    - compatibility with gradient infrastructure
    - control dependencies (including automatic)
  """

  __slots__ = [
      "_name",
      "_bound_context",
      "_function_type",
      "_children",
      "_call_options",
      "_cached_definition",
      "_cached_graph",
      "_generated_graph",
  ]

  def __init__(
      self,
      name,
      bound_context,
      function_type,
      children=None,
      call_options=CallOptions(),
      cached_graph=None,
  ):
    """Construct a new AtomicFunction.

    Args:
      name: str/bytes name of the runtime function in the bound context.
      bound_context: interface to the runtime for the AtomicFunction.
      function_type: input/output contract for the AtomicFunction
      children: list of AtomicFunctions that are needed to call this one.
      call_options: extra configuration options for the call.
      cached_graph: FuncGraph that this AtomicFunction was generated from (if
        known). Otherwise it will lazily construct a new corresponding FuncGraph
        if ever needed.
    """
    self._name = compat.as_bytes(name)
    self._bound_context = bound_context
    self._function_type = function_type
    self._children = children if children else []
    self._call_options = call_options
    self._cached_definition = None

    self._cached_graph = cached_graph
    self._generated_graph = None

    ref_key = (self._bound_context.function_scope_id, self.name)
    if ref_key not in RUNTIME_FUNCTION_REFS:
      RUNTIME_FUNCTION_REFS[ref_key] = 1
    else:
      RUNTIME_FUNCTION_REFS[ref_key] += 1

  @property
  def name(self):
    """Name represented in UTF-8 encoded bytes."""
    return self._name

  @property
  def function_type(self):
    """Represents the input/output contract of this function."""
    return self._function_type

  @property
  def children(self):
    """AtomicFunctions needed as dependencies for this one."""
    return self._children

  @property
  def definition(self):
    """Current FunctionDef in the Runtime."""
    return self._bound_context.get_function_def(self.name)

  @property
  def graph_debug_info(self):
    """A GraphDebugInfo proto mapping nodes to corresponding stack traces."""
    return self._bound_context.get_graph_debug_info(self.name)

  @property
  def call_options(self) -> CallOptions:
    """Call options declared for this AtomicFunction."""
    return self._call_options

  @property
  def graph_call_attrs(self):
    """Returns a dictionary of attributes needed to add a call in graph."""
    attrs = {
        "is_stateful": self.call_options.is_stateful,
        "tout": [
            o.dtype.as_datatype_enum for o in self.function_type.flat_outputs
        ],
        "xla_compile_attr": self.cached_definition.attr.get(
            attributes_lib.XLA_COMPILE, None
        ),
    }
    attrs.update(self._bound_context.function_call_options.as_attrs())
    return attrs

  @property
  def _c_func(self):
    """Returns a scoped pybind object containing FunctionRecord in runtime."""
    return self._bound_context.get_c_function(self.name)

  # TODO(fmuham): Move caching to dependent code and remove method.
  @property
  def cached_definition(self):
    """Cached FunctionDef (not guaranteed to be fresh)."""
    if self._cached_definition is None:
      self._cached_definition = self.definition

    return self._cached_definition

  @property
  def graph(self):
    """Returns a FuncGraph corresponding to the AtomicFunction."""
    if self._cached_graph:
      return self._cached_graph

    # Lazily generate the graph if one is not specified.
    if not self._generated_graph:
      self._generated_graph = to_func_graph(self)

    return self._generated_graph

  def __call__(self, *args):
    """Calls with flat tensor inputs and returns flat tensor outputs.

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

    with InterpolateRuntimeError(self):
      with ops.control_dependencies(self._call_options.control_captures):
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
                len(self.function_type.flat_outputs),
            )
          else:
            outputs = make_call_op_in_graph(
                self,
                list(args),
                self._bound_context.function_call_options.as_attrs(),
            )

    for i, output_type in enumerate(self.function_type.flat_outputs):
      handle_data = output_type.dtype._handle_data
      if handle_data:
        handle_data_util.set_handle_data(outputs[i], handle_data)

    # TODO(fmuham): Use FunctionType cast here for all cases.
    if not self._bound_context.executing_eagerly():
      for i, output_type in enumerate(self.function_type.flat_outputs):
        outputs[i].set_shape(output_type.shape)

    return outputs

  def __del__(self):
    if self._generated_graph:
      func_graph_module.dismantle_func_graph(self._generated_graph)

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
  ops.set_int_list_attr(
      op, acd.READ_ONLY_RESOURCE_INPUTS_ATTR, read_only_indices
  )


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
          type=[x.dtype.as_datatype_enum for x in args]
      )
  )
  tout_attr = attr_value_pb2.AttrValue(
      list=attr_value_pb2.AttrValue.ListValue(type=tout)
  )
  func_attr = attr_value_pb2.AttrValue(
      func=attr_value_pb2.NameAttrList(name=name)
  )
  executor_type_attr = attr_value_pb2.AttrValue(
      s=compat.as_bytes(executor_type)
  )

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


def make_call_op_in_graph(atomic, tensor_inputs, context_call_attrs):
  """Adds an AtomicFunction to graph."""
  graph = ops.get_default_graph()
  graph._add_function_recursive(atomic)  # pylint: disable=protected-access

  op = partitioned_call_op(
      name=atomic.name,
      args=tensor_inputs,
      is_stateful=atomic.call_options.is_stateful,
      tout=[
          o.dtype.as_datatype_enum for o in atomic.function_type.flat_outputs
      ],
      config=context_call_attrs["config_proto"],
      executor_type=context_call_attrs["executor_type"],
      xla_compile_attr=atomic.cached_definition.attr.get(
          attributes_lib.XLA_COMPILE, None
      ),
  )
  _set_read_only_resource_inputs_attr(op, atomic.graph)

  ops.set_int_list_attr(
      op,
      acd.COLLECTIVE_MANAGER_IDS,
      atomic._call_options.collective_manager_ids_used,  # pylint: disable=protected-access
  )

  return op.outputs if op.outputs else op


def from_func_graph(name, graph, inputs, outputs, attrs, overwrite=False):
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

  call_options = CallOptions(
      collective_manager_ids_used=getattr(
          graph, "collective_manager_ids_used", []
      ),
      control_captures=graph.function_captures.control,
      is_stateful=any(op._is_stateful for op in operations),  # pylint: disable=protected-access
  )

  # TODO(fmuham): Include structure info from structured_inputs
  input_signature = (
      tuple(trace_type.from_value(i) for i in inputs),
      {},
  )

  # TODO(fmuham): Include output structure info from structured_outputs
  output_signature = tuple(trace_type.from_value(o) for o in outputs)

  function_type = function_type_lib.from_structured_signature(
      input_signature,
      output_signature,
      graph.function_captures.capture_types,
  )

  return AtomicFunction(
      name,
      bound_context,
      function_type,
      list(graph._functions.values()),  # pylint: disable=protected-access,
      call_options,
      cached_graph=graph,
  )


def to_func_graph(atomic):
  """Generate a FuncGraph from an AtomicFunction."""
  # pylint: disable=protected-access
  input_signature, output_signature = function_type_lib.to_structured_signature(
      atomic.function_type
  )

  with ops.Graph().as_default():
    # Insert dependencies in the default graph so the new graph can pull them.
    for f in atomic.children:
      ops.get_default_graph()._add_function(f)

    result = function_def_to_graph.function_def_to_graph(
        atomic.definition,
        structured_input_signature=input_signature,
        structured_outputs=output_signature,
        propagate_device_spec=True,
        include_library_functions=False,
    )
    for f in atomic.children:
      result._add_function(f)

  # Set input shapes and handle data
  for i, input_type in enumerate(atomic.function_type.flat_inputs):
    handle_data = input_type.dtype._handle_data
    if handle_data:
      handle_data_util.set_handle_data(result.inputs[i], handle_data)
    result.inputs[i].set_shape(input_type.shape)

  # Set output shapes and handle data
  for i, output_type in enumerate(atomic.function_type.flat_outputs):
    handle_data = output_type.dtype._handle_data
    if handle_data:
      handle_data_util.set_handle_data(result.outputs[i], handle_data)
    result.outputs[i].set_shape(output_type.shape)

  result.collective_manager_ids_used = (
      atomic.call_options.collective_manager_ids_used,
  )

  # pylint: enable=protected-access
  return result


class InterpolateRuntimeError(object):
  """Context Manager that interpolates exceptions received by AtomicFunction."""

  DENY_LIST_PHRASES = ["<embedded"]

  def __init__(self, top_level_func):
    self._func = top_level_func

  def interpolate(self, message, node_names, graph_debug_info):
    """Uses the GraphDebugInfo to generate an error message."""
    error_message = ["Graph execution error:", ""]
    for node_name in node_names:
      error_message.append(
          f"Detected at node {node_name} defined at (most recent call last):"
      )
      if node_name in graph_debug_info.traces:
        stack_trace = graph_debug_info.traces[node_name]
        tb_frames = []
        for frame in stack_trace.file_line_cols:
          tb_frames.append(
              traceback.FrameSummary(
                  graph_debug_info.files[frame.file_index],
                  frame.line,
                  frame.func,
              )
          )
          for formatted_frame in traceback.format_list(tb_frames):
            if not any(p in formatted_frame for p in self.DENY_LIST_PHRASES):
              error_message.append(formatted_frame)
      else:
        error_message.append("<stack traces unavailable>")

    error_message.append(message.strip())
    return "\n".join(error_message)

  def __enter__(self):
    pass

  def __exit__(self, typ, exc, tb):
    if not exc or not isinstance(exc, errors.OpError):
      return False
    message = compat.as_text(exc.message)
    parsed_message, func_tags, node_tags = error_interpolation.parse_message(
        message
    )
    deepest_func = None
    for func_tag in func_tags:
      if func_tag.name == compat.as_str(self._func.name):
        deepest_func = self._func
      elif deepest_func:
        next_func = None
        for child_func in deepest_func.children:
          if func_tag.name == compat.as_str(child_func.name):
            next_func = child_func
            break
        if next_func is not None and isinstance(next_func, AtomicFunction):
          deepest_func = next_func
    if deepest_func:
      exc._message = self.interpolate(
          parsed_message,
          [t.name for t in node_tags],
          deepest_func.graph_debug_info,
      )
    return False
