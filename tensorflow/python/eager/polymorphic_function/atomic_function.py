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

from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.eager import cancellation
from tensorflow.python.eager import context
from tensorflow.python.eager import execute
from tensorflow.python.eager import tape
from tensorflow.python.eager.polymorphic_function import attributes as attributes_lib
from tensorflow.python.framework import auto_control_deps_utils as acd
from tensorflow.python.framework import error_interpolation
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import functional_ops
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
  attrs: Any
  graph: Any
  stateful_ops: Any

# Maps the name in runtime to the number of associated AtomicFunctions.
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
      "_graph_artifacts",
      "_cached_definition",
  ]

  def __init__(self, name, bound_context, graph_artifacts):
    self._name = compat.as_bytes(name)
    self._bound_context = bound_context
    self._graph_artifacts = graph_artifacts
    self._cached_definition = None

    if self.name not in RUNTIME_FUNCTION_REFS:
      RUNTIME_FUNCTION_REFS[self.name] = 1
    else:
      RUNTIME_FUNCTION_REFS[self.name] += 1

  @property
  def _c_func(self):
    return context.get_c_function(self.name)

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
    return self._name

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

    function_call_options = self._bound_context.function_call_options
    if function_call_options.config_proto_serialized is None:
      config = function_utils.get_disabled_rewriter_config()
    else:
      config = function_call_options.config_proto_serialized
    executor_type = function_call_options.executor_type or ""

    executing_eagerly = self._bound_context.executing_eagerly()
    attrs = ("executor_type", executor_type, "config_proto", config)
    if executing_eagerly:
      with _InterpolateFunctionError(self):
        if cancellation.context() is None:
          outputs = execute.execute(
              str(self.cached_definition.signature.name),
              num_outputs=self._graph_artifacts.num_outputs,
              inputs=list(args),
              attrs=attrs,
              ctx=self._bound_context,
          )
        else:
          outputs = execute.execute_with_cancellation(
              str(self.cached_definition.signature.name),
              num_outputs=self._graph_artifacts.num_outputs,
              inputs=list(args),
              attrs=attrs,
              ctx=self._bound_context,
              cancellation_manager=cancellation.context(),
          )
      # Replace empty list with None
      outputs = outputs or None
    else:
      with _InterpolateFunctionError(self):
        with ops.control_dependencies(self._graph_artifacts.control_captures):
          # The caller must use record_operation to record this operation in the
          # eager case, so we enforce the same requirement for the non-eager
          # case by explicitly pausing recording. We don't have a gradient
          # registered for PartitionedCall, so recording this operation confuses
          # forwardprop code (GradientTape manages to ignore it).
          with tape.stop_recording():
            graph = ops.get_default_graph()
            graph._add_function_recursive(self)  # pylint: disable=protected-access

            op = functional_ops.partitioned_call_op(
                name=self.name,
                args=list(args),
                is_stateful=len(self.stateful_ops) > 0,  # pylint: disable=g-explicit-length-test
                tout=self._graph_artifacts.output_types,
                config=config,
                executor_type=executor_type,
                xla_compile_attr=self.cached_definition.attr.get(
                    attributes_lib.XLA_COMPILE, None
                ),
            )
            _set_read_only_resource_inputs_attr(op, self.graph)
            if hasattr(self.graph, "collective_manager_ids_used"):
              ops.set_int_list_attr(
                  op,
                  acd.COLLECTIVE_MANAGER_IDS,
                  self.graph.collective_manager_ids_used,
              )
            outputs = op.outputs if op.outputs else op

    for i, func_graph_output in enumerate(
        self._graph_artifacts.func_graph_outputs
    ):
      handle_data_util.copy_handle_data(func_graph_output, outputs[i])
    if executing_eagerly:
      return outputs
    else:
      # TODO(b/128924522): This additional set_shape should not be
      # necessary. ShapeRefiner likely needs to inspect handle_data. Remove this
      # once that's done.
      for i, shape in enumerate(self._graph_artifacts.output_shapes):
        outputs[i].set_shape(shape)
      return outputs

  def __del__(self):
    RUNTIME_FUNCTION_REFS[self.name] -= 1
    if RUNTIME_FUNCTION_REFS[self.name] < 0:
      raise RuntimeError(
          f"AtomicFunction Refcounting for {self.name} is invalid."
      )

    if RUNTIME_FUNCTION_REFS[self.name] == 0:
      try:
        self._bound_context.remove_function(self.name)
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
      control_captures=graph._function_captures.control,  # pylint: disable=protected-access
      func_graph_outputs=list(outputs),
      attrs=attrs,
      graph=graph,
      stateful_ops=tuple(op for op in operations if op._is_stateful),  # pylint: disable=protected-access
  )

  return AtomicFunction(name, bound_context, graph_artifacts)
