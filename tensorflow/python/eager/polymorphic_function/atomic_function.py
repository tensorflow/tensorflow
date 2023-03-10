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

from tensorflow.core.framework import function_pb2
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.eager import context
from tensorflow.python.eager import execute
from tensorflow.python.eager import tape
from tensorflow.python.framework import c_api_util
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
        if next_func is not None and isinstance(next_func,
                                                EagerDefinedFunction):
          g = next_func.graph
    if g:
      exc._message = error_interpolation.interpolate(message, g)  # pylint: disable=protected-access
    return False


class _EagerDefinedFunctionDeleter(object):
  """Unregister function from eager context."""

  __slots__ = ["name"]

  def __init__(self, name):
    self.name = name

  def __del__(self):
    try:
      context.remove_function(self.name)
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


# TODO(b/232961485): Remove after quarantined `add_function_callback` removed.
function_callbacks = set()


# TODO(apassos) get rid of this by splitting framework.function._DefinedFunction
# so it doesn't have the definition-generating logic and is just a container for
# an already-defined function.
class EagerDefinedFunction(object):
  """Callable with the interface of `framework.function._DefinedFunction`.

  `_EagerDefinedFunction` encapsulates a function definition and its properties,
  and it provides a method for calling the encapsulated function. Some Ops
  take functions as attributes, which have type `func`; an instance of this
  class may be provided as the value of these `func` attributes.
  """

  def __init__(self, name, graph, inputs, outputs, attrs):
    """Initializes an eager defined function.

    Args:
      name: str, the name for the created function.
      graph: Graph, the graph containing the operations in the function
      inputs: the tensors in the graph to be used as inputs to the function
      outputs: the tensors in the graph which will be outputs from the function
      attrs: dict mapping names of attributes to their AttrValue values
    """
    for function_callback in function_callbacks:
      function_callback(self, name, graph, tuple(inputs), tuple(outputs))

    input_ops = set(arg.op for arg in inputs)
    operations = [op for op in graph.get_operations() if op not in input_ops]

    graph_output_names = graph._output_names  # pylint: disable=protected-access
    if (graph_output_names is not None and
        all(ops.tensor_id(t) in graph_output_names for t in outputs)):
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
          compat.as_str(""))

    self._c_func = c_api_util.ScopedTFFunction(fn, name)

    for name, attr_value in attrs.items():
      serialized = attr_value.SerializeToString()
      # TODO(iga): this creates and deletes a new TF_Status for every attr.
      # It might be worth creating a convenient way to re-use status.
      pywrap_tf_session.TF_FunctionSetAttrValueProto(fn, compat.as_str(name),
                                                     serialized)

    # NOTE(feyu): Do not cache signature and definition at initialization to
    # save memory usage of concrete functions never called through Python. We
    # cache them on the first call of .definition and .signature.
    signature = self._get_definition().signature

    self._name = compat.as_bytes(signature.name)
    with ops.init_scope():
      if context.executing_eagerly():
        context.ensure_initialized()
        context.add_function(fn)
        self._function_deleter = _EagerDefinedFunctionDeleter(self.name)
        self._registered_on_context = True

    self._num_outputs = len(signature.output_arg)
    self._output_types = [o.type for o in signature.output_arg]
    self._output_shapes = [o.shape for o in outputs]
    self._control_captures = graph._function_captures.control  # pylint: disable=protected-access
    # Shallow copy outputs since ConcreteFunction may mutate it.
    self._func_graph_outputs = list(outputs)
    self.grad_func_name = None
    self.python_grad_func = None
    self._grad_func = None
    self.graph = graph
    self._stateful_ops = tuple(op for op in operations if op._is_stateful)  # pylint: disable=protected-access

  @property
  def signature(self):
    try:
      return self._signature
    except AttributeError:
      self._signature = self.definition.signature
    return self._signature

  @property
  def definition(self):
    try:
      return self._definition
    except AttributeError:
      self._definition = self._get_definition()
    return self._definition

  def _get_definition(self):
    # TODO(apassos) avoid creating a FunctionDef (specially to grab the
    # signature, but also in general it's nice not to depend on it.
    with c_api_util.tf_buffer() as buffer_:
      with self._c_func.get() as func:
        pywrap_tf_session.TF_FunctionToFunctionDef(func, buffer_)
      proto_data = pywrap_tf_session.TF_GetBuffer(buffer_)
    function_def = function_pb2.FunctionDef()
    function_def.ParseFromString(compat.as_bytes(proto_data))
    return function_def

  def add_to_graph(self, g=None, overwrite=False):
    """Add the function to the current context or a graph, if supplied.

    Args:
      g: the graph to add the function to. If not supplied, the function will
        be added to the current context.
      overwrite: A bool. If True, this function will overwrite any existing
        function of the same signature name in the graph `g` or context.
    """
    # pylint: disable=protected-access
    if not g and context.executing_eagerly():
      ctx = context.context()
      if ctx.has_function(self.name):
        if overwrite:
          ctx.remove_function(self.name)
          ctx.add_function_def(self.definition)
      else:
        ctx.add_function_def(self.definition)
    else:
      if g._is_function(self.name):
        if overwrite:
          g._remove_function(self.name)
          g._add_function(self)
      else:
        g._add_function(self)

      for f in self.graph._functions.values():
        if g._is_function(f.name):
          if overwrite:
            g._remove_function(f.name)
            g._add_function(f)
        else:
          g._add_function(f)
    # pylint: enable=protected-access

  @property
  def name(self):
    return self._name

  @property
  def stateful_ops(self):
    return self._stateful_ops

  def call(self, ctx, args, cancellation_manager=None):
    """Calls this function with `args` as inputs.

    `ConcreteFunction` execution respects device annotations only if the
    function won't be compiled with xla.

    Args:
      ctx: a Context object
      args: a list of arguments to supply this function with.
      cancellation_manager: a `CancellationManager` object that can be used to
        cancel function execution.

    Returns:
      The outputs of the function call.

    Raises:
      ValueError: if the number of arguments is incorrect.
      FunctionAlreadyGarbageCollectedError: if the function is no longer
        available to be called because it has been garbage collected.
    """
    if len(args) != len(self.signature.input_arg):
      raise ValueError(
          f"Signature specifies {len(list(self.signature.input_arg))} "
          f"arguments, got: {len(args)}.")

    function_call_options = ctx.function_call_options
    if function_call_options.config_proto_serialized is None:
      config = function_utils.get_disabled_rewriter_config()
    else:
      config = function_call_options.config_proto_serialized
    executor_type = function_call_options.executor_type or ""

    executing_eagerly = ctx.executing_eagerly()
    attrs = ("executor_type", executor_type, "config_proto", config)
    if executing_eagerly:
      with _InterpolateFunctionError(self):
        if cancellation_manager is None:
          outputs = execute.execute(
              str(self.signature.name),
              num_outputs=self._num_outputs,
              inputs=args,
              attrs=attrs,
              ctx=ctx)
        else:
          outputs = execute.execute_with_cancellation(
              str(self.signature.name),
              num_outputs=self._num_outputs,
              inputs=args,
              attrs=attrs,
              ctx=ctx,
              cancellation_manager=cancellation_manager)
      # Replace empty list with None
      outputs = outputs or None
    else:
      # TODO(akshayka): Either remove this if the FunctionLibraryRuntime
      # creates `PartitionedCallOp` kernels by default, or remove the previous
      # branch if a TPU kernel is registered for `PartitionedCall`.
      with _InterpolateFunctionError(self):
        with ops.control_dependencies(self._control_captures):
          # The caller must use record_operation to record this operation in the
          # eager case, so we enforce the same requirement for the non-eager
          # case by explicitly pausing recording. We don't have a gradient
          # registered for PartitionedCall, so recording this operation confuses
          # forwardprop code (GradientTape manages to ignore it).
          with tape.stop_recording():
            outputs = functional_ops.partitioned_call(
                args=args,
                f=self,
                tout=self._output_types,
                executing_eagerly=executing_eagerly,
                config=config,
                executor_type=executor_type)

    for i, func_graph_output in enumerate(self._func_graph_outputs):
      handle_data_util.copy_handle_data(func_graph_output, outputs[i])
    if executing_eagerly:
      return outputs
    else:
      # TODO(b/128924522): This additional set_shape should not be
      # necessary. ShapeRefiner likely needs to inspect handle_data. Remove this
      # once that's done.
      for i, shape in enumerate(self._output_shapes):
        outputs[i].set_shape(shape)
      return outputs

