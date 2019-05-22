# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""FuncGraph and related functionality."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections as py_collections
import itertools
import weakref

from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager import execute
from tensorflow.python.eager import tape
from tensorflow.python.eager.graph_only_ops import graph_placeholder
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework.auto_control_deps import AutomaticControlDependencies
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import compat
from tensorflow.python.util import memory
from tensorflow.python.util import nest
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_decorator
from tensorflow.python.util.lazy_loader import LazyLoader

# This is to avoid a circular dependency:
# function -> func_graph
function = LazyLoader("function", globals(),
                      "tensorflow.python.eager.function")
def_function = LazyLoader(
    "def_function", globals(),
    "tensorflow.python.eager.def_function")

WHITELIST_COLLECTIONS = [
    ops.GraphKeys.GLOBAL_VARIABLES,
    ops.GraphKeys.LOCAL_VARIABLES,
    ops.GraphKeys.TRAINABLE_VARIABLES,
    variable_scope._VARSTORE_KEY,  # pylint: disable=protected-access
    variable_scope._VARSCOPESTORE_KEY  # pylint: disable=protected-access
]


class UnknownArgument(object):
  """Signifies an argument which is not currently handled."""
  pass


def convert_structure_to_signature(structure, arg_names=None):
  """Convert a potentially nested structure to a signature.

  Args:
    structure: Structure to convert, where top level collection is a list or a
      tuple.
    arg_names: Optional list of arguments that has equal number of elements as
      `structure` and is used for naming corresponding TensorSpecs.

  Returns:
    Identical structure that has TensorSpec objects instead of Tensors and
    UknownArgument instead of any unsupported types.
  """
  def encode_arg(arg, path):
    """A representation for this argument, for converting into signatures."""
    if isinstance(arg, ops.Tensor):
      user_specified_name = None
      try:
        user_specified_name = compat.as_str(
            arg.op.get_attr("_user_specified_name"))
      except ValueError:
        pass

      if path and user_specified_name and user_specified_name != path[0]:
        # The user has explicitly named the argument differently than the name
        # of the function argument.
        name = user_specified_name
      else:
        name = "/".join([str(p) for p in path])
      return tensor_spec.TensorSpec(arg.shape, arg.dtype, name)
    if isinstance(arg, (
        int,
        float,
        bool,
        type(None),
        dtypes.DType,
        tensor_spec.TensorSpec,
    )):
      return arg
    return UnknownArgument()

  # We are using the flattened paths to name the TensorSpecs. We need an
  # explicit name for them downstream.
  flattened = nest.flatten_with_tuple_paths(structure, expand_composites=True)
  if arg_names:
    if len(arg_names) != len(structure):
      raise ValueError(
          "Passed in arg_names don't match actual signature (%s)." % arg_names)
    # Replace all top-level names with their actual arg_names. If a path before
    # was "(2,'a',1)", it will become "(arg_names[2],'a',1)".
    flattened = [
        ((arg_names[path[0]],) + path[1:], arg) for path, arg in flattened
    ]

  mapped = [encode_arg(arg, path) for path, arg in flattened]
  return nest.pack_sequence_as(structure, mapped, expand_composites=True)


class FuncGraph(ops.Graph):
  """Graph representing a function body.

  Attributes:
    name: The name of the function.
    inputs: Placeholder tensors representing the inputs to this function. The
      tensors are in this FuncGraph. This represents "regular" inputs as well as
      captured inputs (i.e. the values of self.captures), with the regular
      inputs coming first.
    outputs: Tensors that will be returned by this function. The tensors are in
      this FuncGraph.
    control_outputs: Operations that must be executed before the function
      represented by this graph can be said to have been executed.
    structured_input_signature: A tuple of (args, kwargs), which are both
      possibly-nested python objects that were received by this function. Note
      that these structures might contain Python `None`s.
    structured_outputs: A possibly-nested python object which will be returned
      by this function. The Tensors in this structure are the same as those of
      self.outputs. Note that this structure might contain Python `None`s.
    variables: Variables that should be watched during function execution.
    outer_graph: The graph this function is defined in. May be another FuncGraph
      or the global default Graph.
    captures: Maps external tensor -> internal tensor (i.e. input placeholder).
      The entries are in the order they were captured.
    control_captures: Set of external ops on which this graph has a control
      dependency.
    seed: The graph-level random seed.
    capture_by_value: If True, the func graph will capture Variables by value
      instead of reference.
  """

  def __init__(self, name, collections=None, capture_by_value=None):
    """Construct a new FuncGraph.

    The graph will inherit its graph key, collections, seed, and distribution
    strategy stack from the current context or graph.

    Args:
      name: the name of the function.
      collections: a dictionary of collections this FuncGraph should start
        with. If not specified (None), the FuncGraph will read (but not write
        to) the outer graph's collections that are not whitelisted, and both
        read and write to the outer graph's collections that are whitelisted.
        The current whitelisted collections are the global variables, the
        local variables, and the trainable variables.
        Defaults to None.
      capture_by_value: An optional boolean. If True, the func graph will
        capture Variables by value instead of reference. By default inherit
        from outer graphs, and failing that will default to False.
    """
    super(FuncGraph, self).__init__()

    self.name = name
    self.inputs = []
    self.outputs = []
    self.control_outputs = []
    self.control_captures = set()
    self.structured_input_signature = None
    self.structured_outputs = None
    self._weak_variables = []
    self._watched_variables = weakref.WeakSet()
    self.outer_graph = ops.get_default_graph()
    self.captures = py_collections.OrderedDict()
    # Inherit capture-by-value from outer graph.
    if capture_by_value is not None:
      self.capture_by_value = capture_by_value
    elif self.outer_graph is not None and isinstance(
        self.outer_graph, FuncGraph):
      self.capture_by_value = self.outer_graph.capture_by_value
    else:
      self.capture_by_value = False

    self._building_function = True
    # Map from resource tensor name to last op (in program order) which uses
    # this tensor. Used to enforce that execution order matches program order
    # for resource tensors.
    self._last_op_using_resource_tensor = {}

    graph = self.outer_graph

    if context.executing_eagerly():
      self.seed = context.global_seed()
      # [for tf-data user migration from TF1.0 to 2.0] seed_used keep track of
      # any None op_seed for random_op in the function, in which case we end up
      # using function seed, which could be unintended behavior for the op.
      self._seed_used = False
    else:
      self.seed = graph.seed
      self._seed_used = False
      # TODO(allenl): Figure out if we can remove colocation stack
      # specialization (currently used in cond_v2), here and in the cache key.
      self._colocation_stack = graph._colocation_stack.copy()  # pylint: disable=protected-access

    if collections is None:
      for collection_name in graph.get_all_collection_keys():
        if collection_name not in WHITELIST_COLLECTIONS:
          self._collections[collection_name] = graph.get_collection(
              collection_name)
      for collection_name in WHITELIST_COLLECTIONS:
        self._collections[collection_name] = graph.get_collection_ref(
            collection_name)
    else:
      self._collections = collections

  def __str__(self):
    return "FuncGraph(name=%s, id=%s)" % (self.name, id(self))

  def watch_variable(self, v):
    """Marks the variable v as accessed while building this graph."""
    while self is not None and isinstance(self, FuncGraph):
      self._watched_variables.add(v)
      self = self.outer_graph

  def control_dependencies(self, control_inputs):
    """Handles control dependencies.

    FuncGraph wraps Graph's control_dependencies logic by first filtering out
    any external tensors / operations and storing them in the graph's
    control_captures member. Any consumers of this function graph must then
    decide how to handle the control captures.

    Args:
      control_inputs: A list of `Operation` or `Tensor` objects which
        must be executed or computed before running the operations
        defined in the context.  Can also be `None` to clear the control
        dependencies.

    Returns:
     A context manager that specifies control dependencies for all
     operations constructed within the context.

    Raises:
      TypeError: If `control_inputs` is not a list of `Operation` or
        `Tensor` objects.
    """
    if control_inputs is None:
      return super(FuncGraph, self).control_dependencies(control_inputs)

    filtered_control_inputs = []
    for c in control_inputs:
      # Check for _UnreadVariable
      if (isinstance(c, ops.IndexedSlices) or
          (hasattr(c, "_handle") and hasattr(c, "op"))):
        c = c.op
      graph_element = ops._as_graph_element(c)  # pylint: disable=protected-access
      if graph_element is None:
        graph_element = c
      if graph_element is not None and getattr(
          graph_element, "graph", None) is not self:
        self.control_captures.add(graph_element)
      else:
        filtered_control_inputs.append(graph_element)
    return super(FuncGraph, self).control_dependencies(filtered_control_inputs)

  def as_default(self):
    outer_cm = super(FuncGraph, self).as_default()

    @tf_contextlib.contextmanager
    def inner_cm():
      """Context manager for copying distribute.Strategy scope information."""
      graph = ops.get_default_graph()
      # pylint: disable=protected-access
      # TODO(b/112906995, nareshmodi): distribution strategy depends on
      # inheriting this stack from the default graph even in eager mode. Maybe
      # it should be part of the eager context? This would also allow us to
      # remove a get_default_graph() call from the function cache lookup.
      old_strategy_stack = self._distribution_strategy_stack
      self._distribution_strategy_stack = list(
          graph._distribution_strategy_stack)
      # We ignore device placements from any outer scopes while tracing the
      # function when possible, to avoid hard-coding them in the function
      # graph. "Default" placements come from the PartitionedCallOp's placement,
      # so that the same trace of the Python function may be placed on several
      # different devices and saved functions may be placed on new devices when
      # restored.
      old_device_stack = self._device_function_stack
      if context.executing_eagerly():
        if self._distribution_strategy_stack:
          self._add_device_to_stack(context.context().device_name)
      else:
        if (self._distribution_strategy_stack
            or device_stack_has_callable(graph._device_function_stack)):
          # Hard-code devices from device functions in the function body
          self._device_function_stack = graph._device_function_stack.copy()

      old_creator_stack = self._variable_creator_stack
      self._variable_creator_stack = graph._variable_creator_stack
      # Inherit the graph key, since this is used for matching variables in
      # optimizers.
      old_graph_key = self._graph_key
      self._graph_key = graph._graph_key
      # pylint: enable=protected-access

      with outer_cm as g:
        try:
          yield g
        finally:
          self._distribution_strategy_stack = old_strategy_stack
          self._device_function_stack = old_device_stack
          self._variable_creator_stack = old_creator_stack
          self._graph_key = old_graph_key
    return inner_cm()

  @property
  def output_types(self):
    return [t.dtype for t in self.outputs]

  @property
  def output_shapes(self):
    return [t.shape for t in self.outputs]

  @property
  def variables(self):
    """A list of variables accessed by this FuncGraph.

    Note that functions keep only weak references to variables. Calling the
    function after a variable it accesses has been deleted is an error.

    Yields:
      Strong references to variables accessed by this FuncGraph.
    """
    for weak_v in self._weak_variables:
      v = weak_v()
      if v is None:
        raise AssertionError(
            "Called a function referencing variables which have been deleted. "
            "This likely means that function-local variables were created and "
            "not referenced elsewhere in the program. This is generally a "
            "mistake; consider storing variables in an object attribute on "
            "first call.")
      yield v

  @variables.setter
  def variables(self, var_list):
    self._weak_variables = [weakref.ref(v) for v in var_list]

  def _capture_by_value(
      self,
      op_type,
      inputs,
      dtypes,  # pylint: disable=redefined-outer-name
      input_types=None,
      name=None,
      attrs=None,
      op_def=None,
      compute_device=True):
    # When capturing by value, do the read outside
    reverse_captures = dict((v, k) for k, v in self.captures.items())
    uncaptured_inputs = [reverse_captures.get(t, t) for t in inputs]
    with ops.init_scope():
      if context.executing_eagerly():
        attr_list = ("dtype", int(attrs["dtype"].type))
        value, = execute.execute(
            compat.as_bytes(op_type), 1, uncaptured_inputs, attr_list,
            context.context())
      else:
        op = ops.get_default_graph().create_op(
            op_type,
            uncaptured_inputs,
            dtypes,
            input_types,
            name,
            attrs,
            op_def,
            compute_device=compute_device)
        value = op.outputs[0]
    captured_value = self.capture(value)
    return captured_value.op

  def create_op(
      self,
      op_type,
      inputs,
      dtypes=None,  # pylint: disable=redefined-outer-name
      input_types=None,
      name=None,
      attrs=None,
      op_def=None,
      compute_shapes=True,
      compute_device=True):
    """Like Graph.create_op, except handles external input tensors.

    This overload adds functionality to create_op to "capture" any external
    input tensors, i.e. tensors from the eager context or outer function graphs
    if this is a nested function. See `capture` for more information.

    Args:
      op_type: The `Operation` type to create. This corresponds to the
        `OpDef.name` field for the proto that defines the operation.
      inputs: A list of `Tensor` objects that will be inputs to the `Operation`.
      dtypes: (Optional) A list of `DType` objects that will be the types of the
        tensors that the operation produces.
      input_types: (Optional.) A list of `DType`s that will be the types of
        the tensors that the operation consumes. By default, uses the base
        `DType` of each input in `inputs`. Operations that expect
        reference-typed inputs must specify `input_types` explicitly.
      name: (Optional.) A string name for the operation. If not specified, a
        name is generated based on `op_type`.
      attrs: (Optional.) A dictionary where the key is the attribute name (a
        string) and the value is the respective `attr` attribute of the
        `NodeDef` proto that will represent the operation (an `AttrValue`
        proto).
      op_def: (Optional.) The `OpDef` proto that describes the `op_type` that
        the operation will have.
      compute_shapes: (Optional.) Deprecated. Has no effect (shapes are always
        computed).
      compute_device: (Optional.) If True, device functions will be executed
        to compute the device property of the Operation.

    Returns:
      An `Operation` object.
    """
    del compute_shapes
    if self.capture_by_value and op_type in ["ReadVariableOp",
                                             "ResourceGather"]:
      return self._capture_by_value(op_type, inputs, dtypes, input_types, name,
                                    attrs, op_def, compute_device)

    # This capturing logic interacts poorly with control flow contexts which
    # want to replace inputs of ops far too late in the process. This can lead
    # the context to get confused and try to create an Enter for an Enter. We
    # can detect this here and skip the additional Enter which can confuse loop
    # validation logic.
    if op_type == "Enter" and inputs[0].op.type == "Enter":
      if inputs[0].op.get_attr("frame_name") == attrs["frame_name"].s:
        return inputs[0].op
    # Calling AddValue on the control flow contexts to force creation of the
    # backward accumulators in the original graph before we create placeholders
    # to capture the inputs.
    ctxt = ops.get_default_graph()._control_flow_context  # pylint: disable=protected-access
    for i, inp in enumerate(inputs):
      # TPU Estimator defines a control flow context with no AddValue method.
      if ctxt is not None and hasattr(ctxt, "AddValue"):
        inp = ctxt.AddValue(inp)
      inp = self.capture(inp)
      inputs[i] = inp
    return super(FuncGraph, self).create_op(
        op_type, inputs, dtypes, input_types, name, attrs, op_def,
        compute_device=compute_device)

  def capture(self, tensor, name=None):
    """Captures `tensor` if it's external to this graph.

    If `tensor` is from a different graph, returns a placeholder for it.
    `tensor` and the placeholder will appear in self.captures, and the
    placeholder will appear in self.inputs.  Multiple calls to this method with
    the same `tensor` argument will return the same placeholder. If `tensor` is
    from this graph, returns `tensor`.

    Args:
      tensor: Tensor. May be from this FuncGraph or a different graph.
      name: Optional name if a placeholder is created.

    Returns:
      Tensor from this FuncGraph.
    """
    # Note: _forward_func_graph is currently only set when building the gradient
    # graph graph of a defun call. If the backwards graph tries to capture
    # tensors those will be captured first in the forward graph. This
    # makes sure that any tensor needed by a custom_gradient is correctly
    # captured.
    if (getattr(tensor, "graph", None) is not self and
        hasattr(self, "_forward_func_graph") and
        isinstance(self._forward_func_graph, FuncGraph)):
      tensor = self._forward_func_graph.capture(tensor)
    if isinstance(tensor, ops.EagerTensor):
      if name is None:
        name = str(ops.uid())
      return self._capture_helper(tensor, name)
    if tensor.graph is not self:
      if name is None:
        name = tensor.op.name
      inner_graph = tensor.graph
      while inner_graph is not None and isinstance(inner_graph, FuncGraph):
        if inner_graph is self:
          raise ValueError(
              "Trying to capture a tensor from an inner function. This can be "
              "caused by accessing a tensor defined inside a loop or "
              "conditional body, or a subfunction, from a calling function, "
              "without going through the proper return value mechanism. "
              "Consider using TensorFlow mechanisms such as TensorArrays "
              "to return tensors from inner functions or loop / conditional "
              "bodies. Tensor: %s; tensor graph: %s; this graph: %s"
              % (tensor, tensor.graph, self))
        inner_graph = inner_graph.outer_graph
      return self._capture_helper(tensor, name)
    return tensor

  def _capture_helper(self, tensor, name):
    captured_tensor = self.captures.get(tensor, None)
    if captured_tensor is None:
      captured_tensor = _create_substitute_placeholder(tensor, name=name,
                                                       dtype=tensor.dtype)
      self.captures[tensor] = captured_tensor
      self.inputs.append(captured_tensor)
    tape.record_operation("captured_value", [captured_tensor], [tensor],
                          lambda x: [x])
    return captured_tensor

  @property
  def external_captures(self):
    """External tensors captured by this function."""
    return list(self.captures.keys())

  @property
  def internal_captures(self):
    """Placeholders in this function corresponding captured tensors."""
    return list(self.captures.values())


def func_graph_from_py_func(name,
                            python_func,
                            args,
                            kwargs,
                            signature=None,
                            func_graph=None,
                            autograph=False,
                            autograph_options=None,
                            add_control_dependencies=True,
                            arg_names=None,
                            op_return_value=None,
                            collections=None,
                            capture_by_value=None,
                            override_flat_arg_shapes=None):
  """Returns a `FuncGraph` generated from `python_func`.

  Args:
    name: an identifier for the function.
    python_func: the Python function to trace.
    args: the positional args with which the Python function should be called;
      ignored if a signature is provided.
    kwargs: the keyword args with which the Python function should be called;
      ignored if a signature is provided.
    signature: a possibly nested sequence of `TensorSpecs` specifying the shapes
      and dtypes of the arguments. When a signature is provided, `args` and
      `kwargs` are ignored, and `python_func` is traced with Tensors conforming
      to `signature`. If `None`, the shapes and dtypes are inferred from the
      inputs.
    func_graph: Optional. An instance of FuncGraph. If provided, we will use
      this graph else a new one is built and returned.
    autograph: whether to use autograph to compile `python_func`.
      See https://www.tensorflow.org/guide/autograph for more information.
    autograph_options: additional knobs to control when `autograph=True`.
      See https://www.tensorflow.org/guide/autograph for more information.
    add_control_dependencies: If True, automatically adds control dependencies
      to ensure program order matches execution order and stateful ops always
      execute.
    arg_names: Optional list of argument names, used to give input placeholders
      recognizable names.
    op_return_value: Optional. A Tensor. If set and `python_func` returns
      Operations, those return values will be replaced with this value. If not
      set, returning an Operation triggers an error.
    collections: a dictionary of collections this FuncGraph should start
      with. If not specified (None), the FuncGraph will read (but not write to)
      the outer graph's collections that are not whitelisted, and both
      read and write to the outer graph's collections that are whitelisted.
      The current whitelisted collections are the global variables, the
      local variables, and the trainable variables.
      Defaults to None.
    capture_by_value: An optional boolean. If True, the func graph will capture
      Variables by value instead of reference. By default inherit from outer
      graphs, and failing that will default to False.
    override_flat_arg_shapes: An optional list of instances that are either
      `None` or `TensorShape`.  The length must match that of
      `nest.flatten((args, kwargs), expand_composites=True)`.  The entries
      containing value `None` must match entries in flattened arguments
      containing non-tensors, while entries containing a `TensorShape` must
      match entries in the flattened arguments containing tensors.

  Returns:
    A FuncGraph.

  Raises:
    TypeError: If any of `python_func`'s return values is neither `None` nor a
      `Tensor`.
    ValueError: If both `signature` and `override_flat_arg_shapes` are
      passed in.
  """
  if op_return_value is not None:
    assert isinstance(op_return_value, ops.Tensor), op_return_value
  if func_graph is None:
    func_graph = FuncGraph(name, collections=collections,
                           capture_by_value=capture_by_value)
  assert isinstance(func_graph, FuncGraph)
  if add_control_dependencies:
    control_manager = AutomaticControlDependencies()
  else:
    control_manager = ops.NullContextmanager()
  with func_graph.as_default(), control_manager as a:
    current_scope = variable_scope.get_variable_scope()
    default_use_recource = current_scope.use_resource
    current_scope.set_use_resource(True)

    if signature is not None and override_flat_arg_shapes is not None:
      raise ValueError(
          "Passed both signature and override_flat_arg_shapes: %s and %s."
          % (signature, override_flat_arg_shapes))

    if signature is not None:
      args = signature
      kwargs = {}

    # Creates and names placeholders for all arguments.
    if override_flat_arg_shapes is not None:
      flat_args = nest.flatten(args, expand_composites=True)
      arg_shapes = override_flat_arg_shapes[:len(flat_args)]
      kwarg_shapes = override_flat_arg_shapes[len(flat_args):]
    else:
      arg_shapes = None
      kwarg_shapes = None
    func_args = _get_defun_inputs_from_args(
        args, arg_names, flat_shapes=arg_shapes)
    func_kwargs = _get_defun_inputs_from_kwargs(
        kwargs, flat_shapes=kwarg_shapes)

    # Convert all Tensors into TensorSpecs before saving the structured inputs.
    # If storing pure concrete functions that are not called through polymorphic
    # functions, we don't have access to FunctionSpec, so we need to call the
    # TensorSpecs by their `arg_names` for later binding.
    func_graph.structured_input_signature = (
        convert_structure_to_signature(func_args, arg_names),
        convert_structure_to_signature(func_kwargs))

    flat_func_args = nest.flatten(func_args, expand_composites=True)
    flat_func_kwargs = nest.flatten(func_kwargs, expand_composites=True)
    # Temporarily set inputs to allow graph building code to inspect
    # them. Reassigned below.
    func_graph.inputs = [arg for arg in flat_func_args + flat_func_kwargs
                         if isinstance(arg, ops.Tensor)]

    # Note: `nest.flatten` sorts by keys, as does `_deterministic_dict_values`.
    # Variables to help check whether mutation happens in calling the function
    # Copy the recursive list, tuple and map structure, but not base objects
    func_args_before = nest.pack_sequence_as(func_args, flat_func_args,
                                             expand_composites=True)
    func_kwargs_before = nest.pack_sequence_as(
        func_kwargs, flat_func_kwargs, expand_composites=True)

    def convert(x):
      """Converts a function output to a Tensor."""
      if x is None:
        return None
      if op_return_value is not None and isinstance(x, ops.Operation):
        # TODO(b/79881896): we currently can't capture external control deps, so
        # this won't work if x needs to be captured (i.e. if python_func returns
        # captured Operations).
        with ops.control_dependencies([x]):
          x = array_ops.identity(op_return_value)
      elif not isinstance(x, tensor_array_ops.TensorArray):
        try:
          x = ops.convert_to_tensor_or_composite(x)
        except (ValueError, TypeError):
          raise TypeError(
              "To be compatible with tf.contrib.eager.defun, Python functions "
              "must return zero or more Tensors; in compilation of %s, found "
              "return value of type %s, which is not a Tensor." %
              (str(python_func), type(x)))
      if add_control_dependencies:
        x = a.mark_as_return(x)
      return x

    try:
      if autograph:
        from tensorflow.python import autograph  # pylint: disable=g-import-not-at-top
        _, original_func = tf_decorator.unwrap(python_func)

        def wrapper(*args, **kwargs):
          """Calls a converted version of original_func."""
          # TODO(mdan): Push this block higher in tf.function's call stack.
          try:
            return autograph.converted_call(
                original_func, None,
                autograph.ConversionOptions(
                    recursive=True,
                    optional_features=autograph_options,
                    force_conversion=True,
                ), args, kwargs)
          except Exception as e:  # pylint:disable=broad-except
            if hasattr(e, "ag_error_metadata"):
              raise e.ag_error_metadata.to_exception(type(e))
            else:
              raise

        # Wrapping around a decorator allows checks like tf_inspect.getargspec
        # to be accurate.
        converted_func = tf_decorator.make_decorator(original_func, wrapper)
        python_func = tf_decorator.rewrap(python_func, original_func,
                                          converted_func)

      func_outputs = python_func(*func_args, **func_kwargs)

      # invariant: `func_outputs` contains only Tensors, CompositeTensors,
      # TensorArrays and `None`s.
      func_outputs = nest.map_structure(convert, func_outputs,
                                        expand_composites=True)

      check_mutation(func_args_before, func_args)
      check_mutation(func_kwargs_before, func_kwargs)
    finally:
      current_scope.set_use_resource(default_use_recource)

    # Variables in `func_args`, `func_kwargs` should be explicit inputs
    # to the function, not captured inputs.
    graph_variables = list(func_graph._watched_variables)  # pylint: disable=protected-access
    arg_variables = set()
    inputs = []
    for arg in (nest.flatten(func_args, expand_composites=True) +
                nest.flatten(func_kwargs, expand_composites=True)):
      if isinstance(arg, resource_variable_ops.ResourceVariable):
        # Even if an argument variable was not used in the function, we've
        # already manually captured the resource Tensor when creating argument
        # placeholders.
        resource_placeholder = func_graph.captures.pop(arg.handle, None)
        if resource_placeholder is None:
          continue
        arg_variables.add(arg)
        inputs.append(resource_placeholder)
      elif isinstance(arg, ops.Tensor):
        inputs.append(arg)
    variables = [v for v in graph_variables if v not in arg_variables]
    func_graph.inputs = inputs + list(func_graph.captures.values())

    func_graph.structured_outputs = func_outputs
    # Returning a closed-over tensor does not trigger convert_to_tensor.
    func_graph.outputs.extend(
        func_graph.capture(x)
        for x in flatten(func_graph.structured_outputs)
        if x is not None)

    func_graph.variables = variables

  if add_control_dependencies:
    func_graph.control_outputs.extend(control_manager.ops_which_must_run)

# Register any other functions defined in the graph.
  with ops.init_scope():
    if context.executing_eagerly():
      for f in func_graph._functions.values():  # pylint: disable=protected-access
        # TODO(ashankar): What about the gradient registry?
        context.add_function(f._c_func.func)  # pylint: disable=protected-access

  return func_graph


def maybe_captured(tensor):
  """If t is a captured value placeholder, returns the original captured value.

  Args:
    tensor: Tensor.

  Returns:
    A tensor, potentially from a different Graph/FuncGraph.
  """
  if (not isinstance(tensor, ops.EagerTensor) and
      tensor.op.graph.building_function and tensor.op.type == "Placeholder"):
    for input_t, placeholder_t in tensor.op.graph.captures.items():
      if tensor == placeholder_t:
        return maybe_captured(input_t)
  # pylint: enable=protected-access
  return tensor


def device_stack_has_callable(device_stack):
  """Checks whether a device stack contains a callable."""
  return any(callable(spec._device_name_or_function)  # pylint: disable=protected-access
             for spec in device_stack.peek_objs())


def check_mutation(n1, n2):
  """Check if two list of arguments are exactly the same."""
  errmsg = ("Function to be traced should not modify structure of input "
            "arguments. Check if your function has list and dictionary "
            "operations that alter input arguments, "
            "such as `list.pop`, `list.append`")
  try:
    nest.assert_same_structure(n1, n2, expand_composites=True)
  except ValueError:
    raise ValueError(errmsg)

  for arg1, arg2 in zip(nest.flatten(n1, expand_composites=True),
                        nest.flatten(n2, expand_composites=True)):
    if arg1 is not arg2:
      raise ValueError(errmsg)


# TODO(edloper): If TensorArray becomes a CompositeTensor, then delete this.
def flatten(sequence):
  """Like nest.flatten w/ expand_composites, but returns flow for TensorArrays.

  Args:
    sequence: A nested structure of Tensors, CompositeTensors, and
      TensorArrays.

  Returns:
    A list of tensors.
  """
  flat_sequence = nest.flatten(sequence, expand_composites=True)
  return [
      item.flow if isinstance(item, tensor_array_ops.TensorArray) else item
      for item in flat_sequence]


# TODO(edloper): If TensorArray becomes a CompositeTensor, then delete this.
def pack_sequence_as(structure, flat_sequence):
  """Like `nest.pack_sequence_as` but also builds TensorArrays from flows.

  Args:
    structure: The structure to pack into. May contain Tensors,
      CompositeTensors, or TensorArrays.
    flat_sequence: An iterable containing tensors.

  Returns:
    A nested structure.

  Raises:
    AssertionError if `structure` and `flat_sequence` are not compatible.
  """
  flat_sequence = list(flat_sequence)
  flattened_structure = nest.flatten(structure, expand_composites=True)
  if len(flattened_structure) != len(flat_sequence):
    raise ValueError("Mismatch in element count")
  for i in range(len(flat_sequence)):
    if isinstance(flattened_structure[i], tensor_array_ops.TensorArray):
      flat_sequence[i] = tensor_array_ops.build_ta_with_new_flow(
          old_ta=flattened_structure[i], flow=flat_sequence[i])
  return nest.pack_sequence_as(structure, flat_sequence, expand_composites=True)


def _create_substitute_placeholder(value, name=None, dtype=None):
  """Creates a placeholder for `value` and propagates shape info to it."""
  # Note: setting ops.control_dependencies(None) ensures we always put
  # capturing placeholders outside of any control flow context.
  with ops.control_dependencies(None):
    placeholder = graph_placeholder(
        dtype=dtype or value.dtype, shape=value.shape, name=name)
  custom_gradient.copy_handle_data(value, placeholder)
  return placeholder


def _get_defun_inputs_from_args(args, names, flat_shapes=None):
  """Maps Python function positional args to graph-construction inputs."""
  return _get_defun_inputs(
      args, names, structure=args, flat_shapes=flat_shapes)


def _get_defun_inputs(args, names, structure, flat_shapes=None):
  """Maps python function args to graph-construction inputs.

  Args:
    args: A flat list of user-specified arguments.
    names: A list of strings with user-specified argument names, same length as
      `args`. May be `None`, in which case a generic name is used.
    structure: The original argument list or dictionary.
    flat_shapes: A flat list of values that are either `None` or
      instances of `TensorShape`.  If provided, then length must match
      that of `nest.flatten(args, expand_composites=True)`; and locations where
      `args` are instances of `Tensor` must have a corresponding `TensorShape`
      in `flat_shapes`.  May be `None`, in which case exact shapes are read
      directly from the args.

  Returns:
    Placeholders with the same structure as `structure`.

  Raises:
    RuntimeError: if `flat_shapes` is provided, but
     `len(flat_shapes) != len(nest.flatten(args, expand_composites=True))`.
    RuntimeError: if a shape from `flat_shapes` is not None
     for an argument that is not a `Tensor`, `TensorSpec`,
     or `ResourceVariable`.
  """
  func_graph = ops.get_default_graph()
  function_inputs = []
  if names is None:
    names = [None] * len(args)
  if flat_shapes is None:
    shapes_iter = itertools.repeat(None)
  else:
    len_flat_args = len(nest.flatten(args, expand_composites=True))
    if len_flat_args != len(flat_shapes):
      raise RuntimeError(
          "Length of fully flat shapes (%d) must match that of "
          "flatten(args) (%d).  args: %s, flat_shapes: %s"
          % (len(flat_shapes),
             len_flat_args,
             args,
             flat_shapes))
    shapes_iter = iter(flat_shapes)
  for arg_value, name in zip(args, names):
    flattened = nest.flatten(arg_value, expand_composites=True)
    tensor_specs = [
        arg for arg in flattened if isinstance(arg, tensor_spec.TensorSpec)
    ]
    specified_names = [arg.name for arg in tensor_specs if arg.name]
    if specified_names and len(specified_names) < len(tensor_specs):
      raise ValueError("If specifying TensorSpec names for nested structures, "
                       "either zero or all names have to be specified.")

    for arg in flattened:
      # We have a shape entry for each arg, regadless of whether it's a real
      # Tensor or not.  For non-tensor entries it should be None.
      shape = next(shapes_iter)
      if isinstance(arg, (ops.Tensor, tensor_spec.TensorSpec)):
        if isinstance(arg, tensor_spec.TensorSpec) and arg.name:
          requested_name = arg.name
        else:
          requested_name = name
        placeholder_shape = shape if shape is not None else arg.shape
        try:
          placeholder = graph_placeholder(
              arg.dtype, placeholder_shape,
              name=requested_name)
        except ValueError:
          # Sometimes parameter names are not valid op names, so fall back to
          # unnamed placeholders.
          placeholder = graph_placeholder(arg.dtype, placeholder_shape)
        if name is not None:
          # Record the requested/user-specified name in case it's different than
          # the uniquified name, for validation when exporting signatures.
          placeholder.op._set_attr(  # pylint: disable=protected-access
              "_user_specified_name",
              attr_value_pb2.AttrValue(s=compat.as_bytes(requested_name)))
        function_inputs.append(placeholder)
      elif isinstance(arg, resource_variable_ops.ResourceVariable):
        # Capture arg variables to create placeholders for them. These will be
        # removed as captures after the function is traced (since otherwise we'd
        # just add it back with a new placeholder when the variable was
        # referenced).
        placeholder = func_graph.capture(arg.handle, name=name)
        placeholder.op._set_attr(  # pylint: disable=protected-access
            "_user_specified_name",
            attr_value_pb2.AttrValue(s=compat.as_bytes(name)))
        function_inputs.append(arg)
      else:
        if shape is not None:
          raise RuntimeError(
              "Expected provided shape override to be None for arg that isn't "
              "a Tensor, but saw arg: '%s', shape: '%s'.  args: %s"
              % (arg, shape, args))
        function_inputs.append(arg)
  return nest.pack_sequence_as(structure, function_inputs,
                               expand_composites=True)


def _get_defun_inputs_from_kwargs(kwargs, flat_shapes):
  """Maps Python function keyword args to graph-construction inputs."""
  if kwargs:
    names, args = zip(*sorted(kwargs.items()))
  else:
    names = []
    args = []
  return _get_defun_inputs(
      args, names, structure=kwargs, flat_shapes=flat_shapes)


def dismantle_func_graph(func_graph):
  """Removes reference cycles in `func_graph` FuncGraph.

  Helpful for making sure the garbage collector doesn't need to run when
  the FuncGraph goes out of scope, e.g. in tests using defun with
  @test_util.run_in_graph_and_eager_modes(assert_no_eager_garbage=True).

  Args:
    func_graph: A `FuncGraph` object to destroy. `func_graph` is unusable
      after this function.
  """
  # TODO(b/115366440): Delete this method when a custom OrderedDict is added.
  # Clearing captures using clear() leaves some cycles around.
  while func_graph.captures:
    func_graph.captures.popitem()
  memory.dismantle_ordered_dict(func_graph.captures)
  ops.dismantle_graph(func_graph)
