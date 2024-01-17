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

import traceback
from typing import Any, Callable, Hashable
import weakref

from tensorflow.core.function import trace_type
from tensorflow.core.function.capture import capture_container
from tensorflow.python.eager import context
from tensorflow.python.eager import execute
from tensorflow.python.eager.polymorphic_function import composite_tensor_utils
from tensorflow.python.framework import auto_control_deps
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.saved_model import save_context
from tensorflow.python.types import core
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util import variable_utils
from tensorflow.python.util.tf_export import tf_export


ALLOWLIST_COLLECTIONS = [
    ops.GraphKeys.GLOBAL_VARIABLES,
    ops.GraphKeys.LOCAL_VARIABLES,
    ops.GraphKeys.TRAINABLE_VARIABLES,
    variable_scope._VARSTORE_KEY,  # pylint: disable=protected-access
    variable_scope._VARSCOPESTORE_KEY  # pylint: disable=protected-access
]


class UnknownArgument(object):
  """Signifies an argument which is not currently handled."""


def convert_structure_to_signature(structure, arg_names=None,
                                   signature_context=None):
  """Convert a potentially nested structure to a signature.

  Args:
    structure: Structure to convert, where top level collection is a list or a
      tuple.
    arg_names: Optional list of arguments that has equal number of elements as
      `structure` and is used for naming corresponding TensorSpecs.
    signature_context: TraceType InternalTracingContext to generate alias_ids
      for mutable objects, like ResourceVariables.

  Returns:
    Identical structure that has TensorSpec objects instead of Tensors and
    UnknownArgument instead of any unsupported types.
  """

  def encode_arg(arg, path):
    """A representation for this argument, for converting into signatures."""
    if isinstance(arg, tensor_lib.Tensor):
      user_specified_name = None
      try:
        user_specified_name = compat.as_str(
            arg.op.get_attr("_user_specified_name"))
      except (ValueError, AttributeError):
        pass

      if path and user_specified_name and user_specified_name != path[0]:
        # The user has explicitly named the argument differently than the name
        # of the function argument.
        name = user_specified_name
      else:
        name = tensor_lib.sanitize_spec_name("_".join(str(p) for p in path))
      return tensor_lib.TensorSpec(arg.shape, arg.dtype, name)
    if isinstance(arg, resource_variable_ops.ResourceVariable):
      return trace_type.from_value(arg, signature_context)
    if isinstance(arg, composite_tensor.CompositeTensor):
      # TODO(b/133606651) Do we need to inject arg_name?
      return arg._type_spec  # pylint: disable=protected-access
    if isinstance(arg, (
        int,
        float,
        bool,
        str,
        type(None),
        dtypes.DType,
        tensor_lib.TensorSpec,
        type_spec.TypeSpec,
    )):
      return arg
    return UnknownArgument()

  # We are using the flattened paths to name the TensorSpecs. We need an
  # explicit name for them downstream.
  flattened = nest.flatten_with_tuple_paths(structure)
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
  return nest.pack_sequence_as(structure, mapped)


@tf_export("__internal__.FuncGraph", v1=[])
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
    seed: The graph-level random seed.
    capture_by_value: If True, the func graph will capture Variables by value
      instead of reference.
  """

  def __init__(self,
               name,
               collections=None,
               capture_by_value=None,
               structured_input_signature=None,
               structured_outputs=None):
    """Construct a new FuncGraph.

    The graph will inherit its graph key, collections, seed, and distribution
    strategy stack from the current context or graph.

    Args:
      name: the name of the function.
      collections: a dictionary of collections this FuncGraph should start with.
        If not specified (None), the FuncGraph will read (but not write to) the
        outer graph's collections that are not allowlisted, and both read and
        write to the outer graph's collections that are allowlisted. The current
        allowlisted collections are the global variables, the local variables,
        and the trainable variables. Defaults to None.
      capture_by_value: An optional boolean. If True, the func graph will
        capture Variables by value instead of reference. By default inherit from
        outer graphs, and failing that will default to False.
      structured_input_signature: Optional. The structured input signature to
        use for initializing the FuncGraph. See the docstring for FuncGraph for
        more information.
      structured_outputs: Optional. The structured outputs to use for
        initializing the FuncGraph. See the docstring for FuncGraph for more
        information.
    """
    super().__init__()
    self.name = name
    # TODO(panzf): Separate captures from non-captures inputs in self.inputs
    self.inputs = []
    self.outputs = []
    self.control_outputs = []
    self.structured_input_signature = structured_input_signature
    self.structured_outputs = structured_outputs
    self._resource_tensor_inputs = object_identity.ObjectIdentitySet()
    self._weak_variables = []
    self._watched_variables = object_identity.ObjectIdentityWeakSet()
    self.is_control_flow_graph = False

    self._function_captures = capture_container.FunctionCaptures()
    outer_graph = ops.get_default_graph()
    self._weak_outer_graph = weakref.ref(outer_graph)
    while outer_graph.building_function:
      outer_graph = outer_graph.outer_graph
    # If self._weak_outer_graph is deleted, we revert to the outermost Graph
    # active when the FuncGraph was traced. This will not be a FuncGraph.
    self._fallback_outer_graph = outer_graph
    # If not None, records the names of output args of this function. Used to
    # preserve the output names in the signature of a serialized+deserialized
    # function. Private at the moment mostly because it's often out of date.
    self._output_names = None
    # Inherit capture-by-value from outer graph.
    if capture_by_value is not None:
      self.capture_by_value = capture_by_value
    elif self.outer_graph is not None and isinstance(self.outer_graph,
                                                     FuncGraph):
      self.capture_by_value = self.outer_graph.capture_by_value
    else:
      self.capture_by_value = False

    self._building_function = True

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
        if collection_name not in ALLOWLIST_COLLECTIONS:
          self._collections[collection_name] = graph.get_collection(
              collection_name)
      for collection_name in ALLOWLIST_COLLECTIONS:
        self._collections[collection_name] = graph.get_collection_ref(
            collection_name)
    else:
      self._collections = collections

    # Keep track of whether this FuncGraph is exportable to SavedModel. Use
    # `graph.mark_as_unsaveable(reason)` to mark this FuncGraph and any
    # dependent functions as unsaveable.
    self._saveable = True
    self._saving_errors = set()

    # Keep track of callbacks to run when this graph exits default scope
    self._scope_exit_callbacks = None

  def __str__(self):
    return "FuncGraph(name=%s, id=%s)" % (self.name, id(self))

  def watch_variable(self, v):
    """Marks the variable v as accessed while building this graph."""
    # Don't watch `v` if it is one of ResourceVariable input arguments.
    if (isinstance(v, resource_variable_ops.ResourceVariable) and
        v.handle in self._resource_tensor_inputs):
      return

    while self is not None and isinstance(self, FuncGraph):
      self._watched_variables.add(v)
      self = self.outer_graph

  def capture_call_time_value(self,
                              closure,
                              spec,
                              key=None,
                              default_value=None,
                              placeholder=None):
    """Returns a placeholder which at call time has the value closure().

    The `tf.function` supports the notion of captures, that is, it allows Python
    functions to have closure variables, which bind over some value outside the
    function. However, this name binding is "early binding" performed before the
    program is run, i.e.,
    ```
    @tf.function
    def f():
      return x

    x = tf.constant(1)
    f()  # returns 1

    x = tf.constant(2)
    f()  # still returns 1!
    ```
    while in Python, name binding is performed as the program is running.
    ```
    def f():
      return x

    x = 1
    f()  # returns 1

    x = 2
    f()  # returns 2
    ```
    `capture_call_time_value` allows tf.function to mimic late binding as a
    Python function does, by passing in a `closure` callable argument to be
    executed when the tf.function is invoked eagerly.  E.g.
    ```
    @tf.function
    def f():
      return ops.get_default_graph.capture_call_time_value(lambda: x)

    x = tf.constant(1)
    f()  # returns 1

    x = tf.constant(2)
    f()  # returns 2
    ```
    Note that a `capture_call_time_value` function itself does not work well in
    the saving process (since the tf.function in which it's called is not
    invoked eagerly) unless passed a `default_value` argument. At saving time,
    the `default_value` argument is returned instead.

    Args:
      closure: function which takes no arguments, to be evaluated at function
        call time, returning a nest of tensors compatible with `spec`.
      spec: nest of TypeSpec for the value to capture.
      key: optional. If not None, multiple calls to lazy_capture with the same
        key in the same graph will return the same placeholder, and the first
        closure will be used at function call time.
      default_value: optional value to return in environments that cannot safely
        evaluate closure.
      placeholder: optional. If not None, the graph will take the passed-in
        `placeholder` as the internal capture instead of creating a new one.
        This is useful when loading from a SavedModel.

    Returns:
      Nest of placeholders which, at function call time, will be fed with the
      result of calling closure().

    Raises:
      ValueError: at function call time, if the return value of closure() is
       not compatible with `spec`.
    """
    if key is None:
      key = object()
    if key not in self._function_captures.by_ref_internal:
      trace_ctx = trace_type.InternalTracingContext(True)
      spec = trace_type.from_value(spec, trace_ctx)

      if placeholder is None:
        placeholder_ctx = trace_type.InternalPlaceholderContext(self)
        placeholder = spec.placeholder_value(placeholder_ctx)

      def wrapped_closure():

        # One major case requiring returning a `default_value` is when passing a
        # concrete function to `save`, i.e.
        # serving_fn = serve_fn.get_concrete_function(...)
        # model.save(save_dir, signatures={"serving_default": serving_fn})
        # `serving_fn` has deferred captures added through
        # `capture_call_time_value`. It can't be saved correctly since
        # `wrapped_closure` will end up executing under a default Graph instead
        # of FuncGraph. The user of `capture_call_time_value` also cannot
        # conditionally avoid this call since presence of `save_context` when
        # executing `wrapped_closure` is not known at tracing time of
        # `serving_fn`.
        if save_context.in_save_context() and default_value is not None:
          return default_value
        # TODO(wxinyi): raise an error if in save context but no default value.

        if not context.executing_eagerly():
          graph = ops.get_default_graph()
          assert isinstance(
              graph,
              FuncGraph), "This API should only be used in TF2 enviroment."

          with graph.as_default():
            ret_nest = graph.capture_call_time_value(
                closure, spec, key=key, default_value=default_value)
        else:
          ret_nest = closure()

        ret_nest = spec.cast(ret_nest, trace_type.InternalCastContext)
        return spec.to_tensors(ret_nest)

      wrapped_closure.output_spec = spec
      self._function_captures.add_or_replace(
          key=key,
          external=wrapped_closure,
          internal=placeholder,
          tracetype=spec,
          is_by_ref=True)
    return self._function_captures.by_ref_internal[key]

  def control_dependencies(self, control_inputs):
    """Handles control dependencies.

    FuncGraph wraps Graph's control_dependencies logic by first filtering out
    any external tensors / operations and storing them in the graph's
    control_captures member. Any consumers of this function graph must then
    decide how to handle the control captures.

    Args:
      control_inputs: A list of `Operation` or `Tensor` objects which must be
        executed or computed before running the operations defined in the
        context.  Can also be `None` to clear the control dependencies.

    Returns:
     A context manager that specifies control dependencies for all
     operations constructed within the context.

    Raises:
      TypeError: If `control_inputs` is not a list of `Operation` or
        `Tensor` objects.
    """
    if control_inputs is None:
      return super().control_dependencies(control_inputs)

    filtered_control_inputs = []
    for c in control_inputs:
      # Check for _UnreadVariable
      if (isinstance(c, indexed_slices.IndexedSlices) or
          (hasattr(c, "_handle") and hasattr(c, "op"))):
        c = c.op
      graph_element = ops._as_graph_element(c)  # pylint: disable=protected-access
      if graph_element is None:
        graph_element = c
      if graph_element is not None and getattr(graph_element, "graph",
                                               None) is not self:
        self._function_captures.control.add(graph_element)
      else:
        filtered_control_inputs.append(graph_element)
    return super().control_dependencies(filtered_control_inputs)

  def as_default(self):
    outer_cm = super().as_default()

    @tf_contextlib.contextmanager
    def inner_cm():
      """Context manager for copying distribute.Strategy scope information."""
      # pylint: disable=protected-access
      # TODO(b/112906995, nareshmodi): distribution strategy depends on
      # inheriting this stack from the default graph even in eager mode. Maybe
      # it should be part of the eager context? This would also allow us to
      # remove a get_default_graph() call from the function cache lookup.
      graph = ops.get_default_graph()
      old_strategy_stack = self._distribution_strategy_stack
      self._distribution_strategy_stack = list(
          graph._distribution_strategy_stack)

      # We ignore device placements from any outer scopes while tracing the
      # function when possible, to avoid hard-coding them in the function
      # graph. "Default" placements come from the PartitionedCallOp's placement,
      # so that the same trace of the Python function may be placed on several
      # different devices and saved functions may be placed on new devices when
      # restored.
      # However, we need to preserve the outer device stack in the following
      # cases in non eager context:
      # 1. device stack is callable
      # 2. When using distribution strategy with legacy graph mode.
      old_device_stack = self._device_function_stack
      if (not context.executing_eagerly() and
          (device_stack_has_callable(graph._device_function_stack) or
           (self._distribution_strategy_stack and
            not ops.executing_eagerly_outside_functions()))):
        # Hard-code devices from device functions in the function body
        self._device_function_stack = graph._device_function_stack.copy()

      old_creator_stack = self._variable_creator_stack
      self._variable_creator_stack = graph._variable_creator_stack
      # Inherit the graph key, since this is used for matching variables in
      # optimizers.
      old_graph_key = self._graph_key
      self._graph_key = graph._graph_key
      # pylint: enable=protected-access

      old_scope_exit_callbacks = self._scope_exit_callbacks
      self._scope_exit_callbacks = []

      with outer_cm as g:
        try:
          yield g
        finally:
          try:
            for fn in self._scope_exit_callbacks:
              fn()
          finally:
            self._scope_exit_callbacks = old_scope_exit_callbacks
            self._distribution_strategy_stack = old_strategy_stack
            self._device_function_stack = old_device_stack
            self._variable_creator_stack = old_creator_stack
            self._graph_key = old_graph_key

    return inner_cm()

  @property
  def outer_graph(self):
    """The Graph this FuncGraph is nested in.

    Functions may capture Tensors from graphs they are nested in (transitive).

    Returns:
      A Graph object. Initially set to the current default graph when the
      FuncGraph was created. If the previous `outer_graph` was deleted because
      the function that owns it was deleted, `outer_graph` is reset to the
      outermost default graph active when the FuncGraph was created. This
      FuncGraph won't have captured anything from the new `outer_graph` (and
      likely not from the previous setting, since that would have created a
      strong reference), but it is returned so that FuncGraphs always have a
      parent.
    """
    current = self._weak_outer_graph()
    if current is None:
      return self._fallback_outer_graph
    return current

  @outer_graph.setter
  def outer_graph(self, new_outer_graph):
    """Sets `outer_graph` to `new_outer_graph`."""
    self._weak_outer_graph = weakref.ref(new_outer_graph)

  @property
  def output_types(self):
    return [t.dtype for t in self.outputs]

  @property
  def output_shapes(self):
    return [t.shape for t in self.outputs]

  @property
  def trainable_variables(self):
    """A sequence of trainable variables accessed by this FuncGraph.

    Note that functions keep only weak references to variables. Calling the
    function after a variable it accesses has been deleted is an error.

    Returns:
      Sequence of trainable variables for this func graph.
    """
    return tuple(v for v in self.variables if v.trainable)

  @property
  def variables(self):
    """A sequence of variables accessed by this FuncGraph.

    Note that functions keep only weak references to variables. Calling the
    function after a variable it accesses has been deleted is an error.

    Returns:
      Sequence of variables for this func graph.
    """

    def deref(weak_v):
      v = weak_v()
      if v is None:
        raise AssertionError(
            "Called a function referencing variables which have been deleted. "
            "This likely means that function-local variables were created and "
            "not referenced elsewhere in the program. This is generally a "
            "mistake; consider storing variables in an object attribute on "
            "first call.")
      return v

    return tuple(deref(v) for v in self._weak_variables)

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
    reverse_captures = dict((id(v), k) for k, v in self.captures)
    uncaptured_inputs = [reverse_captures.get(id(t), t) for t in inputs]
    with ops.init_scope():
      if context.executing_eagerly():
        attr_list = ("dtype", int(attrs["dtype"].type))
        value, = execute.execute(
            compat.as_bytes(op_type), 1, uncaptured_inputs, attr_list,
            context.context())
      else:
        op = ops.get_default_graph()._create_op_internal(  # pylint: disable=protected-access
            op_type, uncaptured_inputs, dtypes, input_types, name, attrs,
            op_def, compute_device)
        value = op.outputs[0]
    captured_value = self.capture(value)
    return captured_value.op

  def _create_op_internal(
      self,
      op_type,
      inputs,
      dtypes=None,  # pylint: disable=redefined-outer-name
      input_types=None,
      name=None,
      attrs=None,
      op_def=None,
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
      input_types: (Optional.) A list of `DType`s that will be the types of the
        tensors that the operation consumes. By default, uses the base `DType`
        of each input in `inputs`. Operations that expect reference-typed inputs
        must specify `input_types` explicitly.
      name: (Optional.) A string name for the operation. If not specified, a
        name is generated based on `op_type`.
      attrs: (Optional.) A dictionary where the key is the attribute name (a
        string) and the value is the respective `attr` attribute of the
        `NodeDef` proto that will represent the operation (an `AttrValue`
        proto).
      op_def: (Optional.) The `OpDef` proto that describes the `op_type` that
        the operation will have.
      compute_device: (Optional.) If True, device functions will be executed to
        compute the device property of the Operation.

    Returns:
      An `Operation` object.
    """
    if self.capture_by_value and op_type in [
        "ReadVariableOp", "ResourceGather"
    ]:
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
    # Use a different list to avoid modifying the original inputs list.
    captured_inputs = []
    for inp in inputs:
      # TPU Estimator defines a control flow context with no AddValue method.
      if ctxt is not None and hasattr(ctxt, "AddValue"):
        inp = ctxt.AddValue(inp)
      inp = self.capture(inp)
      captured_inputs.append(inp)
    return super()._create_op_internal(  # pylint: disable=protected-access
        op_type, captured_inputs, dtypes, input_types, name, attrs, op_def,
        compute_device)

  def capture(self, tensor, name=None, shape=None):
    return self._function_captures.capture_by_value(self, tensor, name)

  def _validate_in_scope(self, tensor):
    inner_graph = tensor.graph
    while inner_graph is not None and isinstance(inner_graph, FuncGraph):
      if inner_graph is self:
        try:
          tb = tensor.op.traceback
        except AttributeError:
          tensor_traceback = "<unknown>"
        else:
          tensor_traceback_list = []
          for frame in traceback.format_list(tb.get_user_frames()):
            tensor_traceback_list.extend(
                [f"  {line}" for line in frame.split("\n") if line.strip()])
          tensor_traceback = "\n".join(tensor_traceback_list)
        # Keep in sync with tfe_wrapper.cc.
        # TODO(b/200991648): Unify those two paths.
        raise errors.InaccessibleTensorError(
            f"{tensor!r} is out of scope and cannot be used here. Use return "
            "values, explicit Python locals or TensorFlow collections to "
            "access it.\n"
            "Please see https://www.tensorflow.org/guide/function#all_outputs_of_a_tffunction_must_be_return_values "  # pylint: disable=line-too-long
            "for more information.\n\n"
            f"{tensor!r} was defined here:\n{tensor_traceback}\n\n"
            f"The tensor {tensor!r} cannot be accessed from {self}, because "
            f"it was defined in {tensor.graph}, which is out of scope.")
      inner_graph = inner_graph.outer_graph

  # TODO(panzf): Rename this method along with usages in cond/while graph.
  def _capture_helper(self, tensor, name):
    return self._function_captures._create_placeholder_helper(  # pylint: disable=protected-access
        self, tensor, name)

  def _experimental_capture_side_input_by_ref(self, identifier: Hashable,
                                              func: Callable[[], Any]) ->...:
    """Implement capturing side input by reference for tf.function.

    Note that this API will only register the capture in the func_graph where
    it is called. In the case of nested graph, like nested tf.function or
    tf.while, the outer graph is not aware of this capture in the inner graph.
    Thus, the outer tf.function will not retrace when the by-ref capture
    changes. It's the user's responsibility to call this API in the outer
    func_graph as well if proper retracing is needed.

    For example:

    ```
    x = 1

    # Correct usage
    @tf.function
    def f_1():
      graph = tf.compat.v1.get_default_graph()
      # Capture the same x for the outer tf.function
      graph._experimental_capture_side_input_by_ref("x", lambda: x)

      @tf.function
      def g():
        graph = tf.compat.v1.get_default_graph()
        cap_x = graph._experimental_capture_side_input_by_ref("x", lambda: x)
        return cap_x + 1

      return g()

    # Incorrect usage
    @tf.function
    def f_2():

      @tf.function
      def g():
        graph = tf.compat.v1.get_default_graph()
        cap_x = graph._experimental_capture_side_input_by_ref("x", lambda: x)
        return cap_x + 1

      return g()

    assert f_1() == 2
    assert f_2() == 2
    x = 2
    assert f_1() == 3
    assert f_2() == 2  # This is incorrect
    ```

    Args:
      identifier: A hashable object as the key for the capture.
      func: A Python function that takes no arguments and returns the value of
        side input. The function is evaluated at function call time.

    Returns:
      A nested structure with the same structure as the side input. Tensors
        are replaced with placehoders, and non-tensors remain the same.

    """
    if context.executing_eagerly():
      return func()

    def maybe_convert_to_tensor():
      value = func()
      if not (isinstance(value, core.Value) or isinstance(value, core.Symbol)):
        value = constant_op.constant(value)
      return value

    placeholder = self._function_captures._capture_by_ref(  # pylint: disable=protected-access
        self, maybe_convert_to_tensor, identifier)
    return placeholder

  @property
  def captures(self):
    """Order list of tuples containing external and internal captures."""
    return self._function_captures.by_val_capture_tuples

  def add_capture(self, tensor, placeholder):
    """Capture a specific tensor and utilize the provided placeholder.

    Args:
      tensor: Tensor to captures.
      placeholder: Provided placeholder for the tensor.
    """
    self._function_captures.add_or_replace(
        key=id(tensor),
        external=tensor,
        internal=placeholder,
        is_by_ref=False)
    self.inputs.append(placeholder)

  def replace_capture(self, tensor, placeholder):
    """Replace already existing capture."""
    self._function_captures.add_or_replace(
        key=id(tensor),
        external=tensor,
        internal=placeholder,
        is_by_ref=False)

  def replace_capture_with_deferred_capture(self,
                                            tensor,
                                            closure,
                                            spec,
                                            placeholder,
                                            default_value=None):
    """Replaces existing capture `tensor` with a deferred capture `closure`.

    Caution: It is the caller's responsibility to make sure that, after calling
    this function, the TypeSpec of the `inputs` (i.e. internal placeholders) and
    the `_captured_inputs` (i.e. external captures) of a concrete function that
    wraps this function graph are still compatible. Thus user should pairing
    usage of this function with `ConcreteFunction.set_external_captures` to make
    sure the order still matches. For example,
    ```
    # concrete_fn._captured_inputs == [tensor1, tensor2, tensor3]
    # concrete_fn.inputs == [placeholder1, placeholder2, placeholder3]
    # replace external capture `tensor2` with a deferred_capture, i.e., a
    # closure, `closure2`
    concrete_fn.graph.replace_capture_with_deferred_capture(tensor2,
                                                            closure2,
                                                            placeholder2,
                                                            some_spec,
                                                            some_default)
    concrete_fn.set_external_captures([tensor1, closure2, tensor3])
    ```

    Args:
      tensor: Tensor already captured.
      closure: function which takes no arguments, to be evaluated at function
        call time, returning a nest of tensors compatible with `spec`.
      spec: nest of TypeSpec for the value to capture.
      placeholder: the internal placeholder corresponding to the captured
        `tensor`.
      default_value: optional value to use in environments that cannot safely
        evaluate closure.
    """
    self._function_captures.pop(id(tensor), is_by_ref=False)
    self.capture_call_time_value(
        closure,
        spec,
        key=id(tensor),
        default_value=default_value,
        placeholder=placeholder)

  @property
  def external_captures(self):
    """External tensors captured by this function."""
    return list(self._function_captures.by_val_external.values())

  @property
  def internal_captures(self):
    """Placeholders in this function corresponding captured tensors."""
    return list(self._function_captures.by_val_internal.values())

  @property
  def deferred_external_captures(self):
    """Ordered nest of tensors whose placeholders will be fed at call time."""
    return list(self._function_captures.by_ref_external.values())

  @property
  def deferred_internal_captures(self):
    """List of nest of placeholders which at call time will be fed."""
    return list(self._function_captures.by_ref_internal.values())

  @property
  def variable_captures(self):
    """Map of python object ids of variables to variables which are captured."""
    return self.variables

  @property
  def function_captures(self):
    return self._function_captures

  def mark_as_unsaveable(self, error_message):
    """Marks this FuncGraph as unsaveable.

    Any attempts to export this FuncGraph will raise an error with the specified
    message.

    Args:
      error_message: List or string containing the error message to be raised
        when saving this FuncGraph to SavedModel.
    """
    self._saveable = False
    if isinstance(error_message, str):
      error_message = [error_message]
    self._saving_errors.update(error_message)

  @property
  def saveable(self):
    """Returns whether this FuncGraph is saveable."""
    return self._saveable

  @property
  def saving_errors(self):
    """Returns set of errors preventing this FuncGraph from being saved."""
    return self._saving_errors

  def _add_scope_exit_callback(self, fn):
    """Add a function to call when this graph exits the default scope."""
    if not callable(fn):
      raise TypeError("fn is not callable: {}".format(fn))
    if self._scope_exit_callbacks is None:
      raise RuntimeError(
          "Attempting to add a scope exit callback, but the default graph is "
          "not the context scope graph.  Did you forget to call "
          "'with graph.as_default(): ...'?")
    self._scope_exit_callbacks.append(fn)


def func_graph_from_py_func(name,
                            python_func,
                            args,
                            kwargs,
                            signature=None,
                            func_graph=None,
                            add_control_dependencies=True,
                            arg_names=None,
                            op_return_value=None,
                            collections=None,
                            capture_by_value=None,
                            create_placeholders=True):
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
    add_control_dependencies: If True, automatically adds control dependencies
      to ensure program order matches execution order and stateful ops always
      execute.
    arg_names: Optional list of argument names, used to give input placeholders
      recognizable names.
    op_return_value: Optional. A Tensor. If set and `python_func` returns
      Operations, those return values will be replaced with this value. If not
      set, returning an Operation triggers an error.
    collections: a dictionary of collections this FuncGraph should start with.
      If not specified (None), the FuncGraph will read (but not write to) the
      outer graph's collections that are not allowlisted, and both read and
      write to the outer graph's collections that are allowlisted. The current
      allowlisted collections are the global variables, the local variables, and
      the trainable variables. Defaults to None.
    capture_by_value: An optional boolean. If True, the func graph will capture
      Variables by value instead of reference. By default inherit from outer
      graphs, and failing that will default to False.
    create_placeholders: An optional boolean. If True, then func graph will
      create placeholders for the inputs as graph ops. If False, the input args
      and kwargs will be treated as the input placeholders.

  Returns:
    A FuncGraph.

  Raises:
    TypeError: If any of `python_func`'s return values is neither `None`, a
      `Tensor` or a `tf.experimental.ExtensionType`.
  """
  if op_return_value is not None:
    assert isinstance(op_return_value, tensor_lib.Tensor), op_return_value
  if func_graph is None:
    func_graph = FuncGraph(
        name, collections=collections, capture_by_value=capture_by_value)
  assert isinstance(func_graph, FuncGraph)
  if add_control_dependencies:
    deps_control_manager = auto_control_deps.AutomaticControlDependencies()
  else:
    deps_control_manager = ops.NullContextmanager()

  with func_graph.as_default(), deps_control_manager as deps_ctx:
    current_scope = variable_scope.get_variable_scope()
    default_use_resource = current_scope.use_resource
    current_scope.set_use_resource(True)

    if signature is not None:
      args = signature
      kwargs = {}

    if create_placeholders:
      func_args, func_kwargs = _create_placeholders(args, kwargs, arg_names)
    else:
      func_args, func_kwargs = args, kwargs

    input_trace_types = trace_type.from_value([func_args, func_kwargs])
    func_graph.inputs = input_trace_types.to_tensors([func_args, func_kwargs])  # pylint: disable=protected-access

    # Reset variables watched while deconstructing inputs.
    func_graph._watched_variables = object_identity.ObjectIdentityWeakSet()  # pylint: disable=protected-access

    for arg in func_graph.inputs:
      if arg.dtype == dtypes.resource:
        func_graph._resource_tensor_inputs.add(arg)  # pylint:disable=protected-access

    signature_context = trace_type.InternalTracingContext()
    # Convert all Tensors into TensorSpecs before saving the structured inputs.
    # If storing pure concrete functions that are not called through polymorphic
    # functions, we don't have access to FunctionSpec, so we need to call the
    # TensorSpecs by their `arg_names` for later binding.
    func_graph.structured_input_signature = (
        convert_structure_to_signature(
            func_args, arg_names, signature_context=signature_context),
        convert_structure_to_signature(
            func_kwargs, signature_context=signature_context))

    # Note: `nest.flatten` sorts by keys, as does `_deterministic_dict_values`.
    # Variables to help check whether mutation happens in calling the function
    # Copy the recursive list, tuple and map structure, but not base objects
    func_args_before = nest.pack_sequence_as(
        func_args,
        nest.flatten(func_args, expand_composites=True),
        expand_composites=True)
    func_kwargs_before = nest.pack_sequence_as(
        func_kwargs,
        nest.flatten(func_kwargs, expand_composites=True),
        expand_composites=True)

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
              "To be compatible with tf.function, Python functions "
              "must return zero or more Tensors or ExtensionTypes or None "
              f"values; in compilation of {str(python_func)}, found return "
              f"value of type {type(x).__name__}, which is not a Tensor or "
              "ExtensionType.")
      if add_control_dependencies:
        x = deps_ctx.mark_as_return(x)
      return x

    _, original_func = tf_decorator.unwrap(python_func)
    func_outputs = python_func(*func_args, **func_kwargs)

    # invariant: `func_outputs` contains only Tensors, CompositeTensors,
    # TensorArrays and `None`s.
    func_outputs = variable_utils.convert_variables_to_tensors(func_outputs)
    func_outputs = nest.map_structure(
        convert, func_outputs, expand_composites=True)

    # flatten and unflatten func_args and func_kwargs to maintain parity
    # from flattening which sorts by key
    func_args = nest.pack_sequence_as(
        func_args,
        nest.flatten(func_args, expand_composites=True),
        expand_composites=True)
    func_kwargs = nest.pack_sequence_as(
        func_kwargs,
        nest.flatten(func_kwargs, expand_composites=True),
        expand_composites=True)
    check_func_mutation(func_args_before, func_kwargs_before, func_args,
                        func_kwargs, original_func)
    current_scope.set_use_resource(default_use_resource)

    inputs = []
    for arg in composite_tensor_utils.flatten_with_variables([func_args,
                                                              func_kwargs]):
      if isinstance(arg, resource_variable_ops.BaseResourceVariable):
        # Even if an argument variable was not used in the function, we've
        # already manually captured the resource Tensor when creating argument
        # placeholders.
        capture = func_graph._function_captures.pop(id(arg.handle), False)  # pylint: disable=protected-access
        assert len(capture) >= 2
        resource_placeholder = capture[1]
        if resource_placeholder is None:
          continue
        inputs.append(resource_placeholder)
      elif isinstance(arg, tensor_lib.Tensor):
        inputs.append(arg)
    func_graph.inputs = (
        inputs + func_graph.internal_captures + nest.flatten(
            func_graph.deferred_internal_captures, expand_composites=True))
    func_graph.structured_outputs = func_outputs
    # Returning a closed-over tensor does not trigger convert_to_tensor.
    func_graph.outputs.extend(
        func_graph.capture(x)
        for x in flatten(func_graph.structured_outputs)
        if x is not None)

    func_graph.variables = func_graph._watched_variables  # pylint: disable=protected-access

  if add_control_dependencies:
    func_graph.control_outputs.extend(deps_control_manager.ops_which_must_run)
    func_graph.collective_manager_ids_used = (
        deps_control_manager.collective_manager_ids_used)

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
    for input_t, placeholder_t in tensor.op.graph.captures:
      if tensor == placeholder_t:
        return maybe_captured(input_t)
  # pylint: enable=protected-access
  return tensor


def device_stack_has_callable(device_stack):
  """Checks whether a device stack contains a callable."""
  return any(
      callable(spec._device_name_or_function)  # pylint: disable=protected-access
      for spec in device_stack.peek_objs())


def has_mutation(n1, n2):
  """Returns true if n1 and n2 are different (using `is` to compare leaves)."""
  try:
    nest.assert_same_structure(n1, n2, expand_composites=True)
  except ValueError:
    return True

  for arg1, arg2 in zip(
      nest.flatten(n1, expand_composites=True),
      nest.flatten(n2, expand_composites=True)):
    if arg1 is not arg2:
      return True

  return False


def check_func_mutation(old_args, old_kwargs, new_args, new_kwargs, func):
  """Checks that the arguments to a function are not modified."""
  if not has_mutation((old_args, old_kwargs), (new_args, new_kwargs)):
    return

  # Mutation detected; construct a useful error message.
  func_name = getattr(func, "__qualname__", getattr(func, "__name__", func))
  signature = tf_inspect.signature(func)
  try:
    old_bound = signature.bind(*old_args, **old_kwargs).arguments
    new_bound = signature.bind(*new_args, **new_kwargs).arguments
  except TypeError as e:
    # This occurs when the function is called with the (deprecated)
    # "flat signature".  See ConcreteFunction._call_with_flat_signature.  In
    # this case, we can't report which arguments were modified.
    raise ValueError(
        f"{func_name}{signature} should not modify its Python input "
        f"arguments. Check if it modifies any lists or dicts passed as "
        f"arguments. Modifying a copy is allowed.") from e

  assert set(old_bound) == set(new_bound)
  modified_args = [
      arg_name for arg_name in new_bound
      if has_mutation(old_bound[arg_name], new_bound[arg_name])
  ]
  changes = ", ".join(modified_args)
  raise ValueError(f"{func_name}{signature} should not modify its Python "
                   f"input arguments. Modifying a copy is allowed. The "
                   f"following parameter(s) were modified: {changes}")


# TODO(edloper): If TensorArray becomes a CompositeTensor, then delete this.
def flatten(sequence):
  """Like nest.flatten w/ expand_composites, but returns flow for TensorArrays.

  Args:
    sequence: A nested structure of Tensors, CompositeTensors, and TensorArrays.

  Returns:
    A list of tensors.
  """
  flat_sequence = nest.flatten(sequence, expand_composites=True)
  return [
      item.flow if isinstance(item, tensor_array_ops.TensorArray) else item
      for item in flat_sequence
  ]


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


def _create_placeholders(args, kwargs, arg_names=None):
  """Create placeholders given positional args and keyword args."""
  signature_context = trace_type.InternalTracingContext(
      is_legacy_signature=True)
  arg_trace_types = trace_type.from_value(tuple(args), signature_context)
  kwarg_trace_types = trace_type.from_value(kwargs, signature_context)

  placeholder_mapping = signature_context.get_placeholder_mapping()
  placeholder_context = trace_type.InternalPlaceholderContext(
      ops.get_default_graph(), placeholder_mapping)

  if arg_names is None:
    arg_names = [None] * len(arg_trace_types.components)

  # Create placeholders for trace type args and trace type kwargs
  func_args = []
  for name, trace_type_arg in zip(arg_names, arg_trace_types.components):
    placeholder_context.update_naming_scope(name)
    placeholder = trace_type_arg.placeholder_value(placeholder_context)
    func_args.append(placeholder)

  func_kwargs = {}
  for name, trace_type_kwarg in zip(*sorted(kwarg_trace_types.mapping.items())):
    placeholder_context.update_naming_scope(name)
    placeholder = trace_type_kwarg.placeholder_value(placeholder_context)
    func_kwargs[name] = placeholder

  return tuple(func_args), func_kwargs


def dismantle_func_graph(func_graph):
  """Removes reference cycles in `func_graph` FuncGraph.

  Helpful for making sure the garbage collector doesn't need to run when
  the FuncGraph goes out of scope, e.g. in tests using defun with
  @test_util.run_in_graph_and_eager_modes(assert_no_eager_garbage=True).

  Args:
    func_graph: A `FuncGraph` object to destroy. `func_graph` is unusable after
      this function.
  """
  func_graph._function_captures.clear()  # pylint: disable=protected-access
  ops.dismantle_graph(func_graph)


def override_func_graph_name_scope(func_graph, name_scope):
  func_graph._name_stack = name_scope  # pylint: disable=protected-access
