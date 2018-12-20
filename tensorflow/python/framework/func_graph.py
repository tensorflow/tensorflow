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

import collections
import weakref

from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager import tape
from tensorflow.python.eager.graph_only_ops import graph_placeholder
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
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
    structured_outputs: A possibly-nested python object which will be returned
      by this function. The Tensors in this structure are the same as those of
      self.outputs. Note that this structure might contain Python `None`s.
    variables: Variables that should be watched during function execution.
    outer_graph: The graph this function is defined in. May be another FuncGraph
      or the global default Graph.
    captures: Maps external tensor -> internal tensor (i.e. input placeholder).
      The entries are in the order they were captured.
    seed: The graph-level random seed.
  """

  def __init__(self, name, read_only_collections=True):
    """Construct a new FuncGraph.

    The graph will inherit its graph key, collections, seed, and distribution
    strategy stack from the current context or graph.

    Args:
      name: the name of the function.
      read_only_collections: whether to not write function graph collections
        back to default graph. Defaults to True.
    """
    super(FuncGraph, self).__init__()

    self.name = name
    self.inputs = []
    self.outputs = []
    self.structured_outputs = None
    self._read_only_collections = read_only_collections
    self._weak_variables = []
    self.outer_graph = ops.get_default_graph()
    self.captures = collections.OrderedDict()

    self._building_function = True
    # Map from resource tensor name to last op (in program order) which uses
    # this tensor. Used to enforce that execution order matches program order
    # for resource tensors.
    self._last_op_using_resource_tensor = {}

    graph = self.outer_graph

    if context.executing_eagerly():
      self.seed = context.global_seed()
      device_type = context.context().device_spec.device_type
      self._xla_compile = (device_type == "TPU" or device_type == "XLA_GPU"
                           or device_type == "XLA_CPU")
    else:
      self.seed = graph.seed
      self._xla_compile = getattr(graph, "_xla_compile", False)
      # TODO(allenl): Figure out if we can remove colocation stack
      # specialization (currently used in cond_v2), here and in the cache key.
      self._colocation_stack = graph._colocation_stack.copy()  # pylint: disable=protected-access

    if not self._read_only_collections:
      self._collections = graph._collections  # pylint: disable=protected-access
    else:
      for collection_name in graph.get_all_collection_keys():
        if collection_name not in WHITELIST_COLLECTIONS:
          self._collections[collection_name] = graph.get_collection(
              collection_name)
      for collection_name in WHITELIST_COLLECTIONS:
        self._collections[collection_name] = graph.get_collection_ref(
            collection_name)

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
        if self._distribution_strategy_stack or self._xla_compile:
          self._add_device_to_stack(context.context().device_name)
      else:
        if (self._distribution_strategy_stack
            or self._xla_compile
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

  def create_op(
      self,
      op_type,
      inputs,
      dtypes,
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
      dtypes: A list of `DType` objects that will be the types of the tensors
        that the operation produces.
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
    if isinstance(tensor, ops.EagerTensor):
      if name is None:
        name = str(ops.uid())
      return self._capture_helper(tensor, name)
    if tensor.graph is not self:
      if name is None:
        name = tensor.op.name
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
                            add_control_dependencies=True,
                            arg_names=None,
                            op_return_value=None):
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
    add_control_dependencies: If True, automatically adds control dependencies
      to ensure program order matches execution order and stateful ops always
      execute.
    arg_names: Optional list of argument names, used to give input placeholders
      recognizable names.
    op_return_value: Optional. A Tensor. If set and `python_func` returns
      Operations, those return values will be replaced with this value. If not
      set, returning an Operation triggers an error.

  Returns:
    A FuncGraph.

  Raises:
    TypeError: If any of `python_func`'s return values is neither `None` nor a
      `Tensor`.
  """
  if op_return_value is not None:
    assert isinstance(op_return_value, ops.Tensor), op_return_value
  if func_graph is None:
    func_graph = FuncGraph(name)
  assert isinstance(func_graph, FuncGraph)
  if add_control_dependencies:
    control_manager = AutomaticControlDependencies
  else:
    control_manager = ops.NullContextmanager
  with func_graph.as_default(), control_manager() as a:
    current_scope = variable_scope.get_variable_scope()
    default_use_recource = current_scope.use_resource
    current_scope.set_use_resource(True)

    if signature is not None:
      args = signature
      kwargs = {}

    # Creates and names placeholders for all arguments.
    func_args = _get_defun_inputs_from_args(args, arg_names)
    func_kwargs = _get_defun_inputs_from_kwargs(kwargs)

    # Note: `nest.flatten` sorts by keys, as does `_deterministic_dict_values`.
    # Variables to help check whether mutation happens in calling the function
    # Copy the recursive list, tuple and map structure, but not base objects
    func_args_before = nest.pack_sequence_as(func_args, nest.flatten(func_args))
    func_kwargs_before = nest.pack_sequence_as(
        func_kwargs, nest.flatten(func_kwargs))

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
          x = ops.convert_to_tensor_or_indexed_slices(x)
        except (ValueError, TypeError):
          raise TypeError(
              "To be compatible with tf.contrib.eager.defun, Python functions "
              "must return zero or more Tensors; in compilation of %s, found "
              "return value of type %s, which is not a Tensor." %
              (str(python_func), type(x)))
      if add_control_dependencies:
        x = a.mark_as_return(x)
      return x

    this_tape = tape.push_new_tape()
    try:
      if autograph:
        from tensorflow.python import autograph  # pylint: disable=g-import-not-at-top
        _, original_func = tf_decorator.unwrap(python_func)

        def wrapper(*args, **kwargs):
          # Note: functions annotated with @tf.function should always be
          # converted even though they would meet autograph's whitelisting
          # criteria.
          # If this assumption is ever broken, converted_call will need to
          # handle the possibility of original_func still being a shim, e.g.
          # bound to WeakrefSelf.
          return autograph.converted_call(
              original_func, None,
              autograph.ConversionOptions(
                  verbose=autograph.Verbosity.BRIEF,
                  recursive=True,
                  strip_decorators=(def_function.function,),
                  optional_features=(),
                  force_conversion=True,
              ), *args, **kwargs)

        # Wrapping around a decorator allows checks like tf_inspect.getargspec
        # to be accurate.
        converted_func = tf_decorator.make_decorator(original_func, wrapper)
        tf_decorator.rewrap(python_func, original_func, converted_func)

      func_outputs = python_func(*func_args, **func_kwargs)

      # invariant: `func_outputs` contains only Tensors, IndexedSlices,
      # SparseTensors, TensorArrays and `None`s.
      func_outputs = nest.map_structure(convert, func_outputs)

      check_mutation(func_args_before, func_args)
      check_mutation(func_kwargs_before, func_kwargs)
    finally:
      tape.pop_tape(this_tape)
      current_scope.set_use_resource(default_use_recource)

    # Variables in `func_args`, `func_kwargs` should be explicit inputs
    # to the function, not captured inputs.
    tape_variables = this_tape.watched_variables()
    arg_variables = set()
    inputs = []
    for arg in nest.flatten(func_args) + nest.flatten(func_kwargs):
      if isinstance(arg, resource_variable_ops.ResourceVariable):
        # Even if an argument variable was not used in the function, we've
        # already manually captured the resource Tensor when creating argument
        # placeholders.
        resource_placeholder = func_graph.captures.pop(arg.handle)
        arg_variables.add(arg)
        inputs.append(resource_placeholder)
      elif isinstance(arg, ops.Tensor):
        inputs.append(arg)
    variables = [v for v in tape_variables if v not in arg_variables]
    func_graph.inputs = inputs + list(func_graph.captures.values())

    func_graph.structured_outputs = func_outputs
    # Returning a closed-over tensor does not trigger convert_to_tensor.
    func_graph.outputs.extend(
        func_graph.capture(x)
        for x in flatten(func_graph.structured_outputs)
        if x is not None)

    func_graph.variables = variables

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
    nest.assert_same_structure(n1, n2)
  except ValueError:
    raise ValueError(errmsg)

  for arg1, arg2 in zip(nest.flatten(n1), nest.flatten(n2)):
    if arg1 is not arg2:
      raise ValueError(errmsg)


def flatten(sequence):
  """Like `nest.flatten` but also unpacks other Tensor-like objects.

  Flattens non-tensor objects into their constituent tensors.

  Args:
    sequence: A nested structure of Tensors, IndexedSlices, SparseTensors and
      TensorArrays.

  Returns:
    A list of tensors.
  """
  # TODO(akshayka): Support `SparseTensor` in a similar fashion.
  flat_sequence = nest.flatten(sequence)
  outputs = []
  for item in flat_sequence:
    if isinstance(item, ops.IndexedSlices):
      if item.dense_shape is not None:
        outputs.extend([item.values, item.indices, item.dense_shape])
      else:
        outputs.extend([item.values, item.indices])
    elif isinstance(item, sparse_tensor.SparseTensor):
      outputs.extend([item.indices, item.values, item.dense_shape])
    elif isinstance(item, tensor_array_ops.TensorArray):
      outputs.append(item.flow)
    else:
      outputs.append(item)
  return outputs


def pack_sequence_as(structure, flat_sequence):
  """Like `nest.pack_sequence_as` but also packs other Tensor-like objects.

  Args:
    structure: The structure to pack into. May contain Tensors, IndexedSlices,
      TensorArrays or SparseTensors.
    flat_sequence: An iterable containing tensors.

  Returns:
    A nested structure.

  Raises:
    AssertionError if `structure` and `flat_sequence` are not compatible.
  """
  flattened_structure = nest.flatten(structure)
  flat_sequence_with_slices_and_tas = []
  index = 0
  for t in flattened_structure:
    if isinstance(t, ops.IndexedSlices):
      if t.dense_shape is not None:
        flat_sequence_with_slices_and_tas.append(
            ops.IndexedSlices(*flat_sequence[index:index + 3]))
        index += 3
      else:
        flat_sequence_with_slices_and_tas.append(
            ops.IndexedSlices(*flat_sequence[index:index + 2]))
        index += 2
    elif isinstance(t, sparse_tensor.SparseTensor):
      flat_sequence_with_slices_and_tas.append(
          sparse_tensor.SparseTensor(*flat_sequence[index:index + 3]))
      index += 3
    elif isinstance(t, tensor_array_ops.TensorArray):
      flow = flat_sequence[index]
      ta = tensor_array_ops.build_ta_with_new_flow(t, flow)
      flat_sequence_with_slices_and_tas.append(ta)
      index += 1
    else:
      flat_sequence_with_slices_and_tas.append(flat_sequence[index])
      index += 1
  assert len(flattened_structure) == len(flat_sequence_with_slices_and_tas)
  return nest.pack_sequence_as(structure, flat_sequence_with_slices_and_tas)


def _create_substitute_placeholder(value, name=None, dtype=None):
  """Creates a placeholder for `value` and propagates shape info to it."""
  # Note: setting ops.control_dependencies(None) ensures we always put
  # capturing placeholders outside of any control flow context.
  with ops.control_dependencies(None):
    placeholder = graph_placeholder(
        dtype=dtype or value.dtype, shape=value.shape, name=name)
  custom_gradient.copy_handle_data(value, placeholder)
  return placeholder


def _get_defun_inputs_from_args(args, names):
  """Maps Python function positional args to graph-construction inputs."""
  return _get_defun_inputs(args, names, structure=args)


def _get_defun_inputs(flat_args, names, structure):
  """Maps python function args to graph-construction inputs.

  Args:
    flat_args: A flat list of user-specified arguments.
    names: A list of strings with user-specified argument names, same length as
      `flat_args`. May be `None`, in which case a generic name is used.
    structure: The original argument list or dictionary.

  Returns:
    Placeholders with the same structure as `structure`.
  """
  func_graph = ops.get_default_graph()
  function_inputs = []
  if names is None:
    names = [None] * len(flat_args)
  for arg_value, name in zip(flat_args, names):
    for arg in nest.flatten(arg_value):
      if isinstance(arg, (ops.Tensor, tensor_spec.TensorSpec)):
        if isinstance(arg, tensor_spec.TensorSpec) and arg.name:
          requested_name = arg.name
        else:
          requested_name = name
        placeholder = graph_placeholder(
            arg.dtype, arg.shape,
            name=requested_name)
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
        function_inputs.append(arg)
  return nest.pack_sequence_as(structure, function_inputs)


def _get_defun_inputs_from_kwargs(kwargs):
  """Maps Python function keyword args to graph-construction inputs."""
  if kwargs:
    names, flat_args = zip(*sorted(kwargs.items()))
  else:
    names = []
    flat_args = []
  return _get_defun_inputs(flat_args, names, structure=kwargs)


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
