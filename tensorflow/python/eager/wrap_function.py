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
# pylint: disable=unidiomatic-typecheck
"""Prototype decorator for defining legacy-graph-mode functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import weakref

from tensorflow.python.eager import def_function
from tensorflow.python.eager import function
from tensorflow.python.eager import lift_to_graph
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.training.tracking import data_structures
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export


class VariableHolder(object):
  """Holds variables for a python function."""

  def __init__(self, fn=None, share_variables=False):
    self._fn = fn

    self._share_variables = share_variables
    self._variables_by_name = data_structures.Mapping()

  @property
  def variables(self):
    return self._variables_by_name

  def variable_creator_scope(self, next_creator, **kwargs):
    """Creates variables & adds them to collections to match legacy code."""
    collections = kwargs.pop("collections", None)
    v = None

    # Get expected variable name.
    with ops.name_scope(kwargs.get("name", None), "Variable") as name:
      variable_name = ops.name_from_scope_name(name)
      kwargs["name"] = name

    if self._share_variables:
      v = self._variables_by_name.get(variable_name, None)

    if v is None:
      v = next_creator(**kwargs)
      self._variables_by_name[variable_name] = v

    if collections is None:
      collections = [ops.GraphKeys.GLOBAL_VARIABLES]
    if v.trainable and ops.GraphKeys.TRAINABLE_VARIABLES not in collections:
      collections = list(collections) + [ops.GraphKeys.TRAINABLE_VARIABLES]

    ops.add_to_collections(collections, v)

    return v

  def __call__(self, *args, **kwargs):
    return self.call_with_variable_creator_scope(self._fn)(*args, **kwargs)

  def call_with_variable_creator_scope(self, fn):
    def wrapped(*args, **kwargs):
      with variable_scope.variable_creator_scope(self.variable_creator_scope):
        return fn(*args, **kwargs)
    return wrapped


# TODO(allenl): make this trackable
class WrappedFunction(function.ConcreteFunction):
  """Wraps a tf V1 piece of code in a function."""

  def __init__(self, fn_graph, variable_holder, attrs=None, signature=None):
    super(WrappedFunction, self).__init__(
        fn_graph, attrs=attrs, signature=signature)
    self._variable_holder = variable_holder
    if ops.executing_eagerly_outside_functions():
      # TODO(allenl): Make this work in 1.x?
      self._lift_unlifted_variables()

  def _lift_unlifted_variables(self):
    """Finds resource variables and lifts them into the outer context.

    When we import a GraphDef inside a wrap_function, no Python graph building
    code runs. This means we get VarHandleOps which create variable resources,
    but no corresponding Python objects. Leaving them like this works but gives
    the user no way to interact with or modify the variables outside the graph.

    This method searches for variables and lifts them out as regular variable
    objects when possible, indicating to the FuncGraph that they are captures.
    """
    with self.graph.as_default():
      collection_variables = (
          ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
          + ops.get_collection(ops.GraphKeys.LOCAL_VARIABLES))
      existing_captures = set(self.graph.internal_captures)
      lifted_variables = {}
      for old_variable in collection_variables:
        if (old_variable._in_graph_mode  # pylint: disable=protected-access
            and isinstance(old_variable,
                           resource_variable_ops.ResourceVariable)):
          if old_variable.handle in existing_captures:
            continue
          new_variable = def_function.UnliftedInitializerVariable(
              array_ops.placeholder(
                  name="unused_{}_initializer".format(old_variable.op.name),
                  shape=old_variable.shape,
                  dtype=old_variable.dtype),
              name=old_variable.op.name,
              trainable=old_variable.trainable)
          self.graph.captures[new_variable.handle] = old_variable.handle
          existing_captures.add(old_variable.handle)
          lifted_variables[old_variable] = new_variable
          # pylint: disable=protected-access
          variable_name = new_variable.name.split(":")[0]
          self._variable_holder._variables_by_name[variable_name] = new_variable
          self.graph._weak_variables.append(weakref.ref(new_variable))
          # pylint: enable=protected-access
      # Update the graph's collections, partly for the user and partly so this
      # function is idempotent when it runs again in prune() calls.
      for collection_name in [ops.GraphKeys.GLOBAL_VARIABLES,
                              ops.GraphKeys.LOCAL_VARIABLES]:
        mutable_collection = ops.get_collection_ref(collection_name)
        for index, current in enumerate(mutable_collection):
          mutable_collection[index] = lifted_variables.get(current, current)

  def prune(self, feeds, fetches, name=None, input_signature=None):
    # TODO(b/129646028): Add support for CompositeTensors.
    name = name or "pruned"
    flat_feeds, flat_fetches = nest.flatten(feeds), nest.flatten(fetches)
    for f in flat_feeds:
      if not isinstance(f, ops.Tensor):
        raise ValueError("Feeds must be tensors.")

    # Ignoring all feeds that are captures allows prune to be called
    # using wrapped_func.inputs even when it uses variables
    internal_captures = self.graph.internal_captures
    flat_feeds = [f for f in flat_feeds
                  if f not in internal_captures]

    tensor_fetches = []
    operation_fetches = []
    for f in flat_fetches:
      if isinstance(f, ops.Tensor):
        tensor_fetches.append(f)
      elif isinstance(f, ops.Operation):
        operation_fetches.append(f)
      else:
        raise ValueError("Fetches must be tensors or operations.")
    for f in flat_feeds + flat_fetches:
      if f.graph is not self._func_graph:
        raise ValueError(
            "Can only prune function whose feeds and fetches "
            "are from this graph (%s). Tensor %s from graph %s" % (
                self._func_graph, f, f.graph))
    with self._func_graph.as_default():
      pruned_graph = func_graph.FuncGraph(name)
      with ops.control_dependencies(operation_fetches):
        if tensor_fetches:
          identity_fetches = array_ops.identity_n(tensor_fetches)
          sink_tensor = identity_fetches[0]
        else:
          identity_fetches = []
          sink_tensor = array_ops.zeros([])
    lift_map = lift_to_graph.lift_to_graph(
        [sink_tensor], pruned_graph, sources=flat_feeds + internal_captures)
    for original_fetch, identity_fetch in zip(
        tensor_fetches, identity_fetches):
      lift_map[original_fetch] = lift_map[identity_fetch]
    pruned_graph.outputs.extend(
        lift_map[x] for x in flat_fetches if isinstance(x, ops.Tensor))
    pruned_graph.control_outputs.extend(
        [lift_map[operation] for operation in operation_fetches])
    for external_capture, internal_capture in self.graph.captures.items():
      pruned_graph.captures[external_capture] = lift_map[internal_capture]
    pruned_graph.inputs.extend(lift_map[x] for x in flat_feeds)
    pruned_graph.inputs.extend(pruned_graph.captures.values())

    pruned_graph.variables = self.graph.variables

    def _structured_output_mapping(fetched):
      lifted = lift_map[fetched]
      if isinstance(lifted, ops.Operation):
        return None
      return lifted

    pruned_graph.structured_outputs = nest.map_structure(
        _structured_output_mapping, fetches)
    pruned_graph.structured_input_signature = input_signature
    pruned_fn = WrappedFunction(
        pruned_graph, variable_holder=self._variable_holder)
    pruned_fn._num_positional_args = len(flat_feeds)  # pylint: disable=protected-access
    # TODO(kathywu): Enable keyword arguments if an input signature is specified
    pruned_fn._arg_keywords = []  # pylint: disable=protected-access
    return pruned_fn


def _filter_returned_ops(fn):
  """Filtering out any ops returned by function.

  Args:
    fn: a function

  Returns:
    A tuple of (
      Wrapped function that returns `None` in place of any ops,
      dict that maps the index in the flat output structure to the returned op
    )
  """
  returned_ops = {}

  def wrap_and_filter_returned_ops(*args, **kwargs):
    outputs = fn(*args, **kwargs)
    flat_outputs = nest.flatten(outputs)
    for n in range(len(flat_outputs)):
      output = flat_outputs[n]
      if isinstance(output, ops.Operation):
        returned_ops[n] = output
        flat_outputs[n] = None
    return nest.pack_sequence_as(outputs, flat_outputs)

  return wrap_and_filter_returned_ops, returned_ops


class WrappedGraph(object):
  """Class for wrapping multiple TF 1.X functions in a single graph.

  Maintains a dictionary mapping names to wrapped functions. See
  `tf.compat.v1.wrap_function` to learn more about wrapping V1 functions.

  Functions wrapped using this class have access to variables and collections
  created in other wrapped functions, using the standard TF 1.X API (
  `tf.compat.v1.get_variable` or
  `tf.compat.v1.get_default_graph().get_collection(...)`)

  Outside a function, variables and collections may be accessed using the
  `variables` and `graph` properties.

  Example:

  ```
  def add_v1(x):
    with tf.compat.v1.variable_scope('vars', reuse=tf.AUTO_REUSE):
      v = tf.compat.v1.get_variable('v', shape=[], dtype=tf.int32)
    return v + x

  def increment_var_v1(x):
    with tf.compat.v1.variable_scope('vars', reuse=tf.AUTO_REUSE):
      v = tf.compat.v1.get_variable('v', shape=[], dtype=tf.int32)
    return v.assign_add(x)

  g = WrappedGraph()
  add = g.wrap_function(add_v1, [tf.TensorSpec([], tf.int32)])
  increment_var = g.wrap_function(increment_var_v1,
                                  [tf.TensorSpec([], tf.int32)])

  assert len(g.variables) == 1
  assert g.variables[0].numpy() == 0
  increment_var(tf.constant(5))
  assert g.variables[0].numpy() == 5

  ```
  """

  def __init__(self, variable_holder=None, **kwargs):
    self._variable_holder = (
        variable_holder or VariableHolder(share_variables=True))

    name = kwargs.pop("name", "wrapped_function_graph")
    # Always start with empty collections, unless otherwise specified. Setting
    # `collections=None` will copy the collections from the outer graph.
    collections = kwargs.pop("collections", {})
    self.graph = func_graph.FuncGraph(name, collections=collections, **kwargs)

    self._wrapped_function = WrappedFunction(self.graph, self._variable_holder)
    self._functions = {}

  @property
  def functions(self):
    return self._functions

  @property
  def variables(self):
    return self._variable_holder.variables

  def wrap_function(self, fn, signature, name=None):
    """Wraps a TF 1.X function and returns an eager-compatible function.

    All functions wrapped in the same `WrappedGraph` will have access to the
    same graph (`tf.get_default_graph` to get the graph object within a
    function, or `WrappedGraph.graph` to get the graph outside a function).
    Variables created within the function will be added to the `variables` list.

    Function inputs: All inputs to the function must be tensors (nested ok),
    with their shapes and dtypes defined in the `signature` argument.

    Function outputs:

      * The 1.X function may return tensors, variables, and ops. The wrapped
        eager-compatible function will always return tensors in the same nested
        structure.
      * Variables are replaced with a tensor containing the latest read values.
      * Returned ops are executed, and replaced with None.
      * The order of op execution and variable reads in the return is
        nondeterministic. For example:

        ```
        def update_var(x):
          v = tf.Variable(0)
          op = tf.compat.v1.assign(v, x).op
          return v, op

        g = WrappedGraph()
        fn = g.wrap_function(update_var)
        read_value, _ = fn(tf.constant(3))
        print(read_value.numpy())  # could be 0 or 3
        print(g.variables[0].numpy()) # always 3
        ```

    To ensure that ops in the function are executed (e.g. ops added to the
    `tf.GraphKeys.UPDATE_OPS` collection), include them in the function returns.

    Args:
      fn: a 1.X tensorflow function.
      signature: a possibly nested sequence of `TensorSpecs` specifying the
        shapes and dtypes of the arguments.
      name: an optional string name for the function. The function will be saved
        with key `name` in the `functions` dictionary.

    Returns:
      An eager-compatible function.
    """
    return self._wrap_function(fn, signature=signature, name=name)

  def _wrap_function(
      self, fn, args=None, kwargs=None, signature=None, name=None):
    """Internal wrap function method with extended func_graph arguments."""
    fn_with_filter_and_scope, returned_ops = _filter_returned_ops(
        self._variable_holder.call_with_variable_creator_scope(fn))

    func_graph.func_graph_from_py_func(
        None,  # Name is unused.
        fn_with_filter_and_scope,
        args=args, kwargs=kwargs, signature=signature,
        add_control_dependencies=False,
        func_graph=self.graph)

    # This code relies on questional behavior from `func_graph_from_py_func`.
    # If an existing FuncGraph is passed into the `func_graph` arg, the inputs
    # and structured outputs are overwritten. Pretty sure this is a bug,
    # because structured outputs doesn't match up with the outputs...
    fn_inputs = self.graph.inputs[:-len(self.graph.captures)]

    # Return filtered ops to the flattened outputs.
    flat_fn_outputs = nest.flatten(self.graph.structured_outputs)
    for index, op in returned_ops.items():
      flat_fn_outputs[index] = op
    fn_outputs = nest.pack_sequence_as(self.graph.structured_outputs,
                                       flat_fn_outputs)

    name = name or fn.__name__
    wrapped_function = self._wrapped_function.prune(
        fn_inputs, fn_outputs, name, self.graph.structured_input_signature)
    self._functions[name] = wrapped_function
    return wrapped_function


@tf_export(v1=["wrap_function"])
def wrap_function(fn, signature, name=None):
  """Wraps the TF 1.x function fn into a graph function.

  The python function `fn` will be called once with symbolic arguments specified
  in the `signature`, traced, and turned into a graph function. Any variables
  created by `fn` will be owned by the object returned by `wrap_function`. The
  resulting graph function can be called with tensors which match the
  signature.

  ```python
  def f(x, do_add):
    v = tf.Variable(5.0)
    if do_add:
      op = v.assign_add(x)
    else:
      op = v.assign_sub(x)
    with tf.control_dependencies([op]):
      return v.read_value()

  f_add = tf.compat.v1.wrap_function(f, [tf.TensorSpec((), tf.float32), True])

  assert float(f_add(1.0)) == 6.0
  assert float(f_add(1.0)) == 7.0

  # Can call tf.compat.v1.wrap_function again to get a new trace, a new set
  # of variables, and possibly different non-template arguments.
  f_sub= tf.compat.v1.wrap_function(f, [tf.TensorSpec((), tf.float32), False])

  assert float(f_sub(1.0)) == 4.0
  assert float(f_sub(1.0)) == 3.0
  ```

  Both `tf.compat.v1.wrap_function` and `tf.function` create a callable
  TensorFlow graph. But while `tf.function` runs all stateful operations
  (e.g. `tf.print`) and sequences operations to provide the same semantics as
  eager execution, `wrap_function` is closer to the behavior of `session.run` in
  TensorFlow 1.x. It will not run any operations unless they are required to
  compute the function's outputs, either through a data dependency or a control
  dependency. Nor will it sequence operations.

  Unlike `tf.function`, `wrap_function` will only trace the Python function
  once. As with placeholders in TF 1.x, shapes and dtypes must be provided to
  `wrap_function`'s `signature` argument.

  Since it is only traced once, variables and state may be created inside the
  function and owned by the function wrapper object.

  Args:
    fn: python function to be wrapped
    signature: the placeholder and python arguments to be passed to the
      wrapped function
    name: Optional. The name of the function.

  Returns:
    the wrapped graph function.
  """
  holder = VariableHolder(fn)
  func_graph_name = "wrapped_function"
  if name is not None:
    func_graph_name = "wrapped_function_" + name
  return WrappedFunction(
      func_graph.func_graph_from_py_func(
          func_graph_name,
          holder,
          args=None, kwargs=None, signature=signature,
          add_control_dependencies=False,
          collections={}),
      variable_holder=holder,
      signature=signature)


def function_from_graph_def(graph_def, inputs, outputs):
  """Creates a ConcreteFunction from a GraphDef.

  Args:
    graph_def: A GraphDef to make a function out of.
    inputs: A Tensor name or nested structure of names in `graph_def` which
      should be inputs to the function.
    outputs: A Tensor name or nested structure of names in `graph_def` which
      should be outputs of the function.

  Returns:
    A ConcreteFunction.
  """
  def _imports_graph_def():
    importer.import_graph_def(graph_def, name="")

  wrapped_import = wrap_function(_imports_graph_def, [])
  import_graph = wrapped_import.graph
  return wrapped_import.prune(
      nest.map_structure(import_graph.as_graph_element, inputs),
      nest.map_structure(import_graph.as_graph_element, outputs))
