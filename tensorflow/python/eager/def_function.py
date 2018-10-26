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
"""Prototype decorator for defining graph functions with eager semantics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import weakref

from tensorflow.python.eager import context
from tensorflow.python.eager import function as function_lib
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.training.checkpointable import base as checkpointable
from tensorflow.python.util import tf_decorator


def _graph_inputs(op):
  return [x.op for x in op.inputs] + list(op.control_inputs)


def _lift_to_graph(init_tensor, graph):
  """Copies the tensor and all its inputs recursively to the outer graph."""
  # Check that the initializer does not depend on any placeholders.
  visited_ops = set([])
  ops_to_visit = [init_tensor.op]
  op_outputs = collections.defaultdict(set)
  while ops_to_visit:
    op = ops_to_visit.pop()
    if op in visited_ops:
      continue
    visited_ops.add(op)
    # TODO(apassos) distinguish arg placeholders, capture placeholders,
    # and placeholders the user might directly use to initialize
    # variables.
    if op.type == "Placeholder":
      raise ValueError(
          "Unable to lift tensor", init_tensor,
          "because it depends transitively on placeholder ", op)
    for inp in _graph_inputs(op):
      op_outputs[inp].add(op)
      if inp not in visited_ops:
        ops_to_visit.append(inp)
  # Topologically sort the nodes we've extracted. Now we know how many of their
  # outputs are part of this subgraph.
  ops_to_copy = []
  marked_ops = set([])
  ops_to_visit = [init_tensor.op]
  while ops_to_visit:
    op = ops_to_visit.pop()
    if op in marked_ops:
      continue
    marked_ops.add(op)
    ops_to_copy.append(op)
    for inp in _graph_inputs(op):
      if all(x in marked_ops for x in op_outputs[inp]):
        ops_to_visit.append(inp)
  assert len(ops_to_copy) == len(visited_ops)
  # ops_to_copy now holds a reverse topologically sorted list of ops which
  # ends in the initializer. We copy those to the outermost graph and
  # build the initialization op there.
  with graph.as_default():
    op_map = {}
    for op in reversed(ops_to_copy):
      copied_inputs = [op_map[x] for x in op.inputs]
      copied_control_inputs = [op_map[x] for x in op.control_inputs]
      with ops.control_dependencies(copied_control_inputs):
        copied_op = graph.create_op(
            op.type, copied_inputs, [x.dtype for x in op.outputs],
            attrs=op.node_def.attr)
      op_map[op] = copied_op
      for i, o in enumerate(op.outputs):
        op_map[o] = copied_op.outputs[i]
    return op_map[init_tensor]


class UnliftedInitializerVariable(resource_variable_ops.ResourceVariable):
  """Variable which does not lift its initializer out of function context.

  Instances of this variable, when created, build a graph which runs their
  initializer inside a tf.cond(is_initialized) block.

  This can only be created inside a defun called from (eventually) eager
  mode. That is, non-function-building graphs are not supported.
  """

  def __init__(self,  # pylint: disable=super-init-not-called
               initial_value=None,
               trainable=None,
               caching_device=None,
               name=None,
               dtype=None,
               constraint=None,
               **unused_kwargs):
    """Creates a variable.

    Args:
      initial_value: A `Tensor`, or Python object convertible to a `Tensor`,
        which is the initial value for the Variable. The initial value must have
        a shape specified unless `validate_shape` is set to False. Can also be a
        callable with no argument that returns the initial value when called.
        (Note that initializer functions from init_ops.py must first be bound
         to a shape before being used here.)
      trainable: If `True`, GradientTapes automatically watch uses of this
        Variable.
      caching_device: Optional device string or function describing where the
        Variable should be cached for reading.  Defaults to the Variable's
        device.  If not `None`, caches on another device.  Typical use is to
        cache on the device where the Ops using the Variable reside, to
        deduplicate copying through `Switch` and other conditional statements.
      name: Optional name for the variable. Defaults to `'Variable'` and gets
        uniquified automatically.
      dtype: If set, initial_value will be converted to the given type.
        If None, either the datatype will be kept (if initial_value is
       a Tensor) or float32 will be used (if it is a Python object convertible
       to a Tensor).
      constraint: An optional projection function to be applied to the variable
        after being updated by an `Optimizer` (e.g. used to implement norm
        constraints or value constraints for layer weights). The function must
        take as input the unprojected Tensor representing the value of the
        variable and return the Tensor for the projected value
        (which must have the same shape). Constraints are not safe to
        use when doing asynchronous distributed training.

    Raises:
      ValueError: If the initial value is not specified, or does not have a
        shape and `validate_shape` is `True`.
      RuntimeError: If called outside of a function definition.
    """
    if context.executing_eagerly():
      # If we've been init_scope()d out of the function definition nothing to do
      # here; we can't really do the capturing or conditional logic.
      resource_variable_ops.ResourceVariable.__init__(
          self, initial_value=initial_value, trainable=trainable,
          caching_device=caching_device, name=name, dtype=dtype,
          constraint=constraint)
      return
    with ops.init_scope():
      self._in_graph_mode = not context.executing_eagerly()
    if initial_value is None:
      raise ValueError("initial_value must be specified.")
    init_from_fn = callable(initial_value)

    if constraint is not None and not callable(constraint):
      raise ValueError("The `constraint` argument must be a callable.")

    if isinstance(initial_value, checkpointable.CheckpointInitialValue):
      self._maybe_initialize_checkpointable()
      self._update_uid = initial_value.checkpoint_position.restore_uid
      initial_value = initial_value.wrapped_value

    if trainable is None:
      trainable = True
    self._trainable = trainable
    self._save_slice_info = None
    self._initial_value = None
    self._initializer_op = None
    self._is_initialized_op = None
    self._graph_element = None
    self._cached_value = None
    # Store the graph key so optimizers know how to only retrieve variables from
    # this graph. Guaranteed to be the same as the eager graph_key.
    self._graph_key = ops.get_default_graph()._graph_key  # pylint: disable=protected-access
    with ops.name_scope(name, "Variable", []
                        if init_from_fn else [initial_value]) as name:
      # pylint: disable=protected-access
      with ops.init_scope():
        shared_name = ops._name_from_scope_name(name)
        shared_name = "%s_%d" % (shared_name, ops.uid())
      with ops.name_scope("Initializer"), ops.device(None):
        initial_value = ops.convert_to_tensor(
            initial_value() if init_from_fn else initial_value,
            name="initial_value", dtype=dtype)
      with ops.init_scope():
        self._handle = resource_variable_ops.eager_safe_variable_handle(
            shape=initial_value.get_shape(),
            dtype=initial_value.dtype.base_dtype,
            shared_name=shared_name,
            name=name,
            graph_mode=self._in_graph_mode)
      self._shape = initial_value.shape
      self._unique_id = shared_name
      self._handle_name = shared_name + ":0"
      self._dtype = initial_value.dtype.base_dtype
      self._constraint = constraint
      assert initial_value is not None
      if self._in_graph_mode:
        with ops.init_scope():
          outer_graph = ops.get_default_graph()
        lifted_initializer = _lift_to_graph(initial_value, outer_graph)
        with ops.init_scope():
          self._initial_value = lifted_initializer
          with ops.name_scope("IsInitialized"):
            self._is_initialized_op = (
                resource_variable_ops.var_is_initialized_op(self._handle))
          if initial_value is not None:
            with ops.name_scope("Assign") as n, ops.colocate_with(self._handle):
              self._initializer_op = resource_variable_ops.assign_variable_op(
                  self._handle, lifted_initializer, name=n)
          with ops.name_scope("Read"), ops.colocate_with(self._handle):
            # Manually assign reads to the handle's device to avoid log
            # messages.
            with ops.device(self._handle.device):
              value = self._read_variable_op()
            self._graph_element = value
          ops.add_to_collection(ops.GraphKeys.GLOBAL_VARIABLES, self)
      else:
        def assign_fn():
          with ops.name_scope("Assign") as n, ops.colocate_with(self._handle):
            resource_variable_ops.assign_variable_op(
                self._handle,
                initial_value,
                name=n)
            # Returning values to keep tf.cond happy.
          return ops.convert_to_tensor(1)
        def not_assign_fn():
          return ops.convert_to_tensor(0)
        # Note: this cond is always guaranteed to run because we're inside a
        # defun which will insert automatic control dependencies.
        control_flow_ops.cond(
            resource_variable_ops.var_is_initialized_op(self._handle),
            not_assign_fn, assign_fn)

    # After the handle has been created, set up a way to clean it up when
    # executing eagerly. We'll hold the only reference to the deleter, so that
    # when this object is garbage collected the deleter will be too. This
    # means ResourceVariables can be part of reference cycles without those
    # cycles being uncollectable.
    if not self._in_graph_mode:
      self._handle_deleter = resource_variable_ops.EagerResourceDeleter(
          handle=self._handle, handle_device=self._handle.device)
    self._cached_shape_as_list = None


class PolymorphicFunction(object):
  """Wrapper class for the graph functions defined for a Python function.

  See the documentation for `tf.function` for more information on the semantics
  of defined functions.

  PolymorphicFunction is thread-compatible.
  """

  def __init__(self,
               python_function,
               name,
               input_signature=None,
               autograph=False,
               experimental_autograph_options=None):
    """Initializes a polymorphic function.

    Args:
      python_function: the function to be wrapped.
      name: the name given to it.
      input_signature: a possibly nested sequence of `TensorSpec` objects
        specifying the input signature of this function. If `None`, a separate
        function is instantiated for each inferred input signature.
      autograph: whether `python_function` should be converted to graph mode.
        See https://www.tensorflow.org/guide/autograph for more information.
      experimental_autograph_options: optional tuple of
        tensorflow.autograph.Feature values. Allows enabling additional
        conversion options when autograph is set to True.

    Raises:
      ValueError: if `input_signature` is not None and the `python_function`'s
        argspec has keyword arguments.
    """
    self._python_function = python_function
    self._input_signature = input_signature
    self._autograph = autograph
    self._experimental_autograph_options = experimental_autograph_options
    if self._experimental_autograph_options is not None:
      raise NotImplementedError()
    self._created_variables = None
    self._stateful_fn = None
    self._descriptor_cache = weakref.WeakKeyDictionary()
    self._name = name

  def _defun_with_scope(self, scope):
    """Creates a defun wrapped inside a variable creator scope."""

    fn = self._python_function

    def wrapped_fn(*args, **kwds):
      with variable_scope.variable_creator_scope(scope):
        return fn(*args, **kwds)

    # TODO(mdan): Pipe self._experimental_autograph_options through.
    return function_lib.defun(
        tf_decorator.make_decorator(fn, wrapped_fn),
        input_signature=self._input_signature,
        experimental_autograph=self._autograph)

  def _initialize(self, args, kwds):
    """Initializes, on the first call."""

    self._created_variables = []

    def variable_capturing_scope(unused_next_creator, **kwds):
      """Creates UnliftedInitializerVariables and saves references to them."""
      v = UnliftedInitializerVariable(**kwds)
      self._created_variables.append(weakref.ref(v))
      return v

    self._stateful_fn = self._defun_with_scope(variable_capturing_scope)
    self._stateful_fn._name = self._name  # pylint: disable=protected-access

    # Force the definition of the function for these arguments
    self._concrete_stateful_fn = (
        self._stateful_fn._get_concrete_function_internal(*args, **kwds))  # pylint: disable=protected-access

    def invalid_creator_scope(*unused_args, **unused_kwds):
      """Disables variable creation."""
      raise ValueError(
          "tf.function-decorated function tried to create "
          "variables on non-first call.")

    self._stateless_fn = self._defun_with_scope(invalid_creator_scope)
    self._stateless_fn._name = self._name  # pylint: disable=protected-access
    return self._stateful_fn._canonicalize_function_inputs(*args, **kwds)  # pylint: disable=protected-access

  def __call__(self, *args, **kwds):
    """Calls the graph function."""
    if self._created_variables:
      # In this case we have created variables on the first call, so we run the
      # defunned version which is guaranteed to never create variables.
      return self._stateless_fn(*args, **kwds)  # pylint: disable=not-callable
    elif self._stateful_fn is not None:
      # In this case we have not created variables on the first call. So we can
      # run the first trace but we should fail if variables are created.
      results = self._stateful_fn(*args, **kwds)
      if self._created_variables:
        raise ValueError("Creating variables on a non-first call to a function"
                         " decorated with tf.function.")
      return results

    canon_args, canon_kwds = self._initialize(args, kwds)

    if not self._created_variables:
      # If we did not create any variables the trace we have is good enough.
      return self._concrete_stateful_fn._filtered_call(canon_args, canon_kwds)  # pylint: disable=protected-access

    def fn_with_cond(*inner_args, **inner_kwds):
      """Conditionally runs initialization if it's needed."""
      condition = True
      for wr in self._created_variables:
        variable = wr()
        if variable is None:
          raise ValueError(
              "Variable created in a tf.function garbage-collected. Code needs"
              " to keep python references to variables created in a"
              " tf.function.")
        condition = math_ops.logical_and(
            condition, resource_variable_ops.var_is_initialized_op(
                variable.handle))
      # We want to call stateless_fn if possible because it avoids recomputing
      # potentially expensive initializers.
      return control_flow_ops.cond(
          condition,
          lambda: self._stateless_fn(*inner_args, **inner_kwds),
          functools.partial(self._concrete_stateful_fn._filtered_call,  # pylint: disable=protected-access
                            inner_args, inner_kwds))

    return function_lib.defun(fn_with_cond)(*canon_args, **canon_kwds)

  @property
  def python_function(self):
    """The python function wrapped in this tf.function."""
    return self._python_function

  def get_concrete_function(self, *args, **kwargs):
    """Returns a `Function` object specialized to inputs and execution context.

    If this `PolymorphicFunction` was created with an `input_signature`, `args`
    and `kwargs` may be omitted. With an input signature there is only one
    concrete function associated with this `PolymorphicFunction`.

    If there is no fixed `input_signature` associated with this
    `PolymorphicFunction`, positional and keyword arguments to
    `get_concrete_function` follow the same rules as input signature
    specification, with `tf.TensorSpec` objects describing `tf.Tensor`s which
    will be passed to the concrete function.

    Each `tf.Tensor` argument to the concrete function must have a unique name,
    either because it is the only one associated with a named argument of the
    Python function or because an explicit `name=` was passed to its
    `tf.TensorSpec` object. These names become the argument names for the
    concrete function.

    Arguments to the concrete function may always be specified as keyword
    arguments, naming the Tensor input. Positional arguments may be used instead
    when each preceding argument to the Python function is a Tensor.

    ```python
    @tf.function
    def f(x):
      return x

    f_concrete = f.get_concrete_function(tf.TensorSpec([], tf.float64))
    f_concrete(tf.constant(1.))
    f_concrete(x=tf.constant(1.))
    ```

    Nested structures containing Tensors may be specified when retrieving
    concrete functions. Structures with multiple Tensors are expanded into
    multiple arguments of the concrete function. Since multiple concrete
    function arguments are associated with one argument to the original
    function, these Tensors must be named explicitly. Tensors in nested
    structures may not be passed using positional arguments when calling the
    concrete function.

    ```python
    f_concrete2 = f.get_concrete_function(
        (tf.TensorSpec(None, tf.float64, name="first"),
         tf.TensorSpec([], tf.float32, name="second")))
    # Keyword arguments are required when identifying Tensors in nested
    # structures.
    f_concrete2(first=tf.constant([1.]), second=tf.constant(0.))
    ```

    Functions with fixed input signatures have only one concrete function
    associated with them, which can be retrieved without specifying any
    arguments. As before Tensors must have unique names, either inferred from
    the argument names in the original Python function or specified
    explicitly.

    ```python
    @tf.function(input_signature=(tf.TensorSpec(None, tf.float32)))
    def f_sig(y):
      return y

    f_sig_concrete = f.get_concrete_function()
    f_sig_concrete(tf.constant(1.))
    f_sig_concrete(y=tf.constant(1.))
    ```

    Args:
      *args: inputs to specialize on.
      **kwargs: inputs to specialize on.

    Returns:
      A TensorFlow function which takes exactly one `tf.Tensor` per argument.

    Raises:
      ValueError: if this object has not yet been called on concrete values.
    """
    # TODO(apassos) figure out how to handle this case (what should we return
    # here?)
    if self._stateful_fn is None:
      raise ValueError(
          "Call this function with concrete values before asking for a"
          " concrete function. Calling the function will ensure that, in"
          " case this function creates variables, that those are properly"
          " initialized.")
    if self._created_variables:
      # In this case we have created variables on the first call, so we run the
      # defunned version which is guaranteed to never create variables.
      return self._stateless_fn.get_concrete_function(*args, **kwargs)
    elif self._stateful_fn is not None:
      # In this case we have not created variables on the first call. So we can
      # run the first trace but we should fail if variables are created.
      concrete = self._stateful_fn.get_concrete_function(*args, **kwargs)
      if self._created_variables:
        raise ValueError("Creating variables on a non-first call to a function"
                         " decorated with tf.function.")
      return concrete

  def __get__(self, instance, owner):
    """Makes it possible to defun instance methods."""
    del owner
    # `instance` here is the instance that this `PolymorphicFunction` was
    # accessed through; e.g., for
    #
    #   class Foo(object):
    #
    #     @function.defun
    #     def bar(self):
    #       ...
    #
    #   foo = Foo()
    #   foo.bar()  # `foo.bar` is a `PolymorphicFunction` instance
    #
    # then `instance` will be `foo` (and `owner` will be `Foo`).  We create a
    # new instance of PolymorphicFunction here to allow different instances each
    # to create variables once, thereby allowing methods to be decorated with
    # tf.function. Keeps a cache to avoid retracing the function every time the
    # descriptor is accessed.
    if instance not in self._descriptor_cache:
      if instance is None:
        return self
      self._descriptor_cache[instance] = (
          function_lib.class_method_to_instance_method(self, instance))
    return self._descriptor_cache[instance]


def function(func=None,
             input_signature=None,
             autograph=False,
             experimental_autograph_options=None):
  """Defines a function as per the "functions, not sessions" document."""
  if input_signature is not None:
    function_lib.validate_signature(input_signature)

  def decorated(inner_function):
    try:
      name = inner_function.__name__
    except AttributeError:
      name = "function"
    return tf_decorator.make_decorator(
        inner_function,
        PolymorphicFunction(
            inner_function,
            name,
            input_signature=input_signature,
            autograph=autograph,
            experimental_autograph_options=experimental_autograph_options))

  # This code path is for the `foo = tf.function(foo, ...)` use case
  if func is not None:
    return decorated(func)

  # This code path is for the
  #
  # @tf.function(...)
  # def foo(...):
  #    ...
  #
  # use case, which is equivalent to `foo = tf.function(...)(foo)`
  return decorated
