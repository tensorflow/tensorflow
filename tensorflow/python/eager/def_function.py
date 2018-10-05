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
"""Prototype decorator for defining graph-mode functions with eager semantics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import context
from tensorflow.python.eager import function
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.training.checkpointable import base as checkpointable


class UnliftedInitializerVariable(resource_variable_ops.ResourceVariable):
  """Variable which does not lift its initializer out of function context.

  Instances of this variable, when created, build a graph which runs their
  initializer inside a tf.cond(is_initialized) block.

  This can only be created inside a defun called from (eventually) eager
  mode. That is, non-function-building graphs are not supported.
  """

  def __init__(self,  # pylint: disable=super-init-not-called
               initial_value=None,
               trainable=True,
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
      raise RuntimeError(
          "UnliftedInitializerVariable should not be created "
          "outside of functions.")
    with ops.init_scope():
      if not context.executing_eagerly():
        raise RuntimeError(
            "UnliftedInitializerVariable does not support legacy graph mode.")
    self._in_graph_mode = False
    if initial_value is None:
      raise ValueError("initial_value must be specified.")
    init_from_fn = callable(initial_value)

    if constraint is not None and not callable(constraint):
      raise ValueError("The `constraint` argument must be a callable.")

    if isinstance(initial_value, checkpointable.CheckpointInitialValue):
      self._maybe_initialize_checkpointable()
      self._update_uid = initial_value.checkpoint_position.restore_uid
      initial_value = initial_value.wrapped_value

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
        assert context.executing_eagerly()
        shared_name = ops._name_from_scope_name(name)
        shared_name = "%s_%d" % (shared_name, ops.uid())
      # Use attr_scope and device(None) to simulate the behavior of
      # colocate_with when the variable we want to colocate with doesn't
      # yet exist.
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
            graph_mode=False)
      self._shape = initial_value.shape
      self._unique_id = shared_name
      self._handle_name = shared_name + ":0"
      self._dtype = initial_value.dtype.base_dtype
      self._constraint = constraint
      assert initial_value is not None
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
      # Note: this cond is always guaranteed to run because we're inside a defun
      # which will insert automatic control dependencies.
      control_flow_ops.cond(
          resource_variable_ops.var_is_initialized_op(self._handle),
          not_assign_fn, assign_fn)

    # After the handle has been created, set up a way to clean it up when
    # executing eagerly. We'll hold the only reference to the deleter, so that
    # when this object is garbage collected the deleter will be too. This
    # means ResourceVariables can be part of reference cycles without those
    # cycles being uncollectable.
    self._handle_deleter = resource_variable_ops.EagerResourceDeleter(
        handle=self._handle, handle_device=self._handle.device)
    self._cached_shape_as_list = None


def _defun_with_scope(scope, fn):

  def wrapped_fn(*args, **kwds):
    with variable_scope.variable_creator_scope(scope):
      return fn(*args, **kwds)

  return function.defun(wrapped_fn)


def def_function(fn):
  """Defines a function as per the "functions, not sessions" document."""

  # Wrapping the values in lists to bypass python's lack of way to mutate
  # symbols from an outer scope.
  first_call = [True]
  function_to_call = []

  # TODO(apassos) represent this as an object and not as a closure.
  def decorated_fn(*args, **kwds):
    """Graph function for fn."""
    if not first_call[0]:
      return function_to_call[0](*args, **kwds)

    first_call[0] = False
    created_variables = []

    def variable_creator_scope(unused_next_creator, **kwds):
      """Creates UnliftedInitializerVariables and saves references to them."""
      v = UnliftedInitializerVariable(**kwds)
      created_variables.append(v)
      return v

    first_graph_function = _defun_with_scope(variable_creator_scope, fn)

    # Force the definition of the function for these arguments
    first_concrete = first_graph_function.get_concrete_function(*args, **kwds)

    def invalid_creator_scope(*unused_args, **unused_kwds):
      """Disables variable creation."""
      raise ValueError(
          "def_function-decorated function tried to create "
          "variables on second call.")

    second_graph_function = _defun_with_scope(invalid_creator_scope, fn)

    function_to_call.append(second_graph_function)
    if not created_variables:
      # Note: this retracing might be unnecessary, but running the function
      # forever in the scope which disallows variable creation is safer than not
      # doing so.
      return second_graph_function(*args, **kwds)

    def fn_with_cond(*inner_args, **inner_kwds):
      """Conditionally runs initialization if it's needed."""
      condition = True
      for variable in created_variables:
        condition = condition and resource_variable_ops.var_is_initialized_op(
            variable.handle)
      # We want to call second_graph_function if possible because it avoids
      # recomputing potentially expensive initializers.
      return control_flow_ops.cond(
          condition,
          lambda: second_graph_function(*inner_args, **inner_kwds),
          lambda: first_concrete(*inner_args, **inner_kwds))

    return function.defun(fn_with_cond)(*args, **kwds)

  return decorated_fn
