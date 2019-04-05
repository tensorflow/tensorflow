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
"""EXPERIMENTAL utilities for parameter server training with eager execution.

Note: this should eventually be merged with the distribution strategy for
ParameterServer.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import time

from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.training.tracking import base as trackable


def _eager_safe_variable_handle(shape, dtype, shared_name, name, graph_mode):
  """Creates a variable handle with information to do shape inference."""
  container = ops.get_default_graph()._container  # pylint: disable=protected-access
  if container is None:
    container = ""
  handle = resource_variable_ops.var_handle_op(shape=shape, dtype=dtype,
                                               shared_name=shared_name,
                                               name=name,
                                               container=container)
  if graph_mode:
    return handle

  with context.graph_mode(), ops.Graph().as_default() as graph:
    h = resource_variable_ops.var_handle_op(shape=shape, dtype=dtype,
                                            shared_name=shared_name,
                                            name=name,
                                            container=container)

    # Tensor._handle_data contains information for the shape-inference code to
    # know the shape and dtype of the variable pointed to by a handle. Since
    # shape inference doesn't run in eager mode we copy this data here for when
    # the handle is captured by an eager mode function.
    # pylint: disable=protected-access
    handle._handle_data = resource_variable_ops.get_resource_handle_data(h)
    # pylint: enable=protected-access
  # Clean up op->graph->op reference cycles.
  ops.dismantle_graph(graph)
  return handle


class SharedVariable(resource_variable_ops.ResourceVariable):
  """Experimental Variable designed for parameter server training.

  A SharedVariable has a name and two instances of SharedVariable with the
  same name will have the same value, even if they are in different Sessions,
  as long as they are placed on the same device.

  The storage associated with SharedVariables is also not deleted when they go
  out of scope.
  """

  def __init__(self,  # pylint: disable=super-init-not-called
               initial_value=None,
               trainable=True,
               name=None,
               dtype=None,
               constraint=None,
               initialize=True,
               **unused_kwargs):
    """Creates a variable.

    Args:
      initial_value: A `Tensor`, or Python object convertible to a `Tensor`,
        which is the initial value for the Variable. The initial value must have
        a shape specified unless `validate_shape` is set to False. Can also be a
        callable with no argument that returns the initial value when called.
        (Note that initializer functions from init_ops.py must first be bound
         to a shape before being used here.)
      trainable: If `True`, automatically watches this variable on GradientTape
        whenever it's used.
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
      initialize: if True, runs initialization in eager execution; leaves the
        variable uninitialized otherwise.

    Raises:
      ValueError: If the initial value is not specified, or does not have a
        shape and `validate_shape` is `True`.
    """
    if initial_value is None:
      raise ValueError("initial_value must be specified.")
    init_from_fn = callable(initial_value)

    if isinstance(initial_value, ops.Tensor) and hasattr(
        initial_value, "graph") and initial_value.graph.building_function:
      raise ValueError("Tensor-typed variable initializers must either be "
                       "wrapped in an init_scope or callable "
                       "(e.g., `tf.Variable(lambda : "
                       "tf.truncated_normal([10, 40]))`) when building "
                       "functions. Please file a feature request if this "
                       "restriction inconveniences you.")

    if constraint is not None and not callable(constraint):
      raise ValueError("The `constraint` argument must be a callable.")

    if isinstance(initial_value, trackable.CheckpointInitialValue):
      self._maybe_initialize_trackable()
      self._update_uid = initial_value.checkpoint_position.restore_uid
      initial_value = initial_value.wrapped_value

    self._trainable = trainable
    self._save_slice_info = None
    # Store the graph key so optimizers know how to only retrieve variables from
    # this graph.
    self._graph_key = ops.get_default_graph()._graph_key  # pylint: disable=protected-access
    with ops.init_scope():
      self._in_graph_mode = not context.executing_eagerly()
      with ops.name_scope(name, "Variable", []
                          if init_from_fn else [initial_value]) as name:
        # pylint: disable=protected-access
        handle_name = ops.name_from_scope_name(name)
        shared_name = handle_name
        if init_from_fn:
          # Use attr_scope and device(None) to simulate the behavior of
          # colocate_with when the variable we want to colocate with doesn't
          # yet exist.
          if self._in_graph_mode:
            with ops.name_scope("Initializer"), ops.device(None):
              initial_value = ops.convert_to_tensor(
                  initial_value(), name="initial_value", dtype=dtype)
            self._handle = _eager_safe_variable_handle(
                shape=initial_value.get_shape(),
                dtype=initial_value.dtype.base_dtype,
                shared_name=shared_name,
                name=name,
                graph_mode=self._in_graph_mode)
            self._shape = initial_value.get_shape()
          else:
            initial_value = initial_value()
            with ops.name_scope("Initializer"):
              initial_value = ops.convert_to_tensor(
                  initial_value, name="initial_value", dtype=dtype)
            self._handle = _eager_safe_variable_handle(
                shape=initial_value.get_shape(),
                dtype=initial_value.dtype.base_dtype,
                shared_name=shared_name,
                name=name,
                graph_mode=False)
            self._shape = initial_value.get_shape()
        # pylint: enable=protected-access

        # Or get the initial value from a Tensor or Python object.
        else:
          with ops.name_scope("Initializer"):
            initial_value = ops.convert_to_tensor(
                initial_value, name="initial_value", dtype=dtype)
          # pylint: disable=protected-access
          if (self._in_graph_mode and initial_value is not None and
              initial_value.op._get_control_flow_context() is not None):
            raise ValueError(
                "Initializer for variable %s is from inside a control-flow "
                "construct, such as a loop or conditional. When creating a "
                "variable inside a loop or conditional, use a lambda as the "
                "initializer." % name)
          # pylint: enable=protected-access
          self._handle = _eager_safe_variable_handle(
              shape=initial_value.get_shape(),
              dtype=initial_value.dtype.base_dtype,
              shared_name=shared_name,
              name=name,
              graph_mode=self._in_graph_mode)
          self._shape = initial_value.get_shape()

        self._unique_id = shared_name
        self._initial_value = initial_value if self._in_graph_mode else None
        self._handle_name = handle_name + ":0"
        self._dtype = initial_value.dtype.base_dtype
        self._constraint = constraint

        if self._in_graph_mode:
          with ops.name_scope("IsInitialized"):
            self._is_initialized_op = (
                resource_variable_ops.var_is_initialized_op(self._handle))
          if initial_value is not None:
            with ops.name_scope("Assign") as n, ops.colocate_with(self._handle):
              self._initializer_op = (
                  resource_variable_ops.assign_variable_op(
                      self._handle,
                      self._try_guard_against_uninitialized_dependencies(
                          initial_value),
                      name=n))
          with ops.name_scope("Read"), ops.colocate_with(self._handle):
            # Manually assign reads to the handle's device to avoid log
            # messages.
            with ops.device(self._handle.device):
              value = self._read_variable_op()
            self._graph_element = value
            self._cached_value = None
        else:
          if initialize:
            resource_variable_ops.assign_variable_op(self._handle,
                                                     initial_value)
          self._is_initialized_op = None
          self._initializer_op = None
          self._graph_element = None
          self._cached_value = None

    self._handle_deleter = None
    self._cached_shape_as_list = None


@contextlib.contextmanager
def parameter_server_scope(is_chief, ps_job_name, num_ps_tasks):
  """Strategy to use parameter servers in eager.

  Creates SharedVariable objects for variables created in this scope. These
  SharedVariable objects will be placed round-robin on the parameter servers
  specified by the ps_job_name and num_ps_tasks arguments.

  To use parameter servers you need only to wrap your model initialization in
  this scope:

  ```
  with tf.contrib.eager.parameter_server_scope(
      is_chief, ps_job_name, num_ps_tasks):
    my_model = tf.keras.Sequential([...])  # Or
    input = tf.keras.Input(...)
    ....
    my_model = tf.keras.Model(input, output)
  my_model.compile(...)
  # or other usages of the model.
  ```

  Args:
    is_chief: Boolean. Whether this worker is responsible for initializing
      variables.
    ps_job_name: The name of the ps job in this cluster.
    num_ps_tasks: The number of ps tasks to use.

  Yields:
    a context manager.
  """
  # Note: capturing in a list to allow assignment.
  ps_index = [0]

  def variable_creator_scope(unused_next_creator, **kwargs):
    kwargs["initialize"] = is_chief
    with ops.device(
        "/job:%s/task:%s" % (ps_job_name, ps_index[0] % num_ps_tasks)):
      ps_index[0] += 1
      v = SharedVariable(**kwargs)
      if not is_chief:
        while not resource_variable_ops.var_is_initialized_op(v.handle):
          time.sleep(10)
      return v

  with variable_scope.variable_creator_scope(variable_creator_scope):
    yield
