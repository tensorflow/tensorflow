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
"""Condition Variable object and execution logic."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# TODO(ebrevdo): Re-enable once ConditionVariable is in core.
# from tensorflow.core.protobuf import \
#    condition_variable_pb2

from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_resource_variable_ops

# Graph Keys
CONDITION_VARIABLES = "condition_variables"


class ConditionVariable(object):
  """Critical section.

  A `ConditionVariable` object is  a resource in the graph which allows multiple
  threads or workers to wait for a predetermined amount of time, or to
  synchronize while waiting for a notification.

  ```python
  cv = ConditionVariable()
  waiter = cv.wait(1e6)  # Wait for 1 second.
  session.run(waiter)
  ```

  ```python

  with tf.device("/job:ps/task:0"):
    cv = ConditionVariable(shared_name="shared_cv")
    counter = tf.get_variable('counter', initializer=0, use_resource=True)
    param = tf.get_variable('param', initializer=..., use_resource=True)

  # On workers 0 ... N - 1
  wait_for_notification = cv.wait()
  with tf.control_dependencies([wait_for_notification]):
    # Request variables from parameter server
    with tf.control_dependencies([counter.assign_add(1, use_locking=True)]):
      value = param.value()
      err = f(value)
      update = param.assign_add(err)
  sess.run(update)

  # On controller
  with tf.control_dependencies([counter.assign(0, use_locking=True)]):
    notifier = cv.notify()
  session.run(notifier)
  ```
  """

  def __init__(self,
               name=None,
               shared_name=None,
               condition_variable_def=None,
               import_scope=None):
    """Creates a Condition Variable."""
    if condition_variable_def and name is not None:
      raise ValueError("condition_variable_def and shared_name are "
                       "mutually exclusive.")
    if condition_variable_def:
      self._init_from_proto(condition_variable_def, import_scope=import_scope)
    else:
      self._init_from_args(name, shared_name)

  def _init_from_proto(self, condition_variable_def, import_scope):  # pylint: disable=invalid-name
    raise NotImplementedError("Not yet implemented")
    # TODO(ebrevdo): Re-enable once ConditionVariable is in core.
    # assert isinstance(
    #     condition_variable_def, condition_variable_pb2.ConditionVariableDef)
    # # Create from condition_variable_def.
    # g = ops.get_default_graph()
    # self._handle = g.as_graph_element(
    #     ops.prepend_name_scope(
    #         condition_variable_def.condition_variable_name,
    #         import_scope=import_scope))

  def _init_from_args(self, name, shared_name):  # pylint: disable=invalid-name
    """Initialize the ConditionVariable from constructor arguments."""
    with ops.name_scope(name, "ConditionVariable", []) as name:
      with ops.init_scope():
        # pylint: disable=protected-access
        container = ops.get_default_graph()._container
        # pylint: enable=protected-access
        if shared_name is None:
          shared_name = name
        if container is None:
          container = ""
        self._handle = gen_resource_variable_ops.condition_variable(
            shared_name=shared_name, container=container, name=name)

    if not context.executing_eagerly():
      ops.add_to_collections(CONDITION_VARIABLES, self)

  @property
  def name(self):
    return self._handle.op.name

  def notify(self, name=None):
    """Notify all waiters on this ConditionVariable.

    Args:
      name: (optional).  The name to prefix to any created `Operations`.

    Returns:
      An `Operation` that, when executed, notifies all waiters.
    """
    with ops.name_scope(name, "notify_condition_variable",
                        [self._handle]) as name:
      return gen_resource_variable_ops.notify_condition_variable(
          self._handle, name=name)

  def wait(self, timeout_in_us=None, name=None):
    """Add a waiter on this ConditionVariable.

    Args:
      timeout_in_us: (optional).  The timeout for the waiter.  Default is
        to wait forever.
      name: (optional).  The name to prefix to any created `Operations`.

    Returns:
      A scalar `Tensor` that, when executed, waits on the Conditional Variable.
      If a timeout is provided, the Tensor will return `True` if a notification
      occurred within the timeout range and `False` otherwise.  If no timeout
      is provided, then the waiter will wait forever and always return `True`
      upon eventual notification.
    """
    with ops.name_scope(name, "wait_for_condition_variable",
                        [self._handle]) as name:
      if timeout_in_us is None:
        timeout_in_us = -1
      timeout_in_us = ops.convert_to_tensor(
          timeout_in_us, dtype=dtypes.int64, name="timeout_in_us")
      return gen_resource_variable_ops.wait_for_condition_variable(
          self._handle, timeout_in_us=timeout_in_us, name=name)

  # TODO(ebrevdo): Re-enable once ConditionVariable is in core.

  # def to_proto(self, export_scope=None):
  #   """Converts a `ConditionVariable` to a `ConditionVariableDef` protobuf.

  #   Args:
  #     export_scope: Optional `string`. Name scope to remove.

  #   Returns:
  #     A `ConditionVariableDef` protocol buffer, or `None` if the
  #     `ConditionVariable` is not in the specified name scope.
  #   """
  #   if export_scope is None or self.handle.name.startswith(export_scope):
  #     cs_def = condition_variable_pb2.ConditionVariableDef()
  #     cs_def.condition_variable_name = ops.strip_name_scope(
  #         self._handle.name, export_scope)
  #     return cs_def
  #   else:
  #     return None

  # @staticmethod
  # def from_proto(condition_variable_def, import_scope=None):
  #   return ConditionVariable(
  #       condition_variable_def=condition_variable_def,
  #       import_scope=import_scope)


# TODO(ebrevdo): Re-enable once ConditionVariable is in core.

# ops.register_proto_function(
#     CONDITION_VARIABLES,
#     proto_type=condition_variable_pb2.ConditionVariableDef,
#     to_proto=ConditionVariable.to_proto,
#     from_proto=ConditionVariable.from_proto)
