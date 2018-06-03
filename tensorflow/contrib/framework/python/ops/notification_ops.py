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
"""Notification object and execution logic."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# TODO(ebrevdo): Re-enable once Notification is in core.
# from tensorflow.core.protobuf import \
#    notification_pb2

from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_resource_variable_ops

# Graph Keys
NOTIFICATIONS = "notifications"


class Notification(object):
  """Notification.

  A `Notification` object is  a resource in the graph which allows multiple
  threads or workers to wait for a predetermined amount of time, or to
  synchronize while waiting for a notification.

  The object is stateful.  Waiters will pause (or time out) until the
  Notification object is notified via the result of `notifier()`.  By default,
  notifying the object atomically releases all waiters, and resets the state.
  Any new `wait` operations executing after this will go into a waiting state.
  Alternatively, executing `notifier(immediately_reset=False)` will also set the
  state of the Notification object.  Any `wait` operations executing after this
  will immediately continue and will not time out.  The state of the object can
  be reset by executing the `Operation` `notifier()` (with default arguments) or
  the `Operation` `resetter()`.

  ```python
  n = Notification()
  waiter = n.wait(1e6)  # Wait for 1 second.
  session.run(waiter)
  ```

  ```python

  with tf.device("/job:ps/task:0"):
    n = Notification(shared_name="shared_n")
    counter = tf.get_variable('counter', initializer=0, use_resource=True)
    param = tf.get_variable('param', initializer=..., use_resource=True)

  # On workers 0 ... N - 1
  wait_for_notification = n.wait()
  with tf.control_dependencies([wait_for_notification]):
    # Request variables from parameter server
    with tf.control_dependencies([counter.assign_add(1, use_locking=True)]):
      value = param.value()
      err = f(value)
      update = param.assign_add(err)
  sess.run(update)

  # On controller
  with tf.control_dependencies([counter.assign(0, use_locking=True)]):
    notifier = n.notifier()
  session.run(notifier)
  ```
  """

  def __init__(self,
               name=None,
               shared_name=None,
               notification_def=None,
               import_scope=None):
    """Creates a Condition Variable."""
    if notification_def and name is not None:
      raise ValueError("notification_def and shared_name are "
                       "mutually exclusive.")
    if notification_def:
      self._init_from_proto(notification_def, import_scope=import_scope)
    else:
      self._init_from_args(name, shared_name)

  def _init_from_proto(self, notification_def, import_scope):  # pylint: disable=invalid-name
    raise NotImplementedError("Not yet implemented")
    # TODO(ebrevdo): Re-enable once Notification is in core.
    # assert isinstance(
    #     notification_def, notification_pb2.NotificationDef)
    # # Create from notification_def.
    # g = ops.get_default_graph()
    # self._handle = g.as_graph_element(
    #     ops.prepend_name_scope(
    #         notification_def.notification_name,
    #         import_scope=import_scope))

  def _init_from_args(self, name, shared_name):  # pylint: disable=invalid-name
    """Initialize the Notification from constructor arguments."""
    with ops.name_scope(name, "Notification", []) as name:
      with ops.init_scope():
        # pylint: disable=protected-access
        container = ops.get_default_graph()._container
        # pylint: enable=protected-access
        if shared_name is None:
          shared_name = name
        if container is None:
          container = ""
        # Build the notification resource outside of any control dependencies.
        with ops.control_dependencies(None):
          self._handle = gen_resource_variable_ops.notification(
              shared_name=shared_name, container=container, name=name)

    if not context.executing_eagerly():
      ops.add_to_collections(NOTIFICATIONS, self)

  @property
  def name(self):
    return self._handle.op.name

  @property
  def graph(self):
    return self._handle.graph

  def notifier(self, immediately_reset=True, name=None):
    """Operation that, when executed, notifies all waiters on this Notification.

    Args:
      immediately_reset: Python bool.  Whether the Notification object
        should immediately be reset or not.  If `True` (default), then any
        waiters that execute after this notification will wait for the next
        notification.  If `False`, any waiters that execute after this
        notification will immediately return.  To reset the notification,
        use the `resetter` operation.
      name: (optional).  The name to prefix to any created `Operations`.

    Returns:
      An `Operation` that, when executed, notifies all waiters and possibly
      permanently sets the `Notification` object (see also `resetter`).
    """
    with ops.name_scope(name, "notify_notification",
                        [self._handle]) as name:
      return gen_resource_variable_ops.notify_notification(
          self._handle, immediately_reset=immediately_reset, name=name)

  def resetter(self, name=None):
    """Operation that, when executed, resets this Notification.

    For use when a `Notification` object has been set by using the
    `notifier(immediately_reset=False)` method.

    Args:
      name: (optional).  The name to prefix to any created `Operations`.

    Returns:
      An `Operation` that, when executed, resets the `Notification` object
      so any future waiters will pause and wait.
    """
    return gen_resource_variable_ops.reset_notification(
        self._handle, name=name)


  def wait(self, timeout_in_us=None, name=None):
    """Add a waiter on this Notification.

    Args:
      timeout_in_us: (optional).  The timeout for the waiter.  Default is
        to wait forever.
      name: (optional).  The name to prefix to any created `Operations`.

    Returns:
      A scalar `Tensor` that, when executed, waits on the Notification.
      If a timeout is provided, the Tensor will return `True` if a notification
      occurred within the timeout range and `False` otherwise.  If no timeout
      is provided, then the waiter will wait forever and always return `True`
      upon eventual notification.
    """
    with ops.name_scope(name, "wait_for_notification",
                        [self._handle]) as name:
      if timeout_in_us is None:
        timeout_in_us = -1
      timeout_in_us = ops.convert_to_tensor(
          timeout_in_us, dtype=dtypes.int64, name="timeout_in_us")
      return gen_resource_variable_ops.wait_for_notification(
          self._handle, timeout_in_us=timeout_in_us, name=name)

  # TODO(ebrevdo): Re-enable once Notification is in core.

  # def to_proto(self, export_scope=None):
  #   """Converts a `Notification` to a `NotificationDef` protobuf.

  #   Args:
  #     export_scope: Optional `string`. Name scope to remove.

  #   Returns:
  #     A `NotificationDef` protocol buffer, or `None` if the
  #     `Notification` is not in the specified name scope.
  #   """
  #   if export_scope is None or self.handle.name.startswith(export_scope):
  #     cs_def = notification_pb2.NotificationDef()
  #     cs_def.notification_name = ops.strip_name_scope(
  #         self._handle.name, export_scope)
  #     return cs_def
  #   else:
  #     return None

  # @staticmethod
  # def from_proto(notification_def, import_scope=None):
  #   return Notification(
  #       notification_def=notification_def,
  #       import_scope=import_scope)


# TODO(ebrevdo): Re-enable once Notification is in core.

# ops.register_proto_function(
#     NOTIFICATIONS,
#     proto_type=notification_pb2.NotificationDef,
#     to_proto=Notification.to_proto,
#     from_proto=Notification.from_proto)
