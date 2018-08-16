# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Critical Section object and execution logic."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

# TODO(ebrevdo): Re-enable once CriticalSection is in core.
# from tensorflow.core.protobuf import critical_section_pb2

from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.util import nest


# Graph Keys
CRITICAL_SECTIONS = "critical_sections"
CRITICAL_SECTION_EXECUTIONS = "critical_section_executions"


class _ExecutionSignature(
    collections.namedtuple("_ExecutionSignature",
                           ("op", "handle",
                            "resources", "exclusive_resource_access"))):
  """A class storing an `ExecuteInCriticalResource` op and associated attrs."""
  pass


def _identity(x):
  """Identity op that recognizes `TensorArray`, `Operation`, and `Tensor`."""
  if isinstance(x, tensor_array_ops.TensorArray):
    return x.identity()
  elif isinstance(x, ops.Operation):
    return control_flow_ops.group(x)
  elif context.executing_eagerly() and x is None:
    return None
  else:
    return array_ops.identity(x)


def _get_colocation(op):
  """Get colocation symbol from op, if any."""
  try:
    return op.get_attr("_class")
  except ValueError:
    return None


class CriticalSection(object):
  """Critical section.

  A `CriticalSection` object is a resource in the graph which executes subgraphs
  in **serial** order.  A common example of a subgraph one may wish to run
  exclusively is the one given by the following function:

  ```python
  v = resource_variable_ops.ResourceVariable(0.0, name="v")

  def count():
    value = v.read_value()
    with tf.control_dependencies([value]):
      with tf.control_dependencies([v.assign_add(1)]):
        return tf.identity(value)
  ```

  Here, a snapshot of `v` is captured in `value`; and then `v` is updated.
  The snapshot value is returned.

  If multiple workers or threads all execute `count` in parallel, there is no
  guarantee that access to the variable `v` is atomic at any point within
  any thread's calculation of `count`.  In fact, even implementing an atomic
  counter that guarantees that the user will see each value `0, 1, ...,` is
  currently impossible.

  The solution is to ensure any access to the underlying resource `v` is
  only processed through a critical section:

  ```python
  cs = CriticalSection()
  f1 = cs.execute(count)
  f2 = cs.execute(count)
  output = f1 + f2
  session.run(output)
  ```
  The functions `f1` and `f2` will be executed serially, and updates to `v`
  will be atomic.

  **NOTES**

  All resource objects, including the critical section and any captured
  variables of functions executed on that critical section, will be
  colocated to the same device (host and cpu/gpu).

  When using multiple critical sections on the same resources, there is no
  guarantee of exclusive access to those resources.  This behavior is disallowed
  by default (but see the kwarg `exclusive_resource_access`).

  For example, running the same function in two separate critical sections
  will not ensure serial execution:

  ```python
  v = tf.get_variable("v", initializer=0.0, use_resource=True)
  def accumulate(up):
    x = v.read_value()
    with tf.control_dependencies([x]):
      with tf.control_dependencies([v.assign_add(up)]):
        return tf.identity(x)
  ex1 = CriticalSection().execute(
    accumulate, 1.0, exclusive_resource_access=False)
  ex2 = CriticalSection().execute(
    accumulate, 1.0, exclusive_resource_access=False)
  bad_sum = ex1 + ex2
  sess.run(v.initializer)
  sess.run(bad_sum)  # May return 0.0
  ```
  """

  def __init__(self, name=None, shared_name=None,
               critical_section_def=None, import_scope=None):
    """Creates a critical section."""
    if critical_section_def and name is not None:
      raise ValueError("critical_section_def and shared_name are "
                       "mutually exclusive.")
    if critical_section_def:
      self._init_from_proto(critical_section_def, import_scope=import_scope)
    else:
      self._init_from_args(name, shared_name)

  def _init_from_proto(self, critical_section_def, import_scope):  # pylint: disable=invalid-name
    raise NotImplementedError("Not yet implemented")
    # TODO(ebrevdo): Re-enable once CriticalSection is in core.
    # assert isinstance(
    #     critical_section_def, critical_section_pb2.CriticalSectionDef)
    # # Create from critical_section_def.
    # g = ops.get_default_graph()
    # self._handle = g.as_graph_element(
    #     ops.prepend_name_scope(
    #         critical_section_def.critical_section_name,
    #         import_scope=import_scope))

  def _init_from_args(self, name, shared_name):  # pylint: disable=invalid-name
    """Initialize the CriticalSection from constructor arguments."""
    with ops.name_scope(name, "CriticalSection", []) as name:
      with ops.init_scope():
        # pylint: disable=protected-access
        container = ops.get_default_graph()._container
        # pylint: enable=protected-access
        if shared_name is None:
          shared_name = name
        if container is None:
          container = ""
        self._handle = gen_resource_variable_ops.mutex_v2(
            shared_name=shared_name, container=container, name=name)

    if not context.executing_eagerly():
      ops.add_to_collections(CRITICAL_SECTIONS, self)

  @property
  def name(self):
    return self._handle.op.name

  def execute(self, fn, *args, **kwargs):
    """Execute function `fn(*args, **kwargs)` inside the CriticalSection.

    Args:
      fn: The function to execute.  Must return at least one tensor.
      *args: Additional positional arguments to `fn`.
      **kwargs: Additional keyword arguments to `fn`.
        Several keywords are reserved for `execute`.  These are:

        - name; The name to use when creating the execute operation.
        - exclusive_resource_access; Whether the resources required by
          `fn` should be exclusive to this `CriticalSection`.  Default: `True`.
          You may want to set this to `False` if you will be accessing a
          resource in read-only mode in two different CriticalSections.

    Returns:
      The tensors returned from `fn(*args, **kwargs)`.

    Raises:
      ValueError: If `fn` attempts to lock this `CriticalSection` in any nested
        or lazy way that may cause a deadlock.
      ValueError: If `exclusive_resource_access` is not provided (is `True`) and
        another `CriticalSection` has an execution requesting the same
        resources as in `*args`, `**kwargs`, and any additionally captured
        inputs in `fn`.  Note, even if `exclusive_resource_access` is `True`,
        if another execution in another `CriticalSection` was created without
        `exclusive_resource_access=True`, a `ValueError` will be raised.
    """
    name = kwargs.pop("name", None)
    exclusive_resource_access = kwargs.pop("exclusive_resource_access", True)

    with ops.name_scope(name, "critical_section_execute", []):

      # Ensure that mutex locking only happens *after* all args and
      # kwargs have been executed.  This avoids certain types of deadlocks.
      lock = gen_resource_variable_ops.mutex_lock(self._handle)

      if not context.executing_eagerly():
        # NOTE(ebrevdo): This is to ensure we don't pick up spurious
        # Operations created by other threads.
        with ops.get_default_graph()._lock:  # pylint: disable=protected-access
          existing_ops = ops.get_default_graph().get_operations()
          with ops.control_dependencies([lock]):
            r = fn(*args, **kwargs)
          # TODO(ebrevdo): If creating critical sections in a python loop, this
          # makes graph creation time quadratic.  Revisit if this
          # becomes a problem.
          created_ops = (set(ops.get_default_graph().get_operations())
                         .difference(existing_ops))
      else:
        with ops.control_dependencies([lock]):
          r = fn(*args, **kwargs)

      if not context.executing_eagerly():
        self._add_control_dependencies_to_lock(created_ops, lock.op)

        # captured_resources is a list of resources that are directly
        # accessed only by ops created during fn(), not by any
        # ancestors of those ops in the graph.
        captured_resources = set([
            input_ for op in created_ops
            for input_ in op.inputs
            if input_.dtype == dtypes.resource
        ])

        # NOTE(ebrevdo): The only time self._is_self_handle() is True
        # in this call is if one of the recently created ops, within
        # the execute(), themselves attempt to access the
        # CriticalSection.  This will cause a deadlock.
        if any(self._is_self_handle(x) for x in captured_resources):
          raise ValueError("The function fn attempts to directly access the "
                           "CriticalSection in which it would be running.  "
                           "This is illegal and would cause deadlocks.")

        self._check_multiple_access_to_resources(
            captured_resources, exclusive_resource_access)

      r_flat = [_identity(x) for x in nest.flatten(r)]

      with ops.control_dependencies(r_flat):
        # The identity must run on the same machine as self._handle
        with ops.colocate_with(self._handle):
          # Do not use array_ops.identity as there are special
          # optimizations within TensorFlow which seem to elide it
          # even when optimizations are disabled(!).
          ensure_lock_exists = gen_resource_variable_ops.consume_mutex_lock(
              lock)

        # Make sure that if any element of r is accessed, all of
        # them are executed together.
        r = nest.pack_sequence_as(r, control_flow_ops.tuple(nest.flatten(r)))

      with ops.control_dependencies([ensure_lock_exists]):
        outputs = nest.map_structure(_identity, r)

      if not context.executing_eagerly():
        signature = _ExecutionSignature(
            op=lock.op,
            handle=self._handle,
            resources=list(captured_resources),
            exclusive_resource_access=exclusive_resource_access)
        ops.add_to_collections(
            CRITICAL_SECTION_EXECUTIONS, signature)

      return outputs

  def _add_control_dependencies_to_lock(self, created_ops, lock_op):
    """To avoid deadlocks, all args must be executed before lock_op."""
    # Get all arguments (explicit and captured) of all ops created by fn().
    all_args = set([input_.op for op in created_ops for input_ in op.inputs])
    all_args.update(
        input_op for op in created_ops for input_op in op.control_inputs)
    # Unfortunately, we can't use sets throughout because TF seems to
    # create new Operation objects for the same op sometimes; and we
    # can't rely on id(op).

    # pylint: disable=protected-access
    all_args_dict = dict((op._id, op) for op in all_args)

    # Remove ops created within fn, or that lock_op already has a
    # control dependency on.  Also remove a possible self-loop.
    for op in created_ops:
      all_args_dict.pop(op._id, None)
    for op in lock_op.control_inputs:
      all_args_dict.pop(op._id, None)
    for input_ in lock_op.inputs:
      all_args_dict.pop(input_.op._id, None)
    all_args_dict.pop(lock_op._id, None)

    all_args = all_args_dict.values()

    if not all_args:
      # No control dependencies to add; return early.
      return

    # This group is important: it ensures that any ops in all_args
    # outside the control context of the lock_op (and this fn, which
    # runs in the same context) are added to this context before
    # being added to the control dependencies of lock_op.
    all_args = control_flow_ops.group(*all_args)

    lock_op._add_control_input(all_args)
    # pylint: enable=protected-access

  def _is_self_handle(self, x):
    """Check if the tensor `x` is the same Mutex as `self._handle`."""
    if isinstance(x, ops.EagerTensor):
      return x is self._handle
    return (x.op.type == "MutexV2"
            # blank shared_name means the op will create a unique one.
            and x.op.get_attr("shared_name")
            and (x.op.get_attr("shared_name") ==
                 self._handle.op.get_attr("shared_name"))
            and (x.op.device == self._handle.op.device
                 or _get_colocation(x.op) == _get_colocation(self._handle.op)))

  def _check_multiple_access_to_resources(
      self, captured_resources, exclusive_resource_access):
    """Raise if captured_resources are accessed by another CriticalSection.

    Args:
      captured_resources: Set of tensors of type resource.
      exclusive_resource_access: Whether this execution requires exclusive
        resource access.

    Raises:
      ValueError: If any tensors in `captured_resources` are also accessed
        by another `CriticalSection`, and at least one of them requires
        exclusive resource access.
    """
    # Collections and op introspection does not work in eager
    # mode.  This is generally ok; since eager mode (as of
    # writing) executes sequentially anyway.
    for sg in ops.get_collection(CRITICAL_SECTION_EXECUTIONS):
      if self._is_self_handle(sg.handle):
        # Other executions in the same critical section are allowed.
        continue
      if not (exclusive_resource_access or sg.exclusive_resource_access):
        # Neither execution requested exclusive access.
        continue
      resource_intersection = captured_resources.intersection(sg.resources)
      if resource_intersection:
        raise ValueError(
            "This execution would access resources: %s.  Either this "
            "lock (CriticalSection: %s) or lock '%s' "
            "(CriticalSection: %s) requested exclusive resource access "
            "of this resource.  Did you mean to call execute with keyword "
            "argument exclusive_resource_access=False?" %
            (list(resource_intersection), self._handle, sg, sg.handle))

  # TODO(ebrevdo): Re-enable once CriticalSection is in core.

  # def to_proto(self, export_scope=None):
  #   """Converts a `CriticalSection` to a `CriticalSectoinDef` protocol buffer.

  #   Args:
  #     export_scope: Optional `string`. Name scope to remove.

  #   Returns:
  #     A `CriticalSectionDef` protocol buffer, or `None` if the
  #     `CriticalSection` is not in the specified name scope.
  #   """
  #   if export_scope is None or self.handle.name.startswith(export_scope):
  #     cs_def = critical_section_pb2.CriticalSectionDef()
  #     cs_def.critical_section_name = ops.strip_name_scope(
  #         self._handle.name, export_scope)
  #     return cs_def
  #   else:
  #     return None

  # @staticmethod
  # def from_proto(critical_section_def, import_scope=None):
  #   return CriticalSection(
  #       critical_section_def=critical_section_def, import_scope=import_scope)


# TODO(ebrevdo): Re-enable once CriticalSection is in core.

# def _execution_to_proto_fn(execution_signature, export_scope=None):
#   """Converts `_ExecutionSignature` to a `CriticalSectionExecutionDef`.
#   # TODO(ebrevdo): Update for _ExecutionSignature storing resource list.

#   Args:
#     execution_signature: Instance of `_ExecutionSignature`.
#     export_scope: The export scope, if any.

#   Returns:
#     An instance of `CriticalSectionExecutionDef`.
#   """
#   if (export_scope is None
#       or execution_signature.op.name.startswith(export_scope)):
#     op_def = critical_section_pb2.CriticalSectionExecutionDef()
#     op_def.execute_in_critical_section_name = ops.strip_name_scope(
#         execution_signature.op.name, export_scope)
#     op_def.exclusive_resource_access = (
#         execution_signature.exclusive_resource_access)
#     return op_def
#   else:
#     return None


# def _execution_from_proto_fn(op_def, import_scope=None):
#   """Converts a `CriticalSectionExecutionDef` to a `_ExecutionSignature`."""
#   # TODO(ebrevdo): Update for _ExecutionSignature storing resource list.
#   assert isinstance(
#       op_def, critical_section_pb2.CriticalSectionExecutionDef)

#   # Create from op_def.
#   g = ops.get_default_graph()
#   execution_op = g.as_graph_element(
#       ops.prepend_name_scope(
#           op_def.execute_in_critical_section_name,
#           import_scope=import_scope))
#   return _ExecutionSignature(
#       op=execution_op,
#       exclusive_resource_access=op_def.exclusive_resource_access)

# ops.register_proto_function(
#     CRITICAL_SECTIONS,
#     proto_type=critical_section_pb2.CriticalSectionDef,
#     to_proto=CriticalSection.to_proto,
#     from_proto=CriticalSection.from_proto)

# ops.register_proto_function(
#     CRITICAL_SECTION_EXECUTIONS,
#     proto_type=critical_section_pb2.CriticalSectionExecutionDef,
#     to_proto=_execution_to_proto_fn,
#     from_proto=_execution_from_proto_fn)
