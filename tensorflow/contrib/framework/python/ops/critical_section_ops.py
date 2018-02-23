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
from tensorflow.python.eager import function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
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
      with ops.control_dependencies(None):
        # pylint: disable=protected-access
        container = ops.get_default_graph()._container
        # pylint: enable=protected-access
        if shared_name is None:
          shared_name = name
        if container is None:
          container = ""
        self._handle = gen_resource_variable_ops.mutex_v2(
            shared_name=shared_name, container=container, name=name)

    if context.in_graph_mode():
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
      ValueError: If `fn` attempts to use this `CriticalSection` in any nested
        way.
      ValueError: If `exclusive_resource_access` is not provided (is `True`) and
        another `CriticalSection` has an execution requesting the same
        resources as in `*args`, `**kwargs`, and any additionaly captured
        inputs in `fn`.  Note, even if `exclusive_resource_access` is `True`,
        if another execution in another `CriticalSection` was created without
        `exclusive_resource_access=True`, a `ValueError` will be raised.
    """
    name = kwargs.pop("name", None)
    exclusive_resource_access = kwargs.pop("exclusive_resource_access", True)

    with ops.name_scope(name, "critical_section_execute", []):
      lock = gen_resource_variable_ops.mutex_lock(self._handle)

      with ops.control_dependencies([lock]):
        c_known_ops = set()
        c_captured_tensors = set()

        def add_op_internal(op):
          c_known_ops.add(op)
          for i in op.inputs:
            if i.op not in c_known_ops:
              c_captured_tensors.add(i)

        c = function.HelperContext(add_op_internal)
        with c:
          r = fn(*args, **kwargs)

        resource_inputs = set([
            x for x in
            list(nest.flatten(args)) + nest.flatten(kwargs.values()) +
            list(c_captured_tensors)
            if tensor_util.is_tensor(x) and x.dtype == dtypes.resource])

      if self._handle in resource_inputs:
        raise ValueError("The function fn attempts to access the "
                         "CriticalSection in which it would be running.  "
                         "This is illegal and would cause deadlocks.  "
                         "CriticalSection: %s." % self._handle)

      if context.in_graph_mode():
        # Collections and op introspection does not work in eager
        # mode.  This is generally ok; since eager mode (as of
        # writing) executes sequentially anyway.
        for sg in ops.get_collection(CRITICAL_SECTION_EXECUTIONS):
          if sg.handle.name == self._handle.name:
            # Other executions in the same critical section are allowed.
            continue
          if not (exclusive_resource_access or sg.exclusive_resource_access):
            # Neither execution requested exclusive access.
            continue
          resource_intersection = resource_inputs.intersection(sg.resources)
          if resource_intersection:
            raise ValueError(
                "This execution would access resources: %s.  Either this "
                "lock (CriticalSection: %s) or lock '%s' "
                "(CriticalSection: %s) requested exclusive resource access "
                "of this resource.  Did you mean to call execute with keyword "
                "argument exclusive_resource_access=False?" %
                (list(resource_intersection), self._handle.name,
                 sg.op.name, sg.handle.name))

      def identity(x):  # pylint: disable=invalid-name
        if isinstance(x, tensor_array_ops.TensorArray):
          return x.identity()
        elif isinstance(x, ops.Operation):
          return control_flow_ops.group(x)
        elif context.in_eager_mode() and x is None:
          return None
        else:
          return array_ops.identity(x)

      r_flat = [identity(x) for x in nest.flatten(r)]

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
        r = nest.pack_sequence_as(
            r, control_flow_ops.tuple(nest.flatten(r)))

      with ops.control_dependencies([ensure_lock_exists]):
        outputs = nest.map_structure(identity, r)

      if context.in_graph_mode():
        signature = _ExecutionSignature(
            op=lock.op,
            handle=self._handle,
            resources=list(resource_inputs),
            exclusive_resource_access=exclusive_resource_access)
        ops.add_to_collections(
            CRITICAL_SECTION_EXECUTIONS, signature)

      return outputs

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
