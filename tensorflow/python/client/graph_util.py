"""Helpers to manipulate a tensor graph in python.
"""

import tensorflow.python.platform

from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import ops
from tensorflow.python.framework import types
from tensorflow.python.platform import logging

_VARIABLE_OPS = {
    "Assign",
    "AssignAdd",
    "AssignSub",
    "Queue",
    "RandomParameters",
    "ScatterAdd",
    "ScatterSub",
    "ScatterUpdate",
    "Variable",
}


def _is_variable_op(op):
  """Returns true if 'op' refers to a Variable node."""
  return op in _VARIABLE_OPS


def set_cpu0(device_string):
  """Creates a new device string based on `device_string' but using /CPU:0.

   If the device is already on /CPU:0, this is a no-op.

   Args:
     device_string: A device string.

   Returns:
     A device string.
  """
  parsed_device = pydev.from_string(device_string)
  parsed_device.device_type = "CPU"
  parsed_device.device_index = 0
  return parsed_device.to_string()


def must_run_on_cpu(node, pin_variables_on_cpu=False):
  """Returns True if the given node_def must run on CPU, otherwise False.

  Args:
    node: The node to be assigned to a device. Could be either an ops.Operation
      or NodeDef.
    pin_variables_on_cpu: If True, this function will return False if node_def
      represents a variable-related op.

  Returns:
    True if the given node must run on CPU, otherwise False.
  """

  if isinstance(node, ops.Operation):
    node_def = node.node_def
  else:
    assert isinstance(node, graph_pb2.NodeDef)
    node_def = node

  # If the op is a variable-related op, should we pin it on CPU?
  if pin_variables_on_cpu and _is_variable_op(node_def.op):
    return True

  # Constant operations producing a string or int32 must run on CPU.
  if node_def.op == "Const":
    # Get the value of the 'dtype' attr
    dtype = node_def.attr["dtype"].type
    if dtype == types.string or dtype == types.int32:
      return True

  if node_def.op == "DynamicStitch":
    dtype = node_def.attr["T"].type
    if dtype == types.int32:
      # DynamicStitch on GPU only works for int32 values.
      return True

  if node_def.op in ["Cast"]:
    dtype = node_def.attr["SrcT"].type
    if dtype == types.int32:
      # Cast on GPU does not works for int32 values.
      return True
  return False


################################################################################
#
# device functions for use in with g.device(...)
#
################################################################################


def pin_variables_on_cpu(op):
  """Returns a CPU device for Variable nodes if the device is not specified.

  Args:
    op: The ops.Operation object describing the node for which a device
      should be chosen. The op.device field is respected.

  Returns:
    A device containing "/device:CPU:0" if the node is related to a variable.
  """
  device = op.device if op.device is not None else ""
  dev = pydev.from_string(device)

  # If a device type exists already, do not override.
  if dev.device_type:
    return device

  if isinstance(op, ops.Operation):
    node_def = op.node_def
  else:
    assert isinstance(op, graph_pb2.NodeDef)
    node_def = op

  if _is_variable_op(node_def.op):
    return set_cpu0(device)
  return device


def pin_to_cpu(op):
  """Returns a CPU device for the given node."""
  device = op.device if op.device is not None else ""
  dev = pydev.from_string(device)

  if not dev.device_type:
    return set_cpu0(device)
  if dev.device_type == "CPU":
    return device

  logging.info("Operation %s has been assigned to a non-CPU (%s), so "
               "it will not be pinned to the CPU.", op.name, dev.device_type)
  return device
