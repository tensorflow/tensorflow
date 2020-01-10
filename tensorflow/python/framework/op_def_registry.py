"""Global registry for OpDefs."""

from tensorflow.core.framework import op_def_pb2


_registered_ops = {}


def register_op_list(op_list):
  """Register all the ops in an op_def_pb2.OpList."""
  if not isinstance(op_list, op_def_pb2.OpList):
    raise TypeError("%s is %s, not an op_def_pb2.OpList" %
                    (op_list, type(op_list)))
  for op_def in op_list.op:
    if op_def.name in _registered_ops:
      assert _registered_ops[op_def.name] == op_def
    else:
      _registered_ops[op_def.name] = op_def


def get_registered_ops():
  """Returns a dictionary mapping names to OpDefs."""
  return _registered_ops
