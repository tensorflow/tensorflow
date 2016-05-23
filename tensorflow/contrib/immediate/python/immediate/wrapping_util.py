# Utilities to help with wrapping of TF namespace
# This module helps obtain list of all gen op modules
# And will generate the correct order of wrapping Python op module
# (not all orders work because modules include each other)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__all__ = ["get_op_input_argnames_argtypes"]

def get_op_input_argnames_argtypes():
  """Parses op_history.v0.pbtxt to get list of input args for ops."""

  from tensorflow.core.framework import graph_pb2
  from tensorflow.core.framework import op_def_pb2
  from google.protobuf import text_format
  ops_file = "/Users/yaroslavvb/tfimmediate_src/tensorflow/tensorflow/core/ops/compat/ops_history.v0.pbtxt"

  oplist = op_def_pb2.OpList()
  text_format.Merge(open(ops_file).read(), oplist)
  # dictionary of opname->input names pairs
  argnames = {}
  argtypes = {}
  for op in oplist.op:
    argnames[op.name] = [arg.name for arg in op.input_arg]
    argtypes0 = {}
    for arg in op.input_arg:
      if arg.number_attr or arg.type_list_attr:
        argtypes0[arg.name] = "list"
      else:
        argtypes0[arg.name] = "single"
    argtypes[op.name] = argtypes0

  return argnames, argtypes
