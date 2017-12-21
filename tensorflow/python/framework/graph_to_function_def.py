# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================
"""Utility to convert a Graph to a FunctionDef."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

from tensorflow.core.framework import function_pb2
from tensorflow.core.framework import op_def_pb2
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import op_def_registry


def _make_argname_from_tensor_name(name):
  return re.sub(":0$", "", name).replace(":", "_o")


def _tensor_to_argdef(t, name=None, used_names=None):
  """Convert tensor t to an argdef, with a specified name or a unique name."""
  arg = op_def_pb2.OpDef.ArgDef()
  if name is None:
    arg.name = _make_argname_from_tensor_name(t.name)
    if used_names is not None:
      if arg.name in used_names:
        i = 0
        while True:
          new_name = "%s_U%d" % (arg.name, i)
          if new_name not in used_names:
            arg.name = new_name
            break
          i += 1
      used_names.add(arg.name)
  else:
    arg.name = name
  arg.type = t.dtype.as_datatype_enum
  return arg


def _is_in_placeholders(op, func_arg_placeholders):
  """Checks whether any output of this op is in func_arg_placeholders."""
  return op.values() and any(x.name in func_arg_placeholders
                             for x in op.values())


def _get_node_def(op):
  return op._node_def  # pylint: disable=protected-access


def _get_op_def(op):
  return op.op_def or op_def_registry.get_registered_ops()[op.type]


def _create_input_dict(function_graph,
                       func_arg_placeholders,
                       initial_value=None):
  """Create a mapping from graph tensor names to function tensor names."""
  if initial_value is None:
    input_dict = {}
  else:
    input_dict = dict(initial_value)
  for op in function_graph.get_operations():
    if _is_in_placeholders(op, func_arg_placeholders):
      input_dict[op.name] = op.name
    else:
      op_def = _get_op_def(op)
      attrs = _get_node_def(op).attr
      o = 0
      for arg_def in op_def.output_arg:
        if arg_def.number_attr:
          num = attrs[arg_def.number_attr].i
        elif arg_def.type_list_attr:
          num = len(attrs[arg_def.type_list_attr].list.type)
        else:
          num = 1
        for i in range(num):
          result = "%s:%s:%d" % (op.name, arg_def.name, i)
          input_dict[op.values()[o].name] = result
          if o == 0:
            input_dict[op.name] = result
          o += 1
  return input_dict


def _add_op_node(op, func, input_dict):
  """Converts an op to a function def node and add it to `func`."""
  # Add an entry in func.node_def

  # Note that extend() makes a copy in this case, see:
  # https://developers.google.com/protocol-buffers/docs/reference/python-generated#repeated-message-fields
  func.node_def.extend([_get_node_def(op)])
  node_def = func.node_def[-1]
  for i in range(len(node_def.input)):
    if not node_def.input[i].startswith("^"):
      assert node_def.input[i] in input_dict, ("%s missing from %s" %
                                               (node_def.input[i],
                                                input_dict.items()))
      node_def.input[i] = input_dict[node_def.input[i]]
  # The function is stateful if any of its operations are stateful.
  # NOTE(mrry): The "Const" node typically does not have an `OpDef` associated
  # with it, so we assume any nodes without an `OpDef` are stateless.
  # TODO(skyewm): Remove the `is not None` test after we transition to the C
  # API.
  if op.op_def is not None and op.op_def.is_stateful:
    func.signature.is_stateful = True


def graph_to_function_def(graph, operations, inputs, outputs, out_names=None):
  """Returns `graph` as a `FunctionDef` protocol buffer.

  This method creates a [`FunctionDef`](
  https://www.tensorflow.org/code/tensorflow/core/framework/function.proto)
  protocol buffer that contains all the ops in `operations`.  The
  operations become the body of the function.

  The arguments `inputs` and `outputs` will be listed as the inputs
  and outputs tensors of the function.  They must be lists of
  tensors present in the graph.  The lists can optionally be empty.

  Args:
    graph: Graph.
    operations: the operations to put in the function. Must be a subset of
     the operations in the graph.
    inputs: List of tensors. Inputs to the function.
    outputs: List of tensors. Outputs of the function.
    out_names: Optional list of string names for the outputs.

  Returns:
    A FunctionDef protocol buffer.

  Raises:
    ValueError: if out_names is specified and the wrong length.
  """
  func = function_pb2.FunctionDef()
  func.signature.name = "_"
  used_names = set()
  func.signature.input_arg.extend(
      [_tensor_to_argdef(i, used_names=used_names) for i in inputs])
  # Initializes the input map with all placeholder input tensors.
  initial_dict = {}
  for o, m in zip(inputs, func.signature.input_arg):
    initial_dict[o.name] = m.name
  if out_names is None:
    used_names = set()
    func.signature.output_arg.extend(
        [_tensor_to_argdef(o, used_names=used_names) for o in outputs])
  elif len(outputs) != len(out_names):
    raise errors_impl.InvalidArgumentError(
        None, None,
        "output names must be either empty or equal in size to outputs. "
        "output names size = %d outputs size = %d" %
        (len(out_names), len(outputs)))
  elif len(out_names) != len(set(out_names)):
    raise ValueError(
        "Must not have duplicates in out_names: %s" % ", ".join(out_names))
  else:
    func.signature.output_arg.extend(
        [_tensor_to_argdef(o, name=n) for o, n in zip(outputs, out_names)])
  func_arg_placeholders = set([i.name for i in inputs])
  input_dict = _create_input_dict(graph, func_arg_placeholders,
                                  initial_value=initial_dict)

  for op in operations:
    if _is_in_placeholders(op, func_arg_placeholders):
      continue
    _add_op_node(op, func, input_dict)

  if out_names is None:
    for index, o in enumerate(outputs):
      k = func.signature.output_arg[index].name
      func.ret[k] = input_dict[o.name]
  else:
    for o, n in zip(outputs, out_names):
      func.ret[n] = input_dict[o.name]

  return func
