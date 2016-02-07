# Copyright 2015 Google Inc. All Rights Reserved.
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
"""Python front-end supports for functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import re

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import function_pb2
from tensorflow.core.framework import op_def_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops


def _make_argname_from_tensor_name(name):
  return re.sub(":0$", "", name).replace(":", "_o")


def _tensor_to_argdef(t):
  arg = op_def_pb2.OpDef.ArgDef()
  arg.name = _make_argname_from_tensor_name(t.name)
  arg.type = t.dtype.as_datatype_enum
  return arg


def _get_node_def_attr(op):
  # pylint: disable=protected-access
  return op._node_def.attr
  # pylint: enable=protected-access


def _add_input_array(op, start, limit, dtype, func):
  """Adds a _ListToArray node in the func for op.inputs[start:limit]."""
  node = function_pb2.FunctionDef.Node()
  node.op = "_ListToArray"
  ret_name = op.name + "_L2A_" + str(start)
  node.ret.extend([ret_name])
  node.arg.extend([_make_argname_from_tensor_name(x.name)
                   for x in op.inputs[start:limit]])
  num = limit - start
  node.attr["Tin"].CopyFrom(attr_value_pb2.AttrValue(
      list=attr_value_pb2.AttrValue.ListValue(type=[dtype] * num)))
  node.attr["T"].CopyFrom(attr_value_pb2.AttrValue(type=dtype))
  node.attr["N"].CopyFrom(attr_value_pb2.AttrValue(i=num))
  func.node.extend([node])
  return ret_name


def _add_output_array(op, start, limit, dtype, func):
  """Adds a _ArrayToList node in the func for op.outputs[start:limit]."""
  dtype_proto = attr_value_pb2.AttrValue(type=dtype)
  # A node converting N*T to list(T)
  node = function_pb2.FunctionDef.Node()
  node.op = "_ArrayToList"
  arg_name = op.name + "_A2L_" + str(start)
  ret_name = arg_name + "_out"
  node.ret.append(ret_name)
  node.arg.append(arg_name)
  node.attr["T"].CopyFrom(dtype_proto)
  num = limit - start
  node.attr["N"].CopyFrom(attr_value_pb2.AttrValue(i=num))
  node.attr["out_types"].CopyFrom(attr_value_pb2.AttrValue(
      list=attr_value_pb2.AttrValue.ListValue(type=[dtype] * num)))
  func.node.extend([node])
  num = limit - start
  # Adds an identity node for each element in the array N*T so that
  # uses of each element can be added easily later. These Identity
  # will be eliminated before graph execution.
  for i in xrange(num):
    node = function_pb2.FunctionDef.Node()
    node.op = "Identity"
    node.arg.append(ret_name + ":" + str(i))
    node.ret.append(_make_argname_from_tensor_name(op.outputs[i].name))
    node.attr["T"].CopyFrom(dtype_proto)
    func.node.extend([node])
  return arg_name


def _add_output_list(op, start, limit, dtype_lst, func):
  """Adds a _ArrayToList node in the func for op.outputs[start:limit]."""
  ret_name = op.name + "_Lst_" + str(start) + "_" + str(limit)
  num = limit - start
  assert len(dtype_lst) == num
  # Adds an identity node for each element in the array N*T so that
  # uses of each element can be added easily later. These Identity
  # will be eliminated before graph execution.
  for i in xrange(num):
    node = function_pb2.FunctionDef.Node()
    node.op = "Identity"
    node.arg.append(ret_name + ":" + str(i))
    node.ret.append(_make_argname_from_tensor_name(op.outputs[i].name))
    node.attr["T"].CopyFrom(attr_value_pb2.AttrValue(type=dtype_lst[i]))
    func.node.extend([node])
  return ret_name


def _add_op_node(graph, op, func):
  """Converts an op to a function def node and add it to `func`."""
  node = function_pb2.FunctionDef.Node()
  node.op = op.type
  # pylint: disable=protected-access
  if graph._is_function(op.type):
    op_def = graph._get_function(op.type).signature
  else:
    op_def = op_def_registry.get_registered_ops()[op.type]
  # pylint: enable=protected-access
  attrs = _get_node_def_attr(op)
  out_index = 0
  for arg_def in op_def.output_arg:
    if arg_def.number_attr:
      dtype = arg_def.type or attrs[arg_def.type_attr].type
      num = attrs[arg_def.number_attr].i
      node.ret.append(_add_output_array(op, out_index, out_index + num, dtype,
                                        func))
      out_index += num
    elif arg_def.type_list_attr:
      dtype_lst = attrs[arg_def.type_list_attr].list.type
      num = len(dtype_lst)
      node.ret.append(_add_output_list(op, out_index, out_index + num,
                                       dtype_lst, func))
      out_index += num
    else:
      node.ret.append(_make_argname_from_tensor_name(op.outputs[
          out_index].name))
      out_index += 1
  inp_index = 0
  for arg_def in op_def.input_arg:
    if arg_def.number_attr:
      dtype = arg_def.type or attrs[arg_def.type_attr].type
      num = attrs[arg_def.number_attr].i
      node.arg.append(_add_input_array(op, inp_index, inp_index + num, dtype,
                                       func))
      inp_index += num
    elif arg_def.type_list_attr:
      num = len(attrs[arg_def.type_list_attr].list.type)
      node.arg.extend([_make_argname_from_tensor_name(op.inputs[i].name)
                       for i in range(inp_index, inp_index + num)])
      inp_index += num
    else:
      node.arg.append(_make_argname_from_tensor_name(op.inputs[inp_index].name))
      inp_index += 1
  node.dep.extend([_make_argname_from_tensor_name(x.name)
                   for x in op.control_inputs])
  for k, v in _get_node_def_attr(op).items():
    node.attr[k].CopyFrom(v)
  func.node.extend([node])


# pylint: disable=line-too-long
def graph_to_function_def(graph, name, inputs, outputs):
  """Returns `graph` as a `FunctionDef` protocol buffer.

  This method creates a [`FunctionDef`](
  https://www.tensorflow.org/code/tensorflow/core/framework/function.proto)
  protocol buffer that contains all the ops present in the graph.  The
  graph effectively becomes the body of the function.

  The arguments `inputs` and `outputs` will be listed as the inputs
  and outputs tensors of the function.  They must be lists of
  tensors present in the graph.  The lists can optionally be empty.

  The returned protocol buffer can be passed to the
  [`Graph.add_function()`](#Graph.add_function) method of a
  different graph to make it available there.

  Args:
    graph: GraphDef proto.
    name: string. The name to use for the function.
    inputs: List of tensors. Inputs to the function.
    outputs: List of tensors. Outputs of the function.

  Returns:
    A FunctionDef protocol buffer.
  """
  # pylint: enable=line-too-long
  func = function_pb2.FunctionDef()
  func.signature.name = name
  func.signature.input_arg.extend([_tensor_to_argdef(graph.get_tensor_by_name(
      i.name)) for i in inputs])
  func.signature.output_arg.extend([_tensor_to_argdef(graph.get_tensor_by_name(
      o.name)) for o in outputs])
  func_arg_placeholders = set([i.name for i in inputs])
  g = ops.get_default_graph()
  for op in graph.get_operations():
    tensor_name = op.values()[0].name
    if tensor_name not in func_arg_placeholders:
      _add_op_node(g, op, func)
  return func


def call_function(func_def, *inputs, **kwargs):
  """Calls the function described by `func_def`.

  This adds a `call` op to the default graph that calls the function described
  by `func_def` with the tensors listed in `inputs` as arguments.  It returns
  the outputs of the call, which are one or more tensors.

  `func_def` is a
  [`FunctionDef`](
  https://www.tensorflow.org/code/tensorflow/core/framework/function.proto)
  protcol buffer describing a
  TensorFlow function.  See [`define_function()`](#define_function) for an
  easy way to create one from a Python function.

  You can pass an optional keyword parameters `name=string` to name the
  added operation.

  `func_def` is automatically added to the function library of the graph if
  needed.

  Args:
    func_def: A `FunctionDef` protocol buffer.
    *inputs: A list of tensors
    **kwargs: Optional keyword arguments.  Can only contain 'name'.

  Returns:
    A list of tensors representing the outputs of the call to `func_def`.

  Raises:
    ValueError: if the arguments are invalid.
  """
  name = kwargs.pop("name", None)
  if kwargs:
    raise ValueError("Unknown keyword arguments: %s" % kwargs.keys())
  func_name = func_def.signature.name
  with ops.op_scope(inputs, name, func_name) as name:
    if len(inputs) != len(func_def.signature.input_arg):
      raise ValueError("Expected number of arguments: %d" %
                       len(func_def.signature.input_arg))
    output_types = [dtypes.DType(x.type) for x in func_def.signature.output_arg]
    # TODO(touts): Pass compute_shapes as "try if function exists"
    g = ops.get_default_graph()
    op = g.create_op(func_name,
                     list(inputs),
                     output_types,
                     name=name,
                     compute_shapes=False)
    if op.outputs:
      if len(op.outputs) == 1:
        return op.outputs[0]
      else:
        return tuple(op.outputs)
    else:
      return op


def define_function(func, input_types):
  """Creates a `FunctionDef` for a python function.

  `func` is a Python function that receives zero or more tensors and returns at
  least one tensor.  It should add ops to the default graph the usual way by
  calling TensorFlow functions such as `tf.constant()`, `tf.matmul()`, etc.

  `input_types` is a dictionary of strings to `tf.Dtype` objects.  Keys are
  names arguments to `func`.  The value indicate the type of tensor expected
  by the function.

  The returned `FunctionDef` protocol buffer is also added to the
  default graph library.  After it has been added you can add calls to
  the function by passing it to `tf.call_function()`, together with a
  list of tensors to use as inputs for the function.

  Notes:

  *  `func` is called once, with `placeholder` tensors of the types specified in
     `input_types` as arguments.
  *  Values returned by `func` must be tensors and they are recorded as being
     the output of the function def.
  *  While `func` is a called, an empty graph is temporarily pushed as the
     default graph.  All ops added by `func` to that graph are part of the body
     of the returned function def.

  Example, but also see the [How To on functions](link_needed).

  ```python
  # A function that receives two tensors x, y and returns their
  # sum and difference.
  def my_func(x, y):
    return x + y, x - y

  # Create a FunctionDef for 'my_func'. (This does not change the default
  graph.)
  my_func_def = tf.define_function(my_func, {'x': tf.float32, 'y': tf.float32})

  # Build the graph, calling the function.
  a = tf.constant([1.0])
  b = tf.constant([2.0])
  c, d = tf.call_function(my_func_def, a, b, name='mycall')
  ```

  Args:
    func: a Python function.
    input_types: dict.  Keys are the names of the arguments of `func`, values
      are their expected `tf.DType`.

  Returns:
    A FunctionDef protocol buffer.

  Raises:
    ValueError: if the arguments are invalid.

  """
  # TODO(touts): Lift the limitation that func can only receive Tensor args.
  if inspect.isfunction(func):
    func_name = func.__name__
  elif inspect.ismethod(func):
    func_name = func.__self__.__name__ + "." + func.__name__
  else:
    raise ValueError("Argument must be a function")
  argspec = inspect.getargspec(func)
  if argspec.varargs or argspec.keywords or argspec.defaults:
    raise ValueError("Only functions with plain arglists are supported.")
  if inspect.isfunction(func):
    if len(argspec.args) != len(input_types):
      raise ValueError("The function must have the same number of arguments "
                       "as the number of specified input types.")
    args = argspec.args
  elif inspect.ismethod(func):
    if len(argspec.args) != 1 + len(input_types):
      raise ValueError(
          "The class function must have the same number of arguments "
          "as the number of specified input types.")
    args = argspec.args[1:]  # 1st argument is the "class" type.

  # Create the func_def object.
  temp_graph = ops.Graph()
  with temp_graph.as_default():
    # List of placeholders for the function_def.
    inputs = []
    # Arglist to call 'func'
    kwargs = {}
    for argname in args:
      if argname not in input_types:
        raise ValueError("Missing type for argument: " + argname)
      argholder = array_ops.placeholder(input_types[argname], name=argname)
      inputs.append(argholder)
      kwargs[argname] = argholder
    # Call func and gather the output tensors.
    outputs = func(**kwargs)
    if not outputs:
      raise ValueError("Function must return at least one tensor")
    # Convenience: if func only returned one value, make it a tuple.
    if not isinstance(outputs, (list, tuple)):
      outputs = (outputs,)
  # Build the FunctionDef
  func_def = graph_to_function_def(temp_graph, func_name, inputs, outputs)
  g = ops.get_default_graph()
  g._add_function(func_def)  # pylint: disable=protected-access
  return func_def


class Defun(object):
  """Decorator used to define TensorFlow functions.

  Use this decorator to make a Python function usable directly as a TensorFlow
  function.

  The decorated function must add ops to the default graph and return zero or
  more `Tensor` objects.  Call the decorator with named arguments, one for each
  argument of the function to decorate, with the expected type of the argument
  as value.

  For example if the function to decorate accepts to `tf.float32` arguments
  named `x` and `y`, call the decorator with:

      @Defun(x=tf.float32, y=tf.float32)
      def foo(x, y):
        ...

  When you call the decorated function it will add `call` ops to the graph.

  Example, but also see the [How To on functions](link_needed).

  ```python
  # Defining the function.
  @tf.Defun(x=tf.float32, y=tf.float32)
  def MyFunc(x, y):
    return x + y, x - y

  # Building the graph.
  a = tf.Constant([1.0])
  b = tf.Constant([2.0])
  c, d = MyFunc(a, b, name='mycall')
  ```

  @@__init__
  """

  def __init__(self, **input_types):
    """Create a `Defun` decorator.

    Args:
      **input_types: Dict mapping string with `tf.DType`
        One key for each argument of the function to decorate.
    """
    self._input_types = input_types

  def __call__(self, f):
    func_def = define_function(f, self._input_types)
    return lambda *args, **kwargs: call_function(func_def, *args, **kwargs)
