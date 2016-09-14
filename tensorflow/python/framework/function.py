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
  node.attr["Tin"].CopyFrom(
      attr_value_pb2.AttrValue(list=attr_value_pb2.AttrValue.ListValue(
          type=[dtype] * num)))
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
  node.attr["out_types"].CopyFrom(
      attr_value_pb2.AttrValue(list=attr_value_pb2.AttrValue.ListValue(
          type=[dtype] * num)))
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
    op_def = graph._get_function(op.type).definition.signature
  else:
    op_def = op_def_registry.get_registered_ops()[op.type]
  # pylint: enable=protected-access
  attrs = _get_node_def_attr(op)
  out_index = 0
  for arg_def in op_def.output_arg:
    if arg_def.number_attr:
      dtype = arg_def.type or attrs[arg_def.type_attr].type
      num = attrs[arg_def.number_attr].i
      node.ret.append(
          _add_output_array(op, out_index, out_index + num, dtype, func))
      out_index += num
    elif arg_def.type_list_attr:
      dtype_lst = attrs[arg_def.type_list_attr].list.type
      num = len(dtype_lst)
      node.ret.append(
          _add_output_list(op, out_index, out_index + num, dtype_lst, func))
      out_index += num
    else:
      node.ret.append(
          _make_argname_from_tensor_name(op.outputs[out_index].name))
      out_index += 1
  inp_index = 0
  for arg_def in op_def.input_arg:
    if arg_def.number_attr:
      dtype = arg_def.type or attrs[arg_def.type_attr].type
      num = attrs[arg_def.number_attr].i
      node.arg.append(
          _add_input_array(op, inp_index, inp_index + num, dtype, func))
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
    graph: Graph.
    name: string. The name to use for the function.
    inputs: List of tensors. Inputs to the function.
    outputs: List of tensors. Outputs of the function.

  Returns:
    A FunctionDef protocol buffer.
  """
  func = function_pb2.FunctionDef()
  func.signature.name = name
  func.signature.input_arg.extend([_tensor_to_argdef(i) for i in inputs])
  func.signature.output_arg.extend([_tensor_to_argdef(o) for o in outputs])
  func_arg_placeholders = set([i.name for i in inputs])
  for op in graph.get_operations():
    tensor_name = op.values()[0].name
    if tensor_name not in func_arg_placeholders:
      _add_op_node(graph, op, func)
  return func


def call_function(func, *inputs, **kwargs):
  """Calls the function described by `func`.

  This adds a `call` op to the default graph that calls the function described
  by `func` with the tensors listed in `inputs` as arguments.  It returns
  the outputs of the call, which are one or more tensors.

  `func` is a `_DefinedFunction` object. See
  [`define_function()`](#define_function) for an easy way to create
  one from a Python function.

  You can pass an optional keyword parameter `name=string` to name the
  added operation.

  You can pass an optional keyword parameter `noinline=True|False` to instruct
  the runtime not to inline the function body into the call site.

  `func` is automatically added to the function library of the graph if
  needed.

  Args:
    func: A `_DefinedFunction` object.
    *inputs: A list of tensors
    **kwargs: Optional keyword arguments.  Can only contain 'name' or
        'noinline'.

  Returns:
    A list of tensors representing the outputs of the call to `func`.

  Raises:
    ValueError: if the arguments are invalid.

  """
  func_def = func.definition
  name = kwargs.pop("name", None)
  noinline = kwargs.pop("noinline", None)
  if noinline is None:
    attrs = None
  else:
    attrs = {}
    attrs["_noinline"] = attr_value_pb2.AttrValue(b=bool(noinline))
  if kwargs:
    raise ValueError("Unknown keyword arguments: %s" % kwargs.keys())
  func_name = func_def.signature.name
  with ops.name_scope(name, func_name, inputs) as name:
    if len(inputs) != len(func_def.signature.input_arg):
      raise ValueError("Expected number of arguments: %d, received: %d" %
                       (len(func_def.signature.input_arg), len(inputs)))
    output_types = [dtypes.DType(x.type) for x in func_def.signature.output_arg]
    # TODO(touts): Pass compute_shapes as "try if function exists"
    g = ops.get_default_graph()
    op = g.create_op(
        func_name,
        list(inputs),
        output_types,
        name=name,
        attrs=attrs,
        compute_shapes=False)
    if op.outputs:
      if len(op.outputs) == 1:
        return op.outputs[0]
      else:
        return tuple(op.outputs)
    else:
      return op


def _get_func_name(func):
  if isinstance(func, _DefinedFunction):
    return func.name
  elif callable(func):
    if inspect.isfunction(func):
      return func.__name__
    elif inspect.ismethod(func):
      return "%s.%s" % (func.__self__.__name__, func.__name__)
    else:  # Probably a class instance with __call__
      return type(func)
  else:
    raise ValueError("Argument must be callable")


def define_function(func,
                    input_types,
                    func_name=None,
                    grad_func=None,
                    python_grad_func=None):
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
  # Alternatively:
  # my_func_def = tf.define_function(my_func, [tf.float32, tf.float32])

  # Build the graph, calling the function.
  a = tf.constant([1.0])
  b = tf.constant([2.0])
  c, d = tf.call_function(my_func_def, a, b, name='mycall')
  ```

  Args:
    func: a Python function.
    input_types: if a dict, keys are the names of the arguments of
      `func`, values are their expected `tf.DType`. Otherwise,
      a list of `tf.DType`s.
    func_name: Pyton string.  If not None, specifies the name to use when
      creating the Function.  By default, introspection on `func` is used to
      generate a name.
    grad_func: If not None, specifies the gradient function. The
               gradient function must satisify the criterion defined in
               function.proto:GradientDef.
    python_grad_func: If not None, specifies the gradient function with the same
               interface as that expected by `tf.RegisterGradient`. This
               will be called by tf.gradients to add the gradient ops to the
               graph. No more than one of {grad_func, python_grad_func} may be
               specified.

  Returns:
    A FunctionDef protocol buffer.

  Raises:
    ValueError: if the arguments are invalid.

  """
  f = _DefinedFunction(func, input_types, func_name, grad_func,
                       python_grad_func)
  # pylint: disable=protected-access
  f.add_to_graph(ops.get_default_graph())
  # pylint: enable=protected-access
  return f


class _DefinedFunction(object):
  """_DefinedFunction encapsulates a function definition and its properties.

  Attributes:
    name: The function name.
    definition: The definition of this function. A FunctionDef proto.
    grad_func_name: If not None, the name of this function's gradient function.
    python_grad_func: A python callable implementing the gradient of
      the function python-side.
  """

  def __init__(self,
               func,
               input_types,
               func_name=None,
               grad_func=None,
               python_grad_func=None):
    """Creates _DefinedFunction.

    Args:
      func:  A python callable which constructs a tf function body.
      input_types: The function's argument types. Can be a tuple, list of
        tf data types, or a dictionary of argument names to their types.
      func_name: The function name. Defaults to None, in which derives from
        'func'.
      grad_func: This function's gradient function, if not None. Defaults
        to None.
      python_grad_func: A python callable implementing the gradient of
        the function python-side.

    Raises:
      ValueError: The function definition is invalid.
    """
    self._func = func
    self._input_types = input_types
    self._func_name = func_name or _get_func_name(func)
    self._grad_func = grad_func
    self._python_grad_func = python_grad_func
    self._definition = None  # Constructed lazily.

    argspec = inspect.getargspec(func)
    if argspec.keywords or argspec.defaults:
      raise ValueError("Functions with argument defaults or keyword "
                       "arguments are not supported.")
    if inspect.isfunction(func):
      if argspec.varargs and (
          len(argspec.args) > len(input_types)) or not argspec.varargs and (
              len(argspec.args) != len(input_types)):
        raise ValueError("The function has fewer arguments "
                         "than the number of specified input types.")
      argnames = argspec.args
    elif inspect.ismethod(func):
      if argspec.varargs and (
          len(argspec.args) > 1 + len(input_types)) or not argspec.varargs and (
              len(argspec.args) != 1 + len(input_types)):
        raise ValueError("The class function has fewer arguments "
                         "than the number of specified input types.")
      # 1st argument is the "class" type.
      argnames = argspec.args[1:]

    self._args = []
    if isinstance(input_types, (list, tuple)):
      for i in range(len(input_types)):
        argname = argnames[i] if i < len(argnames) else ("arg%d" % i)
        argtype = input_types[i]
        self._args.append((argname, argtype))
    else:
      for name in argnames:
        if name not in input_types:
          raise ValueError("Missing type for argument: " + name)
        self._args.append((name, input_types[name]))

  @property
  def name(self):
    """Function name."""
    return self._func_name

  @property
  def definition(self):
    """Function definition proto."""
    self._create_definition_if_needed()
    return self._definition

  @property
  def grad_func_name(self):
    """Its gradient function's name."""
    return self._grad_func.name if self._grad_func else None

  @property
  def python_grad_func(self):
    """Python gradient function callable."""
    return self._python_grad_func

  def _create_definition_if_needed(self):
    """Creates the function definition if it's not created yet."""

    if self._definition is not None:
      return

    # Create the func_def object.
    temp_graph = ops.Graph()
    with temp_graph.as_default():
      # List of placeholders for the function_def.
      inputs = []
      # Arglist to call 'func'
      kwargs = {}
      for (argname, argtype) in self._args:
        argholder = array_ops.placeholder(argtype, name=argname)
        inputs.append(argholder)
        kwargs[argname] = argholder
      # Call func and gather the output tensors.
      if isinstance(self._input_types, (list, tuple)):
        outputs = self._func(*inputs)
      else:
        outputs = self._func(**kwargs)
      if not isinstance(outputs, ops.Tensor) and not outputs:
        raise ValueError("Function must return at least one tensor")
      # Convenience: if func only returned one value, make it a tuple.
      if not isinstance(outputs, (list, tuple)):
        outputs = (outputs,)

    # Build the FunctionDef
    self._definition = graph_to_function_def(temp_graph, self._func_name,
                                             inputs, outputs)
    # pylint: disable=protected-access
    self._sub_functions = temp_graph._functions
    # pylint: enable=protected-access
    self._hash = hash(self._definition.SerializeToString())
    for item in self._sub_functions.items():  # OrderedDict
      self._hash = hash((self._hash, item))

  def __hash__(self):
    self._create_definition_if_needed()
    return self._hash

  def add_to_graph(self, g):
    """Adds this function into the graph g."""
    self._create_definition_if_needed()

    # pylint: disable=protected-access
    # If 'g' has an identical function already, do nothing.
    prev = g._get_function(self.name)
    if prev and (prev._hash == self._hash):
      return

    # Adds this function into 'g'.
    g._add_function(self)
    # pylint: enable=protected-access

    # Ensures related sub-routines are defined in 'g', too.
    for f in self._sub_functions.values():
      f.add_to_graph(g)

    # Adds its gradient function, too.
    if self._grad_func:
      self._grad_func.add_to_graph(g)

  def __call__(self, *args, **kwargs):
    self.add_to_graph(ops.get_default_graph())
    return call_function(self, *args, **kwargs)


class Defun(object):
  """Decorator used to define TensorFlow functions.

  Use this decorator to make a Python function usable directly as a TensorFlow
  function.

  The decorated function must add ops to the default graph and return zero or
  more `Tensor` objects.  Call the decorator with named arguments, one for each
  argument of the function to decorate, with the expected type of the argument
  as value.

  For example if the function to decorate accepts two `tf.float32` arguments
  named `x` and `y`, call the decorator with:

      @Defun(tf.float32, tf.float32)
      def foo(x, y):
        ...

  When you call the decorated function it will add `call` ops to the
  default graph and adds the definition of the function into the
  default graph. Because the addition of the function into the graph
  is deferred, the decorator can be used anywhere in the program.

  Example, but also see the [How To on functions](link_needed).

  ```python
  # Defining the function.
  @tf.Defun(tf.float32, tf.float32)
  def MyFunc(x, y):
    return x + y, x - y

  # Building the graph.
  a = tf.Constant([1.0])
  b = tf.Constant([2.0])
  c, d = MyFunc(a, b, name='mycall')
  ```

  @@__init__

  """

  def __init__(self, *input_type_list, **input_types):
    """Create a `Defun` decorator.

    Args:
      *input_type_list: A list of `tf.DType`
      **input_types: Dict mapping string with `tf.DType`
        One key for each argument of the function to decorate.

       Note that these optional keyword arguments are also accepted:
         func_name - (optional).  A python string, the name to use to declare
           this `Function` in the graph.

         grad_func - (optional).  A function implementing the gradient of the
           function-to-register.  This is usually a previously
           `Defun`-registered Python callable.

         python_grad_func - (optional).  A function implementing the gradient of
           the function python-side. This function must take the current op and
           the gradients w.r.t. its outputs, and return the gradients w.r.t. the
           inputs (identical to the interface expected by
           `tf.RegisterGradient`).
    """
    self._func_name = input_types.pop("func_name", None)
    self._grad_func = input_types.pop("grad_func", None)
    self._python_grad_func = input_types.pop("python_grad_func", None)
    assert not input_type_list or not input_types, (
        "Can't specify both *input_type_list and **input_types")
    self._input_types = input_types
    self._input_type_list = input_type_list

  def __call__(self, f):
    if self._input_types:
      inp_types = self._input_types
    else:
      inp_types = self._input_type_list
    return _DefinedFunction(f, inp_types, self._func_name, self._grad_func,
                            self._python_grad_func)
