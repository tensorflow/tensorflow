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
"""Python front-end supports for functions.

NOTE: functions are currently experimental and subject to change!
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
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
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import compat


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
  node.arg.extend(
      [_make_argname_from_tensor_name(x.name) for x in op.inputs[start:limit]])
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


def _add_op_node(op, func):
  """Converts an op to a function def node and add it to `func`."""
  node = function_pb2.FunctionDef.Node()
  node.op = op.type
  # pylint: disable=protected-access
  if hasattr(op, "_sig"):
    op_def = getattr(op, "_sig")
  else:
    op_def = op_def_registry.get_registered_ops()[op.type]
  # pylint: enable=protected-access
  attrs = _get_node_def_attr(op)
  if not op_def.output_arg:
    node.ret.append(_make_argname_from_tensor_name(op.name))
  else:
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
      node.arg.extend([
          _make_argname_from_tensor_name(op.inputs[i].name)
          for i in range(inp_index, inp_index + num)
      ])
      inp_index += num
    else:
      node.arg.append(_make_argname_from_tensor_name(op.inputs[inp_index].name))
      inp_index += 1
  node.dep.extend(
      [_make_argname_from_tensor_name(x.name) for x in op.control_inputs])
  for k, v in _get_node_def_attr(op).items():
    node.attr[k].CopyFrom(v)
  func.node.extend([node])


def _graph_to_function_def(graph, inputs, outputs):
  """Returns `graph` as a `FunctionDef` protocol buffer.

  This method creates a [`FunctionDef`](
  https://www.tensorflow.org/code/tensorflow/core/framework/function.proto)
  protocol buffer that contains all the ops present in the graph.  The
  graph effectively becomes the body of the function.

  The arguments `inputs` and `outputs` will be listed as the inputs
  and outputs tensors of the function.  They must be lists of
  tensors present in the graph.  The lists can optionally be empty.

  Args:
    graph: Graph.
    inputs: List of tensors. Inputs to the function.
    outputs: List of tensors. Outputs of the function.

  Returns:
    A FunctionDef protocol buffer.
  """
  func = function_pb2.FunctionDef()
  func.signature.name = "_"
  func.signature.input_arg.extend([_tensor_to_argdef(i) for i in inputs])
  func.signature.output_arg.extend([_tensor_to_argdef(o) for o in outputs])
  func_arg_placeholders = set([i.name for i in inputs])
  for op in graph.get_operations():
    if op.values() and (op.values()[0].name in func_arg_placeholders):
      continue
    _add_op_node(op, func)
  return func


def _parse_kwargs_as_attrs(**kwargs):
  """Parses **kwargs into a node's attributes."""
  attrs = {}
  noinline = kwargs.pop("noinline", None)
  if noinline is not None:
    attrs["_noinline"] = attr_value_pb2.AttrValue(b=bool(noinline))
  if kwargs:
    raise ValueError("Unknown keyword arguments: %s" % kwargs.keys())
  return attrs


def _call(sig, *inputs, **kwargs):
  """Adds a node calling a function.

  This adds a `call` op to the default graph that calls the function
  of signature `sig`, passing the tensors in `inputs` as arguments.
  It returns the outputs of the call, which are one or more tensors.

  `sig` is OpDefArg.a `_DefinedFunction` object.

  You can pass an optional keyword parameter `name=string` to name the
  added operation.

  You can pass an optional keyword parameter `noinline=True|False` to
  instruct the runtime not to inline the function body into the call
  site.

  Args:
    sig: OpDefArg. The signature of the function.
    *inputs: arguments to the function.
    **kwargs: Optional keyword arguments.  Can only contain 'name' or
        'noinline'.

  Returns:
     A Tensor if the function returns a single value; a list of Tensors
     if the functio returns multiple value; the Operation if the function
     returns no values.

  Raises:
    ValueError: if the arguments are invalid.

  """
  if len(inputs) != len(sig.input_arg):
    raise ValueError("Expected number of arguments: %d, received: %d" %
                     (len(sig.input_arg), len(inputs)))
  name = kwargs.pop("name", None)
  attrs = _parse_kwargs_as_attrs(**kwargs)
  g = ops.get_default_graph()
  func_name = sig.name
  output_types = [dtypes.DType(x.type) for x in sig.output_arg]
  with ops.name_scope(name, func_name, inputs) as name:
    op = g.create_op(
        func_name,
        list(inputs),
        output_types,
        name=name,
        attrs=attrs,
        compute_shapes=False)
  setattr(op, "_sig", sig)  # Remember the signature.
  if op.outputs:
    if len(op.outputs) == 1:
      return op.outputs[0]
    else:
      return tuple(op.outputs)
  else:
    return op


def _get_func_name(func):
  if callable(func):
    if inspect.isfunction(func):
      return func.__name__
    elif inspect.ismethod(func):
      return "%s.%s" % (func.__self__.__name__, func.__name__)
    else:  # Probably a class instance with __call__
      return type(func)
  else:
    raise ValueError("Argument must be callable")


class _FuncGraph(ops.Graph):
  """A helper for construction a function.

  _FuncGraph overrides ops.Graph's create_op() so that we can keep
  track of every inputs into every op created inside the function.  If
  any input is from other graphs, we keep track of it in self.capture
  and substitue the input with a place holder.

  Each captured input's corresponding place holder is converted into a
  function argument and the caller passes in the captured tensor.

  """

  def __init__(self, *args, **kwargs):
    super(_FuncGraph, self).__init__(*args, **kwargs)
    self._building_function = True
    self._outer_graph = ops.get_default_graph()
    self._vscope = vs.get_variable_scope()
    self._old_custom_getter = self._vscope.custom_getter
    self._captured = {}
    self.extra_inputs = []
    self.extra_args = []
    self.extra_vars = []

  def getvar(self,
             name,
             shape=None,
             dtype=None,
             initializer=None,
             trainable=True,
             collections=None,
             **kwargs):
    """A custom variable getter."""
    # Here, we switch the default graph to the outer graph and ask the
    # variable scope in which the function is defined to give us the
    # variable. The variable is stashed in extra_vars and returned to
    # the caller.
    #
    # We capture these variables so that the variable definition is
    # hoisted upward to the outer most graph.
    with self._outer_graph.as_default():
      # pylint: disable=protected-access
      var = self._vscope.get_variable(
          vs._get_default_variable_store(),
          name,
          shape=shape,
          dtype=dtype,
          initializer=initializer,
          trainable=trainable,
          collections=collections)
      self.extra_vars.append(var)
      return var

  def create_op(self, op_type, inputs, data_types, **kwargs):
    for i, x in enumerate(inputs):
      if x.graph is not self:
        # Referring to a tensor from other graph.
        if x in self._captured:
          # Captured already.
          inputs[i] = self._captured[x]
        else:
          # Substitute with a placeholder.
          self.extra_inputs.append(x)
          ph = array_ops.placeholder(x.dtype, shape=x.get_shape())
          inputs[i] = ph
          self._captured[x] = ph
          self.extra_args.append(ph)
    return super(_FuncGraph, self).create_op(op_type, inputs, data_types,
                                             **kwargs)


def get_extra_vars():
  """Returns the captured variables by the function.

  Returns:
    If the default graph is being used to define a function, the
    returned list of variables are those created inside the function
    body so far. Otherwise, returns an empty list.
  """
  g = ops.get_default_graph()
  if isinstance(g, _FuncGraph):
    return g.extra_vars
  else:
    return []


def get_extra_inputs():
  """Returns the captured input tensors by the function.

  Returns:
    If the default graph is being used to define a function, the
    returned list of tensors are those accessed inside the function body
    but defined outside the function body so far. Otherwise, returns an
    empty list.

  """
  g = ops.get_default_graph()
  if isinstance(g, _FuncGraph):
    return g.extra_inputs
  else:
    return []


def get_extra_args():
  """Returns the corresponding function arguments for the captured inputs.

  Returns:
    If the default graph is being used to define a function, the
    returned list of place holders are those used inside the function
    body corresponding those returned by get_extra_inputs(). Otherwise,
    returns an empty list.

  """
  g = ops.get_default_graph()
  if isinstance(g, _FuncGraph):
    return g.extra_args
  else:
    return []


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
               argnames,
               input_types,
               func_name=None,
               grad_func=None,
               python_grad_func=None,
               **kwargs):
    """Creates _DefinedFunction.

    Args:
      func:  A python callable which constructs a tf function body.
      argnames: A list of strings for function argument names.
      input_types: The function's argument types. Can be a tuple, list of
        tf data types.
      func_name: The function name. Defaults to None, in which derives from
        'func'.
      grad_func: This function's gradient function, if not None. Defaults
        to None.
      python_grad_func: A python callable implementing the gradient of
        the function python-side.
      **kwargs: The keyword arguments. **kwargs is passed to every call
        site of this function.

    Raises:
      ValueError: The function definition is invalid.

    """
    self._func = func
    self._input_types = input_types
    self._func_name = func_name
    self._grad_func = grad_func
    self._python_grad_func = python_grad_func
    self._extra_kwargs = kwargs
    self._definition = None  # Constructed lazily.

    self._args = []
    assert isinstance(input_types, (list, tuple))
    for i in range(len(input_types)):
      argname = argnames[i] if i < len(argnames) else ("arg%d" % i)
      argtype = input_types[i]
      self._args.append((argname, argtype))

  @property
  def name(self):
    """Function name."""
    self._create_definition_if_needed()
    return self._func_name

  @property
  def definition(self):
    """Function definition proto."""
    self._create_definition_if_needed()
    return self._definition

  def set_grad_func(self, grad_func):
    """Specifies the gradient function of this function."""
    assert not self._grad_func
    assert isinstance(grad_func, _DefinedFunction)
    self._grad_func = grad_func

  @property
  def grad_func_name(self):
    """Its gradient function's name."""
    return self._grad_func.name if self._grad_func else None

  @property
  def python_grad_func(self):
    """Python gradient function callable."""
    return self._python_grad_func

  @property
  def declared_input_types(self):
    """Returns the list of data types of explicit declared inputs."""
    return self._input_types

  @property
  def captured_inputs(self):
    """Returns the list of implicitly captured inputs."""
    return self._extra_inputs

  def _create_definition_if_needed(self):
    """Creates the function definition if it's not created yet."""

    if self._definition is not None:
      return

    # Create the func_def object.
    temp_graph = _FuncGraph()
    with temp_graph.as_default():
      # List of placeholders for the function_def.
      inputs = []
      for (argname, argtype) in self._args:
        argholder = array_ops.placeholder(argtype, name=argname)
        inputs.append(argholder)
      # Call func and gather the output tensors.
      with vs.variable_scope("", custom_getter=temp_graph.getvar):
        outputs = self._func(*inputs)
      # If func only returned one value, make it a tuple.
      if not isinstance(outputs, (list, tuple)):
        outputs = (outputs,)
      if any([_ is None for _ in outputs]):
        raise ValueError("Function can not return None.")
      # Ensures each output is a Tensor.
      outputs = [ops.convert_to_tensor(_) for _ in outputs]
    self._extra_inputs = temp_graph.extra_inputs
    inputs.extend(temp_graph.extra_args)

    # Build the FunctionDef
    self._definition = _graph_to_function_def(temp_graph, inputs, outputs)

    # Extra kwargs are treated as attrs on the function def.
    kwargs_attr = _parse_kwargs_as_attrs(**self._extra_kwargs)
    for k in kwargs_attr:
      self._definition.attr[k].CopyFrom(kwargs_attr[k])

    # Hash the definition and its dependencies.
    hasher = hashlib.sha1()

    def _hash_func_def():
      """Hash the function definition agnostic to node/map ordering."""

      def update_num(n):
        hasher.update(compat.as_bytes("%x" % n))

      def update_str(s):
        update_num(len(s))
        hasher.update(compat.as_bytes(s))

      def update_strs(slist):
        update_num(len(slist))
        for s in slist:
          update_str(s)

      for n in sorted(self._definition.node, key=lambda n: n.ret[0]):
        update_strs(n.ret)
        update_str(n.op)
        update_strs(n.arg)
        update_strs(n.dep)
        update_num(len(n.attr))
        # NOTE: protobuf map serialization does not guarantee ordering.
        for k in sorted(n.attr):
          update_str(k)
          update_str(n.attr[k].SerializeToString())

    _hash_func_def()
    # pylint: disable=protected-access
    self._sub_functions = temp_graph._functions
    for subname in sorted(self._sub_functions.keys()):
      hasher.update(compat.as_bytes(self._sub_functions[subname]._hash_str))
    # pylint: enable=protected-access

    # Uses the first 8 bytes sha1 hash digest as the __hash__.
    self._hash_str = hasher.hexdigest()[:8]
    self._hash = int(self._hash_str, 16)

    # Finally, we decide the function name to use.  If not specified,
    # make up something which is almost certainly unique.
    if not self._func_name:
      self._func_name = "_".join([_get_func_name(self._func), self._hash_str])
    self._definition.signature.name = self._func_name
    if self._func.__doc__:
      self._definition.signature.description = self._func.__doc__

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
    args = [ops.convert_to_tensor(_) for _ in args] + self._extra_inputs
    return _call(self._definition.signature, *args, **kwargs)

# NOTE: The list needs to be extended when more data types are added.
_DTYPE_TO_STR = {
    dtypes.float16: "f16",
    dtypes.float32: "f32",
    dtypes.float64: "f64",
    dtypes.int32: "i32",
    dtypes.uint8: "i8",
    dtypes.uint16: "u16",
    dtypes.int16: "i16",
    dtypes.int8: "i8",
    dtypes.string: "s",
    dtypes.complex64: "c64",
    dtypes.complex128: "c128",
    dtypes.int64: "i64",
    dtypes.bool: "b",
    dtypes.qint8: "qi8",
    dtypes.quint8: "qu8",
    dtypes.qint16: "qi16",
    dtypes.quint16: "qu16",
    dtypes.qint32: "qi32",
    dtypes.bfloat16: "b16"
}


def _type_list_to_str(types):
  if any([_ not in _DTYPE_TO_STR for _ in types]):
    raise ValueError("Unsupported dtypes: %s" % types)
  return "".join([_DTYPE_TO_STR[_] for _ in types])


class _OverloadedFunction(object):
  """_OverloadedFunction encapsulates an overloaded function.

  _OverloadedFunction maintains a mapping from input types to
  instantiated _DefinedFunction in self._overload.

  """

  def __init__(self,
               func,
               argnames,
               func_name=None,
               grad_func=None,
               python_grad_func=None,
               **kwargs):
    """Creates _DefinedFunction.

    Args:
      func:  A python callable which constructs a tf function body.
      argnames: A list of strings for function argument names.
      func_name: The function name. Defaults to None, in which derives from
        'func'.
      grad_func: This function's gradient function, if not None. Defaults
        to None.
      python_grad_func: A python callable implementing the gradient of
        the function python-side.
      **kwargs: The keyword arguments. **kwargs is passed to every call
        site of this function.

    Raises:
      ValueError: The function definition is invalid.

    """
    self._func = func
    self._argnames = argnames
    self._func_name = func_name
    assert grad_func is None or isinstance(grad_func, _OverloadedFunction)
    self._grad_func = grad_func
    self._python_grad_func = python_grad_func
    self._extra_kwargs = kwargs
    self._overload = {}

  def instantiate(self, input_types):
    """Instantiate this function given input argument types.

    Args:
      input_types: A list of data types for the inputs.

    Returns:
      _DefinedFunction for the given input types.

    """
    # Stringify the type list.
    key = _type_list_to_str(input_types)
    defined = self._overload.get(key)
    if not defined:
      # If not defined yet, define the function given the input types.
      name = self._func_name
      if name is not None:
        name = "_".join([name, key])
      defined = _DefinedFunction(self._func, self._argnames, input_types, name,
                                 None, self._python_grad_func,
                                 **self._extra_kwargs)
      _ = defined.name  # Fully instantiate the function definition.
      if self._grad_func:
        # If _grad_func is given, it is another
        # _OverloadedFunction. We need to instantiate it with the
        # right input types.
        output_types = [
            dtypes.DType(_.type)
            for _ in defined.definition.signature.output_arg
        ]
        # pylint: disable=protected-access
        defined._grad_func = self._grad_func.instantiate(input_types +
                                                         output_types)
        # pylint: enable=protected-access
      self._overload[key] = defined
    return defined

  def __call__(self, *args, **kwargs):
    input_types = []
    args = list(args)
    for (i, x) in enumerate(args):
      x = ops.convert_to_tensor(x)
      if not isinstance(x, ops.Tensor):
        raise ValueError("Expect a Tensor but get ", x)
      input_types.append(x.dtype)
      args[i] = x
    return self.instantiate(input_types)(*args, **kwargs)


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

  def __init__(self, *input_types, **kwargs):
    """Create a `Defun` decorator.

    Args:
      *input_types: A list of `tf.DType`
      **kwargs: Optional keyword arguments, including
         func_name - (optional).  A python string, the name to use to
           declare this `Function` in the graph.

         grad_func - (optional).  A function implementing the gradient
           of the function-to-register.  This is either a
           `_DefinedFunction` or a `Declare` object. The gradient
           function must satisify the criterion defined in
           function.proto:GradientDef.

         python_grad_func - (optional).  A function implementing the
           gradient of the function python-side. This function must
           take the current op and the gradients w.r.t. its outputs,
           and return the gradients w.r.t. the inputs. That is it must
           implement the interface expected by `tf.RegisterGradient`).
           This will be called by tf.gradients to add the gradient ops
           to the graph. At most one of grad_func and python_grad_func
           can be specified.
    """
    self._input_types = input_types
    self._func_name = kwargs.pop("func_name", None)
    self._grad_func = kwargs.pop("grad_func", None)
    self._python_grad_func = kwargs.pop("python_grad_func", None)
    self._extra_kwargs = kwargs

  def __call__(self, func):
    # Various sanity checks on the callable func.
    if not callable(func):
      raise ValueError("func %s must be callable" % func)

    # Func should not use kwargs and defaults.
    argspec = inspect.getargspec(func)
    if argspec.keywords or argspec.defaults:
      raise ValueError("Functions with argument defaults or keyword "
                       "arguments are not supported.")

    # Computes how many arguments 'func' has.
    min_args = len(argspec.args)
    max_args = min_args
    if argspec.varargs:
      max_args = 1000000
    argnames = argspec.args
    if inspect.ismethod(func):
      # 1st argument is the "class" type.
      min_args -= 1
      argnames = argnames[1:]

    if self._input_types:
      # If Defun is given a list of types for the inputs, the number
      # of of input types should be compatible with 'func'.
      num = len(self._input_types)
      if num < min_args or num > max_args:
        raise ValueError(
            "The function has fewer arguments than the number of specified "
            "input types.")
      return _DefinedFunction(func, argnames, self._input_types,
                              self._func_name, self._grad_func,
                              self._python_grad_func, **self._extra_kwargs)

    # 'func' expects no arguments and input types is an empty list.
    if min_args == 0 and max_args == 0:
      return _DefinedFunction(func, [], [], self._func_name, self._grad_func,
                              self._python_grad_func, **self._extra_kwargs)

    # Input types are unknown. It's an overloaded function and hence
    # its definition needs to be deferred until it's called.
    return _OverloadedFunction(func, argnames, self._func_name, self._grad_func,
                               self._python_grad_func, **self._extra_kwargs)


class Declare(object):
  """Declares a TensorFlow function.

  The object represents a TensorFlow function which will be defined
  later during a graph construction.

  For example,
    # Declares  a function Foo, which takes a tf.int32 and a
    # tf.float32 as inputs and returns a tf.float32 as its output.
    foo = Declare("Foo", [tf.int32, tf.float32], [tf.float32])

    # Defines a function Bar calls Foo.
    @tf.Defun(tf.float32)
    def Bar(x):
      return foo(6, x)

    # Defines Foo.
    @tf.Defun(tf.int32, tf.float32)
    def Foo(n, x):
       ...  # Calculation.
       return result
  """

  def __init__(self, func_name, input_types, output_types):
    """Creates a `Declare` object.

    Args:
      func_name: The name of the function.
      input_types: A list of data types of function arguments.
      output_types: A list of data types of function return values.
    """
    self._sig = op_def_pb2.OpDef()
    self._sig.name = func_name

    def _to_argdef_list(types):
      return [op_def_pb2.OpDef.ArgDef(type=_.as_datatype_enum) for _ in types]

    self._sig.input_arg.extend(_to_argdef_list(input_types))
    self._sig.output_arg.extend(_to_argdef_list(output_types))

  def __call__(self, *inputs, **kwargs):
    inputs = [ops.convert_to_tensor(_) for _ in inputs]
    return _call(self._sig, *inputs, **kwargs)
