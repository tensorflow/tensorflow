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
"""An experimental fork of the Python TensorFlow-function library.

NOTE: functions are currently experimental and subject to change!
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import function
from tensorflow.python.framework import graph_to_function_def
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import tf_inspect

# NOTE(mrry): This is an experimental extension of a core class that wasn't
# designed to be extended, so we disable protected access checks for the
# whole file.
# pylint: disable=protected-access


class _ExperimentalFuncGraph(function._FuncGraph):
  """A helper for construction a function (supporting capture-by-value).

  _ExperimentalFuncGraph overrides ops.Graph's create_op() so that we can keep
  track of every inputs into every op created inside the function.  If
  any input is from other graphs, we keep track of it in self.capture
  and substitute the input with a place holder.

  Each captured input's corresponding place holder is converted into a
  function argument and the caller passes in the captured tensor.
  """

  def __init__(self, capture_by_value, *args, **kwargs):
    super(_ExperimentalFuncGraph, self).__init__(*args, **kwargs)
    self._capture_by_value = capture_by_value
    self._building_function = True
    self._outer_graph = ops.get_default_graph()
    self._vscope = vs.get_variable_scope()
    self._old_custom_getter = self._vscope.custom_getter
    self._captured = {}
    self.extra_inputs = []
    self.extra_args = []
    self.extra_vars = []

  def create_op(self, op_type, inputs, data_types, **kwargs):
    for i, x in enumerate(inputs):
      if x.graph is not self:
        # Referring to a tensor from other graph.
        if x in self._captured:
          # Captured already.
          inputs[i] = self._captured[x]
        elif self._capture_by_value:
          inputs[i] = self._add_tensor_and_parents(x)
        else:
          # Substitute with a placeholder.
          self.extra_inputs.append(x)
          ph = array_ops.placeholder(x.dtype, shape=x.get_shape())
          # pylint: disable=protected-access
          ph._handle_data = x._handle_data
          # pylint: enable=protected-access
          inputs[i] = ph
          self._captured[x] = ph
          self.extra_args.append(ph)
    return super(_ExperimentalFuncGraph, self).create_op(op_type, inputs,
                                                         data_types, **kwargs)

  def _add_tensor_and_parents(self, tensor):
    op = self._add_op_and_parents(tensor.op)
    return op.outputs[tensor.value_index]

  def _add_op_and_parents(self, op):
    op_def = graph_to_function_def._get_op_def(op)
    if op_def.is_stateful:
      raise ValueError("Cannot capture a stateful node (name:%s, type:%s) "
                       "by value." % (op.name, op.type))
    elif op.type in ("Placeholder", "PlaceholderV2"):
      raise ValueError("Cannot capture a placeholder (name:%s, type:%s) "
                       "by value." % (op.name, op.type))

    captured_inputs = [self._add_tensor_and_parents(x) for x in op.inputs]

    captured_op = self.create_op(op.type, captured_inputs,
                                 [o.dtype for o in op.outputs],
                                 name=op.name, attrs=op.node_def.attr,
                                 op_def=op_def)

    for t, captured_t in zip(op.outputs, captured_op.outputs):
      self._captured[t] = captured_t

    return captured_op


class _ExperimentalDefinedFunction(function._DefinedFunction):
  """Overrides _DefinedFunction with support for capture-by-value."""

  def __init__(self,
               func,
               argnames,
               input_types,
               func_name=None,
               grad_func=None,
               python_grad_func=None,
               out_names=None,
               shape_func=None,
               capture_by_value=False,
               **kwargs):
    """Creates an _ExperimentalDefinedFunction.

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
      out_names: An optional list of strings for the function return value
        names.
      shape_func: An optional function mapping an op to a list of static
        output shapes.
      capture_by_value: Boolean (defaults to False). If True, captured values
        will be copied into the function body.
      **kwargs: The keyword arguments. **kwargs is passed to every call
        site of this function.

    Raises:
      ValueError: The function definition is invalid.
    """
    super(_ExperimentalDefinedFunction, self).__init__(
        func, argnames, input_types, func_name, grad_func, python_grad_func,
        out_names, shape_func, **kwargs)
    self._capture_by_value = capture_by_value

  def _create_definition_if_needed(self):
    """Creates the function definition if it's not created yet."""

    if self._definition is not None:
      return

    # Create the func_def object.
    temp_graph = _ExperimentalFuncGraph(capture_by_value=self._capture_by_value)
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
    self._sub_functions = temp_graph._functions

    # Build the FunctionDef
    self._definition = graph_to_function_def.graph_to_function_def(
        temp_graph, temp_graph.get_operations(), inputs, outputs,
        out_names=self._out_names)

    # Extra kwargs are treated as attrs on the function def.
    sig_pre_func_name = self._func_name or function._get_func_name(self._func)
    kwargs_attr = function._parse_kwargs_as_attrs(
        sig_pre_func_name, **self._extra_kwargs)
    for k in kwargs_attr:
      self._definition.attr[k].CopyFrom(kwargs_attr[k])

    # Hash the definition and its dependencies.
    self._hash_str = self._create_hash_str(
        self._definition.signature.input_arg,
        self._definition.signature.output_arg,
        self._definition.node_def)

    # Finally, we decide the function name to use.  If not specified,
    # make up something which is almost certainly unique (but deterministic).
    if not self._func_name:
      self._func_name = "_".join([function._get_func_name(self._func),
                                  self._hash_str])
    self._definition.signature.name = self._func_name
    if self._func.__doc__:
      self._definition.signature.description = self._func.__doc__


class Defun(function.Defun):
  """Experimental version of Defun supporting capture-by-value."""

  def __init__(self, *input_types, **kwargs):
    """Create an experimental `Defun` decorator.

    Args:
      *input_types: A list of `tf.DType`
      **kwargs: Optional keyword arguments (see `function.Defun`) plus:
        capture_by_value - Boolean (defaults to False). If True, captured values
        will be copied into the function body.
    """
    super(Defun, self).__init__(*input_types, **kwargs)

  def __call__(self, func):
    # Various sanity checks on the callable func.
    if not callable(func):
      raise ValueError("func %s must be callable" % func)

    # Func should not use kwargs and defaults.
    argspec = tf_inspect.getargspec(func)
    if argspec.keywords or argspec.defaults:
      raise ValueError("Functions with argument defaults or keyword "
                       "arguments are not supported.")

    # Computes how many arguments 'func' has.
    min_args = len(argspec.args)
    max_args = min_args
    if argspec.varargs:
      max_args = 1000000
    argnames = argspec.args
    if tf_inspect.ismethod(func):
      # 1st argument is the "class" type.
      min_args -= 1
      argnames = argnames[1:]

    if self._input_types:
      # If Defun is given a list of types for the inputs, the number
      # of input types should be compatible with 'func'.
      num = len(self._input_types)
      if num < min_args or num > max_args:
        raise ValueError(
            "The function has fewer arguments than the number of specified "
            "input types.")
      return _ExperimentalDefinedFunction(
          func, argnames, self._input_types, self._func_name, self._grad_func,
          self._python_grad_func, out_names=self._out_names,
          **self._extra_kwargs)

    # 'func' expects no arguments and input types is an empty list.
    if min_args == 0 and max_args == 0:
      return _ExperimentalDefinedFunction(
          func, [], [], self._func_name, self._grad_func,
          self._python_grad_func, out_names=self._out_names,
          **self._extra_kwargs)

    # Input types are unknown. It's an overloaded function and hence
    # its definition needs to be deferred until it's called.
    return function._OverloadedFunction(
        func, argnames, self._func_name, self._grad_func,
        self._python_grad_func, out_names=self._out_names, **self._extra_kwargs)
