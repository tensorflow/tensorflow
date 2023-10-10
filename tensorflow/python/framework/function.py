# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

NOTE: At this time, functions are experimental and subject to change!. Proceed
with caution.
"""

import collections
import hashlib

from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import function_pb2
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.eager import context
from tensorflow.python.framework import c_api_util
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_to_function_def
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import compat
from tensorflow.python.util import function_utils
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_inspect


# TODO(b/136040013): Drop support for Defun.
class Defun(object):
  """Obsolete. Slated for deletion. Please use tf.function instead.

  Known feature gaps while migrating to tf.function (could be outdated):
  - tf.function doesn’t support Send/Recv capability since it doesn’t share
    rendezvous with the main graph but always creates a new one.
  - tf.function doesn’t support custom gradient function directly, instead you
    need to define the function inside a tf.custom_gradient wrapper together
    with the gradient function.
  - Unlike Defun, Keras layers used inside a tf.function need to be created only
    once to avoid variable recreation.
  - Defun respects the device assignments and applies them to the function body
    but tf.function needs it to be done manually.
  - Defun might prune out unused ops automatically but tf.function doesn't.

  Limitations of Defun:
  - Original source locations are not preserved so errors do not include
    full/valid stack traces.
  - Only supports linear sequence of arguments and return values, putting the
    burden on the caller to pack/unpack everything across a Defun boundary into
    tuples (as opposed to passing list and dict-like structures directly).
  - Does not support overloading or late-bound specializations.
  - Has its own way for defining gradient overrides which does not follow
    current conventions.
  - Cannot support imperative control flow or automatic control dependencies.
  - Does not reflect statefulness in the graph and has a calling convention that
    differs from how more modern tools interact.
  - Is only compatible with graph building mode.

  Decorator used to define TensorFlow functions.

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

  When you call the decorated function, it adds the `call` ops to the
  default graph. In addition, it adds the definition of the function into the
  default graph. Because the addition of the function into the graph
  is deferred, the decorator can be used anywhere in the program.

  Any variables created inside of the function are hoisted into the outer graph.
  Note that the variables are created in the variable scope that was active
  during the first call to the function. Subsequent function calls will refer to
  the same set of variables.

  Definitions of functions in a graph are frozen as soon as the graph is used to
  create a session. However, new functions and new calls to existing functions
  may be added to the graph, with the new functions themselves becoming
  immediately frozen.

  Example, but also see the [How To on functions](link_needed).

  ```python
  # Defining the function.
  @tf.Defun(tf.float32, tf.float32)
  def MyFunc(x, y):
    return x + y, x - y

  # Building the graph.
  a = tf.constant([1.0])
  b = tf.constant([2.0])
  c, d = MyFunc(a, b, name='mycall')
  ```
  """

  def __init__(self, *input_types, **kwargs):
    """Create a `Defun` decorator.

    Args:
      *input_types: A list of `tf.DType`
      **kwargs: Optional keyword arguments, including
         func_name - (optional).  A python string, the name to use to
           declare this `Function` in the graph.

         grad_func - (optional).  A function implementing the gradient
           of the function-to-register.  This is must be a
           `_DefinedFunction` object. The gradient
           function must satisfy the criterion defined in
           function.proto:GradientDef.

         python_grad_func - (optional).  A function implementing the
           gradient of the function python-side. This function must
           take the current op and the gradients w.r.t. its outputs,
           and return the gradients w.r.t. the inputs. That is it must
           implement the interface expected by `tf.RegisterGradient`).
           This will be called by tf.gradients to add the gradient ops
           to the graph. At most one of grad_func and python_grad_func
           can be specified.

         out_names = (optional). A list of strings, one per output
           tensor.

         shape_func - (optional). A function taking the op and returning a list
           of static shapes to set for the function's outputs.
    """
    self._input_types = input_types
    self._func_name = kwargs.pop("func_name", None)
    self._grad_func = kwargs.pop("grad_func", None)
    self._python_grad_func = kwargs.pop("python_grad_func", None)
    self._out_names = kwargs.pop("out_names", None)
    self._extra_kwargs = kwargs

  def __call__(self, func):
    # Various sanity checks on the callable func.
    if not callable(func):
      raise ValueError(f"Function {func} must be a callable.")

    # Func should not use kwargs and defaults.
    argspec = tf_inspect.getargspec(func)
    if argspec.keywords or argspec.defaults:
      raise ValueError(
          "Functions with argument defaults or keywords arguments are not "
          f"supported. {func} has defaults {argspec.defaults} and keywords "
          f"{argspec.keywords}.")

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
            "The number of tf.function input types is not compatible with the "
            f"allowed arguments of {func}. The tf.function have {num} input "
            f"types, while the python function allows minimum {min_args} and "
            f"maximum {max_args} arguments.")
      return _DefinedFunction(
          func,
          argnames,
          self._input_types,
          self._func_name,
          self._grad_func,
          self._python_grad_func,
          out_names=self._out_names,
          **self._extra_kwargs)

    # 'func' expects no arguments and input types is an empty list.
    if min_args == 0 and max_args == 0:
      return _DefinedFunction(
          func, [], [],
          self._func_name,
          self._grad_func,
          self._python_grad_func,
          out_names=self._out_names,
          **self._extra_kwargs)

    # Input types are unknown. It's an overloaded function and hence
    # its definition needs to be deferred until it's called.
    return _OverloadedFunction(
        func,
        argnames,
        self._func_name,
        self._grad_func,
        self._python_grad_func,
        out_names=self._out_names,
        **self._extra_kwargs)


class _DefinedFunctionDeleter(object):
  """Unregister function from eager context."""

  __slots__ = ["name"]

  def __init__(self, name):
    self.name = name

  def __del__(self):
    try:
      context.remove_function(self.name)
    except TypeError:
      # Suppress some exceptions, mainly for the case when we're running on
      # module deletion. Things that can go wrong include the context module
      # already being unloaded, self._handle._handle_data no longer being
      # valid, and so on. Printing warnings in these cases is silly
      # (exceptions raised from __del__ are printed as warnings to stderr).
      pass  # 'NoneType' object is not callable when the handle has been
      # partially unloaded.
    except AttributeError:
      pass  # 'NoneType' object has no attribute 'eager_mode' when context has
      # been unloaded. Will catch other module unloads as well.


class _DefinedFunction(object):
  """_DefinedFunction encapsulates a function definition and its properties.

  Attributes:
    name: The function name.
    definition: The definition of this function. A FunctionDef proto.
    cached_definition: Same as definition. Needed to match AtomicFunction API.
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
               out_names=None,
               shape_func=None,
               capture_by_value=False,
               allowlisted_stateful_ops=None,
               capture_resource_var_by_value=True,
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
      out_names: An optional list of strings for the function return value
        names.
      shape_func: An optional function mapping an op to a list of static
        output shapes.
      capture_by_value: Boolean (defaults to False). If True, captured values
        will be copied into the function body.
      allowlisted_stateful_ops: A set of ops that if stateful we ignore and
        copy into the function body, when `capture_by_value` is True.
      capture_resource_var_by_value: Boolean (defaults to True). If False,
        captured resource variable returns the handle instead of value.
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
    self._out_names = out_names
    self._shape_func = shape_func
    self._capture_by_value = capture_by_value
    self._allowlisted_stateful_ops = allowlisted_stateful_ops
    if self._allowlisted_stateful_ops is None:
      self._allowlisted_stateful_ops = set()
    self._capture_resource_var_by_value = capture_resource_var_by_value
    self._extra_kwargs = kwargs
    # Constructed only when C API is disabled, lazily
    self._definition = None
    # Constructed only when C API is enabled, lazily
    self._c_func = None
    self._function_deleter = None
    self._sub_functions = {}  # Constructed with _definition or _c_func
    # pylint: disable=protected-access
    device_funcs = ops.get_default_graph()._device_functions_outer_to_inner
    # pylint: enable=protected-access

    # Get the innermost device if possible.
    self._caller_device = device_funcs[-1] if device_funcs else None

    # Cached OpDef for this function. When C API is enabled, this is
    # the only part of FunctionDef that we cache in Python. When C API
    # is disabled the whole _definition is available and this is simply
    # another reference to _definition.signature
    self._op_def = None

    assert isinstance(input_types, (list, tuple))
    self._arg_types = input_types
    self._arg_names = [argnames[i] if i < len(argnames) else ("arg%d" % i)
                       for i in range(len(input_types))]

  @property
  def name(self):
    """Function name."""
    self._create_definition_if_needed()
    return self._func_name

  @property
  def cached_definition(self):
    return self.definition

  @property
  def definition(self):
    """Function definition proto."""
    self._create_definition_if_needed()
    if self._c_func:
      with c_api_util.tf_buffer() as buf:
        with self._c_func.get() as func:
          c_api.TF_FunctionToFunctionDef(func, buf)
          fdef = function_pb2.FunctionDef()
          proto_data = c_api.TF_GetBuffer(buf)
          fdef.ParseFromString(compat.as_bytes(proto_data))
          with ops.init_scope():
            if context.executing_eagerly():
              context.add_c_function(func)
              self._function_deleter = _DefinedFunctionDeleter(
                  fdef.signature.name)
      return fdef
    return self._definition

  @property
  def _signature(self):
    self._create_definition_if_needed()
    return self._op_def

  def set_grad_func(self, grad_func):
    """Specifies the gradient function of this function."""
    assert not self._grad_func
    assert isinstance(grad_func, _DefinedFunction)
    self._grad_func = grad_func

  @property
  def grad_func_name(self):
    """Returns the name of the gradient function."""
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
    self._create_definition_if_needed()
    return self._extra_inputs

  @property
  def stateful_ops(self):
    """Returns the list of stateful ops in function definition.

    Returns:
      A list of (op.name, op.type) pairs.
    """
    self._create_definition_if_needed()
    return self._stateful_ops

  def _create_definition_if_needed(self):
    """Creates the function definition if it's not created yet."""
    with context.graph_mode():
      self._create_definition_if_needed_impl()

  def _create_definition_if_needed_impl(self):
    """This is not what you want, see _create_definition_if_needed."""
    if self._definition is not None or self._c_func is not None:
      return

    # Copy variable collections (by reference) from the parent graph such that
    # name based variable sharing (e.g. via tf.make_template) works between the
    # func graph and parent graph.
    variable_keys = []
    variable_keys.extend(ops.GraphKeys._VARIABLE_COLLECTIONS)  # pylint: disable=protected-access
    variable_keys.append(vs._VARSTORE_KEY)  # pylint: disable=protected-access

    parent_graph = ops.get_default_graph()
    collections_ref = {
        key: parent_graph.get_collection_ref(key) for key in variable_keys}

    temp_graph = func_graph_from_py_func(
        self._func,
        self._arg_names,
        self._arg_types,
        self._func_name,
        self._capture_by_value,
        self._caller_device,
        collections_ref=collections_ref,
        allowlisted_stateful_ops=self._allowlisted_stateful_ops,
        capture_resource_var_by_value=self._capture_resource_var_by_value)

    self._extra_inputs = temp_graph.extra_inputs
    # pylint: disable=protected-access
    self._sub_functions = temp_graph._functions
    # pylint: enable=protected-access

    # Extra kwargs are treated as attrs on the function def.
    if self._func_name:
      base_func_name = self._func_name
    else:
      base_func_name = function_utils.get_func_name(self._func)
      if self._grad_func:
        base_func_name += ("_%s" % self._grad_func.name)
    kwargs_attr = _parse_kwargs_as_attrs(base_func_name, **self._extra_kwargs)

    # FIXME(feyu): C API is always enabled now. The if-true branch never runs.
    if not temp_graph._c_graph:  # pylint: disable=protected-access
      # Build the FunctionDef
      self._definition = graph_to_function_def.graph_to_function_def(
          temp_graph,
          temp_graph.get_operations(),
          temp_graph.inputs,
          temp_graph.outputs,
          out_names=self._out_names)

      for k in kwargs_attr:
        self._definition.attr[k].CopyFrom(kwargs_attr[k])

      # Hash the definition and its dependencies.
      self._hash_str = self._create_hash_str(
          self._definition.signature.input_arg,
          self._definition.signature.output_arg, self._definition.node_def)

      # Finally, we decide the function name to use.  If not specified,
      # make up something which is almost certainly unique (but deterministic).
      if not self._func_name:
        self._func_name = "_".join([base_func_name, self._hash_str])
      self._definition.signature.name = self._func_name
      if self._func.__doc__:
        self._definition.signature.description = self._func.__doc__

      self._op_def = self._definition.signature
    else:  # C API is enabled
      output_names = ([compat.as_bytes(x) for x in self._out_names]
                      if self._out_names else [])
      description = self._func.__doc__ or None
      # pylint: disable=protected-access
      with temp_graph._c_graph.get() as c_graph:
        c_func = c_api.TF_GraphToFunction_wrapper(
            c_graph,
            base_func_name,
            self._func_name is None,  # append_hash_to_fn_name
            None,  # opers
            [t._as_tf_output() for t in temp_graph.inputs],
            [t._as_tf_output() for t in temp_graph.outputs],
            output_names,
            [],  # control_outputs
            [],  # control_output_names
            None,  # opts
            description)
      self._c_func = c_api_util.ScopedTFFunction(c_func, base_func_name)
      # pylint: enable=protected-access
      self._set_c_attrs(kwargs_attr)

      # Set cached fields: _op_def and _func_name (if not already set)
      self._op_def = self.definition.signature
      if self._func_name:
        assert self._func_name == self._op_def.name
      else:
        self._func_name = compat.as_str(self._op_def.name)

    self._stateful_ops = [(op.name, op.type)
                          for op in temp_graph.get_operations()
                          if op._is_stateful]  # pylint: disable=protected-access

  def _set_c_attrs(self, attrs):
    """Sets `attrs` as attributes of self._c_func.

    Requires that self._c_func is not None.

    Args:
      attrs: a dictionary from attribute name to attribute proto value
    """
    for name, attr_value in attrs.items():
      serialized = attr_value.SerializeToString()
      # TODO(skyewm): this creates and deletes a new TF_Status for every attr.
      # It might be worth creating a convenient way to re-use the same status.
      with self._c_func.get() as func:
        c_api.TF_FunctionSetAttrValueProto(func, compat.as_str(name),
                                           serialized)

  def _create_hash_str(self, input_arg, output_arg, node_def):
    """Creates an 8-character string unique to this input.

    Args:
      input_arg: the input_arg field of an OpDef
                 (e.g. self._definition.signature.input_arg)
      output_arg: the output_arg field of an OpDef
                 (e.g. self._definition.signature.output_arg)
      node_def: the node_def field of a FunctionDef
                (e.g. self._definition.node_def)

    Returns:
      The unique string for this input
    """
    hasher = hashlib.sha1()

    def update_num(n):
      hasher.update(compat.as_bytes("%x" % n))

    def update_str(s):
      update_num(len(s))
      hasher.update(compat.as_bytes(s))

    def update_strs(slist):
      update_num(len(slist))
      for s in slist:
        update_str(s)

    for adef in input_arg:
      update_str(adef.SerializeToString())

    for adef in output_arg:
      update_str(adef.SerializeToString())

    for n in sorted(node_def, key=lambda n: n.name):
      update_str(n.name)
      update_str(n.op)
      update_strs(n.input)
      update_num(len(n.attr))
      # NOTE: protobuf map serialization does not guarantee ordering.
      for k in sorted(n.attr):
        update_str(k)
        update_str(n.attr[k].SerializeToString())

    return hasher.hexdigest()[:8]

  def add_to_graph(self, g):
    """Adds this function into the graph g."""
    self._create_definition_if_needed()

    # Adds this function into 'g'.
    # pylint: disable=protected-access
    if context.executing_eagerly():
      context.context().add_function_def(self.definition)
    else:
      g._add_function(self)
    # pylint: enable=protected-access

    # Ensures related sub-routines are defined in 'g', too.
    for f in self._sub_functions.values():
      g._add_function_recursive(f)  # pylint: disable=protected-access

    # Adds its gradient function, too.
    if self._grad_func:
      self._grad_func.add_to_graph(g)

  def __call__(self, *args, **kwargs):
    self.add_to_graph(ops.get_default_graph())
    args = [ops.convert_to_tensor(_) for _ in args] + self._extra_inputs
    ret, op = _call(self._signature, *args, **kwargs)

    # Set a hidden attr in 'op' so that gradients_impl can refer back
    # to this _DefinedFunction instance to access python_grad_func.
    assert isinstance(op, ops.Operation)
    setattr(op, "__defun", self)

    if self._shape_func is not None:
      shapes = self._shape_func(op)
      if len(shapes) != len(op.outputs):
        raise ValueError(f"shape_func {self._shape_func} produced "
                         f"{len(shapes):d} shapes, which does not match "
                         f"{len(op.outputs)} outputs.")
      for (t, shape) in zip(op.outputs, shapes):
        t.set_shape(shape)
    return ret


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
               out_names=None,
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
      out_names: A list of strings for the function return value names.
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
    self._out_names = out_names
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
      defined = _DefinedFunction(
          self._func,
          self._argnames,
          input_types,
          name,
          None,
          self._python_grad_func,
          out_names=self._out_names,
          **self._extra_kwargs)
      _ = defined.name  # Fully instantiate the function definition.
      if self._grad_func:
        # If _grad_func is given, it is another
        # _OverloadedFunction. We need to instantiate it with the
        # right input types.
        output_types = [
            dtypes.DType(_.type) for _ in defined._signature.output_arg  # pylint: disable=protected-access
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
      if not isinstance(x, tensor_lib.Tensor):
        raise ValueError(f"Expected a Tensor but got {x} with type {type(x)}.")
      input_types.append(x.dtype)
      args[i] = x
    return self.instantiate(input_types)(*args, **kwargs)


class _FuncGraph(ops.Graph):
  """A helper for constructing a function.

  _FuncGraph overrides ops.Graph's create_op() so that we can keep
  track of all inputs into every op created inside the function.  If
  any input is from other graphs, we keep track of it in self.capture
  and substitute the input with a place holder.

  Each captured input's corresponding place holder is converted into a
  function argument and the caller passes in the captured tensor.
  """

  def __init__(self, name, capture_by_value, allowlisted_stateful_ops,
               capture_resource_var_by_value, *args, **kwargs):
    super(_FuncGraph, self).__init__(*args, **kwargs)
    self._capture_by_value = capture_by_value
    self._allowlisted_stateful_ops = allowlisted_stateful_ops
    self._capture_resource_var_by_value = capture_resource_var_by_value
    self._building_function = True
    self._outer_graph = ops.get_default_graph()
    self._vscope = vs.get_variable_scope()
    self._old_custom_getter = self._vscope.custom_getter

    # The name of the function.
    self.name = name
    # Placeholder tensors representing the inputs to this function. The tensors
    # are in this _FuncGraph.
    self.inputs = []
    # Tensors that will be returned this function. The tensors are in this
    # _FuncGraph.
    self.outputs = []
    # Maps external tensor -> internal tensor (e.g. input placeholder).
    self._captured = {}
    # The external tensors that have been captured as inputs and must be passed
    # to this function (empty if capturing by value, otherwise these are the
    # keys of _captured).
    self.extra_inputs = []
    # Input placeholders that been added for captured values (empty if capturing
    # by value).
    self.extra_args = []
    # Captured variables.
    # TODO(skyewm): is this needed?
    self.extra_vars = []

  # pylint: disable=g-doc-return-or-yield

  @property
  def outer_graph(self):
    """The graph active when this _FuncGraph was created."""
    return self._outer_graph

  @tf_contextlib.contextmanager
  def container(self, container_name):
    """Returns a context manager that specifies the resource container to use.

    Overridden from `tf.Graph` to update both the init_scope container
    and the present inner container. This is necessary to make sure setting
    containers applies correctly both to created variables and to stateful
    ops.

    Args:
      container_name: container name string.

    Returns:
      A context manager for defining resource containers for stateful ops,
        yields the container name.
    """
    original_container = self._container
    # pylint: disable=protected-access
    with ops.init_scope():
      original_init_container = ops.get_default_graph()._container
    try:
      self._container = container_name
      with ops.init_scope():
        ops.get_default_graph()._container = container_name
      yield self._container
    finally:
      self._container = original_container
      with ops.init_scope():
        ops.get_default_graph()._container = original_init_container
    # pylint: enable=protected-access

  # pylint: enable=g-doc-return-or-yield

  def getvar(
      self,
      getter,
      name,
      shape=None,
      dtype=None,
      initializer=None,
      reuse=None,
      trainable=True,
      collections=None,  # pylint: disable=redefined-outer-name
      use_resource=None,
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
          reuse=reuse,
          trainable=trainable,
          collections=collections,
          use_resource=use_resource)
      self.extra_vars.append(var)
      if (isinstance(var, resource_variable_ops.BaseResourceVariable) and
          self._capture_resource_var_by_value):
        # For resource-based variables read the variable outside the function
        # and pass in the value. This ensures that the function is pure and
        # differentiable. TODO(apassos) this may have performance problems if
        # the function will only do embedding lookups on the variable.
        return var.value()
      return var

  def _create_op_internal(
      self,
      op_type,
      inputs,
      dtypes=None,  # pylint: disable=redefined-outer-name
      input_types=None,
      name=None,
      attrs=None,
      op_def=None,
      compute_device=True):
    for i, x in enumerate(inputs):
      if isinstance(x, ops.EagerTensor) or x.graph is not self:
        inputs[i] = self.capture(x)
    return super(_FuncGraph, self)._create_op_internal(
        op_type,
        inputs,
        dtypes=dtypes,
        input_types=input_types,
        name=name,
        attrs=attrs,
        op_def=op_def,
        compute_device=compute_device)

  def capture(self, tensor, name=None):
    """Adds the given tensor to this graph and returns the captured tensor."""
    if tensor.ref() in self._captured:
      # Captured already.
      return self._captured[tensor.ref()]
    elif self._capture_by_value:
      return self._add_tensor_and_parents(tensor)
    else:
      return self._capture_tensor_as_extra_input(tensor, name)

  @property
  def captures(self):
    """Pairs of tensors and captured tensor."""
    return [(k.deref(), v) for k, v in self._captured.items()]

  def _capture_tensor_as_extra_input(self, tensor, name=None):
    # Substitute with a placeholder.
    self.extra_inputs.append(tensor)
    # Hoist the new input placeholder out of any control flow context
    # we're currently in.
    with ops.control_dependencies(None):
      ph = array_ops.placeholder(
          tensor.dtype, shape=tensor.get_shape(), name=name)
    # pylint: disable=protected-access
    if isinstance(tensor, ops.EagerTensor):
      handle_data = tensor._handle_data
      if handle_data:
        handle_data = handle_data.SerializeToString()
    else:
      with tensor.graph._c_graph.get() as c_graph:
        handle_data = c_api.GetHandleShapeAndType(c_graph,
                                                  tensor._as_tf_output())

    if handle_data:
      with ph.graph._c_graph.get() as c_graph:
        c_api.SetHandleShapeAndType(c_graph, ph._as_tf_output(),
                                    compat.as_bytes(handle_data))
    # pylint: enable=protected-access
    self.inputs.append(ph)
    self._captured[tensor.ref()] = ph
    self.extra_args.append(ph)
    if _is_guaranteed_const(tensor):
      with ops.control_dependencies(None):
        return array_ops.guarantee_const(ph)
    else:
      return ph

  def _add_tensor_and_parents(self, tensor):
    op = self._add_op_and_parents(tensor.op)
    return op.outputs[tensor.value_index]

  def _add_op_and_parents(self, op: ops.Operation):
    # pylint: disable=protected-access
    op_def = graph_to_function_def._get_op_def(op)
    if op._is_stateful and op not in self._allowlisted_stateful_ops:
      raise ValueError(f"Cannot capture a stateful node (name:{op.name}, "
                       f"type:{op.type}) by value.")
    elif op.type in ("Placeholder", "PlaceholderV2"):
      raise ValueError(f"Cannot capture a placeholder (name:{op.name}, "
                       f"type:{op.type}) by value.")
    # pylint: enable=protected-access

    captured_inputs = [self._add_tensor_and_parents(x) for x in op.inputs]

    captured_op = self._create_op_internal(
        op.type,
        captured_inputs, [o.dtype for o in op.outputs],
        name=op.name,
        attrs=op.node_def.attr,
        op_def=op_def)

    for t, captured_t in zip(op.outputs, captured_op.outputs):
      self._captured[t.ref()] = captured_t

    return captured_op


def func_graph_from_py_func(func,
                            arg_names,
                            arg_types,
                            name=None,
                            capture_by_value=False,
                            device=None,
                            colocation_stack=None,
                            container=None,
                            collections_ref=None,
                            arg_shapes=None,
                            allowlisted_stateful_ops=None,
                            capture_resource_var_by_value=True):
  """Returns a _FuncGraph generated from `func`.

  Args:
    func: A Python callable which constructs a TF function body. The arguments
      must correspond to `arg_types`. Returns a value or list/tuple of values.
      No returned value can be None.
    arg_names: A sequence of strings for the function argument names.
    arg_types: A sequence of the function's argument types.
    name: The function name. If None, the name is derived from `func`.
    capture_by_value: boolean. If True, captured values will be copied into the
      function body.
    device: device name or function.
    colocation_stack: A colocation stack (list) the _FuncGraph should use.
    container: A container name the _FuncGraph should start with.
    collections_ref: A reference to a collections dict the _FuncGraph should
      use internally.
    arg_shapes: A sequence of the function's argument shapes.
    allowlisted_stateful_ops: A set of ops that if stateful we ignore and
      re-create.
    capture_resource_var_by_value: Boolean (defaults to True). If False,
      captured resource variable returns the handle instead of value.

  Returns:
    A _FuncGraph.

  Raises:
    ValueError: if func returns None.
  """
  if not name:
    name = function_utils.get_func_name(func)
  func_graph = _FuncGraph(name, capture_by_value, allowlisted_stateful_ops,
                          capture_resource_var_by_value)

  with func_graph.as_default(), ops.device(device):
    # pylint: disable=protected-access
    if collections_ref is not None:
      func_graph._collections = collections_ref
    if container is not None:
      func_graph._container = container
    if colocation_stack is not None:
      func_graph._colocation_stack = colocation_stack
    # pylint: enable=protected-access

    if arg_shapes is None:
      arg_shapes = [None] * len(arg_types)

    # Create placeholders for the function arguments.
    for (argname, argtype, argshape) in zip(arg_names, arg_types, arg_shapes):
      argholder = array_ops.placeholder(argtype, shape=argshape, name=argname)
      func_graph.inputs.append(argholder)
    # Call func and gather the output tensors.
    with vs.variable_scope("", custom_getter=func_graph.getvar):
      outputs = func(*func_graph.inputs)

    # There is no way of distinguishing between a function not returning
    # anything and a function returning None in Python.
    # We need to allow the former and ideally want to forbid the latter as
    # it is most likely user error.
    # TODO(iga): Consider adding a @NoOutput decorator on top of @Defun to
    # allow users to explicitly mark the function as not returning anything.
    # For now, we allow a single None return and interpret it as a function
    # with no output.
    if outputs is None:
      outputs = []
    else:
      # If func only returned one value, make it a tuple.
      if not isinstance(outputs, (list, tuple)):
        outputs = (outputs,)
      if any(_ is None for _ in outputs):
        raise ValueError(f"Function {name} can not return None.")
    # Ensures each output is a Tensor in the function graph.
    outputs = [ops.convert_to_tensor(t) for t in outputs]
    outputs = [func_graph.capture(t) if t.graph is not func_graph else t
               for t in outputs]
    func_graph.outputs = outputs
  return func_graph


def _is_guaranteed_const(tensor):
  """Determines whether `tensor` is guaranteed to be a constant.

  A tensor is guaranteed to be a constant if either it was produced by
  a `GuaranteeConst` op or if all of its children are guaranteed to be
  constants.

  Args:
    tensor: The tensor for which to determine const-ness.

  Returns:
    True if `tensor` is guaranteed to be a constant, False otherwise.
  """

  if isinstance(tensor, ops.EagerTensor):
    return False

  class Work(object):

    def __init__(self, op: ops.Operation, leaving):
      self.op = op
      self.leaving = leaving

  is_guaranteed_const = lambda op: op.node_def.op == "GuaranteeConst"
  constants = set([])
  def all_inputs_const(op: ops.Operation):
    # If all inputs of an op are guaranteed constants, then we can infer that
    # the op produces a constant as well.
    return op.inputs and all(inp.op in constants for inp in op.inputs)

  visited = set([])
  stack = [Work(tensor.op, leaving=False)]
  while stack:
    work = stack.pop()
    if work.leaving:
      if all_inputs_const(work.op):
        constants.add(work.op)
      continue
    visited.add(work.op)
    if is_guaranteed_const(work.op):
      constants.add(work.op)
      continue

    # This op will be revisited after all its inputs are checked for const-ness.
    stack.append(Work(work.op, leaving=True))
    for inp in work.op.inputs:
      if inp.op not in visited:
        stack.append(Work(inp.op, leaving=False))
  return tensor.op in constants


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
     A 2-element tuple. First element: a Tensor if the function returns a single
     value; a list of Tensors if the function returns multiple value; the
     Operation if the function returns no values. Second element: the Operation.

  Raises:
    ValueError: if the arguments are invalid.
  """
  if len(inputs) != len(sig.input_arg):
    raise ValueError(f"Expected {len(sig.input_arg):d} arguments, got "
                     f"{len(inputs):d}.")
  name = kwargs.pop("name", None)
  g = ops.get_default_graph()
  func_name = sig.name
  if name is None:
    name = func_name
  attrs = _parse_kwargs_as_attrs(func_name, **kwargs)
  output_types = [dtypes.DType(x.type) for x in sig.output_arg]
  op = g._create_op_internal(  # pylint: disable=protected-access
      func_name, list(inputs), output_types, name=name, attrs=attrs, op_def=sig)
  if op.outputs:
    if len(op.outputs) == 1:
      ret = op.outputs[0]
    else:
      ret = tuple(op.outputs)
  else:
    ret = op
  return ret, op


def _from_definition(fdef, grad_func=None):
  """Creates a _DefinedFunction initialized from a FunctionDef proto.

  Args:
    fdef: a FunctionDef
    grad_func: a _DefinedFunction or None

  Returns:
    A _DefinedFunction representing fdef
  """
  # TODO(iga): This method does major surgery on _DefinedFunction.
  # Make it a named constructor using @classmethod of _DefinedFunction.

  # The Python callable is only needed to create a FunctionDef. Since we have
  # the FunctionDef here, we don't need to set _DefinedFunction._func (nor do we
  # have access to such a callable here).
  func = None
  argnames = [arg.name for arg in fdef.signature.input_arg]
  input_types = tuple(
      dtypes.as_dtype(arg.type) for arg in fdef.signature.input_arg)
  func_name = fdef.signature.name
  # Note: FunctionDefs do not include python gradient functions, so if the
  # original _DefinedFunction included one it will not be reflected here.
  python_grad_func = None
  out_names = [arg.name for arg in fdef.signature.output_arg]
  result = _DefinedFunction(func, argnames, input_types, func_name, grad_func,
                            python_grad_func, out_names)
  # pylint: disable=protected-access
  serialized = fdef.SerializeToString()
  c_func = c_api.TF_FunctionImportFunctionDef(serialized)
  result._c_func = c_api_util.ScopedTFFunction(c_func, func_name)
  result._extra_inputs = []
  result._op_def = fdef.signature
  # pylint: enable=protected-access

  return result


def from_library(lib):
  """Creates _DefinedFunctions initialized from a FunctionDefLibrary proto.

  This method handles assigning the correct gradient functions to each
  function.

  Args:
    lib: a FunctionDefLibrary

  Returns:
    A list of _DefinedFunctions

  Raises:
    ValueError: `lib` is invalid
  """
  if not lib.function and not lib.gradient:
    return []

  # function name -> FunctionDef proto
  funcs = {fdef.signature.name: fdef for fdef in lib.function}

  # Validate that all references function names have function defs
  for g in lib.gradient:
    if g.function_name not in funcs:
      raise ValueError(f"FunctionDefLibrary missing '{g.function_name}' "
                       f"FunctionDef\n{lib}")
    if g.gradient_func not in funcs:
      raise ValueError(f"FunctionDefLibrary missing '{g.gradient_func}' "
                       f"FunctionDef\n{lib}")

  # function name -> gradient function name
  func_to_grad = collections.defaultdict(lambda: None)
  # gradient function name -> names of functions having that grad function
  grad_to_funcs = collections.defaultdict(list)

  for gdef in lib.gradient:
    func_to_grad[gdef.function_name] = gdef.gradient_func
    grad_to_funcs[gdef.gradient_func].append(gdef.function_name)

  # Start with functions without gradients
  ready = [
      fdef for fdef in lib.function if func_to_grad[fdef.signature.name] is None
  ]
  if not ready:
    raise ValueError(
        f"FunctionDefLibrary contains cyclic gradient functions!\n{lib}")
  # function name -> _DefinedFunction
  initialized = {}

  while ready:
    fdef = ready.pop()
    name = fdef.signature.name

    grad = initialized.get(func_to_grad[name])
    if func_to_grad[name]:
      assert grad
    defined_func = _from_definition(fdef, grad_func=grad)
    initialized[name] = defined_func

    ready.extend(funcs[f] for f in grad_to_funcs[name])

  return initialized.values()


def _get_experimental_kwarg_as_attr(attr_name, value):
  """Creates an AttrValue for a python object."""
  if isinstance(value, bool):
    return attr_value_pb2.AttrValue(b=value)
  elif isinstance(value, int):
    return attr_value_pb2.AttrValue(i=value)
  elif isinstance(value, float):
    return attr_value_pb2.AttrValue(f=value)
  elif isinstance(value, str):
    return attr_value_pb2.AttrValue(s=compat.as_bytes(value))
  else:
    raise ValueError(f"Attribute {attr_name} must be bool, int, float, or "
                     f"str. Got {type(value)}.")


def _get_kwarg_as_str_attr(attr_name, value):
  """Creates an AttrValue for a python object."""
  if isinstance(value, str):
    return attr_value_pb2.AttrValue(s=compat.as_bytes(value))
  else:
    raise ValueError(f"Attribute {attr_name} must be str. Got {type(value)}.")


def _parse_kwargs_as_attrs(func_name, **kwargs):
  """Parses **kwargs into a node's attributes."""
  attrs = {}

  noinline = kwargs.pop("noinline", None)
  if noinline is not None:
    attrs["_noinline"] = attr_value_pb2.AttrValue(b=bool(noinline))

  # For compatibility with previous behavior, Defun does not perform shape
  # inference through its function call operations.
  attrs["_disable_call_shape_inference"] = attr_value_pb2.AttrValue(b=True)

  compiled = kwargs.pop("compiled", None)
  separate_compiled_gradients = kwargs.pop("separate_compiled_gradients", None)
  if compiled is not None:
    attrs["_XlaCompile"] = attr_value_pb2.AttrValue(b=bool(compiled))
    attrs["_XlaSeparateCompiledGradients"] = attr_value_pb2.AttrValue(
        b=bool(separate_compiled_gradients))
    # Forward _XlaScope from enclosing context (if set), otherwise create new.
    # pylint: disable=protected-access
    if "_XlaScope" in ops.get_default_graph()._attr_scope_map:
      attrs["_XlaScope"] = ops.get_default_graph()._attr_scope_map["_XlaScope"]
    else:
      attrs["_XlaScope"] = attr_value_pb2.AttrValue(
          s=("function_%s" % func_name).encode())
    # pylint: enable=protected-access

  kwargs_keys = list(kwargs.keys())
  for key in kwargs_keys:
    if key.startswith("experimental_"):
      attrs[key] = _get_experimental_kwarg_as_attr(key, kwargs[key])
      del kwargs[key]
    # Support for https://github.com/tensorflow/community/pull/113/files.
    elif key == "_implements" or key == "_reference":
      attrs[key] = _get_kwarg_as_str_attr(key, kwargs[key])
      del kwargs[key]
  if kwargs:
    raise ValueError(f"Unknown keyword arguments: {kwargs.keys()}.")
  return attrs


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


def _type_list_to_str(types):
  if any(_ not in _DTYPE_TO_STR for _ in types):
    unsupported_types = [type_ for type_ in types if type_ not in _DTYPE_TO_STR]
    raise ValueError(f"Unsupported dtypes {unsupported_types} in "
                     "`types`. Supported dtypes are "
                     f"{_DTYPE_TO_STR.keys()}.")
  return "".join(_DTYPE_TO_STR[_] for _ in types)


# NOTE: The list needs to be extended when more data types are added.
_DTYPE_TO_STR = {
    dtypes.float16: "f16",
    dtypes.float32: "f32",
    dtypes.float64: "f64",
    dtypes.int32: "i32",
    dtypes.uint8: "i8",
    dtypes.uint16: "u16",
    dtypes.uint32: "u32",
    dtypes.uint64: "u64",
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
    dtypes.bfloat16: "b16",
    dtypes.float8_e5m2: "f8e5m2",
    dtypes.float8_e4m3fn: "f8e4m3fn",
    dtypes.int4: "i4",
    dtypes.uint4: "u4",
}
