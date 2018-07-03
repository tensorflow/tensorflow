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
"""Decorator that produces a callable object that executes a TensorFlow graph.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib

from tensorflow.python.eager import context
from tensorflow.python.eager import function
from tensorflow.python.eager import tape
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops as tf_ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect


def _default_initializer(name, shape, dtype):
  """The default initializer for variables."""
  # pylint: disable=protected-access
  store = variable_scope._get_default_variable_store()
  initializer = store._get_default_initializer(name, shape=shape, dtype=dtype)
  # pylint: enable=protected-access
  return initializer[0]


class _CapturedVariable(object):
  """Variable captured by graph_callable.

  Internal to the implementation of graph_callable. Created only by
  _VariableCapturingScope and used only to read the variable values when calling
  the function after the variables are initialized.
  """

  def __init__(self, name, initializer, shape, dtype, trainable):
    self.name = name
    if initializer is None:
      initializer = _default_initializer(name, shape, dtype)
    initial_value = lambda: initializer(shape, dtype=dtype)

    with context.eager_mode():
      self.variable = resource_variable_ops.ResourceVariable(
          initial_value=initial_value, name=name, dtype=dtype,
          trainable=trainable)
    self.shape = shape
    self.dtype = dtype
    self.placeholder = None
    self.trainable = trainable

  def read(self, want_gradients=True):
    if want_gradients and self.trainable:
      v = tape.watch_variable(self.variable)
    else:
      v = self.variable
    return v.read_value()


class _VariableCapturingScope(object):
  """Variable-scope-like object which captures tf.get_variable calls.

  This is responsible for the main difference between the initialization version
  of a function object and the calling version of a function object.

  capturing_scope replaces calls to tf.get_variable with placeholder tensors to
  be fed the variable's current value. TODO(apassos): these placeholders should
  instead be objects implementing a similar API to tf.Variable, for full
  compatibility.

  initializing_scope replaces calls to tf.get_variable with creation of
  variables and initialization of their values. This allows eventual support of
  initialized_value and friends.

  TODO(apassos): once the eager mode layers API is implemented support eager
  func-to-object as well.
  """

  def __init__(self):
    self.variables = {}
    self.tf_variables = {}

  @contextlib.contextmanager
  def capturing_scope(self):
    """Context manager to capture variable creations.

    Replaces variable accesses with placeholders.

    Yields:
      nothing
    """
    # TODO(apassos) ignoring the regularizer and partitioner here; figure out
    # how to deal with these.
    def _custom_getter(  # pylint: disable=missing-docstring
        getter=None,
        name=None,
        shape=None,
        dtype=dtypes.float32,
        initializer=None,
        regularizer=None,
        reuse=None,
        trainable=True,
        collections=None,
        caching_device=None,  # pylint: disable=redefined-outer-name
        partitioner=None,
        validate_shape=True,
        use_resource=None,
        aggregation=variable_scope.VariableAggregation.NONE,
        synchronization=variable_scope.VariableSynchronization.AUTO):
      del getter, regularizer, partitioner, validate_shape, use_resource, dtype
      del collections, initializer, trainable, reuse, caching_device, shape
      del aggregation, synchronization
      assert name in self.variables
      v = self.variables[name]
      return v.variable

    scope = variable_scope.get_variable_scope()
    with variable_scope.variable_scope(scope, custom_getter=_custom_getter):
      yield

  @contextlib.contextmanager
  def initializing_scope(self):
    """Context manager to capture variable creations.

    Forcibly initializes all created variables.

    Yields:
      nothing
    """
    # TODO(apassos) ignoring the regularizer and partitioner here; figure out
    # how to deal with these.
    def _custom_getter(  # pylint: disable=missing-docstring
        getter=None,
        name=None,
        shape=None,
        dtype=dtypes.float32,
        initializer=None,
        regularizer=None,
        reuse=None,
        trainable=True,
        collections=None,
        caching_device=None,  # pylint: disable=redefined-outer-name
        partitioner=None,
        validate_shape=True,
        use_resource=None,
        aggregation=variable_scope.VariableAggregation.NONE,
        synchronization=variable_scope.VariableSynchronization.AUTO):
      del getter, regularizer, collections, caching_device, partitioner
      del use_resource, validate_shape, aggregation, synchronization
      if name in self.tf_variables:
        if reuse:
          return self.tf_variables[name].initialized_value()
        else:
          raise ValueError("Specified reuse=%s but tried to reuse variables."
                           % reuse)
      # TODO(apassos): ensure this is on the same device as above
      v = _CapturedVariable(name, initializer, shape, dtype, trainable)
      self.variables[name] = v

      graph_mode_resource = v.variable.handle
      if initializer is None:
        initializer = _default_initializer(name, shape, dtype)
      resource_variable_ops.shape_safe_assign_variable_handle(
          graph_mode_resource, v.variable.shape, initializer(shape, dtype))
      return v.variable

    scope = variable_scope.get_variable_scope()
    with variable_scope.variable_scope(scope, custom_getter=_custom_getter):
      yield


class _InitializingFunctionObject(object):
  """Responsible for deciding which version of func-to-object to call.

  call_fn is the version which calls the function with the current values of the
  variables and init_fn is the version which calls the function to initialize
  all variables.

  TODO(apassos): figure out a way to support initializing only _some_
  variables. This requires a way to pull out a variable's initialization code
  from the graph, which might not be possible in general.
  """

  def __init__(self, call_fn, init_fn, shape_and_dtypes):
    self._init_fn = init_fn
    self._call_fn = call_fn
    self.shape_and_dtypes = shape_and_dtypes
    self.flattened_shapes = [tensor_shape.as_shape(sd.shape) for sd in
                             nest.flatten(self.shape_and_dtypes)]

  @property
  def variables(self):
    return self._call_fn.variables

  def __call__(self, *args):
    nest.assert_same_structure(self.shape_and_dtypes, args, check_types=False)
    if not all([
        shape.is_compatible_with(arg.shape)
        for shape, arg in zip(self.flattened_shapes, nest.flatten(args))
    ]):
      raise ValueError(
          "Declared shapes do not match argument shapes: Expected %s, found %s."
          % (self.flattened_shapes, [arg.shape for arg in nest.flatten(args)]))

    initialized = [resource_variable_ops.var_is_initialized_op(
        v.handle).numpy() for v in self._call_fn.variables]
    if all(x for x in initialized):
      for v in self._call_fn.variables:
        if v.trainable:
          tape.watch_variable(v)
      return self._call_fn(*args)
    elif all(not x for x in initialized):
      return self._init_fn(*args)
    else:
      raise ValueError("Some, but not all, variables are initialized.")


def _get_graph_callable_inputs(shape_and_dtypes):
  """Maps specified shape_and_dtypes to graph inputs."""
  ret = []
  for x in shape_and_dtypes:
    if isinstance(x, ShapeAndDtype):
      ret.append(array_ops.placeholder(x.dtype, x.shape))
    elif isinstance(x, (tuple, list)):
      ret.append(_get_graph_callable_inputs(x))
    else:
      raise errors.InvalidArgumentError(
          None, None, "Expected the argument to @graph_callable to be a "
          "(possibly nested) list or tuple of ShapeAndDtype objects, "
          "but got an object of type: %s" % type(x))

  return tuple(ret) if isinstance(shape_and_dtypes, tuple) else ret


def _graph_callable_internal(func, shape_and_dtypes):
  """Defines and returns a template version of func.

  Under the hood we make two function objects, each wrapping a different version
  of the graph-mode code. One version immediately runs variable initialization
  before making the variable's Tensors available for use, while the other
  version replaces the Variables with placeholders which become function
  arguments and get the current variable's value.

  Limitations in (2) and (4) are because this does not implement a graph-mode
  Variable class which has a convert_to_tensor(as_ref=True) method and a
  initialized_value method. This is fixable.

  Args:
    func: The tfe Python function to compile.
    shape_and_dtypes: A possibly nested list or tuple of ShapeAndDtype objects.

  Raises:
    ValueError: If any one of func's outputs is not a Tensor.

  Returns:
    Callable graph object.
  """
  container = tf_ops.get_default_graph()._container  # pylint: disable=protected-access
  graph_key = tf_ops.get_default_graph()._graph_key  # pylint: disable=protected-access
  with context.graph_mode():
    # This graph will store both the initialization and the call version of the
    # wrapped function. It will later be used by the backprop code to build the
    # backprop graph, if necessary.
    captures = {}
    tmp_graph = function.CapturingGraph(captures)
    # Inherit the graph key from the original graph to ensure optimizers don't
    # misbehave.
    tmp_graph._container = container  # pylint: disable=protected-access
    tmp_graph._graph_key = graph_key  # pylint: disable=protected-access
    with tmp_graph.as_default():
      # Placeholders for the non-variable inputs.
      func_inputs = _get_graph_callable_inputs(shape_and_dtypes)
      func_num_args = len(tf_inspect.getargspec(func).args)
      if len(func_inputs) != func_num_args:
        raise TypeError("The number of arguments accepted by the decorated "
                        "function `%s` (%d) must match the number of "
                        "ShapeAndDtype objects passed to the graph_callable() "
                        "decorator (%d)." %
                        (func.__name__, func_num_args, len(func_inputs)))

      # First call the function to generate a graph which can initialize all
      # variables. As a side-effect this will populate the variable capturing
      # scope's view of which variables exist.
      variable_captures = _VariableCapturingScope()
      with variable_captures.initializing_scope(
          ), function.AutomaticControlDependencies() as a:
        func_outputs = func(*func_inputs)
        outputs_list = nest.flatten(func_outputs)
        for i, x in enumerate(outputs_list):
          if x is not None:
            outputs_list[i] = a.mark_as_return(x)
      if len(outputs_list) == 1 and outputs_list[0] is None:
        outputs_list = []
      output_shapes = [x.shape for x in outputs_list]
      if not all(isinstance(x, tf_ops.Tensor) for x in outputs_list):
        raise ValueError("Found non-tensor output in %s" % str(outputs_list))
      initializing_operations = tmp_graph.get_operations()

      # Call the function again, now replacing usages of variables with
      # placeholders. This assumes the variable capturing scope created above
      # knows about all variables.
      tmp_graph.clear_resource_control_flow_state()
      with variable_captures.capturing_scope(
          ), function.AutomaticControlDependencies() as a:
        captured_outputs = func(*func_inputs)
      captured_outlist = nest.flatten(captured_outputs)
      for i, x in enumerate(captured_outlist):
        if x is not None:
          captured_outlist[i] = a.mark_as_return(x)
      capturing_operations = tmp_graph.get_operations()[
          len(initializing_operations):]

  sorted_variables = sorted(variable_captures.variables.values(),
                            key=lambda x: x.name)
  ids = list(sorted(captures.keys()))
  if ids:
    extra_inputs, extra_placeholders = zip(*[captures[x] for x in ids])
  else:
    extra_inputs = []
    extra_placeholders = []

  flat_inputs = [x for x in nest.flatten(func_inputs)
                 if isinstance(x, tf_ops.Tensor)]
  placeholder_inputs = flat_inputs+ list(extra_placeholders)

  func_def_outputs = [x for x in outputs_list if isinstance(x, tf_ops.Tensor)]
  initialization_name = function._inference_name(func.__name__)  # pylint: disable=protected-access
  # TODO(ashankar): Oh lord, forgive me for this lint travesty.
  # Also, what about the gradient registry of these functions? Those need to be
  # addressed as well.
  for f in tmp_graph._functions.values():  # pylint: disable=protected-access
    function._register(f._c_func.func)  # pylint: disable=protected-access
  initializer_function = function.GraphModeFunction(
      initialization_name,
      placeholder_inputs,
      extra_inputs,
      tmp_graph,
      initializing_operations,
      func_def_outputs,
      func_outputs,
      output_shapes)

  capture_func_def_outputs = [
      x for x in captured_outlist if isinstance(x, tf_ops.Tensor)]
  captured_function_name = function._inference_name(func.__name__)  # pylint: disable=protected-access
  captured_function = function.GraphModeFunction(
      captured_function_name,
      placeholder_inputs,
      extra_inputs,
      tmp_graph,
      capturing_operations,
      capture_func_def_outputs,
      captured_outputs,
      output_shapes,
      variables=[x.variable for x in sorted_variables])

  return _InitializingFunctionObject(captured_function, initializer_function,
                                     shape_and_dtypes)


class ShapeAndDtype(object):
  """Data type that packages together shape and type information.

  Used for arguments to graph callables. See graph_callable() for an example.
  """

  def __init__(self, shape, dtype):
    self.shape = shape
    self.dtype = dtype


def graph_callable(shape_and_dtypes):
  """Decorator that produces a callable that executes a TensorFlow graph.

  When applied on a function that constructs a TensorFlow graph, this decorator
  produces a callable object that:

  1. Executes the graph when invoked. The first call will initialize any
     variables defined in the graph.

  2. Provides a .variables() method to return the list of TensorFlow variables
     defined in the graph.

  Note that the wrapped function is not allowed to change the values of the
  variables, just use them.

  The return value of the wrapped function must be one of the following:
  (1) None,  (2) a Tensor, or (3) a possibly nested sequence of Tensors.

  Example:

  ```python
  @tfe.graph_callable([tfe.ShapeAndDtype(shape(), dtype=dtypes.float32)])
  def foo(x):
    v = tf.get_variable('v', initializer=tf.ones_initializer(), shape=())
    return v + x

  ret = foo(tfe.Tensor(2.0))  # `ret` here is a Tensor with value 3.0.

  foo.variables[0].assign(7.0)  # Modify the value of variable `v`.
  ret = foo(tfe.Tensor(2.0))  # `ret` here now is a Tensor with value 9.0.
  ```
  Args:
    shape_and_dtypes: A possibly nested list or tuple of ShapeAndDtype objects
      that specifies shape and type information for each of the callable's
      arguments. The length of this list must be equal to the number of
      arguments accepted by the wrapped function.

  Returns:
    A callable graph object.
  """
  # TODO(alive,apassos): support initialized_value and friends from tf.Variable.
  assert context.executing_eagerly(), (
      "graph_callable can only be used when Eager execution is enabled.")
  def decorator(func):
    return tf_decorator.make_decorator(func,
                                       _graph_callable_internal(
                                           func, shape_and_dtypes))

  return decorator
