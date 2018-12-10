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
"""Code for backpropagation using the tape utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import operator
import sys

import six

from tensorflow.python import pywrap_tensorflow
from tensorflow.python.eager import context
from tensorflow.python.eager import execute
from tensorflow.python.eager import imperative_grad
from tensorflow.python.eager import tape
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops.unconnected_gradients import UnconnectedGradients
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.lazy_loader import LazyLoader
from tensorflow.python.util.tf_export import tf_export


# Note that we need to lazy load the following two modules to avoid creating
# circular dependencies.
# TODO(b/119775953): fix the circular dependencies.
pfor_ops = LazyLoader(
    "pfor_ops", globals(),
    "tensorflow.python.ops.parallel_for.control_flow_ops")

function = LazyLoader("function", globals(),
                      "tensorflow.python.eager.function")

_op_attr_type_cache = {}


def op_attr_type(op_type, attr_name):
  try:
    return _op_attr_type_cache[(op_type, attr_name)]
  except KeyError:
    h = context.context()._handle  # pylint: disable=protected-access
    attr_type = pywrap_tensorflow.TFE_OpNameGetAttrType(h, op_type, attr_name)
  _op_attr_type_cache[(op_type, attr_name)] = attr_type
  return attr_type


def make_attr(attr_type, value):
  if attr_type == pywrap_tensorflow.TF_ATTR_TYPE:
    return dtypes.as_dtype(value)
  elif attr_type == [pywrap_tensorflow.TF_ATTR_TYPE]:
    return [dtypes.as_dtype(v) for v in value]
  elif attr_type == pywrap_tensorflow.TF_ATTR_SHAPE:
    return tensor_shape.as_shape(value).as_proto()
  elif attr_type == [pywrap_tensorflow.TF_ATTR_SHAPE]:
    return [tensor_shape.as_shape(v).as_proto() for v in value]
  return value


class _MockOp(object):
  """Pretends to be a tf.Operation for the gradient functions."""

  def __init__(self, attrs, inputs, outputs, typ):
    self.attrs = attrs
    self.inputs = inputs
    self.outputs = outputs
    self.type = typ

  def get_attr(self, attr):
    typ = op_attr_type(self.type, attr)
    for i in range(0, len(self.attrs), 2):
      if self.attrs[i] == attr:
        return make_attr(typ, self.attrs[i + 1])
    raise KeyError(attr)

  def _get_control_flow_context(self):
    raise NotImplementedError(
        "tf.GradientTape.gradients() does not support graph control flow "
        "operations like tf.cond or tf.while at this time. Use tf.gradients() "
        "instead. If you need this feature, please file a feature request at "
        "https://github.com/tensorflow/tensorflow/issues/new"
    )


def _gradient_function(op_name, attr_tuple, num_inputs, inputs, outputs,
                       out_grads):
  """Calls the gradient function of the op.

  Args:
    op_name: the name of the op to be differentiated.
    attr_tuple: the attrs, as a tuple.
    num_inputs: the number of inputs to the op.
    inputs: inputs to the original operation.
    outputs: outputs to the original operation.
    out_grads: gradients of the operation wrt its outputs.

  Returns:
    The gradients with respect to the inputs of the function, as a list.
  """
  mock_op = _MockOp(attr_tuple, inputs, outputs, op_name)
  grad_fn = ops._gradient_registry.lookup(op_name)  # pylint: disable=protected-access
  if grad_fn is None:
    return [None] * num_inputs

  return grad_fn(mock_op, *out_grads)


pywrap_tensorflow.TFE_Py_RegisterGradientFunction(_gradient_function)


def _record_gradient(op_name, inputs, attrs, results, name):
  return pywrap_tensorflow.TFE_Py_RecordGradient(op_name, inputs, attrs,
                                                 results, name)


execute.record_gradient = _record_gradient


def implicit_val_and_grad(f):
  """Returns a function which differentiates f with respect to variables.

  The wrapped function returns the value and the gradient of f when called with
  the same arguments. The gradient is with respect to all trainable TFE
  variables accessed by `f`.

  This function is useful when the exact set of variables to differentiate with
  is not known ahead of time.

  Example:

  ```python
  dense_layer = tf.layers.Dense(1)
  def loss(x, y):
    return tf.reduce_sum(tf.square(dense_layer(x) - y))

  # Obtain the gradient function.
  val_grad_fn = tfe.implicit_value_and_gradients(loss)

  # Invoke the gradient function with concrete values of x and y.
  x = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  y = tf.constant([[10.0], [20.0]])
  value, grads_and_vars = val_grad_fn(x, y)
  print('Value of loss: %s' % value)

  # Apply the gradients to Variables.
  optimizer = tf.train.GradientDescentOptimizer(0.1)
  optimizer.apply_gradients(grads_and_vars)
  ```

  Args:
    f: function to be differentiated. If `f` returns a scalar, this scalar will
      be differentiated. If `f` returns a tensor or list of tensors, by default
      a scalar will be computed by adding all their values to produce a single
      scalar.

  Returns:
    A function which, when called, returns a tuple pair.
    Its first element is the value to which the function evaluates.
    Its second element is list of (gradient, variable) pairs.

  Raises:
    ValueError: if `f` returns None.
  """
  # TODO(cais): Remove calls to tf.constant() once the gradients functions
  # accept lists and np.ndarrays.

  def grad_fn(*args, **kwds):
    """Computes the gradient of the wrapped function."""
    this_tape = tape.push_new_tape()
    try:
      end_node = f(*args, **kwds)
      if end_node is None:
        raise ValueError("Cannot differentiate a function that returns None; "
                         "did you forget to return a value from {}?".format(
                             f.__name__))
    finally:
      tape.pop_tape(this_tape)
    # Note: variables are returned in construction order. This ensures unique
    # order across executions.
    variables = this_tape.watched_variables()
    if not variables:
      raise ValueError("No trainable variables were accessed while the "
                       "function was being computed.")

    sources = [v.handle for v in variables]
    grad = imperative_grad.imperative_grad(this_tape, nest.flatten(end_node),
                                           sources)
    return end_node, list(zip(grad, variables))

  return grad_fn


def implicit_grad(f):
  """Returns a function which differentiates f with respect to variables.

  The wrapped function returns the gradient of f when called with the same
  arguments. The gradient is with respect to all trainable TFE variables
  accessed by `f`.

  This function is useful when the exact set of variables to differentiate with
  is not known ahead of time.

  Example:

  ```python
  dense_layer = tf.layers.Dense(1)
  def loss(x, y):
    return tf.reduce_sum(tf.square(dense_layer(x) - y))

  # Obtain the gradient function.
  grad_fn = tfe.implicit_gradients(loss)

  # Invoke the gradient function with concrete values of x and y.
  x = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  y = tf.constant([[10.0], [20.0]])
  grads_and_vars = grad_fn(x, y)

  # Apply the gradients to Variables.
  optimizer = tf.train.GradientDescentOptimizer(0.1)
  optimizer.apply_gradients(grads_and_vars)
  ```

  Args:
    f: function to be differentiated. If `f` returns a scalar, this scalar will
      be differentiated. If `f` returns a tensor or list of tensors, by default
      a scalar will be computed by adding all their values to produce a single
      scalar.

  Returns:
    A function which, when called, returns a list of (gradient, variable) pairs.
  """
  # TODO(cais): Remove calls to tf.constant() once the gradients functions
  # accept lists and np.ndarrays.

  def grad_fn(*args, **kwds):
    """Computes the gradient of the wrapped function."""
    return implicit_val_and_grad(f)(*args, **kwds)[1]

  return grad_fn


def _get_arg_spec(f, params, param_args):
  """The positions of the parameters of f to be differentiated in param_args."""
  try:
    args = tf_inspect.getfullargspec(f).args
  except TypeError as e:
    # TypeError can happen when f is a callable object.
    if params is None:
      return range(len(param_args))
    elif all(isinstance(x, int) for x in params):
      return params
    raise ValueError("Either callable provided is not a function or could not "
                     "inspect its arguments by name: %s. Original error: %s"
                     % (f, e))
  if params is None:
    if not args:
      return range(len(param_args))
    return range(len(args))
  elif all(isinstance(x, six.string_types) for x in params):
    return [args.index(n) for n in params]
  elif all(isinstance(x, int) for x in params):
    return params
  else:
    raise ValueError(
        "params must be all strings or all integers; got %s." % params)


def gradients_function(f, params=None):
  """Returns a function which differentiates f with respect to params.

  Example:
  ```python
  # f(x, y) = (x ^ 3) * y - x * (y ^ 2)
  # Therefore, the 1st order derivatives are:
  #   df / dx = 3 * (x ^ 2) * y - y ^ 2
  #   df / dy = x ^ 3 - 2 * x * y
  # The 2nd order derivatives with respect to x is:
  #   d^2 f / (dx)^2 = 6 * x * y
  def f(x, y):
    return x * x * x * y - x * y * y

  # Obtain a function that returns 1st order gradients.
  grad_fn = tfe.gradients_function(f)

  x = 2.0
  y = 3.0

  # Invoke the 1st order gradient function.
  x_grad, y_grad = grad_fn(x, y)
  assert x_grad.numpy() == 3 * (2 ** 2) * 3 - 3 ** 2
  assert y_grad.numpy() == (2 ** 3) - 2 * 2 * 3

  # Obtain a function that returns the 2nd order gradient with respect to x.
  gradgrad_fn = tfe.gradients_function(lambda x, y: grad_fn(x, y)[0])

  # Invoke the 2nd order gradient function.
  x_gradgrad = gradgrad_fn(x, y)[0]
  assert x_gradgrad.numpy() == 6 * 2 * 3

  # To obtain a callable that returns the gradient(s) of `f` with respect to a
  # subset of its inputs, use the `params` keyword argument with
  # `gradients_function()`.
  ygrad_fn = tfe.gradients_function(f, params=[1])

  (y_grad,) = ygrad_fn(x, y)
  assert y_grad.numpy() == (2 ** 3) - 2 * 2 * 3
  ```

  Note that only tensors with real or complex dtypes are differentiable.

  Args:
    f: function to be differentiated. If `f` returns a scalar, this scalar will
      be differentiated. If `f` returns a tensor or list of tensors, by default
      a scalar will be computed by adding all their values to produce a single
      scalar. If desired, the tensors can be elementwise multiplied by the
      tensors passed as the `dy` keyword argument to the returned gradient
      function.
    params: list of parameter names of f or list of integers indexing the
      parameters with respect to which we'll differentiate. Passing None
      differentiates with respect to all parameters.

  Returns:
    function which, when called, returns the value of f and the gradient
    of `f` with respect to all of `params`. The function takes an extra optional
    keyword argument `dy`. Setting it allows computation of vector jacobian
    products for vectors other than the vector of ones.

  Raises:
    ValueError: if the params are not all strings or all integers.
  """

  def decorated(*args, **kwds):
    """Computes the gradient of the decorated function."""

    _, grad = val_and_grad_function(f, params=params)(*args, **kwds)
    return grad

  return decorated


def _ensure_unique_tensor_objects(parameter_positions, args):
  """Make each of the parameter_positions in args a unique ops.Tensor object.

  Ensure that each parameter is treated independently.
  For example:

  def f(x, y): return x * y
  g = gradients_function(f)
  one = tf.constant(1.)

  g(one, one) should return [1., 1.]
  (even though the two arguments are the same Tensor object).

  Args:
    parameter_positions: List of indices into args defining the arguments to
      differentiate against.
    args: A list of arguments to the function to be differentiated.

  Returns:
    args, possibly edited in-place.
  """
  s = set()
  for (i, t) in enumerate(args):
    if i in parameter_positions:
      tid = ops.tensor_id(t)
      if tid in s:
        args[i] = gen_array_ops.identity(args[i])
      else:
        s.add(tid)
  return args


def val_and_grad_function(f, params=None):
  """Returns a function that computes f and its derivative w.r.t. params.

  Example:
  ```python
  # f(x, y) = (x ^ 3) * y - x * (y ^ 2)
  # Therefore, the 1st order derivatives are:
  #   df / dx = 3 * (x ^ 2) * y - y ^ 2
  #   df / dy = x ^ 3 - 2 * x * y
  def f(x, y):
    return x * x * x * y - x * y * y

  # Obtain a function that returns the function value and the 1st order
  # gradients.
  val_grads_fn = tfe.value_and_gradients_function(f)

  x = 2.0
  y = 3.0

  # Invoke the value-and-gradients function.
  f_val, (x_grad, y_grad) = val_grads_fn(x, y)
  assert f_val.numpy() == (2 ** 3) * 3 - 2 * (3 ** 2)
  assert x_grad.numpy() == 3 * (2 ** 2) * 3 - 3 ** 2
  assert y_grad.numpy() == (2 ** 3) - 2 * 2 * 3

  # To obtain a callable that returns the value of `f` and the gradient(s) of
  # `f` with respect to a subset of its inputs, use the `params` keyword
  # argument with `value_and_gradients_function()`.
  val_ygrad_fn = tfe.value_and_gradients_function(f, params=[1])

  f_val, (y_grad,) = val_ygrad_fn(x, y)
  assert f_val.numpy() == (2 ** 3) * 3 - 2 * (3 ** 2)
  assert y_grad.numpy() == (2 ** 3) - 2 * 2 * 3
  ```

  Args:
    f: function to be differentiated. If `f` returns a scalar, this scalar will
      be differentiated. If `f` returns a tensor or list of tensors, by default
      a scalar will be computed by adding all their values to produce a single
      scalar. If desired, the tensors can be elementwise multiplied by the
      tensors passed as the `dy` keyword argument to the returned gradient
      function.
    params: list of parameter names of f or list of integers indexing the
      parameters with respect to which we'll differentiate. Passing `None`
      differentiates with respect to all parameters.

  Returns:
    function which, when called, returns the value of f and the gradient
    of f with respect to all of `params`. The function takes an extra optional
    keyword argument "dy". Setting it allows computation of vector jacobian
    products for vectors other than the vector of ones.

  Raises:
    ValueError: if the params are not all strings or all integers.
  """

  def decorated(*args, **kwds):
    """Computes the value and gradient of the decorated function."""
    dy = kwds.pop("dy", None)
    if kwds:
      raise ValueError("Functions to be differentiated cannot "
                       "receive keyword arguments.")
    val, vjp = make_vjp(f, params)(*args, **kwds)
    return val, vjp(dy=dy)

  return decorated


def make_vjp(f, params=None, persistent=True):
  """Returns a function that computes f and is vjp w.r.t. params.

  The term "vjp" here is an abbreviation for vector-jacobian product.

  Args:
    f: the function to be differentiated.
    params: the parameters (numbers or names) to differentiate with respect to.
       A value of None will differentiate with respect to all parameters.
    persistent: Boolean controlling whether the VJP function can be re-used.
      Must be True or False.

  Returns:
    A function, which when called, returns a tuple (value, vjp), where:
    - value is the result of calling f.
    - vjp is a function, which takes a vector as an argument and
      returns the product of that vector with the Jacobian of f.
      Providing no argument to vjp is equivalent to providing a
      vector of ones.

    For example,
    ```python
    def f(x):
      return x * x

    wrapped_fn = tfe.make_vjp(f)
    result, vjp = wrapped_fn(tf.constant(3.0))
    # result is 9.0
    vjp()  # the vjp function rturns 6.0

  Raises:
    ValueError: if `f` returns None.
  """

  def decorated(*args, **kwds):
    """Computes the value and gradient of the decorated function."""
    parameter_positions = _get_arg_spec(f, params, args)
    assert not kwds, "The gradient function can't take keyword arguments."
    this_tape = tape.push_new_tape(persistent=persistent)
    try:
      sources = []
      args = [
          ops.convert_to_tensor(args[i])
          if i in parameter_positions else args[i]
          for i in range(len(args))
      ]
      args = _ensure_unique_tensor_objects(parameter_positions, args)
      for i in parameter_positions:
        sources.append(args[i])
        tape.watch(this_tape, args[i])
      result = f(*args)
      if result is None:
        raise ValueError("Cannot differentiate a function that returns None; "
                         "did you forget to return a value from {}?".format(
                             f.__name__))
      flat_result = nest.flatten(result)
      flat_result = [gen_array_ops.identity(x) for x in flat_result]
      result = nest.pack_sequence_as(result, flat_result)
    finally:
      tape.pop_tape(this_tape)
    def vjp(dy=None):
      if dy is not None:
        dy = [ops.convert_to_tensor(x) for x in nest.flatten(dy)]
      return imperative_grad.imperative_grad(
          this_tape, nest.flatten(result), sources, output_gradients=dy)

    return result, vjp

  return decorated


def _aggregate_grads(gradients):
  """Aggregate gradients from multiple sources.

  Args:
    gradients: A list of 'Tensor' or 'IndexedSlices' gradients.

  Returns:
    If 'gradients' only has 'Tensor', returns an aggregated 'Tensor'.
    Otherwise returns an aggregated 'IndexedSlices'.
  """
  assert gradients, "No gradients to aggregate"

  if len(gradients) == 1:
    return gradients[0]
  if all(isinstance(g, ops.Tensor) for g in gradients):
    return gen_math_ops.add_n(gradients)
  else:
    assert all(isinstance(g, (ops.Tensor, ops.IndexedSlices))
               for g in gradients)
    indexed_slices_list = []
    for grad in gradients:
      # TODO(xpan): Support nested IndexedSlices and core IndexedSlices
      if isinstance(grad, ops.Tensor):
        indexed_slices = ops.IndexedSlices(
            grad,
            math_ops.range(grad.shape[0]),
            constant_op.constant(grad.shape.as_list()))
        indexed_slices_list.append(indexed_slices)
      else:
        indexed_slices_list.append(grad)

    # Dense shapes from all gradients should be the same.
    dense_shape = indexed_slices_list[0].dense_shape
    # For simplicity now, always cast to int64.
    indices = array_ops.concat([math_ops.cast(x.indices, dtypes.int64)
                                for x in indexed_slices_list], 0)
    values = array_ops.concat([x.values for x in indexed_slices_list], 0)
    return ops.IndexedSlices(values, indices, dense_shape)


def _num_elements(grad):
  """The number of elements in the `grad` tensor."""
  if isinstance(grad, ops.Tensor):
    shape_tuple = grad._shape_tuple()  # pylint: disable=protected-access
    if shape_tuple is None or None in shape_tuple:
      return 0
    return functools.reduce(operator.mul, shape_tuple, 1)
  if isinstance(grad, ops.IndexedSlices):
    return functools.reduce(operator.mul, grad.values._shape_tuple(), 1)  # pylint: disable=protected-access
  raise ValueError("`grad` not a Tensor or IndexedSlices.")


def _fast_fill(value, shape, dtype):
  return array_ops.fill(
      constant_op.constant(shape, dtype=dtypes.int32),
      constant_op.constant(value, dtype=dtype))


def _zeros(shape, dtype):
  """Helper to return (possibly cached) zero tensors in eager mode."""
  if dtype == dtypes.variant:
    # TODO(apassos): need to save enough information about variant tensors to do
    # a zeros
    return None

  ctx = context.context()
  if not ctx.executing_eagerly():
    return array_ops.zeros(shape, dtype)

  device = ctx.device_name
  cache_key = shape, dtype, device
  cached = ctx.zeros_cache().get(cache_key)
  if cached is None:
    if dtypes.as_dtype(dtype).is_bool:
      value = False
    else:
      value = 0
    cached = _fast_fill(value, shape, dtype)
    ctx.zeros_cache().put(cache_key, cached)
  return cached


def _ones(shape, dtype):
  if not context.context().executing_eagerly():
    return array_ops.ones(shape, dtype)

  if dtypes.as_dtype(dtype).is_bool:
    value = True
  else:
    value = 1

  if shape == ():  # pylint: disable=g-explicit-bool-comparison
    return constant_op.constant(value, dtype=dtype)
  return _fast_fill(value, shape, dtype)


_default_vspace = imperative_grad.VSpace(
    num_elements_fn=_num_elements,
    aggregate_fn=_aggregate_grads,
    zeros_fn=_zeros,
    ones_fn=_ones,
    graph_shape_fn=gen_array_ops.shape)
pywrap_tensorflow.TFE_Py_RegisterVSpace(_default_vspace)


def _handle_or_self(x):
  """If x is ResourceVariable, return its handle, else x."""
  if resource_variable_ops.is_resource_variable(x):
    x = x.handle
  return x


@tf_export("GradientTape")
class GradientTape(object):
  """Record operations for automatic differentiation.

  Operations are recorded if they are executed within this context manager and
  at least one of their inputs is being "watched".

  Trainable variables (created by `tf.Variable` or `tf.get_variable`, where
  `trainable=True` is default in both cases) are automatically watched. Tensors
  can be manually watched by invoking the `watch` method on this context
  manager.

  For example, consider the function `y = x * x`. The gradient at `x = 3.0` can
  be computed as:

  ```python
  x = tf.constant(3.0)
  with tf.GradientTape() as g:
    g.watch(x)
    y = x * x
  dy_dx = g.gradient(y, x) # Will compute to 6.0
  ```

  GradientTapes can be nested to compute higher-order derivatives. For example,

  ```python
  x = tf.constant(3.0)
  with tf.GradientTape() as g:
    g.watch(x)
    with tf.GradientTape() as gg:
      gg.watch(x)
      y = x * x
    dy_dx = gg.gradient(y, x)     # Will compute to 6.0
  d2y_dx2 = g.gradient(dy_dx, x)  # Will compute to 2.0
  ```

  By default, the resources held by a GradientTape are released as soon as
  GradientTape.gradient() method is called. To compute multiple gradients over
  the same computation, create a persistent gradient tape. This allows multiple
  calls to the gradient() method as resources are released when the tape object
  is garbage collected. For example:

  ```python
  x = tf.constant(3.0)
  with tf.GradientTape(persistent=True) as g:
    g.watch(x)
    y = x * x
    z = y * y
  dz_dx = g.gradient(z, x)  # 108.0 (4*x^3 at x = 3)
  dy_dx = g.gradient(y, x)  # 6.0
  del g  # Drop the reference to the tape
  ```

  By default GradientTape will automatically watch any trainable variables that
  are accessed inside the context. If you want fine grained control over which
  variables are watched you can disable automatic tracking by passing
  `watch_accessed_variables=False` to the tape constructor:

  ```python
  with tf.GradientTape(watch_accessed_variables=False) as tape:
    tape.watch(variable_a)
    y = variable_a ** 2  # Gradients will be available for `variable_a`.
    z = variable_b ** 3  # No gradients will be avaialble since `variable_b` is
                         # not being watched.
  ```

  Note that when using models you should ensure that your variables exist when
  using `watch_accessed_variables=False`. Otherwise it's quite easy to make your
  first iteration not have any gradients:

  ```python
  a = tf.keras.layers.Dense(32)
  b = tf.keras.layers.Dense(32)

  with tf.GradientTape(watch_accessed_variables=False) as tape:
    tape.watch(a.variables)  # Since `a.build` has not been called at this point
                             # `a.variables` will return an empty list and the
                             # tape will not be watching anything.
    result = b(a(inputs))
    tape.gradient(result, a.variables)  # The result of this computation will be
                                        # a list of `None`s since a's variables
                                        # are not being watched.
  ```

  Note that only tensors with real or complex dtypes are differentiable.
  """

  def __init__(self, persistent=False, watch_accessed_variables=True):
    """Creates a new GradientTape.

    Args:
      persistent: Boolean controlling whether a persistent gradient tape
        is created. False by default, which means at most one call can
        be made to the gradient() method on this object.
      watch_accessed_variables: Boolean controlling whether the tape will
        automatically `watch` any (trainable) variables accessed while the tape
        is active. Defaults to True meaning gradients can be requested from any
        result computed in the tape derived from reading a trainable `Variable`.
        If False users must explicitly `watch` any `Variable`s they want to
        request gradients from.
    """
    self._tape = None
    self._persistent = persistent
    self._watch_accessed_variables = watch_accessed_variables
    self._recording = False
    self._created_eagerly = context.executing_eagerly()
    if self._created_eagerly:
      context.context().start_step()

  def __enter__(self):
    """Enters a context inside which operations are recorded on this tape."""
    self._push_tape()
    return self

  def __exit__(self, typ, value, traceback):
    """Exits the recording context, no further operations are traced."""
    if self._recording:
      self._pop_tape()

  def _push_tape(self):
    if self._recording:
      raise ValueError("Tape is already recording.")
    if self._tape is None:
      self._tape = tape.push_new_tape(
          persistent=self._persistent,
          watch_accessed_variables=self._watch_accessed_variables)
    else:
      tape.push_tape(self._tape)
    self._recording = True

  def _pop_tape(self):
    if not self._recording:
      raise ValueError("Tape is not recording.")
    tape.pop_tape(self._tape)
    self._recording = False

  def __del__(self):
    if self._created_eagerly:
      try:
        context.context().end_step()
      except AttributeError:
        pass
      except TypeError:
        pass

  def watch(self, tensor):
    """Ensures that `tensor` is being traced by this tape.

    Args:
      tensor: a Tensor or list of Tensors.
    """
    for t in nest.flatten(tensor):
      if hasattr(t, "handle"):
        # There are many variable-like objects, all of them currently have
        # `handle` attribute that points to a tensor. If this changes, internals
        # of watch_variable need to change as well.
        tape.watch_variable(self._tape, t)
      else:
        tape.watch(self._tape, t)

  @tf_contextlib.contextmanager
  def stop_recording(self):
    """Temporarily stops recording operations on this tape.

    Operations executed while this context manager is active will not be
    recorded on the tape. This is useful for reducing the memory used by tracing
    all computations.

    For example:

    ```
      with tf.GradientTape(persistent=True) as t:
        loss = compute_loss(model)
        with t.stop_recording():
          # The gradient computation below is not traced, saving memory.
          grads = t.gradient(loss, model.variables)
    ```

    Yields:
      None
    Raises:
      RuntimeError: if the tape is not currently recording.
    """
    if self._tape is None:
      raise RuntimeError(
          "Trying to stop recording a tape which is not recording.")
    self._pop_tape()
    try:
      yield
    finally:
      self._push_tape()

  def reset(self):
    """Clears all information stored in this tape.

    Equivalent to exiting and reentering the tape context manager with a new
    tape. For example, the two following code blocks are equivalent:
    ```
    with tf.GradientTape() as t:
      loss = loss_fn()
    with tf.GradientTape() as t:
      loss += other_loss_fn()
    t.gradient(loss, ...)  # Only differentiates other_loss_fn, not loss_fn


    # The following is equivalent to the above
    with tf.GradientTape() as t:
      loss = loss_fn()
      t.reset()
      loss += other_loss_fn()
    t.gradient(loss, ...)  # Only differentiates other_loss_fn, not loss_fn
    ```

    This is useful if you don't want to exit the context manager for the tape,
    or can't because the desired reset point is inside a control flow construct:

    ```
    with tf.GradientTape() as t:
      loss = ...
      if loss > k:
        t.reset()
    ```
    """
    self._pop_tape()
    self._tape = None
    self._push_tape()

  def watched_variables(self):
    """Returns variables watched by this tape in order of construction."""
    return self._tape.watched_variables()

  def gradient(self,
               target,
               sources,
               output_gradients=None,
               unconnected_gradients=UnconnectedGradients.NONE):
    """Computes the gradient using operations recorded in context of this tape.

    Args:
      target: Tensor (or list of tensors) to be differentiated.
      sources: a list or nested structure of Tensors or Variables. `target`
        will be differentiated against elements in `sources`.
      output_gradients: a list of gradients, one for each element of
        target. Defaults to None.
      unconnected_gradients: a value which can either hold 'none' or 'zero' and
        alters the value which will be returned if the target and sources are
        unconnected. The possible values and effects are detailed in
        'UnconnectedGradients' and it defaults to 'none'.

    Returns:
      a list or nested structure of Tensors (or IndexedSlices, or None),
      one for each element in `sources`. Returned structure is the same as
      the structure of `sources`.

    Raises:
      RuntimeError: if called inside the context of the tape, or if called more
       than once on a non-persistent tape.
      ValueError: if the target is a variable or if unconnected gradients is
       called with an unknown value.
    """
    if self._tape is None:
      raise RuntimeError("GradientTape.gradient can only be called once on "
                         "non-persistent tapes.")
    if self._recording:
      if not self._persistent:
        self._pop_tape()
      else:
        logging.log_first_n(logging.WARN,
                            "Calling GradientTape.gradient on a persistent "
                            "tape inside it's context is significantly less "
                            "efficient than calling it outside the context (it "
                            "causes the gradient ops to be recorded on the "
                            "tape, leading to increased CPU and memory usage). "
                            "Only call GradientTape.gradient inside the "
                            "context if you actually want to trace the "
                            "gradient in order to compute higher order "
                            "derrivatives.", 1)

    flat_targets = nest.flatten(target)
    for t in flat_targets:
      if resource_variable_ops.is_resource_variable(t):
        raise ValueError("GradientTape.gradient is not supported for variable "
                         "targets.")

    flat_sources = nest.flatten(sources)
    flat_sources = [_handle_or_self(x) for x in flat_sources]

    if output_gradients is not None:
      output_gradients = [None if x is None else ops.convert_to_tensor(x)
                          for x in nest.flatten(output_gradients)]

    flat_grad = imperative_grad.imperative_grad(
        self._tape,
        flat_targets,
        flat_sources,
        output_gradients=output_gradients,
        unconnected_gradients=unconnected_gradients)

    if not self._persistent:
      self._tape = None

    grad = nest.pack_sequence_as(sources, flat_grad)
    return grad

  def jacobian(self,
               target,
               sources,
               unconnected_gradients=UnconnectedGradients.NONE,
               parallel_iterations=None,
               experimental_use_pfor=True):
    """Computes the jacobian using operations recorded in context of this tape.

    See http://en.wikipedia.org/wiki/jacobian_matrix_and_determinant for the
    definition of a Jacobian.

    Example usage:

    with tf.GradientTape() as g:
      x  = tf.constant([1.0, 2.0])
      g.watch(x)
      y = x * x
    jacobian = g.jacobian(y, x)
    # jacobian value is [[2., 0.], [0., 4.]]

    Args:
      target: Tensor to be differentiated.
      sources: a list or nested structure of Tensors or Variables. `target`
        will be differentiated against elements in `sources`.
      unconnected_gradients: a value which can either hold 'none' or 'zero' and
        alters the value which will be returned if the target and sources are
        unconnected. The possible values and effects are detailed in
        'UnconnectedGradients' and it defaults to 'none'.
      parallel_iterations: A knob to control how many iterations are dispatched
        in parallel. This knob can be used to control the total memory usage.
      experimental_use_pfor: If true, vectorizes the jacobian computation. Else
        falls back to a sequential while_loop. Vectorization can sometimes fail
        or lead to excessive memory usage. This option can be used to disable
        vectorization in such cases.

    Returns:
      a list or nested structure of Tensors (or IndexedSlices, or None),
      one for each element in `sources`. Returned structure is the same as
      the structure of `sources`.

    Raises:
      RuntimeError: If called on a non-persistent tape with eager execution
        enabled and without enabling experimental_use_pfor.
      ValueError: If vectorization of jacobian computation fails.
    """
    flat_sources = nest.flatten(sources)
    target_static_shape = target.shape
    target_shape = array_ops.shape(target)
    # Note that we push and pop the tape here and below. This is needed since we
    # need gradients through the enclosed operations.
    self._push_tape()
    target = array_ops.reshape(target, [-1])
    self._pop_tape()

    def loop_fn(i):
      self._push_tape()
      y = array_ops.gather(target, i)
      self._pop_tape()
      return self.gradient(y, flat_sources,
                           unconnected_gradients=unconnected_gradients)

    try:
      target_size = int(target.shape[0])
    except TypeError:
      target_size = array_ops.shape(target)[0]

    if experimental_use_pfor:
      try:
        output = pfor_ops.pfor(loop_fn, target_size,
                               parallel_iterations=parallel_iterations)
      except ValueError as err:
        six.reraise(
            ValueError,
            ValueError(
                str(err) + "\nEncountered an exception while vectorizing the "
                "jacobian computation. Vectorization can be disabled by setting"
                " experimental_use_pfor to False."),
            sys.exc_info()[2])
    else:
      if context.executing_eagerly() and not self._persistent:
        raise RuntimeError(
            "GradientTape must be created with persistent=True"
            " to compute the jacobian with eager execution enabled and with "
            " experimental_use_pfor set to False.")
      output = pfor_ops.for_loop(
          loop_fn, [target.dtype] * len(flat_sources), target_size,
          parallel_iterations=parallel_iterations)

    for i, out in enumerate(output):
      if out is not None:
        new_shape = array_ops.concat(
            [target_shape, array_ops.shape(out)[1:]], axis=0)
        out = array_ops.reshape(out, new_shape)
        if context.executing_eagerly():
          out.set_shape(target_static_shape.concatenate(flat_sources[i].shape))
      output[i] = out

    return nest.pack_sequence_as(sources, output)

  def batch_jacobian(self,
                     target,
                     source,
                     unconnected_gradients=UnconnectedGradients.NONE,
                     parallel_iterations=None,
                     experimental_use_pfor=True):
    """Computes and stacks per-example jacobians.

    See http://en.wikipedia.org/wiki/jacobian_matrix_and_determinant for the
    definition of a Jacobian.  This function is essentially an efficient
    implementation of the following:
    `tf.stack([self.jacobian(y[i], x[i]) for i in range(x.shape[0])])`.

    Note that compared to `GradientTape.jacobian` which computes gradient of
    each output value w.r.t each input value, this function is useful when
    `target[i,...] is independent of `source[j,...]` for `j != i`. This
    independence assumption allows more efficient computation as compared to
    `GradientTape.jacobian`. The output, as well as intermediate activations,
    are lower dimensional and avoid a bunch of redundant zeros which would
    result in the jacobian computation given the independence assumption.

    Example usage:
    with tf.GradientTape() as g:
      x = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
      g.watch(x)
      y = x * x
    batch_jacobian = g.batch_jacobian(y, x)
    # batch_jacobian is [[[2,  0], [0,  4]], [[6,  0], [0,  8]]]

    Args:
      target: A tensor with rank 2 or higher and with shape [b, y1, ..., y_n].
        `target[i,...]` should only depend on `source[i,...]`.
      source: A tensor with rank 2 or higher and with shape [b, x1, ..., x_m].
      unconnected_gradients: a value which can either hold 'none' or 'zero' and
        alters the value which will be returned if the target and sources are
        unconnected. The possible values and effects are detailed in
        'UnconnectedGradients' and it defaults to 'none'.
      parallel_iterations: A knob to control how many iterations are dispatched
        in parallel. This knob can be used to control the total memory usage.
      experimental_use_pfor: If true, uses pfor for computing the Jacobian. Else
        uses a tf.while_loop.

    Returns:
      A tensor `t` with shape [b, y_1, ..., y_n, x1, ..., x_m] where `t[i, ...]`
      is the jacobian of `target[i, ...]` w.r.t. `source[i, ...]`, i.e. stacked
      per-example jacobians.

    Raises:
      RuntimeError: If called on a non-persistent tape with eager execution
        enabled and without enabling experimental_use_pfor.
      ValueError: If vectorization of jacobian computation fails or if first
        dimension of `target` and `source` do not match.
    """
    target_shape = target.shape
    if not target_shape.with_rank_at_least(2)[0].is_compatible_with(
        source.shape.with_rank_at_least(2)[0]):
      raise ValueError(
          "Need first dimension of target shape (%s) and "
          "source shape (%s) to match." % (target.shape, source.shape))
    if target_shape.is_fully_defined():
      batch_size = int(target_shape[0])
      target_row_size = target_shape.num_elements() // batch_size
    else:
      target_shape = array_ops.shape(target)
      batch_size = target_shape[0]
      target_row_size = array_ops.size(target) // batch_size
    source_shape = array_ops.shape(source)
    # Flatten target to 2-D.
    # Note that we push and pop the tape here and below. This is needed since we
    # need gradients through the enclosed operations.
    self._push_tape()
    with ops.control_dependencies(
        [check_ops.assert_equal(batch_size, source_shape[0])]):
      target = array_ops.reshape(target, [batch_size, target_row_size])
    self._pop_tape()

    def loop_fn(i):
      self._push_tape()
      y = array_ops.gather(target, i, axis=1)
      self._pop_tape()
      return self.gradient(y, source,
                           unconnected_gradients=unconnected_gradients)

    if experimental_use_pfor:
      try:
        output = pfor_ops.pfor(loop_fn, target_row_size,
                               parallel_iterations=parallel_iterations)
      except ValueError as err:
        six.reraise(
            ValueError,
            ValueError(
                str(err) + "\nEncountered an exception while vectorizing the "
                "batch_jacobian computation. Vectorization can be disabled by "
                "setting experimental_use_pfor to False."),
            sys.exc_info()[2])
    else:
      if context.executing_eagerly() and not self._persistent:
        raise RuntimeError(
            "GradientTape must be created with persistent=True"
            " to compute the batch_jacobian with eager execution enabled and "
            " with experimental_use_pfor set to False.")
      output = pfor_ops.for_loop(loop_fn, target.dtype, target_row_size,
                                 parallel_iterations=parallel_iterations)
    if output is None:
      return None
    output = array_ops.reshape(output,
                               [target_row_size, batch_size, -1])
    output = array_ops.transpose(output, [1, 0, 2])
    new_shape = array_ops.concat([target_shape, source_shape[1:]], axis=0)
    return array_ops.reshape(output, new_shape)
