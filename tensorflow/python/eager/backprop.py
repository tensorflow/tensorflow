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
import threading

import six

from tensorflow.python import pywrap_tensorflow
from tensorflow.python.eager import context
from tensorflow.python.eager import execute
from tensorflow.python.eager import imperative_grad
from tensorflow.python.eager import tape
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect


_op_attr_type_cache = {}


def op_attr_type(op_type, attr_name):
  try:
    return _op_attr_type_cache[(op_type, attr_name)]
  except KeyError:
    with errors.raise_exception_on_not_ok_status() as status:
      h = context.context()._handle  # pylint: disable=protected-access
      attr_type = pywrap_tensorflow.TFE_OpNameGetAttrType(
          h, op_type, attr_name, status)
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


def _magic_gradient_function(op_name, attr_tuple, num_inputs,
                             inputs, outputs, out_grads):
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


_gradient_functions = {}
_gradient_functions_lock = threading.Lock()


_tracing = False


# TODO(apassos) replace this with a mechanism which can happen at the op
# gradient function registration site, to be less error-prone
# TODO(apassos) add ops other than those in nn_grad and math_grad
_ops_which_dont_need_outputs = set([
    "MatMul",
    "Conv2DBackpropInput",
    "Conv2DBackpropFilter",
    "Conv3D",
    "Conv3DBackpropInputV2",
    "AvgPool3D",
    "AvgPool3DGrad",
    "MaxPool3D",
    "MaxPool3DGrad",
    "MaxPool3DGradGrad",
    "BiasAdd",
    "BiasAddV1",
    "BiasAddGrad",
    "Relu6",
    "Softplus",
    "SoftplusGrad",
    "Softsign",
    "ReluGrad",
    "Conv2D",
    "DepthwiseConv2dNative",
    "Dilation2D",
    "AvgPool",
    "AvgPoolGrad",
    "BatchNormWithGlobalNormalization",
    "L2Loss",
    "Sum",
    "Prod",
    "SegmentSum",
    "SegmentMean",
    "SparseSegmentSum",
    "SparseSegmentMean",
    "SparseSegmentSqrtN",
    "SegmentMin",
    "SegmentMax",
    "UnsortedSegmentSum",
    "UnsortedSegmentMax",
    "Abs",
    "Neg",
    "ReciprocalGrad",
    "Square",
    "Expm1",
    "Log",
    "Log1p",
    "TanhGrad",
    "SigmoidGrad",
    "Sign",
    "Sin",
    "Cos",
    "Tan",
    "Add",
    "Sub",
    "Mul",
    "Div",
    "RealDiv",
    "Maximum",
    "Minimum",
    "SquaredDifference",
    "Select",
    "SparseMatMul",
    "BatchMatMul",
    "Complex",
    "Real",
    "Imag",
    "Angle",
    "Conj",
    "Cast",
    "Cross",
    "Cumsum",
    "Cumprod",
    "ReadVariableOp",
    "VarHandleOp",
    "Shape",
])

_ops_which_dont_need_inputs = set([
    "Softmax",
    "LogSoftmax",
    "BiasAdd",
    "Relu",
    "Elu",
    "Selu",
    "SparseSoftmaxCrossEntropyWithLogits",
    "Neg",
    "Inv",
    "Reciprocal",
    "Sqrt",
    "Exp",
    "Tanh",
    "Sigmoid",
    "Real",
    "Imag",
    "Conj",
    "ReadVariableOp",
    "VarHandleOp",
    "Shape",
])


# TODO(agarwal): use an automatic mechanism for handling None arguments to
# gradient functions.
# Some gradient functions can accept None arguments for gradients. The following
# maps the operation name to the indices at which the corresponding gradient
# function can accept None values.
# e.g. FusedBatchNorm outputs 5 values and hence receives 5 gradient values
# during backprop. However the gradient function uses only the first of those
# values and ignores the rest. The entry, "FusedBatchNorm": [1, 2, 3, 4],
# indicates that only the gradient corresponding to index 0 is used, and the
# gradient values at indices 1-4 are ignored (and hence can be None). The
# backprop algorithm can then leverage this by not constructing zeros to
# pass for those indices.
_grad_fn_accepts_none_for_indices = {
    "SoftmaxCrossEntropyWithLogits": [1],
    "FusedBatchNorm": [1, 2, 3, 4]
}


def _record_gradient(op_name, inputs, attrs, results, name):
  """Records gradients for a TensorFlow operation.

  Args:
    op_name: Name of the TensorFlow operation (see REGISTER_OP in C++ code) to
      execute.
    inputs: A flat list of Tensor object inputs to the operation.
    attrs: A tuple with alternating string attr names and attr values for this
      operation.
    results: The results of the operation (as a flat list).
    name: Customized name for the operation.

  Returns:
    A list of maybe-wrapped results. Either Tensors or TensorNodes.

  Raises:
    An exception on error.
  """
  if not tape.could_possibly_record():
    return

  if op_name in _ops_which_dont_need_outputs:
    op_outputs = None
  else:
    # TODO(apassos) this line creates a weak circular reference where the
    # backprop function keeps an output alive which in turn keeps the tape entry
    # alive which keeps the backprop function alive. Figure out how to break
    # this up without breaking second derivatives of ops like Exp whose
    # gradients depend only on the outputs.
    op_outputs = results

  if op_name in _ops_which_dont_need_inputs:
    op_inputs = None
  else:
    op_inputs = inputs

  num_inputs = len(inputs)

  def grad_fn(*orig_outputs):
    """Generated gradient function."""
    result = _magic_gradient_function(op_name, attrs, num_inputs,
                                      op_inputs, op_outputs, orig_outputs)
    if _tracing:
      print("Gradient for", (name if name else op_name), "inputs", op_inputs,
            "output_grads", orig_outputs, "gradients", result)
    return nest.flatten(result)

  tape.record_operation(op_name, results, inputs, grad_fn)
  if _tracing:
    print("Computed op", (name if name else op_name), "inputs", inputs,
          "outputs", results)


execute.record_gradient = _record_gradient


def implicit_val_and_grad(f):
  """Returns a function which differentiates f with respect to variables.

  The wrapped function returns the value and the gradient of f when called with
  the same arguments. The gradient is with respect to all TFE variables which
  have `variable.watch()` called on them by f.

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

  def grad_fn(*args):
    """Computes the gradient of the wrapped function."""
    tape.push_new_tape()
    try:
      end_node = f(*args)
      if end_node is None:
        raise ValueError("Cannot differentiate a function that returns None; "
                         "did you forget to return a value from {}?".format(
                             f.__name__))
      variables = tape.top_tape_watched_variables()
    finally:
      popped_tape = tape.pop_tape()
    sources = [x.handle for x in variables]

    if not sources:
      raise ValueError("No trainable variables were accessed while the "
                       "function was being computed.")
    grad = imperative_grad.imperative_grad(_default_vspace,
                                           popped_tape,
                                           nest.flatten(end_node),
                                           sources)
    return end_node, list(zip(grad, variables))

  return grad_fn


def implicit_grad(f):
  """Returns a function which differentiates f with respect to variables.

  The wrapped function returns the gradient of f when called with the same
  arguments. The gradient is with respect to all TFE variables which have
  `variable.watch()` called on them by f.

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
    args = tf_inspect.getargspec(f).args
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
    of f with respect to all of `params`. The function takes an extra optional
    keyword argument "dy". Setting it allows computation of vector jacobian
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
        args[i] = args[i]._dup()  # pylint: disable=protected-access
      else:
        s.add(tid)
  return args


def val_and_grad_function(f, params=None):
  """Returns a function that computes f and is derivative w.r.t. params.

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

  Returns: function which, when called, returns the value of f and the gradient
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


def make_vjp(f, params=None):
  """Returns a function that computes f and is vjp w.r.t. params.

  The term "vjp" here is an abbreviation for vector-jacobian product.

  Args:
    f: the function to be differentiated.
    params: the parameters (numbers or names) to differentiate with respect to.
       A value of None will differentiate with respect to all parameters.

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
    tape.push_new_tape()
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
        tape.watch(args[i])
        result = f(*args)
        if result is None:
          raise ValueError("Cannot differentiate a function that returns None; "
                           "did you forget to return a value from {}?".format(
                               f.__name__))
        flat_result = nest.flatten(result)
        flat_result = [gen_array_ops.identity(x) for x in flat_result]
        result = nest.pack_sequence_as(result, flat_result)
    finally:
      t = tape.pop_tape()
    def vjp(dy=None):
      if dy is not None:
        dy = [ops.convert_to_tensor(x) for x in nest.flatten(dy)]
      return imperative_grad.imperative_grad(
          _default_vspace, t, nest.flatten(result), sources,
          output_gradients=dy)
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
  if all([isinstance(g, ops.Tensor) for g in gradients]):
    return math_ops.add_n(gradients)
  else:
    assert all([isinstance(g, (ops.Tensor, ops.IndexedSlices))
                for g in gradients])
    indexed_slices_list = []
    for grad in gradients:
      # TODO(xpan): Support nested IndexedSlices and core IndexedSlices
      if isinstance(grad, ops.Tensor):
        indexed_slices = ops.IndexedSlices(
            grad,
            constant_op.constant(range(grad.shape[0])),
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
    return functools.reduce(operator.mul, grad._shape_tuple(), 1)  # pylint: disable=protected-access
  if isinstance(grad, ops.IndexedSlices):
    return functools.reduce(operator.mul, grad.values._shape_tuple(), 1)  # pylint: disable=protected-access
  raise ValueError("`grad` not a Tensor or IndexedSlices.")


_default_vspace = imperative_grad.VSpace(
    num_elements_fn=_num_elements,
    aggregate_fn=_aggregate_grads,
    tensor_id=ops.tensor_id,
    zeros=array_ops.zeros,
    ones_like=lambda x: ops.convert_to_tensor(array_ops.ones_like(x)))


class GradientTape(object):
  """Records operations to use to compute gradients.

  Operations are recorded if:
    - they happen in code marked by this context manager
    - at least one of their inputs is being watched

  Outputs of recorded operations are watched. Variables are automatically
  watched and tensors can be manually watched by calling the watch method on the
  context manager.

  Example usage:

  ```python
  with tfe.GradientTape() as g:
    x = tf.constant(3.0)
    g.watch(x)
    y = x * x
  grad = g.gradient(y, [x])[0]
  assert grad.numpy() == 6.0
  ```

  It is possible to use GradientTapes to compute higher-order derivatives as
  follows:

  ```python
  with tfe.GradientTape() as g:
    x = tf.constant(3.0)
    g.watch(x)
    y = x * x
    with tfe.GradientTape() as gg:
      gg.watch(y)
      z = 2 * y
    inner_grad = gg.gradient(z, [y])[0]
    assert inner_grad.numpy() == 2
    y = y + inner_grad
  grad = g.gradient(y, [x])[0]
  assert grad.numpy() == 6.0
  ```
  """

  def __init__(self):
    self._tape = None

  def __enter__(self):
    tape.push_new_tape()
    return self

  def __exit__(self, typ, value, traceback):
    self._tape = tape.pop_tape()

  def watch(self, tensor):
    """Ensures that `tensor` is being traced by this tape.

    Args:
      tensor: a Tensor or Variable a list of Tensors or Variables.
    """
    for t in nest.flatten(tensor):
      if isinstance(t, resource_variable_ops.ResourceVariable):
        t = t.handle
      tape.watch(t)

  def gradient(self, target, sources):
    """Computes the gradient using information traced by the tape.

    Args:
      target: the tensor to be differentiated.
      sources: a list of Tensors or Variables, the target will be
       differentiated with respect to the sources.

    Returns:
      a list of Tensors (or IndexedSlices, or None), one for each element in
      `sources`.

    Raises:
      RuntimeError: if called inside the context of the tape, or if called more
       than once.
    """
    if self._tape is None:
      raise RuntimeError("GradientTape.gradient can only be called once, and "
                         "only when the context manager has exited.")
    sources = [x.handle if isinstance(x, resource_variable_ops.ResourceVariable)
               else x
               for x in sources]
    grad = imperative_grad.imperative_grad(
        _default_vspace, self._tape, [target], sources)
    self.tape = None
    return grad
