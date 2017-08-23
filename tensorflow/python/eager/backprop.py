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

import threading

from autograd import convenience_wrappers
from autograd import core as ag_core

import six

from tensorflow.python import pywrap_tensorflow
from tensorflow.python.eager import context
from tensorflow.python.eager import execute
from tensorflow.python.eager import tape
from tensorflow.python.eager import tensor
# Imports TensorNode to enable autograd tracing of TF ops. We don't need to use
# any symbols here but import the file just to get the right registrations to
# happen.
from tensorflow.python.eager import tensor_node  # pylint: disable=unused-import
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops as tf_ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import tf_inspect


def op_attr_type(op_type, attr_name):
  with errors.raise_exception_on_not_ok_status() as status:
    h = context.context()._handle  # pylint: disable=protected-access
    op = pywrap_tensorflow.TFE_NewOp(h, op_type, status)
    attr_type = pywrap_tensorflow.TFE_OpGetAttrType(op, attr_name, status)
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


def _magic_gradient_function(op_name, attr_tuple, num_inputs, num_outputs,
                             *tensors):
  """Calls the gradient function of the op.

  Args:
    op_name: the name of the op to be differentiated.
    attr_tuple: the attrs, as a tuple
    num_inputs: the number of inputs to the op
    num_outputs: the number of outputs of the op
    *tensors: a list of tensors, composed of, in order, the inputs, the outputs,
      and the gradients with respect to the outputs.

  Returns:
    the gradients with respect to the inputs of the function, as a list.
  """
  inputs = tensors[:num_inputs]
  outputs = tensors[num_inputs:num_inputs + num_outputs]
  out_grads = tensors[num_inputs + num_outputs:]
  mock_op = _MockOp(attr_tuple, inputs, outputs, op_name)
  grad_fn = tf_ops._gradient_registry.lookup(op_name)  # pylint: disable=protected-access
  if grad_fn is None:
    return [None] * num_inputs
  out_grads = [
      o if (o is not None) else array_ops.zeros_like(outputs[i])
      for i, o in enumerate(out_grads)
  ]
  return grad_fn(mock_op, *out_grads)


_gradient_functions = {}
_gradient_functions_lock = threading.Lock()


_tracing = False


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
  if not any(ag_core.isnode(x) for x in inputs):
    return results
  num_outputs = len(results)
  if num_outputs == 0:
    return results
  if attrs is not None:
    attrs = tuple(tuple(x) if isinstance(x, list) else x for x in attrs)

  # It is imperative we make a copy of results here as otherwise we create a
  # dependency cycle in the captured function and this can delay garbage
  # collecting of the tensors arbitrarily.
  result_copies = results[:]

  def grad_fn(*outputs):
    """Generated gradient function."""
    tensors = inputs + result_copies + list(outputs)
    tensors = [ag_core.getval(x) for x in tensors]
    result = _magic_gradient_function(op_name, attrs, len(inputs),
                                      num_outputs, *(tensors))
    if _tracing:
      print("Gradient for", (name if name else op_name), "inputs", inputs,
            "output_grads", outputs)
    return result

  results = tape.record_operation(results, inputs, [], grad_fn)
  if _tracing:
    print("Computed op", (name if name else op_name), "inputs", inputs,
          "outputs", results)
  return results


execute.record_gradient = _record_gradient


def _ones(shape, dtype):
  return array_ops.fill(shape, tensor.Tensor(1, dtype=dtype))


def _aggregate_grads(gradients):
  """Aggregate gradients of the same tensor."""
  grad_lists = dict()
  for t, g in gradients:
    if id(t) not in grad_lists:
      grad_lists[id(t)] = [(t, g)]
    else:
      grad_lists[id(t)].append((t, g))

  ret = []
  for t, g_list in six.iteritems(grad_lists):
    if len(g_list) == 1:
      ret.append(g_list[0])
    else:
      # TODO(xpan): Aggregate IndexedSlices.
      ret.append((g_list[0][0], math_ops.add_n(zip(*g_list)[1])))
  return ret


def implicit_val_and_grad(f):
  """Returns a function which differentiates f with respect to variables.

  The wrapped function returns the value and the gradient of f when called with
  the same arguments. The gradient is with respect to all TFE variables which
  have `variable.watch()` called on them by f.

  This function is useful when the exact set of variables to differentiate with
  is not known ahead of time.

  Args:
    f: The function to be differentiated.

  Returns:
    A function which, when called, returns the value and gradients.
  """

  def grad_fn(*args, **kwds):
    """Computes the gradient of the wrapped function."""
    tape.push_new_tape()
    end_node = f(*args)
    start_node = tape.pop_tape()
    ag_core.active_progenitors.remove(start_node)
    if not ag_core.isnode(end_node):
      raise ValueError(
          "Target not part of a computation being traced. %s" % end_node)
    if start_node not in end_node.progenitors:
      raise ValueError("Target not derived from source. %s %s" %
                       (end_node.progenitors, repr(start_node)))
    output_gradients = kwds.get("output_gradients", None)
    if output_gradients is None:
      output_gradients = _ones(end_node.shape, end_node.dtype)
    grad = ag_core.backward_pass(output_gradients, end_node, start_node)
    return end_node.value, _aggregate_grads(grad.gradients)

  return grad_fn


def implicit_grad(f):
  """Returns a function which differentiates f with respect to variables.

  The wrapped function returns the gradient of f when called with the same
  arguments. The gradient is with respect to all TFE variables which have
  `variable.watch()` called on them by f.

  This function is useful when the exact set of variables to differentiate with
  is not known ahead of time.

  Args:
    f: The function to be differentiated.

  Returns:
    A function which, when called, returns the gradients.
  """

  def grad_fn(*args, **kwds):
    """Computes the gradient of the wrapped function."""
    return implicit_val_and_grad(f)(*args, **kwds)[1]

  return grad_fn


def _get_arg_spec(f, params):
  args = tf_inspect.getargspec(f).args
  if params is None:
    if not args:
      raise ValueError("When params is None the differentiated function cannot"
                       " only take arguments by *args and **kwds.")
    return range(len(args))
  elif all(isinstance(x, six.string_types) for x in params):
    return [args.index(n) for n in params]
  elif all(isinstance(x, int) for x in params):
    return params
  else:
    raise ValueError(
        "params must be all strings or all integers; got %s" % params)


def gradients_function(f, params=None):
  """Returns a function which differentiates f with respect to params.

  Args:
   f: function to be differentiated.
   params: list of parameter names of f or list of integers indexing the
     parameters with respect to which we'll differentiate. Passing None
     differentiates with respect to all parameters.

  Returns:
   function which, when called, returns the gradient of f with
   respect to all of `params`.

  Raises:
   ValueError: if the params are not all strings or all integers.
  """
  parameter_positions = _get_arg_spec(f, params)

  def decorated(*args, **kwargs):
    tensors = convenience_wrappers.multigrad(f, parameter_positions)(*args,
                                                                     **kwargs)
    return [t.tensor() if isinstance(t, tensor.LazyZero)
            else t for t in tensors]

  return decorated


def val_and_grad_function(f, params=None):
  """Returns a function that computes f and is derivative w.r.t. params.

  Args:
   f: function to be differentiated.
   params: list of parameter names of f or list of integers indexing the
     parameters with respect to which we'll differentiate. Passing None
     differentiates with respect to all parameters.

  Returns:
   function which, when called, returns the value of f and the
   gradient of f with respect to all of `params`.

  Raises:
   ValueError: if the params are not all strings or all integers.
  """
  return convenience_wrappers.value_and_multigrad(f, _get_arg_spec(f, params))
