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

import collections
import threading

import six

from tensorflow.python import pywrap_tensorflow
from tensorflow.python.eager import context
from tensorflow.python.eager import execute
from tensorflow.python.eager import tape
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_inspect


# Terminology:
#
#  - op: a possibly composite operation, which has an entry in the tape
#  - target: dy in dx/dy
#  - source: dx in dx/dy
#  - tensor: one of the many inputs or outputs of an operation
#
# Below here we do the gradient algorithm. It works as follows:
#
# First we filter the tape to just the subset of operations we want to
# differentiate. In the process of doing so we count how many times each Tensor
# is used as an input to an op (so we know when we're done computing gradients
# for that Tensor). We also count, for each tape entry, how many of its output
# Tensors need gradients to be computed (Tensors which are not used do not need
# any gradients to be computed).
#
# Finally, we start a backprop stack with a set of tape entries for which we
# have all gradients available. This set usually is a subset of the set of
# targets (not all since targets which have outputs in the tape will not have
# gradients available initially).
#
# Then we repeatedly pop an entry from the stack, run its backprop, and update
# the gradients of its inputs. Once we have computed all gradients for a single
# input we can mark this input as done, and this can trigger adding an entry to
# the stack if all outputs of that entry are now done.
#
# When the stack is empty we have gradients for all tensors we're interested in.


def _prepare_backprop(target, tensor_to_op, op_to_entry, id_sources):
  """Filters the tape to only include relevant entries and counts tensor usages.

  Args:
    target: the target to optimize.
    tensor_to_op: Map from tensor id to key in op_to_entry that produced it.
    op_to_entry: Map from op id to a tape.TapeEntry object
    id_sources: the ids of the sources wrt the gradient is being taken.

  Returns:
    usage counts (how many entries downstream from a tensor use it)
    op_to_entry_map: entry map (a filtered tape, with only the relevant
     entries),
    missing: map from tensor id to how many downstream gradients still need
     to be computed before this tensor's gradient can be computed.
  """
  if isinstance(target, (ops.Tensor)):
    tensor_stack = [ops.tensor_id(target)]
  else:
    tensor_stack = list([ops.tensor_id(x) for x in target])
  tensor_usage_counts = {}
  o_to_e = {}  # Copy of just the bits we need from op_to_entry
  while tensor_stack:
    t = tensor_stack.pop()
    op = tensor_to_op[t]
    # op is None if the tensor is a source (i.e. was watched directly)
    if op is None or op in o_to_e:
      continue
    op_trace = op_to_entry[op]
    o_to_e[op] = op_trace
    for i in op_trace.inputs:
      it = ops.tensor_id(i)
      if it in tensor_usage_counts:
        tensor_usage_counts[it] += 1
      else:
        tensor_usage_counts[it] = 1
        if it not in id_sources and it in tensor_to_op:
          tensor_stack.append(it)
  op_missing_tensor_counts = collections.defaultdict(int)
  for t in tensor_usage_counts:
    if t in tensor_to_op and tensor_to_op[t] is not None:
      op_missing_tensor_counts[tensor_to_op[t]] += 1
  return tensor_usage_counts, o_to_e, op_missing_tensor_counts


def _initialize_backprop_stack(op_to_entry, op_missing_tensor):
  """Returns the set of tape entries which are available for backprop."""
  ready_ops = []
  for op in op_to_entry:
    if op not in op_missing_tensor:
      ready_ops.append(op)
  return ready_ops


def _initial_gradients(target, output_gradients, tensor_usage_counts):
  """Computes the initial gradients for each Tensor."""
  # Initialize the backprop stack
  gradients = collections.defaultdict(list)
  if isinstance(target, ops.Tensor):
    if output_gradients is not None:
      output_gradient = output_gradients
    else:
      output_gradient = array_ops.ones_like(target)
    gradients[ops.tensor_id(target)].append(output_gradient)
  else:
    for i, t in enumerate(target):
      if ops.tensor_id(t) in tensor_usage_counts:
        # Can't provide a gradient of something we're trying to differentiate
        assert output_gradients is None or output_gradients[i] is None
      else:
        if output_gradients is None or output_gradients[i] is None:
          out_grad = ops.ones_like(t)
        else:
          out_grad = output_gradients[i]
        gradients[ops.tensor_id(t)].append(out_grad)
  return gradients


@tf_contextlib.contextmanager
def _no_op():
  yield


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


def imperative_grad(
    target,
    sources,
    output_gradients=None):
  """Computes gradients from the imperatively defined tape on top of the stack.

  Works by filtering the tape, computing how many downstream usages are of each
  tensor and entry, and repeatedly applying backward functions until we have
  gradients for all sources.

  Args:
   target: either a Tensor or list of Tensors to be differentiated.
   sources: list of Tensors for which we want gradients
   output_gradients: if not None, a list of gradient provided for each Target,
    or None if we are to use the target's computed downstream gradient.

  Returns:
   the gradient wrt each of the sources.

  Raises:
    RuntimeError: if something goes wrong.
    ValueError: if there is no sequence of differentiable operations connecting
     a source and any target Tensor. This can happen either if the target is
     not computed based on the source, if the tracing was set up incorrectly,
     or if only non-differentiable functions of the source were used in the
     computation of target.
  """
  if not tape._tape_stack.stack:  # pylint: disable=protected-access
    raise RuntimeError("Computing a gradient with no tape present")
  bp_tape = tape.pop_tape()
  tensor_to_op, op_to_entry, output_to_shape_dtype = bp_tape.export()
  # This overwrites the op_to_entry variable, which will release all memory used
  # to keep traces that are irrelevant to the gradient computation we're doing
  # here.
  id_sources = [ops.tensor_id(t) for t in sources]
  tensor_usage_counts, op_to_entry, op_missing_tensor = _prepare_backprop(
      target, tensor_to_op, op_to_entry, id_sources)
  ready_ops = _initialize_backprop_stack(op_to_entry, op_missing_tensor)
  gradients = _initial_gradients(target, output_gradients,
                                 tensor_usage_counts)
  # Now exhaust the backprop stack
  while ready_ops:
    op = ready_ops.pop()
    op_trace = op_to_entry.pop(op)
    out_gradients = [gradients.pop(t, None) for t in op_trace.output_ids]
    for i in range(len(out_gradients)):
      if out_gradients[i] is None:
        # TODO(apassos) this should be in the right device
        out_gradients[i] = array_ops.zeros(
            *output_to_shape_dtype[op_trace.output_ids[i]])
      else:
        out_gradients[i] = _aggregate_grads(out_gradients[i])

    in_gradients = op_trace.backward_function(
        *(out_gradients + op_trace.side_outputs))
    in_gradients = ([in_gradients]
                    if isinstance(in_gradients, (ops.Tensor,
                                                 ops.IndexedSlices,
                                                 type(None)))
                    else in_gradients)
    for i, t in enumerate(op_trace.inputs):
      if in_gradients[i] is not None:
        gradients[ops.tensor_id(t)].append(in_gradients[i])
      if tensor_usage_counts.get(ops.tensor_id(t), 0) > 0:
        tensor_usage_counts[ops.tensor_id(t)] -= 1
        if ops.tensor_id(t) in tensor_to_op and tensor_usage_counts[
            ops.tensor_id(t)] == 0 and ops.tensor_id(t) not in id_sources:
          in_op = tensor_to_op[ops.tensor_id(t)]
          if in_op is None:
            continue
          if op_missing_tensor.get(in_op, 0) > 0:
            op_missing_tensor[in_op] -= 1
            if op_missing_tensor.get(in_op, 0) == 0:
              ready_ops.append(in_op)
  result = []
  for i, s in enumerate(sources):
    g = gradients.get(ops.tensor_id(s), None)
    if g is None:
      # TODO(apassos): figure out a way to summarize why sources and targets are
      # not connected.
      raise ValueError("There is no sequence of operations connecting source "
                       "tensor %s (%s) to any of the target Tensors. This is "
                       "commonly caused by the tape not recording all "
                       "operations in the forward pass or if by mistake a "
                       "source was only used in non-differentiable operations."
                       % (i, s))
    result.append(_aggregate_grads(g))
  return result


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

  def grad_fn(*orig_outputs):
    """Generated gradient function."""
    result = _magic_gradient_function(op_name, attrs, len(inputs),
                                      inputs, results, orig_outputs)
    if _tracing:
      print("Gradient for", (name if name else op_name), "inputs", inputs,
            "output_grads", orig_outputs, "gradients", result)
    return result

  inputs = [ops.convert_to_tensor(x) for x in inputs]
  tape.record_operation(results, inputs, [], grad_fn)
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
    def train(model, inputs, labels, optimizer):

      def forward_fn():
        prediction = model(inputs)
        return loss_fn(labels, prediction)

      loss, grads_and_vars = implicit_val_and_grad(forward_fn)()
      optimizer.apply_gradients(grads_and_vars)
      return loss
    ```

  Args:
    f: The function to be differentiated.

  Returns:
    A function which, when called, returns a tuple pair.
    Its first element is the value to which the function evaluates.
    Its second element is list of (gradient, variable) pairs.
  """

  def grad_fn(*args):
    """Computes the gradient of the wrapped function."""
    tape.push_new_tape()
    end_node = f(*args)
    variables = tape.top_tape_watched_variables()
    sources = [x.handle for x in variables]
    grad = imperative_grad(end_node, sources)
    return end_node, list(zip(grad, variables))

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
    A function which, when called, returns a list of (gradient, variable) pairs.
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
        "params must be all strings or all integers; got %s." % params)


def gradients_function(f, params=None):
  """Returns a function which differentiates f with respect to params.

  Args:
   f: function to be differentiated.
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

    _, grad = val_and_grad_function(f, params)(*args, **kwds)
    return grad

  return decorated


def val_and_grad_function(f, params):
  """Returns a function that computes f and is derivative w.r.t. params.

  Args:
   f: function to be differentiated.
   params: list of parameter names of f or list of integers indexing the
     parameters with respect to which we'll differentiate. Passing None
     differentiates with respect to all parameters.

  Returns: function which, when called, returns the value of f and the gradient
   of f with respect to all of `params`. The function takes an extra optional
   keyword argument "dy". Setting it allows computation of vector jacobian
   products for vectors other than the vector of ones.

  Raises:
   ValueError: if the params are not all strings or all integers.
  """
  parameter_positions = _get_arg_spec(f, params)

  def decorated(*args, **kwds):
    """Computes the value and gradient of the decorated function."""
    dy = kwds.pop("dy", None)
    if dy is not None:
      dy = ops.convert_to_tensor(dy)
    assert not kwds, "The gradient function can't take keyword arguments."
    tape.push_new_tape()
    sources = []
    args = [ops.convert_to_tensor(x) for x in args]
    for i in parameter_positions:
      sources.append(args[i])
      tape.watch(args[i])
    result = f(*args)
    return result, imperative_grad(
        result,
        sources,
        output_gradients=dy)

  return decorated
