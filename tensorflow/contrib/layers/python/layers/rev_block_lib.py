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
"""Reversible Residual Block.

From
[The Reversible Residual Network: Backpropagation Without Storing
Activations](https://arxiv.org/abs/1707.04585).

Also contains the @recompute_grad decorator, which recomputes the forward
function on the backwards pass.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import re

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.contrib.framework.python import ops as contrib_framework_ops
from tensorflow.python.eager import backprop
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops as framework_ops
from tensorflow.python.layers import base
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect

__all__ = ["rev_block", "RevBlock", "recompute_grad"]

LAYER_RE = re.compile(".*revlayer_([0-9]*)/([fg])/.*")
_USE_DEFAULT = "__rev_block_lib_default"
_WRONG_VARS_ERR = """\
The variables used on recompute were different than the variables originally
used. The function wrapped with @recompute_grad likley creates its own variable
scope with a default name and has been called twice in the same enclosing scope.
To fix, ensure each call to the function happens in its own unique variable
scope.
"""


def _acc_grads(*lists_of_grads):
  """Accumulates lists of gradients."""
  acc_grads = []
  for grads in zip(*lists_of_grads):
    grads = [g for g in grads if g is not None]
    if grads:
      acc_grads.append(math_ops.add_n(grads))
    else:
      acc_grads.append(None)
  return acc_grads


def _rev_layer_forward(xs, f, g, f_side_input, g_side_input,
                       gate_outputs=False):
  """Forward for 1 reversible layer."""
  x1, x2 = xs
  y1 = x1 + (f(x2, f_side_input) if f_side_input else f(x2))
  y2 = x2 + (g(y1, g_side_input) if g_side_input else g(y1))
  if gate_outputs:
    return control_flow_ops.tuple([y1, y2])
  else:
    return (y1, y2)


def _rev_layer_backward(ys, grad_ys, f, g, f_vars, f_side_input, g_vars,
                        g_side_input):
  """Backprop for 1 layer."""
  y1, y2 = ys
  grad_y1, grad_y2 = grad_ys

  # Reconstruct intermediates and inputs (x1, x2)
  # stop_gradients required on fn inputs to prevent infinite recursion into this
  # grad function on the calls to gradients.
  y1_stop = array_ops.stop_gradient(y1)
  g_side_input = [array_ops.stop_gradient(t) for t in g_side_input]
  gy1 = g(y1_stop, g_side_input) if g_side_input else g(y1_stop)

  x2 = y2 - gy1
  x2_stop = array_ops.stop_gradient(x2)
  f_side_input = [array_ops.stop_gradient(t) for t in f_side_input]
  fx2 = f(x2_stop, f_side_input) if f_side_input else f(x2_stop)

  x1 = y1 - fx2

  # Compute gradients wrt to inputs
  # dL/dy2 * dG(y1)/y1
  grad_gy1_y2 = gradients_impl.gradients(gy1, y1_stop, grad_y2)[0]
  grad_x1 = grad_y1 + grad_gy1_y2
  grad_x2 = (
      gradients_impl.gradients(fx2, x2_stop, grad_y1)[0] + grad_y2 +
      gradients_impl.gradients(fx2, x2_stop, grad_gy1_y2)[0])

  # Compute gradients wrt to vars and side inputs in f and g
  grads1 = gradients_impl.gradients(gy1, g_vars + g_side_input, grad_y2)
  grad_g_vars, grad_g_side = grads1[:len(g_vars)], grads1[len(g_vars):]
  grads2 = gradients_impl.gradients(fx2, f_vars + f_side_input, grad_y1)
  grad_f_y1, grad_f_side1 = grads2[:len(f_vars)], grads2[len(f_vars):]
  grads3 = gradients_impl.gradients(fx2, f_vars + f_side_input, grad_gy1_y2)
  grad_f_y2, grad_f_side2 = grads3[:len(f_vars)], grads3[len(f_vars):]
  grad_f_vars = _acc_grads(grad_f_y1, grad_f_y2)

  grad_f_side = _acc_grads(grad_f_side1, grad_f_side2)

  # Put returns in a tuple to ensure a constant memory budget (i.e. don't want
  # the subsequent layer to start computing and consuming memory based on a
  # subset of these values).
  outputs = ((x1, x2), (grad_x1, grad_x2), (grad_f_vars, grad_f_side),
             (grad_g_vars, grad_g_side))
  tupled = control_flow_ops.tuple(nest.flatten(outputs))
  return nest.pack_sequence_as(outputs, tupled)


def _rev_block_forward(x1,
                       x2,
                       f,
                       g,
                       num_layers=1,
                       f_side_input=None,
                       g_side_input=None,
                       gate_outputs=False):
  """Forward for a series of reversible layers."""
  out = (x1, x2)
  for i in xrange(num_layers):
    out = _rev_layer_forward(
        out, f[i], g[i], f_side_input, g_side_input, gate_outputs=gate_outputs)

  y1, y2 = out
  return y1, y2


def _scope_wrap(fn, scope):

  @functools.wraps(fn)
  def wrap(*args, **kwargs):
    with variable_scope.variable_scope(scope, use_resource=True):
      return fn(*args, **kwargs)

  return wrap


class RevBlock(base.Layer):
  """Block of reversible layers. See rev_block."""

  def __init__(self,
               f,
               g,
               num_layers=1,
               f_side_input=None,
               g_side_input=None,
               use_efficient_backprop=True,
               name="revblock",
               **kwargs):
    super(RevBlock, self).__init__(name=name, **kwargs)

    if isinstance(f, list):
      assert len(f) == num_layers
    else:
      f = [f] * num_layers

    if isinstance(g, list):
      assert len(g) == num_layers
    else:
      g = [g] * num_layers

    f = [_scope_wrap(fn, "revlayer_%d/f" % i) for i, fn in enumerate(f)]
    g = [_scope_wrap(fn, "revlayer_%d/g" % i) for i, fn in enumerate(g)]

    self.f = f
    self.g = g

    self.num_layers = num_layers
    self.f_side_input = f_side_input or []
    self.g_side_input = g_side_input or []

    self._use_efficient_backprop = use_efficient_backprop

  def call(self, inputs, forward=True):
    vs = variable_scope.get_variable_scope()
    vars_before = vs.global_variables()

    if forward:
      x1, x2 = inputs
      out = self._forward(x1, x2)
    else:
      y1, y2 = inputs
      out = self._backward(y1, y2)

    # Add any created variables to the Layer's variable stores
    new_vars = vs.global_variables()[len(vars_before):]
    train_vars = vs.trainable_variables()
    for new_var in new_vars:
      if new_var in train_vars:
        self._trainable_weights.append(new_var)
      else:
        self._non_trainable_weights.append(new_var)

    return out

  def forward(self, x1, x2):
    return self.apply([x1, x2])

  def backward(self, y1, y2):
    return self.apply([y1, y2], forward=False)

  def build(self, _):
    logging.warn("RevBlock constructs its variables on first call, not on "
                 "build.")
    self.built = True

  def _make_efficient_grad_fn(self, inputs_, ys_):
    def _efficient_grad_fn(*grad_ys, **kwargs):
      """Custom gradient fn for a block of reversible residual layers."""
      inputs = inputs_
      ys = ys_
      variables = kwargs["variables"]
      side_inputs = inputs[2:]

      f_side_idxs = [None] * len(self.f_side_input)
      g_side_idxs = [None] * len(self.g_side_input)
      assert len(side_inputs) == len(self.f_side_input) + len(self.g_side_input)

      for i, t in enumerate(side_inputs):
        if t in self.f_side_input:
          f_side_idxs[self.f_side_input.index(t)] = i
        elif t in self.g_side_input:
          g_side_idxs[self.g_side_input.index(t)] = i
        else:
          assert False

      f_vars = [[] for _ in range(self.num_layers)]
      g_vars = [[] for _ in range(self.num_layers)]
      f_vars_idxs = [[] for _ in range(self.num_layers)]
      g_vars_idxs = [[] for _ in range(self.num_layers)]

      for i, ref in enumerate(variables):
        # Use the name to identify the layer number and function (f or g)
        regex = LAYER_RE.match(ref.name)
        layer_no = int(regex.group(1))
        fn_name = regex.group(2)
        if fn_name == "f":
          f_vars[layer_no].append(ref)
          f_vars_idxs[layer_no].append(i)
        else:
          assert fn_name == "g"
          g_vars[layer_no].append(ref)
          g_vars_idxs[layer_no].append(i)

      f_var_grads = []
      g_var_grads = []
      f_side_grads = []
      g_side_grads = []

      # Reverse variable containers to go backward
      f_vars.reverse()
      g_vars.reverse()
      f = list(self.f)
      g = list(self.g)
      f.reverse()
      g.reverse()

      with variable_scope.variable_scope(self.scope_name, reuse=True):
        for i in xrange(self.num_layers):
          ys, grad_ys, f_ret, g_ret = _rev_layer_backward(
              ys, grad_ys, f[i], g[i], f_vars[i], self.f_side_input, g_vars[i],
              self.g_side_input)

          grad_f_vars, grad_f_side = f_ret
          grad_g_vars, grad_g_side = g_ret
          f_var_grads.append(grad_f_vars)
          g_var_grads.append(grad_g_vars)
          f_side_grads.append(grad_f_side)
          g_side_grads.append(grad_g_side)

      # Accumulate layer gradients for f_side_input and g_side_input
      acc_f_side_grads = _acc_grads(*f_side_grads)
      acc_g_side_grads = _acc_grads(*g_side_grads)

      # Use the stored idxs to put gradients in the passed-in order.
      side_input_grads = [None] * len(side_inputs)
      variable_grads = [None] * len(variables)

      # Variable gradients were collected in reverse layer order. Reverse to
      # match idxs.
      f_var_grads.reverse()
      g_var_grads.reverse()
      for idxs, grads in list(zip(f_vars_idxs, f_var_grads)) + list(
          zip(g_vars_idxs, g_var_grads)):
        for i, grad in zip(idxs, grads):
          variable_grads[i] = grad

      for i, grad in zip(f_side_idxs, acc_f_side_grads):
        side_input_grads[i] = grad
      for i, grad in zip(g_side_idxs, acc_g_side_grads):
        side_input_grads[i] = grad

      grad_x1, grad_x2 = grad_ys
      return [grad_x1, grad_x2] + side_input_grads, variable_grads
    return _efficient_grad_fn

  def _forward(self, x1, x2):
    """Run forward through the reversible layers."""

    side_inputs = [self.f_side_input, self.g_side_input]
    flat_side_inputs = nest.flatten(side_inputs)

    def _forward_wrap(x1_, x2_, *flat_side_inputs):
      f_side, g_side = nest.pack_sequence_as(side_inputs, flat_side_inputs)
      return _rev_block_forward(
          x1_,
          x2_,
          self.f,
          self.g,
          num_layers=self.num_layers,
          f_side_input=f_side,
          g_side_input=g_side,
          gate_outputs=self._use_efficient_backprop)

    @custom_gradient.custom_gradient
    def _forward_with_custom_grad(*args):
      out = _forward_wrap(*args)  # pylint: disable=no-value-for-parameter
      grad_fn = self._make_efficient_grad_fn(args, out)
      return out, grad_fn

    if self._use_efficient_backprop:
      return _forward_with_custom_grad(x1, x2, *flat_side_inputs)
    else:
      return _forward_wrap(x1, x2, *flat_side_inputs)

  def _backward(self, y1, y2):
    """Run backward through the reversible layers."""

    f = list(self.f)
    g = list(self.g)
    f.reverse()
    g.reverse()

    for i in xrange(self.num_layers):
      gy1 = g[i](y1, self.g_side_input) if self.g_side_input else g[i](y1)
      x2 = y2 - gy1
      fx2 = f[i](x2, self.f_side_input) if self.f_side_input else f[i](x2)
      x1 = y1 - fx2

      y1, y2 = x1, x2

    return x1, x2


def rev_block(x1,
              x2,
              f,
              g,
              num_layers=1,
              f_side_input=None,
              g_side_input=None,
              is_training=True):
  """A block of reversible residual layers.

  A reversible residual layer is defined as:

  ```
  y1 = x1 + f(x2, f_side_input)
  y2 = x2 + g(y1, g_side_input)
  ```

  A reversible residual block, defined here, is a series of reversible residual
  layers.

  Limitations:
  * f and g must not close over any Tensors; all side inputs to f and g should
    be passed in with f_side_input and g_side_input which will be forwarded to
    f and g.
  * f and g must not change the dimensionality of their inputs in order for the
    addition in the equations above to work.

  Args:
    x1: a float Tensor.
    x2: a float Tensor.
    f: a function, (Tensor) -> (Tensor) (or list of such of length num_layers).
      Should not change the shape of the Tensor. Can make calls to get_variable.
      See f_side_input if there are side inputs.
    g: a function, (Tensor) -> (Tensor) (or list of such of length num_layers).
      Should not change the shape of the Tensor. Can make calls to get_variable.
      See g_side_input if there are side inputs.
    num_layers: int, number of reversible residual layers. Each layer will
      apply f and g according to the equations above, with new variables in each
      layer.
    f_side_input: list of Tensors, side input to f. If not None, signature of f
      should be (Tensor, list<Tensor>) -> (Tensor).
    g_side_input: list of Tensors, side input to g. If not None, signature of g
      should be (Tensor, list<Tensor>) -> (Tensor).
    is_training: bool, whether to actually use the efficient backprop codepath.

  Returns:
    y1, y2: tuple of float Tensors.
  """
  block = RevBlock(
      f=f,
      g=g,
      num_layers=num_layers,
      f_side_input=f_side_input,
      g_side_input=g_side_input,
      use_efficient_backprop=is_training,
      _reuse=variable_scope.get_variable_scope().reuse)
  return block.forward(x1, x2)


def enable_with_args(dec):
  """A decorator for decorators to enable their usage with or without args."""

  @functools.wraps(dec)
  def new_dec(*args, **kwargs):
    if len(args) == 1 and not kwargs and callable(args[0]):
      # Used as decorator without args
      fn = args[0]
      return dec(fn)
    else:
      return lambda fn: dec(fn, *args, **kwargs)

  return new_dec


@enable_with_args
def recompute_grad(fn, use_data_dep=_USE_DEFAULT, tupleize_grads=False):
  """Decorator that recomputes the function on the backwards pass.

  To use this function, you must use `ResourceVariable`s (i.e.
  `variable_scope(name, use_resource=True), which are the default in Eager mode
  and when running on TPU.

  Warning: Because the function will be called again on the backwards pass, the
  user should be careful to not use ops in their function that mutate state or
  have randomness (for example, batch normalization or dropout). If the function
  does have such operations, it is recommended that the function take the
  `is_recomputing` keyword argument which will be `False` on the forward pass
  and `True` on the backwards pass so that it can disable state changes when
  `is_recomputing=True` (for example, not updating the moving averages in batch
  normalization).

  Args:
    fn: a function that takes Tensors (all as positional arguments) and returns
      a tuple of Tensors.
    use_data_dep: `bool`, if `True` will use a dummy data dependency to force
      the recompute to happen. If `False` will use a control dependency. By
      default will be `True` if in an XLA context and `False` otherwise. XLA
      ignores control dependencies and so this data dependency is necessary.
    tupleize_grads: `bool`, if `True` will use control dependencies to ensure
      that all gradients are produced before any are consumed by downstream ops.
      If `use_data_dep` is also `True`, will use a data dependency instead of
      a control dependency.

  Returns:
    A wrapped fn that is identical to fn when called, but its activations will
    be discarded and recomputed on the backwards pass (i.e. on a call to
    tf.gradients).
  """

  @functools.wraps(fn)
  def wrapped(*args):
    return _recompute_grad(
        fn, args, use_data_dep=use_data_dep, tupleize_grads=tupleize_grads)

  return wrapped


def _is_on_tpu():
  ctxt = framework_ops.get_default_graph()._get_control_flow_context()  # pylint: disable=protected-access
  return control_flow_util.GetContainingXLAContext(ctxt) is not None


def _recompute_grad(fn, args, use_data_dep=_USE_DEFAULT, tupleize_grads=False):
  """See recompute_grad."""
  has_is_recompute_kwarg = "is_recomputing" in tf_inspect.getargspec(fn).args
  for arg in args:
    if not isinstance(arg, framework_ops.Tensor):
      raise ValueError("All inputs to function must be Tensors")
  use_data_dep_ = use_data_dep
  if use_data_dep_ == _USE_DEFAULT:
    use_data_dep_ = _is_on_tpu()

  @custom_gradient.custom_gradient
  def fn_with_recompute(*args):
    """Wrapper for fn."""
    # Forward pass
    vs = variable_scope.get_variable_scope()
    arg_scope = contrib_framework_ops.current_arg_scope()
    with backprop.GradientTape() as tape:
      fn_kwargs = {}
      if has_is_recompute_kwarg:
        fn_kwargs["is_recomputing"] = False
      outputs = fn(*args, **fn_kwargs)
    original_vars = set(tape.watched_variables())

    # Backward pass
    def grad_fn(*output_grads, **kwargs):
      """Recompute outputs for gradient computation."""
      variables = []
      if original_vars:
        variables = kwargs["variables"]
      if set(variables) != original_vars:
        raise ValueError(_WRONG_VARS_ERR)
      del kwargs
      inputs = list(args)
      # Recompute outputs
      with framework_ops.control_dependencies(output_grads):
        if use_data_dep_:
          inputs = _force_data_dependency(output_grads, inputs)
        with contrib_framework_ops.arg_scope(arg_scope):
          with variable_scope.variable_scope(vs, reuse=True):
            with backprop.GradientTape() as tape:
              fn_kwargs = {}
              if has_is_recompute_kwarg:
                fn_kwargs["is_recomputing"] = True
              outputs = fn(*inputs, **fn_kwargs)
            recompute_vars = set(tape.watched_variables())
            if original_vars != recompute_vars:
              raise ValueError(_WRONG_VARS_ERR)

      if not (isinstance(outputs, list) or isinstance(outputs, tuple)):
        outputs = [outputs]
      outputs = list(outputs)
      grads = gradients_impl.gradients(outputs, inputs + variables,
                                       output_grads)

      if tupleize_grads:
        if use_data_dep_:
          grads = _tuple_with_data_dep(grads)
        else:
          grads = control_flow_ops.tuple(grads)

      grad_inputs = grads[:len(inputs)]
      grad_vars = grads[len(inputs):]
      return grad_inputs, grad_vars

    return outputs, grad_fn

  return fn_with_recompute(*args)


def _underlying_variable_ref(t):
  """Find the underlying variable ref.

  Traverses through Identity, ReadVariableOp, and Enter ops.
  Stops when op type has Variable or VarHandle in name.

  Args:
    t: a Tensor

  Returns:
    a Tensor that is a variable ref, or None on error.
  """
  while t.op.type in ["Identity", "ReadVariableOp", "Enter"]:
    t = t.op.inputs[0]

  op_type = t.op.type
  if "Variable" in op_type or "VarHandle" in op_type:
    return t
  else:
    return None


def _force_data_dependency(first_compute, then_compute):
  """Force all of `then_compute` to depend on all of `first_compute`.

  Uses a dummy data dependency, which is useful when running on TPUs because
  XLA ignores control dependencies. Only supports float arguments.

  Args:
    first_compute: `list<Tensor>`. These will be made to run before the
      `Tensor`s `then_compute`.
    then_compute: `list<Tensor>`. These will run after all the `Tensor`s in
      `first_compute`.

  Returns:
    `list<Tensor>`, same length as `then_compute`.

  Raises:
    ValueError: if ranks are unknown or types are not floating.
  """

  def _first_element(x):
    if x.get_shape().ndims is None:
      raise ValueError("Rank of Tensor %s must be known" % x)
    ndims = x.get_shape().ndims
    begin = framework_ops.convert_to_tensor([0] * ndims, dtype=dtypes.int32)
    size = framework_ops.convert_to_tensor([1] * ndims, dtype=dtypes.int32)
    return array_ops.reshape(array_ops.slice(x, begin, size), [])

  first_compute_sum = math_ops.add_n(
      [_first_element(x) for x in first_compute if x is not None])
  dtype = first_compute_sum.dtype
  if not dtype.is_floating:
    raise ValueError("_force_data_dependency only supports floating dtypes.")
  epsilon = np.finfo(dtype.as_numpy_dtype).tiny
  zero = array_ops.stop_gradient(epsilon * first_compute_sum)

  return [
      array_ops.identity(x) + zero if x is not None else None
      for x in then_compute
  ]


def _tuple_with_data_dep(tensors):
  return _force_data_dependency(tensors, tensors)
