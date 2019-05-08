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
# ==============================================================================
"""Recurrent computation.

The main interface of this module is Recurrent().
A recurrent computation describes an auto-regressive process, where outputs
of one time step are fed to the output of the next time step.

This module uses:
  theta: the "weights" each RNN uses.
  state0: the initial state of each RNN.
  cell_fn: A python function describing RNN cell. It must has the following
    signature:
         cell_fn: (theta, state0, inputs) -> (state1, extras)
    state1 is the next RNN state, extras are computed by cell_fn
    and the library forwards extras to cell_fn's gradient function.
  cell_grad: A python function describing the backprop gradient function
    for the RNN cell. It must has the following signature:
         cell_grad: (theta, state0, inputs, extras, dstate1) -> (
                  dtheta, dstate0, dinputs)
    dstate1 is what the backprop algorithm provides representing
    gradients of state1 w.r.t. the final loss.

In this module, we handle structures of tensors for theta, state0, inputs,
and extras. The structure is an arbitrarily nested python structure, such
as a dictionary of named tuples.

Because the computation is a left-to-right chain, a single in-place accumulator
can be used rather than a stack. Thus a special gradient was written to reduce
unnecessary memory usage.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import inplace_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.inplace_ops import alias_inplace_update
from tensorflow.python.util import nest


def _AssertIsCompatible(a, b):
  """Checks that `a` and `b` are nested structures of the same type."""
  # TODO(drpng): implement.
  del a
  del b


def _Index(struct, index):
  """Returns a structure with `x[index]` for each tensor `x` in the structure.

  Args:
    struct: A structure of tensors.
    index: A scalar integer tensor. Performance is better if `index` is
      on the host memory.

  Returns:
    A structure of tensors congruent to `struct`.
    For each key in `ret`, `rets[key] = struct[key][index]`.
  """
  index = ops.convert_to_tensor(index)
  index.get_shape().assert_has_rank(0)
  return nest.map_structure(lambda x: array_ops.gather(x, index), struct)


def _Update(struct_acc, struct_x, t):
  """Updates t-th row in accumulators.

  Args:
    struct_acc: The accumulators. A structure of tensors.
    struct_x: The new values. A structure of tensors congruent to `struct_acc`.
    t: A scalar integer. Performance is better if `t` is on the device
      memory.

  Returns:
    A structure of tensors. Say, ret is a returned dictionary. Then, for
    each key, we have:
      ret[key] = struct_acc[key];
      ret[key][t, :] = struct_x[key]
  """
  to_skip_update = set()
  acc_lst = nest.flatten(struct_acc)
  x_lst = nest.flatten(struct_x)
  t = math_ops.cast(
      [t], dtypes.int32)  # tf.compat.v1.to_int32 casts on-device tensors.
  lst = []
  for acc, x in zip(acc_lst, x_lst):
    if acc in to_skip_update:
      # Until b/62105730 is fixed, we need to avoid inplace update for tensors
      # of rank 1.  could reshape to handle it, but we don't really need the
      # values applied to these, so just skip their modification.
      lst += [acc]
    else:
      lst += [alias_inplace_update(acc, t, array_ops.expand_dims(x, 0))]
  return nest.pack_sequence_as(struct_acc, lst)


def _SeqLenDim(struct):
  """Returns the 0-th dim size of tensors in a structure of tensors.

  This is the max sequence length according to the shape of the inputs.

  Args:
    struct: A structure of tensors. Every tensor's 0-th dim has the same size.

  Returns:
    A scalar tensor which is the size of 0-th dim of every tensors in struct.
  """
  xs = nest.flatten(struct)
  assert xs
  dim0 = array_ops.shape(xs[0])[0]
  return dim0


def _Flatten(struct):
  """Flattens a structure."""
  return nest.flatten(struct)


def _Pack(elements, struct_template):
  """Packs the list of tensors according to the structure.

  In the event that `elements` should be a scalar, `struct_template` must
  contain exactly one non-trivial element (for instance, `[[], {'x':elt}]`).

  Args:
    elements: Elements to be packed. A list of tensor, or a single tensor.
    struct_template: The container structure in which to pack them.
  Returns:
    A python structure of the same type as `struct_template`, containing
    `elements` as its contained elements.
  """
  if not nest.is_sequence(elements):
    return nest.pack_sequence_as(struct_template, [elements])
  return nest.pack_sequence_as(struct_template, elements)


def _EmptyAcc(slen, struct_template):
  """Creates a set of accumulators for tensors in structure.

  Args:
    slen: The sequence length. A scalar tensor.
    struct_template: A structure of tensors.

  Returns:
    A structure congruent to `struct_template`. Say ret is a returned
    dictionary. Then, `ret.key`, a tensor, has the same dtype as
    `struct_template.key`. The tensor's shape has 1 more dimension
    than the tensor `struct_template.key`. The extra 0-th dimension is of size
    `slen`. E.g., if `slen=10` and `struct_template.key`'s shape is `[3, 5]`,
    then, `ret.key`'s shape is `[10, 3, 5]`.
  """

  def _EmptyAccForTensor(tensor):
    return inplace_ops.empty(
        array_ops.concat([[slen], array_ops.shape(tensor)], axis=0),
        tensor.dtype,
        init=True)

  return nest.map_structure(_EmptyAccForTensor, struct_template)


def _EmptyLike(struct):
  """Creates a set of empty initialized tensors.

  Args:
    struct: A structure of tensors.

  Returns:
    A struct of tensors. Each tensor has the same shape and dtype as
    its corresponding tensor in `struct`. And each tensor is initialized.
  """
  return nest.map_structure(
      lambda x: inplace_ops.empty_like(x, init=True), struct)


def _Add(struct_x, struct_y):
  """Adds tensors in `struct_x` with respective tensors in `struct_y`.

  Args:
    struct_x: A struct of tensors.
    struct_y: A struct of tensors congruent to `struct_x`.

  Returns:
    A struct of tensors. Each element of the returned value
  equals `x + y`, with corresponding values in `struct_x` and `struct_y`.
  """
  list_x = nest.flatten(struct_x)
  list_y = nest.flatten(struct_y)
  z = []
  for x, y in zip(list_x, list_y):
    z += [math_ops.add(x, y)]
  return nest.pack_sequence_as(struct_x, z)


def _Dtypes(struct):
  """Returns all tensors' data types in a list."""
  return [x.dtype for x in nest.flatten(struct)]


def _ConvertNoneGradientToZeros(xs, dxs):
  """Sanitize dxs so that None becomes zeros appropriately.

  Args:
    xs: A list of tensors.
    dxs: A list of tensors. dxs[i] corresponds to xs[i]'s gradient.

  Returns:
    A structure same as `dxs` with `None` replaced by a zero tensor.
  """
  list_xs = nest.flatten(xs)
  list_dxs = nest.flatten(dxs)

  # If x does not get any backprop-ed gradient, propagate zeros.
  rets = []
  for (x, dx) in zip(list_xs, list_dxs):
    if dx is None:
      rets.append(array_ops.zeros_like(x))
    else:
      rets.append(dx)

  return nest.pack_sequence_as(dxs, rets)


# All structures are flattened for use internally. This is for simplicity
# and also to use the Defun construct.
# In the forward pass (inference), the computation is structured as follows.
# Forward: [gradient = _Recurrent.Grad]
#   Flatten structures, create accumulators.
#   for t = 0..max_input_length:
#     Defun ForwardLoopBody:
#       Defun Fwd: flatten/pack around cell_fn
#       state1 = Fwd(inputs[t], state0)
#       acc_state += [state1]
#   Pack structures.
# During the backward pass (backpropping the gradient from the last time
# step to the first, through the structure), the computation is structured
# as follows.
# Grad:
#   Flatten structures.
#   Defun Backward:
#     Create create accumulated derivatives: d_theta, d_inputs, d_acc_state.
#     Regarding the note at the top of the file, there is only one accumulator
#     for d_theta accumulated over the whole sequence.
#     for t = max_input_length -1..0:
#       Defun BackwardLoopBody:
#         Retrieve acc_state[t] computed in the forward pass.
#         Defun Bak: flatten/back around cell_fn_grad.
#         d_state1 is d_state0 from previous step (ie next time).
#         d_acc_state[dev_t] += d_state1
#         d_theta_t, d_state0, d_inputs_t, = Bak()
#         d_inputs[dev_t] += d_inputs
#         d_theta += d_theta_t
#         d_acc_state[t] += d_state1
#   Pack structures and return.
class _Recurrent(object):
  """A helper class to construct a recurrent neural net."""

  def __init__(self,
               cell_fn,
               cell_grad,
               theta,
               state0,
               inputs,
               max_input_length,
               extras,
               use_tpu,
               aligned_end=False):
    """RNN helper class.

    Args:
      cell_fn: A python function, which computes:
         state1, extras = cell_fn(theta, state0, inputs[t, :])
      cell_grad: A python function which computes:
         dtheta, dstate0, dinputs[t, :] = cell_grad(
           theta, state0, inputs[t, :], extras, dstate1)
      theta: weights. A structure of tensors.
      state0: initial state. A structure of tensors.
      inputs: inputs. A structure of tensors.
      max_input_length: None, or the maximum effective length of the input over
        all batches. A scalar tensor.
      extras: A structure of tensors. The 2nd return value of every
        invocation of cell_fn is a structure of tensors with matching keys
        and shapes of this `extras`.
      use_tpu: A boolean indicating whether the computation is mean to
        run on a TPU.
      aligned_end: A boolean indicating whether the sequence is aligned at
        the end.
    """
    self._theta = theta
    self._state = state0
    self._inputs = inputs
    self._max_input_length = self._MaybeComputeMaxInputLength(
        inputs, max_input_length)
    self._cell_fn = cell_fn
    self._cell_grad = cell_grad
    self._extras = extras
    self._aligned_end = aligned_end

    # pylint: disable=unbalanced-tuple-unpacking

    # NOTE: TF Function (Fwd, Bak, ForwardLoopBody, BackwardLoopBody,
    # Forward and Backward defined below) simply takes a list of
    # Tensors and returns a list of Tensors. When we pass in a
    # structure (a list of structures of Tensors), we use _Flatten to
    # convert the structure into a list of tensor. Conversely, the
    # following code often uses _Pack to formulate a structure from a
    # list of tensors based on a "template".

    # Wraps cell_fn in a TF Function:
    #    state1 = cell_fn(theta, state0, inputs)
    fwd_sig = [self._theta, self._state, self._inputs]

    compiled = use_tpu
    noinline = not compiled
    dev_t_type = dtypes.int32 if use_tpu else dtypes.int64

    @function.Defun(*_Dtypes(fwd_sig))
    def Fwd(*args):
      (theta, state0, inputs) = _Pack(args, fwd_sig)
      state1, extras = self._cell_fn(theta, state0, inputs)
      assert not function.get_extra_args(), (
          'cell_fn is not pure with extra args: %s.' %
          (function.get_extra_args()))
      _AssertIsCompatible(state1, self._state)
      _AssertIsCompatible(extras, self._extras)
      return _Flatten([state1, extras])

    # Wraps cell_fn in a TF Function as a for-loop's body.
    #
    # The loop state is composed of:
    #  t: The loop variable. Timestep id.
    #  dev_t: The loop variable mirrored on the device.
    #  theta: the recurrent net's weights.
    #  state0: the previous recurrent state.
    #  inputs: inputs to the recurrent net. inputs[t, :] are for the timestep t.
    #  acc_state: Each timestep's computed new state is also stashed into
    #    acc_state.
    #  acc_extras: Each timestep's computed extras is stashed into acc_extras
    fwdloop_sig = [
        self._theta, self._state, self._inputs, self._state, self._extras
    ]

    @function.Defun(dtypes.int32, dev_t_type, *_Dtypes(fwdloop_sig))
    def ForwardLoopBody(*args):
      """The body of forward loop."""
      t, dev_t = args[0], args[1]
      (theta, state0, inputs, acc_state, acc_extras) = _Pack(
          args[2:], fwdloop_sig)
      inputs_t = _Index(inputs, t)  # external input at time step t.
      fwd = Fwd(*_Flatten([theta, state0, inputs_t]))
      state1, extras = _Pack(fwd, [self._state, self._extras])
      # Saves state1 and extras in their accumulators.
      acc_state = _Update(acc_state, state1, dev_t)
      acc_extras = _Update(acc_extras, extras, dev_t)

      return [math_ops.add(dev_t, 1)] + _Flatten(
          [theta, state1, inputs, acc_state, acc_extras])

    def Grad(op, *args):
      """The python grad function for the Forward function."""

      # NOTE: tf.gradient backprops None for int32/int64 while zeros
      # for float32/float64. For consistency, we always backprop
      # zeros.
      args = list(args)
      for i, dy in enumerate(args):
        if dy is None:
          args[i] = array_ops.zeros_like(op.outputs[i])
      # TODO(drpng): getting the extra state here?
      op_inputs = [x for x in op.inputs]
      op_struct = [
          self._theta, self._state, self._inputs, self._max_input_length,
          self._extras
      ]
      (theta, state0, inputs, max_input_length, _) = _Pack(op_inputs, op_struct)
      # acc_state and acc_extras are computed by the Forward pass and
      # needed by the Backward pass.
      acc_state, _, acc_extras = _Pack([x for x in op.outputs],
                                       [self._state, self._state, self._extras])

      # Forward computes acc_state, the final state and
      # acc_extras. tf.gradients gives us their gradients w.r.t. the
      # final loss. Because acc_extras are not exposed by Compute(),
      # it has no gradients w.r.t. the final loss (i.e., by
      # construction, it must be zeros).
      d_acc_state, d_state1, _ = _Pack(args,
                                       [self._state, self._state, self._extras])
      return Backward(*_Flatten([
          theta, state0, inputs, max_input_length, acc_state, acc_extras,
          d_acc_state, d_state1
      ]))

    # Forward calls ForwardLoopBody n times. Each time computes one
    # time step of the recurrent net.
    forward_sig = [
        self._theta, self._state, self._inputs, self._max_input_length,
        self._extras
    ]

    @function.Defun(
        *_Dtypes(forward_sig), python_grad_func=Grad, noinline=noinline)
    def Forward(*args):
      """Forward pass of the recurrent net."""
      theta, state0, inputs, max_input_length, extras = _Pack(args, forward_sig)

      slen_dim = _SeqLenDim(inputs)

      # Creates accumulators for state0 and extras.
      acc_state = _EmptyAcc(slen_dim, state0)
      acc_extras = _EmptyAcc(slen_dim, extras)

      t = slen_dim - max_input_length if self._aligned_end else 0
      dev_t = math_ops.cast(t, dtypes.int32) if use_tpu else math_ops.cast(
          t, dtypes.int64)
      run = functional_ops.For(
          start=t,
          limit=slen_dim if self._aligned_end else max_input_length,
          delta=1,
          inputs=[dev_t] + _Flatten(
              [theta, state0, inputs, acc_state, acc_extras]),
          body=ForwardLoopBody,
          rewrite_with_while=compiled)
      _, state1, _, acc_state, acc_extras = _Pack(
          run[1:],
          [self._theta, self._state, self._inputs, self._state, self._extras])

      return _Flatten([acc_state, state1, acc_extras])

    # The per-step backward computes:
    #    d_theta, d_state0, d_inputs = cell_grad(
    #        theta, state0, inputs, extras, d_state1)
    # where d_state1 is the backprop-ed gradient for state1, and
    # extras is the computed by the forward step to facilitate the
    # backward step.
    bak_sig = [
        self._theta, self._state, self._inputs, self._extras, self._state
    ]

    @function.Defun(*_Dtypes(bak_sig))
    def Bak(*args):
      """Backward step."""
      (theta, state0, inputs, extras, d_state1) = _Pack(args, bak_sig)
      (dtheta, dstate0, dinputs) = self._cell_grad(theta, state0, inputs,
                                                   extras, d_state1)
      assert not function.get_extra_args(), (
          'cell_grad is not pure with extra args: %s.' %
          (function.get_extra_args()))
      _AssertIsCompatible(dtheta, self._theta)
      _AssertIsCompatible(dstate0, self._state)
      _AssertIsCompatible(dinputs, self._inputs)
      return _Flatten(
          _ConvertNoneGradientToZeros([theta, state0, inputs],
                                      [dtheta, dstate0, dinputs]))

    # Define defuns used by a functional_ops.If in BackwardLoopBody.
    state_if_sig = [self._state, self._state]

    @function.Defun(*_Dtypes(state_if_sig))
    def ReturnOrigState0(*args):
      """Returns original state0 from inputs."""
      (_, orig_state0) = _Pack(args, state_if_sig)
      return nest.flatten(orig_state0)

    @function.Defun(*_Dtypes(state_if_sig))
    def ReturnAccState(*args):
      """Returns acc_state[t-1] from inputs."""
      (acc_state, _) = _Pack(args, state_if_sig)
      return nest.flatten(acc_state)

    # Wraps cell_grad gradient function in a TF Function as a
    # for-loop's body for the Backward pass.
    #
    # The loop state is composed of:
    #  t: The loop variable. Timestep id.
    #  state0: the initial state for the entire backward loop.
    #  dev_t: The loop variable mirrored on the device.
    #  theta: the recurrent net's weights.
    #  inputs: inputs to the recurrent net. inputs[t, :] are for the timestep t.
    #  acc_state: Each timestep's computed new state was stashed into
    #    acc_state by the Forward pass.
    #  acc_extras: Each timestep's computed extras was stashed into
    #    acc_extras by the Forward pass.
    #  d_theta: All timestep's gradient for theta is accumulated (added) into
    #      d_theta.
    #  d_state1: The backprop-ed gradient for the new stated computed by
    #      timestep t.
    #  d_inputs: d_inputs[t, :] is populated by the backward time step t.
    #  d_acc_state: The backprop-ed gradient for acc_state.
    bakloop_sig = [
        self._theta, self._state, self._inputs, self._state, self._extras,
        self._theta, self._state, self._inputs, self._state
    ]

    @function.Defun(dtypes.int32, dev_t_type, *_Dtypes(bakloop_sig))
    def BackwardLoopBody(*args):
      """Backward loop body function."""
      t, dev_t = args[0], args[1]
      (theta, orig_state0, inputs, acc_state, acc_extras, d_theta, d_state1,
       d_inputs, d_acc_state) = _Pack(args[2:], bakloop_sig)

      # The input recurrent state for time step t is previous time step's
      # output, or the original state0 when on time step 0.
      state_from_acc = _Index(acc_state, math_ops.maximum(0, t - 1))
      state0 = functional_ops.If(
          math_ops.equal(t, array_ops.constant(0, dtypes.int32)),
          _Flatten([state_from_acc, orig_state0]), ReturnOrigState0,
          ReturnAccState)
      state0 = nest.pack_sequence_as(orig_state0, state0)

      # The external inputs for time step t.
      inputs_t = _Index(inputs, t)
      # The extras for time step t.
      extras_t = _Index(acc_extras, t)

      d_state1 = _Add(_Index(d_acc_state, t), d_state1)
      (d_theta_t, d_state0, d_inputs_t) = _Pack(
          Bak(*_Flatten([theta, state0, inputs_t, extras_t, d_state1])),
          [self._theta, self._state, self._inputs])
      d_theta = _Add(d_theta, d_theta_t)
      d_inputs = _Update(d_inputs, d_inputs_t, dev_t)
      return [math_ops.subtract(dev_t, 1)] + _Flatten([
          theta, orig_state0, inputs, acc_state, acc_extras, d_theta, d_state0,
          d_inputs, d_acc_state
      ])

    # Backward calls BackwardLoopBody n times.  Each time computes the backprop
    # for one time step of the recurrent net.
    backward_sig = [
        self._theta, self._state, self._inputs, self._max_input_length,
        self._state, self._extras, self._state, self._state
    ]

    @function.Defun(*_Dtypes(backward_sig), noinline=noinline)
    def Backward(*args):
      """Backward pass for the recurrent net."""
      # theta, state0, inputs are Forward's inputs.
      # acc_state is the accumulated 1st output of Forward.
      # acc_extras is the accumulated 2nd output of Forward.
      # d_acc_state is the gradient for acc_state.
      # d_state1 is the gradient for the final state computed by Forward.
      (theta, state0, inputs, max_input_length, acc_state, acc_extras,
       d_acc_state, d_state1) = _Pack(args, backward_sig)

      # Accumulators for gradients.
      d_theta = _EmptyLike(theta)
      d_inputs = _EmptyLike(inputs)

      slen_dim = _SeqLenDim(inputs)

      # Loop backwards. Note the loop's limit is open-ended, so goes through
      # t=0.
      t = slen_dim - 1 if self._aligned_end else max_input_length - 1
      dev_t = math_ops.cast(t, dtypes.int32) if use_tpu else math_ops.cast(
          t, dtypes.int64)
      limit = slen_dim - max_input_length - 1 if self._aligned_end else -1
      run = functional_ops.For(
          start=t,
          limit=limit,
          delta=-1,
          inputs=[dev_t] + _Flatten([
              theta, state0, inputs, acc_state, acc_extras, d_theta, d_state1,
              d_inputs, d_acc_state
          ]),
          body=BackwardLoopBody,
          rewrite_with_while=compiled)

      (theta, state0, inputs, acc_state, acc_extras, d_theta, d_state0,
       d_inputs, d_acc_state) = _Pack(run[1:], bakloop_sig)

      d_max_input_length = array_ops.constant(0, dtype=max_input_length.dtype)
      return _Flatten(
          [d_theta, d_state0, d_inputs, d_max_input_length, acc_extras])

    self._forward = Forward

  def _MaybeComputeMaxInputLength(self, inputs, max_input_length):
    if max_input_length is not None:
      return max_input_length
    return math_ops.reduce_max(array_ops.shape(nest.flatten(inputs)[0])[0])

  def Compute(self):
    return _Pack(
        self._forward(*_Flatten([
            self._theta, self._state, self._inputs, self._max_input_length,
            self._extras
        ])), [self._state, self._state, self._extras])[:2]


def _GetCellGrad(cell_fn, cell_grad):
  """Returns the gradient function for cell_fn.

  Args:
    cell_fn: The recurrent neural net's cell function.
    cell_grad: If not None, cell_fn's gradient function.

  Returns:
    Returns cell_grad if not None. Otherwise, assume cell_fn is a python
    function representing the recurrent neural net's cell function, i.e.,
      cell_fn: (theta, state0, inputs) -> (state1, extra)
    returns its default gradient python function, i.e.,
      cell_grad: (theta, state0, inputs, extras, dstate1) -> (
                  dtheta, dstate0, dinputs)
  """

  if cell_grad:
    return cell_grad

  def CellGrad(theta, state0, inputs, extras, dstate1):
    """Default gradient function for cell_fn."""
    # NOTE: The default grad function recomputes the forward
    # function and does not take advantage of 'extras' returned by
    # the forward function.
    del extras
    state1, extras = cell_fn(theta, state0, inputs)
    ys = _Flatten([state1])
    xs = _Flatten([theta, state0, inputs])
    grad_ys = _Flatten([dstate1])
    grads = gradients_impl.gradients(ys=ys, xs=xs, grad_ys=grad_ys)
    return _ConvertNoneGradientToZeros([theta, state0, inputs],
                                       _Pack(grads, [theta, state0, inputs]))

  return CellGrad


def _IsSingleTimeStep(inputs, max_input_length):
  """Returns True only if the time dimension of inputs is 1."""
  if not isinstance(max_input_length, ops.Tensor):
    return max_input_length == 1
  for x in nest.flatten(inputs):
    if x.shape.dims is None or x.shape[0].value != 1:
      return False
  return True


def Recurrent(theta,
              state0,
              inputs,
              cell_fn,
              cell_grad=None,
              extras=None,
              max_input_length=None,
              use_tpu=False,
              aligned_end=False):
  """Compute a recurrent neural net.

  Roughly, Recurrent() computes the following:
    state = state0
    for t in inputs' sequence length:
      state = cell_fn(theta, state, inputs[t, :])
      accumulate_state[t, :] = state
    return accumulate_state, state

  theta, state, inputs are all structures of tensors.

  inputs[t, :] means taking a slice out from every tensor in the inputs.

  accumulate_state[t, :] = state means that we stash every tensor in
  'state' into a slice of the corresponding tensor in
  accumulate_state.

  cell_fn is a python callable computing (building up a TensorFlow
  graph) the recurrent neural network's one forward step. Two calls of
  cell_fn must describe two identical computations.

  By construction, Recurrent()'s backward computation does not access
  any intermediate values computed by cell_fn during forward
  computation. We may extend Recurrent() to support that by taking a
  customized backward function of cell_fn.

  Args:
    theta: weights. A structure of tensors.
    state0: initial state. A structure of tensors.
    inputs: inputs. A structure of tensors.
    cell_fn: A python function, which computes:
      state1, extras = cell_fn(theta, state0, inputs[t, :])
    cell_grad: A python function which computes:
      dtheta, dstate0, dinputs[t, :] = cell_grad(
        theta, state0, inputs[t, :], extras, dstate1)
    extras: A structure of tensors. The 2nd return value of every
      invocation of cell_fn is a structure of tensors with matching keys
      and shapes of  this `extras`.
    max_input_length: maximum length of effective input. This is used to
      truncate the computation if the inputs have been allocated to a
      larger size. A scalar tensor.
    use_tpu: whether or not we are on TPU.
    aligned_end: A boolean indicating whether the sequence is aligned at
      the end.

  Returns:
    accumulate_state and the final state.
  """
  if cell_grad is None and _IsSingleTimeStep(inputs, max_input_length):
    # The seqlen length is staticly known as 1. Hence, we just need to
    # call cell_fn once without putting it into a loop.
    inputs = nest.map_structure(lambda x: array_ops.squeeze(x, axis=0), inputs)
    state1, _ = cell_fn(theta, state0, inputs)
    acc_state = nest.map_structure(lambda x: array_ops.expand_dims(x, axis=0),
                                   state1)
    return acc_state, state1

  # If cell_grad is not given, derives the gradient function from
  # cell_fn.
  cell_grad = _GetCellGrad(cell_fn, cell_grad)

  if extras is None:
    # Derives 'extras' so that we can allocate extras' accumulator.
    _, extras = cell_fn(theta, state0, _Index(inputs, 0))
    extras = nest.map_structure(array_ops.zeros_like, extras)
  else:
    _, actual = cell_fn(theta, state0, _Index(inputs, 0))
    _AssertIsCompatible(extras, actual)

  return _Recurrent(
      cell_fn=cell_fn,
      cell_grad=cell_grad,
      theta=theta,
      state0=state0,
      inputs=inputs,
      max_input_length=max_input_length,
      extras=extras,
      use_tpu=use_tpu,
      aligned_end=aligned_end).Compute()
