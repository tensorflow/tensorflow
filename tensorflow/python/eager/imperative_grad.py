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

from tensorflow.python.eager import tape as tape_module


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
def _prepare_backprop(vspace, target, tensor_to_op, op_to_entry, id_sources):
  """Filters the tape to only include relevant entries and counts tensor usages.

  Args:
    vspace: information about the space we're differentiating in.
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
  tensor_stack = [vspace.tensor_id(x) for x in target]
  tensor_usage_counts = {}
  o_to_e = {}  # Copy of just the bits we need from op_to_entry
  while tensor_stack:
    t = tensor_stack.pop()
    op = tensor_to_op.get(t, None)
    # op is None or -1 if the tensor is a source (i.e. was watched directly)
    if op is None or op == -1 or op in o_to_e:
      continue
    op_trace = tape_module.TapeEntry(*op_to_entry[op])
    o_to_e[op] = op_trace
    for it in op_trace.input_ids:
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


def _initial_gradients(vspace, target, output_gradients, tensor_usage_counts):
  """Computes the initial gradients for each Tensor."""
  # Initialize the backprop stack
  gradients = collections.defaultdict(list)
  for i, t in enumerate(target):
    if vspace.tensor_id(t) in tensor_usage_counts:
      # Can't provide a gradient of something we're trying to differentiate
      assert output_gradients is None or output_gradients[i] is None
    else:
      if output_gradients is None or output_gradients[i] is None:
        out_grad = vspace.ones_like(t)
      else:
        out_grad = output_gradients[i]
      gradients[vspace.tensor_id(t)].append(out_grad)
  return gradients


VSpace = collections.namedtuple(
    "VSpace",
    ["aggregate_fn", "num_elements_fn", "tensor_id", "zeros", "ones_like"])


# If over MIN_AGGREGATE_COUNT gradients are accumulated and the total
# memory consumption is over MIN_AGGREGATE_BYTES, do an early aggregation
# so as to release the gradient tensor to save memory.
_MIN_AGGREGATE_COUNT = 4
_MIN_AGGREGATE_BYTES = 128 * 1024 * 1024


def imperative_grad(
    vspace,
    tape,
    target,
    sources,
    output_gradients=None):
  """Computes gradients from the imperatively defined tape on top of the stack.

  Works by filtering the tape, computing how many downstream usages are of each
  tensor and entry, and repeatedly applying backward functions until we have
  gradients for all sources.

  Args:
   vspace: the vector space in which to differentiate.
   tape: the gradient tape which stores the trace.
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
  tensor_to_op, op_to_entry = tape.export()
  # This overwrites the op_to_entry variable, which will release all memory used
  # to keep traces that are irrelevant to the gradient computation we're doing
  # here.
  id_sources = [vspace.tensor_id(t) for t in sources]
  tensor_usage_counts, op_to_entry, op_missing_tensor = _prepare_backprop(
      vspace, target, tensor_to_op, op_to_entry, id_sources)
  ready_ops = _initialize_backprop_stack(op_to_entry, op_missing_tensor)
  gradients = _initial_gradients(vspace, target, output_gradients,
                                 tensor_usage_counts)
  gradients_size = dict()
  # Now exhaust the backprop stack
  while ready_ops:
    op = ready_ops.pop()
    op_trace = op_to_entry.pop(op)
    out_gradients = [gradients.pop(t, None) for t in op_trace.output_ids]

    # Cache the last used zero tensor. We reuse it if the next one
    # we need is of the same shape and dtype. This is very helpful in
    # large splits and should have negligible overhead in other cases.
    last_shape_and_dtype = None
    last_zeros = None
    for i in range(len(out_gradients)):
      if out_gradients[i] is None:
        # TODO(apassos) this should be in the right device
        none_indices = _grad_fn_accepts_none_for_indices.get(
            op_trace.op_type, None)
        if none_indices is None or i not in none_indices:
          shape_and_dtype = op_trace.output_shape_and_dtype[i]
          if shape_and_dtype != last_shape_and_dtype:
            last_shape_and_dtype = shape_and_dtype
            last_zeros = vspace.zeros(*shape_and_dtype)
          out_gradients[i] = last_zeros
      else:
        out_gradients[i] = vspace.aggregate_fn(out_gradients[i])

    in_gradients = op_trace.backward_function(*(out_gradients))
    for i, t in enumerate(op_trace.input_ids):
      if in_gradients[i] is not None:
        t_grads = gradients.setdefault(t, [])
        t_grads.append(in_gradients[i])
        if len(t_grads) >= _MIN_AGGREGATE_COUNT:
          if t not in gradients_size:
            gradients_size[t] = vspace.num_elements_fn(t_grads[-1])
          size = gradients_size[t]

          if len(t_grads) * size * 4 > _MIN_AGGREGATE_BYTES:
            t_grads[:] = [vspace.aggregate_fn(t_grads)]
      if tensor_usage_counts.get(t, 0) > 0:
        tensor_usage_counts[t] -= 1
        if (t in tensor_to_op
            and tensor_usage_counts[t] == 0
            and t not in id_sources):
          in_op = tensor_to_op[t]
          if in_op is None or in_op == -1:
            continue
          if op_missing_tensor.get(in_op, 0) > 0:
            op_missing_tensor[in_op] -= 1
            if op_missing_tensor.get(in_op, 0) == 0:
              ready_ops.append(in_op)
  result = []
  for i, s in enumerate(sources):
    g = gradients.get(vspace.tensor_id(s), None)
    if g is None:
      result.append(None)
    else:
      result.append(vspace.aggregate_fn(g))
  return result


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
