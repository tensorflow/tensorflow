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
"""Jacobian ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import gradients as gradient_ops
from tensorflow.python.ops.parallel_for import control_flow_ops
from tensorflow.python.util import nest


def jacobian(output, inputs, use_pfor=True, parallel_iterations=None):
  """Computes jacobian of `output` w.r.t. `inputs`.

  Args:
    output: A tensor.
    inputs: A tensor or a nested structure of tensor objects.
    use_pfor: If true, uses pfor for computing the jacobian. Else uses
      tf.while_loop.
    parallel_iterations: A knob to control how many iterations and dispatched in
      parallel. This knob can be used to control the total memory usage.

  Returns:
    A tensor or a nested structure of tensors with the same structure as
    `inputs`. Each entry is the jacobian of `output` w.r.t. to the corresponding
    value in `inputs`. If output has shape [y_1, ..., y_n] and inputs_i has
    shape [x_1, ..., x_m], the corresponding jacobian has shape
    [y_1, ..., y_n, x_1, ..., x_m]. Note that in cases where the gradient is
    sparse (IndexedSlices), jacobian function currently makes it dense and
    returns a Tensor instead. This may change in the future.
  """
  flat_inputs = nest.flatten(inputs)
  output_tensor_shape = output.shape
  output_shape = array_ops.shape(output)
  output = array_ops.reshape(output, [-1])

  def loop_fn(i):
    y = array_ops.gather(output, i)
    return gradient_ops.gradients(y, flat_inputs)

  try:
    output_size = int(output.shape[0])
  except TypeError:
    output_size = array_ops.shape(output)[0]

  if use_pfor:
    pfor_outputs = control_flow_ops.pfor(
        loop_fn, output_size, parallel_iterations=parallel_iterations)
  else:
    pfor_outputs = control_flow_ops.for_loop(
        loop_fn,
        [output.dtype] * len(flat_inputs),
        output_size,
        parallel_iterations=parallel_iterations)

  for i, out in enumerate(pfor_outputs):
    if isinstance(out, ops.Tensor):
      new_shape = array_ops.concat(
          [output_shape, array_ops.shape(out)[1:]], axis=0)
      out = array_ops.reshape(out, new_shape)
      out.set_shape(output_tensor_shape.concatenate(flat_inputs[i].shape))
      pfor_outputs[i] = out

  return nest.pack_sequence_as(inputs, pfor_outputs)


def batch_jacobian(output, inp, use_pfor=True, parallel_iterations=None):
  """Computes and stacks jacobians of `output[i,...]` w.r.t. `input[i,...]`.

  e.g.
  x = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
  y = x * x
  jacobian = batch_jacobian(y, x)
  # => [[[2,  0], [0,  4]], [[6,  0], [0,  8]]]

  Args:
    output: A tensor with shape [b, y1, ..., y_n]. `output[i,...]` should
      only depend on `inp[i,...]`.
    inp: A tensor with shape [b, x1, ..., x_m]
    use_pfor: If true, uses pfor for computing the Jacobian. Else uses a
      tf.while_loop.
    parallel_iterations: A knob to control how many iterations are vectorized
      and dispatched in parallel. The default value of None, when use_pfor is
      true, corresponds to vectorizing all the iterations. When use_pfor is
      false, the default value of None corresponds to parallel_iterations=10.
      This knob can be used to control the total memory usage.

  Returns:
    A tensor `t` with shape [b, y_1, ..., y_n, x1, ..., x_m] where `t[i, ...]`
    is the jacobian of `output[i, ...]` w.r.t. `inp[i, ...]`, i.e. stacked
    per-example jacobians.

  Raises:
    ValueError: if first dimension of `output` and `inp` do not match.
  """
  output_shape = output.shape
  if not output_shape[0].is_compatible_with(inp.shape[0]):
    raise ValueError("Need first dimension of output shape (%s) and inp shape "
                     "(%s) to match." % (output.shape, inp.shape))
  if output_shape.is_fully_defined():
    batch_size = int(output_shape[0])
    output_row_size = output_shape.num_elements() // batch_size
  else:
    output_shape = array_ops.shape(output)
    batch_size = output_shape[0]
    output_row_size = array_ops.size(output) // batch_size
  inp_shape = array_ops.shape(inp)
  # Flatten output to 2-D.
  with ops.control_dependencies(
      [check_ops.assert_equal(batch_size, inp_shape[0])]):
    output = array_ops.reshape(output, [batch_size, output_row_size])

  def loop_fn(i):
    y = array_ops.gather(output, i, axis=1)
    return gradient_ops.gradients(y, inp)[0]

  if use_pfor:
    pfor_output = control_flow_ops.pfor(loop_fn, output_row_size,
                                        parallel_iterations=parallel_iterations)
  else:
    pfor_output = control_flow_ops.for_loop(
        loop_fn, output.dtype,
        output_row_size,
        parallel_iterations=parallel_iterations)
  if pfor_output is None:
    return None
  pfor_output = array_ops.reshape(pfor_output,
                                  [output_row_size, batch_size, -1])
  output = array_ops.transpose(pfor_output, [1, 0, 2])
  new_shape = array_ops.concat([output_shape, inp_shape[1:]], axis=0)
  return array_ops.reshape(output, new_shape)
