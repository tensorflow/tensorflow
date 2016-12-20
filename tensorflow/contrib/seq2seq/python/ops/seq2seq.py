# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""Seq2seq layer operations for use in neural networks.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import layers
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs

__all__ = ["dynamic_rnn_decoder"]

def dynamic_rnn_decoder(cell, decoder_fn, inputs=None, sequence_length=None,
                        parallel_iterations=None, swap_memory=False,
                        time_major=False, scope=None, name=None):
  """ Dynamic RNN decoder for a sequence-to-sequence model specified by
  RNNCell and decoder function.

  The `dynamic_rnn_decoder` is similar to the `tf.python.ops.rnn.dynamic_rnn`
  as the decoder does not make any assumptions of sequence length and batch
  size of the input.

  The `dynamic_rnn_decoder` has two modes: training or inference and expects
  the user to create seperate functions for each.

  Under both training and inference `cell` and `decoder_fn` is expected. Where
  the `cell` performs computation at every timestep using the `raw_rnn` and
  the `decoder_fn` allows modelling of early stopping, output, state, and next
  input and context.

  When training the user is expected to supply `inputs`. At every time step a
  slice of the supplied input is fed to the `decoder_fn`, which modifies and
  returns the input for the next time step.

  `sequence_length` is needed at training time, i.e., when `inputs` is not
  None, for dynamic unrolling. At test time, when `inputs` is None,
  `sequence_length` is not needed.

  Under inference `inputs` is expected to be `None` and the input is inferred
  solely from the `decoder_fn`.

  Args:
    cell: An instance of RNNCell.
    decoder_fn: A function that takes time, cell state, cell input,
      cell output and context state. It returns a early stopping vector,
      cell state, next input, cell output and context state.
      Examples of decoder_fn can be found in the decoder_fn.py folder.
    inputs: The inputs for decoding (embedded format).

      If `time_major == False` (default), this must be a `Tensor` of shape:
        `[batch_size, max_time, ...]`.

      If `time_major == True`, this must be a `Tensor` of shape:
        `[max_time, batch_size, ...]`.

      The input to `cell` at each time step will be a `Tensor` with dimensions
        `[batch_size, ...]`.
    sequence_length: (optional) An int32/int64 vector sized `[batch_size]`.
      if `inputs` is not None and `sequence_length` is None it is inferred
      from the `inputs` as the maximal possible sequence length.
    parallel_iterations: (Default: 32).  The number of iterations to run in
      parallel.  Those operations which do not have any temporal dependency
      and can be run in parallel, will be.  This parameter trades off
      time for space.  Values >> 1 use more memory but take less time,
      while smaller values use less memory but computations take longer.
    swap_memory: Transparently swap the tensors produced in forward inference
      but needed for back prop from GPU to CPU.  This allows training RNNs
      which would typically not fit on a single GPU, with very minimal (or no)
      performance penalty.
    time_major: The shape format of the `inputs` and `outputs` Tensors.
      If true, these `Tensors` must be shaped `[max_time, batch_size, depth]`.
      If false, these `Tensors` must be shaped `[batch_size, max_time, depth]`.
      Using `time_major = True` is a bit more efficient because it avoids
      transposes at the beginning and end of the RNN calculation.  However,
      most TensorFlow data is batch-major, so by default this function
      accepts input and emits output in batch-major form.
    scope: VariableScope for the `raw_rnn`;
      defaults to None.
    name: NameScope for the decoder;
      defaults to "dynamic_rnn_decoder"

  Returns:
    A pair (outputs, state) where:

      outputs: the RNN output 'Tensor'.

        If time_major == False (default), this will be a `Tensor` shaped:
          `[batch_size, max_time, cell.output_size]`.

        If time_major == True, this will be a `Tensor` shaped:
          `[max_time, batch_size, cell.output_size]`.

      state: The final state and will be shaped
             `[batch_size, cell.state_size]`.

  Raises:
    ValueError: if inputs is not None and has less than three dimensions.
  """
  with ops.name_scope(name, "dynamic_rnn_decoder",
                      [cell, decoder_fn, inputs, sequence_length,
                       parallel_iterations, swap_memory, time_major, scope]):
    if inputs is not None:
      # Convert to tensor
      inputs = ops.convert_to_tensor(inputs)

      # Test input dimensions
      if inputs.get_shape().ndims is not None and (
          inputs.get_shape().ndims < 2):
        raise ValueError("Inputs must have at least two dimensions")
      # Setup of RNN (dimensions, sizes, length, initial state, dtype)
      if not time_major:
        # [batch, seq, features] -> [seq, batch, features]
        inputs = array_ops.transpose(inputs, perm=[1, 0, 2])

      dtype = inputs.dtype
      # Get data input information
      input_depth = int(inputs.get_shape()[2])
      batch_depth = inputs.get_shape()[1].value
      max_time = inputs.get_shape()[0].value
      if max_time is None:
        max_time = array_ops.shape(inputs)[0]
      # Setup decoder inputs as TensorArray
      inputs_ta = tensor_array_ops.TensorArray(dtype, size=max_time)
      inputs_ta = inputs_ta.unpack(inputs)

    def loop_fn(time, cell_output, cell_state, loop_state):
      if cell_state is None:  # first call, before while loop (in raw_rnn)
        if cell_output is not None:
          raise ValueError("Expected cell_output to be None when cell_state "
                           "is None, but saw: %s" % cell_output)
        if loop_state is not None:
          raise ValueError("Expected loop_state to be None when cell_state "
                           "is None, but saw: %s" % loop_state)
        context_state = None
      else:  # subsequent calls, inside while loop, after cell excution
        if isinstance(loop_state, tuple):
          (done, context_state) = loop_state
        else:
          done = loop_state
          context_state = None

      # call decoder function
      if inputs is not None:  # training
        # get next_cell_input
        if cell_state is None:
          next_cell_input = inputs_ta.read(0)
        else:
          if batch_depth is not None:
            batch_size = batch_depth
          else:
            batch_size = array_ops.shape(done)[0]
          next_cell_input = control_flow_ops.cond(
              math_ops.equal(time, max_time),
              lambda: array_ops.zeros([batch_size, input_depth], dtype=dtype),
              lambda: inputs_ta.read(time))
        (next_done, next_cell_state, next_cell_input, emit_output,
         next_context_state) = decoder_fn(time, cell_state, next_cell_input,
                                          cell_output, context_state)
      else:  # inference
        # next_cell_input is obtained through decoder_fn
        (next_done, next_cell_state, next_cell_input, emit_output,
         next_context_state) = decoder_fn(time, cell_state, None, cell_output,
                                          context_state)

      # check if we are done
      if next_done is None:  # training
        next_done = time >= sequence_length

      # build next_loop_state
      if next_context_state is None:
        next_loop_state = next_done
      else:
        next_loop_state = (next_done, next_context_state)

      return (next_done, next_cell_input, next_cell_state,
              emit_output, next_loop_state)

    # Run raw_rnn function
    outputs_ta, state, _ = rnn.raw_rnn(
        cell, loop_fn, parallel_iterations=parallel_iterations,
        swap_memory=swap_memory, scope=scope)
    outputs = outputs_ta.pack()

    if not time_major:
      # [seq, batch, features] -> [batch, seq, features]
      outputs = array_ops.transpose(outputs, perm=[1, 0, 2])
    return outputs, state
