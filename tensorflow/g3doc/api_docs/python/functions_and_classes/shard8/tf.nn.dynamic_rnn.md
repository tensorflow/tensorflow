### `tf.nn.dynamic_rnn(cell, inputs, sequence_length=None, initial_state=None, dtype=None, parallel_iterations=None, swap_memory=False, time_major=False, scope=None)` {#dynamic_rnn}

Creates a recurrent neural network specified by RNNCell `cell`.

This function is functionally identical to the function `rnn` above, but
performs fully dynamic unrolling of `inputs`.

Unlike `rnn`, the input `inputs` is not a Python list of `Tensors`.  Instead,
it is a single `Tensor` where the maximum time is either the first or second
dimension (see the parameter `time_major`).  The corresponding output is
a single `Tensor` having the same number of time steps and batch size.

The parameter `sequence_length` is required and dynamic calculation is
automatically performed.

##### Args:


*  <b>`cell`</b>: An instance of RNNCell.
*  <b>`inputs`</b>: The RNN inputs.
    If time_major == False (default), this must be a tensor of shape:
      `[batch_size, max_time, input_size]`.
    If time_major == True, this must be a tensor of shape:
      `[max_time, batch_size, input_size]`.
*  <b>`sequence_length`</b>: (optional) An int32/int64 vector sized `[batch_size]`.
*  <b>`initial_state`</b>: (optional) An initial state for the RNN.
    If `cell.state_size` is an integer, this must be
    a tensor of appropriate type and shape `[batch_size x cell.state_size]`.
    If `cell.state_size` is a tuple, this should be a tuple of
    tensors having shapes `[batch_size, s] for s in cell.state_size`.
*  <b>`dtype`</b>: (optional) The data type for the initial state.  Required if
    initial_state is not provided.
*  <b>`parallel_iterations`</b>: (Default: 32).  The number of iterations to run in
    parallel.  Those operations which do not have any temporal dependency
    and can be run in parallel, will be.  This parameter trades off
    time for space.  Values >> 1 use more memory but take less time,
    while smaller values use less memory but computations take longer.
*  <b>`swap_memory`</b>: Transparently swap the tensors produced in forward inference
    but needed for back prop from GPU to CPU.  This allows training RNNs
    which would typically not fit on a single GPU, with very minimal (or no)
    performance penalty.
*  <b>`time_major`</b>: The shape format of the `inputs` and `outputs` Tensors.
    If true, these `Tensors` must be shaped `[max_time, batch_size, depth]`.
    If false, these `Tensors` must be shaped `[batch_size, max_time, depth]`.
    Using `time_major = True` is a bit more efficient because it avoids
    transposes at the beginning and end of the RNN calculation.  However,
    most TensorFlow data is batch-major, so by default this function
    accepts input and emits output in batch-major form.
*  <b>`scope`</b>: VariableScope for the created subgraph; defaults to "RNN".

##### Returns:

  A pair (outputs, state) where:

*  <b>`outputs`</b>: The RNN output `Tensor`.
      If time_major == False (default), this will be a `Tensor` shaped:
        `[batch_size, max_time, cell.output_size]`.
      If time_major == True, this will be a `Tensor` shaped:
        `[max_time, batch_size, cell.output_size]`.
*  <b>`state`</b>: The final state.  If `cell.state_size` is a `Tensor`, this
      will be shaped `[batch_size, cell.state_size]`.  If it is a tuple,
      this be a tuple with shapes `[batch_size, s] for s in cell.state_size`.

##### Raises:


*  <b>`TypeError`</b>: If `cell` is not an instance of RNNCell.
*  <b>`ValueError`</b>: If inputs is None or an empty list.

