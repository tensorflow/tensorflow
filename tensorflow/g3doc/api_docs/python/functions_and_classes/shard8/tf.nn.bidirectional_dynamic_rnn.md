### `tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, sequence_length=None, initial_state_fw=None, initial_state_bw=None, dtype=None, parallel_iterations=None, swap_memory=False, time_major=False, scope=None)` {#bidirectional_dynamic_rnn}

Creates a dynamic version of bidirectional recurrent neural network.

Similar to the unidirectional case above (rnn) but takes input and builds
independent forward and backward RNNs. The input_size of forward and
backward cell must match. The initial state for both directions is zero by
default (but can be set optionally) and no intermediate states are ever
returned -- the network is fully unrolled for the given (passed in)
length(s) of the sequence(s) or completely unrolled if length(s) is not
given.

##### Args:


*  <b>`cell_fw`</b>: An instance of RNNCell, to be used for forward direction.
*  <b>`cell_bw`</b>: An instance of RNNCell, to be used for backward direction.
*  <b>`inputs`</b>: The RNN inputs.
    If time_major == False (default), this must be a tensor of shape:
      `[batch_size, max_time, input_size]`.
    If time_major == True, this must be a tensor of shape:
      `[max_time, batch_size, input_size]`.
    [batch_size, input_size].
*  <b>`sequence_length`</b>: An int32/int64 vector, size `[batch_size]`,
    containing the actual lengths for each of the sequences.
*  <b>`initial_state_fw`</b>: (optional) An initial state for the forward RNN.
    This must be a tensor of appropriate type and shape
    `[batch_size, cell_fw.state_size]`.
    If `cell_fw.state_size` is a tuple, this should be a tuple of
    tensors having shapes `[batch_size, s] for s in cell_fw.state_size`.
*  <b>`initial_state_bw`</b>: (optional) Same as for `initial_state_fw`, but using
    the corresponding properties of `cell_bw`.
*  <b>`dtype`</b>: (optional) The data type for the initial states and expected output.
    Required if initial_states are not provided or RNN states have a
    heterogeneous dtype.
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
*  <b>`dtype`</b>: (optional) The data type for the initial state.  Required if
    either of the initial states are not provided.
*  <b>`scope`</b>: VariableScope for the created subgraph; defaults to
    "bidirectional_rnn"

##### Returns:

  A tuple (outputs, output_states) where:

*  <b>`outputs`</b>: A tuple (output_fw, output_bw) containing the forward and
      the backward rnn output `Tensor`.
      If time_major == False (default),
        output_fw will be a `Tensor` shaped:
        `[batch_size, max_time, cell_fw.output_size]`
        and output_bw will be a `Tensor` shaped:
        `[batch_size, max_time, cell_bw.output_size]`.
      If time_major == True,
        output_fw will be a `Tensor` shaped:
        `[max_time, batch_size, cell_fw.output_size]`
        and output_bw will be a `Tensor` shaped:
        `[max_time, batch_size, cell_bw.output_size]`.
      It returns a tuple instead of a single concatenated `Tensor`, unlike
      in the `bidirectional_rnn`. If the concatenated one is preferred,
      the forward and backward outputs can be concatenated as
      `tf.concat_v2(outputs, 2)`.
*  <b>`output_states`</b>: A tuple (output_state_fw, output_state_bw) containing
      the forward and the backward final states of bidirectional rnn.

##### Raises:


*  <b>`TypeError`</b>: If `cell_fw` or `cell_bw` is not an instance of `RNNCell`.

