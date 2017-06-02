'''Beam search decoder from https://gist.github.com/nikitakit/6ab61a73b86c50ad88d409bac3c3d09f
But rewrote some core logic to fix incorrect output.

Beam decoder for tensorflow

Sample usage:

```
beam_decoder = BeamDecoder(NUM_CLASSES, beam_size=10, max_len=MAX_LEN)

_, final_state = tf.nn.seq2seq.rnn_decoder(
                        [beam_decoder.wrap_input(initial_input)] + [None] * (MAX_LEN - 1),
                        beam_decoder.wrap_state(initial_state),
                        beam_decoder.wrap_cell(my_cell),
                        loop_function = lambda prev_symbol, i: tf.nn.embedding_lookup(my_embedding, prev_symbol)
                    )
best_dense = beam_decoder.unwrap_output_dense(final_state) # Dense tensor output, right-aligned
best_sparse = beam_decoder.unwrap_output_sparse(final_state) # Output, this time as a sparse tensor
```
'''

import tensorflow as tf

try:
    from tensorflow.python.util import nest
except ImportError:
    # Backwards-compatibility
    from tensorflow.python.ops import rnn_cell
    class NestModule(object): pass
    nest = NestModule()
    nest.is_sequence = rnn_cell._is_sequence
    nest.flatten = rnn_cell._unpacked_state
    nest.pack_sequence_as = rnn_cell._packed_state

def nest_map(func, nested):
    if not nest.is_sequence(nested):
        return func(nested)
    flat = nest.flatten(nested)
    return nest.pack_sequence_as(nested, list(map(func, flat)))

class BeamDecoder(object):
    def __init__(self, num_classes, stop_token=0, beam_size=7, max_len=20):
        """
        num_classes: int. Number of output classes used
        stop_token: int.
        beam_size: int.
        max-len: int or scalar Tensor. If this cell is called recurrently more
            than max_len times in a row, the outputs will not be valid!
        """
        self.num_classes = num_classes
        self.stop_token = stop_token
        self.beam_size = beam_size
        self.max_len = max_len

    @classmethod
    def _tile_along_beam(cls, beam_size, state):
        if nest.is_sequence(state):
            return nest_map(
                lambda val: cls._tile_along_beam(beam_size, val),
                state
            )

        if not isinstance(state, tf.Tensor):
            raise ValueError("State should be a sequence or tensor")

        tensor = state

        tensor_shape = tensor.get_shape().with_rank_at_least(1)

        try:
            new_first_dim = tensor_shape[0] * beam_size
        except:
            new_first_dim = None

        dynamic_tensor_shape = tf.unpack(tf.shape(tensor))
        res = tf.expand_dims(tensor, 1)
        res = tf.tile(res, [1, beam_size] + [1] * (tensor_shape.ndims-1))
        res = tf.reshape(res, [-1] + list(dynamic_tensor_shape[1:]))
        res.set_shape([new_first_dim] + list(tensor_shape[1:]))
        return res

    def wrap_cell(self, cell):
        """
        Wraps a cell for use with the beam decoder
        """
        return BeamDecoderCellWrapper(cell, self.num_classes, self.max_len, self.stop_token, self.beam_size)

    def wrap_state(self, state):
        dummy = BeamDecoderCellWrapper(None, self.num_classes, self.max_len, self.stop_token, self.beam_size)
        if nest.is_sequence(state):
            batch_size = tf.shape(nest.flatten(state)[0])[0]
            dtype = nest.flatten(state)[0].dtype
        else:
            batch_size = tf.shape(state)[0]
            dtype = state.dtype
        print "wrap state", dtype
        return dummy._create_state(batch_size, dtype, cell_state=state)

    def wrap_input(self, input):
        """
        Wraps an input for use with the beam decoder.

        Should be used for the initial input at timestep zero, as well as any side-channel
        inputs that are per-batch (e.g. attention targets)
        """
        return self._tile_along_beam(self.beam_size, input)

    def unwrap_output_dense(self, final_state, include_stop_tokens=True):
        """
        Retreive the beam search output from the final state.

        Returns a [batch_size, max_len]-sized Tensor.
        """
        res = final_state[0]
        if include_stop_tokens:
            res = tf.concat(1, [res[:,1:], tf.ones_like(res[:,0:1]) * self.stop_token])
        return res

    def unwrap_output_sparse(self, final_state, include_stop_tokens=True):
        """
        Retreive the beam search output from the final state.

        Returns a sparse tensor with underlying dimensions of [batch_size, max_len]
        """
        output_dense = final_state[0]
        mask = tf.not_equal(output_dense, self.stop_token)

        if include_stop_tokens:
            output_dense = tf.concat(1, [output_dense[:,1:], tf.ones_like(output_dense[:,0:1]) * self.stop_token])
            mask = tf.concat(1, [mask[:,1:], tf.cast(tf.ones_like(mask[:,0:1], dtype=tf.int8), tf.bool)])

        return sparse_boolean_mask(output_dense, mask)

    def unwrap_output_logprobs(self, final_state):
        """
        Retreive the log-probabilities associated with the selected beams.
        """
        return final_state[1]

class BeamDecoderCellWrapper(tf.nn.rnn_cell.RNNCell):
    def __init__(self, cell, num_classes, max_len, stop_token=0, beam_size=7):
        # TODO: determine if we can have dynamic shapes instead of pre-filling up to max_len

        self.cell = cell
        self.num_classes = num_classes
        self.stop_token = stop_token
        self.beam_size = beam_size

        self.max_len = max_len

        # Note: masking out entries to -inf plays poorly with top_k, so just subtract out
        # a large number.
        # TODO: consider using slice+fill+concat instead of adding a mask
        # TODO: consider making the large negative constant dtype-dependent
        self._nondone_mask = tf.reshape(
            tf.cast(tf.equal(tf.range(self.num_classes), self.stop_token), tf.float32) * -1e18,
            [1, 1, self.num_classes]
        )
        self._nondone_mask = tf.reshape(tf.tile(self._nondone_mask, [1, self.beam_size, 1]),
            [-1, self.beam_size*self.num_classes])

    def __call__(self, inputs, state, scope=None):
        (
            past_cand_symbols, # [batch_size, max_len]
            past_cand_logprobs,# [batch_size]
            past_beam_symbols, # [batch_size*self.beam_size, max_len], right-aligned!!!
            past_beam_logprobs,# [batch_size*self.beam_size]
            past_cell_state,
                ) = state

        batch_size = tf.shape(past_cand_logprobs)[0] # TODO: get as int, if possible
        
        full_size = batch_size * self.beam_size

        cell_inputs = inputs
        cell_outputs, raw_cell_state = self.cell(cell_inputs, past_cell_state)

        logprobs = tf.nn.log_softmax(cell_outputs)

        logprobs_batched = tf.reshape(logprobs + tf.expand_dims(past_beam_logprobs, 1),
                                      [-1, self.beam_size * self.num_classes])
        logprobs_batched.set_shape((None, self.beam_size * self.num_classes))

        # prints and asserts
        tf.assert_less_equal(logprobs, 0.0)
        tf.assert_less_equal(past_beam_logprobs, 0.0)

        masked_logprobs = tf.reshape(logprobs_batched, [-1, self.beam_size * self.num_classes])
        # print masked_logprobs.get_shape()

        beam_logprobs, indices = tf.nn.top_k(
            masked_logprobs,
            self.beam_size
        )

        beam_logprobs = tf.reshape(beam_logprobs, [-1])

        # For continuing to the next symbols
        symbols = indices % self.num_classes # [batch_size, self.beam_size]
        parent_refs = tf.reshape(indices // self.num_classes, [-1]) # [batch_size*self.beam_size]

        # TODO: this technically doesn't need to be recalculated every loop
        parent_refs_offsets = tf.mul(tf.floordiv(tf.range(full_size), self.beam_size), self.beam_size)
        parent_refs = parent_refs + parent_refs_offsets

        if past_beam_symbols is not None:
            symbols_history = tf.gather(past_beam_symbols, parent_refs)
            beam_symbols = tf.concat(1, [tf.reshape(symbols, [-1, 1]), symbols_history])
        else:
            beam_symbols = tf.reshape(symbols, [-1, 1])

        # Above ends up outputting reversed. Below doesn't work though because tf doesn't support negative indexing.
        # last = past_beam_symbols.get_shape()[1]
        # symbols_history = tf.gather(past_beam_symbols[:,last - 1], parent_refs)
        # beam_symbols = tf.concat(1, [past_beam_symbols[:,:last-1], tf.reshape(symbols_history, [-1, 1]), tf.reshape(symbols, [-1, 1]), ])

        # Handle the output and the cell state shuffling
        outputs = tf.reshape(symbols, [-1]) # [batch_size*beam_size, 1]
        cell_state = nest_map(
            lambda element: tf.gather(element, parent_refs),
            raw_cell_state
        )

        # Handling for getting a done token
        # logprobs_done = tf.reshape(logprobs_batched, [-1, self.beam_size, self.num_classes])[:,:,self.stop_token]
        # done_parent_refs = tf.to_int32(tf.argmax(logprobs_done, 1))
        # done_parent_refs_offsets = tf.range(batch_size) * self.beam_size
        # done_symbols = tf.gather(past_beam_symbols, done_parent_refs + done_parent_refs_offsets)

        # logprobs_done_max = tf.reduce_max(logprobs_done, 1)
        # cand_symbols = tf.select(logprobs_done_max > past_cand_logprobs,
        #                         done_symbols,
        #                         past_cand_symbols)
        # cand_logprobs = tf.maximum(logprobs_done_max, past_cand_logprobs)
        cand_symbols = past_cand_symbols # current last symbol in the beam [batch_size*self.beam_size]
        cand_logprobs = past_cand_logprobs

        return outputs, (
            cand_symbols,
            cand_logprobs,
            beam_symbols,
            beam_logprobs,
            cell_state,
        )

    @property
    def state_size(self):
        return (self.max_len,
                1,
                self.max_len,
                1,
                self.cell.state_size
               )

    @property
    def output_size(self):
        return 1

    def _create_state(self, batch_size, dtype, cell_state=None):
        print "state", batch_size, self.max_len, dtype
        cand_logprobs = tf.ones((batch_size,), dtype=tf.float32) * -float('inf')

        if cell_state is None:
            cell_state = self.cell.zero_state(batch_size*self.beam_size, dtype=dtype)
        else:
            cell_state = BeamDecoder._tile_along_beam(self.beam_size, cell_state)

        full_size = batch_size * self.beam_size
        first_in_beam_mask = tf.equal(tf.range(full_size) % self.beam_size, 0)

        cand_symbols = tf.fill([full_size], tf.constant(self.stop_token, dtype=tf.int32))

        beam_symbols = None
        beam_logprobs = tf.select(
            first_in_beam_mask,
            tf.fill([full_size], 0.0),
            tf.fill([full_size], -1e18), # top_k does not play well with -inf
                                         # TODO: dtype-dependent value here
        )
        return (
            cand_symbols,
            cand_logprobs,
            beam_symbols,
            beam_logprobs,
            cell_state,
        )

    def zero_state(self, batch_size_times_beam_size, dtype):
        """
        Instead of calling this manually, please use
        BeamDecoder.wrap_state(cell.zero_state(...)) instead
        """
        batch_size = batch_size_times_beam_size / self.beam_size
        return self.create_zero_state(batch_size, dtype)

def sparse_boolean_mask(tensor, mask):
    """
    Creates a sparse tensor from masked elements of `tensor`

    Inputs:
      tensor: a 2-D tensor, [batch_size, T]
      mask: a 2-D mask, [batch_size, T]

    Output: a 2-D sparse tensor
    """
    mask_lens = tf.reduce_sum(tf.cast(mask, tf.int32), -1, keep_dims=True)
    mask_shape = tf.shape(mask)
    left_shifted_mask = tf.tile(
        tf.expand_dims(tf.range(mask_shape[1]), 0),
        [mask_shape[0], 1]
    ) < mask_lens
    return tf.SparseTensor(
        indices=tf.where(left_shifted_mask),
        values=tf.boolean_mask(tensor, mask),
        shape=tf.cast(tf.pack([mask_shape[0], tf.reduce_max(mask_lens)]), tf.int64) # For 2D only
    )
