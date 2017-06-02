import tensorflow as tf
import numpy as np
from beam_decoder import BeamDecoder

sess = tf.InteractiveSession()

class MarkovChainCell(tf.nn.rnn_cell.RNNCell):
    """
    This cell type is only used for testing the beam decoder.
    It represents a Markov chain characterized by a probability table p(x_t|x_{t-1},x_{t-2}).
    """
    def __init__(self, table):
        """
        table[a,b,c] = p(x_t=c|x_{t-1}=b,x_{t-2}=a)
        """
        assert len(table.shape) == 3 and table.shape[0] == table.shape[1] == table.shape[2]
        self.log_table = tf.log(np.asarray(table, dtype=np.float32))
        self._output_size = table.shape[0]

    def __call__(self, inputs, state, scope=None):
        """
        inputs: [batch_size, 1] int tensor
        state: [batch_size, 1] int tensor
        """
        logits = tf.reshape(self.log_table, [-1, self.output_size])
        indices = state[0] * self.output_size + inputs
        return tf.gather(logits, tf.reshape(indices, [-1])), (inputs,)

    @property
    def state_size(self):
        return (1,)

    @property
    def output_size(self):
        return self._output_size

table = np.array([[[0.9, 0.1, 0],
                   [0, 0.9, 0.1],
                   [0, 0, 1.0]]] * 3)
cell = MarkovChainCell(table)
initial_state = cell.zero_state(1, tf.int32)
initial_input = initial_state[0]

MAX_LEN = 3
beam_decoder = BeamDecoder(num_classes=3, stop_token=2, beam_size=10, max_len=MAX_LEN)

outputs, final_state = tf.nn.seq2seq.rnn_decoder(
                        [beam_decoder.wrap_input(initial_input)] + [None] * (MAX_LEN-1),
                        beam_decoder.wrap_state(initial_state),
                        beam_decoder.wrap_cell(cell),
                        loop_function = lambda prev_symbol, i: tf.reshape(prev_symbol, [-1, 1])
                    )

# best_dense = beam_decoder.unwrap_output_dense(final_state)
# best_sparse = beam_decoder.unwrap_output_sparse(final_state)
# best_logprobs = beam_decoder.unwrap_output_logprobs(final_state)

# print best_dense.eval()
# print best_logprobs.eval()

beams = final_state[2].eval()
probs = final_state[3].eval()

for i in range(len(beams)):
    print beams[i][::-1], # have to reverse the order
    print probs[i],
    print np.exp(probs)[i]

"""
Correct output:

[0 0 0] -0.316082 0.729
[0 0 1] -2.51331 0.081
[0 1 1] -2.51331 0.081
[1 1 1] -2.51331 0.081
[1 2 2] -4.60517 0.01
[0 1 2] -4.71053 0.009
[1 1 2] -4.71053 0.009
[0 0 0] -1e+18 0.0
[0 0 1] -1e+18 0.0
[0 1 1] -1e+18 0.0
"""
