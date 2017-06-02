import tensorflow as tf

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope

linear = rnn_cell._linear

class AttentionWrapper(tf.nn.rnn_cell.RNNCell):
    def __init__(self, cell, attention_states, batch_size, embedding, initializer=None, num_heads=1, scope=None):
        if not isinstance(cell, tf.nn.rnn_cell.RNNCell):
            raise TypeError("The parameter cell is not RNNCell.")
        self._cell = cell
        self._attention_states = attention_states
        self.embedding = embedding

        with variable_scope.variable_scope(scope or "attention_decoder"):
            # batch_size = attention_states.get_shape()[0].value
            attn_length = attention_states.get_shape()[1].value
            attn_size = attention_states.get_shape()[2].value

            hidden = array_ops.reshape(
            attention_states, [-1, attn_length, 1, attn_size])
            hidden_features = []
            v = []
            attention_vec_size = attn_size  # Size of query vectors for attention.
            for a in xrange(num_heads):
                k = variable_scope.get_variable("AttnW_%d" % a,
                                              [1, 1, attn_size, attention_vec_size])
                hidden_features.append(nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME"))
                v.append(variable_scope.get_variable("AttnV_%d" % a,
                                                   [attention_vec_size]))

            batch_attn_size = array_ops.pack([batch_size, attn_size])
            self.attns = [array_ops.zeros(batch_attn_size, dtype=dtypes.float32)
                 for _ in xrange(num_heads)]

            def attention(query):
                """Put attention masks on hidden using hidden_features and query."""
                ds = []  # Results of attention reads will be stored here.
                for a in xrange(num_heads):
                    with variable_scope.variable_scope("Attention_%d" % a):
                        y = linear(query, attention_vec_size, True)
                        y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size])
                        # Attention mask is a softmax of v^T * tanh(...).
                        s = math_ops.reduce_sum(
                          v[a] * math_ops.tanh(hidden_features[a] + y), [2, 3])
                        a = nn_ops.softmax(s)
                        # Now calculate the attention-weighted vector d.
                        d = math_ops.reduce_sum(
                          array_ops.reshape(a, [-1, attn_length, 1, 1]) * hidden,
                          [1, 2])
                        ds.append(array_ops.reshape(d, [-1, attn_size]))
                return ds

            self.attention = attention

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):
        with variable_scope.variable_scope(scope or "attention_decoder_cell"):
            inp = embedding_ops.embedding_lookup(self.embedding, inputs)
            input_size = inp.get_shape().with_rank(2)[1]
            x = linear([inp] + self.attns, input_size, True)
            cell_output, state = self._cell(x, state)
            self.attns = self.attention(state)
        return cell_output, state