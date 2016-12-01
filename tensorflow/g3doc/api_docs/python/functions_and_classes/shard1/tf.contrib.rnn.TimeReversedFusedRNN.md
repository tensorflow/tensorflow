This is an adaptor to time-reverse a FusedRNNCell.

For example,

```python
cell = tf.nn.rnn_cell.BasicRNNCell(10)
fw_lstm = tf.contrib.rnn.FusedRNNCellAdaptor(cell, use_dynamic_rnn=True)
bw_lstm = tf.contrib.rnn.TimeReversedFusedRNN(fw_lstm)
fw_out, fw_state = fw_lstm(inputs)
bw_out, bw_state = bw_lstm(inputs)
```
- - -

#### `tf.contrib.rnn.TimeReversedFusedRNN.__call__(inputs, initial_state=None, dtype=None, sequence_length=None, scope=None)` {#TimeReversedFusedRNN.__call__}




- - -

#### `tf.contrib.rnn.TimeReversedFusedRNN.__init__(cell)` {#TimeReversedFusedRNN.__init__}




