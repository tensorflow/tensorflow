This directory contains functions for creating recurrent neural networks
and sequence-to-sequence models. Detailed instructions on how to get started
and use them are available in the tutorials.

* [RNN Tutorial](http://tensorflow.org/tutorials/recurrent/)
* [Sequence-to-Sequence Tutorial](http://tensorflow.org/tutorials/seq2seq/)

Here is a short overview of what is in this directory.

File | What's in it?
--- | ---
`linear.py` | Basic helper functions for creating linear layers.
`linear_test.py` | Unit tests for `linear.py`.
`rnn_cell.py` | Cells for recurrent neural networks, e.g., LSTM.
`rnn_cell_test.py` | Unit tests for `rnn_cell.py`.
`rnn.py` | Functions for building recurrent neural networks.
`rnn_test.py` | Unit tests for `rnn.py`.
`seq2seq.py` | Functions for building sequence-to-sequence models.
`seq2seq_test.py` | Unit tests for `seq2seq.py`.
`ptb/` | PTB language model, see the [RNN Tutorial](http://tensorflow.org/tutorials/recurrent/)
`translate/` | Translation model, see the [Sequence-to-Sequence Tutorial](http://tensorflow.org/tutorials/seq2seq/)
