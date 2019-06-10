Using IPU optimized operations
------------------------------

Several custom versions of operators are provided to target operators
available in Poplibs.  See the :ref:`api-section` for more details.

LSTM
~~~~

See :py:class:`tensorflow.contrib.ipu.PopnnLSTM`.

Dropout
~~~~~~~

The Poplibs version of dropout does not need to store the dropout mask
between the forward and backward parts of the graph, saving memory.

See :py:func:`tensorflow.contrib.ipu.dropout`.

Embedding lookup
~~~~~~~~~~~~~~~~

This is a version of embedding lookup which will produce a smaller memory
footprint for small lookups. Instead of using dynamic lookup into the main
embedding dictionary, it uses a one hot operator and a multiply.

See :py:func:`tensorflow.contrib.ipu.embedding_lookup`.

Group normalization
~~~~~~~~~~~~~~~~~~~

Group normalization is an alternative to batch normalization, and produces
smaller and more optimized graphs.

The original paper on group normalization:
`"Group Normalization", Yuxin Wu, Kaiming He <https://arxiv.org/abs/1803.08494>`_.

See :py:func:`tensorflow.contrib.ipu.group_norm`.

