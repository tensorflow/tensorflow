### `tf.contrib.layers.embed_sequence(ids, vocab_size=None, embed_dim=None, unique=False, initializer=None, regularizer=None, trainable=True, scope=None, reuse=None)` {#embed_sequence}

Maps a sequence of symbols to a sequence of embeddings.

Typical use case would be reusing embeddings between an encoder and decoder.

##### Args:


*  <b>`ids`</b>: `[batch_size, doc_length]` `Tensor` of type `int32` or `int64`
    with symbol ids.
*  <b>`vocab_size`</b>: Integer number of symbols in vocabulary.
*  <b>`embed_dim`</b>: Integer number of dimensions for embedding matrix.
*  <b>`unique`</b>: If `True`, will first compute the unique set of indices, and then
       lookup each embedding once, repeating them in the output as needed.
*  <b>`initializer`</b>: An initializer for the embeddings, if `None` default for
      current scope is used.
*  <b>`regularizer`</b>: Optional regularizer for the embeddings.
*  <b>`trainable`</b>: If `True` also add variables to the graph collection
    `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
*  <b>`scope`</b>: Optional string specifying the variable scope for the op, required
      if `reuse=True`.
*  <b>`reuse`</b>: If `True`, variables inside the op will be reused.

##### Returns:

  `Tensor` of `[batch_size, doc_length, embed_dim]` with embedded sequences.

##### Raises:


*  <b>`ValueError`</b>: if `embed_dim` or `vocab_size` are not specified when not
    `reuse` is `None` or `False`.

