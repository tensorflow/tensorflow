### `tf.contrib.distributions.batch_index(vectors, indices, name=None)` {#batch_index}

Indexes into a batch of vectors.

##### Args:


*  <b>`vectors`</b>: An N-D Tensor.
*  <b>`indices`</b>: A K-D integer Tensor, K <= N. The first K - 1 dimensions of indices
      must be broadcastable to the first N - 1 dimensions of vectors.
*  <b>`name`</b>: A name for this operation (optional).

##### Returns:

  An N-D Tensor comprised of one element selected from each of the vectors.

##### Example usage:

  vectors = [[[1, 2, 3], [4, 5, 6]],
             [[7, 8, 9], [1, 2, 3]]]

  batch_index(vectors, 0)
  => [[1, 4],
      [7, 1]]

  batch_index(vectors, [0])
  => [[[1], [4]],
      [[7], [1]]]

  batch_index(vectors, [0, 0, 2, 2])
  => [[[1, 1, 3, 3], [4, 4, 6, 6]],
      [[7, 7, 9, 9], [1, 1, 3, 3]]]

  batch_index(vectors, [[0, 0, 2, 2], [0, 1, 2, 0]])
  => [[[1, 1, 3, 3], [4, 5, 6, 4]],
      [[7, 7, 9, 9], [1, 2, 3, 1]]]

