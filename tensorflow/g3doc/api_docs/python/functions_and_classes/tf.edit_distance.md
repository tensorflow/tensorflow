### `tf.edit_distance(hypothesis, truth, normalize=True, name='edit_distance')` {#edit_distance}

Computes the Levenshtein distance between sequences.

This operation takes variable-length sequences (`hypothesis` and `truth`),
each provided as a `SparseTensor`, and computes the Levenshtein distance.
You can normalize the edit distance by length of `truth` by setting
`normalize` to true.

For example, given the following input:

```python
# 'hypothesis' is a tensor of shape `[2, 1]` with variable-length values:
#   (0,0) = ["a"]
#   (1,0) = ["b"]
hypothesis = tf.SparseTensor(
    [[0, 0, 0],
     [1, 0, 0]],
    ["a", "b"]
    (2, 1, 1))

# 'truth' is a tensor of shape `[2, 2]` with variable-length values:
#   (0,0) = []
#   (0,1) = ["a"]
#   (1,0) = ["b", "c"]
#   (1,1) = ["a"]
truth = tf.SparseTensor(
    [[0, 1, 0],
     [1, 0, 0],
     [1, 0, 1],
     [1, 1, 0]]
    ["a", "b", "c", "a"],
    (2, 2, 2))

normalize = True
```

This operation would return the following:

```python
# 'output' is a tensor of shape `[2, 2]` with edit distances normalized
# by 'truth' lengths.
output ==> [[inf, 1.0],  # (0,0): no truth, (0,1): no hypothesis
           [0.5, 1.0]]  # (1,0): addition, (1,1): no hypothesis
```

##### Args:


*  <b>`hypothesis`</b>: A `SparseTensor` containing hypothesis sequences.
*  <b>`truth`</b>: A `SparseTensor` containing truth sequences.
*  <b>`normalize`</b>: A `bool`. If `True`, normalizes the Levenshtein distance by
    length of `truth.`
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A dense `Tensor` with rank `R - 1`, where R is the rank of the
  `SparseTensor` inputs `hypothesis` and `truth`.

##### Raises:


*  <b>`TypeError`</b>: If either `hypothesis` or `truth` are not a `SparseTensor`.

