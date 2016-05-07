### `tf.range(start, limit=None, delta=1, name='range')` {#range}

Creates a sequence of integers.

Creates a sequence of integers that begins at `start` and extends by
increments of `delta` up to but not including `limit`.

Like the Python builtin `range`, `start` defaults to 0, so that
`range(n) = range(0, n)`.

For example:

```
# 'start' is 3
# 'limit' is 18
# 'delta' is 3
tf.range(start, limit, delta) ==> [3, 6, 9, 12, 15]

# 'limit' is 5
tf.range(limit) ==> [0, 1, 2, 3, 4]
```

##### Args:


*  <b>`start`</b>: A 0-D (scalar) of type `int32`. First entry in sequence.
    Defaults to 0.
*  <b>`limit`</b>: A 0-D (scalar) of type `int32`. Upper limit of sequence,
    exclusive.
*  <b>`delta`</b>: A 0-D `Tensor` (scalar) of type `int32`. Optional. Default is 1.
    Number that increments `start`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  An 1-D `int32` `Tensor`.

