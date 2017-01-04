### `tf.contrib.metrics.set_union(a, b, validate_indices=True)` {#set_union}

Compute set union of elements in last dimension of `a` and `b`.

All but the last dimension of `a` and `b` must match.

Example:

```python
  a = [
    [
      [
        [1, 2],
        [3],
      ],
      [
        [4],
        [5, 6],
      ],
    ],
  ]
  b = [
    [
      [
        [1, 3],
        [2],
      ],
      [
        [4, 5],
        [5, 6, 7, 8],
      ],
    ],
  ]
  set_union(a, b) = [
    [
      [
        [1, 2, 3],
        [2, 3],
      ],
      [
        [4, 5],
        [5, 6, 7, 8],
      ],
    ],
  ]
```

##### Args:


*  <b>`a`</b>: `Tensor` or `SparseTensor` of the same type as `b`. If sparse, indices
      must be sorted in row-major order.
*  <b>`b`</b>: `Tensor` or `SparseTensor` of the same type as `a`. If sparse, indices
      must be sorted in row-major order.
*  <b>`validate_indices`</b>: Whether to validate the order and range of sparse indices
     in `a` and `b`.

##### Returns:

  A `SparseTensor` whose shape is the same rank as `a` and `b`, and all but
  the last dimension the same. Elements along the last dimension contain the
  unions.

