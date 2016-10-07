### `tf.dynamic_stitch(indices, data, name=None)` {#dynamic_stitch}

Interleave the values from the `data` tensors into a single tensor.

Builds a merged tensor such that

```python
    merged[indices[m][i, ..., j], ...] = data[m][i, ..., j, ...]
```

For example, if each `indices[m]` is scalar or vector, we have

```python
    # Scalar indices:
    merged[indices[m], ...] = data[m][...]

    # Vector indices:
    merged[indices[m][i], ...] = data[m][i, ...]
```

Each `data[i].shape` must start with the corresponding `indices[i].shape`,
and the rest of `data[i].shape` must be constant w.r.t. `i`.  That is, we
must have `data[i].shape = indices[i].shape + constant`.  In terms of this
`constant`, the output shape is

    merged.shape = [max(indices)] + constant

Values are merged in order, so if an index appears in both `indices[m][i]` and
`indices[n][j]` for `(m,i) < (n,j)` the slice `data[n][j]` will appear in the
merged result.

For example:

```python
    indices[0] = 6
    indices[1] = [4, 1]
    indices[2] = [[5, 2], [0, 3]]
    data[0] = [61, 62]
    data[1] = [[41, 42], [11, 12]]
    data[2] = [[[51, 52], [21, 22]], [[1, 2], [31, 32]]]
    merged = [[1, 2], [11, 12], [21, 22], [31, 32], [41, 42],
              [51, 52], [61, 62]]
```

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../../images/DynamicStitch.png" alt>
</div>

##### Args:


*  <b>`indices`</b>: A list of at least 1 `Tensor` objects of type `int32`.
*  <b>`data`</b>: A list with the same number of `Tensor` objects as `indices` of `Tensor` objects of the same type.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `data`.

