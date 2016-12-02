### `tf.squeeze(input, axis=None, name=None, squeeze_dims=None)` {#squeeze}

Removes dimensions of size 1 from the shape of a tensor.

Given a tensor `input`, this operation returns a tensor of the same type with
all dimensions of size 1 removed. If you don't want to remove all size 1
dimensions, you can remove specific size 1 dimensions by specifying
`axis`.

For example:

```prettyprint
# 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
shape(squeeze(t)) ==> [2, 3]
```

Or, to remove specific size 1 dimensions:

```prettyprint
# 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
shape(squeeze(t, [2, 4])) ==> [1, 2, 3, 1]
```

##### Args:


*  <b>`input`</b>: A `Tensor`. The `input` to squeeze.
*  <b>`axis`</b>: An optional list of `ints`. Defaults to `[]`.
    If specified, only squeezes the dimensions listed. The dimension
    index starts at 0. It is an error to squeeze a dimension that is not 1.
*  <b>`name`</b>: A name for the operation (optional).
*  <b>`squeeze_dims`</b>: Deprecated keyword argument that is now axis.

##### Returns:

  A `Tensor`. Has the same type as `input`.
  Contains the same data as `input`, but has one or more dimensions of
  size 1 removed.

##### Raises:


*  <b>`ValueError`</b>: When both `squeeze_dims` and `axis` are specified.

