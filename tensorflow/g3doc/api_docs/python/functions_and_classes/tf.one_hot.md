### `tf.one_hot(indices, depth, on_value, off_value, axis=None, name=None)` {#one_hot}

Returns a one-hot tensor.

The locations represented by indices in `indices` take value `on_value`,
while all other locations take value `off_value`.

If the input `indices` is rank `N`, the output will have rank `N+1`,
The new axis is created at dimension `axis` (default: the new axis is
appended at the end).

If `indices` is a scalar the output shape will be a vector of length `depth`.

If `indices` is a vector of length `features`, the output shape will be:
```
  features x depth if axis == -1
  depth x features if axis == 0
```

If `indices` is a matrix (batch) with shape `[batch, features]`,
the output shape will be:
```
  batch x features x depth if axis == -1
  batch x depth x features if axis == 1
  depth x batch x features if axis == 0
```


Examples
=========

Suppose that

```
  indices = [0, 2, -1, 1]
  depth = 3
  on_value = 5.0
  off_value = 0.0
  axis = -1
```

Then output is `[4 x 3]`:

    ```output =
      [5.0 0.0 0.0]  // one_hot(0)
      [0.0 0.0 5.0]  // one_hot(2)
      [0.0 0.0 0.0]  // one_hot(-1)
      [0.0 5.0 0.0]  // one_hot(1)
    ```

Suppose that

```
  indices = [0, 2, -1, 1]
  depth = 3
  on_value = 0.0
  off_value = 3.0
  axis = 0
```

Then output is `[3 x 4]`:

    ```output =
      [0.0 3.0 3.0 3.0]
      [3.0 3.0 3.0 0.0]
      [3.0 3.0 3.0 3.0]
      [3.0 0.0 3.0 3.0]
    //  ^                one_hot(0)
    //      ^            one_hot(2)
    //          ^        one_hot(-1)
    //              ^    one_hot(1)
    ```
Suppose that

```
  indices = [[0, 2], [1, -1]]
  depth = 3
  on_value = 1.0
  off_value = 0.0
  axis = -1
```

Then output is `[2 x 2 x 3]`:

    ```output =
      [
        [1.0, 0.0, 0.0]  // one_hot(0)
        [0.0, 0.0, 1.0]  // one_hot(2)
      ][
        [0.0, 1.0, 0.0]  // one_hot(1)
        [0.0, 0.0, 0.0]  // one_hot(-1)
      ]```

##### Args:


*  <b>`indices`</b>: A `Tensor` of type `int64`. A tensor of indices.
*  <b>`depth`</b>: A `Tensor` of type `int32`.
    A scalar defining the depth of the one hot dimension.
*  <b>`on_value`</b>: A `Tensor`.
    A scalar defining the value to fill in output when `indices[j] = i`.
*  <b>`off_value`</b>: A `Tensor`. Must have the same type as `on_value`.
    A scalar defining the value to fill in output when `indices[j] != i`.
*  <b>`axis`</b>: An optional `int`. Defaults to `-1`.
    The axis to fill (default: -1, a new inner-most axis).
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `on_value`. The one-hot tensor.

