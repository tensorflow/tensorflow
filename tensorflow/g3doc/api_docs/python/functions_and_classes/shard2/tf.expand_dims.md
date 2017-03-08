### `tf.expand_dims(input, axis=None, name=None, dim=None)` {#expand_dims}

Inserts a axisension of 1 into a tensor's shape.

Given a tensor `input`, this operation inserts a axisension of 1 at the
axisension index `axis` of `input`'s shape. The axisension index `axis` starts at
zero; if you specify a negative number for `axis` it is counted backward from
the end.

This operation is useful if you want to add a batch axisension to a single
element. For example, if you have a single image of shape `[height, width,
channels]`, you can make it a batch of 1 image with `expand_axiss(image, 0)`,
which will make the shape `[1, height, width, channels]`.

Other examples:

```prettyprint
# 't' is a tensor of shape [2]
shape(expand_axiss(t, 0)) ==> [1, 2]
shape(expand_axiss(t, 1)) ==> [2, 1]
shape(expand_axiss(t, -1)) ==> [2, 1]

# 't2' is a tensor of shape [2, 3, 5]
shape(expand_axiss(t2, 0)) ==> [1, 2, 3, 5]
shape(expand_axiss(t2, 2)) ==> [2, 3, 1, 5]
shape(expand_axiss(t2, 3)) ==> [2, 3, 5, 1]
```

This operation requires that:

`-1-input.axiss() <= axis <= input.axiss()`

This operation is related to `squeeze()`, which removes axisensions of
size 1.

##### Args:


*  <b>`input`</b>: A `Tensor`.
*  <b>`axis`</b>: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    0-D (scalar). Specifies the axisension index at which to
    expand the shape of `input`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `input`.
  Contains the same data as `input`, but its shape has an additional
  axisension of size 1 added.

