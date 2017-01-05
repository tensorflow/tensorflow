<!-- This file is machine generated: DO NOT EDIT! -->

# Tensor Transformations

Note: Functions taking `Tensor` arguments can also take anything accepted by
[`tf.convert_to_tensor`](framework.md#convert_to_tensor).

[TOC]

## Casting

TensorFlow provides several operations that you can use to cast tensor data
types in your graph.

- - -

### `tf.string_to_number(string_tensor, out_type=None, name=None)` {#string_to_number}

Converts each string in the input Tensor to the specified numeric type.

(Note that int32 overflow results in an error while float overflow
results in a rounded value.)

##### Args:


*  <b>`string_tensor`</b>: A `Tensor` of type `string`.
*  <b>`out_type`</b>: An optional `tf.DType` from: `tf.float32, tf.int32`. Defaults to `tf.float32`.
    The numeric type to interpret each string in `string_tensor` as.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `out_type`.
  A Tensor of the same shape as the input `string_tensor`.


- - -

### `tf.to_double(x, name='ToDouble')` {#to_double}

Casts a tensor to type `float64`.

##### Args:


*  <b>`x`</b>: A `Tensor` or `SparseTensor`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` or `SparseTensor` with same shape as `x` with type `float64`.

##### Raises:


*  <b>`TypeError`</b>: If `x` cannot be cast to the `float64`.


- - -

### `tf.to_float(x, name='ToFloat')` {#to_float}

Casts a tensor to type `float32`.

##### Args:


*  <b>`x`</b>: A `Tensor` or `SparseTensor`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` or `SparseTensor` with same shape as `x` with type `float32`.

##### Raises:


*  <b>`TypeError`</b>: If `x` cannot be cast to the `float32`.


- - -

### `tf.to_bfloat16(x, name='ToBFloat16')` {#to_bfloat16}

Casts a tensor to type `bfloat16`.

##### Args:


*  <b>`x`</b>: A `Tensor` or `SparseTensor`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` or `SparseTensor` with same shape as `x` with type `bfloat16`.

##### Raises:


*  <b>`TypeError`</b>: If `x` cannot be cast to the `bfloat16`.


- - -

### `tf.to_int32(x, name='ToInt32')` {#to_int32}

Casts a tensor to type `int32`.

##### Args:


*  <b>`x`</b>: A `Tensor` or `SparseTensor`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` or `SparseTensor` with same shape as `x` with type `int32`.

##### Raises:


*  <b>`TypeError`</b>: If `x` cannot be cast to the `int32`.


- - -

### `tf.to_int64(x, name='ToInt64')` {#to_int64}

Casts a tensor to type `int64`.

##### Args:


*  <b>`x`</b>: A `Tensor` or `SparseTensor`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` or `SparseTensor` with same shape as `x` with type `int64`.

##### Raises:


*  <b>`TypeError`</b>: If `x` cannot be cast to the `int64`.


- - -

### `tf.cast(x, dtype, name=None)` {#cast}

Casts a tensor to a new type.

The operation casts `x` (in case of `Tensor`) or `x.values`
(in case of `SparseTensor`) to `dtype`.

For example:

```python
# tensor `a` is [1.8, 2.2], dtype=tf.float
tf.cast(a, tf.int32) ==> [1, 2]  # dtype=tf.int32
```

##### Args:


*  <b>`x`</b>: A `Tensor` or `SparseTensor`.
*  <b>`dtype`</b>: The destination type.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` or `SparseTensor` with same shape as `x`.

##### Raises:


*  <b>`TypeError`</b>: If `x` cannot be cast to the `dtype`.


- - -

### `tf.bitcast(input, type, name=None)` {#bitcast}

Bitcasts a tensor from one type to another without copying data.

Given a tensor `input`, this operation returns a tensor that has the same buffer
data as `input` with datatype `type`.

If the input datatype `T` is larger than the output datatype `type` then the
shape changes from [...] to [..., sizeof(`T`)/sizeof(`type`)].

If `T` is smaller than `type`, the operator requires that the rightmost
dimension be equal to sizeof(`type`)/sizeof(`T`). The shape then goes from
[..., sizeof(`type`)/sizeof(`T`)] to [...].

*NOTE*: Bitcast is implemented as a low-level cast, so machines with different
endian orderings will give different results.

##### Args:


*  <b>`input`</b>: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
*  <b>`type`</b>: A `tf.DType` from: `tf.float32, tf.float64, tf.int64, tf.int32, tf.uint8, tf.uint16, tf.int16, tf.int8, tf.complex64, tf.complex128, tf.qint8, tf.quint8, tf.qint32, tf.half`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `type`.


- - -

### `tf.saturate_cast(value, dtype, name=None)` {#saturate_cast}

Performs a safe saturating cast of `value` to `dtype`.

This function casts the input to `dtype` without applying any scaling.  If
there is a danger that values would over or underflow in the cast, this op
applies the appropriate clamping before the cast.

##### Args:


*  <b>`value`</b>: A `Tensor`.
*  <b>`dtype`</b>: The desired output `DType`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  `value` safely cast to `dtype`.



## Shapes and Shaping

TensorFlow provides several operations that you can use to determine the shape
of a tensor and change the shape of a tensor.

- - -

### `tf.broadcast_dynamic_shape(shape_x, shape_y)` {#broadcast_dynamic_shape}

Returns the broadcasted dynamic shape between `shape_x` and `shape_y`.

##### Args:


*  <b>`shape_x`</b>: A rank 1 integer `Tensor`, representing the shape of x.
*  <b>`shape_y`</b>: A rank 1 integer `Tensor`, representing the shape of x.

##### Returns:

  A rank 1 integer `Tensor` representing the broadcasted shape.


- - -

### `tf.broadcast_static_shape(shape_x, shape_y)` {#broadcast_static_shape}

Returns the broadcasted static shape between `shape_x` and `shape_y`.

##### Args:


*  <b>`shape_x`</b>: A `TensorShape`
*  <b>`shape_y`</b>: A `TensorShape`

##### Returns:

  A `TensorShape` representing the broadcasted shape.

##### Raises:


*  <b>`ValueError`</b>: If the two shapes can not be broadcasted.


- - -

### `tf.shape(input, name=None, out_type=tf.int32)` {#shape}

Returns the shape of a tensor.

This operation returns a 1-D integer tensor representing the shape of `input`.

For example:

```python
# 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
shape(t) ==> [2, 2, 3]
```

##### Args:


*  <b>`input`</b>: A `Tensor` or `SparseTensor`.
*  <b>`name`</b>: A name for the operation (optional).
*  <b>`out_type`</b>: (Optional) The specified output type of the operation
    (`int32` or `int64`). Defaults to `tf.int32`.

##### Returns:

  A `Tensor` of type `out_type`.


- - -

### `tf.shape_n(input, out_type=None, name=None)` {#shape_n}

Returns shape of tensors.

This operation returns N 1-D integer tensors representing shape of `input[i]s`.

##### Args:


*  <b>`input`</b>: A list of at least 1 `Tensor` objects of the same type.
*  <b>`out_type`</b>: An optional `tf.DType` from: `tf.int32, tf.int64`. Defaults to `tf.int32`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A list with the same number of `Tensor` objects as `input` of `Tensor` objects of type out_type.


- - -

### `tf.size(input, name=None, out_type=tf.int32)` {#size}

Returns the size of a tensor.

This operation returns an integer representing the number of elements in
`input`.

For example:

```python
# 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]]
size(t) ==> 12
```

##### Args:


*  <b>`input`</b>: A `Tensor` or `SparseTensor`.
*  <b>`name`</b>: A name for the operation (optional).
*  <b>`out_type`</b>: (Optional) The specified output type of the operation
    (`int32` or `int64`). Defaults to tf.int32.

##### Returns:

  A `Tensor` of type `out_type`. Defaults to tf.int32.


- - -

### `tf.rank(input, name=None)` {#rank}

Returns the rank of a tensor.

This operation returns an integer representing the rank of `input`.

For example:

```python
# 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
# shape of tensor 't' is [2, 2, 3]
rank(t) ==> 3
```

**Note**: The rank of a tensor is not the same as the rank of a matrix. The
rank of a tensor is the number of indices required to uniquely select each
element of the tensor. Rank is also known as "order", "degree", or "ndims."

##### Args:


*  <b>`input`</b>: A `Tensor` or `SparseTensor`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `int32`.

@compatibility(numpy)
Equivalent to np.ndim
@end_compatibility


- - -

### `tf.reshape(tensor, shape, name=None)` {#reshape}

Reshapes a tensor.

Given `tensor`, this operation returns a tensor that has the same values
as `tensor` with shape `shape`.

If one component of `shape` is the special value -1, the size of that dimension
is computed so that the total size remains constant.  In particular, a `shape`
of `[-1]` flattens into 1-D.  At most one component of `shape` can be -1.

If `shape` is 1-D or higher, then the operation returns a tensor with shape
`shape` filled with the values of `tensor`. In this case, the number of elements
implied by `shape` must be the same as the number of elements in `tensor`.

For example:

```prettyprint
# tensor 't' is [1, 2, 3, 4, 5, 6, 7, 8, 9]
# tensor 't' has shape [9]
reshape(t, [3, 3]) ==> [[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]]

# tensor 't' is [[[1, 1], [2, 2]],
#                [[3, 3], [4, 4]]]
# tensor 't' has shape [2, 2, 2]
reshape(t, [2, 4]) ==> [[1, 1, 2, 2],
                        [3, 3, 4, 4]]

# tensor 't' is [[[1, 1, 1],
#                 [2, 2, 2]],
#                [[3, 3, 3],
#                 [4, 4, 4]],
#                [[5, 5, 5],
#                 [6, 6, 6]]]
# tensor 't' has shape [3, 2, 3]
# pass '[-1]' to flatten 't'
reshape(t, [-1]) ==> [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6]

# -1 can also be used to infer the shape

# -1 is inferred to be 9:
reshape(t, [2, -1]) ==> [[1, 1, 1, 2, 2, 2, 3, 3, 3],
                         [4, 4, 4, 5, 5, 5, 6, 6, 6]]
# -1 is inferred to be 2:
reshape(t, [-1, 9]) ==> [[1, 1, 1, 2, 2, 2, 3, 3, 3],
                         [4, 4, 4, 5, 5, 5, 6, 6, 6]]
# -1 is inferred to be 3:
reshape(t, [ 2, -1, 3]) ==> [[[1, 1, 1],
                              [2, 2, 2],
                              [3, 3, 3]],
                             [[4, 4, 4],
                              [5, 5, 5],
                              [6, 6, 6]]]

# tensor 't' is [7]
# shape `[]` reshapes to a scalar
reshape(t, []) ==> 7
```

##### Args:


*  <b>`tensor`</b>: A `Tensor`.
*  <b>`shape`</b>: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    Defines the shape of the output tensor.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `tensor`.


- - -

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


- - -

### `tf.expand_dims(input, axis=None, name=None, dim=None)` {#expand_dims}

Inserts a dimension of 1 into a tensor's shape.

Given a tensor `input`, this operation inserts a dimension of 1 at the
dimension index `axis` of `input`'s shape. The dimension index `axis` starts
at zero; if you specify a negative number for `axis` it is counted backward
from the end.

This operation is useful if you want to add a batch dimension to a single
element. For example, if you have a single image of shape `[height, width,
channels]`, you can make it a batch of 1 image with `expand_dims(image, 0)`,
which will make the shape `[1, height, width, channels]`.

Other examples:

```python
# 't' is a tensor of shape [2]
shape(expand_dims(t, 0)) ==> [1, 2]
shape(expand_dims(t, 1)) ==> [2, 1]
shape(expand_dims(t, -1)) ==> [2, 1]

# 't2' is a tensor of shape [2, 3, 5]
shape(expand_dims(t2, 0)) ==> [1, 2, 3, 5]
shape(expand_dims(t2, 2)) ==> [2, 3, 1, 5]
shape(expand_dims(t2, 3)) ==> [2, 3, 5, 1]
```

This operation requires that:

`-1-input.dims() <= dim <= input.dims()`

This operation is related to `squeeze()`, which removes dimensions of
size 1.

##### Args:


*  <b>`input`</b>: A `Tensor`.
*  <b>`axis`</b>: 0-D (scalar). Specifies the dimension index at which to
    expand the shape of `input`.
*  <b>`name`</b>: The name of the output `Tensor`.
*  <b>`dim`</b>: 0-D (scalar). Equivalent to `axis`, to be deprecated.

##### Returns:

  A `Tensor` with the same data as `input`, but its shape has an additional
  dimension of size 1 added.

##### Raises:


*  <b>`ValueError`</b>: if both `dim` and `axis` are specified.


- - -

### `tf.meshgrid(*args, **kwargs)` {#meshgrid}

Broadcasts parameters for evaluation on an N-D grid.

Given N one-dimensional coordinate arrays `*args`, returns a list `outputs`
of N-D coordinate arrays for evaluating expressions on an N-D grid.

Notes:

`meshgrid` supports cartesian ('xy') and matrix ('ij') indexing conventions.
When the `indexing` argument is set to 'xy' (the default), the broadcasting
instructions for the first two dimensions are swapped.

Examples:

Calling `X, Y = meshgrid(x, y)` with the tensors

```prettyprint
  x = [1, 2, 3]
  y = [4, 5, 6]
```

results in

```prettyprint
  X = [[1, 1, 1],
       [2, 2, 2],
       [3, 3, 3]]
  Y = [[4, 5, 6],
       [4, 5, 6],
       [4, 5, 6]]
```

##### Args:


*  <b>`*args`</b>: `Tensor`s with rank 1
*  <b>`indexing`</b>: Either 'xy' or 'ij' (optional, default: 'xy')
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:


*  <b>`outputs`</b>: A list of N `Tensor`s with rank N



## Slicing and Joining

TensorFlow provides several operations to slice or extract parts of a tensor,
or join multiple tensors together.

- - -

### `tf.slice(input_, begin, size, name=None)` {#slice}

Extracts a slice from a tensor.

This operation extracts a slice of size `size` from a tensor `input` starting
at the location specified by `begin`. The slice `size` is represented as a
tensor shape, where `size[i]` is the number of elements of the 'i'th dimension
of `input` that you want to slice. The starting location (`begin`) for the
slice is represented as an offset in each dimension of `input`. In other
words, `begin[i]` is the offset into the 'i'th dimension of `input` that you
want to slice from.

`begin` is zero-based; `size` is one-based. If `size[i]` is -1,
all remaining elements in dimension i are included in the
slice. In other words, this is equivalent to setting:

`size[i] = input.dim_size(i) - begin[i]`

This operation requires that:

`0 <= begin[i] <= begin[i] + size[i] <= Di  for i in [0, n]`

For example:

```python
# 'input' is [[[1, 1, 1], [2, 2, 2]],
#             [[3, 3, 3], [4, 4, 4]],
#             [[5, 5, 5], [6, 6, 6]]]
tf.slice(input, [1, 0, 0], [1, 1, 3]) ==> [[[3, 3, 3]]]
tf.slice(input, [1, 0, 0], [1, 2, 3]) ==> [[[3, 3, 3],
                                            [4, 4, 4]]]
tf.slice(input, [1, 0, 0], [2, 1, 3]) ==> [[[3, 3, 3]],
                                           [[5, 5, 5]]]
```

##### Args:


*  <b>`input_`</b>: A `Tensor`.
*  <b>`begin`</b>: An `int32` or `int64` `Tensor`.
*  <b>`size`</b>: An `int32` or `int64` `Tensor`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` the same type as `input`.


- - -

### `tf.strided_slice(input_, begin, end, strides=None, begin_mask=0, end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=0, var=None, name=None)` {#strided_slice}

Extracts a strided slice from a tensor.

To a first order, this operation extracts a slice of size `end - begin`
from a tensor `input`
starting at the location specified by `begin`. The slice continues by adding
`stride` to the `begin` index until all dimensions are not less than `end`.
Note that components of stride can be negative, which causes a reverse
slice.

This operation can be thought of an encoding of a numpy style sliced
range. Given a python slice input[<spec0>, <spec1>, ..., <specn>]
this function will be called as follows.

`begin`, `end`, and `strides` will be all length n. n is in general
not the same dimensionality as `input`.

For the ith spec,
`begin_mask`, `end_mask`, `ellipsis_mask`, `new_axis_mask`,
and `shrink_axis_mask` will have the ith bit corresponding to
the ith spec.

If the ith bit of `begin_mask` is non-zero, `begin[i]` is ignored and
the fullest possible range in that dimension is used instead.
`end_mask` works analogously, except with the end range.

`foo[5:,:,:3]` on a 7x8x9 tensor is equivalent to `foo[5:7,0:8,0:3]`.
`foo[::-1]` reverses a tensor with shape 8.


If the ith bit of `ellipsis_mask`, as many unspecified dimensions
as needed will be inserted between other dimensions. Only one
non-zero bit is allowed in `ellipsis_mask`.

For example `foo[3:5,...,4:5]` on a shape 10x3x3x10 tensor is
equivalent to `foo[3:5,:,:,4:5]` and
`foo[3:5,...]` is equivalent to `foo[3:5,:,:,:]`.

If the ith bit of `new_axis_mask` is one, then a `begin`,
`end`, and `stride` are ignored and a new length 1 dimension is
added at this point in the output tensor.

For example `foo[3:5,4]` on a 10x8 tensor produces a shape 2 tensor
whereas `foo[3:5,4:5]` produces a shape 2x1 tensor with shrink_mask
being 1<<1 == 2.

If the ith bit of `shrink_axis_mask` is one, then `begin`,
`end[i]`, and `stride[i]` are used to do a slice in the appropriate
dimension, but the output tensor will be reduced in dimensionality
by one. This is only valid if the ith entry of slice[i]==1.

NOTE: `begin` and `end` are zero-indexed`.
`strides` entries must be non-zero.


```python
# 'input' is [[[1, 1, 1], [2, 2, 2]],
#             [[3, 3, 3], [4, 4, 4]],
#             [[5, 5, 5], [6, 6, 6]]]
tf.strided_slice(input, [1, 0, 0], [2, 1, 3], [1, 1, 1]) ==> [[[3, 3, 3]]]
tf.strided_slice(input, [1, 0, 0], [2, 2, 3], [1, 1, 1]) ==> [[[3, 3, 3],
                                                               [4, 4, 4]]]
tf.strided_slice(input, [1, 1, 0], [2, -1, 3], [1, -1, 1]) ==>[[[4, 4, 4],
                                                                [3, 3, 3]]]
```

##### Args:


*  <b>`input_`</b>: A `Tensor`.
*  <b>`begin`</b>: An `int32` or `int64` `Tensor`.
*  <b>`end`</b>: An `int32` or `int64` `Tensor`.
*  <b>`strides`</b>: An `int32` or `int64` `Tensor`.
*  <b>`begin_mask`</b>: An `int32` mask.
*  <b>`end_mask`</b>: An `int32` mask.
*  <b>`ellipsis_mask`</b>: An `int32` mask.
*  <b>`new_axis_mask`</b>: An `int32` mask.
*  <b>`shrink_axis_mask`</b>: An `int32` mask.
*  <b>`var`</b>: The variable corresponding to `input_` or None
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` the same type as `input`.


- - -

### `tf.split(value, num_or_size_splits, axis=0, num=None, name='split')` {#split}

Splits a tensor into sub tensors.

If `num_or_size_splits` is a scalar, `num_split`, then splits `value` along
dimension `axis` into `num_split` smaller tensors.
Requires that `num_split` evenly divides `value.shape[axis]`.

If `num_or_size_splits` is a tensor, `size_splits`, then splits `value` into
`len(size_splits)` pieces. The shape of the `i`-th piece has the same size as
the `value` except along dimension `axis` where the size is `size_splits[i]`.

For example:

```python
# 'value' is a tensor with shape [5, 30]
# Split 'value' into 3 tensors with sizes [4, 15, 11] along dimension 1
split0, split1, split2 = tf.split(value, [4, 15, 11], 1)
tf.shape(split0) ==> [5, 4]
tf.shape(split1) ==> [5, 15]
tf.shape(split2) ==> [5, 11]
# Split 'value' into 3 tensors along dimension 1
split0, split1, split2 = tf.split(value, num_or_size_splits=3, axis=1)
tf.shape(split0) ==> [5, 10]
```

##### Args:


*  <b>`value`</b>: The `Tensor` to split.
*  <b>`num_or_size_splits`</b>: Either an integer indicating the number of splits along
    split_dim or a 1-D Tensor containing the sizes of each output tensor
    along split_dim. If an integer then it must evenly divide
    `value.shape[axis]`; otherwise the sum of sizes along the split
    dimension must match that of the `value`.
*  <b>`axis`</b>: A 0-D `int32` `Tensor`. The dimension along which to split.
    Must be in the range `[0, rank(value))`. Defaults to 0.
*  <b>`num`</b>: Optional, used to specify the number of outputs when it cannot be
    inferred from the shape of `size_splits`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  if `num_or_size_splits` is a scalar returns `num_or_size_splits` `Tensor`
  objects; if `num_or_size_splits` is a 1-D Tensor returns
  `num_or_size_splits.get_shape[0]` `Tensor` objects resulting from splitting
  `value`.

##### Raises:


*  <b>`ValueError`</b>: If `num` is unspecified and cannot be inferred.


- - -

### `tf.tile(input, multiples, name=None)` {#tile}

Constructs a tensor by tiling a given tensor.

This operation creates a new tensor by replicating `input` `multiples` times.
The output tensor's i'th dimension has `input.dims(i) * multiples[i]` elements,
and the values of `input` are replicated `multiples[i]` times along the 'i'th
dimension. For example, tiling `[a b c d]` by `[2]` produces
`[a b c d a b c d]`.

##### Args:


*  <b>`input`</b>: A `Tensor`. 1-D or higher.
*  <b>`multiples`</b>: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    1-D. Length must be the same as the number of dimensions in `input`
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `input`.


- - -

### `tf.pad(tensor, paddings, mode='CONSTANT', name=None)` {#pad}

Pads a tensor.

This operation pads a `tensor` according to the `paddings` you specify.
`paddings` is an integer tensor with shape `[n, 2]`, where n is the rank of
`tensor`. For each dimension D of `input`, `paddings[D, 0]` indicates how
many values to add before the contents of `tensor` in that dimension, and
`paddings[D, 1]` indicates how many values to add after the contents of
`tensor` in that dimension. If `mode` is "REFLECT" then both `paddings[D, 0]`
and `paddings[D, 1]` must be no greater than `tensor.dim_size(D) - 1`. If
`mode` is "SYMMETRIC" then both `paddings[D, 0]` and `paddings[D, 1]` must be
no greater than `tensor.dim_size(D)`.

The padded size of each dimension D of the output is:

`paddings[D, 0] + tensor.dim_size(D) + paddings[D, 1]`

For example:

```python
# 't' is [[1, 2, 3], [4, 5, 6]].
# 'paddings' is [[1, 1,], [2, 2]].
# rank of 't' is 2.
pad(t, paddings, "CONSTANT") ==> [[0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 1, 2, 3, 0, 0],
                                  [0, 0, 4, 5, 6, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0]]

pad(t, paddings, "REFLECT") ==> [[6, 5, 4, 5, 6, 5, 4],
                                 [3, 2, 1, 2, 3, 2, 1],
                                 [6, 5, 4, 5, 6, 5, 4],
                                 [3, 2, 1, 2, 3, 2, 1]]

pad(t, paddings, "SYMMETRIC") ==> [[2, 1, 1, 2, 3, 3, 2],
                                   [2, 1, 1, 2, 3, 3, 2],
                                   [5, 4, 4, 5, 6, 6, 5],
                                   [5, 4, 4, 5, 6, 6, 5]]
```

##### Args:


*  <b>`tensor`</b>: A `Tensor`.
*  <b>`paddings`</b>: A `Tensor` of type `int32`.
*  <b>`mode`</b>: One of "CONSTANT", "REFLECT", or "SYMMETRIC" (case-insensitive)
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `tensor`.

##### Raises:


*  <b>`ValueError`</b>: When mode is not one of "CONSTANT", "REFLECT", or "SYMMETRIC".


- - -

### `tf.concat(*args, **kwargs)` {#concat}

Concatenates tensors along one dimension. (deprecated)

THIS FUNCTION IS DEPRECATED. It will be removed after 2016-12-14.
Instructions for updating:
This op will be removed after the deprecation date. Please switch to tf.concat_v2().

Concatenates the list of tensors `values` along dimension `concat_dim`.  If
`values[i].shape = [D0, D1, ... Dconcat_dim(i), ...Dn]`, the concatenated
result has shape

    [D0, D1, ... Rconcat_dim, ...Dn]

where

    Rconcat_dim = sum(Dconcat_dim(i))

That is, the data from the input tensors is joined along the `concat_dim`
dimension.

The number of dimensions of the input tensors must match, and all dimensions
except `concat_dim` must be equal.

For example:

```python
t1 = [[1, 2, 3], [4, 5, 6]]
t2 = [[7, 8, 9], [10, 11, 12]]
tf.concat(0, [t1, t2]) ==> [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
tf.concat(1, [t1, t2]) ==> [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]

# tensor t3 with shape [2, 3]
# tensor t4 with shape [2, 3]
tf.shape(tf.concat(0, [t3, t4])) ==> [4, 3]
tf.shape(tf.concat(1, [t3, t4])) ==> [2, 6]
```

Note: If you are concatenating along a new axis consider using pack.
E.g.

```python
tf.concat(axis, [tf.expand_dims(t, axis) for t in tensors])
```

can be rewritten as

```python
tf.pack(tensors, axis=axis)
```

##### Args:


*  <b>`concat_dim`</b>: 0-D `int32` `Tensor`.  Dimension along which to concatenate.
*  <b>`values`</b>: A list of `Tensor` objects or a single `Tensor`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` resulting from concatenation of the input tensors.


- - -

### `tf.concat_v2(values, axis, name='concat_v2')` {#concat_v2}

Concatenates tensors along one dimension.

Concatenates the list of tensors `values` along dimension `axis`.  If
`values[i].shape = [D0, D1, ... Daxis(i), ...Dn]`, the concatenated
result has shape

    [D0, D1, ... Raxis, ...Dn]

where

    Raxis = sum(Daxis(i))

That is, the data from the input tensors is joined along the `axis`
dimension.

The number of dimensions of the input tensors must match, and all dimensions
except `axis` must be equal.

For example:

```python
t1 = [[1, 2, 3], [4, 5, 6]]
t2 = [[7, 8, 9], [10, 11, 12]]
tf.concat_v2([t1, t2], 0) ==> [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
tf.concat_v2([t1, t2], 1) ==> [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]

# tensor t3 with shape [2, 3]
# tensor t4 with shape [2, 3]
tf.shape(tf.concat_v2([t3, t4], 0)) ==> [4, 3]
tf.shape(tf.concat_v2([t3, t4], 1)) ==> [2, 6]
```

Note: If you are concatenating along a new axis consider using pack.
E.g.

```python
tf.concat(axis, [tf.expand_dims(t, axis) for t in tensors])
```

can be rewritten as

```python
tf.pack(tensors, axis=axis)
```

##### Args:


*  <b>`values`</b>: A list of `Tensor` objects or a single `Tensor`.
*  <b>`axis`</b>: 0-D `int32` `Tensor`.  Dimension along which to concatenate.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` resulting from concatenation of the input tensors.


- - -

### `tf.stack(values, axis=0, name='stack')` {#stack}

Stacks a list of rank-`R` tensors into one rank-`(R+1)` tensor.

Packs the list of tensors in `values` into a tensor with rank one higher than
each tensor in `values`, by packing them along the `axis` dimension.
Given a list of length `N` of tensors of shape `(A, B, C)`;

if `axis == 0` then the `output` tensor will have the shape `(N, A, B, C)`.
if `axis == 1` then the `output` tensor will have the shape `(A, N, B, C)`.
Etc.

For example:

```prettyprint
# 'x' is [1, 4]
# 'y' is [2, 5]
# 'z' is [3, 6]
stack([x, y, z]) => [[1, 4], [2, 5], [3, 6]]  # Pack along first dim.
stack([x, y, z], axis=1) => [[1, 2, 3], [4, 5, 6]]
```

This is the opposite of unstack.  The numpy equivalent is

    tf.stack([x, y, z]) = np.asarray([x, y, z])

##### Args:


*  <b>`values`</b>: A list of `Tensor` objects with the same shape and type.
*  <b>`axis`</b>: An `int`. The axis to stack along. Defaults to the first dimension.
    Supports negative indexes.
*  <b>`name`</b>: A name for this operation (optional).

##### Returns:


*  <b>`output`</b>: A stacked `Tensor` with the same type as `values`.

##### Raises:


*  <b>`ValueError`</b>: If `axis` is out of the range [-(R+1), R+1).


- - -

### `tf.empty(output_shape, dtype, init=False)` {#empty}

Creates an empty Tensor with shape `output_shape` and type `dtype`.

The memory can optionally be initialized. This is usually useful in
conjunction with in-place operations.

##### Args:


*  <b>`output_shape`</b>: 1-D `Tensor` indicating the shape of the output.
*  <b>`dtype`</b>: The element type of the returned tensor.
*  <b>`init`</b>: `bool` indicating whether or not to zero the allocated memory.

##### Returns:


*  <b>`output`</b>: An empty Tensor of the specified type.


- - -

### `tf.empty_like(value, init=None)` {#empty_like}

Creates an empty Tensor with the same shape and type `dtype` as value.

The memory can optionally be initialized. This op is usually useful in
conjunction with in-place operations.

##### Args:


*  <b>`value`</b>: A `Tensor` whose shape will be used.
*  <b>`init`</b>: Initalize the returned tensor with the default value of
    `value.dtype()` if True.  Otherwise do not initialize.

##### Returns:


*  <b>`output`</b>: An empty Tensor of the specified shape and type.


- - -

### `tf.alias_inplace_update(value, loc, update)` {#alias_inplace_update}

Updates input `value` at `loc` with `update`. Aliases value.

   If `loc` is None, `value` and `update` must be the same size.
   ```
   value = update
   ```

   If `loc` is a scalar, `value` has rank 1 higher than `update`
   ```
   value[i, :] = update
   ```

   If `loc` is a vector, `value` has the same rank as `update`
   ```
   value[loc, :] = update
   ```

   Warning: If you use this function you will almost certainly want to add
   a control dependency as done in the implementation of parallel_stack to
   avoid race conditions.

##### Args:


*  <b>`value`</b>: A `Tensor` object that will be updated in-place.
*  <b>`loc`</b>: None, scalar or 1-D `Tensor`.
*  <b>`update`</b>: A `Tensor` of rank one less than `value` if `loc` is a scalar,
          otherwise of rank equal to `value` that contains the new values
          for `value`.

##### Returns:


*  <b>`output`</b>: `value` that has been updated accordingly.


- - -

### `tf.alias_inplace_add(value, loc, update)` {#alias_inplace_add}

Updates input `value` at `loc` with `update`. Aliases value.

   If `loc` is None, `value` and `update` must be the same size.
   ```
   value += update
   ```

   If `loc` is a scalar, `value` has rank 1 higher than `update`
   ```
   value[i, :] += update
   ```

   If `loc` is a vector, `value` has the same rank as `update`
   ```
   value[loc, :] += update
   ```

   Warning: If you use this function you will almost certainly want to add
   a control dependency as done in the implementation of parallel_stack to
   avoid race conditions.

##### Args:


*  <b>`value`</b>: A `Tensor` object that will be updated in-place.
*  <b>`loc`</b>: None, scalar or 1-D `Tensor`.
*  <b>`update`</b>: A `Tensor` of rank one less than `value` if `loc` is a scalar,
          otherwise of rank equal to `value` that contains the new values
          for `value`.

##### Returns:


*  <b>`output`</b>: `value` that has been updated accordingly.


- - -

### `tf.alias_inplace_subtract(value, loc, update)` {#alias_inplace_subtract}

Updates input `value` at `loc` with `update`. Aliases value.

   If `loc` is None, `value` and `update` must be the same size.
   ```
   value -= update
   ```

   If `loc` is a scalar, `value` has rank 1 higher than `update`
   ```
   value[i, :] -= update
   ```

   If `loc` is a vector, `value` has the same rank as `update`
   ```
   value[loc, :] -= update
   ```

   Warning: If you use this function you will almost certainly want to add
   a control dependency as done in the implementation of parallel_stack to
   avoid race conditions.

##### Args:


*  <b>`value`</b>: A `Tensor` object that will be updated in-place.
*  <b>`loc`</b>: None, Scalar or 1-D `Tensor`.
*  <b>`update`</b>: A `Tensor` of rank one less than `value` if `loc` is a scalar,
          otherwise of rank equal to `value` that contains the new values
          for `value`.

##### Returns:


*  <b>`output`</b>: `value` that has been updated accordingly.


- - -

### `tf.parallel_stack(values, name='parallel_stack')` {#parallel_stack}

Stacks a list of rank-`R` tensors into one rank-`(R+1)` tensor in parallel.

Requires that the shape of inputs be known at graph construction time.

Packs the list of tensors in `values` into a tensor with rank one higher than
each tensor in `values`, by packing them along the first dimension.
Given a list of length `N` of tensors of shape `(A, B, C)`; the `output`
tensor will have the shape `(N, A, B, C)`.

For example:

```prettyprint
# 'x' is [1, 4]
# 'y' is [2, 5]
# 'z' is [3, 6]
parallel_stack([x, y, z]) => [[1, 4], [2, 5], [3, 6]]
```

The difference between stack and parallel_stack is that stack requires all
of the inputs be computed before the operation will begin but doesn't require
that the input shapes be known during graph construction.  Parallel stack
will copy pieces of the input into the output as they become available, in
some situations this can provide a performance benefit.

This is the opposite of unstack.  The numpy equivalent is

    tf.parallel_stack([x, y, z]) = np.asarray([x, y, z])

##### Args:


*  <b>`values`</b>: A list of `Tensor` objects with the same shape and type.
*  <b>`name`</b>: A name for this operation (optional).

##### Returns:


*  <b>`output`</b>: A stacked `Tensor` with the same type as `values`.


- - -

### `tf.pack(*args, **kwargs)` {#pack}

Packs a list of rank-`R` tensors into one rank-`(R+1)` tensor. (deprecated)

THIS FUNCTION IS DEPRECATED. It will be removed after 2016-12-14.
Instructions for updating:
This op will be removed after the deprecation date. Please switch to tf.stack().

Packs the list of tensors in `values` into a tensor with rank one higher than
each tensor in `values`, by packing them along the `axis` dimension.
Given a list of length `N` of tensors of shape `(A, B, C)`;

if `axis == 0` then the `output` tensor will have the shape `(N, A, B, C)`.
if `axis == 1` then the `output` tensor will have the shape `(A, N, B, C)`.
Etc.

For example:

```prettyprint
# 'x' is [1, 4]
# 'y' is [2, 5]
# 'z' is [3, 6]
pack([x, y, z]) => [[1, 4], [2, 5], [3, 6]]  # Pack along first dim.
pack([x, y, z], axis=1) => [[1, 2, 3], [4, 5, 6]]
```

This is the opposite of unpack.  The numpy equivalent is

    tf.pack([x, y, z]) = np.asarray([x, y, z])

##### Args:


*  <b>`values`</b>: A list of `Tensor` objects with the same shape and type.
*  <b>`axis`</b>: An `int`. The axis to pack along. Defaults to the first dimension.
    Supports negative indexes.
*  <b>`name`</b>: A name for this operation (optional).

##### Returns:


*  <b>`output`</b>: A packed `Tensor` with the same type as `values`.

##### Raises:


*  <b>`ValueError`</b>: If `axis` is out of the range [-(R+1), R+1).


- - -

### `tf.unstack(value, num=None, axis=0, name='unstack')` {#unstack}

Unpacks the given dimension of a rank-`R` tensor into rank-`(R-1)` tensors.

Unpacks `num` tensors from `value` by chipping it along the `axis` dimension.
If `num` is not specified (the default), it is inferred from `value`'s shape.
If `value.shape[axis]` is not known, `ValueError` is raised.

For example, given a tensor of shape `(A, B, C, D)`;

If `axis == 0` then the i'th tensor in `output` is the slice
  `value[i, :, :, :]` and each tensor in `output` will have shape `(B, C, D)`.
  (Note that the dimension unpacked along is gone, unlike `split`).

If `axis == 1` then the i'th tensor in `output` is the slice
  `value[:, i, :, :]` and each tensor in `output` will have shape `(A, C, D)`.
Etc.

This is the opposite of pack.  The numpy equivalent is

    tf.unstack(x, n) = list(x)

##### Args:


*  <b>`value`</b>: A rank `R > 0` `Tensor` to be unstacked.
*  <b>`num`</b>: An `int`. The length of the dimension `axis`. Automatically inferred
    if `None` (the default).
*  <b>`axis`</b>: An `int`. The axis to unstack along. Defaults to the first
    dimension. Supports negative indexes.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  The list of `Tensor` objects unstacked from `value`.

##### Raises:


*  <b>`ValueError`</b>: If `num` is unspecified and cannot be inferred.
*  <b>`ValueError`</b>: If `axis` is out of the range [-R, R).


- - -

### `tf.unpack(*args, **kwargs)` {#unpack}

Unpacks the given dimension of a rank-`R` tensor into rank-`(R-1)` tensors. (deprecated)

THIS FUNCTION IS DEPRECATED. It will be removed after 2016-12-14.
Instructions for updating:
This op will be removed after the deprecation date. Please switch to tf.unstack().

Unpacks `num` tensors from `value` by chipping it along the `axis` dimension.
If `num` is not specified (the default), it is inferred from `value`'s shape.
If `value.shape[axis]` is not known, `ValueError` is raised.

For example, given a tensor of shape `(A, B, C, D)`;

If `axis == 0` then the i'th tensor in `output` is the slice
  `value[i, :, :, :]` and each tensor in `output` will have shape `(B, C, D)`.
  (Note that the dimension unpacked along is gone, unlike `split`).

If `axis == 1` then the i'th tensor in `output` is the slice
  `value[:, i, :, :]` and each tensor in `output` will have shape `(A, C, D)`.
Etc.

This is the opposite of pack.  The numpy equivalent is

    tf.unpack(x, n) = list(x)

##### Args:


*  <b>`value`</b>: A rank `R > 0` `Tensor` to be unpacked.
*  <b>`num`</b>: An `int`. The length of the dimension `axis`. Automatically inferred
    if `None` (the default).
*  <b>`axis`</b>: An `int`. The axis to unpack along. Defaults to the first
    dimension. Supports negative indexes.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  The list of `Tensor` objects unpacked from `value`.

##### Raises:


*  <b>`ValueError`</b>: If `num` is unspecified and cannot be inferred.
*  <b>`ValueError`</b>: If `axis` is out of the range [-R, R).


- - -

### `tf.reverse_sequence(input, seq_lengths, seq_axis=None, batch_axis=None, name=None, seq_dim=None, batch_dim=None)` {#reverse_sequence}

Reverses variable length slices.

This op first slices `input` along the dimension `batch_axis`, and for each
slice `i`, reverses the first `seq_lengths[i]` elements along
the dimension `seq_axis`.

The elements of `seq_lengths` must obey `seq_lengths[i] < input.dims[seq_dim]`,
and `seq_lengths` must be a vector of length `input.dims[batch_dim]`.

The output slice `i` along dimension `batch_axis` is then given by input
slice `i`, with the first `seq_lengths[i]` slices along dimension
`seq_axis` reversed.

For example:

```prettyprint
# Given this:
batch_dim = 0
seq_dim = 1
input.dims = (4, 8, ...)
seq_lengths = [7, 2, 3, 5]

# then slices of input are reversed on seq_dim, but only up to seq_lengths:
output[0, 0:7, :, ...] = input[0, 7:0:-1, :, ...]
output[1, 0:2, :, ...] = input[1, 2:0:-1, :, ...]
output[2, 0:3, :, ...] = input[2, 3:0:-1, :, ...]
output[3, 0:5, :, ...] = input[3, 5:0:-1, :, ...]

# while entries past seq_lens are copied through:
output[0, 7:, :, ...] = input[0, 7:, :, ...]
output[1, 2:, :, ...] = input[1, 2:, :, ...]
output[2, 3:, :, ...] = input[2, 3:, :, ...]
output[3, 2:, :, ...] = input[3, 2:, :, ...]
```

In contrast, if:

```prettyprint
# Given this:
batch_dim = 2
seq_dim = 0
input.dims = (8, ?, 4, ...)
seq_lengths = [7, 2, 3, 5]

# then slices of input are reversed on seq_dim, but only up to seq_lengths:
output[0:7, :, 0, :, ...] = input[7:0:-1, :, 0, :, ...]
output[0:2, :, 1, :, ...] = input[2:0:-1, :, 1, :, ...]
output[0:3, :, 2, :, ...] = input[3:0:-1, :, 2, :, ...]
output[0:5, :, 3, :, ...] = input[5:0:-1, :, 3, :, ...]

# while entries past seq_lens are copied through:
output[7:, :, 0, :, ...] = input[7:, :, 0, :, ...]
output[2:, :, 1, :, ...] = input[2:, :, 1, :, ...]
output[3:, :, 2, :, ...] = input[3:, :, 2, :, ...]
output[2:, :, 3, :, ...] = input[2:, :, 3, :, ...]
```

##### Args:


*  <b>`input`</b>: A `Tensor`. The input to reverse.
*  <b>`seq_lengths`</b>: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    1-D with length `input.dims(batch_dim)` and
    `max(seq_lengths) < input.dims(seq_dim)`
*  <b>`seq_axis`</b>: An `int`. The dimension which is partially reversed.
*  <b>`batch_axis`</b>: An optional `int`. Defaults to `0`.
    The dimension along which reversal is performed.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `input`.
  The partially reversed input. It has the same shape as `input`.


- - -

### `tf.reverse(tensor, axis, name=None)` {#reverse}

Reverses specific dimensions of a tensor.

NOTE `tf.reverse` has now changed behavior in preparation for 1.0.
`tf.reverse_v2` is currently an alias that will be deprecated before TF 1.0.

Given a `tensor`, and a `int32` tensor `axis` representing the set of
dimensions of `tensor` to reverse. This operation reverses each dimension
`i` for which there exists `j` s.t. `axis[j] == i`.

`tensor` can have up to 8 dimensions. The number of dimensions specified
in `axis` may be 0 or more entries. If an index is specified more than
once, a InvalidArgument error is raised.

For example:

```prettyprint
# tensor 't' is [[[[ 0,  1,  2,  3],
#                  [ 4,  5,  6,  7],
#                  [ 8,  9, 10, 11]],
#                 [[12, 13, 14, 15],
#                  [16, 17, 18, 19],
#                  [20, 21, 22, 23]]]]
# tensor 't' shape is [1, 2, 3, 4]

# 'dims' is [3] or 'dims' is -1
reverse(t, dims) ==> [[[[ 3,  2,  1,  0],
                        [ 7,  6,  5,  4],
                        [ 11, 10, 9, 8]],
                       [[15, 14, 13, 12],
                        [19, 18, 17, 16],
                        [23, 22, 21, 20]]]]

# 'dims' is '[1]' (or 'dims' is '[-3]')
reverse(t, dims) ==> [[[[12, 13, 14, 15],
                        [16, 17, 18, 19],
                        [20, 21, 22, 23]
                       [[ 0,  1,  2,  3],
                        [ 4,  5,  6,  7],
                        [ 8,  9, 10, 11]]]]

# 'dims' is '[2]' (or 'dims' is '[-2]')
reverse(t, dims) ==> [[[[8, 9, 10, 11],
                        [4, 5, 6, 7],
                        [0, 1, 2, 3]]
                       [[20, 21, 22, 23],
                        [16, 17, 18, 19],
                        [12, 13, 14, 15]]]]
```

##### Args:


*  <b>`tensor`</b>: A `Tensor`. Must be one of the following types: `uint8`, `int8`, `int32`, `int64`, `bool`, `half`, `float32`, `float64`, `complex64`, `complex128`.
    Up to 8-D.
*  <b>`axis`</b>: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    1-D. The indices of the dimensions to reverse.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `tensor`. The same shape as `tensor`.


- - -

### `tf.reverse_v2(tensor, axis, name=None)` {#reverse_v2}

Reverses specific dimensions of a tensor.

NOTE `tf.reverse` has now changed behavior in preparation for 1.0.
`tf.reverse_v2` is currently an alias that will be deprecated before TF 1.0.

Given a `tensor`, and a `int32` tensor `axis` representing the set of
dimensions of `tensor` to reverse. This operation reverses each dimension
`i` for which there exists `j` s.t. `axis[j] == i`.

`tensor` can have up to 8 dimensions. The number of dimensions specified
in `axis` may be 0 or more entries. If an index is specified more than
once, a InvalidArgument error is raised.

For example:

```prettyprint
# tensor 't' is [[[[ 0,  1,  2,  3],
#                  [ 4,  5,  6,  7],
#                  [ 8,  9, 10, 11]],
#                 [[12, 13, 14, 15],
#                  [16, 17, 18, 19],
#                  [20, 21, 22, 23]]]]
# tensor 't' shape is [1, 2, 3, 4]

# 'dims' is [3] or 'dims' is -1
reverse(t, dims) ==> [[[[ 3,  2,  1,  0],
                        [ 7,  6,  5,  4],
                        [ 11, 10, 9, 8]],
                       [[15, 14, 13, 12],
                        [19, 18, 17, 16],
                        [23, 22, 21, 20]]]]

# 'dims' is '[1]' (or 'dims' is '[-3]')
reverse(t, dims) ==> [[[[12, 13, 14, 15],
                        [16, 17, 18, 19],
                        [20, 21, 22, 23]
                       [[ 0,  1,  2,  3],
                        [ 4,  5,  6,  7],
                        [ 8,  9, 10, 11]]]]

# 'dims' is '[2]' (or 'dims' is '[-2]')
reverse(t, dims) ==> [[[[8, 9, 10, 11],
                        [4, 5, 6, 7],
                        [0, 1, 2, 3]]
                       [[20, 21, 22, 23],
                        [16, 17, 18, 19],
                        [12, 13, 14, 15]]]]
```

##### Args:


*  <b>`tensor`</b>: A `Tensor`. Must be one of the following types: `uint8`, `int8`, `int32`, `int64`, `bool`, `half`, `float32`, `float64`, `complex64`, `complex128`.
    Up to 8-D.
*  <b>`axis`</b>: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    1-D. The indices of the dimensions to reverse.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `tensor`. The same shape as `tensor`.


- - -

### `tf.transpose(a, perm=None, name='transpose')` {#transpose}

Transposes `a`. Permutes the dimensions according to `perm`.

The returned tensor's dimension i will correspond to the input dimension
`perm[i]`. If `perm` is not given, it is set to (n-1...0), where n is
the rank of the input tensor. Hence by default, this operation performs a
regular matrix transpose on 2-D input Tensors.

For example:

```python
# 'x' is [[1 2 3]
#         [4 5 6]]
tf.transpose(x) ==> [[1 4]
                     [2 5]
                     [3 6]]

# Equivalently
tf.transpose(x, perm=[1, 0]) ==> [[1 4]
                                  [2 5]
                                  [3 6]]

# 'perm' is more useful for n-dimensional tensors, for n > 2
# 'x' is   [[[1  2  3]
#            [4  5  6]]
#           [[7  8  9]
#            [10 11 12]]]
# Take the transpose of the matrices in dimension-0
tf.transpose(x, perm=[0, 2, 1]) ==> [[[1  4]
                                      [2  5]
                                      [3  6]]

                                     [[7 10]
                                      [8 11]
                                      [9 12]]]
```

##### Args:


*  <b>`a`</b>: A `Tensor`.
*  <b>`perm`</b>: A permutation of the dimensions of `a`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A transposed `Tensor`.


- - -

### `tf.extract_image_patches(images, ksizes, strides, rates, padding, name=None)` {#extract_image_patches}

Extract `patches` from `images` and put them in the "depth" output dimension.

##### Args:


*  <b>`images`</b>: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
    4-D Tensor with shape `[batch, in_rows, in_cols, depth]`.
*  <b>`ksizes`</b>: A list of `ints` that has length `>= 4`.
    The size of the sliding window for each dimension of `images`.
*  <b>`strides`</b>: A list of `ints` that has length `>= 4`.
    1-D of length 4. How far the centers of two consecutive patches are in
    the images. Must be: `[1, stride_rows, stride_cols, 1]`.
*  <b>`rates`</b>: A list of `ints` that has length `>= 4`.
    1-D of length 4. Must be: `[1, rate_rows, rate_cols, 1]`. This is the
    input stride, specifying how far two consecutive patch samples are in the
    input. Equivalent to extracting patches with
    `patch_sizes_eff = patch_sizes + (patch_sizes - 1) * (rates - 1)`, followed by
    subsampling them spatially by a factor of `rates`.
*  <b>`padding`</b>: A `string` from: `"SAME", "VALID"`.
    The type of padding algorithm to use.

    We specify the size-related attributes as:

    ```python
          ksizes = [1, ksize_rows, ksize_cols, 1]
          strides = [1, strides_rows, strides_cols, 1]
          rates = [1, rates_rows, rates_cols, 1]
    ```

*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `images`.
  4-D Tensor with shape `[batch, out_rows, out_cols, ksize_rows *
  ksize_cols * depth]` containing image patches with size
  `ksize_rows x ksize_cols x depth` vectorized in the "depth" dimension.


- - -

### `tf.space_to_batch_nd(input, block_shape, paddings, name=None)` {#space_to_batch_nd}

SpaceToBatch for N-D tensors of type T.

This operation divides "spatial" dimensions `[1, ..., M]` of the input into a
grid of blocks of shape `block_shape`, and interleaves these blocks with the
"batch" dimension (0) such that in the output, the spatial dimensions
`[1, ..., M]` correspond to the position within the grid, and the batch
dimension combines both the position within a spatial block and the original
batch position.  Prior to division into blocks, the spatial dimensions of the
input are optionally zero padded according to `paddings`.  See below for a
precise description.

##### Args:


*  <b>`input`</b>: A `Tensor`.
    N-D with shape `input_shape = [batch] + spatial_shape + remaining_shape`,
    where spatial_shape has `M` dimensions.
*  <b>`block_shape`</b>: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    1-D with shape `[M]`, all values must be >= 1.
*  <b>`paddings`</b>: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    2-D with shape `[M, 2]`, all values must be >= 0.
      `paddings[i] = [pad_start, pad_end]` specifies the padding for input dimension
      `i + 1`, which corresponds to spatial dimension `i`.  It is required that
      `block_shape[i]` divides `input_shape[i + 1] + pad_start + pad_end`.

    This operation is equivalent to the following steps:

    1. Zero-pad the start and end of dimensions `[1, ..., M]` of the
       input according to `paddings` to produce `padded` of shape `padded_shape`.

    2. Reshape `padded` to `reshaped_padded` of shape:

         [batch] +
         [padded_shape[1] / block_shape[0],
           block_shape[0],
          ...,
          padded_shape[M] / block_shape[M-1],
          block_shape[M-1]] +
         remaining_shape

    3. Permute dimensions of `reshaped_padded` to produce
       `permuted_reshaped_padded` of shape:

         block_shape +
         [batch] +
         [padded_shape[1] / block_shape[0],
          ...,
          padded_shape[M] / block_shape[M-1]] +
         remaining_shape

    4. Reshape `permuted_reshaped_padded` to flatten `block_shape` into the batch
       dimension, producing an output tensor of shape:

         [batch * prod(block_shape)] +
         [padded_shape[1] / block_shape[0],
          ...,
          padded_shape[M] / block_shape[M-1]] +
         remaining_shape

    Some examples:

    (1) For the following input of shape `[1, 2, 2, 1]`, `block_shape = [2, 2]`, and
        `paddings = [[0, 0], [0, 0]]`:

    ```prettyprint
    x = [[[[1], [2]], [[3], [4]]]]
    ```

    The output tensor has shape `[4, 1, 1, 1]` and value:

    ```prettyprint
    [[[[1]]], [[[2]]], [[[3]]], [[[4]]]]
    ```

    (2) For the following input of shape `[1, 2, 2, 3]`, `block_shape = [2, 2]`, and
        `paddings = [[0, 0], [0, 0]]`:

    ```prettyprint
    x = [[[[1, 2, 3], [4, 5, 6]],
          [[7, 8, 9], [10, 11, 12]]]]
    ```

    The output tensor has shape `[4, 1, 1, 3]` and value:

    ```prettyprint
    [[[1, 2, 3]], [[4, 5, 6]], [[7, 8, 9]], [[10, 11, 12]]]
    ```

    (3) For the following input of shape `[1, 4, 4, 1]`, `block_shape = [2, 2]`, and
        `paddings = [[0, 0], [0, 0]]`:

    ```prettyprint
    x = [[[[1],   [2],  [3],  [4]],
          [[5],   [6],  [7],  [8]],
          [[9],  [10], [11],  [12]],
          [[13], [14], [15],  [16]]]]
    ```

    The output tensor has shape `[4, 2, 2, 1]` and value:

    ```prettyprint
    x = [[[[1], [3]], [[5], [7]]],
         [[[2], [4]], [[10], [12]]],
         [[[5], [7]], [[13], [15]]],
         [[[6], [8]], [[14], [16]]]]
    ```

    (4) For the following input of shape `[2, 2, 4, 1]`, block_shape = `[2, 2]`, and
        paddings = `[[0, 0], [2, 0]]`:

    ```prettyprint
    x = [[[[1],   [2],  [3],  [4]],
          [[5],   [6],  [7],  [8]]],
         [[[9],  [10], [11],  [12]],
          [[13], [14], [15],  [16]]]]
    ```

    The output tensor has shape `[8, 1, 3, 1]` and value:

    ```prettyprint
    x = [[[[0], [1], [3]]], [[[0], [9], [11]]],
         [[[0], [2], [4]]], [[[0], [10], [12]]],
         [[[0], [5], [7]]], [[[0], [13], [15]]],
         [[[0], [6], [8]]], [[[0], [14], [16]]]]
    ```

    Among others, this operation is useful for reducing atrous convolution into
    regular convolution.

*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `input`.


- - -

### `tf.space_to_batch(input, paddings, block_size, name=None)` {#space_to_batch}

SpaceToBatch for 4-D tensors of type T.

This is a legacy version of the more general SpaceToBatchND.

Zero-pads and then rearranges (permutes) blocks of spatial data into batch.
More specifically, this op outputs a copy of the input tensor where values from
the `height` and `width` dimensions are moved to the `batch` dimension. After
the zero-padding, both `height` and `width` of the input must be divisible by the
block size.

##### Args:


*  <b>`input`</b>: A `Tensor`. 4-D with shape `[batch, height, width, depth]`.
*  <b>`paddings`</b>: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    2-D tensor of non-negative integers with shape `[2, 2]`. It specifies
      the padding of the input with zeros across the spatial dimensions as follows:

          paddings = [[pad_top, pad_bottom], [pad_left, pad_right]]

      The effective spatial dimensions of the zero-padded input tensor will be:

          height_pad = pad_top + height + pad_bottom
          width_pad = pad_left + width + pad_right

    The attr `block_size` must be greater than one. It indicates the block size.

      * Non-overlapping blocks of size `block_size x block size` in the height and
        width dimensions are rearranged into the batch dimension at each location.
      * The batch of the output tensor is `batch * block_size * block_size`.
      * Both height_pad and width_pad must be divisible by block_size.

    The shape of the output will be:

        [batch*block_size*block_size, height_pad/block_size, width_pad/block_size,
         depth]

    Some examples:

    (1) For the following input of shape `[1, 2, 2, 1]` and block_size of 2:

    ```prettyprint
    x = [[[[1], [2]], [[3], [4]]]]
    ```

    The output tensor has shape `[4, 1, 1, 1]` and value:

    ```prettyprint
    [[[[1]]], [[[2]]], [[[3]]], [[[4]]]]
    ```

    (2) For the following input of shape `[1, 2, 2, 3]` and block_size of 2:

    ```prettyprint
    x = [[[[1, 2, 3], [4, 5, 6]],
          [[7, 8, 9], [10, 11, 12]]]]
    ```

    The output tensor has shape `[4, 1, 1, 3]` and value:

    ```prettyprint
    [[[1, 2, 3]], [[4, 5, 6]], [[7, 8, 9]], [[10, 11, 12]]]
    ```

    (3) For the following input of shape `[1, 4, 4, 1]` and block_size of 2:

    ```prettyprint
    x = [[[[1],   [2],  [3],  [4]],
          [[5],   [6],  [7],  [8]],
          [[9],  [10], [11],  [12]],
          [[13], [14], [15],  [16]]]]
    ```

    The output tensor has shape `[4, 2, 2, 1]` and value:

    ```prettyprint
    x = [[[[1], [3]], [[5], [7]]],
         [[[2], [4]], [[10], [12]]],
         [[[5], [7]], [[13], [15]]],
         [[[6], [8]], [[14], [16]]]]
    ```

    (4) For the following input of shape `[2, 2, 4, 1]` and block_size of 2:

    ```prettyprint
    x = [[[[1],   [2],  [3],  [4]],
          [[5],   [6],  [7],  [8]]],
         [[[9],  [10], [11],  [12]],
          [[13], [14], [15],  [16]]]]
    ```

    The output tensor has shape `[8, 1, 2, 1]` and value:

    ```prettyprint
    x = [[[[1], [3]]], [[[9], [11]]], [[[2], [4]]], [[[10], [12]]],
         [[[5], [7]]], [[[13], [15]]], [[[6], [8]]], [[[14], [16]]]]
    ```

    Among others, this operation is useful for reducing atrous convolution into
    regular convolution.

*  <b>`block_size`</b>: An `int` that is `>= 2`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `input`.


- - -

### `tf.required_space_to_batch_paddings(input_shape, block_shape, base_paddings=None, name=None)` {#required_space_to_batch_paddings}

Calculate padding required to make block_shape divide input_shape.

This function can be used to calculate a suitable paddings argument for use
with space_to_batch_nd and batch_to_space_nd.

##### Args:


*  <b>`input_shape`</b>: int32 Tensor of shape [N].
*  <b>`block_shape`</b>: int32 Tensor of shape [N].
*  <b>`base_paddings`</b>: Optional int32 Tensor of shape [N, 2].  Specifies the minimum
    amount of padding to use.  All elements must be >= 0.  If not specified,
    defaults to 0.
*  <b>`name`</b>: string.  Optional name prefix.

##### Returns:

  (paddings, crops), where:

  `paddings` and `crops` are int32 Tensors of rank 2 and shape [N, 2]

*  <b>`satisfying`</b>: 

      paddings[i, 0] = base_paddings[i, 0].
      0 <= paddings[i, 1] - base_paddings[i, 1] < block_shape[i]
      (input_shape[i] + paddings[i, 0] + paddings[i, 1]) % block_shape[i] == 0

      crops[i, 0] = 0
      crops[i, 1] = paddings[i, 1] - base_paddings[i, 1]


*  <b>`Raises`</b>: ValueError if called with incompatible shapes.


- - -

### `tf.batch_to_space_nd(input, block_shape, crops, name=None)` {#batch_to_space_nd}

BatchToSpace for N-D tensors of type T.

This operation reshapes the "batch" dimension 0 into `M + 1` dimensions of shape
`block_shape + [batch]`, interleaves these blocks back into the grid defined by
the spatial dimensions `[1, ..., M]`, to obtain a result with the same rank as
the input.  The spatial dimensions of this intermediate result are then
optionally cropped according to `crops` to produce the output.  This is the
reverse of SpaceToBatch.  See below for a precise description.

##### Args:


*  <b>`input`</b>: A `Tensor`.
    N-D with shape `input_shape = [batch] + spatial_shape + remaining_shape`,
    where spatial_shape has M dimensions.
*  <b>`block_shape`</b>: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    1-D with shape `[M]`, all values must be >= 1.
*  <b>`crops`</b>: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    2-D with shape `[M, 2]`, all values must be >= 0.
      `crops[i] = [crop_start, crop_end]` specifies the amount to crop from input
      dimension `i + 1`, which corresponds to spatial dimension `i`.  It is
      required that
      `crop_start[i] + crop_end[i] <= block_shape[i] * input_shape[i + 1]`.

    This operation is equivalent to the following steps:

    1. Reshape `input` to `reshaped` of shape:
         [block_shape[0], ..., block_shape[M-1],
          batch / prod(block_shape),
          input_shape[1], ..., input_shape[N-1]]

    2. Permute dimensions of `reshaped` to produce `permuted` of shape
         [batch / prod(block_shape),

          input_shape[1], block_shape[0],
          ...,
          input_shape[M], block_shape[M-1],

          input_shape[M+1], ..., input_shape[N-1]]

    3. Reshape `permuted` to produce `reshaped_permuted` of shape
         [batch / prod(block_shape),

          input_shape[1] * block_shape[0],
          ...,
          input_shape[M] * block_shape[M-1],

          input_shape[M+1],
          ...,
          input_shape[N-1]]

    4. Crop the start and end of dimensions `[1, ..., M]` of
       `reshaped_permuted` according to `crops` to produce the output of shape:
         [batch / prod(block_shape),

          input_shape[1] * block_shape[0] - crops[0,0] - crops[0,1],
          ...,
          input_shape[M] * block_shape[M-1] - crops[M-1,0] - crops[M-1,1],

          input_shape[M+1], ..., input_shape[N-1]]

    Some examples:

    (1) For the following input of shape `[4, 1, 1, 1]`, `block_shape = [2, 2]`, and
        `crops = [[0, 0], [0, 0]]`:

    ```prettyprint
    [[[[1]]], [[[2]]], [[[3]]], [[[4]]]]
    ```

    The output tensor has shape `[1, 2, 2, 1]` and value:

    ```prettyprint
    x = [[[[1], [2]], [[3], [4]]]]
    ```

    (2) For the following input of shape `[4, 1, 1, 3]`, `block_shape = [2, 2]`, and
        `crops = [[0, 0], [0, 0]]`:

    ```prettyprint
    [[[1, 2, 3]], [[4, 5, 6]], [[7, 8, 9]], [[10, 11, 12]]]
    ```

    The output tensor has shape `[1, 2, 2, 3]` and value:

    ```prettyprint
    x = [[[[1, 2, 3], [4, 5, 6]],
          [[7, 8, 9], [10, 11, 12]]]]
    ```

    (3) For the following input of shape `[4, 2, 2, 1]`, `block_shape = [2, 2]`, and
        `crops = [[0, 0], [0, 0]]`:

    ```prettyprint
    x = [[[[1], [3]], [[5], [7]]],
         [[[2], [4]], [[10], [12]]],
         [[[5], [7]], [[13], [15]]],
         [[[6], [8]], [[14], [16]]]]
    ```

    The output tensor has shape `[1, 4, 4, 1]` and value:

    ```prettyprint
    x = [[[1],   [2],  [3],  [4]],
         [[5],   [6],  [7],  [8]],
         [[9],  [10], [11],  [12]],
         [[13], [14], [15],  [16]]]
    ```

    (4) For the following input of shape `[8, 1, 3, 1]`, `block_shape = [2, 2]`, and
        `crops = [[0, 0], [2, 0]]`:

    ```prettyprint
    x = [[[[0], [1], [3]]], [[[0], [9], [11]]],
         [[[0], [2], [4]]], [[[0], [10], [12]]],
         [[[0], [5], [7]]], [[[0], [13], [15]]],
         [[[0], [6], [8]]], [[[0], [14], [16]]]]
    ```

    The output tensor has shape `[2, 2, 4, 1]` and value:

    ```prettyprint
    x = [[[[1],   [2],  [3],  [4]],
          [[5],   [6],  [7],  [8]]],
         [[[9],  [10], [11],  [12]],
          [[13], [14], [15],  [16]]]]
    ```

*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `input`.


- - -

### `tf.batch_to_space(input, crops, block_size, name=None)` {#batch_to_space}

BatchToSpace for 4-D tensors of type T.

This is a legacy version of the more general BatchToSpaceND.

Rearranges (permutes) data from batch into blocks of spatial data, followed by
cropping. This is the reverse transformation of SpaceToBatch. More specifically,
this op outputs a copy of the input tensor where values from the `batch`
dimension are moved in spatial blocks to the `height` and `width` dimensions,
followed by cropping along the `height` and `width` dimensions.

##### Args:


*  <b>`input`</b>: A `Tensor`. 4-D tensor with shape
    `[batch*block_size*block_size, height_pad/block_size, width_pad/block_size,
      depth]`. Note that the batch size of the input tensor must be divisible by
    `block_size * block_size`.
*  <b>`crops`</b>: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    2-D tensor of non-negative integers with shape `[2, 2]`. It specifies
    how many elements to crop from the intermediate result across the spatial
    dimensions as follows:

        crops = [[crop_top, crop_bottom], [crop_left, crop_right]]

*  <b>`block_size`</b>: An `int` that is `>= 2`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `input`.
  4-D with shape `[batch, height, width, depth]`, where:

        height = height_pad - crop_top - crop_bottom
        width = width_pad - crop_left - crop_right

  The attr `block_size` must be greater than one. It indicates the block size.

  Some examples:

  (1) For the following input of shape `[4, 1, 1, 1]` and block_size of 2:

  ```prettyprint
  [[[[1]]], [[[2]]], [[[3]]], [[[4]]]]
  ```

  The output tensor has shape `[1, 2, 2, 1]` and value:

  ```prettyprint
  x = [[[[1], [2]], [[3], [4]]]]
  ```

  (2) For the following input of shape `[4, 1, 1, 3]` and block_size of 2:

  ```prettyprint
  [[[1, 2, 3]], [[4, 5, 6]], [[7, 8, 9]], [[10, 11, 12]]]
  ```

  The output tensor has shape `[1, 2, 2, 3]` and value:

  ```prettyprint
  x = [[[[1, 2, 3], [4, 5, 6]],
        [[7, 8, 9], [10, 11, 12]]]]
  ```

  (3) For the following input of shape `[4, 2, 2, 1]` and block_size of 2:

  ```prettyprint
  x = [[[[1], [3]], [[5], [7]]],
       [[[2], [4]], [[10], [12]]],
       [[[5], [7]], [[13], [15]]],
       [[[6], [8]], [[14], [16]]]]
  ```

  The output tensor has shape `[1, 4, 4, 1]` and value:

  ```prettyprint
  x = [[[1],   [2],  [3],  [4]],
       [[5],   [6],  [7],  [8]],
       [[9],  [10], [11],  [12]],
       [[13], [14], [15],  [16]]]
  ```

  (4) For the following input of shape `[8, 1, 2, 1]` and block_size of 2:

  ```prettyprint
  x = [[[[1], [3]]], [[[9], [11]]], [[[2], [4]]], [[[10], [12]]],
       [[[5], [7]]], [[[13], [15]]], [[[6], [8]]], [[[14], [16]]]]
  ```

  The output tensor has shape `[2, 2, 4, 1]` and value:

  ```prettyprint
  x = [[[[1], [3]], [[5], [7]]],
       [[[2], [4]], [[10], [12]]],
       [[[5], [7]], [[13], [15]]],
       [[[6], [8]], [[14], [16]]]]
  ```


- - -

### `tf.space_to_depth(input, block_size, name=None)` {#space_to_depth}

SpaceToDepth for tensors of type T.

Rearranges blocks of spatial data, into depth. More specifically,
this op outputs a copy of the input tensor where values from the `height`
and `width` dimensions are moved to the `depth` dimension.
The attr `block_size` indicates the input block size and how the data is moved.

  * Non-overlapping blocks of size `block_size x block size` are rearranged
    into depth at each location.
  * The depth of the output tensor is `input_depth * block_size * block_size`.
  * The input tensor's height and width must be divisible by block_size.

That is, assuming the input is in the shape:
`[batch, height, width, depth]`,
the shape of the output will be:
`[batch, height/block_size, width/block_size, depth*block_size*block_size]`

This operation requires that the input tensor be of rank 4, and that
`block_size` be >=1 and a divisor of both the input `height` and `width`.

This operation is useful for resizing the activations between convolutions
(but keeping all data), e.g. instead of pooling. It is also useful for training
purely convolutional models.

For example, given this input of shape `[1, 2, 2, 1]`, and block_size of 2:

```prettyprint
x = [[[[1], [2]],
      [[3], [4]]]]
```

This operation will output a tensor of shape `[1, 1, 1, 4]`:

```prettyprint
[[[[1, 2, 3, 4]]]]
```

Here, the input has a batch of 1 and each batch element has shape `[2, 2, 1]`,
the corresponding output will have a single element (i.e. width and height are
both 1) and will have a depth of 4 channels (1 * block_size * block_size).
The output element shape is `[1, 1, 4]`.

For an input tensor with larger depth, here of shape `[1, 2, 2, 3]`, e.g.

```prettyprint
x = [[[[1, 2, 3], [4, 5, 6]],
      [[7, 8, 9], [10, 11, 12]]]]
```

This operation, for block_size of 2, will return the following tensor of shape
`[1, 1, 1, 12]`

```prettyprint
[[[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]]]
```

Similarly, for the following input of shape `[1 4 4 1]`, and a block size of 2:

```prettyprint
x = [[[[1],   [2],  [5],  [6]],
      [[3],   [4],  [7],  [8]],
      [[9],  [10], [13],  [14]],
      [[11], [12], [15],  [16]]]]
```

the operator will return the following tensor of shape `[1 2 2 4]`:

```prettyprint
x = [[[[1, 2, 3, 4],
       [5, 6, 7, 8]],
      [[9, 10, 11, 12],
       [13, 14, 15, 16]]]]
```

##### Args:


*  <b>`input`</b>: A `Tensor`.
*  <b>`block_size`</b>: An `int` that is `>= 2`. The size of the spatial block.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `input`.


- - -

### `tf.depth_to_space(input, block_size, name=None)` {#depth_to_space}

DepthToSpace for tensors of type T.

Rearranges data from depth into blocks of spatial data.
This is the reverse transformation of SpaceToDepth. More specifically,
this op outputs a copy of the input tensor where values from the `depth`
dimension are moved in spatial blocks to the `height` and `width` dimensions.
The attr `block_size` indicates the input block size and how the data is moved.

  * Chunks of data of size `block_size * block_size` from depth are rearranged
    into non-overlapping blocks of size `block_size x block_size`
  * The width the output tensor is `input_width * block_size`, whereas the
    height is `input_height * block_size`.
  * The depth of the input tensor must be divisible by
    `block_size * block_size`.

That is, assuming the input is in the shape:
`[batch, height, width, depth]`,
the shape of the output will be:
`[batch, height*block_size, width*block_size, depth/(block_size*block_size)]`

This operation requires that the input tensor be of rank 4, and that
`block_size` be >=1 and that `block_size * block_size` be a divisor of the
input depth.

This operation is useful for resizing the activations between convolutions
(but keeping all data), e.g. instead of pooling. It is also useful for training
purely convolutional models.

For example, given this input of shape `[1, 1, 1, 4]`, and a block size of 2:

```prettyprint
x = [[[[1, 2, 3, 4]]]]

```

This operation will output a tensor of shape `[1, 2, 2, 1]`:

```prettyprint
   [[[[1], [2]],
     [[3], [4]]]]
```

Here, the input has a batch of 1 and each batch element has shape `[1, 1, 4]`,
the corresponding output will have 2x2 elements and will have a depth of
1 channel (1 = `4 / (block_size * block_size)`).
The output element shape is `[2, 2, 1]`.

For an input tensor with larger depth, here of shape `[1, 1, 1, 12]`, e.g.

```prettyprint
x = [[[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]]]
```

This operation, for block size of 2, will return the following tensor of shape
`[1, 2, 2, 3]`

```prettyprint
   [[[[1, 2, 3], [4, 5, 6]],
     [[7, 8, 9], [10, 11, 12]]]]

```

Similarly, for the following input of shape `[1 2 2 4]`, and a block size of 2:

```prettyprint
x =  [[[[1, 2, 3, 4],
       [5, 6, 7, 8]],
      [[9, 10, 11, 12],
       [13, 14, 15, 16]]]]
```

the operator will return the following tensor of shape `[1 4 4 1]`:

```prettyprint
x = [[ [1],   [2],  [5],  [6]],
     [ [3],   [4],  [7],  [8]],
     [ [9],  [10], [13],  [14]],
     [ [11], [12], [15],  [16]]]

```

##### Args:


*  <b>`input`</b>: A `Tensor`.
*  <b>`block_size`</b>: An `int` that is `>= 2`.
    The size of the spatial block, same as in Space2Depth.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `input`.


- - -

### `tf.gather(params, indices, validate_indices=None, name=None)` {#gather}

Gather slices from `params` according to `indices`.

`indices` must be an integer tensor of any dimension (usually 0-D or 1-D).
Produces an output tensor with shape `indices.shape + params.shape[1:]` where:

```python
    # Scalar indices
    output[:, ..., :] = params[indices, :, ... :]

    # Vector indices
    output[i, :, ..., :] = params[indices[i], :, ... :]

    # Higher rank indices
    output[i, ..., j, :, ... :] = params[indices[i, ..., j], :, ..., :]
```

If `indices` is a permutation and `len(indices) == params.shape[0]` then
this operation will permute `params` accordingly.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../../images/Gather.png" alt>
</div>

##### Args:


*  <b>`params`</b>: A `Tensor`.
*  <b>`indices`</b>: A `Tensor`. Must be one of the following types: `int32`, `int64`.
*  <b>`validate_indices`</b>: An optional `bool`. Defaults to `True`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `params`.


- - -

### `tf.gather_nd(params, indices, name=None)` {#gather_nd}

Gather values or slices from `params` according to `indices`.

`params` is a Tensor of rank `P` and `indices` is a Tensor of rank `Q`.

`indices` must be integer tensor, containing indices into `params`.
It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.

The innermost dimension of `indices` (with length `K`) corresponds to
indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th
dimension of `params`.

Produces an output tensor with shape

```
[d_0, ..., d_{Q-2}, params.shape[K], ..., params.shape[P-1]].
```

Some examples below.

Simple indexing into a matrix:

```python
    indices = [[0, 0], [1, 1]]
    params = [['a', 'b'], ['c', 'd']]
    output = ['a', 'd']
```

Slice indexing into a matrix:

```python
    indices = [[1], [0]]
    params = [['a', 'b'], ['c', 'd']]
    output = [['c', 'd'], ['a', 'b']]
```

Indexing into a 3-tensor:

```python
    indices = [[1]]
    params = [[['a0', 'b0'], ['c0', 'd0']],
              [['a1', 'b1'], ['c1', 'd1']]]
    output = [[['a1', 'b1'], ['c1', 'd1']]]


    indices = [[0, 1], [1, 0]]
    params = [[['a0', 'b0'], ['c0', 'd0']],
              [['a1', 'b1'], ['c1', 'd1']]]
    output = [['c0', 'd0'], ['a1', 'b1']]


    indices = [[0, 0, 1], [1, 0, 1]]
    params = [[['a0', 'b0'], ['c0', 'd0']],
              [['a1', 'b1'], ['c1', 'd1']]]
    output = ['b0', 'b1']
```

Batched indexing into a matrix:

```python
    indices = [[[0, 0]], [[0, 1]]]
    params = [['a', 'b'], ['c', 'd']]
    output = [['a'], ['b']]
```

Batched slice indexing into a matrix:

```python
    indices = [[[1]], [[0]]]
    params = [['a', 'b'], ['c', 'd']]
    output = [[['c', 'd']], [['a', 'b']]]
```

Batched indexing into a 3-tensor:

```python
    indices = [[[1]], [[0]]]
    params = [[['a0', 'b0'], ['c0', 'd0']],
              [['a1', 'b1'], ['c1', 'd1']]]
    output = [[[['a1', 'b1'], ['c1', 'd1']]],
              [[['a0', 'b0'], ['c0', 'd0']]]]

    indices = [[[0, 1], [1, 0]], [[0, 0], [1, 1]]]
    params = [[['a0', 'b0'], ['c0', 'd0']],
              [['a1', 'b1'], ['c1', 'd1']]]
    output = [[['c0', 'd0'], ['a1', 'b1']],
              [['a0', 'b0'], ['c1', 'd1']]]


    indices = [[[0, 0, 1], [1, 0, 1]], [[0, 1, 1], [1, 1, 0]]]
    params = [[['a0', 'b0'], ['c0', 'd0']],
              [['a1', 'b1'], ['c1', 'd1']]]
    output = [['b0', 'b1'], ['d0', 'c1']]
```

##### Args:


*  <b>`params`</b>: A `Tensor`. `P-D`.  The tensor from which to gather values.
*  <b>`indices`</b>: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    `Q-D`.  Index tensor having shape `[d_0, ..., d_{Q-2}, K]`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `params`.
  `(P+Q-K-1)-D`.  Values from `params` gathered from indices given by
  `indices`.


- - -

### `tf.unique_with_counts(x, out_idx=None, name=None)` {#unique_with_counts}

Finds unique elements in a 1-D tensor.

This operation returns a tensor `y` containing all of the unique elements of `x`
sorted in the same order that they occur in `x`. This operation also returns a
tensor `idx` the same size as `x` that contains the index of each value of `x`
in the unique output `y`. Finally, it returns a third tensor `count` that
contains the count of each element of `y` in `x`. In other words:

`y[idx[i]] = x[i] for i in [0, 1,...,rank(x) - 1]`

For example:

```prettyprint
# tensor 'x' is [1, 1, 2, 4, 4, 4, 7, 8, 8]
y, idx, count = unique_with_counts(x)
y ==> [1, 2, 4, 7, 8]
idx ==> [0, 0, 1, 2, 2, 2, 3, 4, 4]
count ==> [2, 1, 3, 1, 2]
```

##### Args:


*  <b>`x`</b>: A `Tensor`. 1-D.
*  <b>`out_idx`</b>: An optional `tf.DType` from: `tf.int32, tf.int64`. Defaults to `tf.int32`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A tuple of `Tensor` objects (y, idx, count).

*  <b>`y`</b>: A `Tensor`. Has the same type as `x`. 1-D.
*  <b>`idx`</b>: A `Tensor` of type `out_idx`. 1-D.
*  <b>`count`</b>: A `Tensor` of type `out_idx`. 1-D.


- - -

### `tf.scatter_nd(indices, updates, shape, name=None)` {#scatter_nd}

Creates a new tensor by applying sparse `updates` to individual

values or slices within a zero tensor of the given `shape` tensor according to
indices.  This operator is the inverse of the [tf.gather_nd](#gather_nd)
operator which extracts values or slices from a given tensor.

TODO(simister): Add a link to Variable.__getitem__ documentation on slice
syntax.

`shape` is a `TensorShape` with rank `P` and `indices` is a `Tensor` of rank
`Q`.

`indices` must be integer tensor, containing indices into `shape`.
It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.

The innermost dimension of `indices` (with length `K`) corresponds to
indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th
dimension of `shape`.

`updates` is Tensor of rank `Q-1+P-K` with shape:

```
[d_0, ..., d_{Q-2}, shape[K], ..., shape[P-1]].
```

The simplest form of scatter is to insert individual elements in a tensor by
index. For example, say we want to insert 4 scattered elements in a rank-1
tensor with 8 elements.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../../images/ScatterNd1.png" alt>
</div>

In Python, this scatter operation would look like this:

    indices = tf.constant([[4], [3], [1], [7]])
    updates = tf.constant([9, 10, 11, 12])
    shape = tf.constant([8])
    scatter = tf.scatter_nd(indices, updates, shape)
    with tf.Session() as sess:
      print sess.run(scatter)

The resulting tensor would look like this:

    [0, 11, 0, 10, 9, 0, 0, 12]

We can also, insert entire slices of a higher rank tensor all at once. For
example, if we wanted to insert two slices in the first dimension of a
rank-3 tensor with two matrices of new values.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../../images/ScatterNd2.png" alt>
</div>

In Python, this scatter operation would look like this:

    indices = tf.constant([[0], [2]])
    updates = tf.constant([[[5, 5, 5, 5], [6, 6, 6, 6],
                            [7, 7, 7, 7], [8, 8, 8, 8]],
                           [[5, 5, 5, 5], [6, 6, 6, 6],
                            [7, 7, 7, 7], [8, 8, 8, 8]]])
    shape = tf.constant([4, 4, 4])
    scatter = tf.scatter_nd(indices, updates, shape)
    with tf.Session() as sess:
      print sess.run(scatter)

The resulting tensor would look like this:

    [[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
     [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
     [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
     [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]

##### Args:


*  <b>`indices`</b>: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    A Tensor. Must be one of the following types: int32, int64.
    A tensor of indices into ref.
*  <b>`updates`</b>: A `Tensor`.
    A Tensor. Must have the same type as tensor. A tensor of updated values
    to store in ref.
*  <b>`shape`</b>: A `Tensor`. Must have the same type as `indices`.
    A vector. The shape of the resulting tensor.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `updates`.
  A new tensor with the given shape and updates applied according
  to the indices.


- - -

### `tf.dynamic_partition(data, partitions, num_partitions, name=None)` {#dynamic_partition}

Partitions `data` into `num_partitions` tensors using indices from `partitions`.

For each index tuple `js` of size `partitions.ndim`, the slice `data[js, ...]`
becomes part of `outputs[partitions[js]]`.  The slices with `partitions[js] = i`
are placed in `outputs[i]` in lexicographic order of `js`, and the first
dimension of `outputs[i]` is the number of entries in `partitions` equal to `i`.
In detail,

```python
    outputs[i].shape = [sum(partitions == i)] + data.shape[partitions.ndim:]

    outputs[i] = pack([data[js, ...] for js if partitions[js] == i])
```

`data.shape` must start with `partitions.shape`.

For example:

```python
    # Scalar partitions.
    partitions = 1
    num_partitions = 2
    data = [10, 20]
    outputs[0] = []  # Empty with shape [0, 2]
    outputs[1] = [[10, 20]]

    # Vector partitions.
    partitions = [0, 0, 1, 1, 0]
    num_partitions = 2
    data = [10, 20, 30, 40, 50]
    outputs[0] = [10, 20, 50]
    outputs[1] = [30, 40]
```

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../../images/DynamicPartition.png" alt>
</div>

##### Args:


*  <b>`data`</b>: A `Tensor`.
*  <b>`partitions`</b>: A `Tensor` of type `int32`.
    Any shape.  Indices in the range `[0, num_partitions)`.
*  <b>`num_partitions`</b>: An `int` that is `>= 1`.
    The number of partitions to output.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A list of `num_partitions` `Tensor` objects of the same type as data.


- - -

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


- - -

### `tf.boolean_mask(tensor, mask, name='boolean_mask')` {#boolean_mask}

Apply boolean mask to tensor.  Numpy equivalent is `tensor[mask]`.

```python
# 1-D example
tensor = [0, 1, 2, 3]
mask = np.array([True, False, True, False])
boolean_mask(tensor, mask) ==> [0, 2]
```

In general, `0 < dim(mask) = K <= dim(tensor)`, and `mask`'s shape must match
the first K dimensions of `tensor`'s shape.  We then have:
  `boolean_mask(tensor, mask)[i, j1,...,jd] = tensor[i1,...,iK,j1,...,jd]`
where `(i1,...,iK)` is the ith `True` entry of `mask` (row-major order).

##### Args:


*  <b>`tensor`</b>: N-D tensor.
*  <b>`mask`</b>: K-D boolean tensor, K <= N and K must be known statically.
*  <b>`name`</b>: A name for this operation (optional).

##### Returns:

  (N-K+1)-dimensional tensor populated by entries in `tensor` corresponding
  to `True` values in `mask`.

##### Raises:


*  <b>`ValueError`</b>: If shapes do not conform.


*  <b>`Examples`</b>: 

```python
# 2-D example
tensor = [[1, 2], [3, 4], [5, 6]]
mask = np.array([True, False, True])
boolean_mask(tensor, mask) ==> [[1, 2], [5, 6]]
```


- - -

### `tf.one_hot(indices, depth, on_value=None, off_value=None, axis=None, dtype=None, name=None)` {#one_hot}

Returns a one-hot tensor.

The locations represented by indices in `indices` take value `on_value`,
while all other locations take value `off_value`.

`on_value` and `off_value` must have matching data types. If `dtype` is also
provided, they must be the same data type as specified by `dtype`.

If `on_value` is not provided, it will default to the value `1` with type
`dtype`

If `off_value` is not provided, it will default to the value `0` with type
`dtype`

If the input `indices` is rank `N`, the output will have rank `N+1`. The
new axis is created at dimension `axis` (default: the new axis is appended
at the end).

If `indices` is a scalar the output shape will be a vector of length `depth`

If `indices` is a vector of length `features`, the output shape will be:

```
  features x depth if axis == -1
  depth x features if axis == 0
```

If `indices` is a matrix (batch) with shape `[batch, features]`, the output
shape will be:

```
  batch x features x depth if axis == -1
  batch x depth x features if axis == 1
  depth x batch x features if axis == 0
```

If `dtype` is not provided, it will attempt to assume the data type of
`on_value` or `off_value`, if one or both are passed in. If none of
`on_value`, `off_value`, or `dtype` are provided, `dtype` will default to the
value `tf.float32`.

Note: If a non-numeric data type output is desired (`tf.string`, `tf.bool`,
etc.), both `on_value` and `off_value` _must_ be provided to `one_hot`.

Examples
=========

Suppose that

```python
  indices = [0, 2, -1, 1]
  depth = 3
  on_value = 5.0
  off_value = 0.0
  axis = -1
```

Then output is `[4 x 3]`:

```python
  output =
  [5.0 0.0 0.0]  // one_hot(0)
  [0.0 0.0 5.0]  // one_hot(2)
  [0.0 0.0 0.0]  // one_hot(-1)
  [0.0 5.0 0.0]  // one_hot(1)
```

Suppose that

```python
  indices = [[0, 2], [1, -1]]
  depth = 3
  on_value = 1.0
  off_value = 0.0
  axis = -1
```

Then output is `[2 x 2 x 3]`:

```python
  output =
  [
    [1.0, 0.0, 0.0]  // one_hot(0)
    [0.0, 0.0, 1.0]  // one_hot(2)
  ][
    [0.0, 1.0, 0.0]  // one_hot(1)
    [0.0, 0.0, 0.0]  // one_hot(-1)
  ]
```

Using default values for `on_value` and `off_value`:

```python
  indices = [0, 1, 2]
  depth = 3
```

The output will be

```python
  output =
  [[1., 0., 0.],
   [0., 1., 0.],
   [0., 0., 1.]]
```

##### Args:


*  <b>`indices`</b>: A `Tensor` of indices.
*  <b>`depth`</b>: A scalar defining the depth of the one hot dimension.
*  <b>`on_value`</b>: A scalar defining the value to fill in output when `indices[j]
    = i`. (default: 1)
*  <b>`off_value`</b>: A scalar defining the value to fill in output when `indices[j]
    != i`. (default: 0)
*  <b>`axis`</b>: The axis to fill (default: -1, a new inner-most axis).
*  <b>`dtype`</b>: The data type of the output tensor.

##### Returns:


*  <b>`output`</b>: The one-hot tensor.

##### Raises:


*  <b>`TypeError`</b>: If dtype of either `on_value` or `off_value` don't match `dtype`
*  <b>`TypeError`</b>: If dtype of `on_value` and `off_value` don't match one another


- - -

### `tf.sequence_mask(lengths, maxlen=None, dtype=tf.bool, name=None)` {#sequence_mask}

Return a mask tensor representing the first N positions of each row.

Example:

```python
tf.sequence_mask([1, 3, 2], 5) =
  [[True, False, False, False, False],
   [True, True, True, False, False],
   [True, True, False, False, False]]
```

##### Args:


*  <b>`lengths`</b>: 1D integer tensor, all its values < maxlen.
*  <b>`maxlen`</b>: scalar integer tensor, maximum length of each row. Default: use
          maximum over lengths.
*  <b>`dtype`</b>: output type of the resulting tensor.
*  <b>`name`</b>: name of the op.

##### Returns:

  A 2D mask tensor, as shown in the example above, cast to specified dtype.

##### Raises:


*  <b>`ValueError`</b>: if the arguments have invalid rank.


- - -

### `tf.dequantize(input, min_range, max_range, mode=None, name=None)` {#dequantize}

Dequantize the 'input' tensor into a float Tensor.

[min_range, max_range] are scalar floats that specify the range for
the 'input' data. The 'mode' attribute controls exactly which calculations are
used to convert the float values to their quantized equivalents.

In 'MIN_COMBINED' mode, each value of the tensor will undergo the following:

```
if T == qint8, in[i] += (range(T) + 1)/ 2.0
out[i] = min_range + (in[i]* (max_range - min_range) / range(T))
```
here `range(T) = numeric_limits<T>::max() - numeric_limits<T>::min()`

*MIN_COMBINED Mode Example*

If the input comes from a QuantizedRelu6, the output type is
quint8 (range of 0-255) but the possible range of QuantizedRelu6 is
0-6.  The min_range and max_range values are therefore 0.0 and 6.0.
Dequantize on quint8 will take each value, cast to float, and multiply
by 6 / 255.
Note that if quantizedtype is qint8, the operation will additionally add
each value by 128 prior to casting.

If the mode is 'MIN_FIRST', then this approach is used:

```
number_of_steps = 1 << (# of bits in T)
range_adjust = number_of_steps / (number_of_steps - 1)
range = (range_max - range_min) * range_adjust
range_scale = range / number_of_steps
const double offset_input = static_cast<double>(input) - lowest_quantized;
result = range_min + ((input - numeric_limits<T>::min()) * range_scale)
```

##### Args:


*  <b>`input`</b>: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint16`, `quint16`, `qint32`.
*  <b>`min_range`</b>: A `Tensor` of type `float32`.
    The minimum scalar value possibly produced for the input.
*  <b>`max_range`</b>: A `Tensor` of type `float32`.
    The maximum scalar value possibly produced for the input.
*  <b>`mode`</b>: An optional `string` from: `"MIN_COMBINED", "MIN_FIRST"`. Defaults to `"MIN_COMBINED"`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `float32`.


- - -

### `tf.quantize_v2(input, min_range, max_range, T, mode=None, name=None)` {#quantize_v2}

Quantize the 'input' tensor of type float to 'output' tensor of type 'T'.

[min_range, max_range] are scalar floats that specify the range for
the 'input' data. The 'mode' attribute controls exactly which calculations are
used to convert the float values to their quantized equivalents.

In 'MIN_COMBINED' mode, each value of the tensor will undergo the following:

```
out[i] = (in[i] - min_range) * range(T) / (max_range - min_range)
if T == qint8, out[i] -= (range(T) + 1) / 2.0
```
here `range(T) = numeric_limits<T>::max() - numeric_limits<T>::min()`

*MIN_COMBINED Mode Example*

Assume the input is type float and has a possible range of [0.0, 6.0] and the
output type is quint8 ([0, 255]). The min_range and max_range values should be
specified as 0.0 and 6.0. Quantizing from float to quint8 will multiply each
value of the input by 255/6 and cast to quint8.

If the output type was qint8 ([-128, 127]), the operation will additionally
subtract each value by 128 prior to casting, so that the range of values aligns
with the range of qint8.

If the mode is 'MIN_FIRST', then this approach is used:

```
number_of_steps = 1 << (# of bits in T)
range_adjust = number_of_steps / (number_of_steps - 1)
range = (range_max - range_min) * range_adjust
range_scale = number_of_steps / range
quantized = round(input * range_scale) - round(range_min * range_scale) +
  numeric_limits<T>::min()
quantized = max(quantized, numeric_limits<T>::min())
quantized = min(quantized, numeric_limits<T>::max())
```

The biggest difference between this and MIN_COMBINED is that the minimum range
is rounded first, before it's subtracted from the rounded value. With
MIN_COMBINED, a small bias is introduced where repeated iterations of quantizing
and dequantizing will introduce a larger and larger error.

One thing to watch out for is that the operator may choose to adjust the
requested minimum and maximum values slightly during the quantization process,
so you should always use the output ports as the range for further calculations.
For example, if the requested minimum and maximum values are close to equal,
they will be separated by a small epsilon value to prevent ill-formed quantized
buffers from being created. Otherwise, you can end up with buffers where all the
quantized values map to the same float value, which causes problems for
operations that have to perform further calculations on them.

##### Args:


*  <b>`input`</b>: A `Tensor` of type `float32`.
*  <b>`min_range`</b>: A `Tensor` of type `float32`.
    The minimum scalar value possibly produced for the input.
*  <b>`max_range`</b>: A `Tensor` of type `float32`.
    The maximum scalar value possibly produced for the input.
*  <b>`T`</b>: A `tf.DType` from: `tf.qint8, tf.quint8, tf.qint16, tf.quint16, tf.qint32`.
*  <b>`mode`</b>: An optional `string` from: `"MIN_COMBINED", "MIN_FIRST"`. Defaults to `"MIN_COMBINED"`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A tuple of `Tensor` objects (output, output_min, output_max).

*  <b>`output`</b>: A `Tensor` of type `T`. The quantized data produced from the float input.
*  <b>`output_min`</b>: A `Tensor` of type `float32`. The actual minimum scalar value used for the output.
*  <b>`output_max`</b>: A `Tensor` of type `float32`. The actual maximum scalar value used for the output.


- - -

### `tf.quantized_concat(concat_dim, values, input_mins, input_maxes, name=None)` {#quantized_concat}

Concatenates quantized tensors along one dimension.

##### Args:


*  <b>`concat_dim`</b>: A `Tensor` of type `int32`.
    0-D.  The dimension along which to concatenate.  Must be in the
    range [0, rank(values)).
*  <b>`values`</b>: A list of at least 2 `Tensor` objects of the same type.
    The `N` Tensors to concatenate. Their ranks and types must match,
    and their sizes must match in all dimensions except `concat_dim`.
*  <b>`input_mins`</b>: A list with the same number of `Tensor` objects as `values` of `Tensor` objects of type `float32`.
    The minimum scalar values for each of the input tensors.
*  <b>`input_maxes`</b>: A list with the same number of `Tensor` objects as `values` of `Tensor` objects of type `float32`.
    The maximum scalar values for each of the input tensors.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A tuple of `Tensor` objects (output, output_min, output_max).

*  <b>`output`</b>: A `Tensor`. Has the same type as `values`. A `Tensor` with the concatenation of values stacked along the
    `concat_dim` dimension.  This tensor's shape matches that of `values` except
    in `concat_dim` where it has the sum of the sizes.
*  <b>`output_min`</b>: A `Tensor` of type `float32`. The float value that the minimum quantized output value represents.
*  <b>`output_max`</b>: A `Tensor` of type `float32`. The float value that the maximum quantized output value represents.


- - -

### `tf.setdiff1d(x, y, index_dtype=tf.int32, name=None)` {#setdiff1d}

Computes the difference between two lists of numbers or strings.

Given a list `x` and a list `y`, this operation returns a list `out` that
represents all values that are in `x` but not in `y`. The returned list `out`
is sorted in the same order that the numbers appear in `x` (duplicates are
preserved). This operation also returns a list `idx` that represents the
position of each `out` element in `x`. In other words:

`out[i] = x[idx[i]] for i in [0, 1, ..., len(out) - 1]`

For example, given this input:

```prettyprint
x = [1, 2, 3, 4, 5, 6]
y = [1, 3, 5]
```

This operation would return:

```prettyprint
out ==> [2, 4, 6]
idx ==> [1, 3, 5]
```

##### Args:


*  <b>`x`</b>: A `Tensor`. 1-D. Values to keep.
*  <b>`y`</b>: A `Tensor`. Must have the same type as `x`. 1-D. Values to remove.
*  <b>`out_idx`</b>: An optional `tf.DType` from: `tf.int32, tf.int64`. Defaults to `tf.int32`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A tuple of `Tensor` objects (out, idx).

*  <b>`out`</b>: A `Tensor`. Has the same type as `x`. 1-D. Values present in `x` but not in `y`.
*  <b>`idx`</b>: A `Tensor` of type `out_idx`. 1-D. Positions of `x` values preserved in `out`.



## Fake quantization
Operations used to help train for better quantization accuracy.

- - -

### `tf.fake_quant_with_min_max_args(inputs, min=None, max=None, name=None)` {#fake_quant_with_min_max_args}

Fake-quantize the 'inputs' tensor, type float to 'outputs' tensor of same type.

Attributes [min; max] define the clamping range for the 'inputs' data.  Op
divides this range into 255 steps (total of 256 values), then replaces each
'inputs' value with the closest of the quantized step values.

Quantization is called fake since the output is still in floating point.

##### Args:


*  <b>`inputs`</b>: A `Tensor` of type `float32`.
*  <b>`min`</b>: An optional `float`. Defaults to `-6`.
*  <b>`max`</b>: An optional `float`. Defaults to `6`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `float32`.


- - -

### `tf.fake_quant_with_min_max_args_gradient(gradients, inputs, min=None, max=None, name=None)` {#fake_quant_with_min_max_args_gradient}

Compute gradients for a FakeQuantWithMinMaxArgs operation.

##### Args:


*  <b>`gradients`</b>: A `Tensor` of type `float32`.
    Backpropagated gradients above the FakeQuantWithMinMaxArgs operation.
*  <b>`inputs`</b>: A `Tensor` of type `float32`.
    Values passed as inputs to the FakeQuantWithMinMaxArgs operation.
*  <b>`min`</b>: An optional `float`. Defaults to `-6`.
*  <b>`max`</b>: An optional `float`. Defaults to `6`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `float32`.
  Backpropagated gradients below the FakeQuantWithMinMaxArgs operation:
  `gradients * (inputs >= min && inputs <= max)`.


- - -

### `tf.fake_quant_with_min_max_vars(inputs, min, max, name=None)` {#fake_quant_with_min_max_vars}

Fake-quantize the 'inputs' tensor of type float and shape `[b, h, w, d]` via

global float scalars `min` and `max` to 'outputs' tensor of same shape as
`inputs`.

[min; max] is the clamping range for the 'inputs' data.  Op divides this range
into 255 steps (total of 256 values), then replaces each 'inputs' value with the
closest of the quantized step values.

This operation has a gradient and thus allows for training `min` and `max` values.

##### Args:


*  <b>`inputs`</b>: A `Tensor` of type `float32`.
*  <b>`min`</b>: A `Tensor` of type `float32`.
*  <b>`max`</b>: A `Tensor` of type `float32`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `float32`.


- - -

### `tf.fake_quant_with_min_max_vars_gradient(gradients, inputs, min, max, name=None)` {#fake_quant_with_min_max_vars_gradient}

Compute gradients for a FakeQuantWithMinMaxVars operation.

##### Args:


*  <b>`gradients`</b>: A `Tensor` of type `float32`.
    Backpropagated gradients above the FakeQuantWithMinMaxVars operation.
*  <b>`inputs`</b>: A `Tensor` of type `float32`.
    Values passed as inputs to the FakeQuantWithMinMaxVars operation.
    min, max: Quantization interval, scalar floats.
*  <b>`min`</b>: A `Tensor` of type `float32`.
*  <b>`max`</b>: A `Tensor` of type `float32`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A tuple of `Tensor` objects (backprops_wrt_input, backprop_wrt_min, backprop_wrt_max).

*  <b>`backprops_wrt_input`</b>: A `Tensor` of type `float32`. Backpropagated gradients w.r.t. inputs:
    `gradients * (inputs >= min && inputs <= max)`.
*  <b>`backprop_wrt_min`</b>: A `Tensor` of type `float32`. Backpropagated gradients w.r.t. min parameter:
    `sum(gradients * (inputs < min))`.
*  <b>`backprop_wrt_max`</b>: A `Tensor` of type `float32`. Backpropagated gradients w.r.t. max parameter:
    `sum(gradients * (inputs > max))`.


- - -

### `tf.fake_quant_with_min_max_vars_per_channel(inputs, min, max, name=None)` {#fake_quant_with_min_max_vars_per_channel}

Fake-quantize the 'inputs' tensor of type float and one of the shapes: `[d]`,

`[b, d]` `[b, h, w, d]` via per-channel floats `min` and `max` of shape `[d]`
to 'outputs' tensor of same shape as `inputs`.

[min; max] is the clamping range for the 'inputs' data in the corresponding
depth channel.  Op divides this range into 255 steps (total of 256 values), then
replaces each 'inputs' value with the closest of the quantized step values.

This operation has a gradient and thus allows for training `min` and `max` values.

##### Args:


*  <b>`inputs`</b>: A `Tensor` of type `float32`.
*  <b>`min`</b>: A `Tensor` of type `float32`.
*  <b>`max`</b>: A `Tensor` of type `float32`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `float32`.


- - -

### `tf.fake_quant_with_min_max_vars_per_channel_gradient(gradients, inputs, min, max, name=None)` {#fake_quant_with_min_max_vars_per_channel_gradient}

Compute gradients for a FakeQuantWithMinMaxVarsPerChannel operation.

##### Args:


*  <b>`gradients`</b>: A `Tensor` of type `float32`.
    Backpropagated gradients above the FakeQuantWithMinMaxVars operation,
    shape one of: `[d]`, `[b, d]`,  `[b, h, w, d]`.
*  <b>`inputs`</b>: A `Tensor` of type `float32`.
    Values passed as inputs to the FakeQuantWithMinMaxVars operation, shape
      same as `gradients`.
    min, max: Quantization interval, floats of shape `[d]`.
*  <b>`min`</b>: A `Tensor` of type `float32`.
*  <b>`max`</b>: A `Tensor` of type `float32`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A tuple of `Tensor` objects (backprops_wrt_input, backprop_wrt_min, backprop_wrt_max).

*  <b>`backprops_wrt_input`</b>: A `Tensor` of type `float32`. Backpropagated gradients w.r.t. inputs, shape same as
    `inputs`:
      `gradients * (inputs >= min && inputs <= max)`.
*  <b>`backprop_wrt_min`</b>: A `Tensor` of type `float32`. Backpropagated gradients w.r.t. min parameter, shape `[d]`:
    `sum_per_d(gradients * (inputs < min))`.
*  <b>`backprop_wrt_max`</b>: A `Tensor` of type `float32`. Backpropagated gradients w.r.t. max parameter, shape `[d]`:
    `sum_per_d(gradients * (inputs > max))`.


