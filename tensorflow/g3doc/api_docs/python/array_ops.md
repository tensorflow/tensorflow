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
    The numeric type to interpret each string in string_tensor as.
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

### `tf.shape(input, name=None)` {#shape}

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

##### Returns:

  A `Tensor` of type `int32`.


- - -

### `tf.size(input, name=None)` {#size}

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

##### Returns:

  A `Tensor` of type `int32`.


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
*  <b>`shape`</b>: A `Tensor` of type `int32`. Defines the shape of the output tensor.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `tensor`.


- - -

### `tf.squeeze(input, squeeze_dims=None, name=None)` {#squeeze}

Removes dimensions of size 1 from the shape of a tensor.

Given a tensor `input`, this operation returns a tensor of the same type with
all dimensions of size 1 removed. If you don't want to remove all size 1
dimensions, you can remove specific size 1 dimensions by specifying
`squeeze_dims`.

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
*  <b>`squeeze_dims`</b>: An optional list of `ints`. Defaults to `[]`.
    If specified, only squeezes the dimensions listed. The dimension
    index starts at 0. It is an error to squeeze a dimension that is not 1.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `input`.
  Contains the same data as `input`, but has one or more dimensions of
  size 1 removed.


- - -

### `tf.expand_dims(input, dim, name=None)` {#expand_dims}

Inserts a dimension of 1 into a tensor's shape.

Given a tensor `input`, this operation inserts a dimension of 1 at the
dimension index `dim` of `input`'s shape. The dimension index `dim` starts at
zero; if you specify a negative number for `dim` it is counted backward from
the end.

This operation is useful if you want to add a batch dimension to a single
element. For example, if you have a single image of shape `[height, width,
channels]`, you can make it a batch of 1 image with `expand_dims(image, 0)`,
which will make the shape `[1, height, width, channels]`.

Other examples:

```prettyprint
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
*  <b>`dim`</b>: A `Tensor` of type `int32`.
    0-D (scalar). Specifies the dimension index at which to
    expand the shape of `input`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `input`.
  Contains the same data as `input`, but its shape has an additional
  dimension of size 1 added.


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

```
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

### `tf.strided_slice(input_, begin, end, strides, begin_mask=0, end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=0, name=None)` {#strided_slice}

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
and `shrink_axis_mask` will have the ith bit corrsponding to
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


```
# 'input' is [[[1, 1, 1], [2, 2, 2]],
#             [[3, 3, 3], [4, 4, 4]],
#             [[5, 5, 5], [6, 6, 6]]]
tf.slice(input, [1, 0, 0], [2, 1, 3], [1, 1, 1]) ==> [[[3, 3, 3]]]
tf.slice(input, [1, 0, 0], [2, 2, 3], [1, 1, 1]) ==> [[[3, 3, 3],
                                                       [4, 4, 4]]]
tf.slice(input, [1, 1, 0], [2, -1, 3], [1, -1, 1]) ==>[[[4, 4, 4],
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
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` the same type as `input`.


- - -

### `tf.split(split_dim, num_split, value, name='split')` {#split}

Splits a tensor into `num_split` tensors along one dimension.

Splits `value` along dimension `split_dim` into `num_split` smaller tensors.
Requires that `num_split` evenly divide `value.shape[split_dim]`.

For example:

```python
# 'value' is a tensor with shape [5, 30]
# Split 'value' into 3 tensors along dimension 1
split0, split1, split2 = tf.split(1, 3, value)
tf.shape(split0) ==> [5, 10]
```

Note: If you are splitting along an axis by the length of that axis, consider
using unpack, e.g.

```python
num_items = t.get_shape()[axis].value
[tf.squeeze(s, [axis]) for s in tf.split(axis, num_items, t)]
```

can be rewritten as

```python
tf.unpack(t, axis=axis)
```

##### Args:


*  <b>`split_dim`</b>: A 0-D `int32` `Tensor`. The dimension along which to split.
    Must be in the range `[0, rank(value))`.
*  <b>`num_split`</b>: A Python integer. The number of ways to split.
*  <b>`value`</b>: The `Tensor` to split.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  `num_split` `Tensor` objects resulting from splitting `value`.


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
*  <b>`multiples`</b>: A `Tensor` of type `int32`.
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
*  <b>`mode`</b>: One of "CONSTANT", "REFLECT", or "SYMMETRIC".
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `tensor`.

##### Raises:


*  <b>`ValueError`</b>: When mode is not one of "CONSTANT", "REFLECT", or "SYMMETRIC".


- - -

### `tf.concat(concat_dim, values, name='concat')` {#concat}

Concatenates tensors along one dimension.

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

### `tf.pack(values, axis=0, name='pack')` {#pack}

Packs a list of rank-`R` tensors into one rank-`(R+1)` tensor.

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

### `tf.unpack(value, num=None, axis=0, name='unpack')` {#unpack}

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

### `tf.reverse_sequence(input, seq_lengths, seq_dim, batch_dim=None, name=None)` {#reverse_sequence}

Reverses variable length slices.

This op first slices `input` along the dimension `batch_dim`, and for each
slice `i`, reverses the first `seq_lengths[i]` elements along
the dimension `seq_dim`.

The elements of `seq_lengths` must obey `seq_lengths[i] < input.dims[seq_dim]`,
and `seq_lengths` must be a vector of length `input.dims[batch_dim]`.

The output slice `i` along dimension `batch_dim` is then given by input
slice `i`, with the first `seq_lengths[i]` slices along dimension
`seq_dim` reversed.

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
*  <b>`seq_lengths`</b>: A `Tensor` of type `int64`.
    1-D with length `input.dims(batch_dim)` and
    `max(seq_lengths) < input.dims(seq_dim)`
*  <b>`seq_dim`</b>: An `int`. The dimension which is partially reversed.
*  <b>`batch_dim`</b>: An optional `int`. Defaults to `0`.
    The dimension along which reversal is performed.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `input`.
  The partially reversed input. It has the same shape as `input`.


- - -

### `tf.reverse(tensor, dims, name=None)` {#reverse}

Reverses specific dimensions of a tensor.

Given a `tensor`, and a `bool` tensor `dims` representing the dimensions
of `tensor`, this operation reverses each dimension i of `tensor` where
`dims[i]` is `True`.

`tensor` can have up to 8 dimensions. The number of dimensions
of `tensor` must equal the number of elements in `dims`. In other words:

`rank(tensor) = size(dims)`

For example:

```prettyprint
# tensor 't' is [[[[ 0,  1,  2,  3],
#                  [ 4,  5,  6,  7],
#                  [ 8,  9, 10, 11]],
#                 [[12, 13, 14, 15],
#                  [16, 17, 18, 19],
#                  [20, 21, 22, 23]]]]
# tensor 't' shape is [1, 2, 3, 4]

# 'dims' is [False, False, False, True]
reverse(t, dims) ==> [[[[ 3,  2,  1,  0],
                        [ 7,  6,  5,  4],
                        [ 11, 10, 9, 8]],
                       [[15, 14, 13, 12],
                        [19, 18, 17, 16],
                        [23, 22, 21, 20]]]]

# 'dims' is [False, True, False, False]
reverse(t, dims) ==> [[[[12, 13, 14, 15],
                        [16, 17, 18, 19],
                        [20, 21, 22, 23]
                       [[ 0,  1,  2,  3],
                        [ 4,  5,  6,  7],
                        [ 8,  9, 10, 11]]]]

# 'dims' is [False, False, True, False]
reverse(t, dims) ==> [[[[8, 9, 10, 11],
                        [4, 5, 6, 7],
                        [0, 1, 2, 3]]
                       [[20, 21, 22, 23],
                        [16, 17, 18, 19],
                        [12, 13, 14, 15]]]]
```

##### Args:


*  <b>`tensor`</b>: A `Tensor`. Must be one of the following types: `uint8`, `int8`, `int32`, `bool`, `half`, `float32`, `float64`, `complex64`, `complex128`.
    Up to 8-D.
*  <b>`dims`</b>: A `Tensor` of type `bool`. 1-D. The dimensions to reverse.
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
    `patch_sizes_eff = patch_sizes + (patch_sizes - 1) * (rates - 1), followed by
    subsampling them spatially by a factor of `rates`.
*  <b>`padding`</b>: A `string` from: `"SAME", "VALID"`.
    The type of padding algorithm to use.

    We specify the size-related attributes as:

          ksizes = [1, ksize_rows, ksize_cols, 1]
          strides = [1, strides_rows, strides_cols, 1]
          rates = [1, rates_rows, rates_cols, 1]

*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `images`.
  4-D Tensor with shape `[batch, out_rows, out_cols, ksize_rows *
  ksize_cols * depth]` containing image patches with size
  `ksize_rows x ksize_cols x depth` vectorized in the "depth" dimension.


- - -

### `tf.space_to_batch(input, paddings, block_size, name=None)` {#space_to_batch}

SpaceToBatch for 4-D tensors of type T.

Zero-pads and then rearranges (permutes) blocks of spatial data into batch.
More specifically, this op outputs a copy of the input tensor where values from
the `height` and `width` dimensions are moved to the `batch` dimension. After
the zero-padding, both `height` and `width` of the input must be divisible by the
block size.

##### Args:


*  <b>`input`</b>: A `Tensor`. 4-D with shape `[batch, height, width, depth]`.
*  <b>`paddings`</b>: A `Tensor` of type `int32`.
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

### `tf.batch_to_space(input, crops, block_size, name=None)` {#batch_to_space}

BatchToSpace for 4-D tensors of type T.

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
*  <b>`crops`</b>: A `Tensor` of type `int32`.
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
  * The width the output tensor is `input_depth * block_size`, whereas the
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

    # Scalar indices
    output[:, ..., :] = params[indices, :, ... :]

    # Vector indices
    output[i, :, ..., :] = params[indices[i], :, ... :]

    # Higher rank indices
    output[i, ..., j, :, ... :] = params[indices[i, ..., j], :, ..., :]

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

`params` is a Tensor of rank `R` and `indices` is a Tensor of rank `M`.

`indices` must be integer tensor, containing indices into `params`.
It must be shape `[d_0, ..., d_N, R]` where `0 < R <= M`.

The innermost dimension of `indices` (with length `R`) corresponds to
indices into elements (if `R = M`) or slices (if `R < M`) along the `N`th
dimension of `params`.

Produces an output tensor with shape

    [d_0, ..., d_{n-1}, params.shape[R], ..., params.shape[M-1]].

Some examples below.

Simple indexing into a matrix:

    indices = [[0, 0], [1, 1]]
    params = [['a', 'b'], ['c', 'd']]
    output = ['a', 'd']

Slice indexing into a matrix:

    indices = [[1], [0]]
    params = [['a', 'b'], ['c', 'd']]
    output = [['c', 'd'], ['a', 'b']]

Indexing into a 3-tensor:

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

Batched indexing into a matrix:

    indices = [[[0, 0]], [[0, 1]]]
    params = [['a', 'b'], ['c', 'd']]
    output = [['a'], ['b']]

Batched slice indexing into a matrix:

    indices = [[[1]], [[0]]]
    params = [['a', 'b'], ['c', 'd']]
    output = [[['c', 'd']], [['a', 'b']]]

Batched indexing into a 3-tensor:

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

##### Args:


*  <b>`params`</b>: A `Tensor`. `M-D`.  The tensor from which to gather values.
*  <b>`indices`</b>: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    `(N+1)-D`.  Index tensor having shape `[d_0, ..., d_N, R]`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `params`.
  `(N+M-R)-D`.  Values from `params` gathered from indices given by
  `indices`.


- - -

### `tf.dynamic_partition(data, partitions, num_partitions, name=None)` {#dynamic_partition}

Partitions `data` into `num_partitions` tensors using indices from `partitions`.

For each index tuple `js` of size `partitions.ndim`, the slice `data[js, ...]`
becomes part of `outputs[partitions[js]]`.  The slices with `partitions[js] = i`
are placed in `outputs[i]` in lexicographic order of `js`, and the first
dimension of `outputs[i]` is the number of entries in `partitions` equal to `i`.
In detail,

    outputs[i].shape = [sum(partitions == i)] + data.shape[partitions.ndim:]

    outputs[i] = pack([data[js, ...] for js if partitions[js] == i])

`data.shape` must start with `partitions.shape`.

For example:

    # Scalar partitions
    partitions = 1
    num_partitions = 2
    data = [10, 20]
    outputs[0] = []  # Empty with shape [0, 2]
    outputs[1] = [[10, 20]]

    # Vector partitions
    partitions = [0, 0, 1, 1, 0]
    num_partitions = 2
    data = [10, 20, 30, 40, 50]
    outputs[0] = [10, 20, 50]
    outputs[1] = [30, 40]

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

    merged[indices[m][i, ..., j], ...] = data[m][i, ..., j, ...]

For example, if each `indices[m]` is scalar or vector, we have

    # Scalar indices
    merged[indices[m], ...] = data[m][...]

    # Vector indices
    merged[indices[m][i], ...] = data[m][i, ...]

Each `data[i].shape` must start with the corresponding `indices[i].shape`,
and the rest of `data[i].shape` must be constant w.r.t. `i`.  That is, we
must have `data[i].shape = indices[i].shape + constant`.  In terms of this
`constant`, the output shape is

    merged.shape = [max(indices)] + constant

Values are merged in order, so if an index appears in both `indices[m][i]` and
`indices[n][j]` for `(m,i) < (n,j)` the slice `data[n][j]` will appear in the
merged result.

For example:

    indices[0] = 6
    indices[1] = [4, 1]
    indices[2] = [[5, 2], [0, 3]]
    data[0] = [61, 62]
    data[1] = [[41, 42], [11, 12]]
    data[2] = [[[51, 52], [21, 22]], [[1, 2], [31, 32]]]
    merged = [[1, 2], [11, 12], [21, 22], [31, 32], [41, 42],
              [51, 52], [61, 62]]

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../../images/DynamicStitch.png" alt>
</div>

##### Args:


*  <b>`indices`</b>: A list of at least 2 `Tensor` objects of type `int32`.
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
mask = [True, False, True, False]
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

  Tensor populated by entries in `tensor` corresponding to `True` values in
    `mask`.

##### Raises:


*  <b>`ValueError`</b>: If shapes do not conform.


*  <b>`Examples`</b>: 

```python
# 2-D example
tensor = [[1, 2], [3, 4], [5, 6]]
mask = [True, False, True]
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
value `tf.float32`

Note: If a non-numeric data type output is desired (tf.string, tf.bool, etc.),
both `on_value` and `off_value` _must_ be provided to `one_hot`

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

```
  output =
  [5.0 0.0 0.0]  // one_hot(0)
  [0.0 0.0 5.0]  // one_hot(2)
  [0.0 0.0 0.0]  // one_hot(-1)
  [0.0 5.0 0.0]  // one_hot(1)
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

```
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

```
  indices = [0, 1, 2]
  depth = 3
```

The output will be

```
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



## Other Functions and Classes
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

### `tf.contrib.graph_editor.copy(sgv, dst_graph=None, dst_scope='', src_scope='')` {#copy}

Copy a subgraph.

##### Args:


*  <b>`sgv`</b>: the source subgraph-view. This argument is converted to a subgraph
    using the same rules than the function subgraph.make_view.
*  <b>`dst_graph`</b>: the destination graph.
*  <b>`dst_scope`</b>: the destination scope.
*  <b>`src_scope`</b>: the source scope.

##### Returns:

  the subgraph view of the copied subgraph.

##### Raises:


*  <b>`TypeError`</b>: if dst_graph is not a tf.Graph.
*  <b>`StandardError`</b>: if sgv cannot be converted to a SubGraphView using
    the same rules than the function subgraph.make_view.


- - -

### `tf.shape_n(input, name=None)` {#shape_n}

Returns shape of tensors.

This operation returns N 1-D integer tensors representing shape of `input[i]s`.

##### Args:


*  <b>`input`</b>: A list of at least 1 `Tensor` objects of the same type.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A list with the same number of `Tensor` objects as `input` of `Tensor` objects of type `int32`.


- - -

### `tf.unique_with_counts(x, name=None)` {#unique_with_counts}

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
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A tuple of `Tensor` objects (y, idx, count).

*  <b>`y`</b>: A `Tensor`. Has the same type as `x`. 1-D.
*  <b>`idx`</b>: A `Tensor` of type `int32`. 1-D.
*  <b>`count`</b>: A `Tensor` of type `int32`. 1-D.


