/* Copyright 2015 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/op.h"

namespace tensorflow {

REGISTER_OP("SparseAddGrad")
    .Input("backprop_val_grad: T")
    .Input("a_indices: int64")
    .Input("b_indices: int64")
    .Input("sum_indices: int64")
    .Output("a_val_grad: T")
    .Output("b_val_grad: T")
    .Attr("T: numbertype")
    .Doc(R"doc(
The gradient operator for the SparseAdd op.

The SparseAdd op calculates A + B, where A, B, and the sum are all represented
as `SparseTensor` objects.  This op takes in the upstream gradient w.r.t.
non-empty values of the sum, and outputs the gradients w.r.t. the non-empty
values of A and B.

backprop_val_grad: 1-D with shape `[nnz(sum)]`.  The gradient with respect to
  the non-empty values of the sum.
a_indices: 2-D.  The `indices` of the `SparseTensor` A, size `[nnz(A), ndims]`.
b_indices: 2-D.  The `indices` of the `SparseTensor` B, size `[nnz(B), ndims]`.
sum_indices: 2-D.  The `indices` of the sum `SparseTensor`, size
  `[nnz(sum), ndims]`.
a_val_grad: 1-D with shape `[nnz(A)]`. The gradient with respect to the
  non-empty values of A.
b_val_grad: 1-D with shape `[nnz(B)]`. The gradient with respect to the
  non-empty values of B.
)doc");

REGISTER_OP("SparseAdd")
    .Input("a_indices: int64")
    .Input("a_values: T")
    .Input("a_shape: int64")
    .Input("b_indices: int64")
    .Input("b_values: T")
    .Input("b_shape: int64")
    .Input("thresh: Treal")
    .Output("sum_indices: int64")
    .Output("sum_values: T")
    .Output("sum_shape: int64")
    .Attr("T: numbertype")
    .Attr("Treal: realnumbertype")
    .Doc(R"doc(
Adds two `SparseTensor` objects to produce another `SparseTensor`.

The input `SparseTensor` objects' indices are assumed ordered in standard
lexicographic order.  If this is not the case, before this step run
`SparseReorder` to restore index ordering.

By default, if two values sum to zero at some index, the output `SparseTensor`
would still include that particular location in its index, storing a zero in the
corresponding value slot.  To override this, callers can specify `thresh`,
indicating that if the sum has a magnitude strictly smaller than `thresh`, its
corresponding value and index would then not be included.  In particular,
`thresh == 0` (default) means everything is kept and actual thresholding happens
only for a positive value.

In the following shapes, `nnz` is the count after taking `thresh` into account.

a_indices: 2-D.  The `indices` of the first `SparseTensor`, size `[nnz, ndims]` Matrix.
a_values: 1-D.  The `values` of the first `SparseTensor`, size `[nnz]` Vector.
a_shape: 1-D.  The `shape` of the first `SparseTensor`, size `[ndims]` Vector.
b_indices: 2-D.  The `indices` of the second `SparseTensor`, size `[nnz, ndims]` Matrix.
b_values: 1-D.  The `values` of the second `SparseTensor`, size `[nnz]` Vector.
b_shape: 1-D.  The `shape` of the second `SparseTensor`, size `[ndims]` Vector.
thresh: 0-D.  The magnitude threshold that determines if an output value/index
pair takes space.
)doc");

REGISTER_OP("SparseTensorDenseMatMul")
    .Input("a_indices: int64")
    .Input("a_values: T")
    .Input("a_shape: int64")
    .Input("b: T")
    .Output("product: T")
    .Attr("T: type")
    .Attr("adjoint_a: bool = false")
    .Attr("adjoint_b: bool = false")
    .Doc(R"doc(
Multiply SparseTensor (of rank 2) "A" by dense matrix "B".

No validity checking is performed on the indices of A.  However, the following
input format is recommended for optimal behavior:

if adjoint_a == false:
  A should be sorted in lexicographically increasing order.  Use SparseReorder
  if you're not sure.
if adjoint_a == true:
  A should be sorted in order of increasing dimension 1 (i.e., "column major"
  order instead of "row major" order).

a_indices: 2-D.  The `indices` of the `SparseTensor`, size `[nnz, 2]` Matrix.
a_values: 1-D.  The `values` of the `SparseTensor`, size `[nnz]` Vector.
a_shape: 1-D.  The `shape` of the `SparseTensor`, size `[2]` Vector.
b: 2-D.  A dense Matrix.
adjoint_a: Use the adjoint of A in the matrix multiply.  If A is complex, this
  is transpose(conj(A)).  Otherwise it's transpose(A).
adjoint_b: Use the adjoint of B in the matrix multiply.  If B is complex, this
  is transpose(conj(B)).  Otherwise it's transpose(B).
)doc");

REGISTER_OP("SerializeSparse")
    .Input("sparse_indices: int64")
    .Input("sparse_values: T")
    .Input("sparse_shape: int64")
    .Attr("T: type")
    .Output("serialized_sparse: string")
    .Doc(R"doc(
Serialize a `SparseTensor` into a string 3-vector (1-D `Tensor`) object.

sparse_indices: 2-D.  The `indices` of the `SparseTensor`.
sparse_values: 1-D.  The `values` of the `SparseTensor`.
sparse_shape: 1-D.  The `shape` of the `SparseTensor`.
)doc");

REGISTER_OP("SerializeManySparse")
    .Input("sparse_indices: int64")
    .Input("sparse_values: T")
    .Input("sparse_shape: int64")
    .Attr("T: type")
    .Output("serialized_sparse: string")
    .Doc(R"doc(
Serialize an `N`-minibatch `SparseTensor` into an `[N, 3]` string `Tensor`.

The `SparseTensor` must have rank `R` greater than 1, and the first dimension
is treated as the minibatch dimension.  Elements of the `SparseTensor`
must be sorted in increasing order of this first dimension.  The serialized
`SparseTensor` objects going into each row of `serialized_sparse` will have
rank `R-1`.

The minibatch size `N` is extracted from `sparse_shape[0]`.

sparse_indices: 2-D.  The `indices` of the minibatch `SparseTensor`.
sparse_values: 1-D.  The `values` of the minibatch `SparseTensor`.
sparse_shape: 1-D.  The `shape` of the minibatch `SparseTensor`.
)doc");

REGISTER_OP("DeserializeManySparse")
    .Input("serialized_sparse: string")
    .Attr("dtype: type")
    .Output("sparse_indices: int64")
    .Output("sparse_values: dtype")
    .Output("sparse_shape: int64")
    .Doc(R"doc(
Deserialize and concatenate `SparseTensors` from a serialized minibatch.

The input `serialized_sparse` must be a string matrix of shape `[N x 3]` where
`N` is the minibatch size and the rows correspond to packed outputs of
`SerializeSparse`.  The ranks of the original `SparseTensor` objects
must all match.  When the final `SparseTensor` is created, it has rank one
higher than the ranks of the incoming `SparseTensor` objects
(they have been concatenated along a new row dimension).

The output `SparseTensor` object's shape values for all dimensions but the
first are the max across the input `SparseTensor` objects' shape values
for the corresponding dimensions.  Its first shape value is `N`, the minibatch
size.

The input `SparseTensor` objects' indices are assumed ordered in
standard lexicographic order.  If this is not the case, after this
step run `SparseReorder` to restore index ordering.

For example, if the serialized input is a `[2 x 3]` matrix representing two
original `SparseTensor` objects:

    index = [ 0]
            [10]
            [20]
    values = [1, 2, 3]
    shape = [50]

and

    index = [ 2]
            [10]
    values = [4, 5]
    shape = [30]

then the final deserialized `SparseTensor` will be:

    index = [0  0]
            [0 10]
            [0 20]
            [1  2]
            [1 10]
    values = [1, 2, 3, 4, 5]
    shape = [2 50]

serialized_sparse: 2-D, The `N` serialized `SparseTensor` objects.
  Must have 3 columns.
dtype: The `dtype` of the serialized `SparseTensor` objects.
)doc");

REGISTER_OP("SparseToDense")
    .Input("sparse_indices: Tindices")
    .Input("output_shape: Tindices")
    .Input("sparse_values: T")
    .Input("default_value: T")
    .Attr("validate_indices: bool = true")
    .Attr("T: type")
    .Output("dense: T")
    .Attr("Tindices: {int32, int64}")
    .Doc(R"doc(
Converts a sparse representation into a dense tensor.

Builds an array `dense` with shape `output_shape` such that

```prettyprint
# If sparse_indices is scalar
dense[i] = (i == sparse_indices ? sparse_values : default_value)

# If sparse_indices is a vector, then for each i
dense[sparse_indices[i]] = sparse_values[i]

# If sparse_indices is an n by d matrix, then for each i in [0, n)
dense[sparse_indices[i][0], ..., sparse_indices[i][d-1]] = sparse_values[i]
```

All other values in `dense` are set to `default_value`.  If `sparse_values` is a
scalar, all sparse indices are set to this single value.

Indices should be sorted in lexicographic order, and indices must not
contain any repeats. If `validate_indices` is true, these properties
are checked during execution.

sparse_indices: 0-D, 1-D, or 2-D.  `sparse_indices[i]` contains the complete
  index where `sparse_values[i]` will be placed.
output_shape: 1-D.  Shape of the dense output tensor.
sparse_values: 1-D.  Values corresponding to each row of `sparse_indices`,
  or a scalar value to be used for all sparse indices.
default_value: Scalar value to set for indices not specified in
  `sparse_indices`.
validate_indices: If true, indices are checked to make sure they are sorted in
  lexicographic order and that there are no repeats.
dense: Dense output tensor of shape `output_shape`.
)doc");

REGISTER_OP("SparseConcat")
    .Input("indices: N * int64")
    .Input("values: N * T")
    .Input("shapes: N * int64")
    .Output("output_indices: int64")
    .Output("output_values: T")
    .Output("output_shape: int64")
    .Attr("concat_dim: int >= 0")
    .Attr("N: int >= 2")
    .Attr("T: type")
    .Doc(R"doc(
Concatenates a list of `SparseTensor` along the specified dimension.

Concatenation is with respect to the dense versions of these sparse tensors.
It is assumed that each input is a `SparseTensor` whose elements are ordered
along increasing dimension number.

All inputs' shapes must match, except for the concat dimension.  The
`indices`, `values`, and `shapes` lists must have the same length.

The output shape is identical to the inputs', except along the concat
dimension, where it is the sum of the inputs' sizes along that dimension.

The output elements will be resorted to preserve the sort order along
increasing dimension number.

This op runs in `O(M log M)` time, where `M` is the total number of non-empty
values across all inputs. This is due to the need for an internal sort in
order to concatenate efficiently across an arbitrary dimension.

For example, if `concat_dim = 1` and the inputs are

    sp_inputs[0]: shape = [2, 3]
    [0, 2]: "a"
    [1, 0]: "b"
    [1, 1]: "c"

    sp_inputs[1]: shape = [2, 4]
    [0, 1]: "d"
    [0, 2]: "e"

then the output will be

    shape = [2, 7]
    [0, 2]: "a"
    [0, 4]: "d"
    [0, 5]: "e"
    [1, 0]: "b"
    [1, 1]: "c"

Graphically this is equivalent to doing

    [    a] concat [  d e  ] = [    a   d e  ]
    [b c  ]        [       ]   [b c          ]

indices: 2-D.  Indices of each input `SparseTensor`.
values: 1-D.  Non-empty values of each `SparseTensor`.
shapes: 1-D.  Shapes of each `SparseTensor`.
output_indices: 2-D.  Indices of the concatenated `SparseTensor`.
output_values: 1-D.  Non-empty values of the concatenated `SparseTensor`.
output_shape: 1-D.  Shape of the concatenated `SparseTensor`.
concat_dim: Dimension to concatenate along.
)doc");

REGISTER_OP("SparseSplit")
    .Input("split_dim: int64")
    .Input("indices: int64")
    .Input("values: T")
    .Input("shape: int64")
    .Output("output_indices: num_split * int64")
    .Output("output_values:  num_split * T")
    .Output("output_shape:   num_split * int64")
    .Attr("num_split: int >= 1")
    .Attr("T: type")
    .Doc(R"doc(
Split a `SparseTensor` into `num_split` tensors along one dimension.

If the `shape[split_dim]` is not an integer multiple of `num_split`. Slices
`[0 : shape[split_dim] % num_split]` gets one extra dimension.
For example, if `split_dim = 1` and `num_split = 2` and the input is

    input_tensor = shape = [2, 7]
    [    a   d e  ]
    [b c          ]

Graphically the output tensors are:

    output_tensor[0] = shape = [2, 4]
    [    a  ]
    [b c    ]

    output_tensor[1] = shape = [2, 3]
    [ d e  ]
    [      ]

split_dim: 0-D.  The dimension along which to split.  Must be in the range
  `[0, rank(shape))`.
num_split: The number of ways to split.
indices: 2-D tensor represents the indices of the sparse tensor.
values: 1-D tensor represents the values of the sparse tensor.
shape: 1-D. tensor represents the shape of the sparse tensor.
output indices: A list of 1-D tensors represents the indices of the output
sparse tensors.
output_values: A list of 1-D tensors represents the values of the output sparse
  tensors.
output_shape: A list of 1-D tensors represents the shape of the output sparse
  tensors.
)doc");

REGISTER_OP("SparseReorder")
    .Input("input_indices: int64")
    .Input("input_values: T")
    .Input("input_shape: int64")
    .Output("output_indices: int64")
    .Output("output_values: T")
    .Attr("T: type")
    .Doc(R"doc(
Reorders a SparseTensor into the canonical, row-major ordering.

Note that by convention, all sparse ops preserve the canonical ordering along
increasing dimension number. The only time ordering can be violated is during
manual manipulation of the indices and values vectors to add entries.

Reordering does not affect the shape of the SparseTensor.

If the tensor has rank `R` and `N` non-empty values, `input_indices` has
shape `[N, R]`, input_values has length `N`, and input_shape has length `R`.

input_indices: 2-D.  `N x R` matrix with the indices of non-empty values in a
  SparseTensor, possibly not in canonical ordering.
input_values: 1-D.  `N` non-empty values corresponding to `input_indices`.
input_shape: 1-D.  Shape of the input SparseTensor.
output_indices: 2-D.  `N x R` matrix with the same indices as input_indices, but
  in canonical row-major ordering.
output_values: 1-D.  `N` non-empty values corresponding to `output_indices`.
)doc");

REGISTER_OP("SparseTensorDenseAdd")
    .Input("a_indices: Tindices")
    .Input("a_values: T")
    .Input("a_shape: Tindices")
    .Input("b: T")
    .Output("output: T")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Doc(R"doc(
Adds up a `SparseTensor` and a dense `Tensor`, producing a dense `Tensor`.

This Op does not require `a_indices` be sorted in standard lexicographic order.

a_indices: 2-D.  The `indices` of the `SparseTensor`, with shape `[nnz, ndims]`.
a_values: 1-D.  The `values` of the `SparseTensor`, with shape `[nnz]`.
a_shape: 1-D.  The `shape` of the `SparseTensor`, with shape `[ndims]`.
b: `ndims`-D Tensor.  With shape `a_shape`.
)doc");

REGISTER_OP("SparseReduceSum")
    .Input("input_indices: int64")
    .Input("input_values: T")
    .Input("input_shape: int64")
    .Input("reduction_axes: int32")
    .Attr("keep_dims: bool = False")
    .Output("output: T")
    .Attr("T: numbertype")
    .Doc(R"doc(
Computes the sum of elements across dimensions of a SparseTensor.

This Op takes a SparseTensor and is the sparse counterpart to
`tf.reduce_sum()`.  In particular, this Op also returns a dense `Tensor`
instead of a sparse one.

Reduces `sp_input` along the dimensions given in `reduction_axes`.  Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`reduction_axes`. If `keep_dims` is true, the reduced dimensions are retained
with length 1.

If `reduction_axes` has no entries, all dimensions are reduced, and a tensor
with a single element is returned.

input_indices: 2-D.  `N x R` matrix with the indices of non-empty values in a
  SparseTensor, possibly not in canonical ordering.
input_values: 1-D.  `N` non-empty values corresponding to `input_indices`.
input_shape: 1-D.  Shape of the input SparseTensor.
reduction_axes: 1-D.  Length-`K` vector containing the reduction axes.
keep_dims: If true, retain reduced dimensions with length 1.
output: `R-K`-D.  The reduced Tensor.
)doc");

REGISTER_OP("SparseDenseCwiseMul")
    .Input("sp_indices: int64")
    .Input("sp_values: T")
    .Input("sp_shape: int64")
    .Input("dense: T")
    .Output("output: T")
    .Attr("T: numbertype")
    .Doc(R"doc(
Component-wise multiplies a SparseTensor by a dense Tensor.

*Limitation*: this Op only broadcasts the dense side to the sparse side, but not
the other direction.

sp_indices: 2-D.  `N x R` matrix with the indices of non-empty values in a
  SparseTensor, possibly not in canonical ordering.
sp_values: 1-D.  `N` non-empty values corresponding to `sp_indices`.
sp_shape: 1-D.  Shape of the input SparseTensor.
dense: `R`-D.  The dense Tensor operand.
output: 1-D.  The `N` values that are operated on.
)doc");

REGISTER_OP("SparseDenseCwiseDiv")
    .Input("sp_indices: int64")
    .Input("sp_values: T")
    .Input("sp_shape: int64")
    .Input("dense: T")
    .Output("output: T")
    .Attr("T: numbertype")
    .Doc(R"doc(
Component-wise divides a SparseTensor by a dense Tensor.

*Limitation*: this Op only broadcasts the dense side to the sparse side, but not
the other direction.

sp_indices: 2-D.  `N x R` matrix with the indices of non-empty values in a
  SparseTensor, possibly not in canonical ordering.
sp_values: 1-D.  `N` non-empty values corresponding to `sp_indices`.
sp_shape: 1-D.  Shape of the input SparseTensor.
dense: `R`-D.  The dense Tensor operand.
output: 1-D.  The `N` values that are operated on.
)doc");

}  // namespace tensorflow
