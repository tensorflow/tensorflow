#include "tensorflow/core/framework/op.h"

namespace tensorflow {

REGISTER_OP("SparseToDense")
    .Input("sparse_indices: Tindices")
    .Input("output_shape: Tindices")
    .Input("sparse_values: T")
    .Input("default_value: T")
    .Output("dense: T")
    .Attr("T: type")
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

sparse_indices: 0-D, 1-D, or 2-D.  `sparse_indices[i]` contains the complete
  index where `sparse_values[i]` will be placed.
output_shape: 1-D.  Shape of the dense output tensor.
sparse_values: 1-D.  Values corresponding to each row of `sparse_indices`,
  or a scalar value to be used for all sparse indices.
default_value: Scalar value to set for indices not specified in
  `sparse_indices`.
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

}  // namespace tensorflow
