/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

namespace {

Status SparseSparseMinOrMaxShapeFn(InferenceContext* c) {
  ShapeHandle unused;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &unused));  // a_indices
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &unused));  // a_values
  TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &unused));  // a_shape
  TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 2, &unused));  // b_indices
  TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 1, &unused));  // b_values
  TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 1, &unused));  // b_shape
  c->set_output(0, c->Matrix(InferenceContext::kUnknownDim,
                             InferenceContext::kUnknownDim));
  c->set_output(1, c->Vector(InferenceContext::kUnknownDim));
  return Status::OK();
}

}  // namespace

REGISTER_OP("SparseAddGrad")
    .Input("backprop_val_grad: T")
    .Input("a_indices: int64")
    .Input("b_indices: int64")
    .Input("sum_indices: int64")
    .Output("a_val_grad: T")
    .Output("b_val_grad: T")
    .Attr("T: numbertype")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle a_indices;
      ShapeHandle b_indices;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &a_indices));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &b_indices));
      c->set_output(0, c->Vector(c->Dim(a_indices, 0)));
      c->set_output(1, c->Vector(c->Dim(b_indices, 0)));
      return Status::OK();
    })
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
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle a_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &a_shape));
      c->set_output(
          0, c->Matrix(InferenceContext::kUnknownDim, c->Dim(a_shape, 0)));
      c->set_output(1, c->Vector(InferenceContext::kUnknownDim));
      c->set_output(2, a_shape);
      return Status::OK();
    })
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
    .Input("a_indices: Tindices")
    .Input("a_values: T")
    .Input("a_shape: int64")
    .Input("b: T")
    .Output("product: T")
    .Attr("T: type")
    .Attr("Tindices: {int32,int64} = DT_INT64")
    .Attr("adjoint_a: bool = false")
    .Attr("adjoint_b: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      DimensionHandle unused_dim;
      ShapeHandle unused;
      ShapeHandle b;
      ShapeHandle a_shape;
      ShapeHandle a_shape_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &unused));  // a_indices
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &unused));  // a_values
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(2, &a_shape));
      TF_RETURN_IF_ERROR(c->WithRank(a_shape, 2, &a_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 2, &b));

      bool adjoint_a;
      bool adjoint_b;
      TF_RETURN_IF_ERROR(c->GetAttr("adjoint_a", &adjoint_a));
      TF_RETURN_IF_ERROR(c->GetAttr("adjoint_b", &adjoint_b));

      DimensionHandle output_right = c->Dim(b, adjoint_b ? 0 : 1);
      DimensionHandle output_left = c->Dim(a_shape, adjoint_a ? 1 : 0);
      DimensionHandle inner_left = c->Dim(a_shape, adjoint_a ? 0 : 1);
      DimensionHandle inner_right = c->Dim(b, adjoint_b ? 1 : 0);
      TF_RETURN_IF_ERROR(c->Merge(inner_left, inner_right, &unused_dim));
      c->set_output(0, c->Matrix(output_left, output_right));
      return Status::OK();
    })
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
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &unused));
      c->set_output(0, c->Vector(3));
      return Status::OK();
    })
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
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &unused));
      c->set_output(0, c->Matrix(InferenceContext::kUnknownDim, 3));
      return Status::OK();
    })
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
    .SetShapeFn([](InferenceContext* c) {
      // serialized sparse is [?,3] matrix.
      ShapeHandle serialized_sparse;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &serialized_sparse));
      DimensionHandle unused;
      TF_RETURN_IF_ERROR(
          c->WithValue(c->Dim(serialized_sparse, 1), 3, &unused));

      c->set_output(0, c->Matrix(InferenceContext::kUnknownDim,
                                 InferenceContext::kUnknownDim));
      c->set_output(1, c->Vector(InferenceContext::kUnknownDim));
      c->set_output(2, c->Vector(InferenceContext::kUnknownDim));
      return Status::OK();
    })
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
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle out;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(1, &out));
      c->set_output(0, out);
      return Status::OK();
    })
    .Doc(R"doc(
Converts a sparse representation into a dense tensor.

Builds an array `dense` with shape `output_shape` such that

```
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
    .Attr("concat_dim: int")
    .Attr("N: int >= 2")
    .Attr("T: type")
    .SetShapeFn([](InferenceContext* c) {
      // These accumulates the sum.
      DimensionHandle output_row_count = c->MakeDim(0ll);

      // These are only merged.
      DimensionHandle output_ind_cols = c->UnknownDim();
      ShapeHandle output_shape = c->UnknownShape();

      const int n = c->num_inputs() / 3;
      for (int i = 0; i < n; i++) {
        ShapeHandle ind;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 2, &ind));
        ShapeHandle val;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(i + n), 1, &val));
        ShapeHandle shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(i + 2 * n), 1, &shape));

        // Add to output_ind_rows.
        DimensionHandle num_dim;
        TF_RETURN_IF_ERROR(c->Merge(c->Dim(ind, 0), c->Dim(val, 0), &num_dim));
        TF_RETURN_IF_ERROR(
            c->Add(output_row_count, num_dim, &output_row_count));

        // Merge into output_ind_cols and output_shape.
        TF_RETURN_IF_ERROR(
            c->Merge(output_ind_cols, c->Dim(ind, 1), &output_ind_cols));
        TF_RETURN_IF_ERROR(c->Merge(output_shape, shape, &output_shape));
      }

      c->set_output(0, c->Matrix(output_row_count, output_ind_cols));
      c->set_output(1, c->Vector(output_row_count));
      c->set_output(2, output_shape);
      return Status::OK();
    })
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
concat_dim: Dimension to concatenate along. Must be in range [-rank, rank),
    where rank is the number of dimensions in each input `SparseTensor`.
)doc");

REGISTER_OP("SparseCross")
    .Input("indices: N * int64")
    .Input("values: sparse_types")
    .Input("shapes: N * int64")
    .Input("dense_inputs: dense_types")
    .Output("output_indices: int64")
    .Output("output_values: out_type")
    .Output("output_shape: int64")
    .Attr("N: int >= 0")
    .Attr("hashed_output: bool")
    .Attr("num_buckets: int >= 0")
    .Attr("hash_key: int")
    .Attr("sparse_types: list({int64, string}) >= 0")
    .Attr("dense_types: list({int64, string}) >= 0")
    .Attr("out_type: {int64, string}")
    .Attr("internal_type: {int64, string}")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Matrix(c->UnknownDim(), 2));
      c->set_output(1, c->Vector(c->UnknownDim()));
      c->set_output(2, c->Vector(2));
      return Status::OK();
    })
    .Doc(R"doc(
Generates sparse cross from a list of sparse and dense tensors.

The op takes two lists, one of 2D `SparseTensor` and one of 2D `Tensor`, each
representing features of one feature column. It outputs a 2D `SparseTensor` with
the batchwise crosses of these features.

For example, if the inputs are

    inputs[0]: SparseTensor with shape = [2, 2]
    [0, 0]: "a"
    [1, 0]: "b"
    [1, 1]: "c"

    inputs[1]: SparseTensor with shape = [2, 1]
    [0, 0]: "d"
    [1, 0]: "e"

    inputs[2]: Tensor [["f"], ["g"]]

then the output will be

    shape = [2, 2]
    [0, 0]: "a_X_d_X_f"
    [1, 0]: "b_X_e_X_g"
    [1, 1]: "c_X_e_X_g"

if hashed_output=true then the output will be

    shape = [2, 2]
    [0, 0]: FingerprintCat64(
                Fingerprint64("f"), FingerprintCat64(
                    Fingerprint64("d"), Fingerprint64("a")))
    [1, 0]: FingerprintCat64(
                Fingerprint64("g"), FingerprintCat64(
                    Fingerprint64("e"), Fingerprint64("b")))
    [1, 1]: FingerprintCat64(
                Fingerprint64("g"), FingerprintCat64(
                    Fingerprint64("e"), Fingerprint64("c")))

indices: 2-D.  Indices of each input `SparseTensor`.
values: 1-D.   values of each `SparseTensor`.
shapes: 1-D.   Shapes of each `SparseTensor`.
dense_inputs: 2-D.    Columns represented by dense `Tensor`.
hashed_output: If true, returns the hash of the cross instead of the string.
  This will allow us avoiding string manipulations.
num_buckets: It is used if hashed_output is true.
  output = hashed_value%num_buckets if num_buckets > 0 else hashed_value.
hash_key: Specify the hash_key that will be used by the `FingerprintCat64`
  function to combine the crosses fingerprints.
output_indices: 2-D.  Indices of the concatenated `SparseTensor`.
output_values: 1-D.  Non-empty values of the concatenated or hashed
  `SparseTensor`.
output_shape: 1-D.  Shape of the concatenated `SparseTensor`.
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
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle input_shape = c->input(3);
      ShapeHandle output_indices =
          c->Matrix(InferenceContext::kUnknownDim, c->NumElements(input_shape));
      ShapeHandle output_values = c->Vector(InferenceContext::kUnknownDim);
      ShapeHandle output_shape = input_shape;

      // Copy the outputs into the output ranges.
      int num_splits = c->num_outputs() / 3;
      int out_idx = 0;
      for (int i = 0; i < num_splits; ++i)
        c->set_output(out_idx++, output_indices);
      for (int i = 0; i < num_splits; ++i)
        c->set_output(out_idx++, output_values);
      for (int i = 0; i < num_splits; ++i)
        c->set_output(out_idx++, output_shape);
      return Status::OK();
    })
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

REGISTER_OP("SparseSlice")
    .Input("indices: int64")
    .Input("values: T")
    .Input("shape: int64")
    .Input("start: int64")
    .Input("size: int64")
    .Output("output_indices: int64")
    .Output("output_values: T")
    .Output("output_shape: int64")
    .Attr("T: type")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle input_shape = c->input(2);
      ShapeHandle output_indices =
          c->Matrix(InferenceContext::kUnknownDim, c->NumElements(input_shape));
      ShapeHandle output_values = c->Vector(InferenceContext::kUnknownDim);
      ShapeHandle output_shape = input_shape;

      c->set_output(0, output_indices);
      c->set_output(1, output_values);
      c->set_output(2, output_shape);
      return Status::OK();
    })
    .Doc(R"doc(
Slice a `SparseTensor` based on the `start` and `size`.

For example, if the input is

    input_tensor = shape = [2, 7]
    [    a   d e  ]
    [b c          ]

Graphically the output tensors are:

    sparse_slice([0, 0], [2, 4]) = shape = [2, 4]
    [    a  ]
    [b c    ]

    sparse_slice([0, 4], [2, 3]) = shape = [2, 3]
    [ d e  ]
    [      ]

indices: 2-D tensor represents the indices of the sparse tensor.
values: 1-D tensor represents the values of the sparse tensor.
shape: 1-D. tensor represents the shape of the sparse tensor.
start: 1-D. tensor represents the start of the slice.
size: 1-D. tensor represents the size of the slice.
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
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle indices;
      ShapeHandle values;
      ShapeHandle unused;

      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &indices));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &values));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &unused));

      c->set_output(0, indices);
      c->set_output(1, values);
      return Status::OK();
    })
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

REGISTER_OP("SparseReshape")
    .Input("input_indices: int64")
    .Input("input_shape: int64")
    .Input("new_shape: int64")
    .Output("output_indices: int64")
    .Output("output_shape: int64")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle indices;
      ShapeHandle unused;
      ShapeHandle new_shape;

      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &indices));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &new_shape));

      c->set_output(0, c->Matrix(c->Dim(indices, 0), c->Dim(new_shape, 0)));
      c->set_output(1, new_shape);
      return Status::OK();
    })
    .Doc(R"doc(
Reshapes a SparseTensor to represent values in a new dense shape.

This operation has the same semantics as reshape on the represented dense
tensor.  The `input_indices` are recomputed based on the requested `new_shape`.

If one component of `new_shape` is the special value -1, the size of that
dimension is computed so that the total dense size remains constant.  At
most one component of `new_shape` can be -1.  The number of dense elements
implied by `new_shape` must be the same as the number of dense elements
originally implied by `input_shape`.

Reshaping does not affect the order of values in the SparseTensor.

If the input tensor has rank `R_in` and `N` non-empty values, and `new_shape`
has length `R_out`, then `input_indices` has shape `[N, R_in]`,
`input_shape` has length `R_in`, `output_indices` has shape `[N, R_out]`, and
`output_shape` has length `R_out`.

input_indices: 2-D.  `N x R_in` matrix with the indices of non-empty values in a
  SparseTensor.
input_shape: 1-D.  `R_in` vector with the input SparseTensor's dense shape.
new_shape: 1-D.  `R_out` vector with the requested new dense shape.
output_indices: 2-D.  `N x R_out` matrix with the updated indices of non-empty
  values in the output SparseTensor.
output_shape: 1-D.  `R_out` vector with the full dense shape of the output
  SparseTensor.  This is the same as `new_shape` but with any -1 dimensions
  filled in.
)doc");

REGISTER_OP("SparseTensorDenseAdd")
    .Input("a_indices: Tindices")
    .Input("a_values: T")
    .Input("a_shape: Tindices")
    .Input("b: T")
    .Output("output: T")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(3));
      return Status::OK();
    })
    .Doc(R"doc(
Adds up a `SparseTensor` and a dense `Tensor`, producing a dense `Tensor`.

This Op does not require `a_indices` be sorted in standard lexicographic order.

a_indices: 2-D.  The `indices` of the `SparseTensor`, with shape `[nnz, ndims]`.
a_values: 1-D.  The `values` of the `SparseTensor`, with shape `[nnz]`.
a_shape: 1-D.  The `shape` of the `SparseTensor`, with shape `[ndims]`.
b: `ndims`-D Tensor.  With shape `a_shape`.
)doc");

REGISTER_OP("SparseReduceMax")
    .Input("input_indices: int64")
    .Input("input_values: T")
    .Input("input_shape: int64")
    .Input("reduction_axes: int32")
    .Attr("keep_dims: bool = False")
    .Output("output: T")
    .Attr("T: realnumbertype")
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
Computes the max of elements across dimensions of a SparseTensor.

This Op takes a SparseTensor and is the sparse counterpart to
`tf.reduce_max()`.  In particular, this Op also returns a dense `Tensor`
instead of a sparse one.

Reduces `sp_input` along the dimensions given in `reduction_axes`.  Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`reduction_axes`. If `keep_dims` is true, the reduced dimensions are retained
with length 1.

If `reduction_axes` has no entries, all dimensions are reduced, and a tensor
with a single element is returned.  Additionally, the axes can be negative,
which are interpreted according to the indexing rules in Python.

input_indices: 2-D.  `N x R` matrix with the indices of non-empty values in a
  SparseTensor, possibly not in canonical ordering.
input_values: 1-D.  `N` non-empty values corresponding to `input_indices`.
input_shape: 1-D.  Shape of the input SparseTensor.
reduction_axes: 1-D.  Length-`K` vector containing the reduction axes.
keep_dims: If true, retain reduced dimensions with length 1.
output: `R-K`-D.  The reduced Tensor.
)doc");

REGISTER_OP("SparseReduceMaxSparse")
    .Input("input_indices: int64")
    .Input("input_values: T")
    .Input("input_shape: int64")
    .Input("reduction_axes: int32")
    .Attr("keep_dims: bool = False")
    .Output("output_indices: int64")
    .Output("output_values: T")
    .Output("output_shape: int64")
    .Attr("T: realnumbertype")
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
Computes the max of elements across dimensions of a SparseTensor.

This Op takes a SparseTensor and is the sparse counterpart to
`tf.reduce_max()`.  In contrast to SparseReduceMax, this Op returns a
SparseTensor.

Reduces `sp_input` along the dimensions given in `reduction_axes`.  Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`reduction_axes`. If `keep_dims` is true, the reduced dimensions are retained
with length 1.

If `reduction_axes` has no entries, all dimensions are reduced, and a tensor
with a single element is returned.  Additionally, the axes can be negative,
which are interpreted according to the indexing rules in Python.

input_indices: 2-D.  `N x R` matrix with the indices of non-empty values in a
  SparseTensor, possibly not in canonical ordering.
input_values: 1-D.  `N` non-empty values corresponding to `input_indices`.
input_shape: 1-D.  Shape of the input SparseTensor.
reduction_axes: 1-D.  Length-`K` vector containing the reduction axes.
keep_dims: If true, retain reduced dimensions with length 1.
)doc");

REGISTER_OP("SparseReduceSum")
    .Input("input_indices: int64")
    .Input("input_values: T")
    .Input("input_shape: int64")
    .Input("reduction_axes: int32")
    .Attr("keep_dims: bool = False")
    .Output("output: T")
    .Attr("T: numbertype")
    .SetShapeFn(shape_inference::UnknownShape)
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
with a single element is returned.  Additionally, the axes can be negative,
which are interpreted according to the indexing rules in Python.

input_indices: 2-D.  `N x R` matrix with the indices of non-empty values in a
  SparseTensor, possibly not in canonical ordering.
input_values: 1-D.  `N` non-empty values corresponding to `input_indices`.
input_shape: 1-D.  Shape of the input SparseTensor.
reduction_axes: 1-D.  Length-`K` vector containing the reduction axes.
keep_dims: If true, retain reduced dimensions with length 1.
output: `R-K`-D.  The reduced Tensor.
)doc");

REGISTER_OP("SparseReduceSumSparse")
    .Input("input_indices: int64")
    .Input("input_values: T")
    .Input("input_shape: int64")
    .Input("reduction_axes: int32")
    .Attr("keep_dims: bool = False")
    .Output("output_indices: int64")
    .Output("output_values: T")
    .Output("output_shape: int64")
    .Attr("T: numbertype")
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
Computes the sum of elements across dimensions of a SparseTensor.

This Op takes a SparseTensor and is the sparse counterpart to
`tf.reduce_sum()`.  In contrast to SparseReduceSum, this Op returns a
SparseTensor.

Reduces `sp_input` along the dimensions given in `reduction_axes`.  Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`reduction_axes`. If `keep_dims` is true, the reduced dimensions are retained
with length 1.

If `reduction_axes` has no entries, all dimensions are reduced, and a tensor
with a single element is returned.  Additionally, the axes can be negative,
which are interpreted according to the indexing rules in Python.

input_indices: 2-D.  `N x R` matrix with the indices of non-empty values in a
  SparseTensor, possibly not in canonical ordering.
input_values: 1-D.  `N` non-empty values corresponding to `input_indices`.
input_shape: 1-D.  Shape of the input SparseTensor.
reduction_axes: 1-D.  Length-`K` vector containing the reduction axes.
keep_dims: If true, retain reduced dimensions with length 1.
)doc");

#define SPARSE_DENSE_CWISE_SIGNATURE()                           \
  Input("sp_indices: int64")                                     \
      .Input("sp_values: T")                                     \
      .Input("sp_shape: int64")                                  \
      .Input("dense: T")                                         \
      .Output("output: T")                                       \
      .Attr("T: numbertype")                                     \
      .SetShapeFn([](InferenceContext* c) {                      \
        ShapeHandle input;                                       \
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input)); \
        c->set_output(0, c->Vector(c->Dim(input, 0)));           \
        return Status::OK();                                     \
      })

REGISTER_OP("SparseDenseCwiseMul")
    .SPARSE_DENSE_CWISE_SIGNATURE()
    .Doc(R"doc(
Component-wise multiplies a SparseTensor by a dense Tensor.

The output locations corresponding to the implicitly zero elements in the sparse
tensor will be zero (i.e., will not take up storage space), regardless of the
contents of the dense tensor (even if it's +/-INF and that INF*0 == NaN).

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
    .SPARSE_DENSE_CWISE_SIGNATURE()
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

REGISTER_OP("SparseDenseCwiseAdd")
    .SPARSE_DENSE_CWISE_SIGNATURE()
    .Doc(R"doc(
Adds up a SparseTensor and a dense Tensor, using these special rules:

(1) Broadcasts the dense side to have the same shape as the sparse side, if
    eligible;
(2) Then, only the dense values pointed to by the indices of the SparseTensor
    participate in the cwise addition.

By these rules, the result is a logical SparseTensor with exactly the same
indices and shape, but possibly with different non-zero values.  The output of
this Op is the resultant non-zero values.

sp_indices: 2-D.  `N x R` matrix with the indices of non-empty values in a
  SparseTensor, possibly not in canonical ordering.
sp_values: 1-D.  `N` non-empty values corresponding to `sp_indices`.
sp_shape: 1-D.  Shape of the input SparseTensor.
dense: `R`-D.  The dense Tensor operand.
output: 1-D.  The `N` values that are operated on.
)doc");

#undef SPARSE_DENSE_CWISE_SIGNATURE

REGISTER_OP("SparseSoftmax")
    .Input("sp_indices: int64")
    .Input("sp_values: T")
    .Input("sp_shape: int64")
    .Output("output: T")
    .Attr("T: {float, double}")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      ShapeHandle values;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &unused));  // sp_indices
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &values));  // sp_values
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &unused));
      c->set_output(0, values);
      return Status::OK();
    })
    .Doc(R"doc(
Applies softmax to a batched N-D `SparseTensor`.

The inputs represent an N-D SparseTensor  with logical shape `[..., B, C]`
(where `N >= 2`), and with indices sorted in the canonical lexicographic order.

This op is equivalent to applying the normal `tf.nn.softmax()` to each innermost
logical submatrix with shape `[B, C]`, but with the catch that *the implicitly
zero elements do not participate*.  Specifically, the algorithm is equivalent
to the following:

  (1) Applies `tf.nn.softmax()` to a densified view of each innermost submatrix
      with shape `[B, C]`, along the size-C dimension;
  (2) Masks out the original implicitly-zero locations;
  (3) Renormalizes the remaining elements.

Hence, the `SparseTensor` result has exactly the same non-zero indices and
shape.

sp_indices: 2-D.  `NNZ x R` matrix with the indices of non-empty values in a
  SparseTensor, in canonical ordering.
sp_values: 1-D.  `NNZ` non-empty values corresponding to `sp_indices`.
sp_shape: 1-D.  Shape of the input SparseTensor.
output: 1-D.  The `NNZ` values for the result `SparseTensor`.
)doc");

REGISTER_OP("SparseSparseMaximum")
    .Input("a_indices: int64")
    .Input("a_values: T")
    .Input("a_shape: int64")
    .Input("b_indices: int64")
    .Input("b_values: T")
    .Input("b_shape: int64")
    .Output("output_indices: int64")
    .Output("output_values: T")
    .Attr("T: realnumbertype")
    .SetShapeFn(SparseSparseMinOrMaxShapeFn)
    .Doc(R"doc(
Returns the element-wise max of two SparseTensors.

Assumes the two SparseTensors have the same shape, i.e., no broadcasting.

a_indices: 2-D.  `N x R` matrix with the indices of non-empty values in a
  SparseTensor, in the canonical lexicographic ordering.
a_values: 1-D.  `N` non-empty values corresponding to `a_indices`.
a_shape: 1-D.  Shape of the input SparseTensor.
b_indices: counterpart to `a_indices` for the other operand.
b_values: counterpart to `a_values` for the other operand; must be of the same dtype.
b_shape: counterpart to `a_shape` for the other operand; the two shapes must be equal.

output_indices: 2-D.  The indices of the output SparseTensor.
output_values: 1-D.  The values of the output SparseTensor.
)doc");

REGISTER_OP("SparseSparseMinimum")
    .Input("a_indices: int64")
    .Input("a_values: T")
    .Input("a_shape: int64")
    .Input("b_indices: int64")
    .Input("b_values: T")
    .Input("b_shape: int64")
    .Output("output_indices: int64")
    .Output("output_values: T")
    .Attr("T: numbertype")
    .SetShapeFn(SparseSparseMinOrMaxShapeFn)
    .Doc(R"doc(
Returns the element-wise min of two SparseTensors.

Assumes the two SparseTensors have the same shape, i.e., no broadcasting.

a_indices: 2-D.  `N x R` matrix with the indices of non-empty values in a
  SparseTensor, in the canonical lexicographic ordering.
a_values: 1-D.  `N` non-empty values corresponding to `a_indices`.
a_shape: 1-D.  Shape of the input SparseTensor.
b_indices: counterpart to `a_indices` for the other operand.
b_values: counterpart to `a_values` for the other operand; must be of the same dtype.
b_shape: counterpart to `a_shape` for the other operand; the two shapes must be equal.

output_indices: 2-D.  The indices of the output SparseTensor.
output_values: 1-D.  The values of the output SparseTensor.
)doc");

REGISTER_OP("AddSparseToTensorsMap")
    .Input("sparse_indices: int64")
    .Input("sparse_values: T")
    .Input("sparse_shape: int64")
    .Output("sparse_handle: int64")
    .Attr("T: type")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &unused));
      c->set_output(0, c->Scalar());
      return Status::OK();
    })
    .Doc(R"doc(
Add a `SparseTensor` to a `SparseTensorsMap` return its handle.

A `SparseTensor` is represented by three tensors: `sparse_indices`,
`sparse_values`, and `sparse_shape`.

This operator takes the given `SparseTensor` and adds it to a container
object (a `SparseTensorsMap`).  A unique key within this container is generated
in the form of an `int64`, and this is the value that is returned.

The `SparseTensor` can then be read out as part of a minibatch by passing
the key as a vector element to `TakeManySparseFromTensorsMap`.  To ensure
the correct `SparseTensorsMap` is accessed, ensure that the same
`container` and `shared_name` are passed to that Op.  If no `shared_name`
is provided here, instead use the *name* of the Operation created by calling
`AddSparseToTensorsMap` as the `shared_name` passed to
`TakeManySparseFromTensorsMap`.  Ensure the Operations are colocated.

sparse_indices: 2-D.  The `indices` of the `SparseTensor`.
sparse_values: 1-D.  The `values` of the `SparseTensor`.
sparse_shape: 1-D.  The `shape` of the `SparseTensor`.
sparse_handle: 0-D.  The handle of the `SparseTensor` now stored in the
  `SparseTensorsMap`.
container: The container name for the `SparseTensorsMap` created by this op.
shared_name: The shared name for the `SparseTensorsMap` created by this op.
  If blank, the new Operation's unique name is used.
)doc");

REGISTER_OP("AddManySparseToTensorsMap")
    .Input("sparse_indices: int64")
    .Input("sparse_values: T")
    .Input("sparse_shape: int64")
    .Output("sparse_handles: int64")
    .Attr("T: type")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &unused));
      c->set_output(0, c->Vector(InferenceContext::kUnknownDim));
      return Status::OK();
    })
    .Doc(R"doc(
Add an `N`-minibatch `SparseTensor` to a `SparseTensorsMap`, return `N` handles.

A `SparseTensor` of rank `R` is represented by three tensors: `sparse_indices`,
`sparse_values`, and `sparse_shape`, where

```sparse_indices.shape[1] == sparse_shape.shape[0] == R```

An `N`-minibatch of `SparseTensor` objects is represented as a `SparseTensor`
having a first `sparse_indices` column taking values between `[0, N)`, where
the minibatch size `N == sparse_shape[0]`.

The input `SparseTensor` must have rank `R` greater than 1, and the first
dimension is treated as the minibatch dimension.  Elements of the `SparseTensor`
must be sorted in increasing order of this first dimension.  The stored
`SparseTensor` objects pointed to by each row of the output `sparse_handles`
will have rank `R-1`.

The `SparseTensor` values can then be read out as part of a minibatch by passing
the given keys as vector elements to `TakeManySparseFromTensorsMap`.  To ensure
the correct `SparseTensorsMap` is accessed, ensure that the same
`container` and `shared_name` are passed to that Op.  If no `shared_name`
is provided here, instead use the *name* of the Operation created by calling
`AddManySparseToTensorsMap` as the `shared_name` passed to
`TakeManySparseFromTensorsMap`.  Ensure the Operations are colocated.

sparse_indices: 2-D.  The `indices` of the minibatch `SparseTensor`.
  `sparse_indices[:, 0]` must be ordered values in `[0, N)`.
sparse_values: 1-D.  The `values` of the minibatch `SparseTensor`.
sparse_shape: 1-D.  The `shape` of the minibatch `SparseTensor`.
  The minibatch size `N == sparse_shape[0]`.
sparse_handles: 1-D.  The handles of the `SparseTensor` now stored in the
  `SparseTensorsMap`.  Shape: `[N]`.
container: The container name for the `SparseTensorsMap` created by this op.
shared_name: The shared name for the `SparseTensorsMap` created by this op.
  If blank, the new Operation's unique name is used.
)doc");

REGISTER_OP("TakeManySparseFromTensorsMap")
    .Input("sparse_handles: int64")
    .Output("sparse_indices: int64")
    .Output("sparse_values: dtype")
    .Output("sparse_shape: int64")
    .Attr("dtype: type")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      // serialized sparse is [?,1] matrix.
      ShapeHandle sparse_handles;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &sparse_handles));

      c->set_output(0, c->Matrix(InferenceContext::kUnknownDim,
                                 InferenceContext::kUnknownDim));
      c->set_output(1, c->Vector(InferenceContext::kUnknownDim));
      c->set_output(2, c->Vector(InferenceContext::kUnknownDim));
      return Status::OK();
    })
    .Doc(R"doc(
Read `SparseTensors` from a `SparseTensorsMap` and concatenate them.

The input `sparse_handles` must be an `int64` matrix of shape `[N, 1]` where
`N` is the minibatch size and the rows correspond to the output handles of
`AddSparseToTensorsMap` or `AddManySparseToTensorsMap`.  The ranks of the
original `SparseTensor` objects that went into the given input ops must all
match.  When the final `SparseTensor` is created, it has rank one
higher than the ranks of the incoming `SparseTensor` objects
(they have been concatenated along a new row dimension on the left).

The output `SparseTensor` object's shape values for all dimensions but the
first are the max across the input `SparseTensor` objects' shape values
for the corresponding dimensions.  Its first shape value is `N`, the minibatch
size.

The input `SparseTensor` objects' indices are assumed ordered in
standard lexicographic order.  If this is not the case, after this
step run `SparseReorder` to restore index ordering.

For example, if the handles represent an input, which is a `[2, 3]` matrix
representing two original `SparseTensor` objects:

```
    index = [ 0]
            [10]
            [20]
    values = [1, 2, 3]
    shape = [50]
```

and

```
    index = [ 2]
            [10]
    values = [4, 5]
    shape = [30]
```

then the final `SparseTensor` will be:

```
    index = [0  0]
            [0 10]
            [0 20]
            [1  2]
            [1 10]
    values = [1, 2, 3, 4, 5]
    shape = [2 50]
```

sparse_handles: 1-D, The `N` serialized `SparseTensor` objects.
  Shape: `[N]`.
sparse_indices: 2-D.  The `indices` of the minibatch `SparseTensor`.
sparse_values: 1-D.  The `values` of the minibatch `SparseTensor`.
sparse_shape: 1-D.  The `shape` of the minibatch `SparseTensor`.
dtype: The `dtype` of the `SparseTensor` objects stored in the
  `SparseTensorsMap`.
container: The container name for the `SparseTensorsMap` read by this op.
shared_name: The shared name for the `SparseTensorsMap` read by this op.
  It should not be blank; rather the `shared_name` or unique Operation name
  of the Op that created the original `SparseTensorsMap` should be used.
)doc");

REGISTER_OP("SparseFillEmptyRows")
    .Input("indices: int64")
    .Input("values: T")
    .Input("dense_shape: int64")
    .Input("default_value: T")
    .Output("output_indices: int64")
    .Output("output_values: T")
    .Output("empty_row_indicator: bool")
    .Output("reverse_index_map: int64")
    .Attr("T: type")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle input_indices = c->input(0);
      TF_RETURN_IF_ERROR(c->WithRank(input_indices, 2, &input_indices));
      ShapeHandle input_values = c->input(1);
      TF_RETURN_IF_ERROR(c->WithRank(input_values, 1, &input_values));
      ShapeHandle input_shape = c->input(2);
      TF_RETURN_IF_ERROR(c->WithRank(input_shape, 1, &input_shape));
      ShapeHandle default_value = c->input(3);
      TF_RETURN_IF_ERROR(c->WithRank(default_value, 0, &default_value));
      DimensionHandle N = c->Dim(input_indices, 0);
      TF_RETURN_IF_ERROR(c->Merge(N, c->Dim(input_values, 0), &N));
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->Merge(c->Dim(input_indices, 1),
                                  c->Dim(input_shape, 0), &unused_dim));
      ShapeHandle output_indices =
          c->Matrix(InferenceContext::kUnknownDim, c->NumElements(input_shape));
      ShapeHandle output_values = c->Vector(InferenceContext::kUnknownDim);
      ShapeHandle constant_input_shape;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(2, &constant_input_shape));
      ShapeHandle empty_row_indicator =
          c->Vector(c->Dim(constant_input_shape, 0));
      ShapeHandle reverse_index_map = c->Vector(N);
      c->set_output(0, output_indices);
      c->set_output(1, output_values);
      c->set_output(2, empty_row_indicator);
      c->set_output(3, reverse_index_map);
      return Status::OK();
    })
    .Doc(R"doc(
Fills empty rows in the input 2-D `SparseTensor` with a default value.

The input `SparseTensor` is represented via the tuple of inputs
(`indices`, `values`, `dense_shape`).  The output `SparseTensor` has the
same `dense_shape` but with indices `output_indices` and values
`output_values`.

This op inserts a single entry for every row that doesn't have any values.
The index is created as `[row, 0, ..., 0]` and the inserted value
is `default_value`.

For example, suppose `sp_input` has shape `[5, 6]` and non-empty values:

    [0, 1]: a
    [0, 3]: b
    [2, 0]: c
    [3, 1]: d

Rows 1 and 4 are empty, so the output will be of shape `[5, 6]` with values:

    [0, 1]: a
    [0, 3]: b
    [1, 0]: default_value
    [2, 0]: c
    [3, 1]: d
    [4, 0]: default_value

The output `SparseTensor` will be in row-major order and will have the
same shape as the input.

This op also returns an indicator vector shaped `[dense_shape[0]]` such that

    empty_row_indicator[i] = True iff row i was an empty row.

And a reverse index map vector shaped `[indices.shape[0]]` that is used during
backpropagation,

    reverse_index_map[j] = out_j s.t. indices[j, :] == output_indices[out_j, :]


indices: 2-D. the indices of the sparse tensor.
values: 1-D. the values of the sparse tensor.
dense_shape: 1-D. the shape of the sparse tensor.
default_value: 0-D. default value to insert into location `[row, 0, ..., 0]`
  for rows missing from the input sparse tensor.
output indices: 2-D. the indices of the filled sparse tensor.
output_values: 1-D. the values of the filled sparse tensor.
empty_row_indicator: 1-D. whether the dense row was missing in the
  input sparse tensor.
reverse_index_map: 1-D. a map from the input indices to the output indices.
)doc");

REGISTER_OP("SparseFillEmptyRowsGrad")
    .Input("reverse_index_map: int64")
    .Input("grad_values: T")
    .Output("d_values: T")
    .Output("d_default_value: T")
    .Attr("T: type")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle reverse_index_map = c->input(0);
      TF_RETURN_IF_ERROR(c->WithRank(reverse_index_map, 1, &reverse_index_map));
      ShapeHandle grad_values = c->input(1);
      TF_RETURN_IF_ERROR(c->WithRank(grad_values, 1, &grad_values));
      c->set_output(0, reverse_index_map);
      c->set_output(1, c->Scalar());
      return Status::OK();
    })
    .Doc(R"doc(
The gradient of SparseFillEmptyRows.

Takes vectors reverse_index_map, shaped `[N]`, and grad_values,
shaped `[N_full]`, where `N_full >= N` and copies data into either
`d_values` or `d_default_value`.  Here `d_values` is shaped `[N]` and
`d_default_value` is a scalar.

  d_values[j] = grad_values[reverse_index_map[j]]
  d_default_value = sum_{k : 0 .. N_full - 1} (
     grad_values[k] * 1{k not in reverse_index_map})

reverse_index_map: 1-D.  The reverse index map from SparseFillEmptyRows.
grad_values: 1-D.  The gradients from backprop.
d_values: 1-D.  The backprop into values.
d_default_value: 0-D.  The backprop into default_value.
)doc");

}  // namespace tensorflow
