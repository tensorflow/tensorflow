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
#include "tensorflow/core/util/mirror_pad_mode.h"
#include "tensorflow/core/util/padding.h"

namespace tensorflow {

using shape_inference::Dimension;
using shape_inference::InferenceContext;
using shape_inference::Shape;

namespace {

Status GetAxisForPackAndUnpack(InferenceContext* c, int32 rank_after_pack,
                               int32* axis) {
  TF_RETURN_IF_ERROR(c->GetAttr("axis", axis));
  if (*axis < -1 * rank_after_pack || *axis >= rank_after_pack) {
    return errors::InvalidArgument("Invalid axis: ", *axis, "; must be in [",
                                   -1 * rank_after_pack, ",", rank_after_pack,
                                   ")");
  }
  if (*axis < 0) *axis = (rank_after_pack + *axis);
  return Status::OK();
}

Status PadShapeFn(InferenceContext* c) {
  // Paddings is a matrix of [input_rank, 2].
  const Shape* paddings;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &paddings));
  const Dimension* unused;
  TF_RETURN_IF_ERROR(c->WithValue(c->Dim(paddings, 1), 2, &unused));

  // n_dim and input.rank are equivalent.
  const Shape* input = c->input(0);
  const Dimension* n_dim = c->Dim(paddings, 0);
  if (c->ValueKnown(n_dim)) {
    TF_RETURN_IF_ERROR(c->WithRank(input, c->Value(n_dim), &input));
  } else if (c->RankKnown(input)) {
    TF_RETURN_IF_ERROR(c->WithValue(n_dim, c->Rank(input), &n_dim));
  }

  const Tensor* paddings_t = c->input_tensor(1);

  // paddings_t is unknown
  if (paddings_t == nullptr) {
    if (c->ValueKnown(n_dim)) {
      // Make output with n_dim unknown dims.
      std::vector<const Dimension*> dims;
      const auto value = c->Value(n_dim);
      for (int i = 0; i < value; ++i) dims.push_back(c->UnknownDim());
      c->set_output(0, c->MakeShape(dims));
    } else {
      c->set_output(0, c->UnknownShape());
    }
    return Status::OK();
  }

  // tensor value was provided for paddings_t; doublecheck n_dim value is the
  // same.
  const auto num_dims = c->Value(n_dim);
  DCHECK_EQ(num_dims, paddings_t->shape().dim_size(0));

  // paddings_t is known.
  auto paddings_data = paddings_t->matrix<int32>();
  std::vector<const Dimension*> dims(num_dims);
  for (int i = 0; i < num_dims; ++i) {
    const int32 pad0 = paddings_data(i, 0);
    const int32 pad1 = paddings_data(i, 1);
    if (pad0 < 0 || pad1 < 0) {
      return errors::InvalidArgument("Paddings must be non-negative");
    }
    TF_RETURN_IF_ERROR(c->Add(c->Dim(input, i), pad0 + pad1, &dims[i]));
  }
  c->set_output(0, c->MakeShape(dims));
  return Status::OK();
}

}  // namespace

REGISTER_OP("Pack")
    .Input("values: N * T")
    .Output("output: T")
    .Attr("N: int >= 1")
    .Attr("T: type")
    .Attr("axis: int = 0")
    .SetShapeFn([](InferenceContext* c) {
      // Validate shapes of all inputs are compatible
      const Shape* cur = c->input(c->num_inputs() - 1);
      for (int i = c->num_inputs() - 2; i >= 0; --i) {
        TF_RETURN_WITH_CONTEXT_IF_ERROR(c->Merge(c->input(i), cur, &cur),
                                        "From merging shape ", i,
                                        " with other shapes.");
      }
      if (!c->RankKnown(cur)) {
        c->set_output(0, c->UnknownShape());
        return Status::OK();
      }
      // Determine the axis that will be added, converting from negative
      // axes to a positive point per negative indexing rules.
      int32 rank = c->Rank(cur);
      int32 axis;
      TF_RETURN_IF_ERROR(GetAxisForPackAndUnpack(c, rank + 1, &axis));

      // Copy all dimensions over, inserting a dimension of value #inputs
      // at <axis>.
      std::vector<const Dimension*> dims;
      int index = 0;
      while (index < axis) dims.push_back(c->Dim(cur, index++));
      dims.push_back(c->MakeDim(c->num_inputs()));
      while (index < rank) dims.push_back(c->Dim(cur, index++));

      c->set_output(0, c->MakeShape(dims));
      return Status::OK();
    })
    .Doc(R"doc(
Packs a list of `N` rank-`R` tensors into one rank-`(R+1)` tensor.

Packs the `N` tensors in `values` into a tensor with rank one higher than each
tensor in `values`, by packing them along the `axis` dimension.
Given a list of tensors of shape `(A, B, C)`;

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

This is the opposite of `unpack`.

values: Must be of same shape and type.
axis: Dimension along which to pack.  Negative values wrap around, so the
  valid range is `[-(R+1), R+1)`.
output: The packed tensor.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("Unpack")
    .Input("value: T")
    .Output("output: num * T")
    .Attr("num: int >= 0")
    .Attr("T: type")
    .Attr("axis: int = 0")
    .SetShapeFn([](InferenceContext* c) {
      const Shape* s = c->input(0);
      const Shape* out;
      if (c->RankKnown(s)) {
        // Determine the axis that will be removed, converting from negative
        // axes to a positive point per negative indexing rules.
        int32 rank = c->Rank(s);
        int32 axis;
        TF_RETURN_IF_ERROR(GetAxisForPackAndUnpack(c, rank, &axis));

        // The axis dim matches the number of outputs.
        const Dimension* unused;
        TF_RETURN_IF_ERROR(
            c->WithValue(c->Dim(s, axis), c->num_outputs(), &unused));

        // Copy all dimensions, removing the <axis> dimension.
        std::vector<const Dimension*> dims;
        for (int i = 0; i < rank; ++i) {
          if (i != axis) dims.push_back(c->Dim(s, i));
        }
        out = c->MakeShape(dims);
      } else {
        // All outputs are the same shape, but it's not known.
        out = c->UnknownShape();
      }
      for (int i = 0; i < c->num_outputs(); ++i) c->set_output(i, out);
      return Status::OK();
    })
    .Doc(R"doc(
Unpacks a given dimension of a rank-`R` tensor into `num` rank-`(R-1)` tensors.

Unpacks `num` tensors from `value` by chipping it along the `axis` dimension.
For example, given a tensor of shape `(A, B, C, D)`;

If `axis == 0` then the i'th tensor in `output` is the slice `value[i, :, :, :]`
  and each tensor in `output` will have shape `(B, C, D)`. (Note that the
  dimension unpacked along is gone, unlike `split`).

If `axis == 1` then the i'th tensor in `output` is the slice `value[:, i, :, :]`
  and each tensor in `output` will have shape `(A, C, D)`.
Etc.

This is the opposite of `pack`.

value: 1-D or higher, with `axis` dimension size equal to `num`.
axis: Dimension along which to unpack.  Negative values wrap around, so the
  valid range is `[-R, R)`.
output: The list of tensors unpacked from `value`.
)doc");

// --------------------------------------------------------------------------
// TODO(josh11b): Remove the >= 2 constraint, once we can rewrite the graph
// in the N == 1 case to remove the node.
REGISTER_OP("Concat")
    .Input("concat_dim: int32")
    .Input("values: N * T")
    .Output("output: T")
    .Attr("N: int >= 2")
    .Attr("T: type")
    .SetShapeFn([](InferenceContext* c) {
      const Shape* unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));

      const Tensor* concat_dim_t = c->input_tensor(0);
      if (concat_dim_t == nullptr) {
        // Return an unknown shape with same rank as inputs, or an unknown rank
        // if no input's rank is known.

        // Find rank.
        int32 rank = InferenceContext::kUnknownRank;
        for (int i = 1; i < c->num_inputs(); ++i) {
          if (rank == InferenceContext::kUnknownRank)
            rank = c->Rank(c->input(i));
          if (rank != InferenceContext::kUnknownRank) {
            TF_RETURN_IF_ERROR(c->WithRank(c->input(i), rank, &unused));
          }
        }
        if (rank == InferenceContext::kUnknownRank) {
          c->set_output(0, c->UnknownShape());
          return Status::OK();
        }
        if (rank == 0) {
          return errors::InvalidArgument(
              "Can't concatenate scalars (use tf.pack instead)");
        }
        // Build result of <rank> different unknown dims.
        std::vector<const Dimension*> dims;
        for (int i = 0; i < rank; ++i) dims.push_back(c->UnknownDim());
        c->set_output(0, c->MakeShape(dims));
        return Status::OK();
      }

      // Merge all the non-concat dims, and sum the concat dim to make an output
      // shape.
      const int32 concat_dim = concat_dim_t->scalar<int32>()();
      if (concat_dim < 0) {
        return errors::InvalidArgument("Expected concat_dim >= 0, but got ",
                                       concat_dim);
      }

      const Shape* output_before;
      const Shape* output_after;

      const Shape* input = c->input(c->num_inputs() - 1);
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(input, concat_dim + 1, &input));
      TF_RETURN_IF_ERROR(c->Subshape(input, 0, concat_dim, &output_before));
      const Dimension* output_middle = c->Dim(input, concat_dim);
      TF_RETURN_IF_ERROR(c->Subshape(input, concat_dim + 1, &output_after));

      for (int i = c->num_inputs() - 2; i > 0; --i) {
        const Shape* before;
        const Shape* after;
        input = c->input(i);
        TF_RETURN_IF_ERROR(c->WithRankAtLeast(input, concat_dim + 1, &input));
        TF_RETURN_IF_ERROR(c->Subshape(input, 0, concat_dim, &before));
        const Dimension* middle = c->Dim(input, concat_dim);
        TF_RETURN_IF_ERROR(c->Subshape(input, concat_dim + 1, &after));

        TF_RETURN_IF_ERROR(c->Merge(before, output_before, &output_before));
        TF_RETURN_IF_ERROR(c->Add(output_middle, middle, &output_middle));
        TF_RETURN_IF_ERROR(c->Merge(after, output_after, &output_after));
      }

      const Shape* s;
      TF_RETURN_IF_ERROR(
          c->Concatenate(output_before, c->Vector(output_middle), &s));
      TF_RETURN_IF_ERROR(c->Concatenate(s, output_after, &s));
      c->set_output(0, s);
      return Status::OK();
    })
    .Doc(R"doc(
Concatenates tensors along one dimension.

concat_dim: 0-D.  The dimension along which to concatenate.  Must be in the
  range [0, rank(values)).
values: The `N` Tensors to concatenate. Their ranks and types must match,
  and their sizes must match in all dimensions except `concat_dim`.
output: A `Tensor` with the concatenation of values stacked along the
  `concat_dim` dimension.  This tensor's shape matches that of `values` except
  in `concat_dim` where it has the sum of the sizes.
)doc");

REGISTER_OP("ConcatOffset")
    .Input("concat_dim: int32")
    .Input("shape: N * int32")
    .Output("offset: N * int32")
    .Attr("N: int >= 2")
    .SetShapeFn([](InferenceContext* c) {
      for (int i = 1; i < c->num_inputs(); ++i) {
        c->set_output(i - 1, c->input(i));
      }
      return Status::OK();
    })
    .Doc(R"doc(
Computes offsets of concat inputs within its output.

For example:

```prettyprint
# 'x' is [2, 2, 7]
# 'y' is [2, 3, 7]
# 'z' is [2, 5, 7]
concat_offset(2, [x, y, z]) => [0, 0, 0], [0, 2, 0], [0, 5, 0]
```

concat_dim: The dimension along which to concatenate.
shape: The `N` int32 vectors representing shape of tensors being concatenated.
offset: The `N` int32 vectors representing the starting offset
        of input tensors within the concatenated output.

This is typically used by gradient computations for a concat operation.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("Split")
    .Input("split_dim: int32")
    .Input("value: T")
    .Output("output: num_split * T")
    .Attr("num_split: int >= 1")
    .Attr("T: type")
    .SetShapeFn([](InferenceContext* c) {
      const Dimension* split_dimension;
      TF_RETURN_IF_ERROR(c->MakeDimForScalarInput(0, &split_dimension));
      int num_split = c->num_outputs();
      const Shape* input = c->input(1);
      const Shape* out;
      if (!c->ValueKnown(split_dimension)) {
        if (c->RankKnown(input)) {
          std::vector<const Dimension*> dims;
          dims.resize(c->Rank(input));
          for (int i = 0; i < dims.size(); ++i) dims[i] = c->UnknownDim();
          out = c->MakeShape(dims);
        } else {
          out = c->UnknownShape();
        }
      } else {
        int64 split_dim = c->Value(split_dimension);
        TF_RETURN_IF_ERROR(c->WithRankAtLeast(input, split_dim + 1, &input));
        const Dimension* split_dim_size;
        TF_RETURN_WITH_CONTEXT_IF_ERROR(
            c->Divide(c->Dim(input, split_dim), num_split, &split_dim_size),
            "Number of ways to split should evenly divide the split dimension");
        TF_RETURN_IF_ERROR(
            c->ReplaceDim(input, split_dim, split_dim_size, &out));
      }
      for (int i = 0; i < num_split; ++i) c->set_output(i, out);
      return Status::OK();
    })
    .Doc(R"doc(
Splits a tensor into `num_split` tensors along one dimension.

split_dim: 0-D.  The dimension along which to split.  Must be in the range
  `[0, rank(value))`.
num_split: The number of ways to split.  Must evenly divide
  `value.shape[split_dim]`.
value: The tensor to split.
output: They are identically shaped tensors, whose shape matches that of `value`
  except along `split_dim`, where their sizes are
  `values.shape[split_dim] / num_split`.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("Const")
    .Output("output: dtype")
    .Attr("value: tensor")
    .Attr("dtype: type")
    .SetShapeFn([](InferenceContext* c) {
      const TensorProto* proto = nullptr;
      TF_RETURN_IF_ERROR(c->GetAttr("value", &proto));
      TF_RETURN_IF_ERROR(TensorShape::IsValidShape(proto->tensor_shape()));
      TensorShape shape(proto->tensor_shape());
      std::vector<const Dimension*> dims;
      for (int i = 0; i < shape.dims(); ++i) {
        dims.push_back(c->MakeDim(shape.dim_size(i)));
      }
      c->set_output(0, c->MakeShape(dims));
      return Status::OK();
    })
    .Doc(R"doc(
Returns a constant tensor.

value: Attr `value` is the tensor to return.
)doc");

// --------------------------------------------------------------------------
// TODO(mgubin): Update the doc when the freeze_graph script supports converting
// into memmapped format.
REGISTER_OP("ImmutableConst")
    .Attr("dtype: type")
    .Attr("shape: shape")
    .Attr("memory_region_name: string")
    .Output("tensor: dtype")
    .SetShapeFn([](InferenceContext* c) {
      TensorShape shape_from_attr;
      TF_RETURN_IF_ERROR(c->GetAttr("shape", &shape_from_attr));
      TensorShapeProto shape_proto;
      shape_from_attr.AsProto(&shape_proto);
      const Shape* output_shape;
      TF_RETURN_IF_ERROR(
          c->MakeShapeFromShapeProto(shape_proto, &output_shape));
      c->set_output(0, output_shape);
      return Status::OK();
    })
    .Doc(R"doc(
Returns immutable tensor from memory region.

The current implementation memmaps the tensor from a file.

dtype: Type of the returned tensor.
shape: Shape of the returned tensor.
memory_region_name: Name of readonly memory region used by the tensor, see
  NewReadOnlyMemoryRegionFromFile in tensorflow::Env.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("ZerosLike")
    .Input("x: T")
    .Output("y: T")
    .Attr("T: type")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
Returns a tensor of zeros with the same shape and type as x.

x: a tensor of type T.
y: a tensor of the same shape and type as x but filled with zeros.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("Diag")
    .Input("diagonal: T")
    .Output("output: T")
    .Attr("T: {float, double, int32, int64, complex64}")
    .SetShapeFn([](InferenceContext* c) {
      const Shape* in = c->input(0);
      TF_RETURN_IF_ERROR(c->WithRankAtMost(in, 3, &in));
      // Output shape is original concatenated with itself.
      const Shape* out;
      TF_RETURN_IF_ERROR(c->Concatenate(in, in, &out));
      c->set_output(0, out);
      return Status::OK();
    })
    .Doc(R"doc(
Returns a diagonal tensor with a given diagonal values.

Given a `diagonal`, this operation returns a tensor with the `diagonal` and
everything else padded with zeros. The diagonal is computed as follows:

Assume `diagonal` has dimensions [D1,..., Dk], then the output is a tensor of
rank 2k with dimensions [D1,..., Dk, D1,..., Dk] where:

`output[i1,..., ik, i1,..., ik] = diagonal[i1, ..., ik]` and 0 everywhere else.

For example:

```prettyprint
# 'diagonal' is [1, 2, 3, 4]
tf.diag(diagonal) ==> [[1, 0, 0, 0]
                       [0, 2, 0, 0]
                       [0, 0, 3, 0]
                       [0, 0, 0, 4]]
```

diagonal: Rank k tensor where k is at most 3.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("DiagPart")
    .Input("input: T")
    .Output("diagonal: T")
    .Attr("T: {float, double, int32, int64, complex64}")
    .SetShapeFn([](InferenceContext* c) {
      const Shape* in = c->input(0);
      if (!c->RankKnown(in)) {
        c->set_output(0, c->UnknownShape());
        return Status::OK();
      }
      // Rank must be even, and result will have rank <rank/2>.
      const int32 rank = c->Rank(in);
      if ((rank % 2) != 0 || rank > 6) {
        return errors::InvalidArgument(
            "Input must have even rank <= 6, input rank is ", rank);
      }
      const int32 mid = rank / 2;

      // output dim[i] is the merge of in.dim[i] and in.dim[i+mid].
      std::vector<const Dimension*> dims(mid);
      for (int i = 0; i < mid; ++i) {
        TF_RETURN_IF_ERROR(
            c->Merge(c->Dim(in, i), c->Dim(in, i + mid), &dims[i]));
      }
      c->set_output(0, c->MakeShape(dims));
      return Status::OK();
    })
    .Doc(R"doc(
Returns the diagonal part of the tensor.

This operation returns a tensor with the `diagonal` part
of the `input`. The `diagonal` part is computed as follows:

Assume `input` has dimensions `[D1,..., Dk, D1,..., Dk]`, then the output is a
tensor of rank `k` with dimensions `[D1,..., Dk]` where:

`diagonal[i1,..., ik] = input[i1, ..., ik, i1,..., ik]`.

For example:

```prettyprint
# 'input' is [[1, 0, 0, 0]
              [0, 2, 0, 0]
              [0, 0, 3, 0]
              [0, 0, 0, 4]]

tf.diag_part(input) ==> [1, 2, 3, 4]
```

input: Rank k tensor where k is 2, 4, or 6.
diagonal: The extracted diagonal.

)doc");

// --------------------------------------------------------------------------
REGISTER_OP("BatchMatrixDiag")
    .Input("diagonal: T")
    .Output("output: T")
    .Attr("T: type")
    .SetShapeFn([](InferenceContext* c) {
      const Shape* in;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 1, &in));
      if (!c->RankKnown(in)) {
        c->set_output(0, c->UnknownShape());
        return Status::OK();
      }
      const int32 rank = c->Rank(in);
      const Shape* out;
      TF_RETURN_IF_ERROR(
          c->Concatenate(in, c->Vector(c->Dim(in, rank - 1)), &out));
      c->set_output(0, out);
      return Status::OK();
    })
    .Doc(R"doc(
Returns a batched diagonal tensor with a given batched diagonal values.

Given a `diagonal`, this operation returns a tensor with the `diagonal` and
everything else padded with zeros. The diagonal is computed as follows:

Assume `diagonal` has `k` dimensions `[I, J, K, ..., N]`, then the output is a
tensor of rank `k+1` with dimensions [I, J, K, ..., N, N]` where:

`output[i, j, k, ..., m, n] = 1{m=n} * diagonal[i, j, k, ..., n]`.

For example:

```prettyprint
# 'diagonal' is [[1, 2, 3, 4], [5, 6, 7, 8]]

and diagonal.shape = (2, 4)

tf.batch_matrix_diag(diagonal) ==> [[[1, 0, 0, 0]
                                     [0, 2, 0, 0]
                                     [0, 0, 3, 0]
                                     [0, 0, 0, 4]],
                                    [[5, 0, 0, 0]
                                     [0, 6, 0, 0]
                                     [0, 0, 7, 0]
                                     [0, 0, 0, 8]]]

which has shape (2, 4, 4)
```

diagonal: Rank `k`, where `k >= 1`.
output: Rank `k+1`, with `output.shape = diagonal.shape + [diagonal.shape[-1]]`.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("BatchMatrixSetDiag")
    .Input("input: T")
    .Input("diagonal: T")
    .Output("output: T")
    .Attr("T: type")
    .SetShapeFn([](InferenceContext* c) {
      const Shape* input;
      const Shape* diag;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 2, &input));
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(1), 1, &diag));

      const Dimension* square_dim;
      TF_RETURN_IF_ERROR(
          c->Merge(c->Dim(input, -2), c->Dim(input, -1), &square_dim));
      TF_RETURN_IF_ERROR(c->Merge(square_dim, c->Dim(diag, -1), &square_dim));

      const Shape* output;
      TF_RETURN_IF_ERROR(c->Concatenate(diag, c->Vector(square_dim), &output));
      TF_RETURN_IF_ERROR(c->Merge(input, output, &output));

      c->set_output(0, output);
      return Status::OK();
    })
    .Doc(R"doc(
Returns a batched matrix tensor with new batched diagonal values.

Given `input` and `diagonal`, this operation returns a tensor with the
same shape and values as `input`, except for the diagonals of the innermost
matrices.  These will be overwritten by the values in `diagonal`.
The batched matrices must be square.

The output is computed as follows:

Assume `input` has `k+1` dimensions `[I, J, K, ..., N, N]` and `diagonal` has
`k` dimensions `[I, J, K, ..., N]`.  Then the output is a
tensor of rank `k+1` with dimensions [I, J, K, ..., N, N]` where:

  * `output[i, j, k, ..., m, n] = diagonal[i, j, k, ..., n]` for `m == n`.
  * `output[i, j, k, ..., m, n] = input[i, j, k, ..., m, n]` for `m != n`.

input: Rank `k+1`, where `k >= 1`.
diagonal: Rank `k`, where `k >= 1`.
output: Rank `k+1`, with `output.shape = input.shape`.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("BatchMatrixDiagPart")
    .Input("input: T")
    .Output("diagonal: T")
    .Attr("T: type")
    .SetShapeFn([](InferenceContext* c) {
      const Shape* in;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 2, &in));
      if (!c->RankKnown(in)) {
        c->set_output(0, c->UnknownShape());
        return Status::OK();
      }
      const int32 rank = c->Rank(in);
      // Last two dims must match.
      const Dimension* unused;
      TF_RETURN_IF_ERROR(
          c->Merge(c->Dim(in, rank - 1), c->Dim(in, rank - 2), &unused));

      // Output shape has all dims but last of input.
      std::vector<const Dimension*> dims;
      for (int i = 0; i < rank - 1; ++i) dims.push_back(c->Dim(in, i));
      c->set_output(0, c->MakeShape(dims));
      return Status::OK();
    })
    .Doc(R"doc(
Returns the batched diagonal part of a batched tensor.

This operation returns a tensor with the `diagonal` part
of the batched `input`. The `diagonal` part is computed as follows:

Assume `input` has `k` dimensions `[I, J, K, ..., N, N]`, then the output is a
tensor of rank `k - 1` with dimensions `[I, J, K, ..., N]` where:

`diagonal[i, j, k, ..., n] = input[i, j, k, ..., n, n]`.

The input must be at least a matrix.

For example:

```prettyprint
# 'input' is [[[1, 0, 0, 0]
               [0, 2, 0, 0]
               [0, 0, 3, 0]
               [0, 0, 0, 4]],
              [[5, 0, 0, 0]
               [0, 6, 0, 0]
               [0, 0, 7, 0]
               [0, 0, 0, 8]]]

and input.shape = (2, 4, 4)

tf.batch_matrix_diag_part(input) ==> [[1, 2, 3, 4], [5, 6, 7, 8]]

which has shape (2, 4)
```

input: Rank `k` tensor where `k >= 2` and the last two dimensions are equal.
diagonal: The extracted diagonal(s) having shape
  `diagonal.shape = input.shape[:-1]`.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("BatchMatrixBandPart")
    .Input("input: T")
    .Input("num_lower: int64")
    .Input("num_upper: int64")
    .Output("band: T")
    .Attr("T: type")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
Copy a tensor setting everything outside a central band in each innermost matrix
to zero.

The `band` part is computed as follows:
Assume `input` has `k` dimensions `[I, J, K, ..., M, N]`, then the output is a
tensor with the same shape where

`band[i, j, k, ..., m, n] = in_band(m, n) * input[i, j, k, ..., m, n]`.

The indicator function 'in_band(m, n)` is one if
`(num_lower < 0 || (m-n) <= num_lower)) &&
(num_upper < 0 || (n-m) <= num_upper)`, and zero otherwise.

For example:

```prettyprint
# if 'input' is [[ 0,  1,  2, 3]
                 [-1,  0,  1, 2]
                 [-2, -1,  0, 1]
                 [-3, -2, -1, 0]],

tf.batch_matrix_band_part(input, 1, -1) ==> [[ 0,  1,  2, 3]
                                             [-1,  0,  1, 2]
                                             [ 0, -1,  0, 1]
                                             [ 0,  0, -1, 0]],

tf.batch_matrix_band_part(input, 2, 1) ==> [[ 0,  1,  0, 0]
                                            [-1,  0,  1, 0]
                                            [-2, -1,  0, 1]
                                            [ 0, -2, -1, 0]]
```

Useful special cases:

```prettyprint
 tf.batch_matrix_band_part(input, 0, -1) ==> Upper triangular part.
 tf.batch_matrix_band_part(input, -1, 0) ==> Lower triangular part.
 tf.batch_matrix_band_part(input, 0, 0) ==> Diagonal.
```

input: Rank `k` tensor.
num_lower: 0-D tensor. Number of subdiagonals to keep. If negative, keep entire
           lower triangle.
num_upper: 0-D tensor. Number of superdiagonals to keep. If negative, keep
           entire upper triangle.
band: Rank `k` tensor of the same shape as input. The extracted banded tensor.

)doc");

// --------------------------------------------------------------------------
REGISTER_OP("Reverse")
    .Input("tensor: T")
    .Input("dims: bool")
    .Output("output: T")
    .Attr(
        "T: {uint8, int8, int32, bool, half, float, double, complex64, "
        "complex128}")
    .SetShapeFn([](InferenceContext* c) {
      const Shape* input = c->input(0);
      const Shape* dims;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &dims));
      const Dimension* dims_dim = c->Dim(dims, 0);
      if (c->ValueKnown(dims_dim)) {
        TF_RETURN_IF_ERROR(c->WithRank(input, c->Value(dims_dim), &input));
      }
      if (c->Rank(input) > 8) {
        return errors::InvalidArgument(
            "reverse does not work on tensors with more than 8 dimensions");
      }
      c->set_output(0, input);
      return Status::OK();
    })
    .Doc(R"Doc(
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

tensor: Up to 8-D.
dims: 1-D. The dimensions to reverse.
output: The same shape as `tensor`.
)Doc");

// --------------------------------------------------------------------------
REGISTER_OP("EditDistance")
    .Input("hypothesis_indices: int64")
    .Input("hypothesis_values: T")
    .Input("hypothesis_shape: int64")
    .Input("truth_indices: int64")
    .Input("truth_values: T")
    .Input("truth_shape: int64")
    .Attr("normalize: bool = true")
    .Attr("T: type")
    .Output("output: float")
    .SetShapeFn([](InferenceContext* c) {
      const Tensor* hypothesis_shape_t = c->input_tensor(2);
      const Tensor* truth_shape_t = c->input_tensor(5);
      if (hypothesis_shape_t == nullptr || truth_shape_t == nullptr) {
        // We need to know the runtime shape of the two tensors,
        // or else the output shape is unknown.
        return shape_inference::UnknownShape(c);
      }

      if (hypothesis_shape_t->NumElements() != truth_shape_t->NumElements()) {
        return errors::InvalidArgument(
            "Num elements of hypothesis_shape does not match truth_shape: ",
            hypothesis_shape_t->NumElements(), " vs. ",
            truth_shape_t->NumElements());
      }

      auto h_values = hypothesis_shape_t->flat<int64>();
      auto t_values = truth_shape_t->flat<int64>();
      std::vector<const Dimension*> dims(hypothesis_shape_t->NumElements() - 1);
      for (int i = 0; i < dims.size(); ++i) {
        dims[i] = c->MakeDim(std::max(h_values(i), t_values(i)));
      }

      c->set_output(0, c->MakeShape(dims));
      return Status::OK();
    })
    .Doc(R"doc(
Computes the (possibly normalized) Levenshtein Edit Distance.

The inputs are variable-length sequences provided by SparseTensors
  (hypothesis_indices, hypothesis_values, hypothesis_shape)
and
  (truth_indices, truth_values, truth_shape).

The inputs are:

hypothesis_indices: The indices of the hypothesis list SparseTensor.
  This is an N x R int64 matrix.
hypothesis_values: The values of the hypothesis list SparseTensor.
  This is an N-length vector.
hypothesis_shape: The shape of the hypothesis list SparseTensor.
  This is an R-length vector.
truth_indices: The indices of the truth list SparseTensor.
  This is an M x R int64 matrix.
truth_values: The values of the truth list SparseTensor.
  This is an M-length vector.
truth_shape: The shape of the truth list SparseTensor.
  This is an R-length vector.
truth_shape: truth indices, vector.
normalize: boolean (if true, edit distances are normalized by length of truth).

The output is:

output: A dense float tensor with rank R - 1.

For the example input:

    // hypothesis represents a 2x1 matrix with variable-length values:
    //   (0,0) = ["a"]
    //   (1,0) = ["b"]
    hypothesis_indices = [[0, 0, 0],
                          [1, 0, 0]]
    hypothesis_values = ["a", "b"]
    hypothesis_shape = [2, 1, 1]

    // truth represents a 2x2 matrix with variable-length values:
    //   (0,0) = []
    //   (0,1) = ["a"]
    //   (1,0) = ["b", "c"]
    //   (1,1) = ["a"]
    truth_indices = [[0, 1, 0],
                     [1, 0, 0],
                     [1, 0, 1],
                     [1, 1, 0]]
    truth_values = ["a", "b", "c", "a"]
    truth_shape = [2, 2, 2]
    normalize = true

The output will be:

    // output is a 2x2 matrix with edit distances normalized by truth lengths.
    output = [[inf, 1.0],  // (0,0): no truth, (0,1): no hypothesis
              [0.5, 1.0]]  // (1,0): addition, (1,1): no hypothesis
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("Fill")
    .Input("dims: int32")
    .Input("value: T")
    .Output("output: T")
    .Attr("T: type")
    .SetShapeFn([](InferenceContext* c) {
      const Shape* out;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(0, &out));
      c->set_output(0, out);
      return Status::OK();
    })
    .Doc(R"doc(
Creates a tensor filled with a scalar value.

This operation creates a tensor of shape `dims` and fills it with `value`.

For example:

```prettyprint
# Output tensor has shape [2, 3].
fill([2, 3], 9) ==> [[9, 9, 9]
                     [9, 9, 9]]
```

dims: 1-D. Represents the shape of the output tensor.
value: 0-D (scalar). Value to fill the returned tensor.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("Gather")
    .Input("params: Tparams")
    .Input("indices: Tindices")
    .Attr("validate_indices: bool = true")
    .Output("output: Tparams")
    .Attr("Tparams: type")
    .Attr("Tindices: {int32,int64}")
    .SetShapeFn([](InferenceContext* c) {
      const Shape* unused;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 1, &unused));
      const Shape* params_subshape;
      TF_RETURN_IF_ERROR(c->Subshape(c->input(0), 1, &params_subshape));
      const Shape* indices_shape = c->input(1);
      const Shape* out;
      TF_RETURN_IF_ERROR(c->Concatenate(indices_shape, params_subshape, &out));
      c->set_output(0, out);
      return Status::OK();
    })
    .Doc(R"doc(
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
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("GatherNd")
    .Input("params: Tparams")
    .Input("indices: Tindices")
    .Output("output: Tparams")
    .Attr("Tparams: type")
    .Attr("Tindices: {int32,int64}")
    .SetShapeFn([](InferenceContext* c) {
      const Shape* params = c->input(0);
      const Shape* indices;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(1), 1, &indices));
      const Dimension* r_dim = c->Dim(indices, -1);

      if (!c->RankKnown(params) || !c->ValueKnown(r_dim)) {
        c->set_output(0, c->UnknownShape());
        return Status::OK();
      }

      if (c->Value(r_dim) > c->Rank(params)) {
        return errors::InvalidArgument(
            "indices.shape[-1] must be <= params.rank, but saw indices shape: ",
            c->DebugString(indices), " and params shape: ",
            c->DebugString(params));
      }

      // Remove r_dim from indices to get output.
      const Shape* indices_slice;
      const Shape* params_slice;
      TF_RETURN_IF_ERROR(c->Subshape(indices, 0, -1, &indices_slice));
      TF_RETURN_IF_ERROR(c->Subshape(params, c->Value(r_dim), &params_slice));
      const Shape* out;
      TF_RETURN_IF_ERROR(c->Concatenate(indices_slice, params_slice, &out));
      c->set_output(0, out);
      return Status::OK();
    })
    .Doc(R"doc(
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


params: `M-D`.  The tensor from which to gather values.
indices: `(N+1)-D`.  Index tensor having shape `[d_0, ..., d_N, R]`.
output: `(N+M-R)-D`.  Values from `params` gathered from indices given by
  `indices`.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("Identity")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: type")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"Doc(
Return a tensor with the same shape and contents as the input tensor or value.
)Doc");

// --------------------------------------------------------------------------
REGISTER_OP("RefIdentity")
    .Input("input: Ref(T)")
    .Output("output: Ref(T)")
    .Attr("T: type")
    .SetShapeFn(shape_inference::UnchangedShape)
    .SetAllowsUninitializedInput()
    .Doc(R"Doc(
Return the same ref tensor as the input ref tensor.
)Doc");

// --------------------------------------------------------------------------
REGISTER_OP("StopGradient")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: type")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"Doc(
Stops gradient computation.

When executed in a graph, this op outputs its input tensor as-is.

When building ops to compute gradients, this op prevents the contribution of
its inputs to be taken into account.  Normally, the gradient generator adds ops
to a graph to compute the derivatives of a specified 'loss' by recursively
finding out inputs that contributed to its computation.  If you insert this op
in the graph it inputs are masked from the gradient generator.  They are not
taken into account for computing gradients.

This is useful any time you want to compute a value with TensorFlow but need
to pretend that the value was a constant. Some examples include:

*  The *EM* algorithm where the *M-step* should not involve backpropagation
   through the output of the *E-step*.
*  Contrastive divergence training of Boltzmann machines where, when
   differentiating the energy function, the training must not backpropagate
   through the graph that generated the samples from the model.
*  Adversarial training, where no backprop should happen through the adversarial
   example generation process.
)Doc");

// --------------------------------------------------------------------------
REGISTER_OP("CheckNumerics")
    .Input("tensor: T")
    .Output("output: T")
    .Attr("T: {half, float, double}")
    .Attr("message: string")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
Checks a tensor for NaN and Inf values.

When run, reports an `InvalidArgument` error if `tensor` has any values
that are not a number (NaN) or infinity (Inf). Otherwise, passes `tensor` as-is.

message: Prefix of the error message.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("Reshape")
    .Input("tensor: T")
    .Input("shape: int32")
    .Output("output: T")
    .Attr("T: type")
    .SetShapeFn([](InferenceContext* c) {
      const Shape* in = c->input(0);
      const Shape* out;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(1, &out));

      // If the rank and all dimensions of the input tensor are known, we may
      // infer missing shape information or perform shape checks.
      // NumElements conveniently returns kUnknownDim upon missing rank or
      // dimension information.
      // Additionally, if the rank of the out shape is unknown we have no shape
      // information to go off of.
      const Dimension* num_in_elems = c->NumElements(in);
      const Dimension* num_out_elems = c->NumElements(out);
      if (!c->ValueKnown(num_in_elems) || !c->RankKnown(out)) {
        // Do nothing. We have no shape information to infer from so we directly
        // return out as our shape.
      } else if (c->ValueKnown(num_out_elems)) {
        // If we know the number of output elements, we ensure that they
        // are equal to the number of input elements.
        if (c->Value(num_in_elems) != c->Value(num_out_elems)) {
          return errors::InvalidArgument(
              "Cannot reshape a tensor with ", c->DebugString(num_in_elems),
              " elements to shape ", c->DebugString(out), " (",
              c->DebugString(num_out_elems), " elements)");
        }
      } else {
        // If we don't know the number of output elements, we can infer
        // the missing dimension.
        int32 unknown_idx = -1;
        const Dimension* known_elems = c->MakeDim(1);
        for (int32 i = 0; i < c->Rank(out); ++i) {
          const Dimension* dim = c->Dim(out, i);
          if (!c->ValueKnown(dim)) {
            if (unknown_idx >= 0) {
              return errors::InvalidArgument(
                  "Cannot infer multiple unknown dimensions in shape ",
                  c->DebugString(out));
            }
            unknown_idx = i;
          } else {
            TF_RETURN_IF_ERROR(c->Multiply(known_elems, dim, &known_elems));
          }
        }
        const Dimension* inferred_dim;
        TF_RETURN_IF_ERROR(
            c->Divide(num_in_elems, c->Value(known_elems), &inferred_dim));
        TF_RETURN_IF_ERROR(c->ReplaceDim(out, unknown_idx, inferred_dim, &out));
      }

      c->set_output(0, out);
      return Status::OK();
    })
    .Doc(R"Doc(
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

shape: Defines the shape of the output tensor.
)Doc");

// --------------------------------------------------------------------------
REGISTER_OP("InvertPermutation")
    .Input("x: int32")
    .Output("y: int32")
    .SetShapeFn([](InferenceContext* c) {
      const Shape* x;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &x));
      c->set_output(0, x);
      return Status::OK();
    })
    .Doc(R"doc(
Computes the inverse permutation of a tensor.

This operation computes the inverse of an index permutation. It takes a 1-D
integer tensor `x`, which represents the indices of a zero-based array, and
swaps each value with its index position. In other words, for an output tensor
`y` and an input tensor `x`, this operation computes the following:

`y[x[i]] = i for i in [0, 1, ..., len(x) - 1]`

The values must include 0. There can be no duplicate values or negative values.

For example:

```prettyprint
# tensor `x` is [3, 4, 0, 2, 1]
invert_permutation(x) ==> [2, 4, 3, 0, 1]
```

x: 1-D.
y: 1-D.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("Transpose")
    .Input("x: T")
    .Input("perm: int32")
    .Output("y: T")
    .Attr("T: type")
    .SetShapeFn([](InferenceContext* c) {
      const Shape* input = c->input(0);
      const Shape* perm_shape = c->input(1);
      const Tensor* perm = c->input_tensor(1);
      const Dimension* perm_elems = c->NumElements(perm_shape);
      // If we don't have rank information on the input or value information on
      // perm we can't return any shape information, otherwise we have enough
      // information to at least find the rank of the output.
      if (!c->RankKnown(input) && !c->ValueKnown(perm_elems) &&
          perm == nullptr) {
        c->set_output(0, c->UnknownShape());
        return Status::OK();
      }

      // Find our value of the rank.
      int64 rank;
      if (c->RankKnown(input)) {
        rank = c->Rank(input);
      } else if (c->ValueKnown(perm_elems)) {
        rank = c->Value(perm_elems);
      } else {
        rank = perm->NumElements();
      }
      std::vector<const Dimension*> dims;
      dims.resize(rank);
      TF_RETURN_IF_ERROR(c->WithRank(input, rank, &input));
      // Ensure that perm is a vector and has rank elements.
      TF_RETURN_IF_ERROR(c->WithRank(perm_shape, 1, &perm_shape));
      TF_RETURN_IF_ERROR(c->WithValue(perm_elems, rank, &perm_elems));

      // If we know the rank of the input and the value of perm, we can return
      // all shape informantion, otherwise we can only return rank information,
      // but no information for the dimensions.
      if (perm != nullptr) {
        auto flat_perm = perm->flat<int32>();
        for (int32 i = 0; i < rank; ++i) {
          int32 in_idx = flat_perm(i);
          if (in_idx >= rank) {
            return errors::InvalidArgument(
                "perm dim ", in_idx, " is out of range of input rank ", rank);
          }
          dims[i] = c->Dim(input, in_idx);
        }
      } else {
        for (int i = 0; i < rank; ++i) {
          dims[i] = c->UnknownDim();
        }
      }

      c->set_output(0, c->MakeShape(dims));
      return Status::OK();
    })
    .Doc(R"doc(
Shuffle dimensions of x according to a permutation.

The output `y` has the same rank as `x`. The shapes of `x` and `y` satisfy:
  `y.shape[i] == x.shape[perm[i]] for i in [0, 1, ..., rank(x) - 1]`
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("Unique")
    .Input("x: T")
    .Output("y: T")
    .Output("idx: int32")
    .Attr("T: type")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->Vector(InferenceContext::kUnknownDim));
      c->set_output(1, c->input(0));
      return Status::OK();
    })
    .Doc(R"doc(
Finds unique elements in a 1-D tensor.

This operation returns a tensor `y` containing all of the unique elements of `x`
sorted in the same order that they occur in `x`. This operation also returns a
tensor `idx` the same size as `x` that contains the index of each value of `x`
in the unique output `y`. In other words:

`y[idx[i]] = x[i] for i in [0, 1,...,rank(x) - 1]`

For example:

```prettyprint
# tensor 'x' is [1, 1, 2, 4, 4, 4, 7, 8, 8]
y, idx = unique(x)
y ==> [1, 2, 4, 7, 8]
idx ==> [0, 0, 1, 2, 2, 2, 3, 4, 4]
```

x: 1-D.
y: 1-D.
idx: 1-D.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("UniqueWithCounts")
    .Input("x: T")
    .Output("y: T")
    .Output("idx: int32")
    .Output("count: int32")
    .Attr("T: type")
    .SetShapeFn([](InferenceContext* c) {
      auto* uniq = c->Vector(InferenceContext::kUnknownDim);
      c->set_output(0, uniq);
      c->set_output(1, c->input(0));
      c->set_output(2, uniq);
      return Status::OK();
    })
    .Doc(R"doc(
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

x: 1-D.
y: 1-D.
idx: 1-D.
count: 1-D.
)doc");

namespace {

Status ShapeShapeFn(InferenceContext* c) {
  for (int i = 0; i < c->num_inputs(); ++i) {
    const Dimension* dim;
    if (c->RankKnown(c->input(i))) {
      dim = c->MakeDim(c->Rank(c->input(i)));
    } else {
      dim = c->UnknownDim();
    }
    c->set_output(i, c->Vector(dim));
  }
  return Status::OK();
}

}  // namespace

// --------------------------------------------------------------------------
REGISTER_OP("Shape")
    .Input("input: T")
    .Output("output: int32")
    .Attr("T: type")
    .SetShapeFn(ShapeShapeFn)
    .Doc(R"doc(
Returns the shape of a tensor.

This operation returns a 1-D integer tensor representing the shape of `input`.

For example:

```prettyprint
# 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
shape(t) ==> [2, 2, 3]
```

)doc");

REGISTER_OP("ShapeN")
    .Input("input: N * T")
    .Output("output: N * int32")
    .Attr("N: int")
    .Attr("T: type")
    .SetShapeFn(ShapeShapeFn)
    .Doc(R"doc(
Returns shape of tensors.

This operation returns N 1-D integer tensors representing shape of `input[i]s`.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("ReverseSequence")
    .Input("input: T")
    .Input("seq_lengths: int64")
    .Output("output: T")
    .Attr("seq_dim: int")
    .Attr("batch_dim: int = 0")
    .Attr("T: type")
    .SetShapeFn([](InferenceContext* c) {
      const Shape* input = c->input(0);
      const Shape* seq_lens_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &seq_lens_shape));

      int64 seq_dim;
      TF_RETURN_IF_ERROR(c->GetAttr("seq_dim", &seq_dim));
      int64 batch_dim;
      TF_RETURN_IF_ERROR(c->GetAttr("batch_dim", &batch_dim));

      if (!c->RankKnown(input)) {
        return shape_inference::UnknownShape(c);
      }

      // Validate batch_dim and seq_dim against input.
      const int32 input_rank = c->Rank(input);
      if (batch_dim >= input_rank) {
        return errors::InvalidArgument("batch_dim must be < input rank: ",
                                       batch_dim, " vs. ", input_rank);
      }
      if (seq_dim >= input_rank) {
        return errors::InvalidArgument("seq_dim must be < input rank: ",
                                       seq_dim, " vs. ", input_rank);
      }

      const Dimension* batch_dim_dim = c->Dim(input, batch_dim);
      TF_RETURN_IF_ERROR(
          c->Merge(batch_dim_dim, c->Dim(seq_lens_shape, 0), &batch_dim_dim));

      // Replace batch_dim of input with batch_size
      const Shape* output_shape;
      TF_RETURN_IF_ERROR(
          c->ReplaceDim(input, batch_dim, batch_dim_dim, &output_shape));
      c->set_output(0, output_shape);
      return Status::OK();
    })
    .Doc(R"doc(
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

input: The input to reverse.
seq_lengths: 1-D with length `input.dims(batch_dim)` and
  `max(seq_lengths) < input.dims(seq_dim)`
seq_dim: The dimension which is partially reversed.
batch_dim: The dimension along which reversal is performed.
output: The partially reversed input. It has the same shape as `input`.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("Rank")
    .Input("input: T")
    .Output("output: int32")
    .Attr("T: type")
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Returns the rank of a tensor.

This operation returns an integer representing the rank of `input`.

For example:

```prettyprint
# 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
# shape of tensor 't' is [2, 2, 3]
rank(t) ==> 3
```

**Note**: The rank of a tensor is not the same as the rank of a matrix. The rank
of a tensor is the number of indices required to uniquely select each element
of the tensor. Rank is also known as "order", "degree", or "ndims."
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("Size")
    .Input("input: T")
    .Output("output: int32")
    .Attr("T: type")
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Returns the size of a tensor.

This operation returns an integer representing the number of elements in
`input`.

For example:

```prettyprint
# 't' is [[[1, 1,, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]]
size(t) ==> 12
```

)doc");

// --------------------------------------------------------------------------
REGISTER_OP("Slice")
    .Input("input: T")
    .Input("begin: Index")
    .Input("size: Index")
    .Output("output: T")
    .Attr("T: type")
    .Attr("Index: {int32,int64}")
    .Doc(R"doc(
Return a slice from 'input'.

The output tensor is a tensor with dimensions described by 'size'
whose values are extracted from 'input' starting at the offsets in
'begin'.

*Requirements*:
  0 <= begin[i] <= begin[i] + size[i] <= Di  for i in [0, n)

begin: begin[i] specifies the offset into the 'i'th dimension of
  'input' to slice from.
size: size[i] specifies the number of elements of the 'i'th dimension
  of 'input' to slice. If size[i] is -1, all remaining elements in dimension
  i are included in the slice (i.e. this is equivalent to setting
  size[i] = input.dim_size(i) - begin[i]).
)doc");

REGISTER_OP("StridedSlice")
    .Input("input: T")
    .Input("begin: Index")
    .Input("end: Index")
    .Input("strides: Index")
    .Output("output: T")
    .Attr("T: type")
    .Attr("Index: {int32, int64}")
    .Attr("begin_mask: int = 0")
    .Attr("end_mask: int = 0")
    .Attr("ellipsis_mask: int = 0")
    .Attr("new_axis_mask: int = 0")
    .Attr("shrink_axis_mask: int = 0")
    .Doc(R"doc(
Return a strided slice from `input`.

The output tensor is a tensor with dimensions implied by `begin`,
`end`, and `strides`, whose values are extracted from `begin`.

Specifically, the result tensor at index `(i[0], i[1], ..., i[n-1])`
will obtain the value `input[begin[0] + i[0] * stride[0], ..., `
                            `begin[n-1] + i[n-1] * stride[n-1])]`.

*Requirements*:
  `0 != strides[i] for i in [0, n)`

begin: `begin[i]` specifies the offset into the `i`th dimension of
  `input` to slice from.
end: `end[i]` specifies the first offset into the `i`th dimension of
  `input` that will not be extracted. Out or range values are
  clamped to `[0,dim[i]) if slice[i] > 0` or `[-1,dim[i]-1]`
  `if slice[i] < 0`
strides: `strides[i]` specifies the increment in the `i`th dimension
  after extracting a given element. Negative indices will reverse
  the original order. Out or range values are
  clamped to `[0,dim[i]) if slice[i]>0` or `[-1,dim[i]-1] if slice[i] < 0`
begin_mask: a bitmask where a bit i being 1 means to ignore the begin
  value and instead use the largest interval possible. At runtime
  begin[i] will be replaced with `[0, n-1) if `stride[i] > 0` or
  `[-1, n-1]` if `stride[i] < 0`
end_mask: analogous to `begin_mask`
ellipsis_mask: a bitmask where bit `i` being 1 means the `i`th
  position is actually an ellipsis. One bit at most can be 1.
new_axis_mask: a bitmask where bit `i` being 1 means the `i`th
  position creates a dimension in the tensor of length 1. Thus
  the total number of elements remain unchanged but the shape
  gets a 1 in the appropriate position.
shrink_axis_mask: a bitmask where bit `i` implies that the `i`th
  position should shrink the dimensionality. begin and end
  must imply a slice of size 1 in the dimension. For example in
  python one might do `foo[:,3,:]` which would result in
  `shrink_axis_mask` being 2.
)doc");

REGISTER_OP("StridedSliceGrad")
    .Input("shape: Index")
    .Input("begin: Index")
    .Input("end: Index")
    .Input("strides: Index")
    .Input("dy: T")
    .Output("output: T")
    .Attr("T: type")
    .Attr("Index: {int32, int64}")
    .Attr("begin_mask: int = 0")
    .Attr("end_mask: int = 0")
    .Attr("ellipsis_mask: int = 0")
    .Attr("new_axis_mask: int = 0")
    .Attr("shrink_axis_mask: int = 0")
    .Doc(R"doc(
Returns the gradient of `StridedSlice`.

Since `StridedSlice` cuts out pieces of its `input` which is size
`shape`, its gradient will have the same shape (which is passed here
as `shape`). The gradient will be zero in any element that the slice
does not select.

Arguments are the same as StridedSliceGrad with the exception that
`dy` is the input gradient to be propagated and `shape` is the
shape of `StridedSlice`'s `input`.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("Tile")
    .Input("input: T")
    .Input("multiples: int32")
    .Output("output: T")
    .Attr("T: type")
    .SetShapeFn([](InferenceContext* c) {
      const Shape* input;
      const Shape* multiples;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &multiples));
      const Dimension* multiples_dim0 = c->Dim(multiples, 0);
      if (!c->ValueKnown(multiples_dim0)) {
        // Length of multiples vector unknown, so output is unknown.
        //
        // NOTE: we could potentially merge the input rank with the
        // multiples length.
        return shape_inference::UnknownShape(c);
      }

      int32 rank = c->Value(multiples_dim0);
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), rank, &input));
      const Tensor* multiples_t = c->input_tensor(1);
      if (multiples_t == nullptr) {
        // If multiples vector isn't available, we only know the
        // output rank, not the sizes.
        std::vector<const Dimension*> dims;
        for (int64 i = 0; i < rank; ++i) {
          dims.push_back(c->UnknownDim());
        }
        c->set_output(0, c->MakeShape(dims));
        return Status::OK();
      }

      // Multiply each input dimension by its corresponding value
      // from the multiples tensor.
      auto multiples_data = multiples_t->vec<int32>();
      std::vector<const Dimension*> dims(rank);
      for (int i = 0; i < rank; ++i) {
        const int32 multiple = multiples_data(i);
        TF_RETURN_IF_ERROR(c->Multiply(c->Dim(input, i), multiple, &dims[i]));
      }
      c->set_output(0, c->MakeShape(dims));
      return Status::OK();
    })
    .Doc(R"doc(
Constructs a tensor by tiling a given tensor.

This operation creates a new tensor by replicating `input` `multiples` times.
The output tensor's i'th dimension has `input.dims(i) * multiples[i]` elements,
and the values of `input` are replicated `multiples[i]` times along the 'i'th
dimension. For example, tiling `[a b c d]` by `[2]` produces
`[a b c d a b c d]`.

input: 1-D or higher.
multiples: 1-D. Length must be the same as the number of dimensions in `input`
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("TileGrad")
    .Input("input: T")
    .Input("multiples: int32")
    .Output("output: T")
    .Attr("T: type")
    .Deprecated(3, "TileGrad has been replaced with reduce_sum")
    .Doc(R"doc(
Returns the gradient of `Tile`.

Since `Tile` takes an input and repeats the input `multiples` times
along each dimension, `TileGrad` takes in `multiples` and aggregates
each repeated tile of `input` into `output`.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("Where")
    .Input("input: bool")
    .Output("index: int64")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->Matrix(c->UnknownDim(), c->Rank(c->input(0))));
      return Status::OK();
    })
    .Doc(R"doc(
Returns locations of true values in a boolean tensor.

This operation returns the coordinates of true elements in `input`. The
coordinates are returned in a 2-D tensor where the first dimension (rows)
represents the number of true elements, and the second dimension (columns)
represents the coordinates of the true elements. Keep in mind, the shape of
the output tensor can vary depending on how many true values there are in
`input`. Indices are output in row-major order.

For example:

```prettyprint
# 'input' tensor is [[True, False]
#                    [True, False]]
# 'input' has two true values, so output has two coordinates.
# 'input' has rank of 2, so coordinates have two indices.
where(input) ==> [[0, 0],
                  [1, 0]]

# `input` tensor is [[[True, False]
#                     [True, False]]
#                    [[False, True]
#                     [False, True]]
#                    [[False, False]
#                     [False, True]]]
# 'input' has 5 true values, so output has 5 coordinates.
# 'input' has rank of 3, so coordinates have three indices.
where(input) ==> [[0, 0, 0],
                  [0, 1, 0],
                  [1, 0, 1],
                  [1, 1, 1],
                  [2, 1, 1]]
```

)doc");

// --------------------------------------------------------------------------
REGISTER_OP("BroadcastGradientArgs")
    .Input("s0: int32")
    .Input("s1: int32")
    .Output("r0: int32")
    .Output("r1: int32")
    .SetShapeFn([](InferenceContext* c) {
      // TODO(mrry): Implement constant_value for BroadcastGradientArgs?
      const Shape* unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &unused));
      c->set_output(0, c->Vector(InferenceContext::kUnknownDim));
      c->set_output(1, c->Vector(InferenceContext::kUnknownDim));
      return Status::OK();
    })
    .Doc(R"doc(
Return the reduction indices for computing gradients of s0 op s1 with broadcast.

This is typically used by gradient computations for a broadcasting operation.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("Pad")
    .Input("input: T")
    .Input("paddings: int32")
    .Output("output: T")
    .Attr("T: type")
    .SetShapeFn(PadShapeFn)
    .Doc(R"doc(
Pads a tensor with zeros.

This operation pads a `input` with zeros according to the `paddings` you
specify. `paddings` is an integer tensor with shape `[Dn, 2]`, where n is the
rank of `input`. For each dimension D of `input`, `paddings[D, 0]` indicates
how many zeros to add before the contents of `input` in that dimension, and
`paddings[D, 1]` indicates how many zeros to add after the contents of `input`
in that dimension.

The padded size of each dimension D of the output is:

`paddings(D, 0) + input.dim_size(D) + paddings(D, 1)`

For example:

```prettyprint
# 't' is [[1, 1], [2, 2]]
# 'paddings' is [[1, 1], [2, 2]]
# rank of 't' is 2
pad(t, paddings) ==> [[0, 0, 0, 0, 0, 0]
                      [0, 0, 1, 1, 0, 0]
                      [0, 0, 2, 2, 0, 0]
                      [0, 0, 0, 0, 0, 0]]
```

)doc");

// --------------------------------------------------------------------------
REGISTER_OP("MirrorPad")
    .Input("input: T")
    .Input("paddings: int32")
    .Output("output: T")
    .Attr("T: type")
    .Attr(GetMirrorPadModeAttrString())
    .SetShapeFn(PadShapeFn)
    .Doc(R"doc(
Pads a tensor with mirrored values.

This operation pads a `input` with mirrored values according to the `paddings`
you specify. `paddings` is an integer tensor with shape `[n, 2]`, where n is
the rank of `input`. For each dimension D of `input`, `paddings[D, 0]` indicates
how many values to add before the contents of `input` in that dimension, and
`paddings[D, 1]` indicates how many values to add after the contents of `input`
in that dimension. Both `paddings[D, 0]` and `paddings[D, 1]` must be no greater
than `input.dim_size(D)` (or `input.dim_size(D) - 1`) if `copy_border` is true
(if false, respectively).

The padded size of each dimension D of the output is:

`paddings(D, 0) + input.dim_size(D) + paddings(D, 1)`

For example:

```prettyprint
# 't' is [[1, 2, 3], [4, 5, 6]].
# 'paddings' is [[1, 1]], [2, 2]].
# 'mode' is SYMMETRIC.
# rank of 't' is 2.
pad(t, paddings) ==> [[2, 1, 1, 2, 3, 3, 2]
                      [2, 1, 1, 2, 3, 3, 2]
                      [5, 4, 4, 5, 6, 6, 5]
                      [5, 4, 4, 5, 6, 6, 5]]
```

input: The input tensor to be padded.
paddings: A two-column matrix specifying the padding sizes. The number of
  rows must be the same as the rank of `input`.
mode: Either `REFLECT` or `SYMMETRIC`. In reflect mode the padded regions
  do not include the borders, while in symmetric mode the padded regions
  do include the borders. For example, if `input` is `[1, 2, 3]` and `paddings`
  is `[0, 2]`, then the output is `[1, 2, 3, 2, 1]` in reflect mode, and
  it is `[1, 2, 3, 3, 2]` in symmetric mode.
output: The padded tensor.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("MirrorPadGrad")
    .Input("input: T")
    .Input("paddings: int32")
    .Output("output: T")
    .Attr("T: type")
    .Attr(GetMirrorPadModeAttrString())
    .SetShapeFn([](InferenceContext* c) {
      const Shape* paddings;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &paddings));
      const Dimension* pad_0 = c->Dim(paddings, 0);
      if (!c->ValueKnown(pad_0)) {
        // We don't know the rank of the output since the first
        // padding dimension is unknown.
        c->set_output(0, c->UnknownShape());
        return Status::OK();
      }

      int64 input_rank = c->Value(pad_0);
      const Shape* input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), input_rank, &input));
      TF_RETURN_IF_ERROR(
          c->Merge(paddings, c->Matrix(input_rank, 2), &paddings));

      const Tensor* paddings_t = c->input_tensor(1);
      if (paddings_t == nullptr) {
        // Values of 'paddings' is not available, but we know the
        // input rank, so return the rank of the output with unknown
        // dimensions.
        std::vector<const Dimension*> dims;
        for (int64 i = 0; i < input_rank; ++i) dims.push_back(c->UnknownDim());
        c->set_output(0, c->MakeShape(dims));
        return Status::OK();
      }

      auto paddings_data = paddings_t->matrix<int32>();
      std::vector<const Dimension*> dims(input_rank);
      for (int i = 0; i < input_rank; ++i) {
        const int64 pad0 = static_cast<int64>(paddings_data(i, 0));
        const int64 pad1 = static_cast<int64>(paddings_data(i, 1));
        if (pad0 < 0 || pad1 < 0) {
          return errors::InvalidArgument("Paddings must be non-negative");
        }

        TF_RETURN_IF_ERROR(
            c->Subtract(c->Dim(input, i), pad0 + pad1, &dims[i]));
      }
      c->set_output(0, c->MakeShape(dims));
      return Status::OK();
    })
    .Doc(R"doc(
Gradient op for `MirrorPad` op. This op folds a mirror-padded tensor.

This operation folds the padded areas of `input` by `MirrorPad` according to the
`paddings` you specify. `paddings` must be the same as `paddings` argument
given to the corresponding `MirrorPad` op.

The folded size of each dimension D of the output is:

`input.dim_size(D) - paddings(D, 0) - paddings(D, 1)`

For example:

```prettyprint
# 't' is [[1, 2, 3], [4, 5, 6], [7, 8, 9]].
# 'paddings' is [[0, 1]], [0, 1]].
# 'mode' is SYMMETRIC.
# rank of 't' is 2.
pad(t, paddings) ==> [[ 1,  5]
                      [11, 28]]
```

input: The input tensor to be folded.
paddings: A two-column matrix specifying the padding sizes. The number of
  rows must be the same as the rank of `input`.
mode: The mode used in the `MirrorPad` op.
output: The folded tensor.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("Placeholder")
    .Output("output: dtype")
    .Attr("dtype: type")
    .Attr("shape: shape = {}")
    .SetShapeFn([](InferenceContext* c) {
      PartialTensorShape shape;
      TF_RETURN_IF_ERROR(c->GetAttr("shape", &shape));

      // Placeholder has a legacy bug where we cannot tell
      // the difference between a scalar shape attribute and
      // 'unknown shape'.  So if the shape is a scalar, we return
      // an unknown shape.
      if (shape.dims() == 0) {
        return shape_inference::UnknownShape(c);
      }

      TensorShapeProto shape_proto;
      shape.AsProto(&shape_proto);
      const Shape* out;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeProto(shape_proto, &out));
      c->set_output(0, out);
      return Status::OK();
    })
    .Doc(R"doc(
A placeholder op for a value that will be fed into the computation.

N.B. This operation will fail with an error if it is executed. It is
intended as a way to represent a value that will always be fed, and to
provide attrs that enable the fed value to be checked at runtime.

output: A placeholder tensor that must be replaced using the feed mechanism.
dtype: The type of elements in the tensor.
shape: (Optional) The shape of the tensor. If the shape has 0 dimensions, the
  shape is unconstrained.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("PlaceholderWithDefault")
    .Input("input: dtype")
    .Output("output: dtype")
    .Attr("dtype: type")
    .Attr("shape: shape")
    .SetShapeFn([](InferenceContext* c) {
      const Shape* input = c->input(0);
      PartialTensorShape shape;
      TF_RETURN_IF_ERROR(c->GetAttr("shape", &shape));
      TensorShapeProto shape_proto;
      shape.AsProto(&shape_proto);
      const Shape* out;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeProto(shape_proto, &out));

      // We merge for compatibility checking, but return the output,
      // since output_shape may be less precise than input_shape.
      const Shape* unused;
      TF_RETURN_IF_ERROR(c->Merge(input, out, &unused));
      c->set_output(0, out);
      return Status::OK();
    })
    .Doc(R"doc(
A placeholder op that passes though `input` when its output is not fed.

input: The default value to produce when `output` is not fed.
output: A placeholder tensor that defaults to `input` if it is not fed.
dtype: The type of elements in the tensor.
shape: The (possibly partial) shape of the tensor.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("ExpandDims")
    .Input("input: T")
    .Input("dim: int32")
    .Output("output: T")
    .Attr("T: type")
    .SetShapeFn([](InferenceContext* c) {
      const Shape* input = c->input(0);
      const Shape* expand_dim;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &expand_dim));

      const Tensor* dim_t = c->input_tensor(1);
      if (dim_t == nullptr || !c->RankKnown(input)) {
        c->set_output(0, c->UnknownShape());
        return Status::OK();
      }
      int32 which_dim = dim_t->flat<int32>()(0);
      if (which_dim < 0) {
        which_dim += c->Rank(input) + 1;
      }

      const Shape* end;
      TF_RETURN_IF_ERROR(c->Subshape(input, which_dim, &end));

      // Build output as start + 1 + end.
      const Shape* output;
      TF_RETURN_IF_ERROR(c->Subshape(input, 0, which_dim, &output));
      TF_RETURN_IF_ERROR(c->Concatenate(output, c->Vector(1), &output));
      TF_RETURN_IF_ERROR(c->Concatenate(output, end, &output));
      c->set_output(0, output);
      return Status::OK();
    })
    .Doc(R"doc(
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

dim: 0-D (scalar). Specifies the dimension index at which to
  expand the shape of `input`.
output: Contains the same data as `input`, but its shape has an additional
  dimension of size 1 added.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("Squeeze")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: type")
    .Attr("squeeze_dims: list(int) >= 0 = []")
    .SetShapeFn([](InferenceContext* c) {
      const Shape* input = c->input(0);
      if (!c->RankKnown(input)) {
        // Input shape unknown.
        return shape_inference::UnknownShape(c);
      }

      const int32 input_rank = c->Rank(input);

      // Validate and wrap squeeze dimensions.
      std::vector<int32> squeeze_dims;
      TF_RETURN_IF_ERROR(c->GetAttr("squeeze_dims", &squeeze_dims));
      for (int i = 0; i < squeeze_dims.size(); ++i) {
        if (squeeze_dims[i] < -input_rank || squeeze_dims[i] >= input_rank) {
          return errors::InvalidArgument("squeeze_dims[", i, "] not in [",
                                         -input_rank, ",", input_rank, ").");
        }

        if (squeeze_dims[i] < 0) {
          squeeze_dims[i] += input_rank;
        }
      }

      std::vector<const Dimension*> result_shape;
      for (int i = 0; i < input_rank; ++i) {
        // True if squeeze_dims contains an entry to squeeze this
        // dimension.
        bool is_explicit_match =
            std::find(squeeze_dims.begin(), squeeze_dims.end(), i) !=
            squeeze_dims.end();

        const Dimension* dim = c->Dim(input, i);

        if (!c->ValueKnown(dim)) {
          // Assume that the squeezed dimension will be 1 at runtime.
          if (is_explicit_match) continue;

          // If squeezing all 1 dimensions, and we see an unknown value,
          // give up and return Unknown Shape.
          if (squeeze_dims.empty()) {
            c->set_output(0, c->UnknownShape());
            return Status::OK();
          }
        } else if (c->Value(dim) == 1) {
          if (is_explicit_match || squeeze_dims.empty()) {
            // If explicitly squeezing, or squeezing all 1s, remove
            // this dimension.
            continue;
          }
        } else if (is_explicit_match) {
          return errors::InvalidArgument("Can not squeeze dim[", i,
                                         "], expected a dimension of 1, got ",
                                         c->Value(c->Dim(input, i)));
        }

        result_shape.emplace_back(dim);
      }

      c->set_output(0, c->MakeShape(result_shape));
      return Status::OK();
    })
    .Doc(R"doc(
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

input: The `input` to squeeze.
squeeze_dims: If specified, only squeezes the dimensions listed. The dimension
  index starts at 0. It is an error to squeeze a dimension that is not 1.
output: Contains the same data as `input`, but has one or more dimensions of
  size 1 removed.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("ListDiff")
    .Input("x: T")
    .Input("y: T")
    .Output("out: T")
    .Output("idx: int32")
    .Attr("T: type")
    .SetShapeFn([](InferenceContext* c) {
      const Shape* unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &unused));
      // TODO(mrry): Indicate that the length falls within an interval?
      const Shape* out = c->Vector(InferenceContext::kUnknownDim);
      c->set_output(0, out);
      c->set_output(1, out);
      return Status::OK();
    })
    .Doc(R"doc(
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

x: 1-D. Values to keep.
y: 1-D. Values to remove.
out: 1-D. Values present in `x` but not in `y`.
idx: 1-D. Positions of `x` values preserved in `out`.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("SpaceToBatch")
    .Input("input: T")
    .Input("paddings: int32")
    .Output("output: T")
    .Attr("T: type")
    .Attr("block_size: int >= 2")
    .Doc(R"doc(
SpaceToBatch for 4-D tensors of type T.

Zero-pads and then rearranges (permutes) blocks of spatial data into batch.
More specifically, this op outputs a copy of the input tensor where values from
the `height` and `width` dimensions are moved to the `batch` dimension. After
the zero-padding, both `height` and `width` of the input must be divisible by the
block size.

input: 4-D with shape `[batch, height, width, depth]`.

paddings: 2-D tensor of non-negative integers with shape `[2, 2]`. It specifies
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
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("BatchToSpace")
    .Input("input: T")
    .Input("crops: int32")
    .Output("output: T")
    .Attr("T: type")
    .Attr("block_size: int >= 2")
    .Doc(R"doc(
BatchToSpace for 4-D tensors of type T.

Rearranges (permutes) data from batch into blocks of spatial data, followed by
cropping. This is the reverse transformation of SpaceToBatch. More specifically,
this op outputs a copy of the input tensor where values from the `batch`
dimension are moved in spatial blocks to the `height` and `width` dimensions,
followed by cropping along the `height` and `width` dimensions.

input: 4-D tensor with shape
 `[batch*block_size*block_size, height_pad/block_size, width_pad/block_size,
   depth]`. Note that the batch size of the input tensor must be divisible by
 `block_size * block_size`.

crops: 2-D tensor of non-negative integers with shape `[2, 2]`. It specifies
  how many elements to crop from the intermediate result across the spatial
  dimensions as follows:

      crops = [[crop_top, crop_bottom], [crop_left, crop_right]]

output: 4-D with shape `[batch, height, width, depth]`, where:

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
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("SpaceToDepth")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: type")
    .Attr("block_size: int >= 2")
    .Doc(R"doc(
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

block_size: The size of the spatial block.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("DepthToSpace")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: type")
    .Attr("block_size: int >= 2")
    .Doc(R"doc(
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

block_size: The size of the spatial block, same as in Space2Depth.
)doc");

// --------------------------------------------------------------------------

REGISTER_OP("ExtractImagePatches")
    .Input("images: T")
    .Output("patches: T")
    .Attr("ksizes: list(int) >= 4")
    .Attr("strides: list(int) >= 4")
    .Attr("rates: list(int) >= 4")
    .Attr("T: realnumbertype")
    .Attr(GetPaddingAttrString())
    .SetShapeFn([](InferenceContext* c) {
      const Shape* input_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input_shape));

      std::vector<int32> ksizes;
      TF_RETURN_IF_ERROR(c->GetAttr("ksizes", &ksizes));
      if (ksizes.size() != 4) {
        return errors::InvalidArgument(
            "ExtractImagePatches requires the ksizes attribute to contain 4 "
            "values, but got: ",
            ksizes.size());
      }

      std::vector<int32> strides;
      TF_RETURN_IF_ERROR(c->GetAttr("strides", &strides));
      if (strides.size() != 4) {
        return errors::InvalidArgument(
            "ExtractImagePatches requires the stride attribute to contain 4 "
            "values, but got: ",
            strides.size());
      }

      std::vector<int32> rates;
      TF_RETURN_IF_ERROR(c->GetAttr("rates", &rates));
      if (rates.size() != 4) {
        return errors::InvalidArgument(
            "ExtractImagePatches requires the rates attribute to contain 4 "
            "values, but got: ",
            rates.size());
      }

      int32 ksize_rows = ksizes[1];
      int32 ksize_cols = ksizes[2];

      int32 stride_rows = strides[1];
      int32 stride_cols = strides[2];

      int32 rate_rows = rates[1];
      int32 rate_cols = rates[2];

      int32 ksize_rows_eff = ksize_rows + (ksize_rows - 1) * (rate_rows - 1);
      int32 ksize_cols_eff = ksize_cols + (ksize_cols - 1) * (rate_cols - 1);

      const Dimension* batch_size_dim = c->Dim(input_shape, 0);
      const Dimension* in_rows_dim = c->Dim(input_shape, 1);
      const Dimension* in_cols_dim = c->Dim(input_shape, 2);
      const Dimension* output_depth_dim = c->Dim(input_shape, 3);

      // At the moment we need to know the values of several fields.
      TF_RETURN_IF_ERROR(c->ValidateKnownDim(in_rows_dim, "in_rows"));
      TF_RETURN_IF_ERROR(c->ValidateKnownDim(in_cols_dim, "in_cols"));
      auto in_rows = c->Value(in_rows_dim);
      auto in_cols = c->Value(in_cols_dim);

      Padding padding;
      TF_RETURN_IF_ERROR(c->GetAttr("padding", &padding));

      int64 output_rows, output_cols;
      int64 padding_before, padding_after;
      TF_RETURN_IF_ERROR(GetWindowedOutputSizeVerbose(
          in_rows, ksize_rows_eff, stride_rows, padding, &output_rows,
          &padding_before, &padding_after));
      TF_RETURN_IF_ERROR(GetWindowedOutputSizeVerbose(
          in_cols, ksize_cols_eff, stride_cols, padding, &output_cols,
          &padding_before, &padding_after));
      const Shape* output_shape = c->MakeShape(
          {batch_size_dim, output_rows, output_cols, output_depth_dim});
      c->set_output(0, output_shape);
      return Status::OK();
    })
    .Doc(R"doc(
Extract `patches` from `images` and put them in the "depth" output dimension.

images: 4-D Tensor with shape `[batch, in_rows, in_cols, depth]`.
patches: 4-D Tensor with shape `[batch, out_rows, out_cols, ksize_rows *
  ksize_cols * depth]` containing image patches with size
  `ksize_rows x ksize_cols x depth` vectorized in the "depth" dimension.
ksizes: The size of the sliding window for each dimension of `images`.
strides: 1-D of length 4. How far the centers of two consecutive patches are in
  the images. Must be: `[1, stride_rows, stride_cols, 1]`.
rates: 1-D of length 4. Must be: `[1, rate_rows, rate_cols, 1]`. This is the
  input stride, specifying how far two consecutive patch samples are in the
  input. Equivalent to extracting patches with
  `patch_sizes_eff = patch_sizes + (patch_sizes - 1) * (rates - 1), followed by
  subsampling them spatially by a factor of `rates`.
padding: The type of padding algorithm to use.

We specify the size-related attributes as:

      ksizes = [1, ksize_rows, ksize_cols, 1]
      strides = [1, strides_rows, strides_cols, 1]
      rates = [1, rates_rows, rates_cols, 1]
)doc");

// --------------------------------------------------------------------------

REGISTER_OP("Bitcast")
    .Input("input: T")
    .Output("output: type")
    .Attr("T: numbertype")
    .Attr("type: numbertype")
    .SetShapeFn([](InferenceContext* c) {
      const Shape* input = c->input(0);
      if (!c->RankKnown(input)) {
        // Input shape unknown.
        return shape_inference::UnknownShape(c);
      }

      // Find the size of the input and output data types.
      DataType input_type;
      DataType output_type;
      TF_RETURN_IF_ERROR(c->GetAttr("T", &input_type));
      TF_RETURN_IF_ERROR(c->GetAttr("type", &output_type));
      const int input_type_size = DataTypeSize(input_type);
      const int output_type_size = DataTypeSize(output_type);

      if (input_type_size == 0 || output_type_size == 0) {
        return errors::InvalidArgument("Cannot bitcast types ",
                                       DataTypeString(input_type), " to ",
                                       DataTypeString(output_type),
                                       " because "
                                       "one of the type sizes is zero.");
      }

      const Shape* new_shape;
      if (input_type_size == output_type_size) {
        // No change in size.
        new_shape = input;
      } else if (input_type_size < output_type_size) {
        TF_RETURN_IF_ERROR(c->WithRankAtLeast(input, 1, &new_shape));

        int64 divisor_val = output_type_size / input_type_size;
        const Dimension* last_dim = c->Dim(new_shape, -1);
        if (!c->ValueKnown(last_dim) || c->Value(last_dim) == divisor_val) {
          TF_RETURN_IF_ERROR(c->Subshape(new_shape, 0, -1, &new_shape));
        } else {
          return errors::InvalidArgument("Cannot bitcast due to shape. ",
                                         c->Value(last_dim), " does not match ",
                                         divisor_val);
        }
      } else {
        // Input type size is larger than output type size.
        int64 divisor_val = input_type_size / output_type_size;
        const Shape* extension = c->Vector(divisor_val);
        TF_RETURN_IF_ERROR(c->Concatenate(input, extension, &new_shape));
      }

      c->set_output(0, new_shape);
      return Status::OK();
    })
    .Doc(R"doc(
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
)doc");

REGISTER_OP("OneHot")
    .Input("indices: TI")
    .Input("depth: int32")
    .Input("on_value: T")
    .Input("off_value: T")
    .Attr("axis: int = -1")
    .Output("output: T")
    .Attr("T: type")
    .Attr("TI: {uint8, int32, int64} = DT_INT64")
    .SetShapeFn([](InferenceContext* c) {
      int32 axis;
      TF_RETURN_IF_ERROR(c->GetAttr("axis", &axis));
      if (axis < -1) return errors::InvalidArgument("axis must be >= -1");

      const Dimension* depth;
      TF_RETURN_IF_ERROR(c->MakeDimForScalarInput(1, &depth));

      const Shape* indices = c->input(0);
      if (!c->RankKnown(indices)) return shape_inference::UnknownShape(c);

      int32 new_rank = c->Rank(indices) + 1;
      // We need to add new_rank to axis in the case the axis is -1 because
      // C++ returns negative values from % if the dividend is negative.
      int32 depth_index = (axis + new_rank) % new_rank;
      // Out shape is indices[0:depth_index] + [depth] + indices[depth_index:].
      const Shape* front;
      const Shape* back;
      const Shape* out;
      TF_RETURN_IF_ERROR(c->Subshape(indices, 0, depth_index, &front));
      TF_RETURN_IF_ERROR(c->Subshape(indices, depth_index, &back));
      TF_RETURN_IF_ERROR(c->Concatenate(front, c->Vector(depth), &front));
      TF_RETURN_IF_ERROR(c->Concatenate(front, back, &out));
      c->set_output(0, out);
      return Status::OK();
    })
    .Doc(R"doc(
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

indices: A tensor of indices.
depth: A scalar defining the depth of the one hot dimension.
on_value: A scalar defining the value to fill in output when `indices[j] = i`.
off_value: A scalar defining the value to fill in output when `indices[j] != i`.
axis: The axis to fill (default: -1, a new inner-most axis).
output: The one-hot tensor.
)doc");

// EXPERIMENTAL. DO NOT USE OR DEPEND ON THIS YET.
REGISTER_OP("QuantizeAndDequantize")
    .Input("input: T")
    .Attr("signed_input: bool = true")
    .Attr("num_bits: int = 8")
    .Attr("range_given: bool = false")
    .Attr("input_min: float = 0")
    .Attr("input_max: float = 0")
    .Output("output: T")
    .Attr("T: {float, double}")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
Quantizes then dequantizes a tensor.

This op simulates the precision loss from the quantized forward pass by:
1. Quantizing the tensor to fixed point numbers, which should match the target
   quantization method when it is used in inference.
2. Dequantizing it back to floating point numbers for the following ops, most
   likely matmul.

There are different ways to quantize. This version does not use the full range
of the output type, choosing to elide the lowest possible value for symmetry
(e.g., output range is -127 to 127, not -128 to 127 for signed 8 bit
quantization), so that 0.0 maps to 0.

To perform this op, we first find the range of values in our tensor. The range
we use is always centered on 0, so we find m such that

1. m = max(abs(input_min), abs(input_max)) if range_given is true,
2. m = max(max(abs(min_elem(input)), abs(max_elem(input))) otherwise.

Our input tensor range is then [-m, m].

Next, we choose our fixed-point quantization buckets, [min_fixed, max_fixed].
If signed_input is true, this is

  [min_fixed, max_fixed ] =
      [-(1 << (num_bits - 1) - 1), (1 << (num_bits - 1)) - 1].

Otherwise, if signed_input is false, the fixed-point range is

  [min_fixed, max_fixed] = [0, (1 << num_bits) - 1].

From this we compute our scaling factor, s:

  s = (max_fixed - min_fixed) / (2 * m).

Now we can quantize and dequantize the elements of our tensor.  An element e
is transformed into e':

  e' = (e * s).round_to_nearest() / s.

Note that we have a different number of buckets in the signed vs. unsigned
cases.  For example, if num_bits == 8, we get 254 buckets in the signed case
vs. 255 in the unsigned case.

For example, suppose num_bits = 8 and m = 1.  Then

  [min_fixed, max_fixed] = [-127, 127], and
  s = (127 + 127) / 2 = 127.

Given the vector {-1, -0.5, 0, 0.3}, this is quantized to
{-127, -63, 0, 38}, and dequantized to {-1, -63.0/127, 0, 38.0/127}.

input: Tensor to quantize and then dequantize.
signed_input: If the quantization is signed or unsigned.
num_bits: The bitwidth of the quantization.
range_given: If the range is given or should be computed from the tensor.
input_min: If range is given, this is the min of the range.
input_max: If range is given, this is the max of the range.
)doc");

// EXPERIMENTAL: tfdb debugger-inserted ops.
REGISTER_OP("Copy")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: type")
    .Attr("tensor_name: string = ''")
    .Doc(R"doc(
Copy Op.

Performs CPU-to-CPU or GPU-to-GPU deep-copying of tensor, depending on the
device on which the tensor is allocated.

Unlike the CopyHost Op, this op does not have HostMemory constraint on its
input or output.

input: Input tensor.
output: Output tensor, deep-copied from input.
tensor_name: The name of the input tensor.
)doc");

REGISTER_OP("CopyHost")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: type")
    .Attr("tensor_name: string = ''")
    .Doc(R"doc(
Copy Host Op.

Performs CPU-to-CPU deep-copying of tensor.

Unlike the Copy Op, this op has HostMemory constraint on its input or output.

input: Input tensor.
output: Output tensor, deep-copied from input.
tensor_name: The name of the input tensor.
)doc");

REGISTER_OP("DebugIdentity")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: type")
    .Attr("tensor_name: string = ''")
    .Attr("debug_urls: list(string) = []")
    .Doc(R"doc(
Debug Identity Op.

Provides an identity mapping of the non-Ref type input tensor for debugging.

input: Input tensor, non-Reference type.
output: Output tensor that equals the input tensor.
tensor_name: Name of the input tensor.
debug_urls: List of URLs to debug targets, e.g.,
            file:///foo/tfdbg_dump, grpc:://localhost:11011
)doc");

REGISTER_OP("DebugNanCount")
    .Input("input: T")
    .Output("output: int64")  // The debug signal (nan count) is int64
    .Attr("T: type")
    .Attr("tensor_name: string = ''")
    .Attr("debug_urls: list(string) = []")
    .Doc(R"doc(
Debug NaN Value Counter Op

Counts number of NaNs in the input tensor, for debugging.

input: Input tensor, non-Reference type.
output: An integer output tensor that is the number of NaNs in the input.
tensor_name: Name of the input tensor.
debug_urls: List of URLs to debug targets, e.g.,
            file:///foo/tfdbg_dump, grpc:://localhost:11011
)doc");

}  // namespace tensorflow
