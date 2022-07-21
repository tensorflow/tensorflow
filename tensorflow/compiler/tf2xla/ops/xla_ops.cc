/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include <cstddef>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/errors.h"

// Note: Most of the operators defined in this module are used by the jax2tf
// converter (see go/jax2tf for details) and are used in SavedModel produced
// by jax2tf. Hence, we need to maintain backwards compatibility for these
// operators. Please reach out to the JAX team if you want to make changes.

namespace tensorflow {
namespace {

// Helper shape function for operators that return an output with the same rank
// as their first input.
Status UnchangedRank(shape_inference::InferenceContext* c) {
  if (c->RankKnown(c->input(0))) {
    c->set_output(0, c->UnknownShapeOfRank(c->Rank(c->input(0))));
  } else {
    c->set_output(0, c->input(0));
  }
  return OkStatus();
}

REGISTER_OP("XlaBroadcastHelper")
    .Input("lhs: T")
    .Input("rhs: T")
    .Input("broadcast_dims: Tindices")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Output("lhs_output: T")
    .Output("rhs_output: T")
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
Helper operator for performing XLA-style broadcasts

Broadcasts `lhs` and `rhs` to the same rank, by adding size 1 dimensions to
whichever of `lhs` and `rhs` has the lower rank, using XLA's broadcasting rules
for binary operators.

lhs: the LHS input tensor
rhs: the RHS input tensor
broadcast_dims: an XLA-style broadcast dimension specification
lhs_output: the broadcasted LHS tensor
rhs_output: the broadcasted RHS tensor
)doc");

REGISTER_OP("XlaSelfAdjointEig")
    .Input("a: T")
    .Attr("lower: bool")
    .Attr("max_iter: int")
    .Attr("epsilon: float")
    .Output("w: T")
    .Output("v: T")
    .SetShapeFn(shape_inference::UnknownShape)
    .Attr("T: numbertype")
    .Doc(R"doc(
Computes the eigen decomposition of a batch of self-adjoint matrices
(Note: Only real inputs are supported).

Computes the eigenvalues and eigenvectors of the innermost N-by-N matrices in
tensor such that tensor[...,:,:] * v[..., :,i] = e[..., i] * v[...,:,i], for
i=0...N-1.

a: the input tensor.

lower: a boolean specifies whether the calculation is done with the lower
  triangular part or the upper triangular part.

max_iter: maximum number of sweep update, i.e., the whole lower triangular
  part or upper triangular part based on parameter lower. Heuristically, it has
  been argued that approximately logN sweeps are needed in practice (Ref: Golub &
  van Loan "Matrix Computation").

epsilon: the tolerance ratio.

w: The eigenvalues in ascending order, each repeated according to its
  multiplicity.
v: The column v[..., :, i] is the normalized eigenvector corresponding to the
  eigenvalue w[..., i].
)doc");

REGISTER_OP("XlaSvd")
    .Input("a: T")
    .Attr("max_iter: int")
    .Attr("epsilon: float")
    .Attr("precision_config: string")
    .Output("s: T")
    .Output("u: T")
    .Output("v: T")
    .SetShapeFn(shape_inference::UnknownShape)
    .Attr("T: numbertype")
    .Doc(R"doc(
Computes the eigen decomposition of a batch of self-adjoint matrices
(Note: Only real inputs are supported).

Computes the eigenvalues and eigenvectors of the innermost M-by-N matrices in
tensor such that tensor[...,:,:] = u[..., :, :] * Diag(s[..., :]) * Transpose(v[...,:,:]).

a: the input tensor.

max_iter: maximum number of sweep update, i.e., the whole lower triangular
  part or upper triangular part based on parameter lower. Heuristically, it has
  been argued that approximately log(min (M, N)) sweeps are needed in practice
  (Ref: Golub & van Loan "Matrix Computation").

epsilon: the tolerance ratio.

precision_config: a serialized xla::PrecisionConfig proto.

s: Singular values. The values are sorted in reverse order of magnitude, so
  s[..., 0] is the largest value, s[..., 1] is the second largest, etc.
u: Left singular vectors.
v: Right singular vectors.
)doc");

REGISTER_OP("XlaConv")
    .Input("lhs: T")
    .Input("rhs: T")
    .Input("window_strides: Tindices")
    .Input("padding: Tindices")
    .Input("lhs_dilation: Tindices")
    .Input("rhs_dilation: Tindices")
    .Input("feature_group_count: Tindices")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("dimension_numbers: string")
    .Attr("precision_config: string")
    .Output("output: T")
    .SetShapeFn(UnchangedRank)
    .Doc(R"doc(
Wraps the XLA ConvGeneralDilated operator, documented at
 https://www.tensorflow.org/performance/xla/operation_semantics#conv_convolution
.

lhs: the input tensor
rhs: the kernel tensor
window_strides: the inter-window strides
padding: the padding to apply at the start and end of each input dimensions
lhs_dilation: dilation to apply between input elements
rhs_dilation: dilation to apply between kernel elements
feature_group_count: number of feature groups for grouped convolution.
dimension_numbers: a serialized xla::ConvolutionDimensionNumbers proto.
precision_config: a serialized xla::PrecisionConfig proto.
)doc");

REGISTER_OP("XlaConvV2")
    .Input("lhs: LhsT")
    .Input("rhs: RhsT")
    .Input("window_strides: Tindices")
    .Input("padding: Tindices")
    .Input("lhs_dilation: Tindices")
    .Input("rhs_dilation: Tindices")
    .Input("feature_group_count: Tindices")
    .Attr("LhsT: numbertype")
    .Attr("RhsT: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("dimension_numbers: string")
    .Attr("precision_config: string")
    .Attr("preferred_element_type: numbertype")
    .Attr("batch_group_count: int = 1")
    .Output("output: preferred_element_type")
    .SetShapeFn(UnchangedRank)
    .Doc(R"doc(
Wraps the XLA ConvGeneralDilated operator, documented at
 https://www.tensorflow.org/performance/xla/operation_semantics#conv_convolution
.

lhs: input tensor
rhs: kernel tensor
window_strides: inter-window strides
padding: padding to apply at the start and end of each input dimensions
lhs_dilation: dilation to apply between input elements
rhs_dilation: dilation to apply between kernel elements
feature_group_count: number of feature groups for grouped convolution.
dimension_numbers: serialized xla::ConvolutionDimensionNumbers proto.
precision_config: serialized xla::PrecisionConfig proto.
preferred_element_type: type of the tensor.
batch_group_count: number of batch groups or grouped filters.
)doc");

static Status XlaDotShapeFunction(shape_inference::InferenceContext* c) {
  shape_inference::ShapeHandle lhs_shape_handle = c->input(0);
  shape_inference::ShapeHandle rhs_shape_handle = c->input(1);
  if (!c->RankKnown(lhs_shape_handle) || !c->RankKnown(rhs_shape_handle)) {
    return shape_inference::UnknownShape(c);
  }

  string dimension_numbers_string;
  TF_RETURN_IF_ERROR(
      c->GetAttr("dimension_numbers", &dimension_numbers_string));

  xla::DotDimensionNumbers dimension_numbers;
  dimension_numbers.ParseFromString(dimension_numbers_string);

  // Check that number of contracting dimensions match.
  if (dimension_numbers.lhs_contracting_dimensions_size() !=
      dimension_numbers.rhs_contracting_dimensions_size())
    return errors::InvalidArgument(
        "Must specify the same number of contracting dimensions for lhs "
        "and rhs. Got: ",
        dimension_numbers.lhs_contracting_dimensions_size(), " and ",
        dimension_numbers.rhs_contracting_dimensions_size());

  // Check that contracting dimension sizes match.
  for (int64_t i = 0; i < dimension_numbers.lhs_contracting_dimensions_size();
       ++i) {
    const int64_t lhs_contracting_dimension =
        dimension_numbers.lhs_contracting_dimensions(i);
    const int64_t rhs_contracting_dimension =
        dimension_numbers.rhs_contracting_dimensions(i);
    shape_inference::DimensionHandle unused;
    TF_RETURN_WITH_CONTEXT_IF_ERROR(
        c->Merge(c->DimKnownRank(lhs_shape_handle, lhs_contracting_dimension),
                 c->DimKnownRank(rhs_shape_handle, rhs_contracting_dimension),
                 &unused),
        "For contracting dimension ", i, " which is lhs dimension ",
        lhs_contracting_dimension, " and rhs dimension ",
        rhs_contracting_dimension);
  }

  // Check that number of batch dimensions match.
  if (dimension_numbers.lhs_batch_dimensions_size() !=
      dimension_numbers.rhs_batch_dimensions_size())
    return errors::InvalidArgument(
        "Must specify the same number of batch dimensions for lhs "
        "and rhs. Got: ",
        dimension_numbers.lhs_batch_dimensions_size(), " and ",
        dimension_numbers.rhs_batch_dimensions_size());

  // The ranks of lhs and rhs are decremented by the number of contractions,
  // and added for the rank of the result. When an input tensor
  // is a scalar, its contribution to the rank of the result is 0. Generate
  // the result dimensions in order, batch dimensions, then the
  // non-contracted and non-batch lhs and rhs dimensions.
  std::vector<shape_inference::DimensionHandle> output_dims;

  // Check that batch dimension sizes match, and add them to output_dims.
  for (int64_t i = 0; i < dimension_numbers.lhs_batch_dimensions_size(); ++i) {
    const int64_t lhs_batch_dimension =
        dimension_numbers.lhs_batch_dimensions(i);
    const int64_t rhs_batch_dimension =
        dimension_numbers.rhs_batch_dimensions(i);
    shape_inference::DimensionHandle out;
    TF_RETURN_WITH_CONTEXT_IF_ERROR(
        c->Merge(c->DimKnownRank(lhs_shape_handle, lhs_batch_dimension),
                 c->DimKnownRank(rhs_shape_handle, rhs_batch_dimension), &out),
        "For batch dimension ", i, " which is lhs dimension ",
        lhs_batch_dimension, " and rhs dimension ", rhs_batch_dimension);
    output_dims.emplace_back(out);
  }

  const int32_t lhs_rank = c->Rank(lhs_shape_handle);
  for (int64_t i = 0; i < lhs_rank; ++i) {
    if (absl::c_linear_search(dimension_numbers.lhs_contracting_dimensions(),
                              i) ||
        absl::c_linear_search(dimension_numbers.lhs_batch_dimensions(), i)) {
      continue;
    }
    output_dims.emplace_back(c->Dim(lhs_shape_handle, i));
  }

  const int32_t rhs_rank = c->Rank(rhs_shape_handle);
  for (int64_t i = 0; i < rhs_rank; ++i) {
    if (absl::c_linear_search(dimension_numbers.rhs_contracting_dimensions(),
                              i) ||
        absl::c_linear_search(dimension_numbers.rhs_batch_dimensions(), i)) {
      continue;
    }
    output_dims.emplace_back(c->Dim(rhs_shape_handle, i));
  }

  c->set_output(0, c->MakeShape(output_dims));
  return OkStatus();
}

REGISTER_OP("XlaDot")
    .Input("lhs: T")
    .Input("rhs: T")
    .Attr("T: numbertype")
    .Attr("dimension_numbers: string")
    .Attr("precision_config: string")
    .Output("output: T")
    .SetShapeFn(XlaDotShapeFunction)
    .Doc(R"doc(
Wraps the XLA DotGeneral operator, documented at
 https://www.tensorflow.org/performance/xla/operation_semantics#dotgeneral
.

lhs: the LHS tensor
rhs: the RHS tensor
dimension_numbers: a serialized xla::DotDimensionNumbers proto.
precision_config: a serialized xla::PrecisionConfig proto.
)doc");

REGISTER_OP("XlaDotV2")
    .Input("lhs: LhsT")
    .Input("rhs: RhsT")
    .Attr("LhsT: numbertype")
    .Attr("RhsT: numbertype")
    .Attr("dimension_numbers: string")
    .Attr("precision_config: string")
    .Attr("preferred_element_type: numbertype")
    .Output("output: preferred_element_type")
    .SetShapeFn(XlaDotShapeFunction)
    .Doc(R"doc(
Wraps the XLA DotGeneral operator, documented at
 https://www.tensorflow.org/performance/xla/operation_semantics#dotgeneral
.

lhs: the LHS tensor
rhs: the RHS tensor
dimension_numbers: a serialized xla::DotDimensionNumbers proto.
precision_config: a serialized xla::PrecisionConfig proto.
preferred_element_type: The type of the tensor.
)doc");

REGISTER_OP("XlaSetBound")
    .Input("input: int32")
    .Input("bound: int32")
    .Output("output: int32")
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(
        R"doc(Set a bound for the given input value as a hint to Xla compiler,
        returns the same value.
)doc");

REGISTER_OP("XlaSetDynamicDimensionSize")
    .Input("input: T")
    .Input("dim_index: int32")
    .Input("size: int32")
    .Output("output: T")
    .Attr("T: type")
    // Use unknown shape to prevent constant folding.
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(
        R"doc(Make a static dimension into a xla bounded dynamic dimension.
        The current static dimension size will become the bound and the second
        operand becomes the dynamic size of the dimension.)doc");

REGISTER_OP("XlaRemoveDynamicDimensionSize")
    .Input("input: T")
    .Input("dim_index: int32")
    .Output("output: T")
    .Attr("T: type")
    // Use unknown shape to prevent constant folding.
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
Inverse of XlaSetDynamicDimensionSize.

Make an xla bounded dynamic dimension into a static dimension. The bound of the
size of dimension `dim_index` becomes the static dimension size.
)doc");

REGISTER_OP("XlaDynamicSlice")
    .Input("input: T")
    .Input("start_indices: Tindices")
    .Input("size_indices: Tindices")
    .Output("output: T")
    .Attr("T: type")
    .Attr("Tindices: {int32, int64}")
    .SetShapeFn([](shape_inference::InferenceContext* c) -> Status {
      shape_inference::ShapeHandle size_indices_shape = c->input(2);
      if (!c->RankKnown(size_indices_shape)) {
        return UnchangedRank(c);
      }
      if (c->Rank(size_indices_shape) != 1) {
        return errors::InvalidArgument("size_indices must be a 1D tensor");
      }
      shape_inference::ShapeHandle size_indices_value;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(2, &size_indices_value));
      if (!c->RankKnown(size_indices_value)) {
        // If we cannot tell the rank of the output from the value of
        // size_indices, perhaps we can find it from the rank of first operand.
        return UnchangedRank(c);
      }
      c->set_output(0, size_indices_value);
      return OkStatus();
    })
    .Doc(R"doc(
Wraps the XLA DynamicSlice operator, documented at
 https://www.tensorflow.org/performance/xla/operation_semantics#dynamicslice
.

DynamicSlice extracts a sub-array from the input array at dynamic
start_indices. The size of the slice in each dimension is passed in
size_indices, which specify the end point of exclusive slice intervals in each
dimension -- [start, start + size). The shape of start_indices must have rank 1,
with dimension size equal to the rank of operand.

input: A `Tensor` of type T.

start_indices: Rank 1 tensor of N integers containing the starting indices of
  the slice for each dimension. Value must be greater than or equal to zero.

start_indices: List of N integers containing the slice size for each
  dimension. Each value must be strictly greater than zero, and start + size
  must be less than or equal to the size of the dimension to avoid
  implementation defined behavior.
)doc");

REGISTER_OP("XlaDynamicUpdateSlice")
    .Input("input: T")
    .Input("update: T")
    .Input("indices: Tindices")
    .Output("output: T")
    .Attr("T: type")
    .Attr("Tindices: {int32, int64}")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
Wraps the XLA DynamicUpdateSlice operator, documented at
 https://www.tensorflow.org/performance/xla/operation_semantics#dynamicupdateslice
.

XlaDynamicUpdateSlice generates a result which is the value of the `input`
operand, with a slice update overwritten at `indices`. The shape of `update`
determines the shape of the sub-array of the result which is updated. The shape
of indices must be rank == 1, with dimension size equal to the rank of `input`.

Handling of out-of-bounds slice indices is implementation-defined.

input: A `Tensor` of type T.
indices: A vector of indices into `input`. Must have length equal to the rank of
  `input`.
update: A `Tensor` of type T. Same rank as `input`.
output: A `Tensor` of type T.
)doc");

// TODO(b/37549631) setting the If Op to always be stateful is too
// conservative.
REGISTER_OP("XlaIf")
    .Input("cond: Tcond")
    .Input("inputs: Tin")
    .Output("output: Tout")
    .Attr("Tcond: type")
    .Attr("then_branch: func")
    .Attr("else_branch: func")
    .Attr("Tin: list(type) >= 0")
    .Attr("Tout: list(type) >= 0")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
output = cond ? then_branch(inputs) : else_branch(inputs).

cond: A boolean scalar.
inputs: A list of input tensors.
output: A list of tensors returned by either then_branch(inputs) or
        else_branch(inputs). The input shapes of the then_branch and
        else_branch must match.
then_branch: A function takes 'inputs' and returns a list of tensors,
             whose types are the same as what else_branch returns.
else_branch: A function takes 'inputs' and returns a list of tensors.
             whose types are the same as what then_branch returns.
)doc");

REGISTER_OP("XlaPad")
    .Input("input: T")
    .Input("padding_value: T")
    .Input("padding_low: Tindices")
    .Input("padding_high: Tindices")
    .Input("padding_interior: Tindices")
    .Output("output: T")
    .Attr("T: type")
    .Attr("Tindices: {int32, int64}")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle input_shape_handle = c->input(0);
      if (!c->RankKnown(input_shape_handle)) {
        return UnchangedRank(c);
      }
      const int32_t op_rank = c->Rank(input_shape_handle);

      shape_inference::ShapeHandle padding_shape_handle = c->input(1);
      if (c->RankKnown(padding_shape_handle) &&
          c->Rank(padding_shape_handle) != 0) {
        return errors::InvalidArgument(
            "padding_value input must be scalar, found rank ",
            c->Rank(padding_shape_handle));
      }
      const Tensor* padding_low_tensor = c->input_tensor(2);
      const Tensor* padding_high_tensor = c->input_tensor(3);
      const Tensor* padding_interior_tensor = c->input_tensor(4);
      if (padding_low_tensor == nullptr || padding_high_tensor == nullptr ||
          padding_interior_tensor == nullptr) {
        return UnchangedRank(c);
      }

      if (padding_low_tensor->shape().dims() != 1 ||
          padding_low_tensor->shape().dim_size(0) != op_rank) {
        return errors::InvalidArgument(
            "padding_low must be a 1D tensor of size ", op_rank);
      }
      if (padding_high_tensor->shape().dims() != 1 ||
          padding_high_tensor->shape().dim_size(0) != op_rank) {
        return errors::InvalidArgument(
            "padding_high must be a 1D tensor of size ", op_rank);
      }
      if (padding_interior_tensor->shape().dims() != 1 ||
          padding_interior_tensor->shape().dim_size(0) != op_rank) {
        return errors::InvalidArgument(
            "padding_interior must be a 1D tensor of size ", op_rank);
      }
      std::vector<shape_inference::DimensionHandle> output_dims;
      output_dims.reserve(op_rank);
      for (int64_t i = 0; i < op_rank; ++i) {
        int64_t low, high, interior;
        TF_RETURN_IF_ERROR(c->GetScalarFromTensor(padding_low_tensor, i, &low));
        TF_RETURN_IF_ERROR(
            c->GetScalarFromTensor(padding_high_tensor, i, &high));
        TF_RETURN_IF_ERROR(
            c->GetScalarFromTensor(padding_interior_tensor, i, &interior));
        if (interior < 0) {
          return errors::InvalidArgument(
              "padding_interior must contain only non-negative values, found ",
              interior);
        }

        shape_inference::DimensionHandle orig_size_handle =
            c->Dim(input_shape_handle, i);
        if (c->ValueKnown(orig_size_handle)) {
          auto orig_dim = c->Value(orig_size_handle);
          int64_t new_dim = orig_dim + low + high;
          if (orig_dim > 0) {
            new_dim += interior * (orig_dim - 1);
          }
          if (new_dim < 0) {
            return errors::InvalidArgument(
                "resulting padded dimension has negative size ", new_dim);
          }
          output_dims.emplace_back(c->MakeDim(new_dim));
        } else {
          output_dims.emplace_back(c->UnknownDim());
        }
      }

      c->set_output(0, c->MakeShape(output_dims));
      return OkStatus();
    })
    .Doc(R"doc(
Wraps the XLA Pad operator, documented at
 https://www.tensorflow.org/performance/xla/operation_semantics#pad
.

input: A `Tensor` of type T.
padding_value: A scalar `Tensor` of type T.
padding_low: the padding to apply at the start of each input dimensions. Must
  be a compile-time constant 1D tensor of length equal to rank of input.
padding_high: the padding to apply at the end of each input dimension. Must
  be a compile-time constant 1D tensor of length equal to rank of input.
padding_interior: the padding to apply between each input element. Must
  be a compile-time constant 1D tensor of length equal to rank of input,
  containing only non-negative values.
output: A `Tensor` of type T.
)doc");

REGISTER_OP("XlaRecv")
    .Output("tensor: dtype")
    .Attr("dtype: type")
    .Attr("tensor_name: string")
    .Attr("shape: shape")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      TensorShape shape_attr;
      TF_RETURN_IF_ERROR(c->GetAttr("shape", &shape_attr));
      shape_inference::ShapeHandle s;
      TF_RETURN_IF_ERROR(c->MakeShapeFromTensorShape(shape_attr, &s));
      c->set_output(0, s);
      return OkStatus();
    })
    .Doc(R"doc(
Receives the named tensor from another XLA computation. Wraps the XLA Recv
operator documented at
 https://www.tensorflow.org/performance/xla/operation_semantics#recv .

tensor: The tensor to receive.
dtype: The type of the tensor.
tensor_name: A string key that identifies the channel.
shape: The shape of the tensor.
)doc");

REGISTER_OP("XlaReduce")
    .Input("input: T")
    .Input("init_value: T")
    .Attr("T: {numbertype, bool}")
    .Attr("dimensions_to_reduce: list(int)")
    .Attr("reducer: func")
    .Output("output: T")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      if (c->RankKnown(c->input(0))) {
        int rank = c->Rank(c->input(0));
        std::vector<int64_t> dimensions_to_reduce;
        TF_RETURN_IF_ERROR(
            c->GetAttr("dimensions_to_reduce", &dimensions_to_reduce));
        std::set<int64_t> dims_set(dimensions_to_reduce.begin(),
                                   dimensions_to_reduce.end());
        auto dim_in_range = [rank](int64_t dim) {
          return dim >= 0 && dim < rank;
        };
        const int dimensions_to_reduce_size = dimensions_to_reduce.size();
        if (rank < dimensions_to_reduce_size ||
            dims_set.size() != dimensions_to_reduce.size() ||
            !absl::c_all_of(dimensions_to_reduce, dim_in_range)) {
          return errors::InvalidArgument(
              "Invalid dimensions_to_reduce argument to XlaReduce");
        }
        c->set_output(
            0, c->UnknownShapeOfRank(rank - dimensions_to_reduce.size()));
      } else {
        c->set_output(0, c->input(0));
      }
      return OkStatus();
    })
    .Doc(R"doc(
Wraps the XLA Reduce operator, documented at
 https://www.tensorflow.org/performance/xla/operation_semantics#reduce .

input: the input tensor
init_value: a scalar representing the initial value for the reduction
reducer: a reducer function to apply
dimensions_to_reduce: dimension numbers over which to reduce
)doc");

REGISTER_OP("XlaVariadicReduce")
    .Input("input: N * T")
    .Input("init_value: N * T")
    .Attr("N: int >= 1")
    .Attr("T: {numbertype, bool}")
    .Attr("dimensions_to_reduce: list(int)")
    .Attr("reducer: func")
    .Output("output: N * T")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      int n;
      TF_RETURN_IF_ERROR(c->GetAttr("N", &n));
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
          c->MergeInput(i, c->input(j));
        }
      }
      if (c->RankKnown(c->input(0))) {
        int rank = c->Rank(c->input(0));
        std::vector<int64_t> dimensions_to_reduce;
        TF_RETURN_IF_ERROR(
            c->GetAttr("dimensions_to_reduce", &dimensions_to_reduce));
        std::set<int64_t> dims_set(dimensions_to_reduce.begin(),
                                   dimensions_to_reduce.end());
        auto dim_in_range = [rank](int64_t dim) {
          return dim >= 0 && dim < rank;
        };
        const int dimensions_to_reduce_size = dimensions_to_reduce.size();
        if (rank < dimensions_to_reduce_size ||
            dims_set.size() != dimensions_to_reduce.size() ||
            !absl::c_all_of(dimensions_to_reduce, dim_in_range)) {
          return errors::InvalidArgument(
              "Invalid dimensions_to_reduce argument to XlaVariadicReduce");
        }
        for (int i = 0; i < n; i++) {
          c->set_output(
              i, c->UnknownShapeOfRank(rank - dimensions_to_reduce.size()));
        }
      } else {
        for (int i = 0; i < n; i++) {
          c->set_output(i, c->input(i));
        }
      }
      return OkStatus();
    })
    .Doc(R"doc(
Wraps the variadic XLA Reduce operator.

Semantics are documented at
 https://www.tensorflow.org/performance/xla/operation_semantics#variadic_reduce.

This version is limited to operands of the same dtype.
XlaVariadicReduceV2 is a version that supports heterogeneous operands.

input: the input tensor(s)
init_value: scalar initial value(s) for the reduction
reducer: a reducer function to apply
dimensions_to_reduce: dimension numbers over which to reduce
)doc");

REGISTER_OP("XlaVariadicReduceV2")
    .Input("inputs: T")
    .Input("init_values: T")
    .Attr("T: list(type) >= 1")
    .Attr("dimensions_to_reduce: list(int)")
    .Attr("reducer: func")
    .Output("outputs: T")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      std::vector<shape_inference::ShapeHandle> input_shapes;
      TF_RETURN_IF_ERROR(c->input("inputs", &input_shapes));
      std::vector<shape_inference::ShapeHandle> init_values_shapes;
      TF_RETURN_IF_ERROR(c->input("init_values", &init_values_shapes));
      const int nr_inputs = input_shapes.size();
      if (nr_inputs != init_values_shapes.size()) {
        return errors::InvalidArgument(
            "Must specify the same number of inputs and init_values. ", "Got ",
            nr_inputs, " and ", init_values_shapes.size());
      }
      if (nr_inputs == 0) {
        return errors::InvalidArgument("Must specify at least one input");
      }

      shape_inference::ShapeHandle input_shape = input_shapes[0];
      for (int i = 1; i < nr_inputs; ++i) {
        shape_inference::ShapeHandle merged;
        TF_RETURN_WITH_CONTEXT_IF_ERROR(
            c->Merge(input_shape, input_shapes[i], &merged),
            "All inputs must have the same shape. Input ", i,
            " (zero-based) has shape ", c->DebugString(input_shapes[i]),
            " incompatible with the shape ", "inferred from previous inputs ",
            c->DebugString(input_shape));
        input_shape = merged;
      }
      // All outputs have the same shape
      shape_inference::ShapeHandle output_shape = c->UnknownShape();

      if (c->RankKnown(input_shape)) {
        int rank = c->Rank(input_shape);

        std::vector<int64_t> dimensions_to_reduce;
        TF_RETURN_IF_ERROR(
            c->GetAttr("dimensions_to_reduce", &dimensions_to_reduce));
        std::set<int64_t> dims_set(dimensions_to_reduce.begin(),
                                   dimensions_to_reduce.end());

        auto dim_in_range = [rank](int64_t dim) {
          return dim >= 0 && dim < rank;
        };
        const int dimensions_to_reduce_size = dimensions_to_reduce.size();
        if (rank < dimensions_to_reduce_size ||
            dims_set.size() != dimensions_to_reduce.size() ||
            !absl::c_all_of(dimensions_to_reduce, dim_in_range)) {
          return errors::InvalidArgument(
              "Invalid dimensions_to_reduce argument to XlaVariadicReduceV2");
        }

        std::vector<shape_inference::DimensionHandle> output_dims;
        for (int64_t i = 0; i < rank; ++i) {
          if (dims_set.find(i) == dims_set.end()) {
            output_dims.emplace_back(c->Dim(input_shape, i));
          }
        }
        output_shape = c->MakeShape(output_dims);
      }
      for (int i = 0; i < nr_inputs; ++i) {
        c->set_output(i, output_shape);
      }
      return OkStatus();
    })
    .Doc(R"doc(
Wraps the variadic XLA Reduce operator.

Semantics are documented at
 https://www.tensorflow.org/performance/xla/operation_semantics#variadic_reduce.

This is an expanded version of XlaVariadicReduce, with support for
operands of different dtypes, and improved shape inference.

inputs: the input tensor(s)
init_values: scalar initial value(s) for the reduction
reducer: a reducer function to apply
dimensions_to_reduce: dimension numbers over which to reduce
)doc");

REGISTER_OP("XlaReduceWindow")
    .Input("input: T")
    .Input("init_value: T")
    .Input("window_dimensions: Tindices")
    .Input("window_strides: Tindices")
    .Input("base_dilations: Tindices")
    .Input("window_dilations: Tindices")
    .Input("padding: Tindices")
    .Attr("T: {numbertype, bool}")
    .Attr("Tindices: {int32, int64}")
    .Attr("computation: func")
    .Output("output: T")
    .SetShapeFn(UnchangedRank)
    .Doc(R"doc(
Wraps the XLA ReduceWindow operator, documented at
 https://www.tensorflow.org/performance/xla/operation_semantics#reducewindow .

input: the input tensor
init_value: a scalar representing the initial value for the reduction
computation: a reducer function to apply
window_dimensions: the shape of the window
window_strides: the inter-window strides
padding: the padding to apply at the start and end of each input dimensions
)doc");

REGISTER_OP("XlaRngBitGenerator")
    .Input("algorithm: int32")
    .Input("initial_state: uint64")
    .Input("shape: Tshape")
    .Output("output_key: uint64")
    .Output("output: dtype")
    .Attr("dtype: {int32, int64, uint32, uint64} = DT_UINT64")
    .Attr("Tshape: {int32, int64} = DT_INT32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle algorithm;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &algorithm));
      shape_inference::ShapeHandle initial_state;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &initial_state));

      c->set_output(0, initial_state);
      shape_inference::ShapeHandle output;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(2, &output));
      c->set_output(1, output);
      return OkStatus();
    })
    .Doc(R"doc(
Stateless PRNG bit generator.
Wraps the XLA RngBitGenerator operator, documented at
 https://www.tensorflow.org/performance/xla/operation_semantics#rngbitgenerator.

algorithm: The PRNG algorithm to use, one of
  tf.random.Algorithm.{PHILOX, THREEFRY, AUTO_SELECT}.
initial_state: Initial state for the PRNG algorithm. For THREEFRY, it should be
  a u64[2] and for PHILOX a u64[3].
shape: The output shape of the generated data.
dtype: The type of the tensor.
)doc");

REGISTER_OP("XlaSelectAndScatter")
    .Input("operand: T")
    .Input("window_dimensions: Tindices")
    .Input("window_strides: Tindices")
    .Input("padding: Tindices")
    .Input("source: T")
    .Input("init_value: T")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("select: func")
    .Attr("scatter: func")
    .Output("output: T")
    .SetShapeFn(UnchangedRank)
    .Doc(R"doc(
Wraps the XLA SelectAndScatter operator, documented at
 https://www.tensorflow.org/performance/xla/operation_semantics#selectandscatter
.

operand: the input tensor
window_dimensions: the shape of the window
window_strides: the inter-window strides
padding: the padding to apply at the start and end of each input dimensions
source: a tensor of values to scatter
init_value: a scalar representing the initial value for the output tensor
select: a selection function to apply
scatter: a scatter function to apply
)doc");

REGISTER_OP("XlaSend")
    .Input("tensor: T")
    .Attr("T: type")
    .Attr("tensor_name: string")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
Sends the named tensor to another XLA computation. Wraps the XLA Send operator
documented at
 https://www.tensorflow.org/performance/xla/operation_semantics#send .

tensor: The tensor to send.
tensor_name: A string key that identifies the channel.
)doc");

REGISTER_OP("XlaSort")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: type")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
Wraps the XLA Sort operator, documented at
 https://www.tensorflow.org/performance/xla/operation_semantics#sort
.

Sorts a tensor. Currently only sorts in ascending order are supported.

input: A `Tensor` of type T.
output: A `Tensor` of type T.
)doc");

REGISTER_OP("XlaKeyValueSort")
    .Input("keys: K")
    .Input("values: V")
    .Output("sorted_keys: K")
    .Output("sorted_values: V")
    .Attr("K: realnumbertype")
    .Attr("V: type")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      c->set_output(1, c->input(1));
      return OkStatus();
    })
    .Doc(R"doc(
Wraps the XLA Sort operator, documented at
 https://www.tensorflow.org/performance/xla/operation_semantics#sort
.

Sorts a tensor. Currently only sorts in ascending order are supported.

keys: A `Tensor` of type K.
values: A `Tensor` of type V.
sorted_keys: A `Tensor` of type K.
sorted_values: A `Tensor` of type V.
)doc");

REGISTER_OP("XlaVariadicSort")
    .Input("inputs: T")
    .Input("dimension: int32")
    .Output("outputs: T")
    .Attr("T: list(type) >= 1")
    .Attr("comparator: func")
    .Attr("is_stable: bool")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      std::vector<shape_inference::ShapeHandle> input_shapes;
      TF_RETURN_IF_ERROR(c->input("inputs", &input_shapes));
      TF_RETURN_IF_ERROR(c->set_output("outputs", input_shapes));
      return OkStatus();
    })
    .Doc(R"doc(
Wraps the XLA Sort operator, documented at
 https://www.tensorflow.org/performance/xla/operation_semantics#sort
.

Sorts one or more tensors, with support for custom comparator, dimension, and
is_stable attributes.

inputs: A list of `Tensor` of identical shape but possibly different types.
dimension: The dimension along which to sort. Must be a compile-time constant.
is_stable: Whether to use stable sort.
comparator: A comparator function to apply to 2*N scalars and returning a
  boolean. N is the number of sort inputs. If you want to sort in ascending
  order then the comparator should perform a less-than comparison.
outputs: A list of `Tensor` of same shape and types as the `input`.
)doc");

// TODO(b/37549631) setting the While Op to always be stateful is too
// conservative.
REGISTER_OP("XlaWhile")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: list(type) >= 0")
    .Attr("cond: func")
    .Attr("body: func")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
output = input; While (Cond(output)) { output = Body(output) }

input: A list of input tensors whose types are T.
output: A list of output tensors whose types are T.
cond: A function takes 'input' and returns a tensor.  If the tensor is
      a scalar of non-boolean, the scalar is converted to a boolean
      according to the following rule: if the scalar is a numerical
      value, non-zero means True and zero means False; if the scalar is
      a string, non-empty means True and empty means False. If the
      tensor is not a scalar, non-emptiness means True and False
      otherwise.
body: A function that takes a list of tensors and returns another
      list of tensors. Both lists have the same types as specified by T.
)doc");

REGISTER_OP("XlaDequantize")
    .Input("input: uint32")
    .Output("output: bfloat16")
    .Attr("min_range: float")
    .Attr("max_range: float")
    .Attr("mode: string")
    .Attr("transpose_output: bool")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
Takes the packed uint32 input and unpacks the input to uint8 to do
Dequantization on device.

input: Input tensors whose types is uint32, shape is [d0, ..., dn].
output: Output tensors whose types is bloat16. If transpose_output is true,
     output shape is [dn * 4, dn-1, ..., d1, d0]. If transpose_output
     is false, output shape is [d0,..., dn * 4].
min_range: The minimum scalar value possibly produced for the input.
max_range: The maximum scalar value possibly produced for the input.
mode: String to determine the dequantize mode in {"MIN_COMBINED", "MIN_FIRST", "SCALED"}.
transpose_output: Boolean to determine if output is transposed. transpose_output
     is faster when input is large and rank of input is higher than 1.
)doc");

REGISTER_OP("XlaEinsum")
    .Input("a: T")
    .Input("b: T")
    .Output("product: T")
    .Attr("equation: string")
    .Attr("T: {complex64, bfloat16, float}")
    .SetShapeFn([](shape_inference::InferenceContext* context) {
      string equation;
      TF_RETURN_IF_ERROR(context->GetAttr("equation", &equation));
      // XlaEinsum supports only two-input einsum equations.
      if (!absl::StrContains(equation, ",")) {
        return errors::InvalidArgument("Expected one \",\" in equation. Got: ",
                                       equation);
      }
      // Use EinsumShape for the rest of the inference now that we know we must
      // have a two-input einsum.
      return shape_inference::EinsumShape(context);
    })
    .Doc(R"doc(
An op which supports basic einsum op with 2 inputs and 1 output.

This op has better TPU performance since it doesn't have explicitly reshape and
transpose operations as tf.einsum does.
)doc");

REGISTER_OP("XlaSpmdFullToShardShape")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: type")
    .Attr("manual_sharding: string")
    .Attr("dim: int = -1")
    .Attr("unspecified_dims: list(int) = []")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      auto input_handle = c->input(0);
      if (!c->RankKnown(input_handle)) {
        return shape_inference::UnknownShape(c);
      }
      string sharding_attr;
      TF_RETURN_IF_ERROR(c->GetAttr("manual_sharding", &sharding_attr));
      int32 single_dim;
      TF_RETURN_IF_ERROR(c->GetAttr("dim", &single_dim));
      xla::OpSharding sharding;
      sharding.ParseFromString(sharding_attr);
      if (sharding.type() != xla::OpSharding::OTHER) {
        return shape_inference::UnchangedShape(c);
      }
      std::vector<shape_inference::DimensionHandle> dims;
      for (int64_t i = 0; i < c->Rank(input_handle); ++i) {
        auto dim = c->Value(c->Dim(input_handle, i));
        if (single_dim < 0 || single_dim == i) {
          int64_t partitions_i = sharding.tile_assignment_dimensions(i);
          if (dim != shape_inference::InferenceContext::kUnknownDim &&
              partitions_i != 1) {
            dim = (dim + partitions_i - 1) / partitions_i;
          }
        }
        dims.push_back(c->MakeDim(dim));
      }
      c->set_output(0, c->MakeShape(dims));
      return OkStatus();
    })
    .Doc(R"doc(
An op used by XLA SPMD partitioner to switch from automatic partitioning to
manual partitioning. It annotates the input (full-shape, to be automatically
partitioned) with the same sharding used by manual partitioning, and outputs a
shard-shaped tensor to be consumed by later manually-partitioned ops. If the
shape is not evenly partitionable, the padding region will be masked with 0s.
The conversion can happen partially in subgroups, by specifying the dim
attribute, where only that dim will be converted.
)doc");

REGISTER_OP("XlaSpmdShardToFullShape")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: type")
    .Attr("manual_sharding: string")
    .Attr("full_shape: shape")
    .Attr("dim: int = -1")
    .Attr("unspecified_dims: list(int) = []")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      TensorShape shape_attr;
      TF_RETURN_IF_ERROR(c->GetAttr("full_shape", &shape_attr));
      shape_inference::ShapeHandle s;
      TF_RETURN_IF_ERROR(c->MakeShapeFromTensorShape(shape_attr, &s));
      c->set_output(0, s);
      return OkStatus();
    })
    .Doc(R"doc(
An op used by XLA SPMD partitioner to switch from manual partitioning to
automatic partitioning. It converts the shard-shaped, manually partitioned input
into full-shaped tensor to be partitioned automatically with the same sharding
used by manual partitioning. The conversion can happen partially in subgroups,
by specifying the dim attribute, where only that dim will be converted.
)doc");

REGISTER_OP("XlaSharding")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: type")
    .Attr("sharding: string = ''")
    .Attr("unspecified_dims: list(int) = []")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
An op which shards the input based on the given sharding attribute. It can
selectively annotate a subset of tensor dimensions by skipping unspecified_dims,
and the sharding annotation should be replicated in those dims.
)doc");

REGISTER_OP("XlaReplicaId")
    .Output("id: int32")
    .SetShapeFn([](shape_inference::InferenceContext* context) {
      context->set_output(0, context->MakeShape({}));
      return OkStatus();
    })
    .Doc("Replica ID.");

REGISTER_OP("XlaGather")
    .Input("operand: T")
    .Input("start_indices: Tindices")
    .Input("slice_sizes: Tindices")
    .Attr("dimension_numbers: string")
    .Attr("indices_are_sorted: bool")
    .Attr("T: {numbertype, bool}")
    .Attr("Tindices: {int32, int64}")
    .Output("output: T")
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
Wraps the XLA Gather operator documented at
  https://www.tensorflow.org/xla/operation_semantics#gather
operand: The array we're gathering from.
start_indices: Array containing the starting indices of the slices we gather.
dimension_numbers: A serialized xla::GatherDimensionNumbers proto.
slice_sizes: slice_sizes[i] is the bounds for the slice on dimension i.
indices_are_sorted: Boolean indicating if the indices are sorted.
)doc");

REGISTER_OP("XlaScatter")
    .Input("operand: T")
    .Input("scatter_indices: Tindices")
    .Input("updates: T")
    .Attr("update_computation: func")
    .Attr("dimension_numbers: string")
    .Attr("indices_are_sorted: bool")
    .Attr("T: {numbertype, bool}")
    .Attr("Tindices: {int32, int64}")
    .Output("output: T")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
Wraps the XLA Scatter operator documented at
  https://www.tensorflow.org/xla/operation_semantics#scatter.

operand: Array to be scattered into.
scatter_indices: Array containing the starting indices of the slices that must
  be scattered to.
updates: Array containing the values that must be used for scattering.
update_computation: Computation to be used for combining the existing values in
  the input array and the updates during scatter.
dimension_numbers: A serialized xla::ScatterDimensionNumbers proto.
indices_are_sorted: Boolean indicating if the indices are sorted.
)doc");

REGISTER_OP("XlaAllReduce")
    .Input("input: T")
    .Input("group_assignment: int32")
    .Output("output: T")
    .Attr("T: {half, bfloat16, float, int32, uint32}")
    .Attr("reduce_op: {'Min', 'Max', 'Mul', 'Add', 'Mean'}")
    .Attr("mode: {'CrossReplica', 'CrossReplicaAndPartition'}")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
Wraps the XLA AllReduce operator
  documented at https://www.tensorflow.org/xla/operation_semantics#allreduce.

input: Array or a non-empty tuple of arrays to reduce across replicas.
group_assignment: Groups between which the reductions are performed.
reduce_op: Reduction computation.
mode: group mode.
  CrossReplica: group_assignment contains replica_id. Each group contains the
    replicas for the current partition.
  CrossReplicaAndPartition: group_assignment contains replica_id. Each group
    contains the replicas for all partitions.
)doc");

REGISTER_OP("XlaReduceScatter")
    .Input("input: T")
    .Input("group_assignment: int32")
    .Input("scatter_dimension: int32")
    .Output("output: T")
    .Attr("T: {half, bfloat16, float, int32, uint32}")
    .Attr("reduce_op: {'Min', 'Max', 'Mul', 'Add', 'Mean'}")
    .SetShapeFn(shape_inference::ReduceScatterShape)
    .Doc(R"doc(
Wraps the XLA ReduceScatter operator
  documented at https://www.tensorflow.org/xla/operation_semantics#reducescatter.

input: Array or a non-empty tuple of arrays to reduce across replicas.
group_assignment: Groups between which the reductions are performed.
scatter_dimension: Dimension to scatter.
reduce_op: Reduction computation.
)doc");

Status OptimizationBarrierShape(shape_inference::InferenceContext* c) {
  for (int i = 0; i < c->num_inputs(); ++i) {
    c->set_output(i, c->input(i));
  }
  return OkStatus();
}

REGISTER_OP("XlaOptimizationBarrier")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: list(type) >= 0")
    .SetShapeFn(OptimizationBarrierShape)
    .Doc(R"doc(
Wraps the XLA OptimizationBarrier operator.

Documented at https://www.tensorflow.org/xla/operation_semantics#optimizationbarrier.

input: A Tuple of Arrays of any type.
)doc");

REGISTER_OP("XlaCustomCall")
    .Input("args: T")
    .Output("output: dtype")
    .Attr("target_name: string")
    .Attr("backend_config: string")
    .Attr("T: list(type) >= 0")
    .Attr("dtype: type")
    .Attr("shape: shape")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      TensorShape shape_attr;
      TF_RETURN_IF_ERROR(c->GetAttr("shape", &shape_attr));
      shape_inference::ShapeHandle s;
      TF_RETURN_IF_ERROR(c->MakeShapeFromTensorShape(shape_attr, &s));
      c->set_output(0, s);
      return OkStatus();
    })
    .Doc(R"doc(
Wraps the XLA CustomCall operator
  documented at https://www.tensorflow.org/xla/operation_semantics#customcall.

args: A list of `Tensor` with possibly different types.
target_name: Name of the function. A call instruction will be emitted which
  targets this symbol name.
backend_config: String, used to encode serialized metadata to the backend.
dtype: Output tensor data type.
shape: Output tensor shape.
)doc");

REGISTER_OP("XlaCallModule")
    .Input("args: Tin")
    .Output("output: Tout")
    .Attr("module: string")
    .Attr("Sout: list(shape) >= 0")
    .Attr("Tout: list(type) >= 0")
    .Attr("Tin: list(type) >= 0")
    .Attr("dim_args_spec: list(string) >= 0")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      // For debugging
      VLOG(3) << "XlaCallModule.shape_inference";
      std::vector<shape_inference::ShapeHandle> args_shapes;
      TF_RETURN_IF_ERROR(c->input("args", &args_shapes));
      for (int i = 0; i < args_shapes.size(); ++i) {
        VLOG(3) << "XlaCallModule.shape_inference args[" << i
                << "] : " << c->DebugString(args_shapes[i]);
      }
      std::vector<PartialTensorShape> shapes_attr;
      TF_RETURN_IF_ERROR(c->GetAttr("Sout", &shapes_attr));
      for (int i = 0; i < shapes_attr.size(); ++i) {
        shape_inference::ShapeHandle s;
        TF_RETURN_IF_ERROR(
            c->MakeShapeFromPartialTensorShape(shapes_attr[i], &s));
        VLOG(3) << "XlaCallModule.shape_inference out[" << i
                << "] : " << c->DebugString(s);
        c->set_output(i, s);
      }
      return OkStatus();
    })
    .Doc(R"doc(
Temporary op for experimenting with jax2tf.

DO NOT USE THIS OP. It has no backwards compatibility guarantees. It is also
very likely to change. This op will be used only in jax2tf under an
experimental flag.

This is an experimental op to allow a smooth evolution of jax2tf towards
emitting and serializing MHLO directly from JAX. At the moment this op
carries a serialized MHLO module, therefore there are no backward-compatibility
guarantees, and should not be used for serialization.
Eventually, the op will carry a MHLO object, which will have
backwards-compatibility guarantees.

The serialized module must return a tuple if and only if the Sout is an empty
list or a list with more than 1 elements. The length of Tout and Sout must
match. This op always returns a tuple of results, even if the module returns
a single result.

The handling of dynamic shapes is work-in-progress. At the moment, the
JAX lowering for dynamic shapes will prepend one dimension parameter to the
serialized module for each dimension whose value must be passed in.
The "args" correspond to the non-dimension arguments. During compilation
we compute the values of the dimension arguments based on the static shapes of
the "args". In order to do this, we encode for each dimension argument a
specification of how to compute its value, as a string, in the form
"<arg_idx>.<axis_idx>".
E.g., the specification "2.1" denotes the value args[2].shape[1].

args: A list of `Tensor` with possibly different types to be passed as arguments
  to the HLO module.
module: A serialized computation, a text representation of mlir.Module.
Tout: List of output tensor data types.
Sout: List of output tensor shapes.
dim_args_spec: the specification for the dimension arguments, one for each
  dimension argument. In absence of dynamic shapes this list is empty.
)doc");

}  // namespace
}  // namespace tensorflow
