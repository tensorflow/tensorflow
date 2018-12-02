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

#include "absl/algorithm/container.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/errors.h"

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
  return Status::OK();
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

REGISTER_OP("XlaDot")
    .Input("lhs: T")
    .Input("rhs: T")
    .Attr("T: numbertype")
    .Attr("dimension_numbers: string")
    .Attr("precision_config: string")
    .Output("output: T")
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
Wraps the XLA ConvGeneralDilated operator, documented at
 https://www.tensorflow.org/performance/xla/operation_semantics#dotgeneral
.

lhs: the LHS tensor
rhs: the RHS tensor
dimension_numbers: a serialized xla::DotDimensionNumbers proto.
precision_config: a serialized xla::PrecisionConfig proto.
)doc");

REGISTER_OP("XlaDynamicSlice")
    .Input("input: T")
    .Input("start_indices: Tindices")
    .Input("size_indices: Tindices")
    .Output("output: T")
    .Attr("T: type")
    .Attr("Tindices: {int32, int64}")
    .SetShapeFn(shape_inference::UnknownShape)
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
    .SetShapeFn(UnchangedRank)
    .Doc(R"doc(
Wraps the XLA Pad operator, documented at
 https://www.tensorflow.org/performance/xla/operation_semantics#pad
.

input: A `Tensor` of type T.
padding_value: A scalar `Tensor` of type T.
padding_low: the padding to apply at the start of each input dimensions
padding_high: the padding to apply at the end of each input dimension.
padding_interior: the padding to apply between each input element.
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
      return Status::OK();
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
    .Attr("T: numbertype")
    .Attr("dimensions_to_reduce: list(int)")
    .Attr("reducer: func")
    .Output("output: T")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      if (c->RankKnown(c->input(0))) {
        int rank = c->Rank(c->input(0));
        std::vector<int64> dimensions_to_reduce;
        TF_RETURN_IF_ERROR(
            c->GetAttr("dimensions_to_reduce", &dimensions_to_reduce));
        std::set<int64> dims_set(dimensions_to_reduce.begin(),
                                 dimensions_to_reduce.end());
        auto dim_in_range = [rank](int64 dim) {
          return dim >= 0 && dim < rank;
        };
        if (rank < dimensions_to_reduce.size() ||
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
      return Status::OK();
    })
    .Doc(R"doc(
Wraps the XLA Reduce operator, documented at
 https://www.tensorflow.org/performance/xla/operation_semantics#reduce .

input: the input tensor
init_value: a scalar representing the initial value for the reduction
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
    .Attr("T: numbertype")
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
    .SetShapeFn(shape_inference::UnchangedShape)
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

}  // namespace
}  // namespace tensorflow
