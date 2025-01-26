/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include <vector>

#include "tensorflow/cc/framework/grad_op_registry.h"
#include "tensorflow/cc/framework/gradients.h"
#include "tensorflow/cc/ops/array_ops_internal.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace tensorflow {
namespace ops {
namespace {

REGISTER_NO_GRADIENT_OP("Const");
REGISTER_NO_GRADIENT_OP("StopGradient");
REGISTER_NO_GRADIENT_OP("ConcatOffset");
REGISTER_NO_GRADIENT_OP("EditDistance");
REGISTER_NO_GRADIENT_OP("ZerosLike");
REGISTER_NO_GRADIENT_OP("InvertPermutation");
REGISTER_NO_GRADIENT_OP("Shape");
REGISTER_NO_GRADIENT_OP("ShapeN");
REGISTER_NO_GRADIENT_OP("Rank");
REGISTER_NO_GRADIENT_OP("Size");
REGISTER_NO_GRADIENT_OP("BroadcastGradientArgs");
REGISTER_NO_GRADIENT_OP("OneHot");

absl::Status PackGrad(const Scope& scope, const Operation& op,
                      const std::vector<Output>& grad_inputs,
                      std::vector<Output>* grad_outputs) {
  int N;
  TF_RETURN_IF_ERROR(GetNodeAttr(op.node()->attrs(), "N", &N));
  int axis;
  TF_RETURN_IF_ERROR(GetNodeAttr(op.node()->attrs(), "axis", &axis));

  grad_outputs->reserve(N);
  auto grad_op = Unstack(scope, grad_inputs[0], N, Unstack::Axis(axis));
  for (const Output& o : grad_op.output) {
    grad_outputs->emplace_back(o);
  }
  return scope.status();
}
REGISTER_GRADIENT_OP("Pack", PackGrad);

absl::Status UnpackGrad(const Scope& scope, const Operation& op,
                        const std::vector<Output>& grad_inputs,
                        std::vector<Output>* grad_outputs) {
  int axis;
  TF_RETURN_IF_ERROR(GetNodeAttr(op.node()->attrs(), "axis", &axis));
  grad_outputs->push_back(Stack(scope, grad_inputs, Stack::Axis(axis)));
  return scope.status();
}
REGISTER_GRADIENT_OP("Unpack", UnpackGrad);

absl::Status IdentityGrad(const Scope& scope, const Operation& op,
                          const std::vector<Output>& grad_inputs,
                          std::vector<Output>* grad_outputs) {
  grad_outputs->push_back(Identity(scope, grad_inputs[0]));
  return scope.status();
}
REGISTER_GRADIENT_OP("Identity", IdentityGrad);

absl::Status RefIdentityGrad(const Scope& scope, const Operation& op,
                             const std::vector<Output>& grad_inputs,
                             std::vector<Output>* grad_outputs) {
  grad_outputs->push_back(Identity(scope, grad_inputs[0]));
  return scope.status();
}
REGISTER_GRADIENT_OP("RefIdentity", RefIdentityGrad);

absl::Status QuantizeAndDequantizeGrad(const Scope& scope, const Operation& op,
                                       const std::vector<Output>& grad_inputs,
                                       std::vector<Output>* grad_outputs) {
  grad_outputs->push_back(Identity(scope, grad_inputs[0]));
  return scope.status();
}
REGISTER_GRADIENT_OP("QuantizeAndDequantize", QuantizeAndDequantizeGrad);

absl::Status QuantizeAndDequantizeV4GradHelper(
    const Scope& scope, const Operation& op,
    const std::vector<Output>& grad_inputs, std::vector<Output>* grad_outputs) {
  Input input = Shape(scope, op.input(0));
  Input input_min = op.input(1);
  Input input_max = op.input(2);
  int64_t axis;
  TF_RETURN_IF_ERROR(GetNodeAttr(op.node()->attrs(), "axis", &axis));
  auto qdq_v4_grad = QuantizeAndDequantizeV4Grad(
      scope, grad_inputs[0], input, input_min, input_max,
      QuantizeAndDequantizeV4Grad::Axis(axis));
  grad_outputs->push_back(qdq_v4_grad.input_backprop);
  grad_outputs->push_back(qdq_v4_grad.input_min_backprop);
  grad_outputs->push_back(qdq_v4_grad.input_max_backprop);
  return scope.status();
}
REGISTER_GRADIENT_OP("QuantizeAndDequantizeV4",
                     QuantizeAndDequantizeV4GradHelper);

absl::Status QuantizeAndDequantizeV3Grad(const Scope& scope,
                                         const Operation& op,
                                         const std::vector<Output>& grad_inputs,
                                         std::vector<Output>* grad_outputs) {
  grad_outputs->push_back(Identity(scope, grad_inputs[0]));
  grad_outputs->push_back(NoGradient());
  grad_outputs->push_back(NoGradient());
  grad_outputs->push_back(NoGradient());
  return scope.status();
}
REGISTER_GRADIENT_OP("QuantizeAndDequantizeV3", QuantizeAndDequantizeV3Grad);

absl::Status SplitGrad(const Scope& scope, const Operation& op,
                       const std::vector<Output>& grad_inputs,
                       std::vector<Output>* grad_outputs) {
  grad_outputs->push_back(NoGradient());
  grad_outputs->push_back(Concat(scope, grad_inputs, op.input(0)));
  return scope.status();
}
REGISTER_GRADIENT_OP("Split", SplitGrad);

absl::Status SplitVGrad(const Scope& scope, const Operation& op,
                        const std::vector<Output>& grad_inputs,
                        std::vector<Output>* grad_outputs) {
  if (op.num_inputs() < 3) {
    return errors::InvalidArgument("SplitV requires 3 arguments");
  }
  grad_outputs->push_back(Concat(scope, grad_inputs, op.input(2)));
  for (int i = 0; i < op.num_inputs() - 1; ++i) {
    grad_outputs->push_back(NoGradient());
  }
  return scope.status();
}
REGISTER_GRADIENT_OP("SplitV", SplitVGrad);

absl::Status FillGrad(const Scope& scope, const Operation& op,
                      const std::vector<Output>& grad_inputs,
                      std::vector<Output>* grad_outputs) {
  // y = fill(fill_shape, x)
  // No gradient returned for the fill_shape argument.
  grad_outputs->push_back(NoGradient());
  // The gradient for x (which must be a scalar) is just the sum of
  // all the gradients from the shape it fills.
  // We use ReduceSum to implement this, which needs an argument providing
  // the indices of all the dimensions of the incoming gradient.
  // grad(x) = reduce_sum(grad(y), [0..rank(grad(y))])
  auto all_dims = Range(scope, Const(scope, 0), Rank(scope, grad_inputs[0]),
                        Const(scope, 1));
  grad_outputs->push_back(ReduceSum(scope, grad_inputs[0], all_dims));
  return scope.status();
}
REGISTER_GRADIENT_OP("Fill", FillGrad);

absl::Status DiagGrad(const Scope& scope, const Operation& op,
                      const std::vector<Output>& grad_inputs,
                      std::vector<Output>* grad_outputs) {
  grad_outputs->push_back(DiagPart(scope, grad_inputs[0]));
  return scope.status();
}
REGISTER_GRADIENT_OP("Diag", DiagGrad);

absl::Status DiagPartGrad(const Scope& scope, const Operation& op,
                          const std::vector<Output>& grad_inputs,
                          std::vector<Output>* grad_outputs) {
  grad_outputs->push_back(Diag(scope, grad_inputs[0]));
  return scope.status();
}
REGISTER_GRADIENT_OP("DiagPart", DiagPartGrad);

absl::Status MatrixDiagGrad(const Scope& scope, const Operation& op,
                            const std::vector<Output>& grad_inputs,
                            std::vector<Output>* grad_outputs) {
  grad_outputs->push_back(MatrixDiagPart(scope, grad_inputs[0]));
  return scope.status();
}
REGISTER_GRADIENT_OP("MatrixDiag", MatrixDiagGrad);

absl::Status MatrixBandPartGrad(const Scope& scope, const Operation& op,
                                const std::vector<Output>& grad_inputs,
                                std::vector<Output>* grad_outputs) {
  auto num_lower = op.input(1);
  auto num_upper = op.input(2);
  grad_outputs->push_back(
      MatrixBandPart(scope, grad_inputs[0], num_lower, num_upper));
  grad_outputs->push_back(NoGradient());
  grad_outputs->push_back(NoGradient());
  return scope.status();
}
REGISTER_GRADIENT_OP("MatrixBandPart", MatrixBandPartGrad);

absl::Status GatherNdGrad(const Scope& scope, const Operation& op,
                          const std::vector<Output>& grad_inputs,
                          std::vector<Output>* grad_outputs) {
  auto ref = op.input(0);
  auto indices = op.input(1);
  Shape::Attrs shape_attrs;
  shape_attrs.out_type_ = indices.type();
  auto ref_shape = Shape(scope, ref, shape_attrs);
  grad_outputs->push_back(ScatterNd(scope, indices, grad_inputs[0], ref_shape));
  grad_outputs->push_back(NoGradient());
  return scope.status();
}
REGISTER_GRADIENT_OP("GatherNd", GatherNdGrad);

absl::Status CheckNumericsGrad(const Scope& scope, const Operation& op,
                               const std::vector<Output>& grad_inputs,
                               std::vector<Output>* grad_outputs) {
  string message;
  TF_RETURN_IF_ERROR(GetNodeAttr(op.node()->attrs(), "message", &message));
  string err_msg = strings::StrCat(
      "Not a number (NaN) or infinity (Inf) values detected in gradient. ",
      message);
  grad_outputs->push_back(CheckNumerics(scope, grad_inputs[0], err_msg));
  return scope.status();
}
REGISTER_GRADIENT_OP("CheckNumerics", CheckNumericsGrad);

absl::Status ReshapeGrad(const Scope& scope, const Operation& op,
                         const std::vector<Output>& grad_inputs,
                         std::vector<Output>* grad_outputs) {
  auto input_shape = Shape(scope, op.input(0));
  grad_outputs->push_back(Reshape(scope, grad_inputs[0], input_shape));
  grad_outputs->push_back(NoGradient());
  return scope.status();
}
REGISTER_GRADIENT_OP("Reshape", ReshapeGrad);

absl::Status ExpandDimsGrad(const Scope& scope, const Operation& op,
                            const std::vector<Output>& grad_inputs,
                            std::vector<Output>* grad_outputs) {
  auto input_shape = Shape(scope, op.input(0));
  grad_outputs->push_back(Reshape(scope, grad_inputs[0], input_shape));
  grad_outputs->push_back(NoGradient());
  return scope.status();
}
REGISTER_GRADIENT_OP("ExpandDims", ExpandDimsGrad);

absl::Status SqueezeGrad(const Scope& scope, const Operation& op,
                         const std::vector<Output>& grad_inputs,
                         std::vector<Output>* grad_outputs) {
  auto input_shape = Shape(scope, op.input(0));
  grad_outputs->push_back(Reshape(scope, grad_inputs[0], input_shape));
  return scope.status();
}
REGISTER_GRADIENT_OP("Squeeze", SqueezeGrad);

absl::Status TransposeGrad(const Scope& scope, const Operation& op,
                           const std::vector<Output>& grad_inputs,
                           std::vector<Output>* grad_outputs) {
  auto inverted_perm = InvertPermutation(scope, op.input(1));
  grad_outputs->push_back(Transpose(scope, grad_inputs[0], inverted_perm));
  grad_outputs->push_back(NoGradient());
  return scope.status();
}
REGISTER_GRADIENT_OP("Transpose", TransposeGrad);

absl::Status ReverseSequenceGrad(const Scope& scope, const Operation& op,
                                 const std::vector<Output>& grad_inputs,
                                 std::vector<Output>* grad_outputs) {
  auto seq_lengths = op.input(1);
  int batch_dim;
  TF_RETURN_IF_ERROR(GetNodeAttr(op.node()->attrs(), "batch_dim", &batch_dim));
  int seq_dim;
  TF_RETURN_IF_ERROR(GetNodeAttr(op.node()->attrs(), "seq_dim", &seq_dim));
  grad_outputs->push_back(
      ReverseSequence(scope, grad_inputs[0], seq_lengths, seq_dim,
                      ReverseSequence::BatchDim(batch_dim)));
  grad_outputs->push_back(NoGradient());
  return scope.status();
}
REGISTER_GRADIENT_OP("ReverseSequence", ReverseSequenceGrad);

absl::Status ReverseGrad(const Scope& scope, const Operation& op,
                         const std::vector<Output>& grad_inputs,
                         std::vector<Output>* grad_outputs) {
  auto reverse_dims = op.input(1);
  grad_outputs->push_back(Reverse(scope, grad_inputs[0], reverse_dims));
  grad_outputs->push_back(NoGradient());
  return scope.status();
}
REGISTER_GRADIENT_OP("ReverseV2", ReverseGrad);

absl::Status ScatterNdGrad(const Scope& scope, const Operation& op,
                           const std::vector<Output>& grad_inputs,
                           std::vector<Output>* grad_outputs) {
  auto indices = op.input(0);
  grad_outputs->push_back(NoGradient());
  grad_outputs->push_back(GatherNd(scope, grad_inputs[0], indices));
  grad_outputs->push_back(NoGradient());
  return scope.status();
}
REGISTER_GRADIENT_OP("ScatterNd", ScatterNdGrad);

absl::Status ScatterNdNonAliasingAddGrad(const Scope& scope,
                                         const Operation& op,
                                         const std::vector<Output>& grad_inputs,
                                         std::vector<Output>* grad_outputs) {
  auto indices = op.input(1);
  grad_outputs->push_back(Identity(scope, grad_inputs[0]));
  grad_outputs->push_back(NoGradient());
  grad_outputs->push_back(GatherNd(scope, grad_inputs[0], indices));
  return scope.status();
}
REGISTER_GRADIENT_OP("ScatterNdNonAliasingAdd", ScatterNdNonAliasingAddGrad);

template <bool IsPadV2>
absl::Status PadGrad(const Scope& scope, const Operation& op,
                     const std::vector<Output>& grad_inputs,
                     std::vector<Output>* grad_outputs) {
  auto x = op.input(0);
  auto a = op.input(1);  // [Rank(x), 2]
  // Takes a slice of a. The 1st column. [Rank(x), 1].
  auto size = Stack(scope, {Rank(scope, x), 1});
  auto pad_before = Slice(scope, a, {0, 0}, size);
  // Make it a 1-D tensor.
  auto begin = Reshape(scope, pad_before, {-1});
  grad_outputs->push_back(Slice(scope, grad_inputs[0], begin, Shape(scope, x)));
  grad_outputs->push_back(NoGradient());
  // PadV2 adds a "constant_values" input.
  if (IsPadV2) {
    grad_outputs->push_back(NoGradient());
  }
  return scope.status();
}
REGISTER_GRADIENT_OP("Pad", PadGrad<false>);
REGISTER_GRADIENT_OP("PadV2", PadGrad<true>);

absl::Status SpaceToBatchGrad(const Scope& scope, const Operation& op,
                              const std::vector<Output>& grad_inputs,
                              std::vector<Output>* grad_outputs) {
  int block_size;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(op.node()->attrs(), "block_size", &block_size));
  grad_outputs->push_back(
      BatchToSpace(scope, grad_inputs[0], op.input(1), block_size));
  grad_outputs->push_back(NoGradient());
  return scope.status();
}
REGISTER_GRADIENT_OP("SpaceToBatch", SpaceToBatchGrad);

absl::Status SpaceToBatchNDGrad(const Scope& scope, const Operation& op,
                                const std::vector<Output>& grad_inputs,
                                std::vector<Output>* grad_outputs) {
  grad_outputs->push_back(
      BatchToSpaceND(scope, grad_inputs[0], op.input(1), op.input(2)));
  grad_outputs->push_back(NoGradient());
  grad_outputs->push_back(NoGradient());
  return scope.status();
}
REGISTER_GRADIENT_OP("SpaceToBatchND", SpaceToBatchNDGrad);

absl::Status BatchToSpaceGrad(const Scope& scope, const Operation& op,
                              const std::vector<Output>& grad_inputs,
                              std::vector<Output>* grad_outputs) {
  int block_size;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(op.node()->attrs(), "block_size", &block_size));
  grad_outputs->push_back(
      SpaceToBatch(scope, grad_inputs[0], op.input(1), block_size));
  grad_outputs->push_back(NoGradient());
  return scope.status();
}
REGISTER_GRADIENT_OP("BatchToSpace", BatchToSpaceGrad);

absl::Status BatchToSpaceNDGrad(const Scope& scope, const Operation& op,
                                const std::vector<Output>& grad_inputs,
                                std::vector<Output>* grad_outputs) {
  grad_outputs->push_back(
      SpaceToBatchND(scope, grad_inputs[0], op.input(1), op.input(2)));
  grad_outputs->push_back(NoGradient());
  grad_outputs->push_back(NoGradient());
  return scope.status();
}
REGISTER_GRADIENT_OP("BatchToSpaceND", BatchToSpaceNDGrad);

absl::Status SpaceToDepthGrad(const Scope& scope, const Operation& op,
                              const std::vector<Output>& grad_inputs,
                              std::vector<Output>* grad_outputs) {
  int block_size;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(op.node()->attrs(), "block_size", &block_size));
  grad_outputs->push_back(DepthToSpace(scope, grad_inputs[0], block_size));
  return scope.status();
}
REGISTER_GRADIENT_OP("SpaceToDepth", SpaceToDepthGrad);

absl::Status DepthToSpaceGrad(const Scope& scope, const Operation& op,
                              const std::vector<Output>& grad_inputs,
                              std::vector<Output>* grad_outputs) {
  int block_size;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(op.node()->attrs(), "block_size", &block_size));
  grad_outputs->push_back(SpaceToDepth(scope, grad_inputs[0], block_size));
  return scope.status();
}
REGISTER_GRADIENT_OP("DepthToSpace", DepthToSpaceGrad);

absl::Status MirrorPadGrad(const Scope& scope, const Operation& op,
                           const std::vector<Output>& grad_inputs,
                           std::vector<Output>* grad_outputs) {
  string mode;
  TF_RETURN_IF_ERROR(GetNodeAttr(op.node()->attrs(), "mode", &mode));
  grad_outputs->push_back(tensorflow::ops::internal::MirrorPadGrad(
      scope, grad_inputs[0], op.input(1), mode));
  grad_outputs->push_back(NoGradient());
  return scope.status();
}
REGISTER_GRADIENT_OP("MirrorPad", MirrorPadGrad);

// TODO(suharshs): b/34770860. This gradient was within 1e-3 but not 1e-4.
absl::Status MirrorPadGradGrad(const Scope& scope, const Operation& op,
                               const std::vector<Output>& grad_inputs,
                               std::vector<Output>* grad_outputs) {
  string mode;
  TF_RETURN_IF_ERROR(GetNodeAttr(op.node()->attrs(), "mode", &mode));
  grad_outputs->push_back(MirrorPad(scope, grad_inputs[0], op.input(1), mode));
  grad_outputs->push_back(NoGradient());
  return scope.status();
}
REGISTER_GRADIENT_OP("MirrorPadGrad", MirrorPadGradGrad);

absl::Status StridedSliceGradHelper(const Scope& scope, const Operation& op,
                                    const std::vector<Output>& grad_inputs,
                                    std::vector<Output>* grad_outputs) {
  Input x = Shape(scope, op.input(0));
  Input begin = op.input(1);
  Input end = op.input(2);
  Input strides = op.input(3);
  int64_t begin_mask;
  int64_t end_mask;
  int64_t ellipsis_mask;
  int64_t new_axis_mask;
  int64_t shrink_axis_mask;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(op.node()->attrs(), "begin_mask", &begin_mask));
  TF_RETURN_IF_ERROR(GetNodeAttr(op.node()->attrs(), "end_mask", &end_mask));
  TF_RETURN_IF_ERROR(
      GetNodeAttr(op.node()->attrs(), "ellipsis_mask", &ellipsis_mask));
  TF_RETURN_IF_ERROR(
      GetNodeAttr(op.node()->attrs(), "new_axis_mask", &new_axis_mask));
  TF_RETURN_IF_ERROR(
      GetNodeAttr(op.node()->attrs(), "shrink_axis_mask", &shrink_axis_mask));
  grad_outputs->push_back(
      StridedSliceGrad(scope, x, begin, end, strides, grad_inputs[0],
                       StridedSliceGrad::BeginMask(begin_mask)
                           .EndMask(end_mask)
                           .EllipsisMask(ellipsis_mask)
                           .NewAxisMask(new_axis_mask)
                           .ShrinkAxisMask(shrink_axis_mask)));
  // No gradients returned for begin, end and strides
  grad_outputs->push_back(NoGradient());
  grad_outputs->push_back(NoGradient());
  grad_outputs->push_back(NoGradient());
  return scope.status();
}
REGISTER_GRADIENT_OP("StridedSlice", StridedSliceGradHelper);

absl::Status SliceGrad(const Scope& scope, const Operation& op,
                       const std::vector<Output>& grad_inputs,
                       std::vector<Output>* grad_outputs) {
  // Propagate the incoming gradient along all the selected values,
  // and zero everywhere else. Use the Pad operator for this.
  //
  // First create an Nx2 padding where N is the number of input
  // dimensions. The first column is the number of prepended zeros
  // for each dimension, and the second column is the number of
  // appended zeros.
  //
  // The first column is just the begin vector.
  // The second column is the shape of the input element-wise
  // subtracted by begin+size

  // Running example:
  // input.shape = [3, 5, 3]
  // begin = [1, 2, 1], size = [1, 3, 2]
  Input input = op.input(0);
  Input begin = op.input(1);
  // input_rank = 3
  auto input_rank = Rank(scope, input);
  // slice_size = [1, 3, 2]
  auto slice_size = Shape(scope, op.output(0));
  // padding_shape = [3, 1]
  auto padding_shape = Stack(scope, {input_rank, 1});
  // before_padding = [[1]
  //                   [2]
  //                   [1]]
  Input before_padding = Reshape(scope, begin, padding_shape);
  // after_padding_sizes = shape(input) - slice_size - begin
  //                     = [3, 5, 3] - [1, 3, 2] - [1, 2, 1]
  //                     = [1, 0, 0]
  auto after_padding_sizes =
      Sub(scope, Sub(scope, Shape(scope, input), slice_size), begin);
  // after_padding = [[1]
  //                  [0]
  //                  [0]]
  Input after_padding = Reshape(scope, after_padding_sizes, padding_shape);
  // paddings = [[1 1]
  //             [2 0]
  //             [1 0]]
  auto paddings =
      Concat(scope, {before_padding, after_padding}, Const(scope, 1));
  grad_outputs->push_back(Pad(scope, grad_inputs[0], paddings));
  // Nothing propagated for "begin" and "size" inputs
  grad_outputs->push_back(NoGradient());
  grad_outputs->push_back(NoGradient());
  return scope.status();
}
REGISTER_GRADIENT_OP("Slice", SliceGrad);

absl::Status ConcatGradHelper(const Scope& scope, const Operation& op,
                              const std::vector<Output>& grad_inputs,
                              std::vector<Output>* grad_outputs,
                              int start_value_index, int end_value_index,
                              int dim_index) {
  if (end_value_index >= op.num_inputs()) {
    return errors::Internal("Invalid input index");
  }
  std::vector<Output> inputs;
  inputs.reserve(end_value_index - start_value_index);
  for (int i = start_value_index; i < end_value_index; ++i) {
    inputs.push_back(op.input(i));
  }

  auto shapes = ShapeN(scope, inputs);
  const auto unique_name = scope.GetUniqueNameForOp("ConcatOffset");
  auto builder =
      ::tensorflow::NodeBuilder(unique_name, "ConcatOffset")
          .Input(::tensorflow::ops::AsNodeOut(scope, op.input(dim_index)))
          .Input(::tensorflow::ops::AsNodeOutList(scope, shapes.output));
  scope.UpdateBuilder(&builder);
  ::tensorflow::Node* concat_offset_node;
  scope.UpdateStatus(builder.Finalize(scope.graph(), &concat_offset_node));
  scope.UpdateStatus(scope.DoShapeInference(concat_offset_node));
  if (concat_offset_node->num_outputs() != inputs.size()) {
    return errors::Internal("ConcatOffset has invalid output count");
  }
  if (grad_inputs.size() != 1) {
    return errors::InvalidArgument("Concat grad should have 1 input");
  }

  // For each dx[i], we take a slice of dy. The offset and size of the
  // slice is given by offset[i] and shape[i].
  const Output& dy = grad_inputs[0];
  for (int i = 0; i < inputs.size(); ++i) {
    grad_outputs->push_back(
        Slice(scope, dy, Output(concat_offset_node, i), shapes.output[i]));
  }

  // Insert a NoGradient for the axis.
  grad_outputs->insert(grad_outputs->begin() + dim_index, NoGradient());
  return scope.status();
}

absl::Status ConcatV2Grad(const Scope& scope, const Operation& op,
                          const std::vector<Output>& grad_inputs,
                          std::vector<Output>* grad_outputs) {
  return ConcatGradHelper(scope, op, grad_inputs, grad_outputs,
                          /*start_value_index=*/0,
                          /*end_value_index=*/op.num_inputs() - 1,
                          /*dim+index=*/op.num_inputs() - 1);
}

REGISTER_GRADIENT_OP("ConcatV2", ConcatV2Grad);

absl::Status BroadcastToGrad(const Scope& scope, const Operation& op,
                             const std::vector<Output>& grad_inputs,
                             std::vector<Output>* grad_outputs) {
  if (grad_inputs.size() != 1) {
    return errors::InvalidArgument("BroadcastTo grad should have 1 grad input");
  }
  if (op.num_inputs() != 2) {
    return errors::InvalidArgument("BroadcastTo requires 2 inputs");
  }

  auto x_shape = Shape(scope, op.input(0));
  auto args = internal::BroadcastGradientArgs(scope, x_shape, op.input(1));
  auto sum_gx = Sum(scope, grad_inputs[0], args.r0);
  grad_outputs->push_back(Reshape(scope, sum_gx, x_shape));
  grad_outputs->push_back(NoGradient());
  return scope.status();
}

REGISTER_GRADIENT_OP("BroadcastTo", BroadcastToGrad);

absl::Status TileGrad(const Scope& scope, const Operation& op,
                      const std::vector<Output>& grad_inputs,
                      std::vector<Output>* grad_outputs) {
  if (op.num_inputs() != 2) {
    return errors::InvalidArgument("Tile requires 2 inputs");
  }
  if (grad_inputs.size() != 1) {
    return errors::InvalidArgument("Tile grad requires 1 grad input");
  }

  Shape::Attrs shape_attrs;
  shape_attrs.out_type_ = op.input_type(1);
  auto input_shape = Shape(scope, op.input(0), shape_attrs);
  // We interleave multiples and input_shape to get split_shape,
  // reshape grad to split_shape, and reduce along all even
  // dimensions (the tiled dimensions) to get the result
  // with shape input_shape.  For example
  //   input_shape = [20, 30, 40]
  //   multiples = [2, 3, 4]
  //   split_shape = [2, 20, 3, 30, 4, 40]
  //   axes = [0, 2, 4]
  auto stack = Stack(scope, {op.input(1), input_shape.output});
  auto perm = Range(scope, Sub(scope, Rank(scope, stack), 1), -1, -1);
  auto split_shape = Reshape(scope, Transpose(scope, stack, perm), {-1});
  auto axes = Range(scope, Const(scope, 0), Size(scope, split_shape.output), 2);
  auto input_grad = ReduceSum(
      scope, Reshape(scope, grad_inputs[0], split_shape.output), axes.output);
  grad_outputs->push_back(input_grad.output);
  grad_outputs->push_back(NoGradient());
  return scope.status();
}
REGISTER_GRADIENT_OP("Tile", TileGrad);

// Create a constant of the provided d_type;
Output ConstHelper(const Scope& scope, int value, DataType d_type) {
  return Cast(scope, Const(scope, value), d_type);
}

// Adds the batch offsets to the given indices and returns the results.
Output GetBatchIndices(const Scope& scope, const Output& params_shape,
                       const Output& indices, int batch_dims) {
  Output batch_indices = indices;
  auto indices_ndims = Rank(scope, indices);
  auto casted_params_shape = Cast(scope, params_shape, indices.type());
  Output accum_dim_value = ConstHelper(scope, 1, indices.type());
  for (int dim = batch_dims; dim > 0; dim--) {
    Output dim_value = Slice(scope, casted_params_shape, {dim - 1}, {1});
    accum_dim_value = Multiply(scope, accum_dim_value,
                               Slice(scope, casted_params_shape, {dim}, {1}));
    auto start = ConstHelper(scope, 0, indices.type());
    auto step = ConstHelper(scope, 1, indices.type());
    Output dim_indices = Range(scope, start, Squeeze(scope, dim_value), step);
    dim_indices = Multiply(scope, dim_indices, accum_dim_value);
    auto one = Cast(scope, Const(scope, {1}), indices.type());
    auto dim_shape = Concat(
        scope,
        {Output(Tile(scope, one, Const(scope, {dim - 1}))), dim_value,
         Output(Tile(scope, one,
                     ExpandDims(scope, Sub(scope, indices_ndims, dim), 0)))},
        /*axis=*/0);
    batch_indices =
        Add(scope, batch_indices, Reshape(scope, dim_indices, dim_shape));
  }

  return batch_indices;
}

Output BatchGatherGrad(const Scope& scope, Output params_shape, Output values,
                       Output indices, int batch_dims, Output gather_dim_size) {
  // Axis is the first non-batch dimension.
  auto indices_size = ExpandDims(scope, Size(scope, indices), 0);
  Output outer_shape, flat_values_shape;
  if (batch_dims != 0) {
    auto values_shape = Shape(scope, values);
    // Add the batch offsets to indices and flatten the batch dimensions.
    outer_shape = Slice(scope, values_shape, {0}, {batch_dims});
    auto inner_shape =
        Slice(scope, Slice(scope, values_shape, {batch_dims}, {-1}), {1}, {-1});
    auto batch_size = Prod(scope, outer_shape, /*axis=*/0);
    flat_values_shape = Concat(scope, {{-1}, inner_shape}, /*axis=*/0);
    gather_dim_size = Multiply(scope, gather_dim_size, batch_size);
    indices = GetBatchIndices(scope, params_shape, indices, batch_dims);
    values = Reshape(scope, values, flat_values_shape);
  }

  indices = Reshape(scope, indices, indices_size);
  Output params_grad =
      UnsortedSegmentSum(scope, values, indices, gather_dim_size);

  if (batch_dims != 0) {
    // Put back the batch dimensions.
    params_grad = Reshape(scope, params_grad, params_shape);
  }
  return params_grad;
}

absl::Status GatherV2Grad(const Scope& scope, const Operation& op,
                          const std::vector<Output>& grad_inputs,
                          std::vector<Output>* grad_outputs) {
  if (op.num_inputs() != 3) {
    return errors::InvalidArgument("Gather requires 3 inputs");
  }
  if (grad_inputs.size() != 1) {
    return errors::InvalidArgument("Gather grad requires 1 grad input");
  }

  // params can be large, so colocate the shape calculation with it.
  // params can be very large for sparse model, array_ops.shape raises
  // exception on the Windows platform when any dimension is larger than
  // int32. params_shape is not used in optimizer apply_sparse gradients,
  // so it's fine to convert it back to int32 regardless of truncation.
  auto params = op.input(0);
  auto colocate_scope = scope.ColocateWith(params);
  Shape::Attrs shape_attrs;
  shape_attrs.out_type_ = DT_INT64;
  auto params_shape64 = Shape(colocate_scope, params, shape_attrs);
  Output params_shape = Cast(colocate_scope, params_shape64, DT_INT32);

  auto indices = op.input(1);
  auto indices_size = ExpandDims(scope, Size(scope, indices), 0);
  auto axis = op.input(2);
  auto axis_expand = ExpandDims(scope, axis, 0);

  int batch_dims;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(op.node()->attrs(), "batch_dims", &batch_dims));
  if (batch_dims < 0) {
    // TODO(bdodson): Figure out if we can find the param rank here, like the
    // python implementation does.
    return errors::InvalidArgument(
        "C++ GatherV2 gradient does not support negative batch_dims.");
  }

  // Handle axis by transposing the axis dimension to be the first non-batch
  // dimension, compute the gradient and transpose the result back.
  auto outer_shape = Slice(scope, params_shape, {0}, axis_expand);
  auto inner_shape =
      Slice(scope, Slice(scope, params_shape, axis_expand, {-1}), {1}, {-1});
  auto values_shape = Concat(scope, {outer_shape, {-1}, inner_shape}, 0);
  auto values_dims = Size(scope, values_shape);
  auto axis_dims = Size(scope, outer_shape);

  Output outer_batches_indices = Range(scope, 0, batch_dims, /*delta=*/1);
  Output batch_axis_indices = Range(scope, batch_dims, axis_dims, /*delta=*/1);
  Output inner_axes_indices =
      Range(scope, Add(scope, axis_dims, 1), values_dims, /*delta=*/1);
  Output axis_dims_expand = ExpandDims(scope, axis_dims, 0);

  auto values = Reshape(scope, grad_inputs[0], values_shape);

  // Move values[axis] up to values[batch_dims]
  Output transpose_dims = Concat(scope,
                                 {outer_batches_indices, axis_dims_expand,
                                  batch_axis_indices, inner_axes_indices},
                                 0);
  auto values_transpose = Transpose(scope, values, transpose_dims);
  Output gather_dim_size =
      Squeeze(scope, Slice(scope, params_shape, axis_expand, {1}));
  params_shape = Gather(scope, params_shape, transpose_dims);

  auto params_grad = BatchGatherGrad(scope, params_shape, values_transpose,
                                     indices, batch_dims, gather_dim_size);

  // Inverts the above transpose by moving dimension batch_dims back to its
  // original position.
  Output invert_transpose_dims = Concat(scope,
                                        {outer_batches_indices,
                                         Add(scope, batch_axis_indices, 1),
                                         {batch_dims},
                                         inner_axes_indices},
                                        0);

  params_grad = Transpose(scope, params_grad, invert_transpose_dims);

  grad_outputs->push_back(params_grad);
  grad_outputs->push_back(NoGradient());
  grad_outputs->push_back(NoGradient());
  return scope.status();
}

REGISTER_GRADIENT_OP("GatherV2", GatherV2Grad);

}  // anonymous namespace
}  // namespace ops
}  // namespace tensorflow
