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

#include "tensorflow/cc/ops/array_ops_internal.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/lib/strings/strcat.h"

#include "tensorflow/cc/framework/grad_op_registry.h"
#include "tensorflow/cc/framework/gradients.h"

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

Status PackGrad(const Scope& scope, const Operation& op,
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

Status UnpackGrad(const Scope& scope, const Operation& op,
                  const std::vector<Output>& grad_inputs,
                  std::vector<Output>* grad_outputs) {
  int axis;
  TF_RETURN_IF_ERROR(GetNodeAttr(op.node()->attrs(), "axis", &axis));
  grad_outputs->push_back(Stack(scope, grad_inputs, Stack::Axis(axis)));
  return scope.status();
}
REGISTER_GRADIENT_OP("Unpack", UnpackGrad);

Status IdentityGrad(const Scope& scope, const Operation& op,
                    const std::vector<Output>& grad_inputs,
                    std::vector<Output>* grad_outputs) {
  grad_outputs->push_back(Identity(scope, grad_inputs[0]));
  return scope.status();
}
REGISTER_GRADIENT_OP("Identity", IdentityGrad);

Status RefIdentityGrad(const Scope& scope, const Operation& op,
                       const std::vector<Output>& grad_inputs,
                       std::vector<Output>* grad_outputs) {
  grad_outputs->push_back(Identity(scope, grad_inputs[0]));
  return scope.status();
}
REGISTER_GRADIENT_OP("RefIdentity", RefIdentityGrad);

Status QuantizeAndDequantizeGrad(const Scope& scope, const Operation& op,
                                 const std::vector<Output>& grad_inputs,
                                 std::vector<Output>* grad_outputs) {
  grad_outputs->push_back(Identity(scope, grad_inputs[0]));
  return scope.status();
}
REGISTER_GRADIENT_OP("QuantizeAndDequantize", QuantizeAndDequantizeGrad);

Status QuantizeAndDequantizeV2Grad(const Scope& scope, const Operation& op,
                                   const std::vector<Output>& grad_inputs,
                                   std::vector<Output>* grad_outputs) {
  grad_outputs->push_back(Identity(scope, grad_inputs[0]));
  grad_outputs->push_back(NoGradient());
  grad_outputs->push_back(NoGradient());
  return scope.status();
}
REGISTER_GRADIENT_OP("QuantizeAndDequantizeV2", QuantizeAndDequantizeV2Grad);

Status QuantizeAndDequantizeV3Grad(const Scope& scope, const Operation& op,
                                   const std::vector<Output>& grad_inputs,
                                   std::vector<Output>* grad_outputs) {
  grad_outputs->push_back(Identity(scope, grad_inputs[0]));
  grad_outputs->push_back(NoGradient());
  grad_outputs->push_back(NoGradient());
  grad_outputs->push_back(NoGradient());
  return scope.status();
}
REGISTER_GRADIENT_OP("QuantizeAndDequantizeV3", QuantizeAndDequantizeV3Grad);

Status SplitGrad(const Scope& scope, const Operation& op,
                 const std::vector<Output>& grad_inputs,
                 std::vector<Output>* grad_outputs) {
  grad_outputs->push_back(NoGradient());
  grad_outputs->push_back(Concat(scope, grad_inputs, op.input(0)));
  return scope.status();
}
REGISTER_GRADIENT_OP("Split", SplitGrad);

Status DiagGrad(const Scope& scope, const Operation& op,
                const std::vector<Output>& grad_inputs,
                std::vector<Output>* grad_outputs) {
  grad_outputs->push_back(DiagPart(scope, grad_inputs[0]));
  return scope.status();
}
REGISTER_GRADIENT_OP("Diag", DiagGrad);

Status DiagPartGrad(const Scope& scope, const Operation& op,
                    const std::vector<Output>& grad_inputs,
                    std::vector<Output>* grad_outputs) {
  grad_outputs->push_back(Diag(scope, grad_inputs[0]));
  return scope.status();
}
REGISTER_GRADIENT_OP("DiagPart", DiagPartGrad);

Status MatrixDiagGrad(const Scope& scope, const Operation& op,
                      const std::vector<Output>& grad_inputs,
                      std::vector<Output>* grad_outputs) {
  grad_outputs->push_back(MatrixDiagPart(scope, grad_inputs[0]));
  return scope.status();
}
REGISTER_GRADIENT_OP("MatrixDiag", MatrixDiagGrad);

Status MatrixBandPartGrad(const Scope& scope, const Operation& op,
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

Status GatherNdGrad(const Scope& scope, const Operation& op,
                    const std::vector<Output>& grad_inputs,
                    std::vector<Output>* grad_outputs) {
  auto ref = op.input(0);
  auto ref_shape = Shape(scope, ref);
  auto indices = op.input(1);
  grad_outputs->push_back(ScatterNd(scope, indices, grad_inputs[0], ref_shape));
  grad_outputs->push_back(NoGradient());
  return scope.status();
}
REGISTER_GRADIENT_OP("GatherNd", GatherNdGrad);

Status CheckNumericsGrad(const Scope& scope, const Operation& op,
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

Status ReshapeGrad(const Scope& scope, const Operation& op,
                   const std::vector<Output>& grad_inputs,
                   std::vector<Output>* grad_outputs) {
  auto input_shape = Shape(scope, op.input(0));
  grad_outputs->push_back(Reshape(scope, grad_inputs[0], input_shape));
  grad_outputs->push_back(NoGradient());
  return scope.status();
}
REGISTER_GRADIENT_OP("Reshape", ReshapeGrad);

Status ExpandDimsGrad(const Scope& scope, const Operation& op,
                      const std::vector<Output>& grad_inputs,
                      std::vector<Output>* grad_outputs) {
  auto input_shape = Shape(scope, op.input(0));
  grad_outputs->push_back(Reshape(scope, grad_inputs[0], input_shape));
  grad_outputs->push_back(NoGradient());
  return scope.status();
}
REGISTER_GRADIENT_OP("ExpandDims", ExpandDimsGrad);

Status SqueezeGrad(const Scope& scope, const Operation& op,
                   const std::vector<Output>& grad_inputs,
                   std::vector<Output>* grad_outputs) {
  auto input_shape = Shape(scope, op.input(0));
  grad_outputs->push_back(Reshape(scope, grad_inputs[0], input_shape));
  return scope.status();
}
REGISTER_GRADIENT_OP("Squeeze", SqueezeGrad);

Status TransposeGrad(const Scope& scope, const Operation& op,
                     const std::vector<Output>& grad_inputs,
                     std::vector<Output>* grad_outputs) {
  auto inverted_perm = InvertPermutation(scope, op.input(1));
  grad_outputs->push_back(Transpose(scope, grad_inputs[0], inverted_perm));
  grad_outputs->push_back(NoGradient());
  return scope.status();
}
REGISTER_GRADIENT_OP("Transpose", TransposeGrad);

Status ReverseSequenceGrad(const Scope& scope, const Operation& op,
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

Status ReverseGrad(const Scope& scope, const Operation& op,
                   const std::vector<Output>& grad_inputs,
                   std::vector<Output>* grad_outputs) {
  auto reverse_dims = op.input(1);
  grad_outputs->push_back(Reverse(scope, grad_inputs[0], reverse_dims));
  grad_outputs->push_back(NoGradient());
  return scope.status();
}
REGISTER_GRADIENT_OP("ReverseV2", ReverseGrad);

Status ScatterNdGrad(const Scope& scope, const Operation& op,
                     const std::vector<Output>& grad_inputs,
                     std::vector<Output>* grad_outputs) {
  auto indices = op.input(0);
  grad_outputs->push_back(NoGradient());
  grad_outputs->push_back(GatherNd(scope, grad_inputs[0], indices));
  grad_outputs->push_back(NoGradient());
  return scope.status();
}
REGISTER_GRADIENT_OP("ScatterNd", ScatterNdGrad);

Status ScatterNdNonAliasingAddGrad(const Scope& scope, const Operation& op,
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
Status PadGrad(const Scope& scope, const Operation& op,
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

Status SpaceToBatchGrad(const Scope& scope, const Operation& op,
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

Status SpaceToBatchNDGrad(const Scope& scope, const Operation& op,
                          const std::vector<Output>& grad_inputs,
                          std::vector<Output>* grad_outputs) {
  grad_outputs->push_back(
      BatchToSpaceND(scope, grad_inputs[0], op.input(1), op.input(2)));
  grad_outputs->push_back(NoGradient());
  grad_outputs->push_back(NoGradient());
  return scope.status();
}
REGISTER_GRADIENT_OP("SpaceToBatchND", SpaceToBatchNDGrad);

Status BatchToSpaceGrad(const Scope& scope, const Operation& op,
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

Status BatchToSpaceNDGrad(const Scope& scope, const Operation& op,
                          const std::vector<Output>& grad_inputs,
                          std::vector<Output>* grad_outputs) {
  grad_outputs->push_back(
      SpaceToBatchND(scope, grad_inputs[0], op.input(1), op.input(2)));
  grad_outputs->push_back(NoGradient());
  grad_outputs->push_back(NoGradient());
  return scope.status();
}
REGISTER_GRADIENT_OP("BatchToSpaceND", BatchToSpaceNDGrad);

Status SpaceToDepthGrad(const Scope& scope, const Operation& op,
                        const std::vector<Output>& grad_inputs,
                        std::vector<Output>* grad_outputs) {
  int block_size;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(op.node()->attrs(), "block_size", &block_size));
  grad_outputs->push_back(DepthToSpace(scope, grad_inputs[0], block_size));
  return scope.status();
}
REGISTER_GRADIENT_OP("SpaceToDepth", SpaceToDepthGrad);

Status DepthToSpaceGrad(const Scope& scope, const Operation& op,
                        const std::vector<Output>& grad_inputs,
                        std::vector<Output>* grad_outputs) {
  int block_size;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(op.node()->attrs(), "block_size", &block_size));
  grad_outputs->push_back(SpaceToDepth(scope, grad_inputs[0], block_size));
  return scope.status();
}
REGISTER_GRADIENT_OP("DepthToSpace", DepthToSpaceGrad);

Status MirrorPadGrad(const Scope& scope, const Operation& op,
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
Status MirrorPadGradGrad(const Scope& scope, const Operation& op,
                         const std::vector<Output>& grad_inputs,
                         std::vector<Output>* grad_outputs) {
  string mode;
  TF_RETURN_IF_ERROR(GetNodeAttr(op.node()->attrs(), "mode", &mode));
  grad_outputs->push_back(MirrorPad(scope, grad_inputs[0], op.input(1), mode));
  grad_outputs->push_back(NoGradient());
  return scope.status();
}
REGISTER_GRADIENT_OP("MirrorPadGrad", MirrorPadGradGrad);

Status StridedSliceGradHelper(const Scope& scope, const Operation& op,
                              const std::vector<Output>& grad_inputs,
                              std::vector<Output>* grad_outputs) {
  Input x = Shape(scope, op.input(0));
  Input begin = op.input(1);
  Input end = op.input(2);
  Input strides = op.input(3);
  int64 begin_mask;
  int64 end_mask;
  int64 ellipsis_mask;
  int64 new_axis_mask;
  int64 shrink_axis_mask;
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

Status SliceGrad(const Scope& scope, const Operation& op,
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

}  // anonymous namespace
}  // namespace ops
}  // namespace tensorflow
