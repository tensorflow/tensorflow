/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/dtensor/mlir/expansions/strided_slice_spmd_expander.h"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "absl/numeric/bits.h"
#include "absl/status/status.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/dtensor/cc/slice_util.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/mlir/collectives.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/shape_utils.h"
#include "tensorflow/dtensor/mlir/value_utils.h"
#include "tensorflow/dtensor/proto/layout.pb.h"

namespace tensorflow {
namespace dtensor {

namespace {

// Tokenizes the arguments of a StridedSlice Op to a vector of Tokens.
// Most arguments are converted directly. If begin, end, or strides are dynamic
// shaped then the converted Tokens will have dynamic_mask set to true.
// Fails if the rank is dynamic.
template <typename T>
StatusOr<std::vector<slice_util::Token>> TokenizeOp(T strided_slice) {
  std::vector<slice_util::Token> tokens;

  llvm::SmallVector<int64_t, 4> spec_begin;
  llvm::SmallVector<int64_t, 4> spec_end;
  llvm::SmallVector<int64_t, 4> spec_strides;

  TF_ASSIGN_OR_RETURN(llvm::ArrayRef<int64_t> strides_shape,
                      GetShapeOfValue(strided_slice.getStrides(),
                                      /*fail_on_dynamic=*/true));
  if (strides_shape.size() != 1)
    return absl::InvalidArgumentError(
        "strides input to strided operation is not rank 1");

  int64_t spec_rank = strides_shape[0];
  bool dynamic = false;
  tokens.reserve(spec_rank);

  if (!ExtractConstVectorFromValue(strided_slice.getStrides(), &spec_strides)
           .ok()) {
    spec_strides.resize(spec_rank, 0);
    dynamic = true;
  }
  if (ExtractConstVectorFromValue(strided_slice.getBegin(), &spec_begin).ok()) {
    if (spec_begin.size() != spec_rank)
      return absl::InvalidArgumentError(
          "rank of begin input to strided operation does not equal rank of "
          "strides input");
  } else {
    spec_begin.resize(spec_rank, 0);
    dynamic = true;
  }
  if (ExtractConstVectorFromValue(strided_slice.getEnd(), &spec_end).ok()) {
    if (spec_end.size() != spec_rank)
      return absl::InvalidArgumentError(
          "rank of end input to strided operation does not equal rank of "
          "strides input");
  } else {
    spec_end.resize(spec_rank, 0);
    dynamic = true;
  }
  const uint64_t new_axis_mask = strided_slice.getNewAxisMask();
  const uint64_t shrink_axis_mask = strided_slice.getShrinkAxisMask();
  const uint64_t spec_begin_mask = strided_slice.getBeginMask();
  const uint64_t spec_end_mask = strided_slice.getEndMask();
  uint64_t ellipsis_mask = strided_slice.getEllipsisMask();

  if (absl::popcount(ellipsis_mask) > 1)
    return absl::InvalidArgumentError(
        "strided slice only supports at most one ellipsis");

  for (int64_t token_index = 0; token_index < spec_rank; ++token_index) {
    uint64_t bit = 1 << token_index;
    slice_util::Token::TokenType token_type = slice_util::Token::REGULAR;
    if (bit & new_axis_mask) {
      token_type = slice_util::Token::NEW_AXIS;
    } else if (bit & shrink_axis_mask) {
      token_type = slice_util::Token::SHRINK_AXIS;
    } else if (bit & ellipsis_mask) {
      token_type = slice_util::Token::ELLIPSIS;
    }
    tokens.emplace_back(token_type,
                        /*begin=*/spec_begin[token_index],
                        /*end=*/spec_end[token_index],
                        /*stride=*/spec_strides[token_index],
                        /*dynamic_mask=*/dynamic,
                        /*begin_mask=*/bit & spec_begin_mask,
                        /*end_mask=*/bit & spec_end_mask);
  }

  return tokens;
}

// Updates an Op's inputs and attributes using the Token vector.
// NOTE(feyu): This function only updates the end argument because currently
// this is the only meaningful change when a global Token vector is converted
// to the local Token vector.
template <typename T>
absl::Status UpdateOpFromTokens(T strided_slice,
                                const std::vector<slice_util::Token>& tokens) {
  mlir::OpBuilder builder(strided_slice);
  llvm::SmallVector<int64_t, 4> end;
  end.reserve(tokens.size());
  for (const auto& token : tokens) {
    end.push_back(token.end);
  }
  assert(end.size() == tokens.size());
  mlir::Value new_end = IntConstWithMatchingType(
      builder, strided_slice.getLoc(), end, strided_slice.getBegin().getType());
  strided_slice.getEndMutable().assign(new_end);
  return absl::OkStatus();
}
}  // namespace

//
// StridedSlice
//
StatusOr<mlir::Operation*> StridedSliceSPMDExpander::ExpandOp(
    mlir::Operation* op) {
  mlir::OpBuilder builder(op);

  auto strided_slice_op = mlir::cast<mlir::TF::StridedSliceOp>(op);
  TF_ASSIGN_OR_RETURN(Layout input_layout, ExtractRequiredLayoutFromOperand(
                                               strided_slice_op.getInput()));
  TF_ASSIGN_OR_RETURN(Layout output_layout,
                      ExtractRequiredSingleLayoutFromOp(op));
  TF_ASSIGN_OR_RETURN(
      const llvm::ArrayRef<int64_t> global_input_shape,
      GetGlobalShapeOfValueFromDTensorLayout(strided_slice_op.getInput()));

  TF_ASSIGN_OR_RETURN(auto tokens, TokenizeOp(strided_slice_op));

  TF_ASSIGN_OR_RETURN(
      auto forward,
      slice_util::CreateAndRun<slice_util::ForwardLayoutInference>(
          tokens, input_layout, global_input_shape));

  TF_ASSIGN_OR_RETURN(mlir::Value new_input,
                      EmitRelayout(strided_slice_op.getInput(), input_layout,
                                   forward.expander_input_layout()));

  TF_RETURN_IF_ERROR(
      UpdateOpFromTokens(strided_slice_op, forward.local_tokens()));

  strided_slice_op.getInputMutable().assign(new_input);

  op = InferSPMDExpandedLocalShape(op);

  // Do a final relayout to the correct output layout in case there are any
  // differences between intermediate_output_layout and output_layout.
  llvm::SmallPtrSet<mlir::Operation*, 4> newly_created_ops;

  TF_ASSIGN_OR_RETURN(mlir::Value output,
                      EmitRelayout(strided_slice_op.getOutput(),
                                   forward.expander_value_layout(),
                                   output_layout, &newly_created_ops));

  strided_slice_op.getOutput().replaceAllUsesExcept(output, newly_created_ops);

  return output.getDefiningOp();
}

StatusOr<llvm::DenseMap<int, Layout>>
StridedSliceSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  mlir::TF::StridedSliceOp strided_slice_op =
      mlir::cast<mlir::TF::StridedSliceOp>(op);
  TF_ASSIGN_OR_RETURN(const Mesh mesh, ExtractDeviceMeshEnclosingCluster(op));

  TF_ASSIGN_OR_RETURN(const llvm::ArrayRef<int64_t> global_input_shape,
                      GetShapeOfValue(strided_slice_op.getInput(),
                                      /*fail_on_dynamic=*/true));

  if (input_layouts.find(0) == input_layouts.end()) {
    return llvm::DenseMap<int, Layout>();
  }
  TF_ASSIGN_OR_RETURN(auto tokens, TokenizeOp(strided_slice_op));
  TF_ASSIGN_OR_RETURN(
      auto forward,
      slice_util::CreateAndRun<slice_util::ForwardLayoutInference>(
          tokens, input_layouts.lookup(0), global_input_shape));

  return llvm::DenseMap<int, Layout>({{0, forward.expander_value_layout()}});
}

StatusOr<llvm::DenseMap<int, Layout>>
StridedSliceSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  mlir::TF::StridedSliceOp strided_slice_op =
      mlir::cast<mlir::TF::StridedSliceOp>(op);
  TF_ASSIGN_OR_RETURN(const Mesh mesh, ExtractDeviceMeshEnclosingCluster(op));

  TF_ASSIGN_OR_RETURN(const llvm::ArrayRef<int64_t> global_input_shape,
                      GetShapeOfValue(strided_slice_op.getInput(),
                                      /*fail_on_dynamic=*/true));

  llvm::DenseMap<int, Layout> input_layouts(strided_slice_op.getNumOperands());
  // Set replicated layout for begin, end, and strides operands.
  input_layouts[1] = Layout::ReplicatedOnMesh(mesh, /*rank=*/1);
  input_layouts[2] = Layout::ReplicatedOnMesh(mesh, /*rank=*/1);
  input_layouts[3] = Layout::ReplicatedOnMesh(mesh, /*rank=*/1);

  // input
  if (output_layouts.find(0) == output_layouts.end()) {
    return input_layouts;
  }

  const Layout& output_layout = output_layouts.lookup(0);
  TF_ASSIGN_OR_RETURN(auto tokens, TokenizeOp(strided_slice_op));
  TF_ASSIGN_OR_RETURN(
      auto backward,
      slice_util::CreateAndRun<slice_util::BackwardLayoutInference>(
          tokens, output_layout, global_input_shape));

  input_layouts[0] = backward.expander_input_layout();
  return input_layouts;
}

//
//  TensorStridedSliceUpdate
//
StatusOr<mlir::Operation*> TensorStridedSliceUpdateSPMDExpander::ExpandOp(
    mlir::Operation* op) {
  mlir::TF::TensorStridedSliceUpdateOp strided_slice_op =
      llvm::cast<mlir::TF::TensorStridedSliceUpdateOp>(op);
  TF_ASSIGN_OR_RETURN(
      const Layout input_layout,
      ExtractRequiredLayoutFromOperand(strided_slice_op.getInput()));
  TF_ASSIGN_OR_RETURN(
      const Layout value_layout,
      ExtractRequiredLayoutFromOperand(strided_slice_op.getValue()));
  TF_ASSIGN_OR_RETURN(const Layout output_layout,
                      ExtractRequiredSingleLayoutFromOp(op));

  TF_ASSIGN_OR_RETURN(
      const llvm::ArrayRef<int64_t> global_input_shape,
      GetGlobalShapeOfValueFromDTensorLayout(strided_slice_op.getInput()));

  TF_ASSIGN_OR_RETURN(auto tokens, TokenizeOp(strided_slice_op));

  TF_ASSIGN_OR_RETURN(
      auto forward,
      slice_util::CreateAndRun<slice_util::ForwardLayoutInference>(
          tokens, input_layout, global_input_shape));

  TF_ASSIGN_OR_RETURN(mlir::Value new_input,
                      EmitRelayout(strided_slice_op.getInput(), input_layout,
                                   forward.expander_input_layout()));

  TF_ASSIGN_OR_RETURN(mlir::Value new_value,
                      EmitRelayout(strided_slice_op.getValue(), value_layout,
                                   forward.expander_value_layout()));

  strided_slice_op.getInputMutable().assign(new_input);
  strided_slice_op.getValueMutable().assign(new_value);

  TF_RETURN_IF_ERROR(
      UpdateOpFromTokens(strided_slice_op, forward.local_tokens()));

  op = InferSPMDExpandedLocalShape(op);

  mlir::OpBuilder builder(op);

  // Do a final relayout to the correct output layout in case there are any
  // differences between intermediate_output_layout and output_layout.
  llvm::SmallPtrSet<mlir::Operation*, 4> newly_created_ops;

  TF_ASSIGN_OR_RETURN(mlir::Value output,
                      EmitRelayout(strided_slice_op.getOutput(),
                                   forward.expander_input_layout(),
                                   output_layout, &newly_created_ops));

  strided_slice_op.getOutput().replaceAllUsesExcept(output, newly_created_ops);

  return output.getDefiningOp();
}

StatusOr<llvm::DenseMap<int, Layout>>
TensorStridedSliceUpdateSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  mlir::TF::TensorStridedSliceUpdateOp strided_slice_op =
      mlir::cast<mlir::TF::TensorStridedSliceUpdateOp>(op);
  TF_ASSIGN_OR_RETURN(const llvm::ArrayRef<int64_t> global_input_shape,
                      GetShapeOfValue(strided_slice_op.getInput(),
                                      /*fail_on_dynamic=*/true));

  TF_ASSIGN_OR_RETURN(auto tokens, TokenizeOp(strided_slice_op));
  // We have a choice to determine the output layout, we will default to use
  // input_layout if available, otherwise we will expand value_layout and use
  // that.
  std::vector<Layout> candidates;
  if (input_layouts.find(0) != input_layouts.end()) {
    // If we have an input_layout, prefer to keep it.
    candidates.push_back(input_layouts.lookup(0));
  }

  if (input_layouts.find(4) != input_layouts.end()) {
    // When we don't have the input layout, use value layout to 'create' the
    // input layout. We do this by applying the backward inference.
    // This is because in the case of a normal strided slice the layout of
    // value would be output layout.
    const Layout& value_layout = input_layouts.lookup(4);
    TF_ASSIGN_OR_RETURN(
        auto backward,
        slice_util::CreateAndRun<slice_util::BackwardLayoutInference>(
            tokens, value_layout, global_input_shape));
    candidates.push_back(backward.expander_input_layout());
  }
  if (candidates.empty()) {
    return llvm::DenseMap<int, Layout>();
  }

  TF_ASSIGN_OR_RETURN(auto input_layout, GetLeastShardedLayout(candidates));

  return llvm::DenseMap<int, Layout>({{0, input_layout}});
}

StatusOr<llvm::DenseMap<int, Layout>>
TensorStridedSliceUpdateSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  mlir::TF::TensorStridedSliceUpdateOp strided_slice_op =
      mlir::cast<mlir::TF::TensorStridedSliceUpdateOp>(op);
  TF_ASSIGN_OR_RETURN(const Mesh mesh, ExtractDeviceMeshEnclosingCluster(op));

  TF_ASSIGN_OR_RETURN(const llvm::ArrayRef<int64_t> global_input_shape,
                      GetShapeOfValue(strided_slice_op.getInput(),
                                      /*fail_on_dynamic=*/true));

  TF_ASSIGN_OR_RETURN(auto tokens, TokenizeOp(strided_slice_op));

  llvm::DenseMap<int, Layout> input_layouts(strided_slice_op.getNumOperands());
  // Set replicated layout for begin, end, and strides operands.
  input_layouts[1] = Layout::ReplicatedOnMesh(mesh, /*rank=*/1);
  input_layouts[2] = Layout::ReplicatedOnMesh(mesh, /*rank=*/1);
  input_layouts[3] = Layout::ReplicatedOnMesh(mesh, /*rank=*/1);

  // input and value layouts
  if (output_layouts.find(0) != output_layouts.end()) {
    const Layout& output_layout = output_layouts.lookup(0);
    input_layouts[0] = output_layout;

    // We also need a layout for value as well, and for that we just take the
    // forward inference of the input layout.
    TF_ASSIGN_OR_RETURN(
        auto forward,
        slice_util::CreateAndRun<slice_util::ForwardLayoutInference>(
            tokens, output_layout, global_input_shape));
    input_layouts[4] = forward.expander_value_layout();
  }

  return input_layouts;
}

//
// StridedSliceGrad
//
StatusOr<mlir::Operation*> StridedSliceGradSPMDExpander::ExpandOp(
    mlir::Operation* op) {
  auto strided_slice_grad_op = llvm::cast<mlir::TF::StridedSliceGradOp>(op);
  TF_ASSIGN_OR_RETURN(
      const Layout input_layout,
      ExtractRequiredLayoutFromOperand(strided_slice_grad_op.getDy()));
  TF_ASSIGN_OR_RETURN(const Layout output_layout,
                      ExtractRequiredSingleLayoutFromOp(op));

  TF_ASSIGN_OR_RETURN(const llvm::ArrayRef<int64_t> global_output_shape,
                      GetGlobalShapeOfValueFromDTensorLayout(
                          strided_slice_grad_op.getOutput()));

  TF_ASSIGN_OR_RETURN(auto tokens, TokenizeOp(strided_slice_grad_op));

  TF_ASSIGN_OR_RETURN(
      auto backward,
      slice_util::CreateAndRun<slice_util::BackwardLayoutInference>(
          tokens, input_layout, global_output_shape));

  TF_ASSIGN_OR_RETURN(mlir::Value new_dy,
                      EmitRelayout(strided_slice_grad_op.getDy(), input_layout,
                                   backward.expander_value_layout()));

  TF_RETURN_IF_ERROR(
      UpdateOpFromTokens(strided_slice_grad_op, backward.local_tokens()));
  strided_slice_grad_op.getDyMutable().assign(new_dy);

  mlir::OpBuilder builder(op);

  // The shape input to StridedSliceGrad will still be global, so we need to
  // compute the local shape update it.
  std::vector<int64_t> computed_output_shape =
      backward.expander_input_layout().LocalShapeFromGlobalShape(
          global_output_shape);
  mlir::Value new_shape = IntConstWithMatchingType(
      builder, strided_slice_grad_op.getLoc(), computed_output_shape,
      strided_slice_grad_op.getBegin().getType());
  strided_slice_grad_op.getShapeMutable().assign(new_shape);

  op = InferSPMDExpandedLocalShape(op);

  // Do a final relayout to the correct output layout in case there are any
  // differences between intermediate_output_layout and output_layout.
  llvm::SmallPtrSet<mlir::Operation*, 4> newly_created_ops;

  TF_ASSIGN_OR_RETURN(mlir::Value output,
                      EmitRelayout(strided_slice_grad_op.getOutput(),
                                   backward.expander_input_layout(),
                                   output_layout, &newly_created_ops));

  strided_slice_grad_op.getOutput().replaceAllUsesExcept(output,
                                                         newly_created_ops);

  return output.getDefiningOp();
}

StatusOr<llvm::DenseMap<int, Layout>>
StridedSliceGradSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  // If the input layout is missing, don't return an output layout.
  if (input_layouts.find(4) == input_layouts.end())
    return llvm::DenseMap<int, Layout>();

  mlir::TF::StridedSliceGradOp strided_slice_grad_op =
      mlir::cast<mlir::TF::StridedSliceGradOp>(op);
  TF_ASSIGN_OR_RETURN(const llvm::ArrayRef<int64_t> global_output_shape,
                      GetShapeOfValue(strided_slice_grad_op.getOutput(),
                                      /*fail_on_dynamic=*/true));
  const Layout& input_layout = input_layouts.lookup(4);

  TF_ASSIGN_OR_RETURN(auto tokens, TokenizeOp(strided_slice_grad_op));
  TF_ASSIGN_OR_RETURN(
      auto backward,
      slice_util::CreateAndRun<slice_util::BackwardLayoutInference>(
          tokens, input_layout, global_output_shape));

  return llvm::DenseMap<int, Layout>({{0, backward.expander_input_layout()}});
}

StatusOr<llvm::DenseMap<int, Layout>>
StridedSliceGradSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  mlir::TF::StridedSliceGradOp strided_slice_grad_op =
      mlir::cast<mlir::TF::StridedSliceGradOp>(op);
  TF_ASSIGN_OR_RETURN(const Mesh mesh, ExtractDeviceMeshEnclosingCluster(op));

  TF_ASSIGN_OR_RETURN(const llvm::ArrayRef<int64_t> global_output_shape,
                      GetShapeOfValue(strided_slice_grad_op.getOutput(),
                                      /*fail_on_dynamic=*/true));
  llvm::DenseMap<int, Layout> input_layouts(
      strided_slice_grad_op.getNumOperands());
  // Set replicated layout for shape, begin, end, stride operands.
  input_layouts[0] = Layout::ReplicatedOnMesh(mesh, /*rank=*/1);
  input_layouts[1] = Layout::ReplicatedOnMesh(mesh, /*rank=*/1);
  input_layouts[2] = Layout::ReplicatedOnMesh(mesh, /*rank=*/1);
  input_layouts[3] = Layout::ReplicatedOnMesh(mesh, /*rank=*/1);

  // dy
  if (output_layouts.find(0) != output_layouts.end()) {
    const Layout& output_layout = output_layouts.lookup(0);
    TF_ASSIGN_OR_RETURN(auto tokens, TokenizeOp(strided_slice_grad_op));

    TF_ASSIGN_OR_RETURN(
        auto forward,
        slice_util::CreateAndRun<slice_util::ForwardLayoutInference>(
            tokens, output_layout, global_output_shape));
    input_layouts[4] = forward.expander_value_layout();
  }

  return input_layouts;
}

}  // namespace dtensor
}  // namespace tensorflow
