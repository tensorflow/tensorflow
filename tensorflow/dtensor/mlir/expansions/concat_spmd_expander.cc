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

#include "tensorflow/dtensor/mlir/expansions/concat_spmd_expander.h"

#include <cstdint>
#include <optional>

#include "absl/status/status.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/mlir/collectives.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/shape_utils.h"
#include "tensorflow/dtensor/mlir/spmd_expander_common.h"
#include "tensorflow/dtensor/mlir/value_utils.h"

namespace tensorflow {
namespace dtensor {
namespace {

absl::Status VerifyConcatLayout(mlir::Value concat_dim_operand,
                                const Layout& concat_layout) {
  TF_ASSIGN_OR_RETURN(int64_t concat_dim_value,
                      ExtractConstIntFromValue(concat_dim_operand));
  for (const auto& shard_and_dimension :
       llvm::enumerate(concat_layout.num_shards())) {
    if (shard_and_dimension.index() == concat_dim_value &&
        shard_and_dimension.value() > 1) {
      return errors::InvalidArgument(
          "Concat op SPMD with concat dimension in sharded dimension is "
          "not supported.");
    }
  }

  return absl::OkStatus();
}

StatusOr<Layout> ReduceForConcatOutputLayout(mlir::Value concat_dim_operand,
                                             const Layout& layout) {
  TF_ASSIGN_OR_RETURN(int64_t concat_dim_value,
                      ExtractConstIntFromValue(concat_dim_operand));
  // Set concatenated dimension to replicated.
  return layout.GetLayoutWithReducedDims({concat_dim_value},
                                         /*keep_dims=*/true);
}

}  // namespace

StatusOr<mlir::Operation*> ConcatSPMDExpander::ExpandOp(mlir::Operation* op) {
  if (!llvm::isa<mlir::TF::ConcatOp, mlir::TF::ConcatV2Op>(op))
    return errors::InvalidArgument(
        "Requested SPMD Expansion for op that is not Concat or ConcatV2.");

  // Extract the concat dim. ConcatOp and ConcatV2Op define the dim at
  // different position.
  bool is_concat_v1 = llvm::isa<mlir::TF::ConcatOp>(op);
  const int concat_dim_operand_idx =
      is_concat_v1 ? 0 : op->getNumOperands() - 1;
  mlir::Value concat_dim = op->getOperand(concat_dim_operand_idx);

  // Ensure that Concat op is not sharded on concat-dimension.
  TF_ASSIGN_OR_RETURN(auto concat_output_layout,
                      ExtractRequiredSingleLayoutFromOp(op));
  TF_RETURN_IF_ERROR(VerifyConcatLayout(concat_dim, concat_output_layout));

  // Relayout all inputs to match output layout before concating.
  // This will ensure that a local concat is the correct thing to do.
  for (int i = 0; i < op->getNumOperands(); ++i) {
    // Skip relayout for the concat dim operand.
    if (i == concat_dim_operand_idx) continue;

    TF_ASSIGN_OR_RETURN(Layout layout,
                        ExtractRequiredLayoutFromOperand(op->getOperand(i)));
    TF_ASSIGN_OR_RETURN(
        mlir::Value new_input,
        EmitRelayout(op->getOperand(i), layout, concat_output_layout));

    op->setOperand(i, new_input);
  }

  return InferSPMDExpandedLocalShape(op);
}

StatusOr<llvm::DenseMap<int, Layout>> ConcatSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  // Verify operand layouts to ensure that there are no conflicting concat
  // operand layouts.
  const bool is_concat_v1 = llvm::isa<mlir::TF::ConcatOp>(op);
  auto begin_idx = is_concat_v1 ? 1 : 0;
  auto end_idx =
      is_concat_v1 ? op->getNumOperands() - 1 : op->getNumOperands() - 2;

  llvm::DenseMap<int, Layout> concat_operands_layouts;
  for (auto idx = begin_idx; idx <= end_idx; ++idx) {
    if (input_layouts.find(idx) != input_layouts.end())
      concat_operands_layouts[idx] = input_layouts.lookup(idx);
  }
  TF_ASSIGN_OR_RETURN(std::optional<Layout> concat_operand_layout,
                      GetMergedOperandLayout(concat_operands_layouts, op));
  // Concat/ConcatV2 has different operand index for concat dim. Retrieve the
  // correct concat dim value.
  const int concat_dim_operand_idx =
      is_concat_v1 ? 0 : op->getNumOperands() - 1;
  mlir::Value concat_dim = op->getOperand(concat_dim_operand_idx);

  // If consistent operand layout exists, propagate the operand layout to output
  // layout.
  if (concat_operand_layout) {
    TF_ASSIGN_OR_RETURN(
        const Layout reduced_concat_layout,
        ReduceForConcatOutputLayout(concat_dim, *concat_operand_layout));
    return llvm::DenseMap<int, Layout>({{0, reduced_concat_layout}});
  } else {
    return llvm::DenseMap<int, Layout>();
  }
}

StatusOr<llvm::DenseMap<int, Layout>> ConcatSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  // If output layout does not exist then do not infer any operand layouts.
  if (output_layouts.find(0) == output_layouts.end())
    return llvm::DenseMap<int, Layout>();

  Layout output_layout = output_layouts.lookup(0);
  const bool is_concat_v1 = llvm::isa<mlir::TF::ConcatOp>(op);
  // Concat/ConcatV2 has different operand index for concat dim. Retrieve the
  // correct concat dim value.
  const int concat_dim_operand_idx =
      is_concat_v1 ? 0 : op->getNumOperands() - 1;
  mlir::Value concat_dim = op->getOperand(concat_dim_operand_idx);

  // If suggested output layout exists, verify that concatenated dimension is
  // replicated to ensure no cross device communication is needed, then
  // propagate the output layout to all concat tensor operands' layout.
  // Set concatenated dimension to replicated.
  TF_ASSIGN_OR_RETURN(const Layout inferred_input_layout,
                      ReduceForConcatOutputLayout(concat_dim, output_layout));

  llvm::DenseMap<int, Layout> operand_layouts;

  for (auto index = 0; index < op->getNumOperands(); ++index) {
    if (index == concat_dim_operand_idx) continue;
    operand_layouts[index] = inferred_input_layout;
  }
  return operand_layouts;
}

}  // namespace dtensor
}  // namespace tensorflow
