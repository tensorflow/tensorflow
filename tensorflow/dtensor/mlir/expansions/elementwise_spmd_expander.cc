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

#include "tensorflow/dtensor/mlir/expansions/elementwise_spmd_expander.h"

#include <iterator>
#include <optional>
#include <string>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/UseDefLists.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
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

StatusOr<llvm::SmallVector<int64_t, 4>> GetShape(mlir::Value value) {
  TF_ASSIGN_OR_RETURN(const auto shape, GetShapeOfValue(value));
  return llvm::SmallVector<int64_t, 4>{shape.begin(), shape.end()};
}

}  // namespace

StatusOr<mlir::Operation*> ElementwiseSPMDExpander::ExpandOp(
    mlir::Operation* op) {
  TF_ASSIGN_OR_RETURN(auto output_layout, ExtractSingleLayoutFromOp(op));
  assert(output_layout);

  mlir::OpBuilder builder(op);

  for (auto& operand : op->getOpOperands()) {
    // Verify that all output dimensions (including the dimensions added by
    // broadcasting) is more sharded then the correspdonding layout
    // configuration of the same dimension of every operands.
    TF_ASSIGN_OR_RETURN(auto operand_layout,
                        ExtractLayoutFromOperand(operand.get()));
    if (!operand_layout)
      return errors::InvalidArgument(
          "input layout of elementwise op must be known before SPMD "
          "expansion.");

    // For scalar operands, splitting is not needed.
    if (operand_layout->rank() == 0) continue;

    llvm::SmallPtrSet<mlir::Operation*, 4> newly_created_ops;

    // Note that due to broacasting, inputs and output tensors are aligned to
    // the right. Therefore, we truncate the output layout.
    const int rank_offset = output_layout->rank() - operand_layout->rank();

    // Get the desired layout for this operand.
    //
    // - Truncate: It should be the same layout as the output, truncated to
    //   take into account broadcasting on the rank and removing sharding in
    //   dimensions where the operand has size 1 (where we have broadcasting on
    //   the dimension size).
    //
    // - Relayout: If the output and operand have different sharding spec, we
    //   adjust the operands. For example, if operand is 'z,*' and output is
    //   '*.y', relayout operand to conform output. This means the SPMD safer
    //   and easeier. In future, we might do certain optimization to save FLops.
    //   For example, if all operands are 'x,y' and output is '*,*', relayouting
    //   output could be the choice (saving communications).
    auto truncated_layout = output_layout->Truncate(rank_offset, /*end=*/true);
    mlir::Value output;
    TF_ASSIGN_OR_RETURN(const auto& shape, ExtractGlobalInputShape(operand));
    absl::flat_hash_set<int> size_one_dims;
    for (int i = 0; i < shape.size(); ++i)
      if (shape[i] == 1) size_one_dims.emplace(i);
    TF_ASSIGN_OR_RETURN(truncated_layout,
                        truncated_layout.GetLayoutWithReducedDims(
                            size_one_dims, /*keep_dims=*/true));
    TF_ASSIGN_OR_RETURN(
        output, EmitRelayout(operand.get(), *operand_layout, truncated_layout));
    operand.set(output);
  }

  // If result is a resource, the shape of the result should be adjusted to
  // local value of the resource, based on the layout for output.
  // This logic is similar to VarHandle op SPMD expansion.
  //
  // Resource output is only likely to be for identity op. However, keeping
  // the checkgeneric here.
  auto op_result = op->getOpResult(0);
  if (IsResourceType(op_result)) {
    TF_RETURN_IF_ERROR(InferSPMDExpandedLocalShapeForResourceOutput(
        &op_result, output_layout.value(), builder.getContext()));
  }

  // For element-wise op SPMD expansion, given that operand layouts are
  // compatible to op's layout, op can simply be executed without any changes.
  return InferSPMDExpandedLocalShape(op);
}

// Computes output layouts of elementwise operation using broadcast logic for
// operands.
StatusOr<llvm::DenseMap<int, Layout>>
ElementwiseSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  TF_ASSIGN_OR_RETURN(std::optional<Layout> merged_operand_layout,
                      GetMergedOperandLayout(input_layouts, op));

  if (merged_operand_layout) {
    const int output_rank = ValueRank(op->getOpResult(0));
    if (output_rank == -1)
      return errors::InvalidArgument("Output has unknown rank");

    // We assume that all elementwise operations output a single tensor.
    return llvm::DenseMap<int, Layout>(
        {{0, merged_operand_layout->LeftPad(output_rank)}});
  }
  return llvm::DenseMap<int, Layout>();
}

// Computes input layouts of elementwise operation using broadcast logic for
// operands.
StatusOr<llvm::DenseMap<int, Layout>>
ElementwiseSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  // Do not infer any operand layout if no output layout is present.
  if (output_layouts.find(0) == output_layouts.end())
    return llvm::DenseMap<int, Layout>();

  Layout output_layout = output_layouts.lookup(0);
  llvm::DenseMap<int, Layout> input_layouts;

  for (const auto& operand_and_index : llvm::enumerate(op->getOperands())) {
    const int operand_index = operand_and_index.index();
    auto operand = operand_and_index.value();

    TF_ASSIGN_OR_RETURN(auto operand_shape, GetShape(operand));
    Layout output_layout_truncated = output_layout.Truncate(
        output_layout.sharding_spec_strs().size() - operand_shape.size(),
        /*end=*/true);
    auto inferred_operand_layout_strs =
        output_layout_truncated.sharding_spec_strs();

    if (inferred_operand_layout_strs.size() != operand_shape.size())
      return errors::FailedPrecondition(
          "Mismatch of operand shape size and layout size.");
    for (const auto& dim_shape_and_index : llvm::enumerate(operand_shape)) {
      const int dim_index = dim_shape_and_index.index();
      const int dim_shape = dim_shape_and_index.value();
      if (dim_shape <= 1) {
        inferred_operand_layout_strs[dim_index] = Layout::kUnshardedDim;
      }
    }
    TF_ASSIGN_OR_RETURN(
        auto inferred_operand_layout,
        Layout::GetLayout(inferred_operand_layout_strs, output_layout.mesh()));
    input_layouts[operand_index] = inferred_operand_layout;
  }
  return input_layouts;
}

}  // namespace dtensor
}  // namespace tensorflow
