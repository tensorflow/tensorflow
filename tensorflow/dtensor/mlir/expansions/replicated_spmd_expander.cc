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

#include "tensorflow/dtensor/mlir/expansions/replicated_spmd_expander.h"

#include <algorithm>
#include <vector>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/mlir/collectives.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/op_utils.h"
#include "tensorflow/dtensor/mlir/shape_utils.h"
#include "tensorflow/dtensor/mlir/spmd_expander_common.h"
#include "tensorflow/dtensor/mlir/value_utils.h"

namespace tensorflow {
namespace dtensor {

// Relayout all operands and outputs to replicated layout.
StatusOr<mlir::Operation*>
ReplicatedOpSPMDExpander::ReplicatedRelayoutOperandsAndOutputs(
    mlir::Operation* op, const std::vector<Layout>& operand_layouts,
    const std::vector<Layout>& output_layouts) {
  TF_ASSIGN_OR_RETURN(const auto mesh, ExtractDeviceMeshEnclosingCluster(op));

  // Relayout operands
  for (auto i = 0; i < operand_layouts.size(); ++i) {
    Layout new_layout =
        Layout::ReplicatedOnMesh(mesh, ValueRank(op->getOperand(i)));
    TF_ASSIGN_OR_RETURN(
        const auto new_operand,
        EmitRelayout(op->getOperand(i), operand_layouts[i], new_layout));
    op->setOperand(i, new_operand);
  }
  // Expand to local shape
  op = InferSPMDExpandedLocalShape(op);

  llvm::SmallPtrSet<mlir::Operation*, 4> newly_created_ops;
  llvm::SmallVector<mlir::Value, 4> generated_outputs;
  llvm::SmallVector<mlir::Type, 4> generated_types;

  // Track the op that comes last after splitting.
  mlir::Operation* last_op_after_splitting = op;

  // Relayout outputs
  for (auto i = 0; i < output_layouts.size(); ++i) {
    Layout new_layout =
        Layout::ReplicatedOnMesh(mesh, ValueRank(op->getResult(i)));
    TF_ASSIGN_OR_RETURN(auto new_output,
                        EmitRelayout(op->getOpResult(i), new_layout,
                                     output_layouts[i], &newly_created_ops));
    generated_outputs.emplace_back(new_output);
    generated_types.emplace_back(new_output.getType());
    if (last_op_after_splitting->isBeforeInBlock(new_output.getDefiningOp())) {
      last_op_after_splitting = new_output.getDefiningOp();
    }
  }
  mlir::OpBuilder builder(op);
  builder.setInsertionPointAfter(last_op_after_splitting);

  // Tie all outputs together with identity_n
  auto identity_op = builder.create<mlir::TF::IdentityNOp>(
      op->getLoc(), generated_types, generated_outputs);
  newly_created_ops.insert(identity_op);
  for (int i = 0; i < output_layouts.size(); ++i) {
    op->getOpResult(i).replaceAllUsesExcept(identity_op.getResult(i),
                                            newly_created_ops);
  }

  return identity_op.getOperation();
}

StatusOr<mlir::Operation*> ReplicatedOpSPMDExpander::ExpandOp(
    mlir::Operation* op) {
  TF_ASSIGN_OR_RETURN(const auto output_layouts,
                      ExtractRequiredLayoutFromOp(op));
  TF_ASSIGN_OR_RETURN(const auto operand_layouts,
                      ExtractRequiredLayoutFromOperands(op));
  if (relayout_when_sharded_)
    return ReplicatedRelayoutOperandsAndOutputs(op, operand_layouts,
                                                output_layouts);
  if (!AllReplicated(output_layouts) || !AllReplicated(operand_layouts)) {
    return errors::InvalidArgument(
        llvm::formatv("Expecting {0} to have input and output layouts to be "
                      "fully replicated but was not. ",
                      OpName(op))
            .str());
  }
  return op;
}

// Always return a set of replicated layouts.
StatusOr<llvm::DenseMap<int, Layout>>
ReplicatedOpSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  TF_ASSIGN_OR_RETURN(const auto mesh, ExtractDeviceMeshEnclosingCluster(op));

  llvm::DenseMap<int, Layout> output_layouts(op->getNumResults());
  for (int i = 0; i < op->getNumResults(); ++i) {
    output_layouts[i] =
        Layout::ReplicatedOnMesh(mesh, ValueRank(op->getResult(i)));
  }
  return output_layouts;
}

// Always return a set of replicated layouts.
StatusOr<llvm::DenseMap<int, Layout>>
ReplicatedOpSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  TF_ASSIGN_OR_RETURN(const auto mesh, ExtractDeviceMeshEnclosingCluster(op));

  llvm::DenseMap<int, Layout> input_layouts(op->getNumOperands());
  for (int i = 0; i < op->getNumOperands(); ++i) {
    input_layouts[i] =
        Layout::ReplicatedOnMesh(mesh, ValueRank(op->getOperand(i)));
  }
  return input_layouts;
}

}  // namespace dtensor
}  // namespace tensorflow
