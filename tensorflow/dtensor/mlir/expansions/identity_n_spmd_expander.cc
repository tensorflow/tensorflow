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

#include "tensorflow/dtensor/mlir/expansions/identity_n_spmd_expander.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
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
#include "tensorflow/dtensor/mlir/shape_utils.h"

namespace tensorflow {
namespace dtensor {

StatusOr<mlir::Operation*> IdentityNSPMDExpander::ExpandOp(
    mlir::Operation* op) {
  TF_ASSIGN_OR_RETURN(auto layouts, ExtractLayoutFromOp(op));

  llvm::SmallPtrSet<mlir::Operation*, 4> newly_created_ops;
  llvm::SmallVector<mlir::Value, 4> generated_outputs;
  llvm::SmallVector<mlir::Type, 4> generated_types;

  mlir::OpBuilder builder(op);
  // Track the op that comes last after splitting.
  mlir::Operation* last_op_after_splitting = op;
  for (int i = 0; i < layouts.size(); ++i) {
    auto output_layout = layouts[i];
    if (!output_layout)
      return errors::InvalidArgument(
          "layout of (", i,
          "-th output of IdentityNOp must be known before SPMD expansion.");

    TF_ASSIGN_OR_RETURN(auto operand_layout,
                        ExtractLayoutFromOperand(op->getOperand(i)));
    if (!operand_layout)
      return errors::InvalidArgument(
          "layout of (", i,
          "-th input of IdentityNOp must be known before SPMD expansion.");

    TF_ASSIGN_OR_RETURN(const mlir::Value output,
                        EmitRelayout(op->getOperand(i), *operand_layout,
                                     *output_layout, &newly_created_ops));
    generated_outputs.emplace_back(output);
    generated_types.emplace_back(output.getType());
    // InsertionPoint has to come after all newly created Ops.
    if (last_op_after_splitting->isBeforeInBlock(output.getDefiningOp())) {
      last_op_after_splitting = output.getDefiningOp();
    }
  }

  builder.setInsertionPointAfter(last_op_after_splitting);
  auto identity_op = builder.create<mlir::TF::IdentityNOp>(
      op->getLoc(), generated_types, generated_outputs);

  for (int i = 0; i < layouts.size(); ++i)
    op->getOpResult(i).replaceAllUsesExcept(identity_op.getResult(i),
                                            newly_created_ops);

  op->erase();

  return InferSPMDExpandedLocalShape(identity_op.getOperation());
}

StatusOr<llvm::DenseMap<int, Layout>>
IdentityNSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  return input_layouts;
}

StatusOr<llvm::DenseMap<int, Layout>>
IdentityNSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  return output_layouts;
}

}  // namespace dtensor
}  // namespace tensorflow
