/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

// This file implements logic for some optimizations to reduce size on export.

#include <memory>

#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Transforms/RegionUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/mlir/xla/transforms/xla_passes.h"
#include "tensorflow/compiler/mlir/xla/transforms/xla_passes_detail.h"

#define DEBUG_TYPE "xla-prepare-for-export"

namespace mlir {
namespace mhlo {
namespace {

// Prepare module for export to XLA HLO.
struct PrepareForExportPass
    : public PrepareForExportPassBase<PrepareForExportPass> {
  void runOnFunction() override;
};

}  // end namespace

// Materializes some splat before export because it may be more efficient in
// HLOInstruction.
void prepareConstantOp(Operation *op, mlir::SplatElementsAttr attr) {
  // Only consider int or floats for now.
  if (!attr.getType().getElementType().isIntOrFloat()) return;
  // Arbitrarialy chosen "small" number. This could be chosen based on the
  // proto size too.
  if (attr.getNumElements() < 32) return;
  ShapedType return_type = op->getResultTypes().front().cast<ShapedType>();
  ImplicitLocOpBuilder b(op->getLoc(), op);
  auto cst = b.create<::mlir::mhlo::ConstOp>(attr.getSplatValue<Attribute>());
  auto broadcast = b.create<::mlir::mhlo::BroadcastInDimOp>(
      return_type, cst, b.getI64TensorAttr({}));
  op->replaceAllUsesWith(broadcast);
  op->erase();
}

// Ensure that there aren't any implicit capture before exporting.
void prepareWhileOp(WhileOp while_op) {
  llvm::SetVector<Value> implicit_inputs;
  getUsedValuesDefinedAbove(while_op->getRegions(), implicit_inputs);
  // Each captured value has to be passed as operand to the while, become then
  // an operand to the condition region and the body region, and an extra
  // operand to the return op in the body. It also becomes an extra result for
  // the while operation, even if it is unused.
  // We'll process the captured values one at a time and patch the body and
  // condition regions as we go, but we'll accumulate the new operands and
  // result type and recreate a new while op to replace the existing one at the
  // end.
  SmallVector<Type> returned_types(while_op->getResultTypes().begin(),
                                   while_op->getResultTypes().end());
  SmallVector<Value> operands(while_op->getOperands().begin(),
                              while_op->getOperands().end());
  Region &cond_region = while_op.cond();
  Region &body_region = while_op.body();

  for (Value input : implicit_inputs) {
    returned_types.push_back(input.getType());
    operands.push_back(input);

    Value cond_arg =
        cond_region.front().addArgument(input.getType(), input.getLoc());
    Value body_arg =
        body_region.front().addArgument(input.getType(), input.getLoc());
    for (OpOperand &operand : input.getUses()) {
      if (cond_region.isAncestor(operand.getOwner()->getParentRegion()))
        operand.set(cond_arg);
      else if (body_region.isAncestor(operand.getOwner()->getParentRegion()))
        operand.set(body_arg);
    }
    auto return_op = cast<mhlo::ReturnOp>(body_region.front().back());
    return_op->insertOperands(return_op->getNumOperands(), body_arg);
  }
  OpBuilder builder(while_op);
  auto new_while_op = builder.create<mhlo::WhileOp>(while_op.getLoc(),
                                                    returned_types, operands);
  new_while_op.cond().getBlocks().clear();
  new_while_op.cond().takeBody(while_op.cond());
  new_while_op.body().getBlocks().clear();
  new_while_op.body().takeBody(while_op.body());
  for (auto zipped_results :
       llvm::zip_first(while_op.getResults(), new_while_op.getResults()))
    std::get<0>(zipped_results).replaceAllUsesWith(std::get<1>(zipped_results));
  while_op->erase();
}

void PrepareForExportPass::runOnFunction() {
  getFunction().walk([&](Operation *op) {
    mlir::SplatElementsAttr attr;
    if (matchPattern(op, m_Constant(&attr))) return prepareConstantOp(op, attr);

    if (auto while_op = dyn_cast<WhileOp>(op)) return prepareWhileOp(while_op);
  });
}

std::unique_ptr<OperationPass<FuncOp>> CreatePrepareForExport() {
  return std::make_unique<PrepareForExportPass>();
}

}  // end namespace mhlo
}  // end namespace mlir
