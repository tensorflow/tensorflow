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

#include <cstdint>
#include <memory>

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
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
  void runOnOperation() override;
};

}  // end namespace

// Materializes some splat before export because it may be more efficient in
// HLOInstruction.
void prepareConstantOp(Operation *op, SplatElementsAttr attr) {
  // Arbitrarialy chosen "small" number. This could be chosen based on the
  // proto size too.
  if (attr.getNumElements() < 32) return;
  ShapedType return_type = op->getResultTypes().front().cast<ShapedType>();
  ImplicitLocOpBuilder b(op->getLoc(), op);
  ConstantOp cst;
  if (auto complexTy = return_type.getElementType().dyn_cast<ComplexType>()) {
    auto tensorType = RankedTensorType::get({}, return_type.getElementType());
    assert(complexTy.getElementType().isa<FloatType>() &&
           "unexpected int complex in MHLO");
    auto complexVal = attr.getSplatValue<std::complex<APFloat>>();
    cst = b.create<ConstantOp>(DenseElementsAttr::get(tensorType, complexVal));
  } else {
    cst = b.create<ConstantOp>(attr.getSplatValue<Attribute>());
  }
  auto broadcast =
      b.create<BroadcastInDimOp>(return_type, cst, b.getI64TensorAttr({}));
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
    for (OpOperand &operand : llvm::make_early_inc_range(input.getUses())) {
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

void prepareBroadcastInDim(BroadcastInDimOp bcast) {
  DenseIntElementsAttr dims = bcast.broadcast_dimensions();
  // If dimensions aren't sorted, there is a transpose fused into the op, which
  // XLA Builder does not support, we unfuse here.
  if (llvm::is_sorted(dims.getValues<int64_t>())) return;

  // We need to compute a permutation that sorts the dimension before the
  // broadcast.
  // If the dims are [2, 4, 1], we create an array of indices: [0, 1, 2] and we
  // sort it using the values of the first array to produce [2, 0, 1] which
  // gives us the operand for the transpose.
  SmallVector<int64_t> transposedDim =
      to_vector(llvm::seq<int64_t>(0, dims.size()));
  auto rawDims = dims.getValues<int64_t>();
  llvm::sort(transposedDim, [&](int64_t lhs, int64_t rhs) {
    return rawDims[lhs] < rawDims[rhs];
  });
  OpBuilder builder(bcast);
  bcast.setOperand(builder.create<TransposeOp>(
      bcast.getLoc(), bcast.operand(),
      DenseIntElementsAttr::get(dims.getType(), transposedDim)));
  // Now reuse the original broadcast_dimensions and sort it.
  transposedDim.assign(rawDims.begin(), rawDims.end());
  llvm::sort(transposedDim);
  bcast.broadcast_dimensionsAttr(
      DenseIntElementsAttr::get(dims.getType(), transposedDim));
}

void PrepareForExportPass::runOnOperation() {
  getOperation().walk([&](Operation *op) {
    mlir::SplatElementsAttr attr;
    if (matchPattern(op, m_Constant(&attr))) return prepareConstantOp(op, attr);

    if (auto whileOp = dyn_cast<WhileOp>(op)) return prepareWhileOp(whileOp);
    if (auto bcastOp = dyn_cast<BroadcastInDimOp>(op))
      return prepareBroadcastInDim(bcastOp);
  });
}

std::unique_ptr<OperationPass<func::FuncOp>> CreatePrepareForExport() {
  return std::make_unique<PrepareForExportPass>();
}

}  // end namespace mhlo
}  // end namespace mlir
