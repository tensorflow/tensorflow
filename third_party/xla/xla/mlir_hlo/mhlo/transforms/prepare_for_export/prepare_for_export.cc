/* Copyright 2021 The OpenXLA Authors.

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

#include <cassert>
#include <complex>
#include <cstdint>

#include "llvm/ADT/STLExtras.h"
#include "mhlo/IR/hlo_ops.h"
#include "mhlo/transforms/passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/RegionUtils.h"

#define DEBUG_TYPE "xla-prepare-for-export"

namespace mlir {
namespace mhlo {

constexpr char kShardingAttr[] = "mhlo.sharding";

#define GEN_PASS_DEF_PREPAREFOREXPORTPASS
#include "mhlo/transforms/mhlo_passes.h.inc"

namespace {
// Prepare module for export to XLA HLO.
struct PrepareForExportPass
    : public impl::PrepareForExportPassBase<PrepareForExportPass> {
  void runOnOperation() override;
};

}  // end namespace

// Materializes some splat before export because it may be more efficient in
// HLOInstruction.
static void prepareConstantOp(Operation *op, SplatElementsAttr attr) {
  // Arbitrarily chosen "small" number. This could be chosen based on the proto
  // size too.
  if (attr.getNumElements() < 32) return;
  ShapedType returnType = mlir::cast<ShapedType>(op->getResultTypes().front());
  ImplicitLocOpBuilder b(op->getLoc(), op);
  ConstantOp cst;
  if (auto complexTy =
          mlir::dyn_cast<ComplexType>(returnType.getElementType())) {
    auto tensorType = RankedTensorType::get({}, returnType.getElementType());
    assert(mlir::isa<FloatType>(complexTy.getElementType()) &&
           "unexpected int complex in MHLO");
    auto complexVal = attr.getSplatValue<std::complex<APFloat>>();
    cst = b.create<ConstantOp>(DenseElementsAttr::get(tensorType, complexVal));
  } else {
    cst = b.create<ConstantOp>(attr.getSplatValue<Attribute>());
  }
  auto broadcast =
      b.create<BroadcastInDimOp>(returnType, cst, b.getI64TensorAttr({}));
  if (auto sharding = op->getAttrOfType<mlir::StringAttr>(kShardingAttr)) {
    // The added broadcast inherits the kShardingAttr from op.
    broadcast->setAttr(kShardingAttr, sharding);
  }
  op->replaceAllUsesWith(broadcast);
  op->erase();
}

static void prepareBroadcastInDim(BroadcastInDimOp bcast) {
  DenseIntElementsAttr dims = bcast.getBroadcastDimensions();
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
      bcast.getLoc(), bcast.getOperand(),
      DenseIntElementsAttr::get(dims.getType(), transposedDim)));
  // Now reuse the original broadcast_dimensions and sort it.
  transposedDim.assign(rawDims.begin(), rawDims.end());
  llvm::sort(transposedDim);
  bcast.setBroadcastDimensionsAttr(
      DenseIntElementsAttr::get(dims.getType(), transposedDim));
}

// Make implicitly captured constant explicit before exporting
static void prepareExplicitCapturedConstants(Operation *op) {
  for (Region &region : op->getRegions()) {
    assert(region.getBlocks().size() == 1 &&
           "Only OPs with single block regions are allowed");
    llvm::SetVector<Value> implicitInputs;
    // Get implicit inputs, i.e. those are used in the region
    // but defined outside
    getUsedValuesDefinedAbove(region, implicitInputs);
    Block &block = region.getBlocks().front();
    OpBuilder builder(&block.front());
    for (Value input : implicitInputs) {
      // If the captured value is defined by a constant OP,
      // Create a clone constant OP within a block to make
      // it explicit and replace uses within the block
      Operation *definingOp = input.getDefiningOp();
      mlir::DenseElementsAttr attr;
      if (matchPattern(input, m_Constant(&attr))) {
        Operation *clonedOp = builder.clone(*definingOp);
        // Find which uses belong to the block and replace
        // with the cloned/explicit one
        input.replaceUsesWithIf(
            clonedOp->getResult(0), [&block](OpOperand &use) {
              return block.getParentOp()->isProperAncestor(use.getOwner());
            });
      }
    }
  }
}

void PrepareForExportPass::runOnOperation() {
  getOperation().walk([&](Operation *op) {
    mlir::SplatElementsAttr attr;
    if (matchPattern(op, m_Constant(&attr))) return prepareConstantOp(op, attr);

    if (auto bcastOp = dyn_cast<BroadcastInDimOp>(op))
      return prepareBroadcastInDim(bcastOp);
    // IfOp, CaseOp, WhileOp are already being handled during
    // mhlo --> hlo translation. MapOp soon be deprecated.
    if (mlir::isa<ReduceOp, AllReduceOp, ReduceScatterOp, ReduceWindowOp,
                  ScatterOp, SelectAndScatterOp, SortOp>(op))
      return prepareExplicitCapturedConstants(op);
  });
}

}  // end namespace mhlo
}  // end namespace mlir
