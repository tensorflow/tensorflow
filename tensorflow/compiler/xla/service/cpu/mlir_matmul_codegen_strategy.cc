/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/cpu/mlir_matmul_codegen_strategy.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/SliceAnalysis.h"  // from @llvm-project
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"  // from @llvm-project
#include "mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"  // from @llvm-project
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"  // from @llvm-project
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"  // from @llvm-project
#include "mlir/Dialect/Linalg/Utils/Utils.h"  // from @llvm-project
#include "mlir/Dialect/SCF/SCF.h"  // from @llvm-project
#include "mlir/Dialect/SCF/Utils.h"  // from @llvm-project
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/Dialect/Vector/EDSC/Intrinsics.h"  // from @llvm-project
#include "mlir/Dialect/Vector/VectorOps.h"  // from @llvm-project
#include "mlir/Dialect/Vector/VectorTransforms.h"  // from @llvm-project
#include "mlir/IR/AffineExpr.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BlockAndValueMapping.h"  // from @llvm-project
#include "mlir/IR/Dominance.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/StandardTypes.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/LoopUtils.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project

// TODO(kramerb): Remove this once strategy is in mlir core.

using namespace mlir;          // NOLINT
using namespace mlir::linalg;  // NOLINT

#define DEBUG_TYPE "matmul-codegen-strategy"

namespace xla {
namespace cpu {
namespace mlir_strategy {

//===----------------------------------------------------------------------===//
// TODO: Cleanup and upstream these to go into core. Please ignore for now !
//===----------------------------------------------------------------------===//
static void hoistRedundantCopies(FuncOp func) {
  bool changed = true;
  while (changed) {
    changed = false;
    func.walk([&](linalg::FillOp op) {
      auto loop = op.getParentOfType<scf::ForOp>();
      if (!loop) return;

      for (auto operand : op.getOperands())
        if (!loop.isDefinedOutsideOfLoop(operand)) return;

      // Hoist fill before.
      op.getOperation()->moveBefore(loop);
      changed = true;
    });

    func.walk([&](linalg::CopyOp op) {
      auto loop = op.getParentOfType<scf::ForOp>();
      if (!loop) return;

      for (auto operand : op.getOperands())
        if (!loop.isDefinedOutsideOfLoop(operand)) return;

      Value sourceView = op.getInput(0);
      while (auto subViewOp = sourceView.getDefiningOp<SubViewOp>())
        sourceView = subViewOp.getViewSource();

      // Source traces back to a block argument.
      if (sourceView.isa<BlockArgument>()) {
        op.getOperation()->moveBefore(loop);
      } else {
        assert(sourceView.getDefiningOp<ViewOp>() ||
               sourceView.getDefiningOp<AllocOp>() ||
               sourceView.getDefiningOp<AllocaOp>());
        op.getOperation()->moveAfter(loop);
      }
      changed = true;
    });
  }
}

/// Substitute scf.for = %lb to %ub step %step by an AffineExpr expressing:
///   `%lb + %step * new_dim` where
/// 1. the AffineExpr for %lb is either an AffineConstantExpr or an
/// AffineDimExpr depending on whether the value is constant or not.
/// 2. the AffineExpr for %step is either an AffineConstantExpr or an
/// AffineSymbolExpr depending on whether the value is constant or not.
///
static void substitute(scf::ForOp forOp, SmallVectorImpl<AffineExpr> &exprs,
                       SmallVectorImpl<Value> &dims,
                       SmallVectorImpl<Value> &symbols) {
  MLIRContext *ctx = forOp.getContext();
  auto lbConstant = forOp.lowerBound().getDefiningOp<ConstantIndexOp>();
  AffineExpr lb = lbConstant ? getAffineConstantExpr(lbConstant.getValue(), ctx)
                             : getAffineDimExpr(dims.size(), ctx);

  auto stepConstant = forOp.step().getDefiningOp<ConstantIndexOp>();
  AffineExpr step = stepConstant
                        ? getAffineConstantExpr(stepConstant.getValue(), ctx)
                        : getAffineSymbolExpr(symbols.size(), ctx);

  if (!lbConstant) dims.push_back(forOp.lowerBound());
  if (!stepConstant) symbols.push_back(forOp.step());
  exprs.push_back(lb + step * getAffineDimExpr(dims.size(), ctx));

  auto ubConstant = forOp.upperBound().getDefiningOp<ConstantIndexOp>();
  AffineExpr ub = ubConstant ? getAffineConstantExpr(ubConstant.getValue(), ctx)
                             : getAffineDimExpr(dims.size(), ctx);
  if (!ubConstant) dims.push_back(forOp.upperBound());
  exprs.push_back(ub);

  dims.push_back(forOp.getInductionVar());
}

/// Traverse the .
static void substitute(AffineMinOp minOp, SmallVectorImpl<AffineExpr> &exprs,
                       SmallVectorImpl<Value> &dims,
                       SmallVectorImpl<Value> &symbols) {
  MLIRContext *ctx = minOp.getContext();
  for (Value v : minOp.getDimOperands()) {
    if (auto forOp = scf::getForInductionVarOwner(v)) {
      substitute(forOp, exprs, dims, symbols);
      continue;
    }
    if (auto parentMinOp = v.getDefiningOp<AffineMinOp>()) {
      substitute(parentMinOp, exprs, dims, symbols);
      continue;
    }
    exprs.push_back(getAffineDimExpr(dims.size(), ctx));
    dims.push_back(v);
  }
}

/// Perform folding of chains of AffineMinOp.
struct AffineMinCanonicalizationPattern : public OpRewritePattern<AffineMinOp> {
  using OpRewritePattern<AffineMinOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineMinOp minOp,
                                PatternRewriter &rewriter) const override;
};

LogicalResult AffineMinCanonicalizationPattern::matchAndRewrite(
    AffineMinOp minOp, PatternRewriter &rewriter) const {
  LLVM_DEBUG(llvm::dbgs() << "\nCanonicalize AffineMin: "
                          << *minOp.getOperation() << "\n");

  int64_t min = std::numeric_limits<int64_t>::max();
  for (auto e : minOp.map().getResults())
    if (auto cstExpr = e.dyn_cast<AffineConstantExpr>())
      min = std::min(min, cstExpr.getValue());
  if (min == std::numeric_limits<int64_t>::max()) return failure();

  SmallVector<AffineExpr, 4> exprs;
  SmallVector<Value, 4> dims, symbols;
  substitute(minOp, exprs, dims, symbols);

  SmallVector<Value, 4> operands = dims;
  operands.append(symbols.begin(), symbols.end());

  MLIRContext *ctx = minOp.getContext();
  auto map = AffineMap::get(dims.size(), symbols.size(), exprs, ctx);
  LLVM_DEBUG(llvm::dbgs() << "Substitution map: " << map << "\n");

  SmallVector<AffineExpr, 4> modExprs;
  for (unsigned idx = 0, e = map.getNumResults(); idx < e; ++idx)
    modExprs.push_back(getAffineDimExpr(idx, ctx) % min);
  map = AffineMap::get(map.getNumResults(), 0, modExprs, ctx).compose(map);
  canonicalizeMapAndOperands(&map, &operands);
  map = simplifyAffineMap(map);

  LLVM_DEBUG(llvm::dbgs() << "Post mod: " << map << "\n";
             llvm::interleaveComma(operands, llvm::dbgs()));

  if (!llvm::all_of(map.getResults(), [](AffineExpr e) {
        if (auto cst = e.dyn_cast<AffineConstantExpr>())
          return cst.getValue() == 0;
        return false;
      }))
    return failure();

  rewriter.replaceOpWithNewOp<ConstantIndexOp>(minOp, min);
  return success();
}
//===----------------------------------------------------------------------===//
// END TODO
//===----------------------------------------------------------------------===//

void MatmulCodegenStrategy::transform(FuncOp func) const {
  MLIRContext *context = func.getContext();
  // Emplace patterns one at a time while also maintaining a simple chained
  // state transition.
  unsigned stepCount = 0;
  SmallVector<OwningRewritePatternList, 4> stage1Patterns;
  auto zeroState = Identifier::get(std::to_string(stepCount), context);
  auto currentState = zeroState;
  for (auto &t : transformation_sequence) {
    auto nextState = Identifier::get(std::to_string(++stepCount), context);
    auto marker = (currentState == zeroState)
                      ? linalg::LinalgMarker({}, nextState)
                      : linalg::LinalgMarker(currentState, nextState);
    stage1Patterns.emplace_back(t->buildRewritePatterns(context, marker));
    currentState = nextState;
  }

  OwningRewritePatternList stage2Patterns =
      linalg::getLinalgTilingCanonicalizationPatterns(context);
  stage2Patterns.insert<AffineMinCanonicalizationPattern>(context);

  auto stage3Transforms = [](Operation *op) {
    // Some of these may be too aggressive as a stage 3 that is applied on each
    // stage 1 application and may have to be split out to post staged patterns
    // application (in which case they could just be passes, TBD).
    PassManager pm(op->getContext());
    pm.addPass(createLoopInvariantCodeMotionPass());
    if (failed(pm.run(op->getParentOfType<ModuleOp>())))
      llvm_unreachable("Unexpected failure in cleanup pass pipeline.");
    promoteSingleIterationLoops(cast<FuncOp>(op));
    hoistViewAllocOps(cast<FuncOp>(op));
    hoistRedundantVectorTransfers(cast<FuncOp>(op));
    hoistRedundantCopies(cast<FuncOp>(op));
    return success();
  };
  linalg::applyStagedPatterns(func, stage1Patterns, stage2Patterns,
                              stage3Transforms);

  //===--------------------------------------------------------------------===//
  // Post staged patterns transforms
  //===--------------------------------------------------------------------===//
  // Programmatic controlled lowering of vector.contract only.
  OwningRewritePatternList vectorContractLoweringPatterns;
  vectorContractLoweringPatterns
      .insert<ContractionOpToOuterProductOpLowering,
              ContractionOpToMatmulOpLowering, ContractionOpLowering>(
          vector_transforms_options, context);
  applyPatternsAndFoldGreedily(func, vectorContractLoweringPatterns);

  // Programmatic controlled lowering of vector.transfer only.
  OwningRewritePatternList vectorToLoopsPatterns;
  populateVectorToSCFConversionPatterns(vectorToLoopsPatterns, context,
                                        vector_to_scf_options);
  applyPatternsAndFoldGreedily(func, vectorToLoopsPatterns);
}

}  // namespace mlir_strategy
}  // namespace cpu
}  // namespace xla
