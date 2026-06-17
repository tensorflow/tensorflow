/* Copyright 2025 The OpenXLA Authors.

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

#include <cassert>
#include <cstdint>
#include <memory>
#include <utility>

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // IWYU pragma: keep
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/backends/cpu/codegen/tiled/transforms/passes.h"

namespace xla::cpu {

#define GEN_PASS_DECL_MEMREFCOPYTOLOOPSPASS
#define GEN_PASS_DEF_MEMREFCOPYTOLOOPSPASS
#include "xla/backends/cpu/codegen/tiled/transforms/passes.h.inc"

namespace {

// Super simple lowering of memref.copies that would otherwise be lowered to a
// external call by the default memref lowering.
// TODO(willfroom): look into vectorizing these.
struct LowerMemRefCopyPattern
    : public mlir::OpRewritePattern<mlir::memref::CopyOp> {
  using mlir::OpRewritePattern<mlir::memref::CopyOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::memref::CopyOp op, mlir::PatternRewriter& rewriter) const override {
    mlir::Location loc = op.getLoc();
    auto source =
        mlir::cast<mlir::TypedValue<mlir::MemRefType>>(op.getSource());
    auto dest = mlir::cast<mlir::TypedValue<mlir::MemRefType>>(op.getTarget());

    mlir::MemRefType src_type = source.getType();
    mlir::MemRefType dest_type = dest.getType();

    // These will be lowered by the default memref -> llvm pipeline to a memcpy
    // intrinsic.
    // TODO(willfroom): We should update the default memref lowering to allow
    // the same layout rather than requiring identity.
    if (mlir::memref::isStaticShapeAndContiguousRowMajor(src_type) &&
        mlir::memref::isStaticShapeAndContiguousRowMajor(dest_type)) {
      return rewriter.notifyMatchFailure(
          op, "memref.copy will be lowered to a memcpy intrinsic");
    }

    int64_t rank = src_type.getRank();

    llvm::SmallVector<mlir::Value> lbs, ubs, steps;
    lbs.reserve(rank);
    ubs.reserve(rank);
    steps.reserve(rank);

    mlir::Value c1 = mlir::arith::ConstantIndexOp::create(rewriter, loc, 1);
    mlir::Value c0 = mlir::arith::ConstantIndexOp::create(rewriter, loc, 0);

    for (int64_t idx = 0; idx < rank; ++idx) {
      lbs.push_back(c0);
      steps.push_back(c1);

      // Source & destination must have the same shape as defined by the copy op
      // spec so we can just extract it from the source without checking the
      // destination.
      if (src_type.isDynamicDim(idx)) {
        ubs.push_back(mlir::memref::DimOp::create(rewriter, loc, source, idx));
      } else {
        ubs.push_back(mlir::arith::ConstantIndexOp::create(
            rewriter, loc, src_type.getDimSize(idx)));
      }
    }

    // TODO(willfroom): We should ensure that the loop order is major-to-minor.
    mlir::scf::buildLoopNest(
        rewriter, loc, lbs, ubs, steps,
        [source, dest](mlir::OpBuilder& builder, mlir::Location loc,
                       mlir::ValueRange ivs) {
          mlir::Value element =
              mlir::memref::LoadOp::create(builder, loc, source, ivs);
          mlir::memref::StoreOp::create(builder, loc, element, dest, ivs);
        });

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class MemrefCopyToLoopsPass
    : public impl::MemrefCopyToLoopsPassBase<MemrefCopyToLoopsPass> {
 public:
  using MemrefCopyToLoopsPassBase::MemrefCopyToLoopsPassBase;

  void runOnOperation() override {
    mlir::MLIRContext* context = &getContext();
    mlir::RewritePatternSet patterns(context);
    patterns.add<LowerMemRefCopyPattern>(context);
    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> CreateMemrefCopyToLoopsPass() {
  return std::make_unique<MemrefCopyToLoopsPass>();
}

}  // namespace xla::cpu
