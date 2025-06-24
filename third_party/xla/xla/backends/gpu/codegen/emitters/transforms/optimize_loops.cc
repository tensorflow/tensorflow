/* Copyright 2024 The OpenXLA Authors.

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
#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/backends/gpu/codegen/emitters/ir/xla_gpu_ops.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/service/gpu/gpu_fusible.h"

namespace xla {
namespace gpu {

#define GEN_PASS_DEF_OPTIMIZELOOPSPASS
#include "xla/backends/gpu/codegen/emitters/transforms/passes.h.inc"

namespace {

int GetUnrollingFactor(mlir::scf::ForOp op) {
  // We only unroll loops with a step of 1 and a lower bound of 0. That's the
  // only type we generate.
  if (auto step = op.getConstantStep(); !step || step->getSExtValue() != 1) {
    return 1;
  }
  llvm::APInt lb, ub;
  if (!mlir::matchPattern(op.getLowerBound(), mlir::m_ConstantInt(&lb)) ||
      !mlir::matchPattern(op.getUpperBound(), mlir::m_ConstantInt(&ub))) {
    return 1;
  }
  if (lb.getSExtValue() != 0) {
    return 1;
  }

  int64_t trip_count = ub.getSExtValue();
  constexpr int kMaxSize = 400;  // Chosen empirically.

  // Get a rough estimate of the size of the loop body.
  int64_t size = 0;
  bool can_unroll = true;
  op.getBodyRegion().walk([&](mlir::Operation* op) {
    if (mlir::isa<mlir::func::CallOp, mlir::scf::ForOp>(op)) {
      can_unroll = false;
      return;
    }

    int64_t this_size = 1;
    if (mlir::isa<mlir::math::MathDialect>(op->getDialect())) {
      // Integer instructions in math are ok, but many float ops lower to lots
      // of instructions.
      if (!op->getResultTypes().front().isIntOrIndex()) {
        namespace mm = mlir::math;
        // We err on the side of not unrolling, so we maintain a list of ops
        // known to be cheap.
        if (!mlir::isa<mm::AbsFOp, mm::CeilOp, mm::CopySignOp, mm::FloorOp,
                       mm::FmaOp, mm::RoundEvenOp, mm::RoundOp, mm::RsqrtOp,
                       mm::SqrtOp, mm::TruncOp>(op)) {
          this_size = 20;  // Rough estimate.
        }
      }
    }

    if (!op->getResultTypes().empty()) {
      if (auto vector_ty =
              mlir::dyn_cast<mlir::VectorType>(op->getResultTypes().front())) {
        this_size *= vector_ty.getNumElements();
      }
    }

    size += this_size;
  });

  if (!can_unroll) {
    return 1;
  }

  // Always unroll if the trip count is smaller than the max unroll factor,
  // because it's very likely that the loop was meant to be unrolled.
  if (trip_count <= MaxUnrollFactor()) {
    return trip_count;
  }

  int factor = std::min(trip_count, kMaxSize / size);
  while (factor > 1 && trip_count % factor) {
    --factor;
  }
  return factor;
}

struct UnrollLoops : mlir::OpRewritePattern<mlir::scf::ForOp> {
  using mlir::OpRewritePattern<mlir::scf::ForOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::scf::ForOp op, mlir::PatternRewriter& rewriter) const override {
    if (int factor = GetUnrollingFactor(op); factor > 1) {
      return mlir::loopUnrollByFactor(op, factor);
    }
    return rewriter.notifyMatchFailure(op, "loop can't be unrolled");
  }
};

class OptimizeLoopsPass
    : public impl::OptimizeLoopsPassBase<OptimizeLoopsPass> {
 public:
  void runOnOperation() override {
    // First unroll loops. If unrolling is possible, we prefer it.
    mlir::RewritePatternSet unroll_patterns(&getContext());
    unroll_patterns.add<UnrollLoops>(&getContext());
    if (mlir::failed(mlir::applyPatternsGreedily(getOperation(),
                                                 std::move(unroll_patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateOptimizeLoopsPass() {
  return std::make_unique<OptimizeLoopsPass>();
}

}  // namespace gpu
}  // namespace xla
