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
#include <memory>
#include <optional>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Math/IR/Math.h"  // from @llvm-project
#include "mlir/Dialect/SCF/IR/SCF.h"  // from @llvm-project
#include "mlir/Dialect/SCF/Utils/Utils.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/Dialect/Vector/IR/VectorOps.h"  // from @llvm-project
#include "mlir/IR/AffineExpr.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/ValueRange.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "xla/service/gpu/fusions/mlir/ir/xla_gpu_ops.h"

namespace xla {
namespace gpu {

#define GEN_PASS_DEF_OPTIMIZELOOPSPASS
#include "xla/service/gpu/fusions/mlir/passes.h.inc"

namespace {

mlir::Value GetSource(mlir::vector::TransferReadOp op) {
  return op.getSource();
}

bool DoIndicesDependOnInductionVar(mlir::ValueRange indices,
                                   mlir::scf::ForOp loop) {
  // We assume LICM ran, so we can just check if any index is defined in the
  // loop.
  return absl::c_any_of(indices, [&](mlir::Value v) {
    return v.getParentRegion() == &loop.getBodyRegion();
  });
}

bool CanReplaceInductionVar(mlir::ValueRange indices) {
  return absl::c_all_of(indices, [&](mlir::Value v) {
    if (mlir::isa<mlir::BlockArgument>(v)) {
      return true;
    }
    auto* op = v.getDefiningOp();
    return op &&
           mlir::isa<mlir::arith::ConstantOp, ApplyIndexingOp,
                     mlir::arith::MaxSIOp, mlir::arith::MinSIOp,
                     mlir::arith::IndexCastOp, mlir::arith::IndexCastUIOp>(
               op) &&
           CanReplaceInductionVar(op->getOperands());
  });
}

llvm::SmallVector<mlir::Value> ReplaceInductionVar(
    mlir::Value induction_var, mlir::Value replacement,
    llvm::SmallVector<mlir::Value> indices,
    mlir::ImplicitLocOpBuilder& builder) {
  for (mlir::Value& index : indices) {
    if (mlir::isa<mlir::BlockArgument>(index)) {
      if (index == induction_var) {
        index = replacement;
      }
      continue;
    }

    auto* op = index.getDefiningOp();
    CHECK(op) << "Did CanReplaceInductionVar() fail?";
    if (mlir::isa<mlir::arith::ConstantOp>(op)) {
      continue;
    }

    CHECK(
        (mlir::isa<ApplyIndexingOp, mlir::arith::MaxSIOp, mlir::arith::MinSIOp,
                   mlir::arith::IndexCastOp, mlir::arith::IndexCastUIOp>(op)))
        << "Did CanReplaceInductionVar() fail?";
    auto replaced_args = ReplaceInductionVar(induction_var, replacement,
                                             op->getOperands(), builder);
    index = builder
                .create(builder.getLoc(), op->getName().getIdentifier(),
                        replaced_args, op->getResultTypes(), op->getAttrs())
                ->getResult(0);
  }
  return indices;
}

mlir::Value GetSource(mlir::tensor::ExtractOp op) { return op.getTensor(); }

// TODO(jreiffers): Use a shared memory queue for pipelining instead of
// registers.
template <typename Op>
struct PipelineLoad : mlir::OpRewritePattern<Op> {
  using mlir::OpRewritePattern<Op>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      Op op, mlir::PatternRewriter& rewriter) const override {
    auto loop = mlir::dyn_cast_or_null<mlir::scf::ForOp>(op->getParentOp());
    if (!loop) {
      return rewriter.notifyMatchFailure(op, "no loop found");
    }

    if (auto step = loop.getConstantStep();
        !step || step->getSExtValue() != 1) {
      return rewriter.notifyMatchFailure(op, "loop step is not 1");
    }

    llvm::APInt lb, ub;
    if (!mlir::matchPattern(loop.getLowerBound(), mlir::m_ConstantInt(&lb)) ||
        !mlir::matchPattern(loop.getUpperBound(), mlir::m_ConstantInt(&ub))) {
      return rewriter.notifyMatchFailure(op, "bounds are not constants");
    }
    if (lb.getSExtValue() != 0) {
      return rewriter.notifyMatchFailure(op, "lower bound is not 0");
    }

    auto source = GetSource(op);
    if (!source.getParentRegion()->isProperAncestor(&loop.getBodyRegion())) {
      return rewriter.notifyMatchFailure(
          op, "source is not defined outside the loop");
    }

    if (!DoIndicesDependOnInductionVar(op.getIndices(), loop)) {
      // We don't run LICM between iterations, so this could happen.
      // Just hoist the load out of the loop.
      rewriter.moveOpBefore(op, loop);
      return mlir::success();
    }

    if (!CanReplaceInductionVar(op.getIndices())) {
      return rewriter.notifyMatchFailure(op, "unable to replace indices");
    }

    mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    mlir::Value zero = b.create<mlir::arith::ConstantIndexOp>(0);

    b.setInsertionPoint(loop);
    auto first_args =
        ReplaceInductionVar(loop.getInductionVar(), zero, op.getOperands(), b);
    auto loaded_first =
        b.create<Op>(op->getResultTypes(), first_args, op->getAttrs());
    auto ub_minus_one =
        b.create<mlir::arith::ConstantIndexOp>(ub.getSExtValue() - 1);

    b.setInsertionPointToStart(loop.getBody());

    auto needs_load = b.create<mlir::arith::CmpIOp>(
        mlir::arith::CmpIPredicate::ult, loop.getInductionVar(), ub_minus_one);
    auto next_value =
        b.create<mlir::scf::IfOp>(op->getResultTypes(), needs_load, true, true);
    auto new_for =
        mlir::cast<mlir::scf::ForOp>(*loop.replaceWithAdditionalYields(
            rewriter, loaded_first->getResult(0),
            /*replaceInitOperandUsesInLoop=*/false,
            [&](mlir::OpBuilder&, mlir::Location,
                llvm::ArrayRef<mlir::BlockArgument>) {
              return llvm::SmallVector<mlir::Value>{next_value->getResult(0)};
            }));
    rewriter.replaceAllUsesWith(op, new_for.getRegionIterArgs().back());

    b.setInsertionPointToStart(next_value.thenBlock());
    auto yield = b.create<mlir::scf::YieldOp>(op->getResult(0));

    // We use this convoluted way to add 1 so folding works properly.
    auto plus_one_map = mlir::AffineMap::get(
        1, 0, mlir::getAffineDimExpr(0, this->getContext()) + 1);
    b.setInsertionPoint(next_value);
    auto induction_plus_one =
        b.create<ApplyIndexingOp>(new_for.getInductionVar(), plus_one_map, 0,
                                  ub.getSExtValue() - 1)
            ->getResult(0);

    // Create the new apply_indexing ops outside the if, to improve CSE.
    rewriter.modifyOpInPlace(op, [&]() {
      op->setOperands(ReplaceInductionVar(
          new_for.getInductionVar(), induction_plus_one, op->getOperands(), b));
    });
    rewriter.moveOpBefore(op, yield);

    b.setInsertionPointToStart(next_value.elseBlock());
    b.create<mlir::scf::YieldOp>(new_for.getRegionIterArgs().back());
    return mlir::success();
  }
};

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
  op.getBodyRegion().walk([&](mlir::Operation* op) {
    if (mlir::isa<mlir::func::CallOp, mlir::scf::ForOp>(op)) {
      size += kMaxSize;
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
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(
            getOperation(), std::move(unroll_patterns)))) {
      signalPassFailure();
      return;
    }

    // Then pipeline the remaining loops.
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<PipelineLoad<mlir::vector::TransferReadOp>,
                 PipelineLoad<mlir::tensor::ExtractOp>>(&getContext());
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(getOperation(),
                                                        std::move(patterns)))) {
      signalPassFailure();
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
