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

#include "llvm/ADT/APInt.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/backends/gpu/codegen/emitters/ir/xla_gpu_ops.h"  // IWYU pragma: keep
#include "xla/backends/gpu/codegen/emitters/transforms/passes.h"  // IWYU pragma: keep
#include "xla/service/gpu/gpu_fusible.h"

namespace xla {
namespace gpu {

#define GEN_PASS_DEF_OPTIMIZELOOPSPASS
#include "xla/backends/gpu/codegen/emitters/transforms/passes.h.inc"

namespace {

bool IsExpensiveToUnroll(mlir::Operation* op) {
  return mlir::isa<
      // clang-format off
      // go/keep-sorted start
      mlir::math::AcosOp,
      mlir::math::AcoshOp,
      mlir::math::AsinOp,
      mlir::math::AsinhOp,
      mlir::math::AtanhOp,
      mlir::math::SinhOp
      // go/keep-sorted end
      // clang-format on
      >(op);
}

std::optional<int64_t> GetConstantTripCount(mlir::scf::ForOp for_op) {
  std::optional<llvm::APInt> step = for_op.getConstantStep();
  if (!step) {
    return std::nullopt;
  }

  llvm::APInt lower_bound, upper_bound;
  if (mlir::matchPattern(for_op.getLowerBound(),
                         mlir::m_ConstantInt(&lower_bound)) &&
      mlir::matchPattern(for_op.getUpperBound(),
                         mlir::m_ConstantInt(&upper_bound))) {
    return (upper_bound.getSExtValue() - lower_bound.getSExtValue()) /
           step->getSExtValue();
  }
  return std::nullopt;
}

// Returns a rough estimate of the number of instructions in the operation.
// Returns std::nullopt if the operation is expensive to unroll.
std::optional<int64_t> EstimateSize(mlir::Operation* op,
                                    const mlir::SymbolTable& symbolTable) {
  if (IsExpensiveToUnroll(op)) {
    return std::nullopt;
  }

  if (auto call = mlir::dyn_cast<mlir::CallOpInterface>(op)) {
    if (auto symbol = mlir::dyn_cast_or_null<mlir::SymbolRefAttr>(
            call.getCallableForCallee())) {
      if (auto callee = symbolTable.lookup<mlir::func::FuncOp>(
              symbol.getLeafReference())) {
        return EstimateSize(callee, symbolTable);
      }
    }
  }

  int64_t size = 0;
  if (!mlir::isa<mlir::func::FuncOp, mlir::func::ReturnOp, mlir::scf::ForOp,
                 mlir::scf::YieldOp, xla::YieldOp, mlir::scf::IfOp>(op)) {
    size = 1;
    if (mlir::isa<mlir::math::MathDialect, mlir::arith::ArithDialect>(
            op->getDialect()) &&
        !op->getResultTypes().empty() &&
        !op->getResultTypes().front().isIntOrIndex()) {
      namespace mm = mlir::math;
      namespace ma = mlir::arith;
      if (!mlir::isa<mm::AbsFOp, mm::CeilOp, mm::CopySignOp, mm::FloorOp,
                     mm::FmaOp, mm::RoundEvenOp, mm::RoundOp, mm::RsqrtOp,
                     mm::SqrtOp, mm::TruncOp, ma::AddFOp, ma::SubFOp,
                     ma::MulFOp, ma::ExtFOp, ma::TruncFOp>(op)) {
        size = 20;
      }
    }

    if (!op->getResultTypes().empty()) {
      if (auto vector_ty =
              mlir::dyn_cast<mlir::VectorType>(op->getResultTypes().front())) {
        size *= vector_ty.getNumElements();
      }
    }
  }

  for (mlir::Region& region : op->getRegions()) {
    for (mlir::Operation& nested_op : region.getOps()) {
      auto nested_size = EstimateSize(&nested_op, symbolTable);
      if (!nested_size) {
        return std::nullopt;
      }
      size += *nested_size;
    }
  }

  if (auto for_op = mlir::dyn_cast<mlir::scf::ForOp>(op)) {
    constexpr int kDefaultAssumedTripCount = 10;
    return size *
           GetConstantTripCount(for_op).value_or(kDefaultAssumedTripCount);
  }

  return size;
}

int GetUnrollingFactor(mlir::scf::ForOp op,
                       const mlir::SymbolTable& symbolTable) {
  // We only unroll loops with a step of 1 and a lower bound of 0. That's the
  // only type we generate.
  if (auto step = op.getConstantStep();
      !step || step->getSExtValue() != 1 ||
      !mlir::matchPattern(op.getLowerBound(), mlir::m_Zero())) {
    return 1;
  }

  auto trip_count = GetConstantTripCount(op);
  if (!trip_count) {
    return 1;
  }

  constexpr int kMaxSize = 400;  // Chosen empirically.

  auto size = EstimateSize(op, symbolTable);
  if (!size) {
    return 1;
  }

  // Always unroll if the trip count is smaller than the max unroll factor,
  // because it's very likely that the loop was meant to be unrolled.
  // We also check the total size to avoid massive register pressure.
  if (*trip_count <= MaxUnrollFactor() && *size <= kMaxSize) {
    return *trip_count;
  }

  int factor = std::min<int64_t>(*trip_count, kMaxSize / (*size / *trip_count));
  while (factor > 1 && *trip_count % factor) {
    --factor;
  }
  return factor;
}

struct UnrollLoops : mlir::OpRewritePattern<mlir::scf::ForOp> {
  UnrollLoops(mlir::MLIRContext* context, const mlir::SymbolTable& symbolTable)
      : mlir::OpRewritePattern<mlir::scf::ForOp>(context),
        symbolTable(symbolTable) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::scf::ForOp op, mlir::PatternRewriter& rewriter) const override {
    if (int factor = GetUnrollingFactor(op, symbolTable); factor > 1) {
      return mlir::loopUnrollByFactor(op, factor);
    }
    return rewriter.notifyMatchFailure(op, "loop can't be unrolled");
  }

  const mlir::SymbolTable& symbolTable;
};

class OptimizeLoopsPass
    : public impl::OptimizeLoopsPassBase<OptimizeLoopsPass> {
 public:
  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    mlir::SymbolTable symbolTable(func->getParentOfType<mlir::ModuleOp>());

    // First unroll loops. If unrolling is possible, we prefer it.
    mlir::RewritePatternSet unroll_patterns(&getContext());
    unroll_patterns.add<UnrollLoops>(&getContext(), symbolTable);
    if (mlir::failed(mlir::applyPatternsGreedily(getOperation(),
                                                 std::move(unroll_patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> CreateOptimizeLoopsPass() {
  return std::make_unique<OptimizeLoopsPass>();
}

}  // namespace gpu
}  // namespace xla
