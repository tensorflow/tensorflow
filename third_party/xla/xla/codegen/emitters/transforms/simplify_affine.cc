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
#include <limits>
#include <utility>
#include <vector>

#include "absl/base/optimization.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/codegen/emitters/ir/xla_ops.h"
#include "xla/codegen/emitters/transforms/passes.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/analysis/symbolic_map.h"
#include "xla/hlo/analysis/symbolic_map_converter.h"

namespace xla {
namespace emitters {

#define GEN_PASS_DEF_SIMPLIFYAFFINEPASS
#include "xla/codegen/emitters/transforms/passes.h.inc"

namespace {

using mlir::AffineMap;
using mlir::ImplicitLocOpBuilder;
using mlir::LogicalResult;
using mlir::MLIRContext;
using mlir::ModuleOp;
using mlir::OpRewritePattern;
using mlir::PatternRewriter;
using mlir::SmallVector;
using mlir::Value;
using mlir::ValueRange;
using mlir::affine::AffineApplyOp;

namespace arith = mlir::arith;

int Distance(ImplicitLocOpBuilder& builder, Value a) {
  auto* block = builder.getInsertionBlock();
  auto* parent = a.getParentBlock();
  int distance = 0;
  while (block && block != parent) {
    ++distance;
    block = block->getParentOp()->getBlock();
  }
  return distance;
}

void CollectArgs(SymbolicExpr expr, SymbolicExprType type,
                 llvm::SmallVector<SymbolicExpr>& ret) {
  if (expr.GetType() == type) {
    CollectArgs(expr.GetLHS(), type, ret);
    CollectArgs(expr.GetRHS(), type, ret);
    return;
  }
  ret.push_back(expr);
}

// Returns true if the top-level operation of `expr` can be lowered to an
// unsigned arith op. Operands of `expr` are not checked.
bool IsUnsignedLoweringSupported(SymbolicExpr expr,
                                 RangeEvaluator& range_evaluator) {
  if (!expr.IsBinaryOp()) {
    return true;
  }
  // Mod and div can be lowered if their LHS is >= 0 and their RHS is a
  // constant positive number.
  if (expr.GetType() == SymbolicExprType::kMod ||
      expr.GetType() == SymbolicExprType::kFloorDiv ||
      expr.GetType() == SymbolicExprType::kCeilDiv) {
    if (!range_evaluator.IsAlwaysPositiveOrZero(expr.GetLHS())) {
      return false;
    }
    auto rhs_range = range_evaluator.ComputeExpressionRange(expr.GetRHS());
    if (!rhs_range.IsPoint() || rhs_range.lower <= 0) {
      return false;
    }
  }
  // Max and min can be lowered if both operands are >= 0.
  if (expr.GetType() == SymbolicExprType::kMax ||
      expr.GetType() == SymbolicExprType::kMin) {
    if (!range_evaluator.IsAlwaysPositiveOrZero(expr.GetLHS()) ||
        !range_evaluator.IsAlwaysPositiveOrZero(expr.GetRHS())) {
      return false;
    }
  }
  return true;
}

struct ExpressionEvaluator {
  // If `range_evaluator` is null, all mod, div, min and max operations are
  // lowered to unsigned arith ops. The caller must ensure that this is correct,
  // e.g. by checking `IsLoweringSupported`. Otherwise, signed arith ops are
  // emitted for the operations whose unsigned lowering is not provably correct.
  ExpressionEvaluator(ImplicitLocOpBuilder& builder, ValueRange operands,
                      RangeEvaluator* range_evaluator = nullptr)
      : builder(builder), operands(operands), range_evaluator(range_evaluator) {
    for (int i = 0; i < operands.size(); ++i) {
      variable_distances.push_back(Distance(builder, operands[i]));
    }
  }

  // Returns the distance (in basic blocks) from the insertion point to the
  // values used in the given expression.
  int ExprDistance(SymbolicExpr e, int depth = 0) {
    if (e.GetType() == SymbolicExprType::kVariable) {
      return variable_distances[e.GetValue()];
    }
    if (e.IsBinaryOp()) {
      return std::min(ExprDistance(e.GetLHS(), depth + 1),
                      ExprDistance(e.GetRHS(), depth + 1));
    }
    if (depth == 0) {
      // Top-level constant. Always add these last.
      return std::numeric_limits<int>::min();
    }
    // Nested constant. Ignore these for distances.
    return std::numeric_limits<int>::max();
  }

  Value EvaluateExpression(SymbolicExpr expr);

  template <typename Op>
  Value EvaluateAddMul(SymbolicExpr expr);

  ImplicitLocOpBuilder& builder;
  ValueRange operands;
  RangeEvaluator* range_evaluator;
  SmallVector<int> variable_distances;
};

template <typename Op>
Value ExpressionEvaluator::EvaluateAddMul(SymbolicExpr expr) {
  llvm::SmallVector<SymbolicExpr> args;
  CollectArgs(expr, expr.GetType(), args);
  // Sort the args so that the ones that are closest to the insertion point
  // are evaluated last - this improves LICM.
  llvm::stable_sort(args, [&](SymbolicExpr a, SymbolicExpr b) {
    int dist_a = ExprDistance(a);
    int dist_b = ExprDistance(b);
    return dist_a > dist_b;
  });

  Value result = nullptr;
  for (auto arg : args) {
    Value arg_evaluated = EvaluateExpression(arg);
    if (result) {
      result = builder.create<Op>(result, arg_evaluated,
                                  arith::IntegerOverflowFlags::nsw);
    } else {
      result = arg_evaluated;
    }
  }

  return result;
}

Value ExpressionEvaluator::EvaluateExpression(SymbolicExpr expr) {
  bool use_signed = range_evaluator != nullptr &&
                    !IsUnsignedLoweringSupported(expr, *range_evaluator);
  switch (expr.GetType()) {
    case SymbolicExprType::kAdd:
      return EvaluateAddMul<arith::AddIOp>(expr);
    case SymbolicExprType::kMul:
      return EvaluateAddMul<arith::MulIOp>(expr);
    case SymbolicExprType::kMod: {
      Value lhs = EvaluateExpression(expr.GetLHS());
      Value rhs = EvaluateExpression(expr.GetRHS());
      if (!use_signed) {
        return builder.create<arith::RemUIOp>(lhs, rhs);
      }
      // SymbolicExpr uses floor semantics: lhs mod rhs has the sign of rhs.
      Value quotient = builder.create<arith::FloorDivSIOp>(lhs, rhs);
      return builder.create<arith::SubIOp>(
          lhs, builder.create<arith::MulIOp>(quotient, rhs));
    }
    case SymbolicExprType::kFloorDiv: {
      Value lhs = EvaluateExpression(expr.GetLHS());
      Value rhs = EvaluateExpression(expr.GetRHS());
      if (use_signed) {
        return builder.create<arith::FloorDivSIOp>(lhs, rhs);
      }
      return builder.create<arith::DivUIOp>(lhs, rhs);
    }
    case SymbolicExprType::kCeilDiv: {
      Value lhs = EvaluateExpression(expr.GetLHS());
      Value rhs = EvaluateExpression(expr.GetRHS());
      if (use_signed) {
        return builder.create<arith::CeilDivSIOp>(lhs, rhs);
      }
      return builder.create<arith::CeilDivUIOp>(lhs, rhs);
    }
    case SymbolicExprType::kMax: {
      Value lhs = EvaluateExpression(expr.GetLHS());
      Value rhs = EvaluateExpression(expr.GetRHS());
      if (use_signed) {
        return builder.create<arith::MaxSIOp>(lhs, rhs);
      }
      return builder.create<arith::MaxUIOp>(lhs, rhs);
    }
    case SymbolicExprType::kMin: {
      Value lhs = EvaluateExpression(expr.GetLHS());
      Value rhs = EvaluateExpression(expr.GetRHS());
      if (use_signed) {
        return builder.create<arith::MinSIOp>(lhs, rhs);
      }
      return builder.create<arith::MinUIOp>(lhs, rhs);
    }
    case SymbolicExprType::kConstant:
      return builder.create<arith::ConstantIndexOp>(expr.GetValue());
    case SymbolicExprType::kVariable:
      return operands[expr.GetValue()];
    default:
      ABSL_UNREACHABLE();
  }
}

// Returns true if all operations in `expr` can be lowered to unsigned arith
// ops.
bool IsLoweringSupported(SymbolicExpr expr, RangeEvaluator& range_evaluator) {
  if (!expr.IsBinaryOp()) {
    return true;
  }
  return IsUnsignedLoweringSupported(expr, range_evaluator) &&
         IsLoweringSupported(expr.GetLHS(), range_evaluator) &&
         IsLoweringSupported(expr.GetRHS(), range_evaluator);
}

// TODO: b/446856305 - Create a RewriteSymbolicApply pattern that takes a
// SymbolicMap. For now, we convert the AffineMap to SymbolicMap.
struct RewriteAffineApply : OpRewritePattern<mlir::affine::AffineApplyOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::affine::AffineApplyOp op,
                                PatternRewriter& rewriter) const override {
    SymbolicMap symbolic_map = AffineMapToSymbolicMap(op.getAffineMap());
    std::vector<IndexingMap::Variable> dim_ranges(symbolic_map.GetNumDims());
    std::vector<IndexingMap::Variable> symbol_ranges(
        symbolic_map.GetNumSymbols());

    for (int i = 0;
         i < symbolic_map.GetNumDims() + symbolic_map.GetNumSymbols(); ++i) {
      if (auto range = GetRange(op->getOperand(i))) {
        if (i >= dim_ranges.size()) {
          symbol_ranges[i - dim_ranges.size()] = IndexingMap::Variable{*range};
        } else {
          dim_ranges[i] = IndexingMap::Variable{*range};
        }
      } else {
        return rewriter.notifyMatchFailure(op, "failed to deduce range");
      }
    }

    IndexingMap indexing_map(symbolic_map, std::move(dim_ranges),
                             std::move(symbol_ranges),
                             /*rt_vars=*/{});
    indexing_map.Simplify();
    auto result_expr = indexing_map.GetSymbolicMap().GetResult(0);

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    RangeEvaluator range_evaluator = indexing_map.GetRangeEvaluator();
    if (!IsLoweringSupported(result_expr, range_evaluator)) {
      return rewriter.notifyMatchFailure(op,
                                         "unable to lower the affine apply");
    }
    b.setInsertionPoint(op);
    auto result = ExpressionEvaluator(b, op->getOperands())
                      .EvaluateExpression(result_expr);
    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct RewriteApplyIndexingOp : OpRewritePattern<ApplyIndexingOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ApplyIndexingOp op,
                                PatternRewriter& rewriter) const override {
    auto indexing_map = op.getIndexingMap();
    indexing_map.Simplify();
    auto symbolic_map = indexing_map.GetSymbolicMap();
    auto operands = op->getOperands();

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    RangeEvaluator range_evaluator = indexing_map.GetRangeEvaluator();

    b.setInsertionPoint(op);
    SmallVector<Value, 4> results;
    results.reserve(symbolic_map.GetNumResults());
    for (unsigned i = 0; i < symbolic_map.GetNumResults(); ++i) {
      SymbolicExpr result_expr = symbolic_map.GetResult(i);
      if (IsLoweringSupported(result_expr, range_evaluator)) {
        results.push_back(
            ExpressionEvaluator(b, operands).EvaluateExpression(result_expr));
        continue;
      }
      // If the expression cannot be lowered to unsigned ops, we convert it to
      // affine.apply, since it supports more expression types.
      // TODO: b/446856305 - Create a SymbolicApplyOp. For now, we convert the
      // SymbolicMap back to AffineMap and fall back to AffineApplyOp.
      if (AffineMap sub_map =
              SymbolicMapToAffineMap(symbolic_map.GetSubMap({i}))) {
        results.push_back(b.create<AffineApplyOp>(
            sub_map, operands.take_front(symbolic_map.GetNumDims() +
                                         symbolic_map.GetNumSymbols())));
        continue;
      }
      // affine.apply does not support min and max. Lower the expression to
      // arith ops, using signed ops wherever unsigned ones are not provably
      // correct.
      results.push_back(ExpressionEvaluator(b, operands, &range_evaluator)
                            .EvaluateExpression(result_expr));
    }
    rewriter.replaceOp(op, results);
    return mlir::success();
  }
};

struct SimplifyAffinePass
    : public impl::SimplifyAffinePassBase<SimplifyAffinePass> {
 public:
  void runOnOperation() override {
    MLIRContext* ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);
    patterns.add<RewriteAffineApply, RewriteApplyIndexingOp>(ctx);
    mlir::GreedyRewriteConfig config;
    // There's no point simplifying more than once.
    config.setStrictness(mlir::GreedyRewriteStrictness::ExistingOps);
    if (mlir::failed(mlir::applyPatternsGreedily(
            getOperation(), std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

}  // namespace

}  // namespace emitters
}  // namespace xla
