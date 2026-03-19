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
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/codegen/emitters/ir/xla_ops.h"
#include "xla/hlo/analysis/indexing_map.h"

namespace xla::cpu {

#define GEN_PASS_DECL_PEELWORKGROUPLOOPPASS
#define GEN_PASS_DEF_PEELWORKGROUPLOOPPASS
#include "xla/backends/cpu/codegen/emitters/transforms/passes.h.inc"

namespace {

struct PeelWorkgroupLoopPattern : public mlir::OpRewritePattern<xla::LoopOp> {
  using OpRewritePattern::OpRewritePattern;

  struct WorkGroupInfo {
    mlir::Value operand;
    uint64_t dim_id;
    Interval interval;
  };

  mlir::LogicalResult matchAndRewrite(
      xla::LoopOp loop_op, mlir::PatternRewriter& rewriter) const override {
    mlir::MLIRContext* ctx = loop_op.getContext();
    xla::IndexingMap indexing_map = loop_op.getIndexingMap();
    indexing_map.Simplify();

    if (indexing_map.GetConstraintsCount() == 0) {
      return rewriter.notifyMatchFailure(loop_op, "No constraints to peel.");
    }

    // We need to check this to be able to just check the constraints are
    // satisfied by setting the non-query dimensions to their upper bound.
    for (const auto& [constraint, interval] : indexing_map.GetConstraints()) {
      if (!IsMonotonicIncreasing(constraint)) {
        return rewriter.notifyMatchFailure(
            loop_op, "Indexing map may not be monotonic increasing.");
      }
    }

    std::vector<mlir::AffineExpr> dimension_upper_bounds =
        GetDimensionUpperBounds(indexing_map, ctx);
    std::vector<mlir::AffineExpr> symbol_upper_bounds =
        GetSymbolUpperBounds(indexing_map, ctx);

    auto work_group_dims = GatherWorkGroupDims(loop_op);

    for (const auto& work_group_dim : work_group_dims) {
      const Interval& interval = work_group_dim.interval;
      uint64_t dim_id = work_group_dim.dim_id;
      if (interval.IsPoint()) {
        continue;
      }

      if (interval.lower != 0) {
        continue;
      }

      int64_t query_dimension_upper = interval.upper;

      // Search for the largest dimension upper bound that satisfies the
      // constraints, the bound will be near the existing upper bound so linear
      // search is fine.
      while (!indexing_map.ConstraintsSatisfied(dimension_upper_bounds,
                                                symbol_upper_bounds) &&
             query_dimension_upper != 0) {
        query_dimension_upper--;
        dimension_upper_bounds[dim_id] =
            mlir::getAffineConstantExpr(query_dimension_upper, ctx);
      }

      if (query_dimension_upper == 0 ||
          query_dimension_upper == interval.upper) {
        continue;
      }

      mlir::ImplicitLocOpBuilder builder(loop_op.getLoc(), rewriter);
      auto cmp_op = mlir::arith::CmpIOp::create(
          builder, mlir::arith::CmpIPredicate::sle, work_group_dim.operand,
          mlir::arith::ConstantIndexOp::create(builder, query_dimension_upper));

      auto loop_body_cloner = GetLoopBodyCloner(loop_op);

      auto then_body_builder = [&](mlir::OpBuilder& then_builder,
                                   mlir::Location then_loc) -> void {
        IndexingMap peeled_map = indexing_map;
        peeled_map.ClearConstraints();
        peeled_map.GetMutableDimensionBound(dim_id).upper =
            query_dimension_upper;
        peeled_map.Simplify();

        auto peeled_loop = LoopOp::create(then_builder, then_loc, peeled_map,
                                          loop_op.getDims(), loop_op.getInits(),
                                          loop_body_cloner);
        mlir::scf::YieldOp::create(then_builder, then_loc,
                                   peeled_loop.getResults());
      };
      auto else_body_builder = [&](mlir::OpBuilder& else_builder,
                                   mlir::Location else_loc) -> void {
        IndexingMap tail_map = indexing_map;
        tail_map.GetMutableDimensionBound(dim_id).lower =
            query_dimension_upper + 1;
        tail_map.Simplify();

        auto tail_loop =
            LoopOp::create(else_builder, else_loc, tail_map, loop_op.getDims(),
                           loop_op.getInits(), loop_body_cloner);
        mlir::scf::YieldOp::create(else_builder, else_loc,
                                   tail_loop.getResults());
      };

      auto if_op = mlir::scf::IfOp::create(builder, cmp_op, then_body_builder,
                                           else_body_builder);

      rewriter.replaceOp(loop_op, if_op.getResults());
      return mlir::success();
    }

    return rewriter.notifyMatchFailure(loop_op, "No constraints to peel.");
  }

 private:
  static std::vector<mlir::AffineExpr> GetDimensionUpperBounds(
      const xla::IndexingMap& indexing_map, mlir::MLIRContext* ctx) {
    std::vector<mlir::AffineExpr> dimension_upper_bounds;
    for (auto [lower, upper] : indexing_map.GetDimensionBounds()) {
      dimension_upper_bounds.push_back(mlir::getAffineConstantExpr(upper, ctx));
    }

    return dimension_upper_bounds;
  }

  static std::vector<mlir::AffineExpr> GetSymbolUpperBounds(
      const xla::IndexingMap& indexing_map, mlir::MLIRContext* ctx) {
    std::vector<mlir::AffineExpr> symbol_upper_bounds;
    for (auto [lower, upper] : indexing_map.GetSymbolBounds()) {
      symbol_upper_bounds.push_back(mlir::getAffineConstantExpr(upper, ctx));
    }

    return symbol_upper_bounds;
  }

  // Check if the constraint is monotonic increasing, it is conservative in
  // that it may return false in some cases where the constraint may actually be
  // monotonic increasing but it will never return true in the case where it is
  // monotonically decreasing.
  static bool IsMonotonicIncreasing(const mlir::AffineExpr& constraint) {
    bool is_monotonic_increasing = true;
    constraint.walk([&is_monotonic_increasing](mlir::AffineExpr expr) {
      if (expr.getKind() == mlir::AffineExprKind::Mod) {
        is_monotonic_increasing = false;
        return;
      }
      if (auto binary_op = mlir::dyn_cast<mlir::AffineBinaryOpExpr>(expr)) {
        auto is_negative_constant = [&](mlir::AffineExpr expr) {
          if (auto const_expr =
                  mlir::dyn_cast<mlir::AffineConstantExpr>(expr)) {
            return const_expr.getValue() < 0;
          }
          return false;
        };
        if (is_negative_constant(binary_op.getLHS()) ||
            is_negative_constant(binary_op.getRHS())) {
          is_monotonic_increasing = false;
          return;
        }
      }
    });

    return is_monotonic_increasing;
  }

  static absl::InlinedVector<WorkGroupInfo, 3> GatherWorkGroupDims(
      xla::LoopOp& loop_op) {
    IndexingMap indexing_map = loop_op.getIndexingMap();
    absl::InlinedVector<WorkGroupInfo, 3> work_group_dims;
    for (auto [dim_id, dim_operand] : llvm::enumerate(loop_op.getDims())) {
      if (!mlir::isa_and_present<WorkGroupIdOp>(dim_operand.getDefiningOp())) {
        continue;
      }
      Interval interval = indexing_map.GetDimensionBound(dim_id);
      work_group_dims.push_back(WorkGroupInfo{dim_operand, dim_id, interval});
    }

    return work_group_dims;
  }

  absl::AnyInvocable<void(mlir::OpBuilder&, mlir::Location, mlir::ValueRange,
                          mlir::ValueRange, mlir::ValueRange)>
  GetLoopBodyCloner(xla::LoopOp& loop_op) const {
    return [&loop_op](mlir::OpBuilder& nested_b, mlir::Location nested_loc,
                      mlir::ValueRange ivs, mlir::ValueRange map_results,
                      mlir::ValueRange iter_args) {
      mlir::OpBuilder::InsertionGuard guard(nested_b);
      mlir::IRMapping mapping;
      mapping.map(loop_op.getInductionVars(), ivs);
      mapping.map(loop_op.getIndexingMapResults(), map_results);
      mapping.map(loop_op.getRegionIterArgs(), iter_args);
      for (auto& op : loop_op.getBody()->getOperations()) {
        nested_b.clone(op, mapping);
      }
    };
  }
};

class PeelWorkgroupLoopPass
    : public impl::PeelWorkgroupLoopPassBase<PeelWorkgroupLoopPass> {
 public:
  using PeelWorkgroupLoopPassBase::PeelWorkgroupLoopPassBase;

  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<PeelWorkgroupLoopPattern>(&getContext());
    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> CreatePeelWorkgroupLoopPass() {
  return std::make_unique<PeelWorkgroupLoopPass>();
}

}  // namespace xla::cpu
