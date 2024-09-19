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
#include <cassert>
#include <cstdint>
#include <memory>
#include <utility>

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/service/gpu/fusions/ir/xla_gpu_ops.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/model/indexing_analysis.h"
#include "xla/util.h"

namespace xla {
namespace gpu {

#define GEN_PASS_DEF_REWRITEREDUCTIONSPASS
#include "xla/service/gpu/fusions/transforms/passes.h.inc"

namespace {

class RewriteReductionsPass
    : public impl::RewriteReductionsPassBase<RewriteReductionsPass> {
 public:
  void runOnOperation() override;
};

mlir::ShapedType GetInputType(ReduceOp op) {
  return mlir::cast<mlir::ShapedType>(op.getOperand(0).getType());
}

mlir::ShapedType GetOutputType(ReduceOp op) {
  return mlir::cast<mlir::ShapedType>(op.getResult(0).getType());
}

int GetNumThreads(mlir::Operation* op) {
  auto grid =
      op->getParentOfType<mlir::func::FuncOp>()->getAttrOfType<LaunchGridAttr>(
          "xla_gpu.launch_grid");
  assert(grid);
  return Product(grid.getThreadCounts());
}

std::pair<int, int64_t> GetNumAndSizeOfMinorReducedDimensions(ReduceOp op) {
  llvm::ArrayRef<int64_t> dims = op.getDimensions();
  auto input_ty = GetInputType(op);
  int64_t cumulative_size = 1;
  for (int i = 0; i < dims.size(); ++i) {
    // The expected next reduction dimension if it is contiguous with the
    // previously reduced dimensions.
    int expected_dim = input_ty.getRank() - 1 - i;
    // If the next reduced dimension is not the expected one, it is not
    // contiguous (i.e., it's not part of the minor reduced dimensions, there is
    // a kept dimension in between).
    if (dims[dims.size() - 1 - i] != expected_dim) {
      return {i, cumulative_size};
    }
    cumulative_size *= input_ty.getDimSize(input_ty.getRank() - 1 - i);
  }
  return {dims.size(), cumulative_size};
}

// Rewrites large row reductions to three reductions:
// 1. to one element per thread.
// 2. to one element per warp.
// 3. to one element per block.
// This also pads the input if the number of threads does not divide the row
// size.
struct RewriteRowReduction : mlir::OpRewritePattern<ReduceOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      ReduceOp op, mlir::PatternRewriter& rewriter) const override {
    auto* ctx = op.getContext();

    auto [num_minor_dims, reduced_size] =
        GetNumAndSizeOfMinorReducedDimensions(op);
    if (num_minor_dims == 0) {
      return rewriter.notifyMatchFailure(op, "not a row reduction");
    }

    if (reduced_size <= WarpSize()) {
      return rewriter.notifyMatchFailure(op, "small minor dimension");
    }

    int64_t num_threads = GetNumThreads(op);
    assert(num_threads % WarpSize() == 0);

    llvm::ArrayRef<int64_t> input_shape = GetInputType(op).getShape();
    llvm::SmallVector<int64_t, 4> projected_input_shape{
        input_shape.begin(), input_shape.end() - num_minor_dims};
    projected_input_shape.push_back(reduced_size);

    // Collapse the minor dimensions into one.
    // [..., 123, 456] -> [..., 123 * 456]
    auto projection_map =
        GetBitcastMap(projected_input_shape, input_shape, ctx);

    // Pad the new minor dimension to a multiple of the number of threads. For
    // example, for 128 threads, 123 * 456 = 56088 is padded to 56192.
    auto padded_projected_input_shape = projected_input_shape;
    int64_t padded_size = RoundUpTo(reduced_size, num_threads);
    padded_projected_input_shape.back() = padded_size;

    // Reshape the padded minor dimension so that we can reduce it per thread
    // and then per warp.
    // [..., 56192] -> [..., 439, 4, 32]
    llvm::SmallVector<int64_t, 4> per_thread_reduction_input_shape(
        input_shape.begin(), input_shape.end() - num_minor_dims);
    per_thread_reduction_input_shape.push_back(padded_size / num_threads);
    per_thread_reduction_input_shape.push_back(num_threads / WarpSize());
    per_thread_reduction_input_shape.push_back(WarpSize());

    int per_thread_input_rank = per_thread_reduction_input_shape.size();

    auto reindex_map = GetBitcastMap(per_thread_reduction_input_shape,
                                     padded_projected_input_shape, ctx) *
                       projection_map;
    reindex_map.AddConstraint(
        mlir::getAffineDimExpr(per_thread_input_rank - 1, ctx) +
            mlir::getAffineDimExpr(per_thread_input_rank - 2, ctx) *
                num_threads,
        {0, reduced_size - 1});

    // Reshape the inputs.
    llvm::SmallVector<mlir::Value, 2> new_operands;
    new_operands.reserve(op.getOperands().size());
    for (auto [operand, init] : llvm::zip(op.getInputs(), op.getInits())) {
      auto new_input_ty = mlir::cast<mlir::ShapedType>(operand.getType())
                              .clone(per_thread_reduction_input_shape);
      new_operands.push_back(rewriter.create<ReindexOp>(
          operand.getLoc(), new_input_ty, operand, init, reindex_map));
    }

    // Reduce the non-minor dimensions and the third to last dimension.
    auto dims_for_first_reduction =
        llvm::to_vector(op.getDimensions().drop_back(num_minor_dims));
    dims_for_first_reduction.push_back(per_thread_input_rank - 3);
    auto first_reduction =
        rewriter.create<ReduceOp>(op.getLoc(), new_operands, op.getInits(),
                                  dims_for_first_reduction, op.getCombiner());

    // Reduce the last and the second-to-last dimensions. First to produce one
    // output element per warp, then to produce one output element per block.
    int rank = GetOutputType(first_reduction).getRank();
    auto second_reduction = rewriter.create<ReduceOp>(
        op.getLoc(), first_reduction.getResults(), op.getInits(),
        llvm::ArrayRef<int64_t>{rank - 1}, op.getCombiner());
    rewriter.replaceOpWithNewOp<ReduceOp>(
        op, second_reduction.getResults(), op.getInits(),
        llvm::ArrayRef<int64_t>{rank - 2}, op.getCombiner());

    return mlir::success();
  }
};

void RewriteReductionsPass::runOnOperation() {
  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<RewriteRowReduction>(&getContext());
  if (mlir::failed(mlir::applyPatternsAndFoldGreedily(getOperation(),
                                                      std::move(patterns)))) {
    signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateRewriteReductionsPass() {
  return std::make_unique<RewriteReductionsPass>();
}

}  // namespace gpu
}  // namespace xla
