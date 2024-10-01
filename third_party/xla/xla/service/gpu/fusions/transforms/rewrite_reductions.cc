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
#include <cassert>
#include <cstdint>
#include <memory>
#include <utility>

#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
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
#include "xla/service/gpu/model/indexing_map.h"
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

struct DimensionGroup {
  int64_t size;
  int64_t stride;
  int first_dimension;
  int num_dimensions;
};

DimensionGroup GetMinorMostReduction(ReduceOp op) {
  llvm::ArrayRef<int64_t> dims = op.getDimensions();

  auto input_ty = GetInputType(op);
  DimensionGroup result{1, 1, static_cast<int>(input_ty.getRank()), 0};
  llvm::SmallBitVector reduced_dims(input_ty.getRank());
  for (int64_t dim : dims) {
    reduced_dims.set(dim);
  }

  // Look for the first group of consecutive reduced dimensions and compute the
  // stride and size of the group.
  bool in_reduction = false;
  for (int dim = input_ty.getRank() - 1;
       dim >= 0 && (!in_reduction || reduced_dims[dim]); --dim) {
    assert(input_ty.getDimSize(dim) > 1 &&
           "degenerate dimensions are not allowed");
    --result.first_dimension;
    if (reduced_dims[dim]) {
      in_reduction = true;
      result.size *= input_ty.getDimSize(dim);
      ++result.num_dimensions;
    } else {
      result.stride *= input_ty.getDimSize(dim);
    }
  }

  return result;
}

llvm::SmallVector<mlir::Value, 2> ReindexTensors(
    mlir::OpBuilder& b, mlir::ValueRange tensors, mlir::ValueRange defaults,
    llvm::ArrayRef<int64_t> new_shape, const IndexingMap& map) {
  llvm::SmallVector<mlir::Value, 2> reindexed;
  reindexed.reserve(tensors.size());
  for (auto [tensor, def] : llvm::zip(tensors, defaults)) {
    auto new_ty =
        mlir::cast<mlir::ShapedType>(tensor.getType()).clone(new_shape);
    reindexed.push_back(
        b.create<ReindexOp>(tensor.getLoc(), new_ty, tensor, def, map));
  }
  return reindexed;
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

    auto minor_reduction = GetMinorMostReduction(op);
    if (minor_reduction.stride > 1) {
      return rewriter.notifyMatchFailure(op, "not a row reduction");
    }

    if (minor_reduction.size <= WarpSize()) {
      return rewriter.notifyMatchFailure(op, "small minor dimension");
    }

    int64_t num_threads = GetNumThreads(op);
    assert(num_threads % WarpSize() == 0);

    llvm::ArrayRef<int64_t> input_shape = GetInputType(op).getShape();
    auto projected_input_shape = llvm::to_vector(
        input_shape.take_front(minor_reduction.first_dimension));
    projected_input_shape.push_back(minor_reduction.size);

    // Collapse the minor dimensions into one.
    // [..., 123, 456] -> [..., 123 * 456]
    auto projection_map =
        GetBitcastMap(projected_input_shape, input_shape, ctx);

    // Pad the new minor dimension to a multiple of the number of threads. For
    // example, for 128 threads, 123 * 456 = 56088 is padded to 56192.
    auto padded_projected_input_shape = projected_input_shape;
    int64_t padded_size = RoundUpTo(minor_reduction.size, num_threads);
    padded_projected_input_shape.back() = padded_size;

    // Reshape the padded minor dimension so that we can reduce it per thread
    // and then per warp.
    // [..., 56192] -> [..., 439, 4, 32]
    auto per_thread_reduction_input_shape = llvm::to_vector(
        input_shape.take_front(minor_reduction.first_dimension));
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
        {0, minor_reduction.size - 1});

    auto new_inputs =
        ReindexTensors(rewriter, op.getInputs(), op.getInits(),
                       per_thread_reduction_input_shape, reindex_map);

    // Reduce the non-minor dimensions and the third to last dimension.
    auto dims_for_first_reduction = llvm::to_vector(
        op.getDimensions().drop_back(minor_reduction.num_dimensions));
    dims_for_first_reduction.push_back(per_thread_input_rank - 3);
    auto first_reduction =
        rewriter.create<ReduceOp>(op.getLoc(), new_inputs, op.getInits(),
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

// Rewrites column reductions to a reduce-transpose-reduce.
struct RewriteColumnReduction : mlir::OpRewritePattern<ReduceOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      ReduceOp op, mlir::PatternRewriter& rewriter) const override {
    auto* ctx = op.getContext();

    auto minor_reduction = GetMinorMostReduction(op);

    if (minor_reduction.stride == 1) {
      return rewriter.notifyMatchFailure(op, "not a column reduction");
    }

    int64_t num_threads = GetNumThreads(op);

    // If the stride is larger than the number of threads, we can efficiently
    // emit this reduction as a simple loop, assuming there's no excessive
    // padding.
    // TODO(jreiffers): Is there anything we can do if the number of threads
    // doesn't divide the stride?
    if (minor_reduction.stride >= num_threads) {
      return rewriter.notifyMatchFailure(op, "efficient loop reduction");
    }

    // A column reduction reduces [a, b] to [b]. We do this in four steps:
    // 1. reshape [a, b] to [a ceildiv c, c, b]
    // 2. reduce [a ceildiv c, c, b] to [c, b] via a loop
    // 3. transpose [c, b] to [b, c]
    // 4. emit a row reduction on [b, c].
    //
    // We are constrained in our choice for `c`:
    //
    // - we need one element of shared memory (or a register) for each element
    //   of the intermediate results, so a larger c needs more shared memory.
    // - we can have at most WarpSize intermediate results per final result,
    //   so c can be at most 32.
    // - c must be a power of two so we can use a warp shuffle.
    // - c * b should be less than the number of threads (but as close to it
    //   as possible, so we don't have excessive padding).
    //
    // All of this assumes no vectorization.
    // TODO(jreiffers): Handle vectorization here.

    // Emitters always choose `c = 32` if `b` is not a small power of two.
    // Also, reductions are tiled so `b = 32`. The number of threads is always
    // 1024. This satisfies all the constraints above.
    // Reduce the size of the reduction dimension. The maximum size we can
    // handle is the warp size.

    assert(num_threads > minor_reduction.stride);
    int64_t c = std::min(WarpSize(), num_threads / minor_reduction.stride);

    llvm::ArrayRef<int64_t> input_shape = GetInputType(op).getShape();
    auto projected_input_shape = llvm::to_vector(
        input_shape.take_front(minor_reduction.first_dimension));
    projected_input_shape.push_back(minor_reduction.size);
    projected_input_shape.push_back(minor_reduction.stride);
    auto projection_map =
        GetBitcastMap(projected_input_shape, input_shape, ctx);
    int64_t projected_rank = projected_input_shape.size();

    // Pad the new minor dimension to a multiple of c.
    auto padded_projected_input_shape = projected_input_shape;
    int64_t padded_size = RoundUpTo(minor_reduction.size, c);
    padded_projected_input_shape[projected_rank - 2] = padded_size;

    // Reshape the input to [..., a ceildiv c, c, b]
    auto reshaped_input_shape = llvm::to_vector(
        input_shape.take_front(minor_reduction.first_dimension));
    reshaped_input_shape.push_back(padded_size / c);
    reshaped_input_shape.push_back(c);
    reshaped_input_shape.push_back(minor_reduction.stride);
    int64_t reshaped_rank = reshaped_input_shape.size();

    auto reindex_map =
        GetBitcastMap(reshaped_input_shape, padded_projected_input_shape, ctx) *
        projection_map;
    reindex_map.AddConstraint(
        mlir::getAffineDimExpr(reshaped_rank - 2, ctx) +
            mlir::getAffineDimExpr(reshaped_rank - 3, ctx) * c,
        {0, minor_reduction.size - 1});

    auto new_inputs = ReindexTensors(rewriter, op.getInputs(), op.getInits(),
                                     reshaped_input_shape, reindex_map);

    // Reduce the non-minor dimensions and the third to last dimension.
    // [..., a ceildiv c, c, b] -> [..., c, b]
    auto dims_for_first_reduction = llvm::to_vector(
        op.getDimensions().drop_back(minor_reduction.num_dimensions));
    dims_for_first_reduction.push_back(reshaped_rank - 3);
    auto first_reduction =
        rewriter.create<ReduceOp>(op.getLoc(), new_inputs, op.getInits(),
                                  dims_for_first_reduction, op.getCombiner());

    // Transpose [..., c, b] to [..., b, c]
    auto shape = GetOutputType(first_reduction).getShape();
    int64_t first_reduction_rank = shape.size();
    llvm::SmallVector<int64_t, 4> permutation(first_reduction_rank);
    std::iota(permutation.begin(), permutation.end(), 0);
    std::swap(permutation[first_reduction_rank - 1],
              permutation[first_reduction_rank - 2]);

    auto transposed_shape = llvm::to_vector(shape);
    std::swap(transposed_shape[first_reduction_rank - 1],
              transposed_shape[first_reduction_rank - 2]);
    IndexingMap transpose_map(
        mlir::AffineMap::getPermutationMap(permutation, ctx),
        DimVarsFromTensorSizes(transposed_shape), {}, {});

    auto transposed =
        ReindexTensors(rewriter, first_reduction.getResults(), op.getInits(),
                       transposed_shape, transpose_map);

    rewriter.replaceOpWithNewOp<ReduceOp>(
        op, transposed, op.getInits(),
        llvm::ArrayRef<int64_t>{first_reduction_rank - 1}, op.getCombiner());
    return mlir::success();
  }
};

void RewriteReductionsPass::runOnOperation() {
  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<RewriteColumnReduction, RewriteRowReduction>(&getContext());
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
