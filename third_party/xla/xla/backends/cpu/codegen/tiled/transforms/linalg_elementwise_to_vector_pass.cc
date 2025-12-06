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
#include <optional>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/numeric/bits.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/backends/cpu/codegen/tiled/transforms/passes.h"

namespace xla::cpu {

#define GEN_PASS_DECL_LINALGELEMENTWISETOVECTORPASS
#define GEN_PASS_DEF_LINALGELEMENTWISETOVECTORPASS
#include "xla/backends/cpu/codegen/tiled/transforms/passes.h.inc"

namespace {

// TODO(willfroom): Make this dependent on the largest element type / target
// platform.
static constexpr int64_t kMaxVectorDim = 8;

// Check that the given shape is possible to be vectorized with with a single
// dimension vector (or one with which will collapse to a single dimension), the
// minor dimension size is of a natural length (one that fits in registers).
bool IsVectorizable(llvm::ArrayRef<int64_t> shape) {
  if (shape.empty()) {
    return true;
  }

  bool leading_dims_unit =
      absl::c_all_of(shape.drop_back(), [](int64_t size) { return size == 1; });
  if (!leading_dims_unit) {
    return false;
  }
  int64_t minor_dim = shape.back();
  if (mlir::ShapedType::isDynamic(minor_dim) || minor_dim > kMaxVectorDim) {
    return false;
  }

  bool minor_dim_power_of_two = absl::has_single_bit<uint64_t>(minor_dim);
  return minor_dim_power_of_two;
}

// We need to tile the op so that it has a size small enough to fit into vector
// registers. We peel any that would otherwise need masking into smaller vector
// sizes.
class TileElementwiseOp
    : public mlir::OpInterfaceRewritePattern<mlir::linalg::LinalgOp> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::linalg::LinalgOp op,
      mlir::PatternRewriter& rewriter) const override {
    if (!mlir::linalg::isElementwise(op)) {
      return rewriter.notifyMatchFailure(op, "Op is not elementwise");
    }

    auto result_shape = op.getShape(op.getDpsInitOperand(0));

    if (IsVectorizable(result_shape)) {
      return rewriter.notifyMatchFailure(op, "Op is already tiled");
    }

    if (mlir::ShapedType::isDynamic(result_shape.back())) {
      return rewriter.notifyMatchFailure(op,
                                         "Op has a zero-size minor dimension");
    }

    mlir::linalg::LinalgTilingOptions options;
    llvm::SmallVector<int64_t> tile_sizes(result_shape.size() - 1, 1);
    auto vector_size = kMaxVectorDim;
    while (vector_size > result_shape.back()) {
      vector_size /= 2;
    }
    tile_sizes.push_back(vector_size);

    options.setTileSizes(tile_sizes);
    mlir::FailureOr<mlir::linalg::TiledLinalgOp> tile_result =
        mlir::linalg::tileLinalgOp(rewriter, op, options);
    if (failed(tile_result)) {
      return rewriter.notifyMatchFailure(op, "failed to tile");
    }

    rewriter.replaceOp(op, tile_result->tensorResults);

    auto loops =
        llvm::map_to_vector(tile_result->loops, [](mlir::Operation* loop) {
          return mlir::cast<mlir::scf::ForOp>(loop);
        });
    mlir::linalg::peelLoops(rewriter, loops);
    return mlir::success();
  }
};

// Once the op is tiled, we can vectorize it directly.
class ElementwiseToVectorPattern
    : public mlir::OpInterfaceRewritePattern<mlir::linalg::LinalgOp> {
 public:
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::linalg::LinalgOp op,
      mlir::PatternRewriter& rewriter) const override {
    if (!mlir::linalg::isElementwise(op)) {
      return rewriter.notifyMatchFailure(op, "Op is not elementwise");
    }

    // Is this possible?
    if (op.getDpsInits().empty()) {
      return rewriter.notifyMatchFailure(op, "op has no outputs");
    }

    auto result_type =
        mlir::dyn_cast<mlir::ShapedType>(op.getDpsInits().front().getType());
    if (!result_type) {
      return rewriter.notifyMatchFailure(op, "could not convert result type");
    }
    if (!result_type.hasStaticShape()) {
      return rewriter.notifyMatchFailure(op,
                                         "only static shapes are supported");
    }

    if (!IsVectorizable(result_type.getShape())) {
      return rewriter.notifyMatchFailure(op, "Op is not yet tiled");
    }

    mlir::FailureOr<mlir::linalg::VectorizationResult> result =
        mlir::linalg::vectorize(rewriter, op);
    if (failed(result)) {
      return rewriter.notifyMatchFailure(op, "failed to vectorize");
    }
    rewriter.replaceOp(op, result->replacements);
    return mlir::success();
  }
};

struct LinalgElementwiseToVectorPass
    : public impl::LinalgElementwiseToVectorPassBase<
          LinalgElementwiseToVectorPass> {
  void runOnOperation() override {
    mlir::MLIRContext* context = &getContext();
    mlir::RewritePatternSet patterns(context);
    patterns.add<TileElementwiseOp, ElementwiseToVectorPattern>(context);
    mlir::scf::ForOp::getCanonicalizationPatterns(patterns, context);
    mlir::memref::SubViewOp::getCanonicalizationPatterns(patterns, context);
    mlir::vector::TransferReadOp::getCanonicalizationPatterns(patterns,
                                                              context);
    mlir::vector::TransferWriteOp::getCanonicalizationPatterns(patterns,
                                                               context);
    mlir::vector::populateCastAwayVectorLeadingOneDimPatterns(patterns);
    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> CreateLinalgElementwiseToVectorPass() {
  return std::make_unique<LinalgElementwiseToVectorPass>();
}

}  // namespace xla::cpu
