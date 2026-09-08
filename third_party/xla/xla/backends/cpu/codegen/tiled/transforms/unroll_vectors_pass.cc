/* Copyright 2026 The OpenXLA Authors.

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
#include <optional>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/VectorInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/backends/cpu/codegen/tiled/transforms/passes.h"  // IWYU pragma: keep

namespace xla::cpu {

#define GEN_PASS_DEF_UNROLLVECTORSPASS
#include "xla/backends/cpu/codegen/tiled/transforms/passes.h.inc"

namespace {

// Returns the target 1D unroll shape for a vector operation by keeping the
// innermost dimension and reducing all leading dimensions to 1.
static std::optional<llvm::SmallVector<int64_t>> GetNativeShape(
    mlir::Operation* op) {
  auto unrollable = mlir::dyn_cast<mlir::VectorUnrollOpInterface>(op);
  if (!unrollable) {
    return std::nullopt;
  }
  auto shape = unrollable.getShapeForUnroll();
  if (!shape || shape->size() <= 1) {
    return std::nullopt;
  }
  if (llvm::all_of(llvm::ArrayRef(*shape).drop_back(),
                   [](int64_t d) { return d == 1; })) {
    return std::nullopt;
  }
  llvm::SmallVector<int64_t> target_shape(shape->size() - 1, 1);
  target_shape.push_back(shape->back());
  return target_shape;
}

// Unrolls `vector.constant_mask` ops, ensuring that if any mask dimension size
// is zero, all mask dimension sizes are set to zero (as required by
// `vector.constant_mask` verifier).
struct UnrollConstantMaskPattern
    : public mlir::OpRewritePattern<mlir::vector::ConstantMaskOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::vector::ConstantMaskOp constant_mask_op,
      mlir::PatternRewriter& rewriter) const override {
    std::optional<llvm::SmallVector<int64_t>> target_shape =
        GetNativeShape(constant_mask_op);
    if (!target_shape) {
      return mlir::failure();
    }

    mlir::VectorType result_type = constant_mask_op.getVectorType();
    std::optional<llvm::SmallVector<int64_t>> original_size =
        constant_mask_op.getShapeForUnroll();
    if (!original_size) {
      return mlir::failure();
    }
    mlir::Location loc = constant_mask_op.getLoc();

    mlir::Value result = mlir::arith::ConstantOp::create(
        rewriter, loc, result_type, rewriter.getZeroAttr(result_type));
    mlir::VectorType target_vector_type =
        mlir::VectorType::get(*target_shape, rewriter.getI1Type());
    llvm::SmallVector<int64_t> strides(target_shape->size(), 1);

    for (const llvm::SmallVector<int64_t>& offsets :
         mlir::StaticTileOffsetRange(*original_size, *target_shape)) {
      llvm::SmallVector<int64_t> unrolled_mask_dims;

      for (auto [i, original_mask_dim] :
           llvm::enumerate(constant_mask_op.getMaskDimSizes())) {
        int64_t adjusted_mask_size =
            std::max(original_mask_dim - offsets[i], static_cast<int64_t>(0));
        int64_t unrolled_mask_dim = std::min(
            adjusted_mask_size, static_cast<int64_t>((*target_shape)[i]));
        unrolled_mask_dims.push_back(unrolled_mask_dim);
      }

      if (llvm::is_contained(unrolled_mask_dims, 0)) {
        std::fill(unrolled_mask_dims.begin(), unrolled_mask_dims.end(), 0);
      }

      auto unrolled_mask = rewriter.createOrFold<mlir::vector::ConstantMaskOp>(
          loc, target_vector_type, unrolled_mask_dims);
      result = rewriter.createOrFold<mlir::vector::InsertStridedSliceOp>(
          loc, unrolled_mask, result, offsets, strides);
    }
    rewriter.replaceOp(constant_mask_op, result);
    return mlir::success();
  }
};

class UnrollVectorsPass
    : public impl::UnrollVectorsPassBase<UnrollVectorsPass> {
 public:
  using UnrollVectorsPassBase::UnrollVectorsPassBase;

  void runOnOperation() override {
    mlir::MLIRContext* context = &getContext();
    mlir::RewritePatternSet patterns(context);
    auto unroll_options =
        mlir::vector::UnrollVectorOptions().setNativeShapeFn(GetNativeShape);
    mlir::vector::populateVectorUnrollPatterns(patterns, unroll_options);
    patterns.add<UnrollConstantMaskPattern>(context, /*benefit=*/2);
    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace
}  // namespace xla::cpu
