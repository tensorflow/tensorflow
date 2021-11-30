/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

// This file provides optional optimization patterns for mhlo, canonocalizing
// operations to equivalent but potentially more efficient operations.

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <numeric>

#include "llvm/ADT/STLExtras.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "mlir-hlo/utils/hlo_utils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

using mlir::OwningRewritePatternList;

namespace mlir {
namespace mhlo {
namespace {

// Returns 1D 64-bit dense elements attribute with the given values.
static DenseIntElementsAttr GetI64ElementsAttr(ArrayRef<int64_t> values,
                                               Builder* builder) {
  RankedTensorType ty = RankedTensorType::get(
      {static_cast<int64_t>(values.size())}, builder->getIntegerType(64));
  return DenseIntElementsAttr::get(ty, values);
}

//===----------------------------------------------------------------------===//
// GatherOp
//===----------------------------------------------------------------------===//

class GatherIsSlice : public OpRewritePattern<GatherOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(GatherOp gather,
                                PatternRewriter& rewriter) const override {
    auto dimension_numbers = gather.dimension_numbers();

    // Inputs need to be ranked to lower.
    if (!gather.operand().getType().cast<ShapedType>().hasRank() ||
        !gather.operand().getType().cast<ShapedType>().hasStaticShape() ||
        !gather.start_indices().getType().cast<ShapedType>().hasRank() ||
        !gather.start_indices().getType().cast<ShapedType>().hasStaticShape()) {
      return rewriter.notifyMatchFailure(gather,
                                         "non-static operand or start_indices");
    }

    if (dimension_numbers.getIndexVectorDim() != 0) {
      return rewriter.notifyMatchFailure(gather, "non-zero index_vector_dim");
    }

    // TODO(suderman): Handle start index map != {0}.
    if (dimension_numbers.getStartIndexMap().empty() ||
        dimension_numbers.getStartIndexMap().size() != 1 ||
        dimension_numbers.getStartIndexMap()[0] != 0) {
      return rewriter.notifyMatchFailure(gather,
                                         "start_index_map not empty or [0]");
    }

    auto result_ty = gather.getResult().getType().dyn_cast<RankedTensorType>();

    if (!result_ty) {
      return rewriter.notifyMatchFailure(gather, "unranked result");
    }
    if (dimension_numbers.getOffsetDims().size() != result_ty.getRank()) {
      return rewriter.notifyMatchFailure(gather,
                                         "offset_dims.size != operand.rank");
    }
    for (auto it : llvm::enumerate(dimension_numbers.getOffsetDims())) {
      if (it.index() != it.value()) {
        return rewriter.notifyMatchFailure(gather,
                                           "offset_dims != [0, result.rank)");
      }
    }

    if (gather.slice_sizes().size() <= result_ty.getRank()) {
      return rewriter.notifyMatchFailure(gather,
                                         "slices_size.size > result.rank");
    }

    for (auto it : llvm::enumerate(result_ty.getShape())) {
      if (gather.slice_sizes().getValues<int64_t>()[it.index() + 1] !=
          it.value()) {
        return failure();
      }
    }

    auto gather_start_indices = gather.start_indices();
    auto gather_start_indices_ty =
        gather_start_indices.getType().cast<ShapedType>();

    llvm::SmallVector<Value, 4> slice_start_indices;

    if (gather_start_indices_ty.getRank() == 0) {
      slice_start_indices.push_back(gather_start_indices);
    } else if (gather_start_indices_ty.getRank() == 1) {
      for (int i = 0; i < gather_start_indices_ty.getDimSize(0); i++) {
        auto start = GetI64ElementsAttr({i}, &rewriter);
        auto limit = GetI64ElementsAttr({i + 1}, &rewriter);
        auto stride = GetI64ElementsAttr({1}, &rewriter);
        auto indicesSlice = rewriter.create<SliceOp>(
            gather.getLoc(), gather_start_indices, start, limit, stride);
        auto reshaped = rewriter.create<ReshapeOp>(
            gather.getLoc(),
            RankedTensorType::get(
                {}, indicesSlice.getType().cast<ShapedType>().getElementType()),
            indicesSlice);
        slice_start_indices.push_back(reshaped);
      }
    } else {
      return rewriter.notifyMatchFailure(gather, "start_indices.rank > 1");
    }

    auto sliceSizesTy = gather.slice_sizes().getType();

    // Start indices have implicit zeros when not specified. This is because
    // Gather occurs similar to slicing where full slices are inferred. Add any
    // missing zeros as necessary.
    auto zero = rewriter.create<ConstOp>(
        gather.getLoc(), rewriter.getZeroAttr(RankedTensorType::get(
                             {}, gather_start_indices_ty.getElementType())));
    while (slice_start_indices.size() < sliceSizesTy.getDimSize(0)) {
      slice_start_indices.push_back(zero);
    }

    SmallVector<int64_t, 5> sliceShape;
    for (auto shapeValue : gather.slice_sizes().getValues<APInt>()) {
      sliceShape.push_back(shapeValue.getSExtValue());
    }

    auto sliceTy =
        RankedTensorType::get(sliceShape, result_ty.getElementType());
    auto slice = rewriter.create<DynamicSliceOp>(
        gather.getLoc(), sliceTy, gather.operand(), slice_start_indices,
        gather.slice_sizes());

    rewriter.replaceOpWithNewOp<ReshapeOp>(gather, gather.getType(), slice);

    return success();
  }
};

}  // end anonymous namespace

void PopulateOptimizeMHLOPatterns(MLIRContext* context,
                                  OwningRewritePatternList* patterns) {
  patterns->insert<GatherIsSlice>(context);
}
}  // end namespace mhlo
}  // end namespace mlir
