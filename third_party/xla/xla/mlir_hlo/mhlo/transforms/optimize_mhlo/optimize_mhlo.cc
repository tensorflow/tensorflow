/* Copyright 2020 The OpenXLA Authors.

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

#include <cstdint>

#include "llvm/ADT/STLExtras.h"
#include "mhlo/IR/hlo_ops.h"
#include "mhlo/transforms/passes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "utils/hlo_utils.h"

namespace mlir {
namespace mhlo {
namespace {

// Returns 1D 64-bit dense elements attribute with the given values.
static DenseIntElementsAttr getI64ElementsAttr(ArrayRef<int64_t> values,
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
    auto dimensionNumbers = gather.getDimensionNumbers();

    // Inputs need to be ranked to lower.
    if (!mlir::cast<ShapedType>(gather.getOperand().getType()).hasRank() ||
        !mlir::cast<ShapedType>(gather.getOperand().getType())
             .hasStaticShape() ||
        !mlir::cast<ShapedType>(gather.getStartIndices().getType()).hasRank() ||
        !mlir::cast<ShapedType>(gather.getStartIndices().getType())
             .hasStaticShape()) {
      return rewriter.notifyMatchFailure(gather,
                                         "non-static operand or start_indices");
    }

    if (dimensionNumbers.getIndexVectorDim() != 0) {
      return rewriter.notifyMatchFailure(gather, "non-zero index_vector_dim");
    }

    // TODO(suderman): Handle start index map != {0}.
    if (dimensionNumbers.getStartIndexMap().empty() ||
        dimensionNumbers.getStartIndexMap().size() != 1 ||
        dimensionNumbers.getStartIndexMap()[0] != 0) {
      return rewriter.notifyMatchFailure(gather,
                                         "start_index_map not empty or [0]");
    }

    auto resultTy =
        mlir::dyn_cast<RankedTensorType>(gather.getResult().getType());

    if (!resultTy) {
      return rewriter.notifyMatchFailure(gather, "unranked result");
    }
    if (static_cast<int64_t>(dimensionNumbers.getOffsetDims().size()) !=
        resultTy.getRank()) {
      return rewriter.notifyMatchFailure(gather,
                                         "offset_dims.size != operand.rank");
    }
    for (const auto& it : llvm::enumerate(dimensionNumbers.getOffsetDims())) {
      if (static_cast<int64_t>(it.index()) != it.value()) {
        return rewriter.notifyMatchFailure(gather,
                                           "offset_dims != [0, result.rank)");
      }
    }

    if (gather.getSliceSizes().size() <= resultTy.getRank()) {
      return rewriter.notifyMatchFailure(gather,
                                         "slices_size.size > result.rank");
    }

    for (const auto& it : llvm::enumerate(resultTy.getShape())) {
      if (gather.getSliceSizes().getValues<int64_t>()[it.index() + 1] !=
          it.value()) {
        return failure();
      }
    }

    auto gatherStartIndices = gather.getStartIndices();
    auto gatherStartIndicesTy =
        mlir::cast<ShapedType>(gatherStartIndices.getType());

    llvm::SmallVector<Value, 4> sliceStartIndices;

    if (gatherStartIndicesTy.getRank() == 0) {
      sliceStartIndices.push_back(gatherStartIndices);
    } else if (gatherStartIndicesTy.getRank() == 1) {
      for (int i = 0; i < gatherStartIndicesTy.getDimSize(0); i++) {
        auto start = getI64ElementsAttr({i}, &rewriter);
        auto limit = getI64ElementsAttr({i + 1}, &rewriter);
        auto stride = getI64ElementsAttr({1}, &rewriter);
        auto indicesSlice = rewriter.create<SliceOp>(
            gather.getLoc(), gatherStartIndices, start, limit, stride);
        auto reshaped = rewriter.create<ReshapeOp>(
            gather.getLoc(),
            RankedTensorType::get({},
                                  mlir::cast<ShapedType>(indicesSlice.getType())
                                      .getElementType()),
            indicesSlice);
        sliceStartIndices.push_back(reshaped);
      }
    } else {
      return rewriter.notifyMatchFailure(gather, "start_indices.rank > 1");
    }

    auto sliceSizesTy = gather.getSliceSizes().getType();

    // Start indices have implicit zeros when not specified. This is because
    // Gather occurs similar to slicing where full slices are inferred. Add any
    // missing zeros as necessary.
    auto zero = rewriter.create<ConstantOp>(
        gather.getLoc(), rewriter.getZeroAttr(RankedTensorType::get(
                             {}, gatherStartIndicesTy.getElementType())));
    while (static_cast<int64_t>(sliceStartIndices.size()) <
           sliceSizesTy.getDimSize(0)) {
      sliceStartIndices.push_back(zero);
    }

    SmallVector<int64_t, 5> sliceShape;
    for (auto shapeValue : gather.getSliceSizes().getValues<APInt>()) {
      sliceShape.push_back(shapeValue.getSExtValue());
    }

    auto sliceTy = RankedTensorType::get(sliceShape, resultTy.getElementType());
    auto slice = rewriter.create<DynamicSliceOp>(
        gather.getLoc(), sliceTy, gather.getOperand(), sliceStartIndices,
        gather.getSliceSizes());

    rewriter.replaceOpWithNewOp<ReshapeOp>(gather, gather.getType(), slice);

    return success();
  }
};

}  // end anonymous namespace

void populateOptimizeMhloPatterns(MLIRContext* context,
                                  RewritePatternSet* patterns) {
  patterns->add<GatherIsSlice>(context);
}
}  // end namespace mhlo
}  // end namespace mlir
