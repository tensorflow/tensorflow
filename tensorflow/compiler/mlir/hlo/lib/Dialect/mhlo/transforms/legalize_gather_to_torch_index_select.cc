/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "mlir-hlo/Dialect/mhlo/transforms/rewriters.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {

namespace mhlo {
namespace {

struct GatherIsTorchIndexSelect : public OpRewritePattern<GatherOp> {
  using OpRewritePattern<GatherOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GatherOp gather,
                                PatternRewriter &rewriter) const override {
    auto start_indices = gather.start_indices();
    auto start_indices_ty = start_indices.getType().cast<ShapedType>();
    if (!start_indices_ty.hasRank()) {
      return failure();
    }

    auto operand = gather.operand();
    auto operand_ty = operand.getType().cast<ShapedType>();
    if (!operand_ty.hasRank()) {
      return failure();
    }

    int64_t index_vector_dim =
        std::max<int64_t>(0, start_indices_ty.getRank() - 1);

    // We can use torch_index_select if the last dimension represents the
    // gather indices.
    auto dimension_numbers = gather.dimension_numbers();
    if (dimension_numbers.index_vector_dim().getValue().getSExtValue() !=
        index_vector_dim) {
      return failure();
    }

    // Index select only works across a single dimension.
    if (!start_indices_ty.getShape().empty() &&
        start_indices_ty.getShape().back() != 1) {
      return failure();
    }

    // Only support the default case for start_index_map.
    if (dimension_numbers.start_index_map().getType().getRank() != 1 ||
        dimension_numbers.start_index_map()
                .getValue(0)
                .cast<IntegerAttr>()
                .getValue() != 0) {
      return failure();
    }

    auto result_ty = gather.getResult().getType().dyn_cast<RankedTensorType>();
    if (!result_ty) {
      return failure();
    }

    // Offset dimensions should be the defaults.
    if (dimension_numbers.offset_dims().getType().getNumElements() !=
        result_ty.getRank() - index_vector_dim) {
      return failure();
    }

    for (auto it : llvm::enumerate(dimension_numbers.offset_dims())) {
      if ((it.index() + index_vector_dim) != it.value()) {
        return failure();
      }
    }

    for (auto it : llvm::enumerate(gather.slice_sizes().getIntValues())) {
      // First shape value must be 1.
      if (it.index() == 0) {
        if (it.value().getSExtValue() != 1) {
          return failure();
        }
        continue;
      }

      // The gather needs to index the entire slice for each other dimension.
      if (it.value().getSExtValue() != operand_ty.getDimSize(it.index())) {
        return failure();
      }
    }

    llvm::SmallVector<int64_t, 4> index_select_shape =
        llvm::to_vector<4>(start_indices_ty.getShape());

    for (auto dim : operand_ty.getShape().drop_front()) {
      index_select_shape.push_back(dim);
    }

    if (!dimension_numbers.collapsed_slice_dims().getType().hasRank() ||
        dimension_numbers.collapsed_slice_dims().getType().getNumElements() !=
            1 ||
        dimension_numbers.collapsed_slice_dims().getValue<int64_t>({0}) != 0) {
      return failure();
    }

    auto torch_index_select = rewriter.create<TorchIndexSelectOp>(
        gather.getLoc(),
        RankedTensorType::get(index_select_shape, operand_ty.getElementType()),
        operand, gather.start_indices(), rewriter.getI64IntegerAttr(0),
        rewriter.getI64IntegerAttr(0));

    rewriter.replaceOpWithNewOp<ReshapeOp>(gather, gather.getType(),
                                           torch_index_select);

    return success();
  }
};

struct LegalizeGatherToTorchIndexSelectPass
    : public PassWrapper<LegalizeGatherToTorchIndexSelectPass, FunctionPass> {
  /// Perform the lowering of standard dialect operations to approximations.
  void runOnFunction() override {
    OwningRewritePatternList patterns;
    PopulateGatherToTorchIndexSelectPatterns(&getContext(), &patterns);
    (void)applyPatternsAndFoldGreedily(getFunction(), std::move(patterns));
  }
};
}  // namespace

void PopulateGatherToTorchIndexSelectPatterns(
    mlir::MLIRContext *context, OwningRewritePatternList *patterns) {
  patterns->insert<GatherIsTorchIndexSelect>(context);
}

std::unique_ptr<FunctionPass> createLegalizeGatherToTorchIndexSelectPass() {
  return std::make_unique<LegalizeGatherToTorchIndexSelectPass>();
}

}  // namespace mhlo
}  // namespace mlir
