/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir-hlo/Dialect/gml_st/IR/gml_st_ops.h"
#include "mlir-hlo/Dialect/gml_st/transforms/pass_detail.h"
#include "mlir-hlo/Dialect/gml_st/transforms/passes.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace gml_st {
namespace {

struct DynamicBroadcastInDimOpPattern
    : public OpRewritePattern<mhlo::DynamicBroadcastInDimOp> {
  using OpRewritePattern<mhlo::DynamicBroadcastInDimOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::DynamicBroadcastInDimOp op,
                                PatternRewriter& rewriter) const override {
    auto loc = op.getLoc();
    Value outputDimensions = op.output_dimensions();
    auto resultTy = op.getType().cast<RankedTensorType>();

    // Create init tensor as none of the operands are reusable/updatable.
    SmallVector<Value> dynamicDims;
    SmallVector<int64_t> staticShapeInfo;
    for (int i = 0; i < resultTy.getRank(); i++) {
      auto iCst = rewriter.create<arith::ConstantIndexOp>(loc, i);
      dynamicDims.push_back(rewriter.create<tensor::ExtractOp>(
          loc, outputDimensions, ValueRange{iCst}));
      staticShapeInfo.push_back(ShapedType::kDynamicSize);
    }
    auto initTensor = rewriter.create<linalg::InitTensorOp>(
        loc, dynamicDims, staticShapeInfo, resultTy.getElementType());

    rewriter.replaceOpWithNewOp<gml_st::DynamicBroadcastInDimOp>(
        op, resultTy, op.operand(), initTensor, op.broadcast_dimensions(),
        op.known_expanding_dimensionsAttr(),
        op.known_nonexpanding_dimensionsAttr());
    return success();
  }
};

// Rewrites simple gather patterns (as checked below).
struct GatherPattern : public OpRewritePattern<mhlo::GatherOp> {
  using OpRewritePattern<mhlo::GatherOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::GatherOp op,
                                PatternRewriter& rewriter) const override {
    auto startIndicesType =
        op.start_indices().getType().dyn_cast<RankedTensorType>();
    auto operandType = op.operand().getType().dyn_cast<RankedTensorType>();

    if (!startIndicesType || !operandType) return failure();

    // index_vector_dim must be the last dimension of start_indices.
    int indexVectorDim = op.dimension_numbers().getIndexVectorDim();
    if (startIndicesType.getRank() - 1 != indexVectorDim) return failure();

    // All slice_sizes must be 1.
    if (!llvm::all_of(op.slice_sizes(), [](auto size) { return size == 1; }))
      return failure();

    // offset_dims must be []
    if (!op.dimension_numbers().getOffsetDims().empty()) return failure();

    // collapsed_slice_dims[] must be range(operand.rank)
    auto collapsedSliceDims = op.dimension_numbers().getCollapsedSliceDims();
    if (!isIotaArray(collapsedSliceDims, operandType.getRank()))
      return failure();

    // start_index_map[] must be range(start_indices.shape[index_vector_dim])
    auto startIndexMap = op.dimension_numbers().getStartIndexMap();
    if (!isIotaArray(startIndexMap,
                     startIndicesType.getShape()[indexVectorDim]))
      return failure();

    // The shape of the result must be statically known.
    if (op.getType().getNumDynamicDims() > 0) return failure();

    auto loc = op.getLoc();
    auto initTensor = rewriter.create<linalg::InitTensorOp>(
        loc, mlir::ValueRange{}, op.getType().getShape(),
        op.getType().getElementType());
    rewriter.replaceOpWithNewOp<gml_st::GatherOp>(
        op, op.getType(), op.operand(), op.start_indices(), initTensor);
    return success();
  }

 private:
  static bool isIotaArray(llvm::ArrayRef<int64_t> array, int expectedSize) {
    if (array.size() != expectedSize) return false;
    for (int i = 0, e = array.size(); i < e; ++i) {
      if (i != array[i]) return false;
    }
    return true;
  }
};

class LegalizeMHLOToGMLPass
    : public LegalizeMHLOToGMLPassBase<LegalizeMHLOToGMLPass> {
  void getDependentDialects(DialectRegistry& registry) const final {
    registry.insert<GmlStDialect, linalg::LinalgDialect>();
  }

  void runOnOperation() final {
    MLIRContext* ctx = &getContext();
    RewritePatternSet patterns(ctx);

    // List of patterns.
    patterns.insert<DynamicBroadcastInDimOpPattern, GatherPattern>(ctx);

    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createLegalizeMHLOToGMLPass() {
  return std::make_unique<LegalizeMHLOToGMLPass>();
}

}  // namespace gml_st
}  // namespace mlir
