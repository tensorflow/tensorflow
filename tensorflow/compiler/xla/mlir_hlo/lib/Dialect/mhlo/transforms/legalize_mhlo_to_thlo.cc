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

#include <memory>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/PassDetail.h"
#include "mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "mlir-hlo/Dialect/thlo/IR/thlo_ops.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace mhlo {
namespace {

bool isIotaArray(llvm::ArrayRef<int64_t> array, int expectedSize = -1) {
  if (expectedSize != -1 && static_cast<int>(array.size()) != expectedSize)
    return false;
  for (int64_t i = 0, e = array.size(); i < e; ++i) {
    if (i != array[i]) return false;
  }
  return true;
}

struct ConcatenateOpPattern : public OpRewritePattern<mhlo::ConcatenateOp> {
  using OpRewritePattern<mhlo::ConcatenateOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::ConcatenateOp op,
                                PatternRewriter& rewriter) const override {
    const int64_t concatDim = op.dimension();
    const Location loc = op.getLoc();
    const Value anyOperand = op.val().front();

    auto resultTy = op.getResult().getType().cast<RankedTensorType>();
    const ArrayRef<int64_t> resultShape = resultTy.getShape();
    const int64_t rank = resultTy.getRank();

    // Determine init tensor size.
    SmallVector<int64_t> staticInitSizes(resultShape.begin(),
                                         resultShape.end());
    SmallVector<Value> dynamicInitSizes;
    for (int64_t i = 0; i < rank; ++i) {
      // No need to materialize anything for static dimensions.
      if (staticInitSizes[i] != ShapedType::kDynamicSize) {
        continue;
      }

      // For all dimensions other than the concatenation dimension, we can copy
      // the size from any operand.
      if (i != concatDim) {
        dynamicInitSizes.push_back(
            rewriter.create<tensor::DimOp>(loc, anyOperand, i));
        continue;
      }

      // For the concatenation dimensions, sum up the sizes of all operands in
      // that dimension.
      int64_t staticSum = 0;
      Value dynamicSum;
      for (const Value operand : op.val()) {
        auto operandTy = operand.getType().cast<RankedTensorType>();
        if (operandTy.getDimSize(concatDim) == ShapedType::kDynamicSize) {
          const Value dynamicSummand =
              rewriter.create<tensor::DimOp>(loc, operand, concatDim);
          if (dynamicSum) {
            dynamicSum =
                rewriter.create<arith::AddIOp>(loc, dynamicSum, dynamicSummand);
          } else {
            dynamicSum = dynamicSummand;
          }
        } else {
          staticSum += operandTy.getDimSize(concatDim);
        }
      }
      assert(dynamicSum && "expect at least one dynamic summand in this case");
      if (staticSum != 0) {
        dynamicSum = rewriter.create<arith::AddIOp>(
            loc, dynamicSum,
            rewriter.create<arith::ConstantIndexOp>(loc, staticSum));
      }
      dynamicInitSizes.push_back(dynamicSum);
    }

    // Create init tensor and the new concat op.
    auto init = rewriter.create<linalg::InitTensorOp>(
        loc, dynamicInitSizes, staticInitSizes, resultTy.getElementType());
    rewriter.replaceOpWithNewOp<thlo::ConcatenateOp>(op, resultTy, op.val(),
                                                     init, concatDim);
    return success();
  }
};

struct DynamicBroadcastInDimOpPattern
    : public OpRewritePattern<mhlo::DynamicBroadcastInDimOp> {
  using OpRewritePattern<mhlo::DynamicBroadcastInDimOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::DynamicBroadcastInDimOp op,
                                PatternRewriter& rewriter) const override {
    auto loc = op.getLoc();
    Value outputDimensions = op.output_dimensions();
    auto operandTy = op.operand().getType().cast<RankedTensorType>();
    auto resultTy = op.getType().cast<RankedTensorType>();

    // Only  apply to broadcasts that cannot be lowered to linalg, i.e. those
    // for which we do not know their expansion behavior at compile time.
    int64_t countKnownExpansionBehavior = 0;
    if (auto expandingDims = op.known_expanding_dimensions()) {
      countKnownExpansionBehavior += expandingDims->size();
    }
    if (auto nonexpandingDims = op.known_nonexpanding_dimensions()) {
      countKnownExpansionBehavior += nonexpandingDims->size();
    }
    if (operandTy.getRank() == countKnownExpansionBehavior) return failure();

    // Create init tensor as none of the operands are reusable/updatable.
    SmallVector<Value> dynamicDims;
    SmallVector<int64_t> staticShapeInfo;
    for (int i = 0; i < resultTy.getRank(); i++) {
      dynamicDims.push_back(rewriter.create<tensor::ExtractOp>(
          loc, outputDimensions,
          ValueRange{rewriter.create<arith::ConstantIndexOp>(loc, i)}));
      staticShapeInfo.push_back(ShapedType::kDynamicSize);
    }
    auto initTensor = rewriter.create<linalg::InitTensorOp>(
        loc, dynamicDims, staticShapeInfo, resultTy.getElementType());

    // TODO(akuegel): Add a builder for getDenseI64ArrayAttr upstream.
    auto broadcastDims = DenseI64ArrayAttr::get(
        rewriter.getContext(),
        llvm::to_vector(
            llvm::map_range(op.broadcast_dimensions(), [](const auto& d) {
              return static_cast<int64_t>(d.getLimitedValue());
            })));
    DenseI64ArrayAttr knownExpandingDims;
    if (op.known_expanding_dimensions().has_value()) {
      knownExpandingDims = DenseI64ArrayAttr::get(
          rewriter.getContext(),
          llvm::to_vector(llvm::map_range(
              op.known_expanding_dimensionsAttr(), [](const auto& d) {
                return static_cast<int64_t>(d.getLimitedValue());
              })));
    }
    DenseI64ArrayAttr knownNonexpandingDims;
    if (op.known_nonexpanding_dimensions().has_value()) {
      knownNonexpandingDims = DenseI64ArrayAttr::get(
          rewriter.getContext(),
          llvm::to_vector(llvm::map_range(
              op.known_nonexpanding_dimensionsAttr(), [](const auto& d) {
                return static_cast<int64_t>(d.getLimitedValue());
              })));
    }

    rewriter.replaceOpWithNewOp<thlo::DynamicBroadcastInDimOp>(
        op, resultTy, op.operand(), initTensor, broadcastDims,
        knownExpandingDims, knownNonexpandingDims);
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
    rewriter.replaceOpWithNewOp<thlo::GatherOp>(op, op.getType(), op.operand(),
                                                op.start_indices(), initTensor);
    return success();
  }
};

// Rewrites simple scatter patterns.
struct ScatterPattern : public OpRewritePattern<mhlo::ScatterOp> {
  using OpRewritePattern<mhlo::ScatterOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::ScatterOp op,
                                PatternRewriter& rewriter) const override {
    // The variadic case is not supported.
    if (op.updates().size() != 1) return failure();

    // update_computation is sum.
    if (matchUpdateComputation(op.update_computation()).failed())
      return failure();

    const auto& dims = op.scatter_dimension_numbers();
    auto scatterIndicesType =
        op.scatter_indices().getType().dyn_cast<RankedTensorType>();
    if (!scatterIndicesType) return failure();

    // Only point updates are supported.
    //  - update_window_dims is []
    //  - inserted_window_dims is range(operand.shape.rank)
    //  - scatter_dims_to_operand_dims is range(scatter_indices.shape.rank)
    //  - index_vector_dim is scatter_indices.shape.rank-1
    if (!dims.getUpdateWindowDims().empty() ||
        !isIotaArray(dims.getInsertedWindowDims()) ||
        !isIotaArray(dims.getScatterDimsToOperandDims()) ||
        dims.getIndexVectorDim() != scatterIndicesType.getRank() - 1)
      return failure();

    auto opType = op.getType(0).dyn_cast<ShapedType>();
    if (!opType)
      return failure();  // Type is a tensor in the non-variadic case.

    rewriter.replaceOpWithNewOp<thlo::ScatterOp>(
        op, opType, op.scatter_indices(), op.updates().front(),
        op.operands().front());
    return success();
  }

  LogicalResult matchUpdateComputation(mlir::Region& computation) const {
    Block& block = computation.front();
    if (block.getNumArguments() != 2) return failure();

    mhlo::ReturnOp returnOp = dyn_cast<mhlo::ReturnOp>(block.getTerminator());
    if (!returnOp || returnOp.getNumOperands() != 1) return failure();

    auto* returnOperand = returnOp.getOperand(0).getDefiningOp();
    auto addOp = dyn_cast<mhlo::AddOp>(returnOperand);
    if (!addOp || addOp->getNumOperands() != 2) return failure();

    auto lhs = addOp->getOperand(0);
    auto rhs = addOp->getOperand(1);
    auto arg0 = block.getArgument(0);
    auto arg1 = block.getArgument(1);

    return success((lhs == arg0 && rhs == arg1) ||
                   (lhs == arg1 && rhs == arg0));
  }
};

class LegalizeMHLOToTHLOPass
    : public LegalizeMHLOToTHLOPassBase<LegalizeMHLOToTHLOPass> {
  void getDependentDialects(DialectRegistry& registry) const final {
    registry.insert<thlo::THLODialect, linalg::LinalgDialect>();
  }

  void runOnOperation() final {
    MLIRContext* ctx = &getContext();
    RewritePatternSet patterns(ctx);

    // List of patterns.
    // clang-format off
    patterns.insert<
        ConcatenateOpPattern,
        DynamicBroadcastInDimOpPattern,
        GatherPattern,
        ScatterPattern>(ctx);
    // clang-format on

    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createLegalizeMHLOToTHLOPass() {
  return std::make_unique<LegalizeMHLOToTHLOPass>();
}

}  // namespace mhlo
}  // namespace mlir
