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
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace gml_st {
namespace {

bool isIotaArray(llvm::ArrayRef<int64_t> array, int expectedSize = -1) {
  if (expectedSize != -1 && array.size() != expectedSize) return false;
  for (int i = 0, e = array.size(); i < e; ++i) {
    if (i != array[i]) return false;
  }
  return true;
}

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

    rewriter.replaceOpWithNewOp<gml_st::DynamicBroadcastInDimOp>(
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
    rewriter.replaceOpWithNewOp<gml_st::GatherOp>(
        op, op.getType(), op.operand(), op.start_indices(), initTensor);
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

    rewriter.replaceOpWithNewOp<gml_st::ScatterOp>(
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

class LegalizeMHLOToGMLPass
    : public LegalizeMHLOToGMLPassBase<LegalizeMHLOToGMLPass> {
  void getDependentDialects(DialectRegistry& registry) const final {
    registry.insert<GmlStDialect, linalg::LinalgDialect>();
  }

  void runOnOperation() final {
    MLIRContext* ctx = &getContext();
    RewritePatternSet patterns(ctx);

    // List of patterns.
    patterns
        .insert<DynamicBroadcastInDimOpPattern, GatherPattern, ScatterPattern>(
            ctx);

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
