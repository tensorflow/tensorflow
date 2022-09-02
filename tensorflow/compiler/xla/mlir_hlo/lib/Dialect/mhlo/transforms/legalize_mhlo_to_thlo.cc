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

#include <algorithm>
#include <memory>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "mlir-hlo/Dialect/mhlo/transforms/type_conversion.h"
#include "mlir-hlo/Dialect/thlo/IR/thlo_ops.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace mhlo {

#define GEN_PASS_DEF_LEGALIZEMHLOTOTHLOPASS
#include "mlir-hlo/Dialect/mhlo/transforms/mhlo_passes.h.inc"

namespace {

bool isIotaArray(llvm::ArrayRef<int64_t> array, int expectedSize = -1) {
  if (expectedSize != -1 && static_cast<int>(array.size()) != expectedSize)
    return false;
  for (int64_t i = 0, e = array.size(); i < e; ++i) {
    if (i != array[i]) return false;
  }
  return true;
}

struct ConcatenateOpPattern : public OpConversionPattern<mhlo::ConcatenateOp> {
  using OpConversionPattern<mhlo::ConcatenateOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::ConcatenateOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    const int64_t concatDim = op.dimension();
    const Location loc = op.getLoc();
    const Value anyOperand = adaptor.val().front();

    auto resultTy = typeConverter->convertType(op.getResult().getType())
                        .cast<RankedTensorType>();
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
      if (i != static_cast<int64_t>(concatDim)) {
        dynamicInitSizes.push_back(
            rewriter.create<tensor::DimOp>(loc, anyOperand, i));
        continue;
      }

      // For the concatenation dimensions, sum up the sizes of all operands in
      // that dimension.
      int64_t staticSum = 0;
      Value dynamicSum;
      for (const Value operand : adaptor.val()) {
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
    rewriter.replaceOpWithNewOp<thlo::ConcatenateOp>(
        op, resultTy, adaptor.val(), init, concatDim);
    return success();
  }
};

struct DynamicBroadcastInDimOpPattern
    : public OpConversionPattern<mhlo::DynamicBroadcastInDimOp> {
  using OpConversionPattern<mhlo::DynamicBroadcastInDimOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::DynamicBroadcastInDimOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto loc = op.getLoc();
    Value outputDimensions = adaptor.output_dimensions();
    auto operandTy = adaptor.operand().getType().cast<RankedTensorType>();
    auto resultTy =
        typeConverter->convertType(op.getType()).cast<RankedTensorType>();

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

    auto broadcastDims = rewriter.getDenseI64ArrayAttr(
        llvm::to_vector(op.broadcast_dimensions().getValues<int64_t>()));

    DenseI64ArrayAttr knownExpandingDims;
    if (op.known_expanding_dimensions().has_value()) {
      knownExpandingDims = rewriter.getDenseI64ArrayAttr(llvm::to_vector(
          op.known_expanding_dimensionsAttr().getValues<int64_t>()));
    }
    DenseI64ArrayAttr knownNonexpandingDims;
    if (op.known_nonexpanding_dimensions().has_value()) {
      knownNonexpandingDims = rewriter.getDenseI64ArrayAttr(llvm::to_vector(
          op.known_nonexpanding_dimensionsAttr().getValues<int64_t>()));
    }

    rewriter.replaceOpWithNewOp<thlo::DynamicBroadcastInDimOp>(
        op, resultTy, adaptor.operand(), initTensor, broadcastDims,
        knownExpandingDims, knownNonexpandingDims);
    return success();
  }
};

// Rewrites simple gather patterns (as checked below).
struct GatherPattern : public OpConversionPattern<mhlo::GatherOp> {
  using OpConversionPattern<mhlo::GatherOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::GatherOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto startIndicesType =
        adaptor.start_indices().getType().dyn_cast<RankedTensorType>();
    auto operandType = adaptor.operand().getType().dyn_cast<RankedTensorType>();

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
    auto resultType =
        typeConverter->convertType(op.getType()).cast<RankedTensorType>();
    if (resultType.getNumDynamicDims() > 0) return failure();

    auto loc = op.getLoc();
    auto initTensor = rewriter.create<linalg::InitTensorOp>(
        loc, mlir::ValueRange{}, resultType.getShape(),
        resultType.getElementType());
    rewriter.replaceOpWithNewOp<thlo::GatherOp>(
        op, resultType, adaptor.operand(), adaptor.start_indices(), initTensor);
    return success();
  }
};

static SmallVector<Value, 8> getReduceOpInitTensorDynSizes(
    OpBuilder& b, Location loc, Value operand, int64_t srcRank,
    RankedTensorType resultType, ArrayRef<int64_t> reductionDims) {
  SmallVector<Value, 8> dynShape;
  for (size_t i = 0, j = 0; i < srcRank; ++i) {
    if (j < reductionDims.size() && reductionDims[j] == i) {
      ++j;
      continue;
    }
    size_t resultIndex = i - j;
    if (!resultType.isDynamicDim(resultIndex)) continue;
    dynShape.push_back(b.create<tensor::DimOp>(loc, operand, resultIndex));
  }
  return dynShape;
}

struct ReductionPattern : public OpConversionPattern<mhlo::ReduceOp> {
  using OpConversionPattern<mhlo::ReduceOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::ReduceOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    auto srcRank =
        adaptor.operands()[0].getType().cast<RankedTensorType>().getRank();
    auto reductionDims = llvm::to_vector(op.dimensions().getValues<int64_t>());
    // mhlo.reduce doesn't specify the order of the reduction dimensions.
    std::sort(reductionDims.begin(), reductionDims.end());

    auto toRankedTensor = [](Value v) -> RankedTensorType {
      return v.getType().dyn_cast<RankedTensorType>();
    };

    SmallVector<Value> outputs;
    SmallVector<RankedTensorType> operandTypes, initTypes;
    SmallVector<Type> resultTypes;
    if (failed(typeConverter->convertTypes(op.getResultTypes(), resultTypes)))
      return failure();

    Location loc = op.getLoc();
    for (auto [operand, initValue, resultType] :
         llvm::zip(adaptor.operands(), adaptor.init_values(), resultTypes)) {
      auto initType = toRankedTensor(initValue);
      if (!initType)
        return rewriter.notifyMatchFailure(op,
                                           "expects known-rank init values");
      initTypes.push_back(initType);
      auto operandType = toRankedTensor(initValue);
      if (!operandType)
        return rewriter.notifyMatchFailure(op, "expects known-rank operands");
      operandTypes.push_back(operandType);
      initValue = rewriter.createOrFold<tensor::ExtractOp>(loc, initValue);
      auto tensorResultType = resultType.cast<RankedTensorType>();

      SmallVector<Value, 8> dynShape = getReduceOpInitTensorDynSizes(
          rewriter, loc, operand, srcRank, tensorResultType, reductionDims);
      Value initTensor = rewriter.create<linalg::InitTensorOp>(
          loc, dynShape, tensorResultType.getShape(),
          tensorResultType.getElementType());
      Value filledTensor =
          rewriter.create<linalg::FillOp>(loc, initValue, initTensor).result();
      outputs.push_back(filledTensor);
    }

    auto thloReduction = rewriter.create<thlo::ReductionOp>(
        loc, resultTypes, adaptor.operands(), outputs,
        rewriter.getDenseI64ArrayAttr(reductionDims));
    Region& region = thloReduction.combiner();
    rewriter.inlineRegionBefore(op.body(), region, region.end());

    // Convert the signature of the body. The reduce op 'computation' region
    // apply function has a signature with tensor types, this is converted to a
    // function with element types. E.g. the signature "(tensor<f32>,
    // tensor<f32>) -> tensor<f32>" will be converted to "(f32, f32) -> f32".
    // Also, we need to swap the operands of the function. The mhlo.reduce op
    // expects the init values to be the first parameters of the apply function,
    // while the thlo.reduction op expects the init values as the last
    // parameters of the 'combiner' region apply function.
    TypeConverter::SignatureConversion signatureConverter(
        thloReduction.getNumInputs() * 2);
    assert(thloReduction.getNumInputs() == thloReduction.getNumOutputs());
    for (const auto& [idx, val] : llvm::enumerate(operandTypes)) {
      signatureConverter.addInputs(
          /*origInputNo=*/idx + thloReduction.getNumInputs(),
          // type for new operand number 'idx'.
          typeConverter->convertType(val.getElementType()));
    }
    for (const auto& [idx, val] : llvm::enumerate(initTypes)) {
      signatureConverter.addInputs(
          /*origInputNo=*/idx,
          // type for new operand number 'idx' + thloReduction.getNumInputs()
          typeConverter->convertType(val.getElementType()));
    }
    rewriter.applySignatureConversion(&region, signatureConverter,
                                      getTypeConverter());

    rewriter.replaceOp(op, thloReduction.getResults());
    return success();
  }
};

static bool isInBodyOfThloReduction(Operation* op) {
  auto* parentOp = op->getParentRegion()->getParentOp();
  return isa<thlo::ReductionOp>(*parentOp);
}

// Rewrites a mhlo::ReturnOp inside a thlo::ReductionOp to thlo::YieldOp.
struct ReduceRegionReturnOpConversion
    : public OpConversionPattern<mhlo::ReturnOp> {
  using OpConversionPattern<mhlo::ReturnOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mhlo::ReturnOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    if (!isInBodyOfThloReduction(op)) return failure();
    SmallVector<Value, 4> operands(adaptor.getOperands());
    auto loc = op.getLoc();
    for (size_t i = 0; i < operands.size(); ++i) {
      if (operands[i].getType().isa<ShapedType>()) {
        operands[i] = rewriter.create<tensor::ExtractOp>(loc, operands[i]);
      }
    }
    rewriter.replaceOpWithNewOp<thlo::YieldOp>(op, operands);
    return success();
  }
};

// Rewrites simple scatter patterns.
struct ScatterPattern : public OpConversionPattern<mhlo::ScatterOp> {
  using OpConversionPattern<mhlo::ScatterOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::ScatterOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    // The variadic case is not supported.
    if (op.updates().size() != 1) return failure();

    // update_computation is sum.
    if (matchUpdateComputation(op.update_computation()).failed())
      return failure();

    const auto& dims = op.scatter_dimension_numbers();
    auto scatterIndicesType =
        adaptor.scatter_indices().getType().dyn_cast<RankedTensorType>();
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

    auto opType =
        typeConverter->convertType(op.getType(0)).dyn_cast<ShapedType>();
    if (!opType)
      return failure();  // Type is a tensor in the non-variadic case.

    rewriter.replaceOpWithNewOp<thlo::ScatterOp>(
        op, opType, adaptor.scatter_indices(), adaptor.updates().front(),
        adaptor.operands().front());
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
    : public impl::LegalizeMHLOToTHLOPassBase<LegalizeMHLOToTHLOPass> {
  void getDependentDialects(DialectRegistry& registry) const final {
    registry.insert<thlo::THLODialect, linalg::LinalgDialect,
                    arith::ArithmeticDialect, tensor::TensorDialect>();
  }

  void runOnOperation() final {
    MLIRContext* ctx = &getContext();
    RewritePatternSet patterns(ctx);
    ConversionTarget target(*ctx);
    target.addLegalDialect<thlo::THLODialect, linalg::LinalgDialect,
                           arith::ArithmeticDialect, tensor::TensorDialect>();
    target.addLegalOp<UnrealizedConversionCastOp>();

    auto typeConverter = std::make_unique<LinalgTypeConverter>();

    // List of patterns.
    // clang-format off
    patterns.insert<
        ConcatenateOpPattern,
        DynamicBroadcastInDimOpPattern,
        GatherPattern,
        ReduceRegionReturnOpConversion,
        ReductionPattern,
        ScatterPattern>(*typeConverter, ctx);
    // clang-format on

    if (failed(applyPartialConversion(getOperation(), target,
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
