/* Copyright 2025 The StableHLO Authors.

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

#include <cstdint>
#include <string>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/transforms/PassUtils.h"

namespace mlir {
namespace stablehlo_ext {

#define GEN_PASS_DEF_STABLEHLOLEGALIZEQUANTCOMPOSITEPASS
#include "stablehlo_ext/transforms/passes.h.inc"

namespace {

/**
 * Determines if a given stablehlo::CompositeOp is a supported quantization
 * operation (quant.fake_quant, quant.quantize, or quant.dequantize) for
 * legalization.
 *
 * - **quant.fake_quant:**
 *   - Requires exactly one operand and one result.
 *   - Operand and result types must be identical.
 *   - Result type must be a float type.
 *   - Must have a "dtype" attribute of type TypeAttr.
 *
 * - **quant.quantize:**
 *   - Requires exactly one operand and one result.
 *   - Operand type must be a float type.
 *   - Result type must be an integer type.
 *   - All users of the quantize op must be quant.dequantize composites.
 *
 * - **quant.dequantize:**
 *   - Requires exactly one operand and one result.
 *   - Legalization is supported only if the operand is:
 *     - quant.quantize composite op
 *     - quantized type.
 *   - Operand type must be an integer type or a quant::QuantizedType. The later
 *      case is for the case where the operand is defined by a
 *      stablehlo::UniformQuantizeOp.
 *   - Result type must be a float type.
 *
 * - Any other composite op name is unsupported.
 **/

bool isRewritableQuantizeCompositeOp(stablehlo::CompositeOp op) {
  return op.getName() == "quant.quantize" && op.getNumOperands() == 1 &&
         op.getNumResults() == 1 &&
         isa<FloatType>(getElementTypeOrSelf(op.getOperand(0).getType())) &&
         isa<IntegerType>(getElementTypeOrSelf(op.getType(0)));
}

bool isRewritableDequantizeCompositeOp(stablehlo::CompositeOp op) {
  return op.getName() == "quant.dequantize" && op.getNumOperands() == 1 &&
         op.getNumResults() == 1 &&
         (isa<IntegerType>(getElementTypeOrSelf(op.getOperand(0).getType())) ||
          isa<quant::QuantizedType>(
              getElementTypeOrSelf(op.getOperand(0).getType()))) &&
         isa<FloatType>(getElementTypeOrSelf(op.getType(0)));
}

bool isRewritableFakeQuantCompositeOp(stablehlo::CompositeOp op) {
  return op.getName() == "quant.fake_quant" && op.getNumOperands() == 1 &&
         op.getNumResults() == 1 &&
         op.getOperand(0).getType() == op.getType(0) &&
         isa<FloatType>(getElementTypeOrSelf(op.getType(0))) &&
         llvm::dyn_cast_or_null<TypeAttr>(
             op.getCompositeAttributes().get("dtype"));
}

/**
 * Retrieves quantization attributes from a stablehlo::CompositeOp.
 */
LogicalResult getQuantCompositeAttributes(
    stablehlo::CompositeOp op, SmallVector<double>& scales,
    SmallVector<int64_t>& zeroPoints, int32_t& quantizedDimension,
    int64_t& storageTypeMin, int64_t& storageTypeMax, Type& storageType) {
  // Extract scales
  auto scaleAttr = llvm::dyn_cast_or_null<DenseFPElementsAttr>(
      op.getCompositeAttributes().get("scale"));
  if (scaleAttr == nullptr) {
    return failure();
  }
  scales.reserve(scaleAttr.getNumElements());
  for (auto floatAttr : scaleAttr.getValues<FloatAttr>()) {
    scales.push_back(floatAttr.getValueAsDouble());
  }

  // Extract zero points. If not present, set to 0.
  zeroPoints.resize(scales.size(), 0);
  auto zeroPointAttr = llvm::dyn_cast_or_null<DenseIntElementsAttr>(
      op.getCompositeAttributes().get("zero_point"));
  if (zeroPointAttr) {
    auto tempVec = llvm::to_vector(zeroPointAttr.getValues<int64_t>());
    zeroPoints.assign(tempVec.begin(), tempVec.end());
  }

  // Extract quantized dimension. If not present, set to -1.
  quantizedDimension = -1;
  auto quantDimensionAttr = llvm::dyn_cast_or_null<IntegerAttr>(
      op.getCompositeAttributes().get("quantization_dimension"));
  if (quantDimensionAttr) {
    quantizedDimension = quantDimensionAttr.getValue().getSExtValue();
  }

  // Extract storage type.
  if (op.getName() == "quant.quantize") {
    storageType = getElementTypeOrSelf(op.getResults().front().getType());
  } else if (op.getName() == "quant.dequantize") {
    storageType = getElementTypeOrSelf(op.getInputs().front().getType());
    if (isa<quant::QuantizedType>(storageType)) {
      storageType = cast<quant::QuantizedType>(storageType).getStorageType();
    }
  } else if (op.getName() == "quant.fake_quant") {
    auto dtypeAttr = cast<TypeAttr>(op.getCompositeAttributes().get("dtype"));
    storageType = dtypeAttr.getValue();
  }

  // Extract storage type min and max. If not present, set to default values.
  auto storageTypeMinAttr = llvm::dyn_cast_or_null<IntegerAttr>(
      op.getCompositeAttributes().get("storage_type_min"));
  if (storageTypeMinAttr == nullptr) {
    if (!storageType.isInteger()) {
      return failure();
    }
    storageTypeMin = quant::QuantizedType::getDefaultMinimumForInteger(
        !storageType.isUnsignedInteger(), storageType.getIntOrFloatBitWidth());
  }
  storageTypeMin = storageTypeMinAttr.getValue().getSExtValue();

  auto storageTypeMaxAttr = llvm::dyn_cast_or_null<IntegerAttr>(
      op.getCompositeAttributes().get("storage_type_max"));
  if (storageTypeMaxAttr == nullptr) {
    if (!storageType.isInteger()) {
      return failure();
    }
    storageTypeMax = quant::QuantizedType::getDefaultMaximumForInteger(
        !storageType.isUnsignedInteger(), storageType.getIntOrFloatBitWidth());
  }
  storageTypeMax = storageTypeMaxAttr.getValue().getSExtValue();

  return success();
}

FailureOr<stablehlo::UniformQuantizeOp> buildUniformQuantizeOp(
    stablehlo::CompositeOp op, PatternRewriter& rewriter) {
  SmallVector<double> scales;
  SmallVector<int64_t> zeroPoints;
  int32_t quantizedDimension;
  int64_t storageTypeMin;
  int64_t storageTypeMax;
  std::string dtypeStr;
  Type storageType;

  if (failed(getQuantCompositeAttributes(op, scales, zeroPoints,
                                         quantizedDimension, storageTypeMin,
                                         storageTypeMax, storageType))) {
    return rewriter.notifyMatchFailure(op,
                                       "Failed to get quantization attributes");
  }

  Type expressedType = getElementTypeOrSelf(op.getInputs().front().getType());
  Type quantizedElementType = stablehlo::getQuantizedElementType(
      op.getLoc(), storageType, expressedType, scales, zeroPoints,
      quantizedDimension, storageTypeMin, storageTypeMax);
  RankedTensorType outputQuantizedType = RankedTensorType::get(
      llvm::cast<ShapedType>(op.getResults().front().getType()).getShape(),
      quantizedElementType);
  return stablehlo::UniformQuantizeOp::create(rewriter, op.getLoc(),
                                              outputQuantizedType,
                                              /*input=*/op.getOperand(0));
}

class RewriteDequantizeCompositeOp
    : public OpRewritePattern<stablehlo::CompositeOp> {
  using OpRewritePattern<stablehlo::CompositeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::CompositeOp op,
                                PatternRewriter& rewriter) const final {
    if (op.getName() != "quant.dequantize") {
      return rewriter.notifyMatchFailure(op, "Not a dequantize composite op");
    }
    if (!isRewritableDequantizeCompositeOp(op)) {
      return rewriter.notifyMatchFailure(
          op, "Not a rewritable dequantize composite op");
    }

    SmallVector<double> scales;
    SmallVector<int64_t> zeroPoints;
    int32_t quantizedDimension;
    int64_t storageTypeMin;
    int64_t storageTypeMax;
    std::string dtypeStr;
    Type storageType;

    if (failed(getQuantCompositeAttributes(op, scales, zeroPoints,
                                           quantizedDimension, storageTypeMin,
                                           storageTypeMax, storageType))) {
      return failure();
    }

    // If operand is already quantized, rewrite
    Value quantizedInput = op.getOperand(0);
    if (isa<quant::QuantizedType>(
            getElementTypeOrSelf(op.getOperand(0).getType()))) {
      rewriter.replaceOpWithNewOp<stablehlo::UniformDequantizeOp>(
          op, op.getType(0),
          /*input=*/quantizedInput);
      return success();
    }

    // Otherwise the operand must be a composite op
    auto quantizeCompositeOp =
        quantizedInput.getDefiningOp<stablehlo::CompositeOp>();
    if (!quantizeCompositeOp ||
        !isRewritableQuantizeCompositeOp(quantizeCompositeOp)) {
      return rewriter.notifyMatchFailure(
          op, "Operand is not quantized or a quantize composite op");
    }

    auto quantizeOp = buildUniformQuantizeOp(quantizeCompositeOp, rewriter);
    if (failed(quantizeOp)) {
      return rewriter.notifyMatchFailure(op, "Failed to build quantize op");
    }

    rewriter.replaceOpWithNewOp<stablehlo::UniformDequantizeOp>(
        op, op.getType(0),
        /*input=*/quantizeOp->getResult());
    return success();
  }
};

class RewriteFakeQuantCompositeOp
    : public OpRewritePattern<stablehlo::CompositeOp> {
  using OpRewritePattern<stablehlo::CompositeOp>::OpRewritePattern;

 public:
  explicit RewriteFakeQuantCompositeOp(MLIRContext* context)
      : OpRewritePattern<stablehlo::CompositeOp>(context) {
    setHasBoundedRewriteRecursion();
  }

  LogicalResult matchAndRewrite(stablehlo::CompositeOp op,
                                PatternRewriter& rewriter) const final {
    if (op.getName() != "quant.fake_quant") {
      return rewriter.notifyMatchFailure(op, "Not a fake quant composite op");
    }
    if (!isRewritableFakeQuantCompositeOp(op)) {
      return rewriter.notifyMatchFailure(
          op, "Not a rewritable fake quant composite op");
    }

    SmallVector<double> scales;
    SmallVector<int64_t> zeroPoints;
    int32_t quantizedDimension;
    int64_t storageTypeMin;
    int64_t storageTypeMax;
    std::string dtypeStr;
    Type storageType;

    if (failed(getQuantCompositeAttributes(op, scales, zeroPoints,
                                           quantizedDimension, storageTypeMin,
                                           storageTypeMax, storageType))) {
      return failure();
    }

    Type expressedType = getElementTypeOrSelf(op.getType(0));
    Type quantizedElementType = stablehlo::getQuantizedElementType(
        op.getLoc(), storageType, expressedType, scales, zeroPoints,
        quantizedDimension, storageTypeMin, storageTypeMax);
    RankedTensorType quantizedType = RankedTensorType::get(
        llvm::cast<ShapedType>(op.getType(0)).getShape(), quantizedElementType);
    auto stablehloQuantizeOp = stablehlo::UniformQuantizeOp::create(
        rewriter, op.getLoc(), quantizedType, /*input=*/op.getOperand(0));
    rewriter.replaceOpWithNewOp<stablehlo::UniformDequantizeOp>(
        op, op.getType(0),
        /*input=*/stablehloQuantizeOp.getResult());
    return success();
  }
};

class StablehloLegalizeQuantCompositePass
    : public impl::StablehloLegalizeQuantCompositePassBase<
          StablehloLegalizeQuantCompositePass> {
 public:
  void runOnOperation() override {
    MLIRContext& ctx = getContext();
    RewritePatternSet patterns(&ctx);
    patterns.add<RewriteDequantizeCompositeOp, RewriteFakeQuantCompositeOp>(
        &ctx);

    GreedyRewriteConfig config;
    config.enableFolding(false);
    config.setMaxIterations(3);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config))) {
      getOperation().emitError("Composite lowering pass failed.");
      signalPassFailure();
    }
  }
};

}  // namespace

}  // namespace stablehlo_ext
}  // namespace mlir
