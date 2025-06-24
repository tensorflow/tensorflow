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
 *   - All users of the quantize op must be either quant.dequantize composite
 *   ops or func::ReturnOps.
 *
 * - **quant.dequantize:**
 *   - Requires exactly one operand and one result.
 *   - Legalization is supported only if the operand is:
 *     - The only user of a block argument, or
 *     - Defined by a stablehlo::UniformQuantizeOp.
 *   - Operand type must be an integer type or a quant::QuantizedType. The later
 *      case is for the case where the operand is defined by a
 *      stablehlo::UniformQuantizeOp.
 *   - Result type must be a float type.
 *
 * - Any other composite op name is unsupported.
 **/

llvm::LogicalResult isSupportedQuantCompositeOp(stablehlo::CompositeOp op) {
  if (op.getName() != "quant.quantize" && op.getName() != "quant.dequantize" &&
      op.getName() != "quant.fake_quant") {
    return failure();
  }

  if (op.getNumOperands() != 1 || op.getNumResults() != 1) {
    return failure();
  }

  if (op.getName() == "quant.fake_quant") {
    if (op.getOperand(0).getType() != op.getType(0)) {
      return failure();
    }
    if (!isa<FloatType>(getElementTypeOrSelf(op.getType(0)))) {
      return failure();
    }

    auto dtypeAttr = llvm::dyn_cast_or_null<TypeAttr>(
        op.getCompositeAttributes().get("dtype"));
    if (dtypeAttr == nullptr) {
      return failure();
    }
    return success();
  }

  if (op.getName() == "quant.quantize") {
    if (!isa<FloatType>(getElementTypeOrSelf(op.getOperand(0).getType()))) {
      return failure();
    }
    if (!isa<IntegerType>(getElementTypeOrSelf(op.getType(0)))) {
      return failure();
    }
    for (auto* user : op->getUsers()) {
      bool isFedToDequantizeComposite =
          isa<stablehlo::CompositeOp>(user) &&
          cast<stablehlo::CompositeOp>(user).getName() == "quant.dequantize";
      bool isFedToReturnOp = isa<func::ReturnOp>(user);
      if (!isFedToDequantizeComposite && !isFedToReturnOp) {
        return failure();
      }
    }

    return success();
  }

  // op.getName() == "quant.dequantize
  bool isOnlyUserOfBlockArgument =
      isa<BlockArgument>(op.getOperand(0)) &&
      cast<BlockArgument>(op.getOperand(0)).hasOneUse();
  bool isDefinedByQuantizeOp =
      op.getOperand(0).getDefiningOp() != nullptr &&
      mlir::isa<stablehlo::UniformQuantizeOp>(op.getOperand(0).getDefiningOp());
  if (!isOnlyUserOfBlockArgument && !isDefinedByQuantizeOp) {
    return failure();
  }
  if (!isa<IntegerType>(getElementTypeOrSelf(op.getOperand(0).getType())) &&
      !isa<quant::QuantizedType>(
          getElementTypeOrSelf(op.getOperand(0).getType()))) {
    return failure();
  }
  if (!isa<FloatType>(getElementTypeOrSelf(op.getType(0)))) {
    return failure();
  }
  return success();
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

class RewriteQuantizeCompositeOp
    : public OpRewritePattern<stablehlo::CompositeOp> {
  using OpRewritePattern<stablehlo::CompositeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::CompositeOp op,
                                PatternRewriter& rewriter) const final {
    if (op.getName() != "quant.quantize") {
      return failure();
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

    Type expressedType = getElementTypeOrSelf(op.getInputs().front().getType());
    Type quantizedElementType = stablehlo::getQuantizedElementType(
        op.getLoc(), storageType, expressedType, scales, zeroPoints,
        quantizedDimension, storageTypeMin, storageTypeMax);
    RankedTensorType outputQuantizedType = RankedTensorType::get(
        llvm::cast<ShapedType>(op.getResults().front().getType()).getShape(),
        quantizedElementType);
    auto stablehloQuantizeOp = rewriter.create<stablehlo::UniformQuantizeOp>(
        op.getLoc(), outputQuantizedType,
        /*input=*/op.getOperand(0));
    rewriter.replaceAllOpUsesWith(op, stablehloQuantizeOp.getResult());
    return success();
  }
};

class RewriteDequantizeCompositeOp
    : public OpRewritePattern<stablehlo::CompositeOp> {
  using OpRewritePattern<stablehlo::CompositeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::CompositeOp op,
                                PatternRewriter& rewriter) const final {
    if (op.getName() != "quant.dequantize") {
      return failure();
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

    Type expressedType =
        getElementTypeOrSelf(op.getResults().front().getType());
    Type quantizedElementType = stablehlo::getQuantizedElementType(
        op.getLoc(), storageType, expressedType, scales, zeroPoints,
        quantizedDimension, storageTypeMin, storageTypeMax);
    auto quantizedType =
        llvm::cast<ShapedType>(op.getResults().front().getType())
            .clone(quantizedElementType);

    // If the operand of the dequantize compposite op defined by block
    // argument, we need to create a new block argument with the quantized type.
    // Otherwise, we can directly use the composite op's operand.
    Value quantizedInput = op.getOperand(0);
    if (isa<BlockArgument>(op.getOperand(0))) {
      auto funcOp = op->getParentOfType<func::FuncOp>();
      if (funcOp == nullptr) {
        return rewriter.notifyMatchFailure(op,
                                           "Failed to find enclosing function");
      }
      SmallVector<Type> newFuncInputTypes;
      auto funcInputTypes = funcOp.getFunctionType().getInputs();
      int updatedArgIdx = -1;
      for (auto [i, arg] : llvm::enumerate(funcOp.getArguments())) {
        if (arg == quantizedInput) {
          newFuncInputTypes.push_back(quantizedType);
          updatedArgIdx = i;
        } else {
          newFuncInputTypes.push_back(funcInputTypes[i]);
        }
      }
      rewriter.modifyOpInPlace(funcOp, [&]() {
        funcOp.setType(rewriter.getFunctionType(
            newFuncInputTypes, funcOp.getFunctionType().getResults()));
      });
      funcOp.getBody()
          .front()
          .getArgument(updatedArgIdx)
          .setType(quantizedType);
      quantizedInput = funcOp.getBody().front().getArgument(updatedArgIdx);
    }

    auto stablehloDeQuantizeOp =
        rewriter.create<stablehlo::UniformDequantizeOp>(
            op.getLoc(), op.getType(0),
            /*input=*/quantizedInput);
    rewriter.eraseOp(op);
    rewriter.replaceAllOpUsesWith(op, stablehloDeQuantizeOp.getResult());

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
      return failure();
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
    auto stablehloQuantizeOp = rewriter.create<stablehlo::UniformQuantizeOp>(
        op.getLoc(), quantizedType, /*input=*/op.getOperand(0));
    auto stablehloDeQuantizeOp =
        rewriter.create<stablehlo::UniformDequantizeOp>(
            op.getLoc(), op.getType(0),
            /*input=*/stablehloQuantizeOp.getResult());
    rewriter.replaceAllOpUsesWith(op, stablehloDeQuantizeOp.getResult());
    return success();
  }
};

/**
 * When there is a quantize op at the output, the return op's operand is a
 * quantized tensor. However, the function's return type is still a simple
 * integer. This pattern makes sure the function's signature is updated so
 * that it's return type conforms the operand of its return op.
 */
struct UpdateFunctionTypePattern : public OpRewritePattern<func::ReturnOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(func::ReturnOp op,
                                PatternRewriter& rewriter) const override {
    auto funcOp = op->getParentOfType<func::FuncOp>();
    if (funcOp == nullptr) {
      return rewriter.notifyMatchFailure(op,
                                         "Failed to find enclosing function");
    }
    funcOp.setType(rewriter.getFunctionType(funcOp.getArgumentTypes(),
                                            op.getOperandTypes()));
    return success();
  }
};

class StablehloLegalizeQuantCompositePass
    : public impl::StablehloLegalizeQuantCompositePassBase<
          StablehloLegalizeQuantCompositePass> {
 public:
  void runOnOperation() override {
    MLIRContext& ctx = getContext();
    auto module = getOperation();

    RewritePatternSet patterns(&ctx);
    patterns.add<RewriteQuantizeCompositeOp, RewriteDequantizeCompositeOp,
                 RewriteFakeQuantCompositeOp>(&ctx);

    ConversionTarget target(getContext());
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<quant::QuantDialect>();

    // Declare all the MHLO ops as legal except for the quantization
    // composites we want to lower.
    target.addDynamicallyLegalDialect<stablehlo::StablehloDialect>(
        [](Operation* op) {
          auto compositeOp = dyn_cast_or_null<stablehlo::CompositeOp>(op);
          if (!compositeOp) {
            return true;
          }
          return failed(isSupportedQuantCompositeOp(compositeOp));
        });

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      getOperation().emitError("Composite lowering pass failed.");
      signalPassFailure();
    }

    GreedyRewriteConfig greedyRewriteConfig;
    RewritePatternSet cleanupPatterns(&ctx);
    cleanupPatterns.add<UpdateFunctionTypePattern>(&ctx);

    (void)applyPatternsGreedily(module, std::move(cleanupPatterns),
                                greedyRewriteConfig);
  }
};

}  // namespace

}  // namespace stablehlo_ext
}  // namespace mlir
