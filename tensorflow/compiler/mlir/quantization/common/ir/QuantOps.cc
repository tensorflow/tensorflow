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

#include "tensorflow/compiler/mlir/quantization/common/ir/QuantOps.h"

#include <numeric>

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Quant/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/common/ir/QuantOpsDialect.cc.inc"

namespace mlir::quant::ir {

using mlir::quant::QuantizedType;

void QuantDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "tensorflow/compiler/mlir/quantization/common/ir/QuantOps.cc.inc"
      >();
}

OpFoldResult StorageCastOp::fold(FoldAdaptor) {
  // Matches x -> [scast -> scast] -> y, replacing the second scast with the
  // value of x if the casts invert each other.
  auto srcScastOp = getArg().getDefiningOp<StorageCastOp>();
  if (!srcScastOp || srcScastOp.getArg().getType() != getType())
    return OpFoldResult();
  return srcScastOp.getArg();
}

/// The quantization specification should match the expressed type.
static bool isValidQuantizationSpec(Attribute quantSpec, Type expressed) {
  if (auto typeAttr = mlir::dyn_cast<TypeAttr>(quantSpec)) {
    Type spec = typeAttr.getValue();
    if (mlir::isa<TensorType, VectorType>(spec)) return false;

    // The spec should be either a quantized type which is compatible to the
    // expressed type, or a primitive type which is as same as the
    // (element type of) the expressed type.
    if (auto quantizedType = mlir::dyn_cast<QuantizedType>(spec))
      return quantizedType.isCompatibleExpressedType(expressed);

    if (auto tensorType = mlir::dyn_cast<TensorType>(expressed))
      return spec == tensorType.getElementType();

    if (auto vectorType = mlir::dyn_cast<VectorType>(expressed))
      return spec == vectorType.getElementType();
  }
  return false;
}

LogicalResult QuantizeRegionOp::verify() {
  // There are specifications for both inputs and outputs.
  if (getNumOperands() != getInputSpecs().size() ||
      getNumResults() != getOutputSpecs().size())
    return emitOpError(
        "has unmatched operands/results number and spec attributes number");

  // Verify that quantization specifications are valid.
  for (auto input : llvm::zip(getOperandTypes(), getInputSpecs())) {
    Type inputType = std::get<0>(input);
    Attribute inputSpec = std::get<1>(input);
    if (!isValidQuantizationSpec(inputSpec, inputType)) {
      return emitOpError() << "has incompatible specification " << inputSpec
                           << " and input type " << inputType;
    }
  }

  for (auto result : llvm::zip(getResultTypes(), getOutputSpecs())) {
    Type outputType = std::get<0>(result);
    Attribute outputSpec = std::get<1>(result);
    if (!isValidQuantizationSpec(outputSpec, outputType)) {
      return emitOpError() << "has incompatible specification " << outputSpec
                           << " and output type " << outputType;
    }
  }
  return success();
}

LogicalResult StatisticsOp::verify() {
  auto tensorArg = mlir::dyn_cast<TensorType>(getArg().getType());
  if (!tensorArg) return emitOpError("arg needs to be tensor type.");

  // Verify layerStats attribute.
  {
    auto layerStatsType = getLayerStats().getShapedType();
    if (!mlir::isa<FloatType>(layerStatsType.getElementType())) {
      return emitOpError("layerStats must have a floating point element type");
    }
    if (layerStatsType.getRank() != 1 || layerStatsType.getDimSize(0) != 2) {
      return emitOpError("layerStats must have shape [2]");
    }
  }
  // Verify axisStats (optional) attribute.
  if (getAxisStats()) {
    if (!getAxis()) return emitOpError("axis must be specified for axisStats");

    auto shape = tensorArg.getShape();
    auto argSliceSize =
        std::accumulate(std::next(shape.begin(), *getAxis()), shape.end(), 1,
                        std::multiplies<int64_t>());

    auto axisStatsType = getAxisStats()->getShapedType();
    if (!mlir::isa<FloatType>(axisStatsType.getElementType())) {
      return emitOpError("axisStats must have a floating point element type");
    }
    if (axisStatsType.getRank() != 2 || axisStatsType.getDimSize(1) != 2 ||
        axisStatsType.getDimSize(0) != argSliceSize) {
      return emitOpError(
          "axisStats must have shape [N,2] "
          "where N = the slice size defined by the axis dim");
    }
  }
  return success();
}

}  // namespace mlir::quant::ir

#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/quantization/common/ir/QuantOps.cc.inc"
