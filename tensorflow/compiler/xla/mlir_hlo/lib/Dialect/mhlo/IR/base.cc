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

#include "mlir-hlo/Dialect/mhlo/IR/base.h"

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Quant/QuantTypes.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"

// Include order matters
#include "mlir-hlo/Dialect/mhlo/IR/base_attr_interfaces.cc.inc"

namespace mlir {
namespace hlo {

namespace {
Type getExpressedTypeOrSelf(Type type) {
  auto quantType = type.dyn_cast<quant::QuantizedType>();
  return quantType ? quantType.getExpressedType() : type;
}

LogicalResult verifyCompatibleShapeWithBounds(Type type1, Type type2) {
  if (failed(verifyCompatibleShape(type1, type2))) return failure();

  // Verify shapes against bounds
  auto isCompatible = [](ArrayRef<int64_t> shape,
                         BoundedAttrInterface boundedAttr) {
    if (shape.empty() || !boundedAttr) return true;
    auto bounds = boundedAttr.getBounds();
    for (auto [dim_size, bound] : llvm::zip(shape, bounds))  // NOLINT
      if (bound != ShapedType::kDynamicSize && bound < dim_size) return false;
    return true;
  };

  RankedTensorType rankedType1 = type1.dyn_cast<RankedTensorType>();
  RankedTensorType rankedType2 = type2.dyn_cast<RankedTensorType>();
  if (rankedType1 && rankedType2) {
    auto boundedAttr1 =
        rankedType1.getEncoding().dyn_cast_or_null<BoundedAttrInterface>();
    auto boundedAttr2 =
        rankedType2.getEncoding().dyn_cast_or_null<BoundedAttrInterface>();
    return LogicalResult::success(
        isCompatible(rankedType1.getShape(), boundedAttr2) &&
        isCompatible(rankedType2.getShape(), boundedAttr1));
  }
  return success();
}
}  // namespace

bool isCompatibleForHloTypeInference(Type tp1, Type tp2) {
  // Dynamism: We don't require shapes to be the same, we only require them
  // to be compatible, which means that:
  //   1) At least one of the shapes is unranked.
  //   2) Or both shapes have the same rank and their dimensions are compatible,
  //     i.e. for each pair of corresponding dimensions:
  //       2.1) At least one of the dimensions is dynamic,
  //       2.2) Or both dimensions are equal.
  // These relaxed rules simplify the implementation of type inference, allowing
  // ops with partially inferred types to pass verification.
  auto stp1 = tp1.dyn_cast<ShapedType>();
  auto stp2 = tp2.dyn_cast<ShapedType>();
  if (stp1 && stp2) {
    return succeeded(verifyCompatibleShapeWithBounds(stp1, stp2)) &&
           isCompatibleForHloTypeInference(stp1.getElementType(),
                                           stp2.getElementType());
  }

  // Quantization: In the most general case, we allow any combination of
  // quantized/non-quantized across any combination of operands/results,
  // and some differences in quantization parameters across operands/results.
  // Individual ops may introduce additional constraints.
  auto qtp1 = tp1.dyn_cast<quant::QuantizedType>();
  auto qtp2 = tp2.dyn_cast<quant::QuantizedType>();
  if (qtp1 && qtp2) {
    if (qtp1.getStorageType() != qtp2.getStorageType() ||
        qtp1.getStorageTypeMin() != qtp2.getStorageTypeMin() ||
        qtp1.getStorageTypeMax() != qtp2.getStorageTypeMax())
      return false;
  }
  auto etp1 = getExpressedTypeOrSelf(tp1);
  auto etp2 = getExpressedTypeOrSelf(tp2);

  // Sparsity: In the most general case, we allow any combination of
  // sparsity/denseness across any combination of operands/results, as well as
  // differences in sparsity encodings for operands and results.
  // Individual ops may introduce additional constraints.
  // No additional code is needed to check this because of how sparsity is
  // currently implemented.

  // Default case: Unless dynamism, quantization and/or sparsity are involved,
  // the types are required to be exactly equal.
  return etp1 == etp2;
}

LogicalResult deriveShapeFromOperand(
    OpBuilder* builder, Operation* op, Value operand,
    SmallVectorImpl<Value>* reifiedReturnShapes) {
  auto shapedTy = operand.getType().dyn_cast<ShapedType>();
  if (!shapedTy) {
    op->emitOpError() << "operand is not a shaped type";
    return failure();
  }
  reifiedReturnShapes->assign(
      {builder->create<shape::ShapeOfOp>(op->getLoc(), operand)});
  return success();
}

TensorType getSameShapeTensorType(TensorType tensorType, Type elementType) {
  if (auto rankedTensorTy = tensorType.dyn_cast<RankedTensorType>()) {
    return RankedTensorType::get(rankedTensorTy.getShape(), elementType,
                                 rankedTensorTy.getEncoding());
  }
  if (auto unrankedTensorTy = tensorType.dyn_cast<UnrankedTensorType>()) {
    return UnrankedTensorType::get(elementType);
  }
  llvm_unreachable("unhandled type");
}

// TODO(hinsu): Add verification for bounds that it has the same size as rank
// of the tensor and static dimensions don't have bounds.
LogicalResult verifyBounds(ArrayRef<int64_t> /*bounds*/, ShapedType /*type*/,
                           function_ref<InFlightDiagnostic()> /*emitError*/) {
  return success();
}

}  // namespace hlo
}  // namespace mlir
