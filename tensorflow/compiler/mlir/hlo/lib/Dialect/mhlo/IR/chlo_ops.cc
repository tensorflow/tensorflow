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

#include "mlir-hlo/Dialect/mhlo/IR/chlo_ops.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir-hlo/utils/broadcast_utils.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

namespace mlir {
namespace chlo {

Value getConstantLikeMaxFiniteValue(OpBuilder& b, Location loc, Value val) {
  auto ty = getElementTypeOrSelf(val.getType()).cast<FloatType>();
  return getConstantLike(
      b, loc, llvm::APFloat::getLargest(ty.getFloatSemantics()), val);
}

Value getConstantLikeInfValue(OpBuilder& b, Location loc, Value val,
                              bool negative) {
  auto ty = getElementTypeOrSelf(val.getType()).cast<FloatType>();
  return getConstantLike(
      b, loc, llvm::APFloat::getInf(ty.getFloatSemantics(), negative), val);
}

Value getConstantLikeSmallestFiniteValue(OpBuilder& b, Location loc,
                                         Value val) {
  auto ty = getElementTypeOrSelf(val.getType()).cast<FloatType>();
  return getConstantLike(
      b, loc, llvm::APFloat::getSmallest(ty.getFloatSemantics()), val);
}

Value getConstantLike(OpBuilder& b, Location loc, const APFloat& constant,
                      Value val) {
  Type ty = getElementTypeOrSelf(val.getType());
  return b.create<ConstantLikeOp>(loc, b.getFloatAttr(ty, constant), val);
}

//===----------------------------------------------------------------------===//
// CompatibleOperandsAndResultType
//===----------------------------------------------------------------------===//

// TODO(b/231358795): Review the use of InferTypeOpInterface for ops that
// support quantization or sparsity.
#define INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(Op)                        \
  LogicalResult Op::inferReturnTypeComponents(                                \
      MLIRContext* context, Optional<Location> location,                      \
      ValueShapeRange operands, DictionaryAttr attributes,                    \
      RegionRange regions,                                                    \
      SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {          \
    return inferReturnTypeComponentsFromOperands(context, location, operands, \
                                                 attributes, regions,         \
                                                 inferredReturnShapes);       \
  }

INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(AcosOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(AcoshOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(AsinOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(AsinhOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(AtanOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(AtanhOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(ConjOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(CoshOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(DigammaOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(ErfOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(ErfcOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(LgammaOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(NextAfterOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(PolygammaOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(SinhOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(TanOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(ZetaOp)

//===----------------------------------------------------------------------===//
// BinaryOps
//===----------------------------------------------------------------------===//

namespace {
// Gets the resulting type from a broadcast between two types.
ShapedTypeComponents getBroadcastType(
    Type x, Type y, Type elementType,
    DenseIntElementsAttr broadcastDimensionsAttr) {
  auto xRanked = x.dyn_cast<RankedTensorType>();
  auto yRanked = y.dyn_cast<RankedTensorType>();
  if (!xRanked || !yRanked) {
    return {elementType};
  }

  auto shapeX = xRanked.getShape();
  auto shapeY = yRanked.getShape();

  // If no broadcast dimensions, assume "numpy" broadcasting.
  if (shapeX.size() == shapeY.size() || !broadcastDimensionsAttr) {
    llvm::SmallVector<int64_t, 4> outShape;
    if (!mlir::OpTrait::util::getBroadcastedShape(shapeX, shapeY, outShape)) {
      // Signal illegal broadcast_dimensions as unranked.
      return {elementType};
    }
    return {outShape, elementType};
  }

  auto shapeLarge = shapeX.size() > shapeY.size() ? shapeX : shapeY;
  auto shapeSmall = shapeX.size() <= shapeY.size() ? shapeX : shapeY;

  auto broadcastDimensions = broadcastDimensionsAttr.getValues<APInt>();
  if (broadcastDimensions.size() != shapeSmall.size()) {
    // Signal illegal broadcast_dimensions as unranked.
    return {elementType};
  }

  llvm::SmallVector<int64_t, 4> shapeLargeFiltered;
  shapeLargeFiltered.reserve(shapeSmall.size());
  for (const auto& dim : broadcastDimensions) {
    if (dim.getZExtValue() >= shapeLarge.size()) return {elementType};
    shapeLargeFiltered.push_back(shapeLarge[dim.getZExtValue()]);
  }
  llvm::SmallVector<int64_t, 4> outShapeFiltered;
  if (!mlir::OpTrait::util::getBroadcastedShape(shapeSmall, shapeLargeFiltered,
                                                outShapeFiltered)) {
    // Signal illegal broadcast_dimensions as unranked.
    return {elementType};
  }

  // Update according to the broadcast dimensions.
  llvm::SmallVector<int64_t, 4> outShape(shapeLarge.begin(), shapeLarge.end());
  for (const auto& indexPair : llvm::enumerate(broadcastDimensions)) {
    auto newValue = outShapeFiltered[indexPair.index()];
    outShape[indexPair.value().getZExtValue()] = newValue;
  }

  return {outShape, elementType};
}

LogicalResult InferBroadcastBinaryOpReturnTypeComponents(
    MLIRContext* context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, Type elementType,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  // Find broadcast_dimensions.
  DenseIntElementsAttr broadcastDimensions =
      attributes.get("broadcast_dimensions")
          .dyn_cast_or_null<DenseIntElementsAttr>();

  ShapedType lhsType = operands[0].getType().dyn_cast<ShapedType>();
  ShapedType rhsType = operands[1].getType().dyn_cast<ShapedType>();
  if (!lhsType || !rhsType ||
      lhsType.getElementType() != rhsType.getElementType()) {
    return emitOptionalError(location, "mismatched operand types");
  }
  if (!elementType) elementType = lhsType.getElementType();
  inferredReturnShapes.push_back(
      getBroadcastType(lhsType, rhsType, elementType, broadcastDimensions));
  return success();
}

LogicalResult ReifyBroadcastBinaryOpReturnTypeShapes(
    OpBuilder& builder, Operation* op, ValueRange operands,
    SmallVectorImpl<Value>& result) {
  assert(operands.size() == 2 && "expect binary op");
  auto loc = op->getLoc();
  auto lhs = operands[0];
  auto rhs = operands[1];

  // Check for "numpy"-style rank broadcast.
  auto broadcastDimensions = op->getAttr("broadcast_dimensions")
                                 .dyn_cast_or_null<DenseIntElementsAttr>();
  if (broadcastDimensions &&
      !hlo::IsLegalNumpyRankedBroadcast(lhs, rhs, broadcastDimensions)) {
    // Note: It is unclear whether the general specification of explicit
    // broadcast_dimensions on binary ops is a feature we want to carry
    // forward. While it can technically be implemented for ranked-dynamic,
    // it is incompatible with unranked inputs. If this warning is emitted
    // in real programs, it is an indication that the feature should be
    // implemented versus just falling back on the more standard definition
    // of numpy-like prefix-padding.
    return op->emitWarning()
           << "unsupported non prefix-padded dynamic rank "
           << "broadcast_dimensions = " << broadcastDimensions;
  }

  result.push_back(hlo::ComputeBinaryElementwiseBroadcastingResultExtents(
      loc, lhs, rhs, builder));
  return success();
}
}  // namespace

//===----------------------------------------------------------------------===//
// BroadcastComplexOp (has custom type inference due to different result type).
//===----------------------------------------------------------------------===//

LogicalResult BroadcastComplexOp::inferReturnTypeComponents(
    MLIRContext* context, Optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, RegionRange /*regions*/,
    SmallVectorImpl<ShapedTypeComponents>& inferedReturnShapes) {
  ShapedType lhsType = operands[0].getType().dyn_cast<ShapedType>();
  if (!lhsType) {
    return emitOptionalError(location, "expected ShapedType");
  }
  Type elementType = ComplexType::get(lhsType.getElementType());
  return InferBroadcastBinaryOpReturnTypeComponents(context, location, operands,
                                                    attributes, elementType,
                                                    inferedReturnShapes);
}
LogicalResult BroadcastComplexOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  return ReifyBroadcastBinaryOpReturnTypeShapes(builder, getOperation(),
                                                operands, reifiedReturnShapes);
}

//===----------------------------------------------------------------------===//
// BroadcastCompareOp (has custom type inference due to different result type).
//===----------------------------------------------------------------------===//

void BroadcastCompareOp::build(OpBuilder& builder, OperationState& result,
                               Value lhs, Value rhs,
                               DenseIntElementsAttr broadcastDimensions,
                               mhlo::ComparisonDirection comparisonDirection,
                               mhlo::ComparisonType compareType) {
  build(builder, result, lhs, rhs, broadcastDimensions,
        mhlo::ComparisonDirectionAttr::get(builder.getContext(),
                                           comparisonDirection),
        mhlo::ComparisonTypeAttr::get(builder.getContext(), compareType));
}

LogicalResult BroadcastCompareOp::inferReturnTypeComponents(
    MLIRContext* context, Optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, RegionRange /*regions*/,
    SmallVectorImpl<ShapedTypeComponents>& inferedReturnShapes) {
  Type elementType = IntegerType::get(context, 1);
  return InferBroadcastBinaryOpReturnTypeComponents(context, location, operands,
                                                    attributes, elementType,
                                                    inferedReturnShapes);
}

LogicalResult BroadcastCompareOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  return ReifyBroadcastBinaryOpReturnTypeShapes(builder, getOperation(),
                                                operands, reifiedReturnShapes);
}

//===----------------------------------------------------------------------===//
// IsInfOp
//===----------------------------------------------------------------------===//

static Type getIsInfLikeReturnType(Value operand) {
  Builder b(operand.getContext());
  return mhlo::getSameShapeTensorType(operand.getType().cast<TensorType>(),
                                      b.getI1Type());
}

LogicalResult IsInfOp::inferReturnTypes(
    MLIRContext* /*ctx*/, Optional<Location>, ValueRange operands,
    DictionaryAttr, RegionRange, SmallVectorImpl<Type>& inferredReturnTypes) {
  inferredReturnTypes.push_back(getIsInfLikeReturnType(operands.front()));
  return success();
}

//===----------------------------------------------------------------------===//
// IsNegInfOp
//===----------------------------------------------------------------------===//

LogicalResult IsNegInfOp::inferReturnTypes(
    MLIRContext* /*ctx*/, Optional<Location>, ValueRange operands,
    DictionaryAttr, RegionRange, SmallVectorImpl<Type>& inferredReturnTypes) {
  inferredReturnTypes.push_back(getIsInfLikeReturnType(operands.front()));
  return success();
}

//===----------------------------------------------------------------------===//
// IsPosInfOp
//===----------------------------------------------------------------------===//

LogicalResult IsPosInfOp::inferReturnTypes(
    MLIRContext* /*ctx*/, Optional<Location>, ValueRange operands,
    DictionaryAttr, RegionRange, SmallVectorImpl<Type>& inferredReturnTypes) {
  inferredReturnTypes.push_back(getIsInfLikeReturnType(operands.front()));
  return success();
}

//===----------------------------------------------------------------------===//
// Macros for method definitions that are common to most broadcasting ops.
//===----------------------------------------------------------------------===//

#define BROADCAST_BINARY_OP_DEFS(Op)                                       \
  LogicalResult Op::inferReturnTypeComponents(                             \
      MLIRContext* context, Optional<Location> location,                   \
      ValueShapeRange operands, DictionaryAttr attributes,                 \
      RegionRange regions,                                                 \
      SmallVectorImpl<ShapedTypeComponents>& inferedReturnShapes) {        \
    return InferBroadcastBinaryOpReturnTypeComponents(                     \
        context, location, operands, attributes, /*element_type=*/nullptr, \
        inferedReturnShapes);                                              \
  }                                                                        \
  LogicalResult Op::reifyReturnTypeShapes(                                 \
      OpBuilder& builder, ValueRange operands,                             \
      SmallVectorImpl<Value>& reifiedReturnShapes) {                       \
    return ReifyBroadcastBinaryOpReturnTypeShapes(                         \
        builder, getOperation(), operands, reifiedReturnShapes);           \
  }

BROADCAST_BINARY_OP_DEFS(BroadcastAddOp);
BROADCAST_BINARY_OP_DEFS(BroadcastAndOp);
BROADCAST_BINARY_OP_DEFS(BroadcastAtan2Op);
BROADCAST_BINARY_OP_DEFS(BroadcastDivOp);
BROADCAST_BINARY_OP_DEFS(BroadcastMaxOp);
BROADCAST_BINARY_OP_DEFS(BroadcastMinOp);
BROADCAST_BINARY_OP_DEFS(BroadcastMulOp);
BROADCAST_BINARY_OP_DEFS(BroadcastNextAfterOp);
BROADCAST_BINARY_OP_DEFS(BroadcastOrOp);
BROADCAST_BINARY_OP_DEFS(BroadcastPolygammaOp);
BROADCAST_BINARY_OP_DEFS(BroadcastPowOp);
BROADCAST_BINARY_OP_DEFS(BroadcastRemOp);
BROADCAST_BINARY_OP_DEFS(BroadcastShiftLeftOp);
BROADCAST_BINARY_OP_DEFS(BroadcastShiftRightArithmeticOp);
BROADCAST_BINARY_OP_DEFS(BroadcastShiftRightLogicalOp);
BROADCAST_BINARY_OP_DEFS(BroadcastSubOp);
BROADCAST_BINARY_OP_DEFS(BroadcastXorOp);
BROADCAST_BINARY_OP_DEFS(BroadcastZetaOp);

#undef BROADCAST_BINARY_OP_DEFS

LogicalResult ConstantLikeOp::verify() {
  if (value().getType() != getType().cast<ShapedType>().getElementType())
    return emitOpError() << "value's type doesn't match element return type";
  return success();
}

//===----------------------------------------------------------------------===//
// MinimumBroadcastShapesOp
//===----------------------------------------------------------------------===//
LogicalResult MinimumBroadcastShapesOp::verify() {
  // Check that the number of operands matches the number of outputs.
  unsigned resultShapesCount = results().size();
  unsigned operandShapesCount = shapes().size();
  if (operandShapesCount != resultShapesCount) {
    return emitOpError() << "number of operand shapes (" << operandShapesCount
                         << ") does not match number of result shapes ("
                         << resultShapesCount << ")";
  }
  if (operandShapesCount < 2) {
    return emitOpError() << "number of operand shapes (" << operandShapesCount
                         << ") should be >= 2";
  }
  return success();
}

LogicalResult ConstantLikeOp::inferReturnTypeComponents(
    MLIRContext* /*context*/, Optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes,
    RegionRange /*regions*/,
    SmallVectorImpl<ShapedTypeComponents>& inferedReturnShapes) {
  ConstantLikeOp::Adaptor op(operands, attributes);
  if (failed(op.verify(location.getValue()))) return failure();
  Type elementType = op.value().getType();
  Type operandType = op.operand().getType();
  if (operandType.isa<UnrankedTensorType>()) {
    inferedReturnShapes.emplace_back(elementType);
  } else {
    const auto& shape = operandType.cast<RankedTensorType>().getShape();
    inferedReturnShapes.emplace_back(shape, elementType);
  }
  return success();
}

LogicalResult ConstantLikeOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  return ::mlir::mhlo::deriveShapeFromOperand(
      &builder, getOperation(), operands.front(), &reifiedReturnShapes);
}

OpFoldResult ConstantLikeOp::fold(ArrayRef<Attribute> /*operands*/) {
  auto opType = operand().getType().cast<ShapedType>();
  if (!opType.hasStaticShape()) return {};
  auto type = RankedTensorType::get(opType.getShape(), value().getType());
  return DenseElementsAttr::get(type, value());
}

LogicalResult BroadcastSelectOp::inferReturnTypeComponents(
    MLIRContext*, Optional<Location> location, ValueShapeRange operands,
    DictionaryAttr, RegionRange,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  BroadcastSelectOp::Adaptor op(operands.getValues());
  auto predType = op.pred().getType().dyn_cast<ShapedType>();
  auto onTrueType = op.on_true().getType().dyn_cast<ShapedType>();
  auto onFalseType = op.on_false().getType().dyn_cast<ShapedType>();

  if (!predType || !onTrueType || !onFalseType ||
      onTrueType.getElementType() != onFalseType.getElementType()) {
    return emitOptionalError(location, "mismatched operand types");
  }

  Type elementType = onTrueType.getElementType();

  // Compute the result shape as two binary broadcasts.
  ShapedTypeComponents& components = inferredReturnShapes.emplace_back(
      getBroadcastType(onTrueType, onFalseType, elementType, nullptr));
  if (components.hasRank()) {
    components = getBroadcastType(
        RankedTensorType::get(components.getDims(), elementType), predType,
        elementType, nullptr);
  }
  return success();
}

LogicalResult BroadcastSelectOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands, SmallVectorImpl<Value>& result) {
  result.push_back(hlo::ComputeNaryElementwiseBroadcastingResultExtents(
      getLoc(), operands, builder));
  return success();
}

//===----------------------------------------------------------------------===//
// RankSpecializationClusterOp
//===----------------------------------------------------------------------===//

void RankSpecializationClusterOp::getSuccessorRegions(
    Optional<unsigned> index, ArrayRef<Attribute> /*operands*/,
    SmallVectorImpl<RegionSuccessor>& regions) {
  // RankSpecializationClusterOp has unconditional control flows into the region
  // and back to the parent, so return the correct RegionSuccessor purely based
  // on the index being None or 0.
  if (index.hasValue()) {
    regions.push_back(RegionSuccessor(getResults()));
    return;
  }
  regions.push_back(RegionSuccessor(&body()));
}

LogicalResult RankSpecializationClusterOp::verify() {
  if (body().getArgumentTypes() != getOperandTypes())
    return emitOpError() << "block argument types must match operand types";

  // All operands of nested ops must be defined in the body or declared by the
  // cluster.
  Block* body = getBody();
  for (Operation& nested : body->without_terminator()) {
    if (!llvm::all_of(nested.getOpOperands(), [&](OpOperand& operand) {
          Operation* def = operand.get().getDefiningOp();
          if (def != nullptr && def->getBlock() == body) return true;
          return llvm::is_contained(body->getArguments(), operand.get());
        })) {
      return emitOpError() << "nested ops must not depend on implicit operands";
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// TopKOp
//===----------------------------------------------------------------------===//

LogicalResult TopKOp::inferReturnTypeComponents(
    MLIRContext* context, Optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  Builder builder(context);
  TopKOp::Adaptor adaptor(operands, attributes, regions);
  Value operand = adaptor.operand();
  uint64_t k = adaptor.k();

  auto operandTy = operand.getType().dyn_cast<RankedTensorType>();
  if (!operandTy) {
    return emitOptionalError(location, "operand must be ranked");
  }
  if (operandTy.getRank() < 1) {
    return emitOptionalError(location, "operand's rank must be at least 1");
  }
  auto operandLastDim = operandTy.getShape()[operandTy.getRank() - 1];
  if (operandLastDim == ShapedType::kDynamicSize) {
    return emitOptionalError(location,
                             "operand's last dimension must be static");
  }
  if (operandLastDim < k) {
    return emitOptionalError(location,
                             "operand's last dimension must be at least ", k);
  }

  SmallVector<int64_t> resultShape;
  append_range(resultShape, operandTy.getShape());
  resultShape[operandTy.getRank() - 1] = k;

  inferredReturnShapes.emplace_back(resultShape, operandTy.getElementType());
  inferredReturnShapes.emplace_back(resultShape, builder.getI32Type());
  return success();
}

}  // namespace chlo
}  // namespace mlir

#define GET_OP_CLASSES
#include "mlir-hlo/Dialect/mhlo/IR/chlo_ops.cc.inc"

namespace mlir {
namespace chlo {

//===----------------------------------------------------------------------===//
// chlo Dialect Constructor
//===----------------------------------------------------------------------===//

Operation* ChloDialect::materializeConstant(OpBuilder& builder, Attribute value,
                                            Type type, Location loc) {
  // Mirror MHLO dialect here.
  if (value.isa<ElementsAttr>())
    return builder.create<mhlo::ConstOp>(loc, type, value.cast<ElementsAttr>());
  return nullptr;
}

void ChloDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir-hlo/Dialect/mhlo/IR/chlo_ops.cc.inc"
      >();
}

}  // namespace chlo
}  // namespace mlir
