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
ShapedTypeComponents GetBroadcastType(
    Type x, Type y, Type element_type,
    DenseIntElementsAttr broadcast_dimensions_attr) {
  auto x_ranked = x.dyn_cast<RankedTensorType>();
  auto y_ranked = y.dyn_cast<RankedTensorType>();
  if (!x_ranked || !y_ranked) {
    return {element_type};
  }

  auto shape_x = x_ranked.getShape();
  auto shape_y = y_ranked.getShape();

  // If no broadcast dimensions, assume "numpy" broadcasting.
  if (shape_x.size() == shape_y.size() || !broadcast_dimensions_attr) {
    llvm::SmallVector<int64_t, 4> out_shape;
    if (!mlir::OpTrait::util::getBroadcastedShape(shape_x, shape_y,
                                                  out_shape)) {
      // Signal illegal broadcast_dimensions as unranked.
      return {element_type};
    }
    return {out_shape, element_type};
  }

  auto shape_large = shape_x.size() > shape_y.size() ? shape_x : shape_y;
  auto shape_small = shape_x.size() <= shape_y.size() ? shape_x : shape_y;

  auto broadcast_dimensions = broadcast_dimensions_attr.getValues<APInt>();
  if (broadcast_dimensions.size() != shape_small.size()) {
    // Signal illegal broadcast_dimensions as unranked.
    return {element_type};
  }

  llvm::SmallVector<int64_t, 4> shape_large_filtered;
  shape_large_filtered.reserve(shape_small.size());
  for (const auto& dim : broadcast_dimensions) {
    if (dim.getZExtValue() >= shape_large.size()) return {element_type};
    shape_large_filtered.push_back(shape_large[dim.getZExtValue()]);
  }
  llvm::SmallVector<int64_t, 4> out_shape_filtered;
  if (!mlir::OpTrait::util::getBroadcastedShape(
          shape_small, shape_large_filtered, out_shape_filtered)) {
    // Signal illegal broadcast_dimensions as unranked.
    return {element_type};
  }

  // Update according to the broadcast dimensions.
  llvm::SmallVector<int64_t, 4> out_shape(shape_large.begin(),
                                          shape_large.end());
  for (const auto& index_pair : llvm::enumerate(broadcast_dimensions)) {
    auto new_value = out_shape_filtered[index_pair.index()];
    out_shape[index_pair.value().getZExtValue()] = new_value;
  }

  return {out_shape, element_type};
}

LogicalResult InferBroadcastBinaryOpReturnTypeComponents(
    MLIRContext* context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, Type element_type,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  // Find broadcast_dimensions.
  DenseIntElementsAttr broadcast_dimensions =
      attributes.get("broadcast_dimensions")
          .dyn_cast_or_null<DenseIntElementsAttr>();

  ShapedType lhs_type = operands[0].getType().dyn_cast<ShapedType>();
  ShapedType rhs_type = operands[1].getType().dyn_cast<ShapedType>();
  if (!lhs_type || !rhs_type ||
      lhs_type.getElementType() != rhs_type.getElementType()) {
    return emitOptionalError(location, "mismatched operand types");
  }
  if (!element_type) element_type = lhs_type.getElementType();
  inferredReturnShapes.push_back(
      GetBroadcastType(lhs_type, rhs_type, element_type, broadcast_dimensions));
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
  auto broadcast_dimensions = op->getAttr("broadcast_dimensions")
                                  .dyn_cast_or_null<DenseIntElementsAttr>();
  if (broadcast_dimensions &&
      !hlo::IsLegalNumpyRankedBroadcast(lhs, rhs, broadcast_dimensions)) {
    // Note: It is unclear whether the general specification of explicit
    // broadcast_dimensions on binary ops is a feature we want to carry
    // forward. While it can technically be implemented for ranked-dynamic,
    // it is incompatible with unranked inputs. If this warning is emitted
    // in real programs, it is an indication that the feature should be
    // implemented versus just falling back on the more standard definition
    // of numpy-like prefix-padding.
    return op->emitWarning()
           << "unsupported non prefix-padded dynamic rank "
           << "broadcast_dimensions = " << broadcast_dimensions;
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
  ShapedType lhs_type = operands[0].getType().dyn_cast<ShapedType>();
  if (!lhs_type) {
    return emitOptionalError(location, "expected ShapedType");
  }
  Type element_type = ComplexType::get(lhs_type.getElementType());
  return InferBroadcastBinaryOpReturnTypeComponents(context, location, operands,
                                                    attributes, element_type,
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
                               DenseIntElementsAttr broadcast_dimensions,
                               mhlo::ComparisonDirection comparison_direction,
                               mhlo::ComparisonType compare_type) {
  build(builder, result, lhs, rhs, broadcast_dimensions,
        mhlo::ComparisonDirectionAttr::get(builder.getContext(),
                                           comparison_direction),
        mhlo::ComparisonTypeAttr::get(builder.getContext(), compare_type));
}

LogicalResult BroadcastCompareOp::inferReturnTypeComponents(
    MLIRContext* context, Optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, RegionRange /*regions*/,
    SmallVectorImpl<ShapedTypeComponents>& inferedReturnShapes) {
  Type element_type = IntegerType::get(context, 1);
  return InferBroadcastBinaryOpReturnTypeComponents(context, location, operands,
                                                    attributes, element_type,
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
  unsigned result_shapes_count = results().size();
  unsigned operand_shapes_count = shapes().size();
  if (operand_shapes_count != result_shapes_count) {
    return emitOpError() << "number of operand shapes (" << operand_shapes_count
                         << ") does not match number of result shapes ("
                         << result_shapes_count << ")";
  }
  if (operand_shapes_count < 2) {
    return emitOpError() << "number of operand shapes (" << operand_shapes_count
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
  Type element_type = op.value().getType();
  Type operand_type = op.operand().getType();
  if (operand_type.isa<UnrankedTensorType>()) {
    inferedReturnShapes.emplace_back(element_type);
  } else {
    const auto& shape = operand_type.cast<RankedTensorType>().getShape();
    inferedReturnShapes.emplace_back(shape, element_type);
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
  auto op_type = operand().getType().cast<ShapedType>();
  if (!op_type.hasStaticShape()) return {};
  auto type = RankedTensorType::get(op_type.getShape(), value().getType());
  return DenseElementsAttr::get(type, value());
}

LogicalResult BroadcastSelectOp::inferReturnTypeComponents(
    MLIRContext*, Optional<Location> location, ValueShapeRange operands,
    DictionaryAttr, RegionRange,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  BroadcastSelectOp::Adaptor op(operands.getValues());
  auto pred_type = op.pred().getType().dyn_cast<ShapedType>();
  auto on_true_type = op.on_true().getType().dyn_cast<ShapedType>();
  auto on_false_type = op.on_false().getType().dyn_cast<ShapedType>();

  if (!pred_type || !on_true_type || !on_false_type ||
      on_true_type.getElementType() != on_false_type.getElementType()) {
    return emitOptionalError(location, "mismatched operand types");
  }

  Type element_type = on_true_type.getElementType();

  // Compute the result shape as two binary broadcasts.
  ShapedTypeComponents& components = inferredReturnShapes.emplace_back(
      GetBroadcastType(on_true_type, on_false_type, element_type, nullptr));
  if (components.hasRank()) {
    components = GetBroadcastType(
        RankedTensorType::get(components.getDims(), element_type), pred_type,
        element_type, nullptr);
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

  auto operand_ty = operand.getType().dyn_cast<RankedTensorType>();
  if (!operand_ty) {
    return emitOptionalError(location, "operand must be ranked");
  }
  if (operand_ty.getRank() < 1) {
    return emitOptionalError(location, "operand's rank must be at least 1");
  }
  auto operand_last_dim = operand_ty.getShape()[operand_ty.getRank() - 1];
  if (operand_last_dim == ShapedType::kDynamicSize) {
    return emitOptionalError(location,
                             "operand's last dimension must be static");
  }
  if (operand_last_dim < k) {
    return emitOptionalError(location,
                             "operand's last dimension must be at least ", k);
  }

  SmallVector<int64_t> result_shape;
  append_range(result_shape, operand_ty.getShape());
  result_shape[operand_ty.getRank() - 1] = k;

  inferredReturnShapes.emplace_back(result_shape, operand_ty.getElementType());
  inferredReturnShapes.emplace_back(result_shape, builder.getI32Type());
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
