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
#include "mlir-hlo/utils/broadcast_utils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"

namespace mlir {
namespace chlo {

template <typename T>
static LogicalResult Verify(T op) {
  return success();
}

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
// BinaryOps
//===----------------------------------------------------------------------===//

namespace {
// Gets the resulting type from a broadcast between two types.
static Type GetBroadcastType(Type x, Type y, Type element_type,
                             DenseIntElementsAttr broadcast_dimensions_attr) {
  auto x_ranked = x.dyn_cast<RankedTensorType>();
  auto y_ranked = y.dyn_cast<RankedTensorType>();
  if (!x_ranked || !y_ranked) {
    return UnrankedTensorType::get(element_type);
  }

  auto shape_x = x_ranked.getShape();
  auto shape_y = y_ranked.getShape();

  if (shape_x.size() == shape_y.size()) {
    llvm::SmallVector<int64_t, 4> out_shape(shape_x.size());
    for (int i = 0, e = shape_x.size(); i < e; i++) {
      auto x_val = shape_x[i];
      auto y_val = shape_y[i];
      if (x_val == -1 || y_val == -1) {
        out_shape[i] = -1;
      } else {
        out_shape[i] = std::max(x_val, y_val);
      }
    }
    return RankedTensorType::get(out_shape, element_type);
  }

  auto shape_large = shape_x.size() > shape_y.size() ? shape_x : shape_y;
  auto shape_small = shape_x.size() <= shape_y.size() ? shape_x : shape_y;

  llvm::SmallVector<int64_t, 4> broadcast_dimensions;
  if (broadcast_dimensions_attr) {
    // Explicit broadcast dimensions.
    for (const APInt& int_value :
         broadcast_dimensions_attr.getValues<APInt>()) {
      broadcast_dimensions.push_back(int_value.getSExtValue());
    }
    if (broadcast_dimensions.size() != shape_small.size()) {
      // Signal illegal broadcast_dimensions as unranked.
      return UnrankedTensorType::get(element_type);
    }
  } else {
    // If no broadcast dimensions, assume "numpy" broadcasting.
    broadcast_dimensions = llvm::to_vector<4>(llvm::seq<int64_t>(
        shape_large.size() - shape_small.size(), shape_large.size()));
  }

  llvm::SmallVector<int64_t, 4> out_shape(shape_large.begin(),
                                          shape_large.end());

  // Update according to the broadcast dimensions.
  for (auto index_pair : llvm::enumerate(broadcast_dimensions)) {
    auto old_value = out_shape[index_pair.value()];
    auto new_value = shape_small[index_pair.index()];
    if (old_value != -1 && (new_value == -1 || new_value > old_value)) {
      out_shape[index_pair.value()] = new_value;
    }
  }

  return RankedTensorType::get(out_shape, element_type);
}

LogicalResult InferBroadcastBinaryOpReturnTypeComponents(
    MLIRContext* context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, Type element_type,
    SmallVectorImpl<ShapedTypeComponents>& inferedReturnShapes) {
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
  Type result_type =
      GetBroadcastType(lhs_type, rhs_type, element_type, broadcast_dimensions);

  if (auto ranked_result_type = result_type.dyn_cast<RankedTensorType>()) {
    inferedReturnShapes.emplace_back(ranked_result_type.getShape(),
                                     element_type);
    return success();
  }

  // TODO(laurenzo): This should be constructing with `element_type` but that
  // constructor variant needs to be added upstream.
  inferedReturnShapes.emplace_back(/* element_type */);
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
    DictionaryAttr attributes, RegionRange regions,
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
                               StringAttr comparison_direction,
                               StringAttr compare_type) {
  auto new_type = GetBroadcastType(lhs.getType(), rhs.getType(),
                                   builder.getI1Type(), broadcast_dimensions);
  build(builder, result, new_type, lhs, rhs, broadcast_dimensions,
        comparison_direction, compare_type);
}

LogicalResult BroadcastCompareOp::inferReturnTypeComponents(
    MLIRContext* context, Optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, RegionRange regions,
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
    MLIRContext* ctx, Optional<Location>, ValueRange operands, DictionaryAttr,
    RegionRange, SmallVectorImpl<Type>& inferredReturnTypes) {
  inferredReturnTypes.push_back(getIsInfLikeReturnType(operands.front()));
  return success();
}

//===----------------------------------------------------------------------===//
// IsNegInfOp
//===----------------------------------------------------------------------===//

LogicalResult IsNegInfOp::inferReturnTypes(
    MLIRContext* ctx, Optional<Location>, ValueRange operands, DictionaryAttr,
    RegionRange, SmallVectorImpl<Type>& inferredReturnTypes) {
  inferredReturnTypes.push_back(getIsInfLikeReturnType(operands.front()));
  return success();
}

//===----------------------------------------------------------------------===//
// IsPosInfOp
//===----------------------------------------------------------------------===//

LogicalResult IsPosInfOp::inferReturnTypes(
    MLIRContext* ctx, Optional<Location>, ValueRange operands, DictionaryAttr,
    RegionRange, SmallVectorImpl<Type>& inferredReturnTypes) {
  inferredReturnTypes.push_back(getIsInfLikeReturnType(operands.front()));
  return success();
}

//===----------------------------------------------------------------------===//
// Macros for method definitions that are common to most broadcasting ops.
//===----------------------------------------------------------------------===//

#define BROADCAST_INFER_SHAPE_TYPE_OP_DEFS(Op)                             \
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

#define BROADCAST_BINARY_OP_DEFS(Op)                                           \
  void Op::build(OpBuilder& builder, OperationState& result, Value left,       \
                 Value right, DenseIntElementsAttr broadcast_dimensions) {     \
    auto type = GetBroadcastType(                                              \
        left.getType().cast<ShapedType>(), right.getType().cast<ShapedType>(), \
        getElementTypeOrSelf(right.getType()), broadcast_dimensions);          \
    return Op::build(builder, result, type, left, right,                       \
                     broadcast_dimensions);                                    \
  }                                                                            \
  BROADCAST_INFER_SHAPE_TYPE_OP_DEFS(Op)

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

#undef BROADCAST_INFER_SHAPE_TYPE_OP_DEFS
#undef BROADCAST_BINARY_OP_DEFS

static LogicalResult Verify(ConstantLikeOp op) {
  if (op.value().getType() != op.getType().cast<ShapedType>().getElementType())
    return op.emitOpError() << "value's type doesn't match element return type";
  return success();
}

//===----------------------------------------------------------------------===//
// MinimumBroadcastShapesOp
//===----------------------------------------------------------------------===//
static LogicalResult Verify(MinimumBroadcastShapesOp op) {
  // Check that the number of operands matches the number of outputs.
  unsigned result_shapes_count = op.results().size();
  unsigned operand_shapes_count = op.shapes().size();
  if (operand_shapes_count != result_shapes_count) {
    return op.emitOpError()
           << "number of operand shapes (" << operand_shapes_count
           << ") does not match number of result shapes ("
           << result_shapes_count << ")";
  }
  if (operand_shapes_count < 2) {
    return op.emitOpError() << "number of operand shapes ("
                            << operand_shapes_count << ") should be >= 2";
  }
  return success();
}

LogicalResult ConstantLikeOp::inferReturnTypeComponents(
    MLIRContext* context, Optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, RegionRange regions,
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

OpFoldResult ConstantLikeOp::fold(ArrayRef<Attribute> operands) {
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
  Type other =
      GetBroadcastType(on_true_type, on_false_type, element_type, nullptr);
  Type output = GetBroadcastType(other, pred_type, element_type, nullptr);

  inferredReturnShapes.push_back(output);
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
    Optional<unsigned> index, ArrayRef<Attribute> operands,
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

static LogicalResult Verify(RankSpecializationClusterOp op) {
  if (failed(RegionBranchOpInterface::verifyTypes(op))) return failure();
  if (op.body().getArgumentTypes() != op.getOperandTypes())
    return op.emitOpError() << "block argument types must match operand types";

  // All operands of nested ops must be defined in the body or declared by the
  // cluster.
  Block* body = op.getBody();
  for (Operation& nested : body->without_terminator()) {
    if (!llvm::all_of(nested.getOpOperands(), [&](OpOperand& operand) {
          Operation* def = operand.get().getDefiningOp();
          if (def != nullptr && def->getBlock() == body) return true;
          return llvm::is_contained(body->getArguments(), operand.get());
        })) {
      return op.emitOpError()
             << "nested ops must not depend on implicit operands";
    }
  }

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

Operation* HloClientDialect::materializeConstant(OpBuilder& builder,
                                                 Attribute value, Type type,
                                                 Location loc) {
  // Mirror MHLO dialect here.
  if (value.isa<ElementsAttr>())
    return builder.create<mhlo::ConstOp>(loc, type, value.cast<ElementsAttr>());
  return nullptr;
}

void HloClientDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir-hlo/Dialect/mhlo/IR/chlo_ops.cc.inc"
      >();
}

}  // namespace chlo
}  // namespace mlir
