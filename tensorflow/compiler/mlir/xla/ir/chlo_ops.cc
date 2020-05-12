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

#include "tensorflow/compiler/mlir/xla/ir/chlo_ops.h"

#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/StandardTypes.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/xla/ir/broadcast_utils.h"

namespace mlir {
namespace xla_chlo {

template <typename T>
static LogicalResult Verify(T op) {
  return success();
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
    for (int i = 0; i < shape_x.size(); i++) {
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
    for (const APInt& int_value : broadcast_dimensions_attr.getIntValues()) {
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
    OpBuilder& builder, Operation* op,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  auto loc = op->getLoc();
  auto lhs = op->getOperand(0);
  auto rhs = op->getOperand(1);

  // Check for "numpy"-style rank broadcast.
  auto broadcast_dimensions = op->getAttr("broadcast_dimensions")
                                  .dyn_cast_or_null<DenseIntElementsAttr>();
  if (broadcast_dimensions &&
      !xla::IsLegalNumpyRankedBroadcast(lhs, rhs, broadcast_dimensions)) {
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

  Value computed_shape = xla::ComputeBinaryElementwiseBroadcastingResultExtents(
      loc, lhs, rhs, builder);
  if (!computed_shape) return failure();
  reifiedReturnShapes.push_back(computed_shape);
  return success();
}
}  // namespace

//===----------------------------------------------------------------------===//
// BroadcastComplexOp (has custom type inference due to different result type).
//===----------------------------------------------------------------------===//

LogicalResult BroadcastComplexOp::inferReturnTypeComponents(
    MLIRContext* context, Optional<Location> location, ValueRange operands,
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
    OpBuilder& builder, SmallVectorImpl<Value>& reifiedReturnShapes) {
  return ReifyBroadcastBinaryOpReturnTypeShapes(builder, getOperation(),
                                                reifiedReturnShapes);
}

//===----------------------------------------------------------------------===//
// BroadcastCompareOp (has custom type inference due to different result type).
//===----------------------------------------------------------------------===//

LogicalResult BroadcastCompareOp::inferReturnTypeComponents(
    MLIRContext* context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferedReturnShapes) {
  Type element_type = IntegerType::get(1, context);
  return InferBroadcastBinaryOpReturnTypeComponents(context, location, operands,
                                                    attributes, element_type,
                                                    inferedReturnShapes);
}
LogicalResult BroadcastCompareOp::reifyReturnTypeShapes(
    OpBuilder& builder, SmallVectorImpl<Value>& reifiedReturnShapes) {
  return ReifyBroadcastBinaryOpReturnTypeShapes(builder, getOperation(),
                                                reifiedReturnShapes);
}

//===----------------------------------------------------------------------===//
// Macros for method definitions that are common to most broadcasting ops.
//===----------------------------------------------------------------------===//

#define BROADCAST_INFER_SHAPE_TYPE_OP_DEFS(Op)                                \
  LogicalResult Op::inferReturnTypeComponents(                                \
      MLIRContext* context, Optional<Location> location, ValueRange operands, \
      DictionaryAttr attributes, RegionRange regions,                         \
      SmallVectorImpl<ShapedTypeComponents>& inferedReturnShapes) {           \
    return InferBroadcastBinaryOpReturnTypeComponents(                        \
        context, location, operands, attributes, /*element_type=*/nullptr,    \
        inferedReturnShapes);                                                 \
  }                                                                           \
  LogicalResult Op::reifyReturnTypeShapes(                                    \
      OpBuilder& builder, SmallVectorImpl<Value>& reifiedReturnShapes) {      \
    return ReifyBroadcastBinaryOpReturnTypeShapes(builder, getOperation(),    \
                                                  reifiedReturnShapes);       \
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
BROADCAST_BINARY_OP_DEFS(BroadcastOrOp);
BROADCAST_BINARY_OP_DEFS(BroadcastPowOp);
BROADCAST_BINARY_OP_DEFS(BroadcastRemOp);
BROADCAST_BINARY_OP_DEFS(BroadcastShiftLeftOp);
BROADCAST_BINARY_OP_DEFS(BroadcastShiftRightArithmeticOp);
BROADCAST_BINARY_OP_DEFS(BroadcastShiftRightLogicalOp);
BROADCAST_BINARY_OP_DEFS(BroadcastSubOp);
BROADCAST_BINARY_OP_DEFS(BroadcastXorOp);

#undef BROADCAST_INFER_SHAPE_TYPE_OP_DEFS
#undef BROADCAST_BINARY_OP_DEFS

#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/xla/ir/chlo_ops.cc.inc"

//===----------------------------------------------------------------------===//
// xla_chlo Dialect Constructor
//===----------------------------------------------------------------------===//

XlaHloClientDialect::XlaHloClientDialect(MLIRContext* context)
    : Dialect(getDialectNamespace(), context) {
  addOperations<
#define GET_OP_LIST
#include "tensorflow/compiler/mlir/xla/ir/chlo_ops.cc.inc"
      >();
}

}  // namespace xla_chlo
}  // namespace mlir
