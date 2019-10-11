/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// This file defines the operations used in the XLA dialect.

#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/IR/Attributes.h"  // TF:local_config_mlir
#include "mlir/IR/Builders.h"  // TF:local_config_mlir
#include "mlir/IR/Dialect.h"  // TF:local_config_mlir
#include "mlir/IR/Location.h"  // TF:local_config_mlir
#include "mlir/IR/MLIRContext.h"  // TF:local_config_mlir
#include "mlir/IR/OpDefinition.h"  // TF:local_config_mlir
#include "mlir/IR/OpImplementation.h"  // TF:local_config_mlir
#include "mlir/IR/Operation.h"  // TF:local_config_mlir
#include "mlir/IR/OperationSupport.h"  // TF:local_config_mlir
#include "mlir/IR/PatternMatch.h"  // TF:local_config_mlir
#include "mlir/IR/StandardTypes.h"  // TF:local_config_mlir
#include "mlir/IR/TypeUtilities.h"  // TF:local_config_mlir
#include "mlir/IR/Types.h"  // TF:local_config_mlir
#include "mlir/IR/Value.h"  // TF:local_config_mlir
#include "mlir/Support/LogicalResult.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h.inc"

namespace mlir {
#include "tensorflow/compiler/mlir/xla/ir/hlo_structs.cc.inc"
}  // namespace mlir

using namespace mlir;
using namespace mlir::xla_hlo;

XlaHloDialect::XlaHloDialect(MLIRContext* context)
    : Dialect(getDialectNamespace(), context) {
  addOperations<
#define GET_OP_LIST
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.cc.inc"
      >();

  // Support unknown operations because not all XLA operations are registered.
  // allowUnknownOperations();
}

Operation* XlaHloDialect::materializeConstant(OpBuilder& builder,
                                              Attribute value, Type type,
                                              Location loc) {
  // HLO dialect constants only support ElementsAttr unlike standard dialect
  // constant which supports all attributes.
  if (value.isa<ElementsAttr>())
    return builder.create<xla_hlo::ConstOp>(loc, type,
                                            value.cast<ElementsAttr>());
  return nullptr;
}

template <typename T>
static LogicalResult Verify(T op) {
  return success();
}

//===----------------------------------------------------------------------===//
// ConstOp
//===----------------------------------------------------------------------===//

static void Print(ConstOp op, OpAsmPrinter* printer) {
  // Use short form only if the result type matches type of attribute 'value'.
  bool use_short_form = op.value().getType() == op.getType();

  // Print op name.
  *printer << op.getOperationName();

  // If short form, elide attribute value while printing the attribute
  // dictionary.
  SmallVector<StringRef, 1> elided_attrs;
  if (use_short_form) elided_attrs.push_back("value");
  printer->printOptionalAttrDict(op.getAttrs(), elided_attrs);

  if (use_short_form) {
    *printer << ' ' << op.value();
  } else {
    *printer << " : " << op.getType();
  }
}

static ParseResult ParseConstOp(OpAsmParser* parser, OperationState* result) {
  if (parser->parseOptionalAttributeDict(result->attributes)) return failure();

  // If colon is not present after attribute dictionary, it should be short form
  // and attribute 'value' is outside the dictionary.
  if (failed(parser->parseOptionalColon())) {
    Attribute value;
    if (parser->parseAttribute(value, "value", result->attributes))
      return failure();
    return parser->addTypeToList(value.getType(), result->types);
  }

  // Long form should have type of the result after colon.
  Type ty;
  if (parser->parseType(ty)) return failure();
  result->types.push_back(ty);
  return success();
}

OpFoldResult ConstOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.empty() && "constant has no operands");

  // Return the held attribute value.
  return value();
}

// Builds a constant op with the specified attribute `value`.
void ConstOp::build(Builder* builder, OperationState& result, Attribute value) {
  Type type;
  if (auto elemAttr = value.dyn_cast<ElementsAttr>()) {
    type = elemAttr.getType();
  } else if (value.isa<BoolAttr>() || value.isa<FloatAttr>() ||
             value.isa<IntegerAttr>()) {
    // All XLA types must be tensor types. In the build() method, we want to
    // provide more flexiblity by allowing attributes of scalar types. But we
    // need to wrap it up with ElementsAttr to construct valid XLA constants.
    type = RankedTensorType::get(/*shape=*/{}, value.getType());
    value = DenseElementsAttr::get(type.cast<TensorType>(), value);
  }

  // TODO: support other XLA specific types.
  assert(type && "unsupported attribute type for building xla_hlo.constant");
  result.types.push_back(type);
  result.addAttribute("value", value);
}

//===----------------------------------------------------------------------===//
// IotaOp
//===----------------------------------------------------------------------===//

OpFoldResult IotaOp::fold(ArrayRef<Attribute> operands) {
  const auto output_type = getResult()->getType().cast<ShapedType>();
  const auto output_size = output_type.getNumElements();
  const auto dimension = iota_dimension().getSExtValue();
  const auto max_dim_size = output_type.getDimSize(dimension);
  int bitwidth = output_type.getElementType().getIntOrFloatBitWidth();

  llvm::SmallVector<APInt, 10> values;
  values.reserve(output_size);

  int64_t increase_stride = output_size;
  for (int i = 0; i <= dimension; i++) {
    increase_stride /= output_type.getDimSize(i);
  }

  int64_t current_value = 0;
  for (int i = 0; i < output_size; i++) {
    int64_t value = (current_value / increase_stride) % max_dim_size;
    values.push_back(APInt(bitwidth, value));
    ++current_value;
  }

  return DenseIntElementsAttr::get(output_type, values);
}

//===----------------------------------------------------------------------===//
// ConvertOp
//===----------------------------------------------------------------------===//

namespace {

// Converts the values of an ElementsAttr into the corresponding type.
ElementsAttr ConvertElements(const ElementsAttr& elements, Type newType) {
  auto oldType = getElementTypeOrSelf(elements);
  size_t bitWidth = newType.isBF16() ? 64 : newType.getIntOrFloatBitWidth();

  if (oldType.isa<FloatType>()) {
    // mapValues always takes a function returning APInt, even when the output
    // is actually float.
    using func_type = APInt(const APFloat&);
    if (auto newFloatType = newType.dyn_cast<FloatType>()) {
      // Float -> Float
      return elements.mapValues(
          newType, llvm::function_ref<func_type>([&newFloatType](
                                                     const APFloat& floatVal) {
            APFloat newDouble(FloatAttr::getValueAsDouble(floatVal));
            bool losesInfo = false;
            newDouble.convert(newFloatType.getFloatSemantics(),
                              llvm::APFloat::rmNearestTiesToEven, &losesInfo);
            return newDouble.bitcastToAPInt();
          }));
    }
    // Float -> Int
    return elements.mapValues(
        newType,
        llvm::function_ref<func_type>([&bitWidth](const APFloat& floatVal) {
          return APInt(bitWidth, FloatAttr::getValueAsDouble(floatVal));
        }));
  }

  // oldType is Integer
  // mapValues always takes a function returning APInt, even when the output
  // is actually float.
  using func_type = APInt(const APInt&);
  if (auto newFloatType = newType.dyn_cast<FloatType>()) {
    // Int -> Float
    return elements.mapValues(
        newType,
        llvm::function_ref<func_type>([&newFloatType](const APInt& intVal) {
          APFloat newDouble(static_cast<double>(intVal.getSExtValue()));
          bool losesInfo = false;
          newDouble.convert(newFloatType.getFloatSemantics(),
                            llvm::APFloat::rmNearestTiesToEven, &losesInfo);
          return newDouble.bitcastToAPInt();
        }));
  }
  // newType is Integer
  // Int -> Int
  return elements.mapValues(
      newType, llvm::function_ref<func_type>([&bitWidth](const APInt& intVal) {
        return APInt(bitWidth, intVal.getSExtValue());
      }));
}

}  // namespace

OpFoldResult ConvertOp::fold(ArrayRef<Attribute> operands) {
  if (getOperand()->getType() == getResult()->getType()) return getOperand();

  // If the operand is constant, we can do the conversion now.
  if (auto elementsAttr = operands.front().dyn_cast_or_null<ElementsAttr>()) {
    return ConvertElements(elementsAttr, getElementTypeOrSelf(getResult()));
  }

  return {};
}

//===----------------------------------------------------------------------===//
// GetTupleElementOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(GetTupleElementOp op) {
  auto indexVal = op.index().getZExtValue();
  auto operandType = op.getOperand()->getType().cast<TupleType>();
  if (indexVal >= operandType.size()) {
    return op.emitOpError(
        llvm::formatv("index {0} is out of bounds of operand with size {1}",
                      indexVal, operandType.size()));
  }

  auto expectedType = operandType.getType(indexVal);
  if (op.getType() != expectedType) {
    return op.emitOpError(llvm::formatv("has return type {0}, but expected {1}",
                                        op.getType(), expectedType));
  }
  return success();
}

OpFoldResult GetTupleElementOp::fold(ArrayRef<Attribute> operands) {
  if (auto tupleOp =
          dyn_cast_or_null<xla_hlo::TupleOp>(getOperand()->getDefiningOp())) {
    return tupleOp.getOperand(index().getLimitedValue());
  }

  return {};
}

//===----------------------------------------------------------------------===//
// TupleOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(TupleOp op) {
  SmallVector<Type, 8> operandTypes = {op.operand_type_begin(),
                                       op.operand_type_end()};
  auto expectedType = TupleType::get(operandTypes, op.getContext());
  if (op.getType() != expectedType) {
    return op.emitOpError(llvm::formatv("has return type {0}, but expected {1}",
                                        op.getType(), expectedType));
  }
  return success();
}

//===----------------------------------------------------------------------===//
// BroadcastOp
//===----------------------------------------------------------------------===//

// TODO(b/129012527) These should be expressed as type constraints.
static LogicalResult Verify(BroadcastOp op) {
  auto sizes = op.broadcast_sizes();
  auto sizesType = sizes.getType();
  auto sizesRank = sizesType.getRank();
  if (sizesRank != 1) {
    return op.emitOpError(llvm::formatv(
        "broadcast_sizes has rank {0} instead of rank 1", sizesRank));
  }

  auto resultType = op.getResult()->getType().cast<RankedTensorType>();
  auto resultRank = resultType.getRank();
  auto operandType = op.operand()->getType().cast<RankedTensorType>();
  auto operandRank = operandType.getRank();
  auto sizesSize = sizesType.getNumElements();
  auto expectedRank = operandRank + sizesSize;

  if (resultRank != expectedRank) {
    return op.emitOpError(
        llvm::formatv("result rank ({0}) does not match operand rank "
                      "({1}) plus size of broadcast_sizes ({2})",
                      resultRank, operandRank, sizesSize));
  }

  llvm::SmallVector<int64_t, 10> expectedShape(sizes.getValues<int64_t>());

  auto operandShape = operandType.getShape();
  expectedShape.insert(expectedShape.end(), operandShape.begin(),
                       operandShape.end());

  auto resultShape = resultType.getShape();
  if (resultShape != llvm::makeArrayRef(expectedShape)) {
    return op.emitOpError(llvm::formatv(
        "result has shape [{0}] instead of [{1}]",
        llvm::make_range(resultShape.begin(), resultShape.end()),
        llvm::make_range(expectedShape.begin(), expectedShape.end())));
  }

  return success();
}

//===----------------------------------------------------------------------===//
// BroadcastInDimOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(BroadcastInDimOp op) {
  auto operandType = op.operand()->getType().cast<RankedTensorType>();
  auto operandRank = operandType.getRank();
  if (!op.broadcast_dimensions()) {
    if (operandRank == 0) {
      return success();
    }
    return op.emitOpError(
        llvm::formatv("broadcast_dimensions is absent, but required because "
                      "operand has non-zero rank ({0})",
                      operandRank));
  }

  auto dimensions = *op.broadcast_dimensions();
  auto dimensionsType = op.broadcast_dimensions()->getType();
  auto dimensionsRank = dimensionsType.getRank();
  if (dimensionsRank != 1) {
    return op.emitOpError(llvm::formatv(
        "broadcast_dimensions has rank {0} instead of rank 1", dimensionsRank));
  }

  auto dimensionsSize = dimensionsType.getNumElements();
  if (dimensionsSize != operandRank) {
    return op.emitOpError(llvm::formatv(
        "broadcast_dimensions size ({0}) does not match operand rank ({1})",
        dimensionsSize, operandRank));
  }

  auto resultType = op.getResult()->getType().cast<RankedTensorType>();
  auto resultRank = resultType.getRank();
  if (resultRank < operandRank) {
    return op.emitOpError(
        llvm::formatv("result rank ({0}) is less than operand rank ({1})",
                      resultRank, operandRank));
  }

  for (int i = 0; i != dimensionsSize; ++i) {
    auto dimIndex = dimensions.getValue<int64_t>(i);
    if (dimIndex >= resultRank) {
      return op.emitOpError(
          llvm::formatv("broadcast_dimensions contains invalid value {0} for "
                        "result result with rank {1}",
                        dimIndex, resultRank));
    }

    auto dimSize = operandType.getDimSize(i);
    auto resultDimSize = resultType.getDimSize(dimIndex);
    if (dimSize != 1 && dimSize != resultDimSize) {
      return op.emitOpError(
          llvm::formatv("size of operand dimension {0} ({1}) is not equal to "
                        "1 or size of result dimension {2} ({3})",
                        i, dimSize, dimIndex, resultDimSize));
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ClampOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(ClampOp op) {
  auto operandType = op.operand()->getType().cast<RankedTensorType>();
  auto operandShape = operandType.getShape();
  auto minType = op.min()->getType().cast<RankedTensorType>();

  auto minShape = minType.getShape();
  if (minShape != operandShape && minType.getRank() != 0) {
    return op.emitOpError(llvm::formatv(
        "min shape [{0}] is not scalar and does not match operand shape [{1}]",
        llvm::make_range(minShape.begin(), minShape.end()),
        llvm::make_range(operandShape.begin(), operandShape.end())));
  }

  auto maxType = op.max()->getType().cast<RankedTensorType>();
  auto maxShape = maxType.getShape();
  if (maxShape != operandShape && maxType.getRank() != 0) {
    return op.emitOpError(llvm::formatv(
        "max shape [{0}] is not scalar and does not match operand shape [{1}]",
        llvm::make_range(maxShape.begin(), maxShape.end()),
        llvm::make_range(operandShape.begin(), operandShape.end())));
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ConcatenateOp
//===----------------------------------------------------------------------===//

OpFoldResult ConcatenateOp::fold(ArrayRef<Attribute> operands) {
  if (getNumOperands() == 1) return getOperand(0);
  return {};
}

static LogicalResult Verify(ConcatenateOp op) {
  auto firstType = op.getOperand(0)->getType().cast<RankedTensorType>();

  auto firstShape = firstType.getShape();
  int numOperands = op.getNumOperands();
  for (int i = 1; i < numOperands; i++) {
    auto secondType = op.getOperand(i)->getType().cast<RankedTensorType>();

    if (firstType.getRank() != secondType.getRank()) {
      return op.emitOpError(
          llvm::formatv("operands (0) and ({0}) do not match rank.", i));
    }

    auto secondShape = secondType.getShape();
    for (int d = 0; d < firstType.getRank(); ++d) {
      if (firstShape[d] != secondShape[d] && d != op.dimension()) {
        return op.emitOpError(llvm::formatv(
            "operands (0) and ({0}) non-concat dimensions do not match "
            "({1}) != ({2}).",
            i, llvm::make_range(firstShape.begin(), firstShape.end()),
            llvm::make_range(secondShape.begin(), secondShape.end())));
      }
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

OpFoldResult ReshapeOp::fold(ArrayRef<Attribute> operands) {
  if (getOperand()->getType() == getType()) {
    return getOperand();
  }

  if (auto prev_op =
          dyn_cast_or_null<ReshapeOp>(getOperand()->getDefiningOp())) {
    setOperand(prev_op.getOperand());
    return getResult();
  }

  if (auto elements = operands.front().dyn_cast_or_null<DenseElementsAttr>()) {
    return elements.reshape(getResult()->getType().cast<ShapedType>());
  }

  return {};
}

//===----------------------------------------------------------------------===//
// ReverseOp
//===----------------------------------------------------------------------===//

OpFoldResult ReverseOp::fold(ArrayRef<Attribute> operands) {
  // No dimensions to reverse.
  if (dimensions().getNumElements() == 0) return operand();
  return nullptr;
}

//===----------------------------------------------------------------------===//
// ReduceOp
//===----------------------------------------------------------------------===//

LogicalResult ReduceOp::fold(ArrayRef<Attribute> operands,
                             SmallVectorImpl<OpFoldResult>& results) {
  // No dimensions to reduce.
  if (dimensions().getNumElements() == 0) {
    for (Value* input : this->operands()) {
      results.push_back(input);
    }
    return success();
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(SelectOp op) {
  auto onTrueType = op.on_true()->getType().cast<RankedTensorType>();
  auto onFalseType = op.on_false()->getType().cast<RankedTensorType>();

  if (onTrueType != onFalseType) {
    return op.emitOpError(
        llvm::formatv("on_true type ({0}) does not match on_false type ({1})",
                      onTrueType, onFalseType));
  }

  auto predType = op.pred()->getType().cast<RankedTensorType>();
  auto predShape = predType.getShape();
  auto predRank = predType.getRank();
  auto selectShape = onTrueType.getShape();

  if (predRank != 0 && predShape != selectShape) {
    return op.emitOpError(llvm::formatv(
        "pred shape ([{0}]) is not scalar and does not match operand shapes "
        "([{1}])",
        llvm::make_range(predShape.begin(), predShape.end()),
        llvm::make_range(selectShape.begin(), selectShape.end())));
  }

  return success();
}

//===----------------------------------------------------------------------===//
// PadOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(PadOp op) {
  auto input_type = op.operand()->getType().cast<RankedTensorType>();
  auto pad_type = op.padding_value()->getType().cast<RankedTensorType>();

  if (pad_type.getRank() != 0) {
    return op.emitOpError(
        llvm::formatv("padding value type should be a rank-0 "
                      "tensor, is rank {0}",
                      pad_type.getRank()));
  }

  const auto& padding_low = op.edge_padding_low();
  if (padding_low.getType().getNumElements() != input_type.getRank()) {
    return op.emitOpError(llvm::formatv(
        "edge_padding_low length ({0}) must match operand rank ({1}).",
        padding_low.getType().getNumElements(), input_type.getRank()));
  }

  const auto& padding_high = op.edge_padding_high();
  if (padding_high.getType().getNumElements() != input_type.getRank()) {
    return op.emitOpError(llvm::formatv(
        "edge_padding_high length ({0}) must match operand rank ({1}).",
        padding_high.getType().getNumElements(), input_type.getRank()));
  }

  auto input_shape = input_type.getShape();
  auto output_shape =
      op.getResult()->getType().cast<RankedTensorType>().getShape();
  if (input_shape.size() != output_shape.size()) {
    return op.emitOpError(
        llvm::formatv("Operand rank ({0}) and result rank({0}) should match",
                      input_shape.size(), output_shape.size()));
  }

  for (int i = 0, e = input_shape.size(); i < e; i++) {
    int expected_output = input_shape[i] +
                          padding_low.getValue<IntegerAttr>(i).getInt() +
                          padding_high.getValue<IntegerAttr>(i).getInt();
    if (expected_output != output_shape[i]) {
      return op.emitOpError(
          llvm::formatv("Expected output shape ({0}) and "
                        "output shape ({1}) should match.",
                        expected_output, output_shape[i]));
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// SliceOp
//===----------------------------------------------------------------------===//

void SliceOp::build(Builder* builder, OperationState& result, Value* operand,
                    DenseIntElementsAttr start_indices,
                    DenseIntElementsAttr limit_indices,
                    DenseIntElementsAttr strides) {
  return build(
      builder, result,
      InferOutputTypes(builder, operand, start_indices, limit_indices, strides),
      operand, start_indices, limit_indices, strides);
}

// Returns output dimension size for slice result for the given arguments.
// Returns -1 if arguments are illegal.
static int64_t InferSliceDim(int64_t input_dim, int64_t start, int64_t end,
                             int64_t stride) {
  if (input_dim == -1 || start < 0 || start > end || end > input_dim ||
      stride == 0)
    return -1;

  return llvm::divideCeil(end - start, stride);
}

Type SliceOp::InferOutputTypes(Builder* builder, Value* operand,
                               DenseIntElementsAttr start_indices,
                               DenseIntElementsAttr limit_indices,
                               DenseIntElementsAttr strides) {
  Type ty = operand->getType();
  RankedTensorType ranked_ty = ty.dyn_cast<RankedTensorType>();
  if (!ranked_ty) return ty;
  int64_t rank = ranked_ty.getRank();

  // Illegal attributes.
  ShapedType attr_ty = start_indices.getType();
  if (attr_ty.getRank() != 1 || attr_ty.getNumElements() != rank ||
      !attr_ty.getElementType().isInteger(64) ||
      limit_indices.getType() != attr_ty || strides.getType() != attr_ty)
    return ty;

  SmallVector<int64_t, 4> start(start_indices.getValues<int64_t>());
  SmallVector<int64_t, 4> limit(limit_indices.getValues<int64_t>());
  SmallVector<int64_t, 4> stride_vals(strides.getValues<int64_t>());

  SmallVector<int64_t, 4> shape;
  shape.reserve(rank);
  for (int64_t i = 0, e = rank; i != e; i++) {
    shape.push_back(InferSliceDim(ranked_ty.getDimSize(i), start[i], limit[i],
                                  stride_vals[i]));
  }
  return builder->getTensorType(shape, ranked_ty.getElementType());
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

OpFoldResult TransposeOp::fold(ArrayRef<Attribute> operands) {
  for (auto it : llvm::enumerate(permutation().getValues<APInt>())) {
    if (it.index() != it.value()) {
      return {};
    }
  }
  return getOperand();
}

static LogicalResult Verify(TransposeOp op) {
  auto permutationType = op.permutation().getType();
  auto permutationRank = permutationType.getRank();
  if (permutationRank != 1) {
    return op.emitOpError(llvm::formatv(
        "permutation has rank {0} instead of rank 1", permutationRank));
  }

  auto operandType = op.operand()->getType().cast<RankedTensorType>();
  auto operandRank = operandType.getRank();
  auto permutationSize = permutationType.getNumElements();
  if (permutationSize != operandRank) {
    return op.emitOpError(llvm::formatv(
        "permutation size ({0}) does not match operand rank ({1})",
        permutationSize, operandRank));
  }

  auto resultType = op.getResult()->getType().cast<RankedTensorType>();
  auto resultRank = resultType.getRank();
  if (resultRank != operandRank) {
    return op.emitOpError(
        llvm::formatv("result rank ({0}) does not match operand rank ({1})",
                      resultRank, operandRank));
  }

  auto resultShape = resultType.getShape();

  auto expectedShape = SmallVector<int64_t, 10>(operandRank);
  for (int i = 0; i != operandRank; ++i) {
    auto permutedDim = op.permutation().getValue<IntegerAttr>(i).getInt();
    expectedShape[i] = operandType.getDimSize(permutedDim);
  }

  if (resultShape != llvm::makeArrayRef(expectedShape)) {
    return op.emitOpError(llvm::formatv(
        "result shape is [{0}"
        "] instead of [{1}"
        "]",
        llvm::make_range(resultShape.begin(), resultShape.end()),
        llvm::make_range(expectedShape.begin(), expectedShape.end())));
  }

  return success();
}

//===----------------------------------------------------------------------===//
// CompareOp
//===----------------------------------------------------------------------===//

void CompareOp::build(Builder* builder, OperationState& result, Value* lhs,
                      Value* rhs, DenseIntElementsAttr broadcast_dimensions,
                      StringAttr comparison_direction) {
  build(builder, result,
        InferOutputTypes(builder, lhs, rhs, broadcast_dimensions,
                         comparison_direction),
        lhs, rhs, broadcast_dimensions, comparison_direction);
}

Type CompareOp::InferOutputTypes(Builder* builder, Value* lhs, Value* rhs,
                                 DenseIntElementsAttr broadcast_dimensions,
                                 StringAttr comparison_direction) {
  if (!lhs->getType().isa<ShapedType>() || !rhs->getType().isa<ShapedType>())
    return builder->getTensorType(builder->getI1Type());
  // TODO(parkers): When binary ops support broadcasting shape inference, reuse
  // that logic.
  auto lhs_type = lhs->getType().cast<ShapedType>();
  auto rhs_type = rhs->getType().cast<ShapedType>();
  if (lhs_type != rhs_type) return builder->getTensorType(builder->getI1Type());
  return builder->getTensorType(lhs_type.getShape(), builder->getI1Type());
}

#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.cc.inc"
