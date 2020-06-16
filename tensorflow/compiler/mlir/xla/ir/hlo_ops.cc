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

#include <algorithm>
#include <functional>

#include "absl/container/flat_hash_set.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/OpImplementation.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/StandardTypes.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/InliningUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/xla/convert_op_folder.h"
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h.inc"
#include "tensorflow/compiler/mlir/xla/ir/hlo_utils.h"

namespace mlir {
#include "tensorflow/compiler/mlir/xla/ir/hlo_structs.cc.inc"
namespace xla_hlo {

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

namespace {

//===----------------------------------------------------------------------===//
// Utilities for the canonicalize patterns
//===----------------------------------------------------------------------===//

// Returns 1D 64-bit dense elements attribute with the given values.
DenseIntElementsAttr GetI64ElementsAttr(ArrayRef<int64_t> values,
                                        Builder* builder) {
  RankedTensorType ty = RankedTensorType::get(
      {static_cast<int64_t>(values.size())}, builder->getIntegerType(64));
  return DenseIntElementsAttr::get(ty, values);
}

// Given the start indices and slice sizes for a dynamic-slice that can be
// converted to a static slice, returns the limits for the static slice.
DenseIntElementsAttr BuildSliceLimits(DenseIntElementsAttr start_indices,
                                      DenseIntElementsAttr slice_sizes,
                                      Builder* builder) {
  SmallVector<int64_t, 4> slice_limits;
  for (int64_t i = 0; i < slice_sizes.getNumElements(); ++i) {
    int64_t start_index = start_indices.getValue<IntegerAttr>(i).getInt();
    int64_t slice_size = slice_sizes.getValue<IntegerAttr>(i).getInt();
    slice_limits.push_back(start_index + slice_size);
  }
  return GetI64ElementsAttr(slice_limits, builder);
}

// Returns the padding value of the given position. If padding_attr is a
// nullptr, returns 0.
static int64_t GetPaddingValue(DenseIntElementsAttr padding_attr,
                               ArrayRef<uint64_t> index) {
  if (!padding_attr) return 0;
  return padding_attr.getValue<int64_t>(index);
}

static bool IsOnlyPaddingSpatialDims(Value lhs,
                                     ConvDimensionNumbers dimension_numbers,
                                     DenseIntElementsAttr edge_padding_low,
                                     DenseIntElementsAttr edge_padding_high) {
  const int64_t batch_dim = dimension_numbers.input_batch_dimension().getInt();
  const int64_t feature_dim =
      dimension_numbers.input_feature_dimension().getInt();
  if (edge_padding_low.getValue<int64_t>(batch_dim) ||
      edge_padding_high.getValue<int64_t>(batch_dim))
    return false;
  if (edge_padding_low.getValue<int64_t>(feature_dim) ||
      edge_padding_high.getValue<int64_t>(feature_dim))
    return false;
  return true;
}

DenseIntElementsAttr BuildConvPaddingAttrs(
    DenseIntElementsAttr edge_padding_low,
    DenseIntElementsAttr edge_padding_high, DenseIntElementsAttr padding_attr,
    ConvDimensionNumbers dimension_numbers, Builder* builder) {
  SmallVector<int64_t, 4> padding_low, padding_high;
  for (const auto& dim : dimension_numbers.input_spatial_dimensions()) {
    unsigned i = dim.getZExtValue();
    padding_low.push_back(edge_padding_low.getValue<int64_t>(i));
    padding_high.push_back(edge_padding_high.getValue<int64_t>(i));
  }

  int rank = padding_low.size();
  SmallVector<int64_t, 8> padding;
  for (unsigned i = 0; i < rank; ++i) {
    padding.push_back(GetPaddingValue(padding_attr, {i, 0}) + padding_low[i]);
    padding.push_back(GetPaddingValue(padding_attr, {i, 1}) + padding_high[i]);
  }
  // padding_attr.getType() doesn't work because it is an optional attribute,
  // which can be a nullptr.
  auto type = RankedTensorType::get({rank, 2}, builder->getIntegerType(64));
  return DenseIntElementsAttr::get(type, padding);
}

#include "tensorflow/compiler/mlir/xla/transforms/generated_canonicalize.inc"
}  // namespace

//===----------------------------------------------------------------------===//
// ConstOp
//===----------------------------------------------------------------------===//

static void Print(ConstOp op, OpAsmPrinter* printer) {
  // Print op name.
  *printer << op.getOperationName();

  // Elide attribute value while printing the attribute dictionary.
  SmallVector<StringRef, 1> elided_attrs;
  elided_attrs.push_back("value");
  printer->printOptionalAttrDict(op.getAttrs(), elided_attrs);

  *printer << ' ' << op.value();
}

static ParseResult ParseConstOp(OpAsmParser* parser, OperationState* result) {
  if (parser->parseOptionalAttrDict(result->attributes)) return failure();

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
void ConstOp::build(OpBuilder& builder, OperationState& result,
                    Attribute value) {
  Type type;
  if (auto elemAttr = value.dyn_cast<ElementsAttr>()) {
    type = elemAttr.getType();
  } else if (value.isa<BoolAttr>() || value.isa<FloatAttr>() ||
             value.isa<IntegerAttr>()) {
    // All XLA types must be tensor types. In the build() method, we want to
    // provide more flexibility by allowing attributes of scalar types. But we
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
// DotGeneralOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(DotGeneralOp op) {
  auto dot_dimension_numbers = op.dot_dimension_numbers();
  int64_t lhs_batching_dimensions_size = llvm::size(
      dot_dimension_numbers.lhs_batching_dimensions().getValues<int64_t>());
  int64_t rhs_batching_dimensions_size = llvm::size(
      dot_dimension_numbers.rhs_batching_dimensions().getValues<int64_t>());
  if (lhs_batching_dimensions_size != rhs_batching_dimensions_size) {
    return op.emitError()
           << "lhs and rhs should have the same number of batching dimensions";
  }
  int64_t lhs_contracting_dimensions_size = llvm::size(
      dot_dimension_numbers.lhs_contracting_dimensions().getValues<int64_t>());
  int64_t rhs_contracting_dimensions_size = llvm::size(
      dot_dimension_numbers.rhs_contracting_dimensions().getValues<int64_t>());
  if (lhs_contracting_dimensions_size != rhs_contracting_dimensions_size) {
    return op.emitError() << "lhs and rhs should have the same number of "
                             "contracting dimensions";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// IotaOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(IotaOp op) {
  auto shape = op.getType().cast<ShapedType>();
  if (!shape.hasRank()) return success();

  if (shape.getRank() == 0)
    return op.emitOpError() << "does not support scalars.";

  auto iota_dimension = op.iota_dimension().getSExtValue();
  if (iota_dimension >= shape.getRank() || iota_dimension < 0)
    return op.emitOpError() << "iota dimension cannot go beyond the output "
                               "rank or be negative.";
  return success();
}

//===----------------------------------------------------------------------===//
// DynamicIotaOp
//===----------------------------------------------------------------------===//

namespace {

struct DynamicIotaIsStatic : public OpRewritePattern<DynamicIotaOp> {
  using OpRewritePattern<DynamicIotaOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(DynamicIotaOp iota,
                                PatternRewriter& rewriter) const override {
    auto result_ty = iota.getType().cast<ShapedType>();
    if (!result_ty.hasStaticShape()) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<IotaOp>(iota, result_ty, iota.iota_dimension());
    return success();
  }
};

}  // namespace

void DynamicIotaOp::getCanonicalizationPatterns(
    OwningRewritePatternList& results, MLIRContext* context) {
  results.insert<DynamicIotaIsStatic>(context);
}

//===----------------------------------------------------------------------===//
// AbsOp
//===----------------------------------------------------------------------===//

void AbsOp::build(OpBuilder& builder, OperationState& result, Value operand) {
  auto shaped_type = operand.getType().cast<ShapedType>();
  Type new_type;
  if (!shaped_type.getElementType().isa<ComplexType>()) {
    new_type = operand.getType();
  } else if (shaped_type.hasRank()) {
    new_type = RankedTensorType::get(shaped_type.getShape(), operand.getType());
  } else {
    new_type = UnrankedTensorType::get(operand.getType());
  }

  return AbsOp::build(builder, result, new_type, operand);
}

//===----------------------------------------------------------------------===//
// CollectivePermuteOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(CollectivePermuteOp op) {
  // Check that source target pair is Nx2 tensor.
  auto type = op.source_target_pairs().getType().dyn_cast<RankedTensorType>();
  if (type.getRank() != 2)
    return op.emitError() << "expect source_target_pairs attribute to be of "
                             "rank 2, but got rank "
                          << type.getRank();
  if (type.getShape()[1] != 2)
    return op.emitError()
           << "expect source_target_pairs attribute of shape (N, 2), but got ("
           << type.getShape() << ")";
  // Check source target pairs for duplicate sources or targets
  absl::flat_hash_set<int64_t> sources;
  absl::flat_hash_set<int64_t> targets;
  for (auto i = op.source_target_pairs().begin(),
            e = op.source_target_pairs().end();
       i != e; ++i) {
    auto val = (*i).getSExtValue();
    if (i.getIndex() % 2 == 0) {
      bool is_unique = sources.insert(val).second;
      if (!is_unique) return op.emitError() << "duplicate sources not allowed.";
    } else {
      bool is_unique = targets.insert(val).second;
      if (!is_unique) return op.emitError() << "duplicate targets not allowed.";
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ConvertOp
//===----------------------------------------------------------------------===//

void ConvertOp::build(OpBuilder& builder, OperationState& result, Value operand,
                      Type result_element_ty) {
  Type result_ty;
  Type operand_ty = operand.getType();
  if (auto ranked_ty = operand_ty.dyn_cast<RankedTensorType>()) {
    result_ty = RankedTensorType::get(ranked_ty.getShape(), result_element_ty);
  } else {
    result_ty = UnrankedTensorType::get(result_element_ty);
  }
  build(builder, result, result_ty, operand);
}

OpFoldResult ConvertOp::fold(ArrayRef<Attribute> operands) {
  if (getOperand().getType() == getResult().getType()) return getOperand();

  // If the result has non-static shape, a convert op is necessary to go from
  // static shape to non-static shape.
  if (!getResult().getType().cast<TensorType>().hasStaticShape()) return {};

  // If the operand is constant, we can do the conversion now.
  if (auto elementsAttr = operands.front().dyn_cast_or_null<ElementsAttr>()) {
    return xla::ConvertElementsAttr(elementsAttr,
                                    getElementTypeOrSelf(getResult()));
  }

  return {};
}

//===----------------------------------------------------------------------===//
// DequantizeOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(DequantizeOp op) {
  auto input_type = op.input().getType().dyn_cast<ShapedType>();
  auto output_type = op.output().getType().dyn_cast<ShapedType>();
  if (!input_type || !output_type) {
    return op.emitError() << "ranked input and output.";
  }
  auto input_shape = input_type.getShape();
  auto output_shape = output_type.getShape().vec();
  if (op.transpose_output()) {
    std::reverse(output_shape.begin(), output_shape.end());
  }

  // Check the input rank and output rank are same, and also the lower
  // dimensions are same.
  if (input_shape.size() != output_shape.size() ||
      !std::equal(input_shape.begin(),
                  std::next(input_shape.begin(), input_shape.size() - 1),
                  output_shape.begin())) {
    return op.emitError() << "mismatched dimensions.";
  }

  // Check that the last dimension of the output is 2x or 4x of that of the
  // input depending on the unpacked input is 16 or 8 bits.
  int input_last_dim = *input_shape.rbegin();
  int output_last_dim = *output_shape.rbegin();
  int scale_factor = op.is_16bits() ? 2 : 4;
  if (output_last_dim != scale_factor * input_last_dim) {
    return op.emitError() << "last dimension of output should be "
                          << scale_factor << "x of the input.";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// GetTupleElementOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(GetTupleElementOp op) {
  auto indexVal = op.index().getZExtValue();
  auto operandType = op.getOperand().getType().cast<TupleType>();
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
          dyn_cast_or_null<xla_hlo::TupleOp>(getOperand().getDefiningOp())) {
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
// AllToAllOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(AllToAllOp op) {
  // If operand is ranked, size of split dimension should be a multiple of split
  // count.
  auto type = op.getOperand().getType().dyn_cast<RankedTensorType>();
  if (!type) return success();
  auto split_dim_size = type.getDimSize(op.split_dimension().getSExtValue());
  auto split_count = op.split_count().getSExtValue();
  if (split_dim_size % split_count != 0) {
    return op.emitError() << "split dimension has size " << split_dim_size
                          << ", expected to be a multiple of split_count "
                          << split_count;
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

  auto resultType = op.getResult().getType().cast<RankedTensorType>();
  auto resultRank = resultType.getRank();
  auto operandType = op.operand().getType().cast<RankedTensorType>();
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
  auto operandType = op.operand().getType().dyn_cast<RankedTensorType>();
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

  auto dimensions = op.broadcast_dimensions();
  auto dimensionsType = op.broadcast_dimensions().getType();
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

  auto resultType = op.getResult().getType().cast<RankedTensorType>();
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

OpFoldResult BroadcastInDimOp::fold(ArrayRef<Attribute>) {
  auto type = getType().cast<RankedTensorType>();
  if (type != getOperand().getType()) {
    return nullptr;
  }
  auto broadcast_values = broadcast_dimensions().getValues<int64_t>();
  if (!std::equal(broadcast_values.begin(), broadcast_values.end(),
                  llvm::seq<int64_t>(0, type.getRank()).begin())) {
    return nullptr;
  }
  return getOperand();
}

//===----------------------------------------------------------------------===//
// DynamicBroadcastInDimOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(DynamicBroadcastInDimOp op) {
  auto operandType = op.operand().getType().dyn_cast<RankedTensorType>();
  auto resultType = op.getResult().getType().dyn_cast<RankedTensorType>();

  // If either the operand or result are unranked, there is very little
  // to verify statically.
  if (!operandType || !resultType) {
    return success();
  }

  auto outputDimensionsType =
      op.output_dimensions().getType().cast<RankedTensorType>();
  auto outputDimensionsSize = outputDimensionsType.getDimSize(0);
  auto operandRank = operandType.getRank();
  auto resultRank = resultType.getRank();

  // Verify broadcast_dimensions.
  auto bcastDimensions = op.broadcast_dimensions();
  auto bcastDimensionsType = op.broadcast_dimensions().getType();
  auto bcastDimensionsRank = bcastDimensionsType.getRank();
  // TODO(laurenzo): Update the BroadcastDimAttr to constrain its rank to 1.
  if (bcastDimensionsRank != 1) {
    return op.emitOpError(
        llvm::formatv("broadcast_dimensions has rank {0} instead of rank 1",
                      bcastDimensionsRank));
  }

  auto bcastDimensionsSize = bcastDimensionsType.getNumElements();
  if (bcastDimensionsSize != operandRank) {
    return op.emitOpError(llvm::formatv(
        "broadcast_dimensions size ({0}) does not match operand rank ({1})",
        bcastDimensionsSize, operandRank));
  }

  if (resultRank < operandRank) {
    return op.emitOpError(
        llvm::formatv("result rank ({0}) is less than operand rank ({1})",
                      resultRank, operandRank));
  }

  for (int i = 0; i != bcastDimensionsSize; ++i) {
    auto dimIndex = bcastDimensions.getValue<int64_t>(i);
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

  if (outputDimensionsSize != resultRank) {
    return op.emitOpError(
        llvm::formatv("result rank ({0}) is not equal to number of output "
                      "dimensions ({1})",
                      resultRank, outputDimensionsSize));
  }

  return success();
}

// If a DynamicBroadCastInDimOp is not actually dynamic, use an ordinary
// BroadcastInDimOp.
class DynamicBroadcastInDimOpNotActuallyDynamic
    : public OpRewritePattern<DynamicBroadcastInDimOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(DynamicBroadcastInDimOp op,
                                PatternRewriter& rewriter) const override {
    auto type = op.getType().dyn_cast<RankedTensorType>();
    if (!type || !type.hasStaticShape()) {
      return rewriter.notifyMatchFailure(op, "requires static shape");
    }
    rewriter.replaceOpWithNewOp<BroadcastInDimOp>(
        op, op.getType(), op.operand(), op.broadcast_dimensions());
    return success();
  }
};

void DynamicBroadcastInDimOp::getCanonicalizationPatterns(
    OwningRewritePatternList& results, MLIRContext* context) {
  results.insert<DynamicBroadcastInDimOpNotActuallyDynamic>(context);
}

//===----------------------------------------------------------------------===//
// ClampOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(ClampOp op) {
  auto operandType = op.operand().getType().cast<RankedTensorType>();
  auto operandShape = operandType.getShape();
  auto minType = op.min().getType().cast<RankedTensorType>();

  auto minShape = minType.getShape();
  if (minShape != operandShape && minType.getRank() != 0) {
    return op.emitOpError(llvm::formatv(
        "min shape [{0}] is not scalar and does not match operand shape [{1}]",
        llvm::make_range(minShape.begin(), minShape.end()),
        llvm::make_range(operandShape.begin(), operandShape.end())));
  }

  auto maxType = op.max().getType().cast<RankedTensorType>();
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
// ComplexOp
//===----------------------------------------------------------------------===//

void ComplexOp::build(OpBuilder& builder, OperationState& state, Value lhs,
                      Value rhs) {
  auto type = lhs.getType();
  auto element_ty = ComplexType::get(getElementTypeOrSelf(type));
  Type result_ty;
  if (auto ranked_type = type.dyn_cast<RankedTensorType>()) {
    result_ty = RankedTensorType::get(ranked_type.getShape(), element_ty);
  } else if (type.isa<UnrankedTensorType>()) {
    result_ty = UnrankedTensorType::get(element_ty);
  } else {
    result_ty = element_ty;
  }

  build(builder, state, result_ty, lhs, rhs);
}

OpFoldResult ComplexOp::fold(ArrayRef<Attribute> operands) {
  auto real_op =
      dyn_cast_or_null<xla_hlo::RealOp>(getOperand(0).getDefiningOp());
  auto imag_op =
      dyn_cast_or_null<xla_hlo::ImagOp>(getOperand(1).getDefiningOp());
  if (real_op && imag_op && real_op.getOperand() == imag_op.getOperand()) {
    return real_op.getOperand();
  }

  return {};
}

namespace {
Type CreateRealType(Type type) {
  auto element_ty = getElementTypeOrSelf(type);
  if (auto complex_ty = element_ty.dyn_cast<ComplexType>()) {
    element_ty = complex_ty.getElementType();
  }

  if (auto ranked_type = type.dyn_cast<RankedTensorType>()) {
    return RankedTensorType::get(ranked_type.getShape(), element_ty);
  } else if (type.dyn_cast<UnrankedTensorType>()) {
    return UnrankedTensorType::get(element_ty);
  }

  return element_ty;
}
}  // namespace

void ImagOp::build(OpBuilder& builder, OperationState& state, Value val) {
  build(builder, state, CreateRealType(val.getType()), val);
}

OpFoldResult ImagOp::fold(ArrayRef<Attribute> operands) {
  if (auto complex_op =
          dyn_cast_or_null<xla_hlo::ComplexOp>(getOperand().getDefiningOp())) {
    return complex_op.getOperand(1);
  }

  return {};
}

void RealOp::build(OpBuilder& builder, OperationState& state, Value val) {
  build(builder, state, CreateRealType(val.getType()), val);
}

OpFoldResult RealOp::fold(ArrayRef<Attribute> operands) {
  if (auto complex_op =
          dyn_cast_or_null<xla_hlo::ComplexOp>(getOperand().getDefiningOp())) {
    return complex_op.getOperand(0);
  }

  return {};
}

//===----------------------------------------------------------------------===//
// ConcatenateOp
//===----------------------------------------------------------------------===//

namespace {
class ConcatenateOperandRemoval : public OpRewritePattern<ConcatenateOp> {
 public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ConcatenateOp op,
                                PatternRewriter& rewriter) const override {
    auto axis = op.dimension().getLimitedValue();
    llvm::SmallVector<Value, 6> new_operands;
    for (auto operand : op.getOperands()) {
      auto ty = operand.getType().cast<ShapedType>();
      if (ty.getDimSize(axis) != 0) {
        new_operands.push_back(operand);
      }
    }

    if (!new_operands.empty() && new_operands.size() < op.getNumOperands()) {
      rewriter.replaceOpWithNewOp<ConcatenateOp>(op, op.getResult().getType(),
                                                 new_operands, op.dimension());
      return success();
    }

    return failure();
  }
};
}  // namespace

LogicalResult ConcatenateOp::inferReturnTypes(
    MLIRContext*, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  if (operands.empty()) {
    return failure();
  }

  auto dimension_attr = attributes.get("dimension").cast<IntegerAttr>();
  auto dimension = dimension_attr.getInt();

  auto first_type = (*operands.begin()).getType().cast<ShapedType>();
  auto out_element = first_type.getElementType();

  for (auto operand : operands.getTypes()) {
    auto element_type = getElementTypeOrSelf(operand);
    if (element_type != out_element) {
      return failure();
    }
  }

  // If an input is unranked the output shape is unranked.
  if (!first_type.hasRank()) {
    inferredReturnTypes.push_back(UnrankedTensorType::get(out_element));
    return success();
  }

  auto out_shape = llvm::to_vector<6>(first_type.getShape());
  out_shape[dimension] = 0;

  for (auto operand : operands.getTypes()) {
    auto type = operand.cast<ShapedType>();
    if (!type.hasRank()) {
      inferredReturnTypes.push_back(UnrankedTensorType::get(out_element));
      return success();
    }

    // If the dimension is dynamic we know the output dimension is dynamic.
    auto dim = type.getShape()[dimension];
    if (dim == -1) {
      out_shape[dimension] = -1;
      break;
    }

    out_shape[dimension] += dim;
  }

  inferredReturnTypes.push_back(RankedTensorType::get(out_shape, out_element));

  return success();
}

void ConcatenateOp::getCanonicalizationPatterns(
    OwningRewritePatternList& results, MLIRContext* context) {
  results.insert<ConcatenateOperandRemoval>(context);
}

template <typename T>
static Attribute foldConcatenateHelper(ConcatenateOp* op,
                                       ArrayRef<Attribute> operands) {
  auto axis = op->dimension().getLimitedValue();
  auto type = op->getType().cast<ShapedType>();

  SmallVector<T, 6> values;
  auto shape = type.getShape();

  size_t top_size = 1;
  for (int i = 0; i < axis; i++) {
    top_size = top_size * shape[i];
  }

  for (size_t i = 0; i < top_size; i++) {
    for (auto operand : operands) {
      DenseElementsAttr attr = operand.cast<DenseElementsAttr>();
      size_t bottom_size = attr.getNumElements() / top_size;
      auto iter = attr.getValues<T>().begin() + i * bottom_size;
      values.append(iter, iter + bottom_size);
    }
  }

  return DenseElementsAttr::get(type, values);
}

static Attribute foldConcatenate(ConcatenateOp* op,
                                 ArrayRef<Attribute> operands) {
  for (auto operand : operands) {
    if (!operand) return {};
  }

  auto type = op->getResult().getType().cast<ShapedType>();
  auto etype = type.getElementType();
  if (etype.isa<IntegerType>()) {
    return foldConcatenateHelper<APInt>(op, operands);
  }

  if (etype.isa<FloatType>()) {
    return foldConcatenateHelper<APFloat>(op, operands);
  }

  return {};
}

OpFoldResult ConcatenateOp::fold(ArrayRef<Attribute> operands) {
  if (getNumOperands() == 1) return getOperand(0);

  ShapedType type = getResult().getType().cast<ShapedType>();
  if (!type.hasStaticShape()) return {};

  auto axis = dimension().getLimitedValue();
  if (auto attr = foldConcatenate(this, operands)) {
    return attr;
  }

  llvm::SmallVector<Value, 6> new_operands;
  for (auto operand : getOperands()) {
    auto ty = operand.getType().cast<ShapedType>();
    if (ty.getDimSize(axis) != 0) {
      return {};
    }
  }

  return DenseElementsAttr::get(type, ArrayRef<Attribute>());
}

static LogicalResult Verify(ConcatenateOp op) {
  Type element_type = getElementTypeOrSelf(op.getOperand(0).getType());
  RankedTensorType first_ranked_type;
  int num_operands = op.getNumOperands();
  for (int i = 0; i < num_operands; i++) {
    auto second_type = op.getOperand(i).getType().dyn_cast<ShapedType>();
    if (second_type.getElementType() != element_type) {
      return op.emitOpError(
          llvm::formatv("operands (0) and ({0}) do not match element type", i));
    }

    if (!second_type.hasRank()) {
      continue;
    }

    if (!first_ranked_type) {
      first_ranked_type = second_type.cast<RankedTensorType>();
      continue;
    }

    if (first_ranked_type.getRank() != second_type.getRank()) {
      return op.emitOpError(
          llvm::formatv("operands (0) and ({0}) do not match rank", i));
    }

    auto first_shape = second_type.getShape();
    auto second_shape = second_type.getShape();
    for (int d = 0; d < first_ranked_type.getRank(); ++d) {
      if (first_shape[d] != second_shape[d] && d != op.dimension()) {
        return op.emitOpError(llvm::formatv(
            "operands (0) and ({0}) non-concat dimensions do not match "
            "({1}) != ({2})",
            i, llvm::make_range(first_shape.begin(), first_shape.end()),
            llvm::make_range(second_shape.begin(), second_shape.end())));
      }
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// DynamicReshapeOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(DynamicReshapeOp op) {
  auto result_type = op.result().getType().dyn_cast<RankedTensorType>();
  auto output_shape_type =
      op.output_shape().getType().dyn_cast<RankedTensorType>();
  if (result_type && output_shape_type && output_shape_type.hasStaticShape() &&
      output_shape_type.getDimSize(0) != result_type.getRank()) {
    return op.emitError() << "output should have a rank equal to the number of "
                             "elements in output_shape";
  }
  return success();
}

namespace {
class DynamicReshapeOpNotActuallyDynamic
    : public OpRewritePattern<DynamicReshapeOp> {
 public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(DynamicReshapeOp op,
                                PatternRewriter& rewriter) const override {
    auto type = op.result().getType().dyn_cast<RankedTensorType>();
    if (!type || !type.hasStaticShape()) {
      return rewriter.notifyMatchFailure(op, "requires static shape tensor");
    }
    rewriter.replaceOpWithNewOp<ReshapeOp>(op, op.getType(), op.operand());
    return success();
  }
};
}  // namespace

void DynamicReshapeOp::getCanonicalizationPatterns(
    OwningRewritePatternList& results, MLIRContext* context) {
  results.insert<DynamicReshapeOpNotActuallyDynamic>(context);
}

//===----------------------------------------------------------------------===//
// DynamicSliceOp
//===----------------------------------------------------------------------===//

namespace {
// Canonicalizes DynamicSlice ops that can be replaced instead with Slice ops.
// This canonicalization is applied the case when the `begin` input values are
// compile time constants and thus can be made into a tensor.
struct DynamicSliceToSlice : public OpRewritePattern<DynamicSliceOp> {
  using OpRewritePattern<DynamicSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(DynamicSliceOp dynamic_slice,
                                PatternRewriter& rewriter) const override {
    Value input = dynamic_slice.operand();
    auto input_tensor = input.getType().dyn_cast<RankedTensorType>();
    if (!input_tensor) return failure();

    SmallVector<int64_t, 4> temp_start_indices;
    for (Value start : dynamic_slice.start_indices()) {
      APInt val;
      if (!matchPattern(start, m_ConstantInt(&val))) {
        return failure();
      }
      temp_start_indices.push_back(*(val.getRawData()));
    }

    // At this point we've determined that the start indices are all constants;
    // pack them into a single tensor.
    auto loc = dynamic_slice.getLoc();
    int64_t input_rank = input_tensor.getRank();
    auto slice_start_indices =
        GetI64ElementsAttr(temp_start_indices, &rewriter);
    DenseIntElementsAttr slice_limits = BuildSliceLimits(
        slice_start_indices, dynamic_slice.slice_sizes(), &rewriter);
    DenseIntElementsAttr slice_strides =
        GetI64ElementsAttr(SmallVector<int64_t, 4>(input_rank, 1), &rewriter);
    auto result = rewriter.create<SliceOp>(loc, input, slice_start_indices,
                                           slice_limits, slice_strides);
    rewriter.replaceOp(dynamic_slice, {result});
    return success();
  }
};

}  // namespace

void DynamicSliceOp::getCanonicalizationPatterns(
    OwningRewritePatternList& results, MLIRContext* context) {
  results.insert<DynamicSliceToSlice>(context);
}

// Verifies that the number of slice sizes and the number of start indices match
static LogicalResult Verify(DynamicSliceOp op) {
  int num_slice_sizes = op.slice_sizes().getNumElements();
  int num_start_indices = op.start_indices().size();
  if (num_start_indices != num_slice_sizes) {
    return op.emitOpError()
           << "has mismatched number of slice sizes (" << num_slice_sizes
           << ") and number of start indices (" << num_start_indices << ")";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// InfeedOp
//===----------------------------------------------------------------------===//

// Checks that the result type is of the form `tuple< any_type, token >`.
static LogicalResult Verify(InfeedOp op) {
  auto result_ty = op.getResult().getType().cast<TupleType>();
  auto subtypes = result_ty.getTypes();
  if (subtypes.size() != 2)
    return op.emitOpError()
           << "result is expected to be a tuple of size 2, but got "
           << subtypes.size();
  if (!subtypes[1].isa<TokenType>())
    return op.emitOpError() << "second element of result tuple is expected to "
                               "be of token type, but got "
                            << subtypes[1];
  return success();
}

//===----------------------------------------------------------------------===//
// MapOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(MapOp op) {
  // Checks if the number of `operands` match the arity of the map `computation`
  // region.
  auto& computation_block = op.computation().front();
  auto computation_args = computation_block.getArguments();
  if (op.operands().size() != computation_args.size())
    return op.emitOpError()
           << "expects number of operands to match the arity "
              "of map computation, but got: "
           << op.operands().size() << " and " << computation_args.size();

  // The parameters of computation should all be scalars and match the element
  // type of operands.
  auto operand_type = op.operands()[0].getType().cast<TensorType>();
  auto operand_elem_ty = operand_type.getElementType();

  for (auto indexed_arg : llvm::enumerate(computation_args)) {
    auto arg_type = indexed_arg.value().getType().dyn_cast<TensorType>();
    if (!arg_type || arg_type.getRank() != 0)
      return op.emitOpError()
             << "computation arguments must be 0-rank tensor, but got: arg #"
             << indexed_arg.index() << " of type "
             << indexed_arg.value().getType();
    if (arg_type.getElementType() != operand_elem_ty) {
      return op.emitOpError()
             << "element type of operands and computation arguments must "
                "match, but got: "
             << operand_elem_ty << " and " << arg_type.getElementType();
    }
  }

  // Mapped computation must return single output
  auto computation_outputs = computation_block.getTerminator()->getOperands();
  if (computation_outputs.size() != 1)
    return op.emitOpError()
           << "computation must return single output, but got: "
           << computation_outputs.size();

  // The output of computation must be scalar and have the same element type
  // as op result.
  auto computation_output_type =
      computation_outputs[0].getType().dyn_cast<TensorType>();
  if (!computation_output_type || computation_output_type.getRank() != 0)
    return op.emitOpError()
           << "computation must return 0-rank tensor, but got: "
           << computation_outputs[0].getType();

  auto result_type = op.getType().cast<TensorType>();
  if (computation_output_type.getElementType() != result_type.getElementType())
    return op.emitOpError() << "element type of result and computation output "
                               "must match, but got: "
                            << result_type.getElementType() << " and "
                            << computation_output_type.getElementType();

  // Checks that the requested map dimension numbers are monotonically
  // increasing.
  auto values = op.dimensions().getValues<int64_t>();
  auto dimensions = std::vector<int64_t>{values.begin(), values.end()};
  for (int i = 0; i < dimensions.size(); ++i) {
    if (dimensions[i] != i)
      return op.emitOpError() << "requires monotonically increasing dimension "
                                 "numbers, but got: "
                              << op.dimensions();
  }

  // Checks that number of dimensions of operands matches the size of
  // `dimensions` since we currently only support mapping across all
  // dimensions: i.e., scalar map functions.
  if (operand_type.hasRank()) {
    if (dimensions.size() != operand_type.getShape().size())
      return op.emitOpError()
             << "applied to a subset of dimensions currently not supported: "
                "operand dimensions = "
             << operand_type.getShape().size()
             << ", requested map dimensions size = " << dimensions.size();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// RecvOp
//===----------------------------------------------------------------------===//

// Checks that the result type is of the form `tuple<any_type, xla_hlo::token>`
static LogicalResult Verify(RecvOp op) {
  auto result_ty = op.getResult().getType().cast<TupleType>();
  auto subtypes = result_ty.getTypes();
  if (subtypes.size() != 2)
    return op.emitOpError()
           << "result is expected to be a tuple of size 2, but got "
           << subtypes.size();
  if (!subtypes[1].isa<TokenType>())
    return op.emitOpError() << "second element of result tuple is expected to "
                               "be of token type, but got "
                            << subtypes[1];
  return success();
}

//===----------------------------------------------------------------------===//
// CopyOp
//===----------------------------------------------------------------------===//

OpFoldResult CopyOp::fold(ArrayRef<Attribute> operands) { return getOperand(); }

//===----------------------------------------------------------------------===//
// ReverseOp
//===----------------------------------------------------------------------===//

OpFoldResult ReverseOp::fold(ArrayRef<Attribute> operands) {
  auto input = operand();

  // No dimensions to reverse.
  if (dimensions().getNumElements() == 0) return input;

  llvm::SmallVector<APInt, 5> new_dims;
  new_dims.reserve(dimensions().getNumElements());

  auto shaped_type = input.getType().cast<ShapedType>();
  for (auto dim : dimensions().getValues<APInt>()) {
    if (shaped_type.getDimSize(dim.getLimitedValue()) != 1) {
      return nullptr;
    }
  }

  return input;
}

//===----------------------------------------------------------------------===//
// ReduceOp
//===----------------------------------------------------------------------===//

// Returns the result type after reducing operand of the given type across the
// specified dimensions.
static TensorType GetReduceResultType(Type operand_ty,
                                      DenseIntElementsAttr dimensions,
                                      Builder* builder) {
  Type element_ty = getElementTypeOrSelf(operand_ty);

  auto ranked_ty = operand_ty.dyn_cast<RankedTensorType>();
  if (!ranked_ty) return UnrankedTensorType::get(element_ty);

  int64_t rank = ranked_ty.getRank();
  llvm::SmallVector<bool, 4> dims_mask(rank, false);
  for (int64_t dim : dimensions.getValues<int64_t>()) dims_mask[dim] = true;

  SmallVector<int64_t, 4> shape;
  for (int64_t i = 0; i < rank; ++i) {
    if (!dims_mask[i]) shape.push_back(ranked_ty.getDimSize(i));
  }

  return RankedTensorType::get(shape, element_ty);
}

void ReduceOp::build(OpBuilder& builder, OperationState& state,
                     ValueRange operands, ValueRange init_values,
                     DenseIntElementsAttr dimensions) {
  SmallVector<Type, 1> result_ty;
  result_ty.reserve(operands.size());

  for (Value operand : operands) {
    result_ty.push_back(
        GetReduceResultType(operand.getType(), dimensions, &builder));
  }
  build(builder, state, result_ty, operands, init_values, dimensions);
}

LogicalResult ReduceOp::fold(ArrayRef<Attribute> operands,
                             SmallVectorImpl<OpFoldResult>& results) {
  // No dimensions to reduce.
  if (dimensions().getNumElements() == 0) {
    for (Value input : this->operands()) {
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
  // TODO(jpienaar): Update to allow broadcastable and unranked inputs. This
  // corresponds to the client side HLO.
  return success();
}

// Makes it such that a SelectOp that is a non-root operation in a DRR infers
// the return type based on operand type.
LogicalResult SelectOp::inferReturnTypes(
    MLIRContext*, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  auto x_type = operands[1].getType();
  auto y_type = operands[2].getType();
  auto x_tensor = x_type.cast<TensorType>();
  auto y_tensor = y_type.cast<TensorType>();

  // Check for type compatibility in the select op. This requires that the two
  // non-predicate operands:
  //   (a) have the same element type
  //   (b) have compatible shapes (i.e. the same shape and/or at least one
  //       dynamic shape)
  if (x_tensor.getElementType() != y_tensor.getElementType() ||
      failed(mlir::verifyCompatibleShape(x_type, y_type))) {
    return emitOptionalError(location, "incompatible operand types: ", x_type,
                             " and ", y_type);
  }

  // TODO(lucyfox): Support output shape inference when operands have compatible
  // shapes. (The output shape should be the most general of the operand shapes
  // at each dimension.) For now, handle the straightforward cases and fail
  // otherwise. When this is fully implemented, this logic should move into
  // reusable functionality in MLIR Core.
  Type output_type;
  if (x_type == y_type || !x_tensor.hasRank()) {
    output_type = x_type;
  } else if (!y_tensor.hasRank()) {
    output_type = y_type;
  } else {
    return emitOptionalError(location,
                             "currently unsupported operand types: ", x_type,
                             " and ", y_type);
  }
  inferredReturnTypes.assign({output_type});
  return success();
}

//===----------------------------------------------------------------------===//
// PadOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(PadOp op) {
  auto input_type = op.operand().getType().cast<RankedTensorType>();
  auto pad_type = op.padding_value().getType().cast<RankedTensorType>();

  if (pad_type.getRank() != 0) {
    return op.emitOpError(
        llvm::formatv("padding value type should be a rank-0 "
                      "tensor, is rank {0}",
                      pad_type.getRank()));
  }

  const auto& padding_low = op.edge_padding_low();
  if (padding_low.getType().getNumElements() != input_type.getRank()) {
    return op.emitOpError(llvm::formatv(
        "edge_padding_low length ({0}) must match operand rank ({1})",
        padding_low.getType().getNumElements(), input_type.getRank()));
  }

  const auto& padding_high = op.edge_padding_high();
  if (padding_high.getType().getNumElements() != input_type.getRank()) {
    return op.emitOpError(llvm::formatv(
        "edge_padding_high length ({0}) must match operand rank ({1})",
        padding_high.getType().getNumElements(), input_type.getRank()));
  }

  const auto& padding_interior = op.interior_padding();
  if (padding_interior.getType().getNumElements() != input_type.getRank()) {
    return op.emitOpError(llvm::formatv(
        "interior_padding length ({0}) must match operand rank ({1})",
        padding_interior.getType().getNumElements(), input_type.getRank()));
  }

  auto input_shape = input_type.getShape();
  auto output_shape =
      op.getResult().getType().cast<RankedTensorType>().getShape();
  if (input_shape.size() != output_shape.size()) {
    return op.emitOpError(
        llvm::formatv("operand rank ({0}) and result rank({0}) should match",
                      input_shape.size(), output_shape.size()));
  }

  for (int i = 0, e = input_shape.size(); i < e; i++) {
    int padding_low_val = padding_low.getValue<IntegerAttr>(i).getInt();
    int padding_high_val = padding_high.getValue<IntegerAttr>(i).getInt();
    int padding_interior_val =
        padding_interior.getValue<IntegerAttr>(i).getInt();
    int expected_output =
        input_shape[i] + padding_low_val + padding_high_val +
        std::max<int64_t>(input_shape[i] - 1, 0LL) * padding_interior_val;
    if (expected_output != output_shape[i]) {
      return op.emitOpError(llvm::formatv(
          "expected output shape's dimension #{0} to be {1} but found {2}", i,
          expected_output, output_shape[i]));
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(ReshapeOp op) {
  // If the operand type is dynamically shaped there is nothing to verify.
  auto operand_ty = op.operand().getType().cast<RankedTensorType>();
  if (!operand_ty || !operand_ty.hasStaticShape()) return success();

  // If the operand type is statically shaped (not required) the number of
  // elements must match that of the result type.
  auto result_ty = op.getType().cast<RankedTensorType>();
  assert(result_ty && result_ty.hasStaticShape() &&
         "result type must be statically shaped");
  int64_t num_result_elements = result_ty.getNumElements();
  int64_t num_operand_elements = operand_ty.getNumElements();
  if (num_result_elements != num_operand_elements)
    return op.emitOpError()
           << "number of output elements (" << num_result_elements
           << ") doesn't match expected number of elements ("
           << num_operand_elements << ")";

  return success();
}

OpFoldResult ReshapeOp::fold(ArrayRef<Attribute> operands) {
  if (getOperand().getType() == getType()) {
    return getOperand();
  }

  if (auto prev_op =
          dyn_cast_or_null<ReshapeOp>(getOperand().getDefiningOp())) {
    setOperand(prev_op.getOperand());
    return getResult();
  }

  if (auto elements = operands.front().dyn_cast_or_null<DenseElementsAttr>()) {
    return elements.reshape(getResult().getType().cast<ShapedType>());
  }

  return {};
}

//===----------------------------------------------------------------------===//
// Case Op
//===----------------------------------------------------------------------===//

static LogicalResult Verify(CaseOp op) {
  auto num_branches = op.branches().size();
  if (op.branch_operands().size() != num_branches)
    return op.emitOpError() << "expects number of branches " << num_branches
                            << " to be same as number of branch operands "
                            << op.branch_operands().size();

  MutableArrayRef<Region> branches = op.branches();
  OperandRange branch_operands = op.branch_operands();
  for (unsigned i = 0; i < num_branches; ++i) {
    mlir::Region& branch_region = branches[i];
    if (branch_region.empty())
      return op.emitOpError() << "cannot have empty regions";
    mlir::Block& entry_block = branch_region.front();
    if (entry_block.getNumArguments() != 1)
      return op.emitOpError()
             << "expects branch regions to have single argument, but found "
             << entry_block.getNumArguments() << " for branch " << i;
    auto operand = branch_operands[i];
    if (entry_block.getArgument(0).getType() != operand.getType())
      return op.emitOpError()
             << "expects operand " << i + 1 << " to be of type "
             << entry_block.getArgument(0).getType() << ", but found "
             << operand.getType();
    WalkResult walker = branch_region.walk([&](ReturnOp return_op) {
      if (return_op.getOperands().getTypes() != op.getResultTypes())
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (walker.wasInterrupted())
      return op.emitOpError()
             << "branch " << i
             << " returned values do not match op result types";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// BinaryOps
//===----------------------------------------------------------------------===//

namespace {

// Updates the element type of a (presumed) tensor type 'x', returning either
// a permuted UnrankedTensorType or RankedTensorType.
static Type UpdateResultElementType(Builder* builder, Type x,
                                    Type element_type) {
  auto x_ranked = x.dyn_cast<RankedTensorType>();
  if (!x_ranked) {
    return UnrankedTensorType::get(element_type);
  }

  auto shape_x = x_ranked.getShape();
  return RankedTensorType::get(shape_x, element_type);
}
}  // namespace

template <typename Op, typename ElementType = Type, typename ValType,
          typename Convert>
static Attribute BinaryFolder(Op* op, ArrayRef<Attribute> attrs) {
  if (!attrs[0] || !attrs[1]) return {};

  DenseElementsAttr lhs = attrs[0].dyn_cast<DenseElementsAttr>();
  DenseElementsAttr rhs = attrs[1].dyn_cast<DenseElementsAttr>();
  if (!lhs || !rhs) return {};

  ShapedType type = op->getType().template cast<ShapedType>();
  if (!type.hasStaticShape()) {
    return {};
  }

  Type etype = type.getElementType();

  // Evaluate for integer values.
  if (!etype.isa<ElementType>()) {
    return {};
  }

  SmallVector<ValType, 6> values;
  values.reserve(lhs.getNumElements());
  for (const auto zip :
       llvm::zip(lhs.getValues<ValType>(), rhs.getValues<ValType>())) {
    values.push_back(Convert()(std::get<0>(zip), std::get<1>(zip)));
  }

  return DenseElementsAttr::get(type, values);
}

template <typename T>
struct divide : std::divides<T> {};

template <>
struct divide<APInt> {
  APInt operator()(const APInt& a, const APInt& b) const { return a.sdiv(b); }
};

template <typename T>
struct max {
  T operator()(const T& a, const T& b) const { return std::max<T>(a, b); }
};

template <>
struct max<APInt> {
  APInt operator()(const APInt& a, const APInt& b) const {
    return llvm::APIntOps::smax(a, b);
  }
};

template <typename T>
struct min {
  T operator()(const T& a, const T& b) const { return std::min<T>(a, b); }
};

template <>
struct min<APInt> {
  APInt operator()(const APInt& a, const APInt& b) const {
    return llvm::APIntOps::smin(a, b);
  }
};

#define BINARY_FOLDER(Op, Func)                                                \
  OpFoldResult Op::fold(ArrayRef<Attribute> attrs) {                           \
    if (getElementTypeOrSelf(getType()).isa<FloatType>())                      \
      return BinaryFolder<Op, FloatType, APFloat, Func<APFloat>>(this, attrs); \
    if (getElementTypeOrSelf(getType()).isa<IntegerType>())                    \
      return BinaryFolder<Op, IntegerType, APInt, Func<APInt>>(this, attrs);   \
    return {};                                                                 \
  }

// Addition, subtraction and multiplication use the std:: versions of the ops.
// Due to the other ops behaving differently in signed vs unsigned integers,
// APInts need a special implementation. Currently, it replicates signed int
// op behavior.
BINARY_FOLDER(AddOp, std::plus);
BINARY_FOLDER(SubOp, std::minus);
BINARY_FOLDER(MulOp, std::multiplies);
BINARY_FOLDER(DivOp, divide);
BINARY_FOLDER(MaxOp, max);
BINARY_FOLDER(MinOp, min);

#undef BINARY_FOLDER

//===----------------------------------------------------------------------===//
// SliceOp
//===----------------------------------------------------------------------===//

void SliceOp::build(OpBuilder& builder, OperationState& result, Value operand,
                    DenseIntElementsAttr start_indices,
                    DenseIntElementsAttr limit_indices,
                    DenseIntElementsAttr strides) {
  return build(builder, result,
               InferOutputTypes(&builder, operand, start_indices, limit_indices,
                                strides),
               operand, start_indices, limit_indices, strides);
}

template <typename I, typename E>
static void SliceElements(I values, ArrayRef<int64_t> sizes,
                          ArrayRef<int64_t> starts, ArrayRef<int64_t> limits,
                          ArrayRef<int64_t> strides,
                          llvm::SmallVectorImpl<E>* out_values) {
  assert(starts.size() == limits.size());
  assert(starts.size() == strides.size());
  if (starts.empty()) return;

  int64_t start = starts.front();
  int64_t limit = limits.front();
  int64_t stride = strides.front();
  if (starts.size() == 1) {
    for (int i = start; i < limit; i += stride) {
      out_values->push_back(*(values + i));
    }
    return;
  }

  for (; start < limit; start += stride) {
    auto begin = values + start * sizes.front();
    SliceElements<I, E>(begin, sizes.drop_front(), starts.drop_front(),
                        limits.drop_front(), strides.drop_front(), out_values);
  }
}

template <typename I, typename E>
static Attribute FoldSlice(SliceOp* op, I values) {
  auto start = llvm::to_vector<6>(op->start_indices().getValues<int64_t>());
  auto limit = llvm::to_vector<6>(op->limit_indices().getValues<int64_t>());
  auto stride = llvm::to_vector<6>(op->strides().getValues<int64_t>());

  auto result_type = op->operand().getType().cast<ShapedType>();
  if (!result_type.hasStaticShape()) return {};

  auto shape = result_type.getShape();
  int64_t count = result_type.getNumElements();
  // Compute the striding for each dimension.
  llvm::SmallVector<int64_t, 6> sizes;
  sizes.reserve(shape.size());
  for (auto v : shape) {
    count = count / v;
    sizes.push_back(count);
  }

  llvm::SmallVector<E, 6> out_values;
  out_values.reserve(result_type.getNumElements());
  SliceElements<I, E>(values, sizes, start, limit, stride, &out_values);

  return DenseElementsAttr::get(op->getResult().getType().cast<ShapedType>(),
                                out_values);
}

OpFoldResult SliceOp::fold(ArrayRef<Attribute> operands) {
  // Check if the SliceOp is a NoOp operation.
  auto operand_shape = getOperand().getType().cast<ShapedType>().getShape();
  auto result_type = getResult().getType().cast<ShapedType>();
  auto result_shape = result_type.getShape();

  if (result_type.hasStaticShape() && (operand_shape == result_shape)) {
    return getOperand();
  }

  if (operands.empty() || !operands.front()) return {};

  // Evaluate for statically valued inputs.
  DenseElementsAttr elements = operands.front().dyn_cast<DenseElementsAttr>();
  if (!elements) return {};

  auto etype = elements.getType().getElementType();
  if (etype.isa<IntegerType>()) {
    return FoldSlice<DenseElementsAttr::IntElementIterator, APInt>(
        this, elements.getIntValues().begin());
  } else if (etype.isa<FloatType>()) {
    return FoldSlice<
        llvm::mapped_iterator<DenseElementsAttr::IntElementIterator,
                              std::function<APFloat(const APInt&)>>,
        APFloat>(this, elements.getFloatValues().begin());
  }

  return {};
}

namespace {
// In cases where a concat is fed into a slice, it is possible the concat
// can be simplified or bypassed. This checks which inputs to the concat are
// used by the slice, either reducing the number of concatenated values or
// entirely removes the concat.
struct SimplifyConcatSlice : public OpRewritePattern<SliceOp> {
  using OpRewritePattern<SliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SliceOp slice,
                                PatternRewriter& rewriter) const override {
    auto result_ty = slice.getType().cast<ShapedType>();
    if (!result_ty.hasStaticShape()) {
      return failure();
    }

    auto slice_input = slice.operand();
    auto slice_input_ty = slice_input.getType().cast<ShapedType>();
    auto concat = dyn_cast_or_null<ConcatenateOp>(slice_input.getDefiningOp());
    if (!concat) {
      return failure();
    }

    auto dimension = concat.dimension().getSExtValue();

    auto start = slice.start_indices().getIntValues();
    auto limit = slice.limit_indices().getIntValues();

    auto slice_start = (*(start.begin() + dimension)).getSExtValue();
    auto slice_limit = (*(limit.begin() + dimension)).getSExtValue();

    // We need to determine what inputs from the concat affect the slice, and
    // how the bounds of the slice need to be updated for the minimally required
    // inputs.
    int64_t running_size = 0;
    int64_t front_offset = slice_input_ty.getShape()[dimension];

    auto subset_start = concat.operand_end();
    auto subset_end = concat.operand_end();
    for (auto it = concat.operand_begin(); it < concat.operand_end(); ++it) {
      auto input = *it;
      ShapedType input_ty = input.getType().cast<ShapedType>();
      if (input_ty.isDynamicDim(dimension)) {
        return failure();
      }
      auto dim_size = input_ty.getShape()[dimension];

      // If this position is in the slice its the start of the subset and we
      // need to update the start and limit values.
      if (running_size + dim_size > slice_start &&
          subset_start == concat.operand_end()) {
        subset_start = it;
        front_offset = running_size;
      }

      // Determine the last required offset.
      if (running_size < slice_limit) {
        subset_end = it + 1;
      }

      running_size += dim_size;
    }

    auto subset_size = subset_end - subset_start;
    // We need all inputs so no optimization.
    if (subset_size == concat.getNumOperands()) {
      return failure();
    }

    if (subset_size > 1 && !concat.getResult().hasOneUse()) {
      return failure();
    }

    auto concat_range = OperandRange(subset_start, subset_end);
    auto new_concat = rewriter.create<ConcatenateOp>(
        concat.getLoc(), concat_range, concat.dimension());

    llvm::SmallVector<APInt, 6> new_start(start);
    llvm::SmallVector<APInt, 6> new_limit(limit);
    new_start[dimension] -= front_offset;
    new_limit[dimension] -= front_offset;

    auto attr_type = slice.start_indices().getType().cast<ShapedType>();
    auto create = rewriter.create<SliceOp>(
        slice.getLoc(), new_concat,
        DenseIntElementsAttr::get(attr_type, new_start),
        DenseIntElementsAttr::get(attr_type, new_limit), slice.strides());
    rewriter.replaceOp(slice, create.getResult());
    return success();
  }
};
}  // namespace

void SliceOp::getCanonicalizationPatterns(OwningRewritePatternList& results,
                                          MLIRContext* context) {
  results.insert<SimplifyConcatSlice>(context);
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

Type SliceOp::InferOutputTypes(Builder* builder, Value operand,
                               DenseIntElementsAttr start_indices,
                               DenseIntElementsAttr limit_indices,
                               DenseIntElementsAttr strides) {
  Type ty = operand.getType();
  RankedTensorType ranked_ty = ty.dyn_cast<RankedTensorType>();
  if (!ranked_ty) return ty;
  int64_t rank = ranked_ty.getRank();

  // Illegal attributes.
  ShapedType attr_ty = start_indices.getType();
  if (attr_ty.getRank() != 1 || attr_ty.getNumElements() != rank ||
      !attr_ty.getElementType().isSignlessInteger(64) ||
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
  return RankedTensorType::get(shape, ranked_ty.getElementType());
}

//===----------------------------------------------------------------------===//
// SortOp
//===----------------------------------------------------------------------===//

void SortOp::build(OpBuilder& builder, OperationState& state,
                   ValueRange operands, int64_t dimension, bool is_stable) {
  state.addOperands(operands);
  state.addAttribute("dimension", builder.getI64IntegerAttr(dimension));
  state.addAttribute("is_stable", builder.getBoolAttr(dimension));

  SmallVector<Type, 2> element_types;
  element_types.reserve(operands.size());
  for (Value operand : operands) element_types.push_back(operand.getType());
  state.addTypes(builder.getTupleType(element_types));

  state.addRegion();
}

static LogicalResult Verify(SortOp op) {
  Operation::operand_range operands = op.operands();
  if (operands.empty()) return op.emitOpError("requires at least one input");

  // TODO(antiagainst): verify partionally dynamic shapes
  if (llvm::all_of(operands, [](Value operand) {
        return operand.getType().cast<ShapedType>().hasRank();
      })) {
    ArrayRef<int64_t> input_shape =
        (*operands.begin()).getType().cast<ShapedType>().getShape();

    if (llvm::any_of(llvm::drop_begin(operands, 1), [&](Value operand) {
          return operand.getType().cast<ShapedType>().getShape() != input_shape;
        }))
      return op.emitOpError("requires all inputs to have the same dimensions");

    int64_t rank = input_shape.size();
    int64_t cmp_dim = op.dimension().getSExtValue();
    if (cmp_dim < -rank || cmp_dim >= rank)
      return op.emitOpError("dimension attribute value must be in range [-")
             << rank << ", " << rank << "), but found " << cmp_dim;
  }

  Block& block = op.comparator().front();
  size_t num_operands = op.getOperation()->getNumOperands();
  if (block.getNumArguments() != 2 * num_operands)
    return op.emitOpError("comparator block should have ")
           << 2 * num_operands << " arguments";

  for (auto indexed_operand : llvm::enumerate(operands)) {
    int index = indexed_operand.index();
    Type element_type =
        indexed_operand.value().getType().cast<ShapedType>().getElementType();
    Type tensor_type = RankedTensorType::get({}, element_type);
    for (int i : {2 * index, 2 * index + 1}) {
      Type arg_type = block.getArgument(i).getType();
      if (arg_type != tensor_type)
        return op.emitOpError("comparator block argument #")
               << i << " should be of type " << tensor_type << " but got "
               << arg_type;
    }
  }

  return success();
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
  // permutation is an attribute of the op so it has static shape.
  auto permutationType = op.permutation().getType();
  auto permutationRank = permutationType.getRank();
  if (permutationRank != 1) {
    return op.emitOpError(llvm::formatv(
        "permutation has rank {0} instead of rank 1", permutationRank));
  }
  auto permutationSize = permutationType.getNumElements();

  auto operandType = op.operand().getType().dyn_cast<RankedTensorType>();
  if (operandType) {
    auto operandRank = operandType.getRank();
    if (operandRank != permutationSize) {
      return op.emitOpError(llvm::formatv(
          "operand rank ({0}) does not match permutation size ({1})",
          operandRank, permutationSize));
    }
  }

  auto resultType = op.getResult().getType().dyn_cast<RankedTensorType>();
  if (resultType) {
    auto resultRank = resultType.getRank();
    if (resultRank != permutationSize) {
      return op.emitOpError(llvm::formatv(
          "result rank ({0}) does not match permutation size ({1})", resultRank,
          permutationSize));
    }
  }

  if (!resultType || !operandType) return success();

  auto operandRank = operandType.getRank();
  SmallVector<int64_t, 4> expectedShape(operandRank);
  for (int i = 0; i != operandRank; ++i) {
    auto permutedDim = op.permutation().getValue<IntegerAttr>(i).getInt();
    expectedShape[i] = operandType.getDimSize(permutedDim);
  }

  auto expectedType =
      RankedTensorType::get(expectedShape, resultType.getElementType());
  if (failed(verifyCompatibleShape(resultType, expectedType))) {
    return op.emitOpError(llvm::formatv(
        "result type {0} is incompatible with the expected type {1}",
        resultType, expectedType));
  }

  return success();
}

//===----------------------------------------------------------------------===//
// TriangularSolveOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(TriangularSolveOp op) {
  auto a_type = op.a().getType().dyn_cast<RankedTensorType>();

  // Skip verifier if a is unranked tensor.
  if (!a_type) return success();

  // Check that a should have rank >= 2
  auto a_rank = a_type.getRank();
  if (a_rank < 2)
    return op.emitOpError()
           << "operand 'a' must have rank >= 2, but got " << a_type;

  // The two minor dimensions of a must have same size.
  if (a_type.getDimSize(a_rank - 2) != a_type.getDimSize(a_rank - 1))
    return op.emitOpError() << "two minor dimensions of operand 'a' must have "
                               "equal size, but got "
                            << a_type;

  auto b_type = op.b().getType().dyn_cast<RankedTensorType>();
  // If b is unranked skip remaining checks.
  if (!b_type) return success();

  // Check that a and b have same rank.
  auto b_rank = b_type.getRank();
  if (a_rank != b_rank)
    return op.emitOpError() << "operands must have equal rank, but got "
                            << a_type << " and " << b_type;

  // The shared dimension of a and b should match.
  if (a_type.getDimSize(a_rank - 1) !=
      b_type.getDimSize(b_rank - (op.left_side() ? 2 : 1)))
    return op.emitOpError() << "shared dimension of operands 'a' and 'b' does "
                               "not match, but got "
                            << a_type << " and " << b_type;

  // The leading batch dimensions of a and b must be equal.
  auto a_batch_dims = a_type.getShape().drop_back(2);
  auto b_batch_dims = b_type.getShape().drop_back(2);
  if (a_batch_dims != b_batch_dims)
    return op.emitOpError()
           << "leading batch dimensions of the operands must be same, but got "
           << a_type << " and " << b_type;

  // Result and argument b must have same shape.
  auto result_type = op.getType().dyn_cast<RankedTensorType>();
  if (!result_type) return success();
  if (result_type != b_type)
    return op.emitOpError()
           << "result and operand 'b' must have same shape, but got "
           << result_type << " and " << b_type;
  return success();
}

//===----------------------------------------------------------------------===//
// GetTupleElementOp
//===----------------------------------------------------------------------===//

void GetTupleElementOp::build(OpBuilder& builder, OperationState& result,
                              Value tuple, int32_t index) {
  if (auto tuple_type = tuple.getType().dyn_cast<TupleType>()) {
    auto element_type = tuple_type.getType(index);
    build(builder, result, element_type, tuple,
          builder.getI32IntegerAttr(index));
    return;
  }

  build(builder, result, tuple.getType(), tuple,
        builder.getI32IntegerAttr(index));
}

//===----------------------------------------------------------------------===//
// TupleOp
//===----------------------------------------------------------------------===//

void TupleOp::build(OpBuilder& builder, OperationState& result,
                    ValueRange values) {
  SmallVector<Type, 4> types;
  types.reserve(values.size());
  for (auto val : values) {
    types.push_back(val.getType());
  }

  build(builder, result, builder.getTupleType(types), values);
}

//===----------------------------------------------------------------------===//
// UnaryEinsumOp
//===----------------------------------------------------------------------===//

void UnaryEinsumOp::getCanonicalizationPatterns(
    OwningRewritePatternList& results, MLIRContext* context) {
  results.insert<UnaryEinsumToEinsum>(context);
}

//===----------------------------------------------------------------------===//
// CompareOp
//===----------------------------------------------------------------------===//

void CompareOp::build(OpBuilder& builder, OperationState& result, Value lhs,
                      Value rhs, StringAttr comparison_direction) {
  auto new_type =
      UpdateResultElementType(&builder, lhs.getType(), builder.getI1Type());
  build(builder, result, new_type, lhs, rhs, comparison_direction);
}

#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.cc.inc"

//===----------------------------------------------------------------------===//
// xla_hlo Dialect Interfaces
//===----------------------------------------------------------------------===//

namespace {
struct HLOInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;
  // We don't have any special restrictions on what can be inlined into
  // destination regions (e.g. while/conditional bodies). Always allow it.
  bool isLegalToInline(Region* dest, Region* src,
                       BlockAndValueMapping& valueMapping) const final {
    return true;
  }
  // Operations in xla_hlo dialect are always legal to inline since they are
  // pure.
  bool isLegalToInline(Operation*, Region*, BlockAndValueMapping&) const final {
    return true;
  }
};
}  // end anonymous namespace

//===----------------------------------------------------------------------===//
// xla_hlo Dialect Constructor
//===----------------------------------------------------------------------===//

XlaHloDialect::XlaHloDialect(MLIRContext* context)
    : Dialect(getDialectNamespace(), context) {
  addOperations<
#define GET_OP_LIST
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.cc.inc"
      >();
  addInterfaces<HLOInlinerInterface>();
  addTypes<TokenType>();
  // Support unknown operations because not all XLA operations are registered.
  // allowUnknownOperations();
}

Type XlaHloDialect::parseType(DialectAsmParser& parser) const {
  StringRef data_type;
  if (parser.parseKeyword(&data_type)) return Type();

  if (data_type == "token") return TokenType::get(getContext());
  parser.emitError(parser.getNameLoc())
      << "unknown xla_hlo type: " << data_type;
  return nullptr;
}

void XlaHloDialect::printType(Type type, DialectAsmPrinter& os) const {
  if (type.isa<TokenType>()) {
    os << "token";
    return;
  }
  os << "<unknown xla_hlo type>";
}

//===----------------------------------------------------------------------===//
// Shape inference
//===----------------------------------------------------------------------===//

LogicalResult deriveShapeFromFirstOperand(
    OpBuilder* builder, Operation* op,
    SmallVectorImpl<Value>* reifiedReturnShapes) {
  Value operand = op->getOperand(0);
  ShapedType operand_type = operand.getType().dyn_cast<ShapedType>();
  if (!operand_type) {
    op->emitOpError() << "first operand is not a shaped type";
    return failure();
  }
  auto loc = op->getLoc();
  SmallVector<Value, 4> shape_values;
  shape_values.reserve(operand_type.getRank());
  auto shape_scalar_type = builder->getIntegerType(64);
  for (auto element : llvm::enumerate(operand_type.getShape())) {
    if (element.value() == ShapedType::kDynamicSize) {
      Value dim = builder->create<DimOp>(loc, operand, element.index());
      shape_values.push_back(
          builder->create<IndexCastOp>(loc, dim, shape_scalar_type));
    } else {
      shape_values.push_back(builder->create<ConstantOp>(
          loc, builder->getI64IntegerAttr(element.value())));
    }
  }
  *reifiedReturnShapes = SmallVector<Value, 1>{
      builder->create<TensorFromElementsOp>(loc, shape_values)};
  return success();
}

//===----------------------------------------------------------------------===//
// ConvOp
//===----------------------------------------------------------------------===//

void ConvOp::getCanonicalizationPatterns(OwningRewritePatternList& results,
                                         MLIRContext* context) {
  results.insert<FoldPadIntoConv>(context);
}

}  // namespace xla_hlo
}  // namespace mlir
