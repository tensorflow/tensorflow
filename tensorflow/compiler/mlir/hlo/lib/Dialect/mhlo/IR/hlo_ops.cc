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

// This file defines the operations used in the MHLO dialect.

#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <functional>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h.inc"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops_common.h"
#include "mlir-hlo/utils/convert_op_folder.h"
#include "mlir-hlo/utils/hlo_utils.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/InliningUtils.h"

namespace mlir {
#include "hlo_patterns.cc.inc"
}  // namespace mlir

namespace mlir {
namespace mhlo {

Operation* MhloDialect::materializeConstant(OpBuilder& builder, Attribute value,
                                            Type type, Location loc) {
  // HLO dialect constants only support ElementsAttr unlike standard dialect
  // constant which supports all attributes.
  if (value.isa<ElementsAttr>())
    return builder.create<mhlo::ConstOp>(loc, type, value.cast<ElementsAttr>());
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

// Verifies that dimension attribute for the op correctly indexes in operand or
// result shape.
template <typename OpT>
static LogicalResult VerifyDimAttr(OpT op) {
  int64_t rank = -1;
  if (auto ty = op.operand().getType().template dyn_cast<RankedTensorType>()) {
    rank = ty.getRank();
  } else if (auto ty = op.getType().template dyn_cast<RankedTensorType>()) {
    rank = ty.getRank();
  } else {
    return success();
  }

  int64_t dim = op.dimension();
  if (dim < 0 || dim >= rank)
    return op.emitOpError() << "requires dimension attribute in range [0, "
                            << rank << "); found (" << dim << ")";
  return success();
}

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

/// Replaces the given op with the contents of the given single-block region,
/// using the operands of the block terminator to replace operation results.
static void ReplaceOpWithRegion(PatternRewriter& rewriter, Operation* op,
                                Region& region, ValueRange blockArgs = {}) {
  assert(llvm::hasSingleElement(region) && "expected single-block region");
  Block* block = &region.front();
  Operation* terminator = block->getTerminator();
  ValueRange results = terminator->getOperands();
  rewriter.mergeBlockBefore(block, op, blockArgs);
  rewriter.replaceOp(op, results);
  rewriter.eraseOp(terminator);
}

#include "mhlo_canonicalize.inc"

// Common shape function helper for RngNormal and RngUniform.
static LogicalResult rngInferReturnTypeComponents(
    MLIRContext* context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  if (operands.size() != 3)
    return emitOptionalError(location, "expected 3 operands");

  SmallVector<int64_t> shapeVector;
  Value shapeOperand = operands[2];
  auto shapeOperandType = shapeOperand.getType().cast<ShapedType>();
  Type elementType = getElementTypeOrSelf(operands[1]);

  // Match constant shape arguments.
  DenseIntElementsAttr shape;
  if (!matchPattern(shapeOperand, m_Constant(&shape))) {
    if (!shapeOperandType.hasRank()) {
      inferredReturnShapes.emplace_back(elementType);
      return success();
    }
    if (shapeOperandType.getRank() != 1)
      return emitOptionalError(location, "shape operand required to be 1D");
    int size = shapeOperandType.getDimSize(0);
    if (size == ShapedType::kDynamicSize) {
      inferredReturnShapes.emplace_back(elementType);
      return success();
    }
    shapeVector.resize(size, ShapedType::kDynamicSize);
    inferredReturnShapes.emplace_back(shapeVector, elementType);
    return success();
  }

  shapeVector.reserve(shape.size());
  for (const APInt& fp : shape.getIntValues())
    shapeVector.push_back(fp.getSExtValue());
  inferredReturnShapes.emplace_back(shapeVector, elementType);
  return success();
}

}  // namespace

//===----------------------------------------------------------------------===//
// ConstOp
//===----------------------------------------------------------------------===//

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
  assert(type && "unsupported attribute type for building mhlo.constant");
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
// GatherOp
//===----------------------------------------------------------------------===//

// Converts gather ops to slice ops in case we have a single set of constant
// indices.
struct GatherSlice : public OpRewritePattern<GatherOp> {
  using OpRewritePattern<GatherOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GatherOp gather,
                                PatternRewriter& rewriter) const override {
    DenseIntElementsAttr index;
    if (!matchPattern(gather.start_indices(), m_Constant(&index)))
      return failure();

    const auto& dnums = gather.dimension_numbers();
    if (dnums.index_vector_dim().getInt() != 0 || index.getType().getRank() > 1)
      return failure();

    // TODO(tberghammer): Remove when the verifier catches this case what is
    // invalid if all previous condition holds.
    if (index.getNumElements() != dnums.start_index_map().getNumElements())
      return failure();

    auto slice_end =
        llvm::to_vector<8>(gather.slice_sizes().getValues<int64_t>());
    llvm::SmallVector<int64_t, 8> slice_start(slice_end.size(), 0);
    for (auto it : llvm::zip(dnums.start_index_map().getIntValues(),
                             index.getIntValues())) {
      int64_t map_index = std::get<0>(it).getSExtValue();
      int64_t offset = std::get<1>(it).getSExtValue();
      slice_start[map_index] += offset;
      slice_end[map_index] += offset;
    }

    llvm::SmallVector<int64_t, 8> slice_stride(slice_end.size(), 1);
    llvm::SmallVector<int64_t, 8> slice_shape(slice_end.size());
    for (size_t i = 0; i < slice_end.size(); ++i) {
      slice_shape[i] = slice_end[i] - slice_start[i];
    }
    Type element_type = gather.getType().cast<TensorType>().getElementType();
    auto slice_type = RankedTensorType::get(slice_shape, element_type);
    Value result = rewriter.create<SliceOp>(
        gather.getLoc(), slice_type, gather.getOperand(0),
        GetI64ElementsAttr(slice_start, &rewriter),
        GetI64ElementsAttr(slice_end, &rewriter),
        GetI64ElementsAttr(slice_stride, &rewriter));

    if (dnums.collapsed_slice_dims().getNumElements() > 0) {
      auto collapsed_slice_dims = llvm::to_vector<8>(llvm::map_range(
          dnums.collapsed_slice_dims().getIntValues(),
          [](const llvm::APInt& i) { return i.getSExtValue(); }));
      llvm::SmallVector<int64_t, 8> reshape_shape;
      for (size_t i = 0; i < slice_shape.size(); ++i) {
        if (llvm::count(collapsed_slice_dims, i) == 0) {
          reshape_shape.push_back(slice_shape[i]);
        }
      }
      auto reshape_type = RankedTensorType::get(reshape_shape, element_type);
      result =
          rewriter.create<ReshapeOp>(gather.getLoc(), reshape_type, result);
    }

    result.setType(gather.getType());
    rewriter.replaceOp(gather, result);
    return success();
  }
};

void GatherOp::getCanonicalizationPatterns(OwningRewritePatternList& results,
                                           MLIRContext* context) {
  results.insert<GatherSlice>(context);
}

//===----------------------------------------------------------------------===//
// GetDimensionSizeOp
//===----------------------------------------------------------------------===//
//
static LogicalResult Verify(GetDimensionSizeOp op) { return VerifyDimAttr(op); }

/// Fold get_dimension_size when the said shape dimension is a constant.
OpFoldResult GetDimensionSizeOp::fold(ArrayRef<Attribute> attrs) {
  RankedTensorType type = operand().getType().dyn_cast<RankedTensorType>();
  if (!type) return {};

  int32_t dim = dimension();
  if (type.isDynamic(dim)) return {};
  // The result type is always is a 0-d i32 tensor.
  return DenseIntElementsAttr::get<int32_t>(
      getResult().getType().cast<RankedTensorType>(), type.getDimSize(dim));
}

//===----------------------------------------------------------------------===//
// IotaOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(IotaOp op) {
  auto shape = op.getType().cast<ShapedType>();
  if (!shape.hasRank()) return success();

  if (shape.getRank() == 0)
    return op.emitOpError() << "does not support scalars.";

  auto iota_dimension = op.iota_dimension();
  if (iota_dimension >= shape.getRank() || iota_dimension < 0)
    return op.emitOpError() << "iota dimension cannot go beyond the output "
                               "rank or be negative.";
  return success();
}

// Iota operations across multiple dimensions can be reduced to an iota and a
// ranked broadcast.
struct IotaBroadcast : public OpRewritePattern<IotaOp> {
  using OpRewritePattern<IotaOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IotaOp iota,
                                PatternRewriter& rewriter) const override {
    auto result_ty = iota.getType().cast<ShapedType>();
    if (!result_ty.hasRank() || result_ty.getRank() < 2) {
      return failure();
    }

    auto iota_dimension = iota.iota_dimension();

    auto iota_type = RankedTensorType::get(
        {result_ty.getDimSize(iota_dimension)}, result_ty.getElementType());

    auto new_iota = rewriter.create<IotaOp>(iota.getLoc(), iota_type,
                                            rewriter.getI64IntegerAttr(0));

    auto broadcast_attr = DenseIntElementsAttr::get(
        RankedTensorType::get({1}, rewriter.getIntegerType(64)),
        {iota_dimension});
    rewriter.replaceOpWithNewOp<BroadcastInDimOp>(iota, result_ty, new_iota,
                                                  broadcast_attr);
    return success();
  }
};

void IotaOp::getCanonicalizationPatterns(OwningRewritePatternList& results,
                                         MLIRContext* context) {
  results.insert<IotaBroadcast>(context);
}

OpFoldResult IotaOp::fold(ArrayRef<Attribute> operands) {
  auto dimension = iota_dimension();
  auto result_ty = getResult().getType().cast<ShapedType>();
  if (result_ty.hasRank() && result_ty.getDimSize(dimension) == 1) {
    Builder builder(getContext());
    return builder.getZeroAttr(result_ty);
  }

  return {};
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

// Dynamic Iota operations across multiple dimensions can be reduced to an iota
// and a ranked broadcast.
struct DynamicIotaBroadcast : public OpRewritePattern<DynamicIotaOp> {
  using OpRewritePattern<DynamicIotaOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(DynamicIotaOp iota,
                                PatternRewriter& rewriter) const override {
    auto result_ty = iota.getType().cast<ShapedType>();
    if (!result_ty.hasRank() || result_ty.getRank() < 2) {
      return failure();
    }

    auto iota_dimension = iota.iota_dimension();
    auto iota_dimension_int = iota_dimension;

    auto converted_shape = rewriter.create<IndexCastOp>(
        iota.getLoc(),
        RankedTensorType::get(
            iota.output_shape().getType().cast<ShapedType>().getShape(),
            rewriter.getI64Type()),
        iota.output_shape());

    auto sliced_shape = rewriter.create<SliceOp>(
        iota.getLoc(), converted_shape,
        GetI64ElementsAttr(iota_dimension_int, &rewriter),
        GetI64ElementsAttr(iota_dimension_int + 1, &rewriter),
        GetI64ElementsAttr(1, &rewriter));

    auto converted_sliced_shape = rewriter.create<IndexCastOp>(
        iota.getLoc(),
        RankedTensorType::get(
            {1},
            iota.output_shape().getType().cast<ShapedType>().getElementType()),
        sliced_shape);

    auto iota_type = RankedTensorType::get(
        {result_ty.getDimSize(iota_dimension_int)}, result_ty.getElementType());

    auto new_iota = rewriter.create<DynamicIotaOp>(
        iota.getLoc(), iota_type, converted_sliced_shape,
        rewriter.getI64IntegerAttr(0));

    auto broadcast_attr = DenseIntElementsAttr::get(
        RankedTensorType::get({1}, rewriter.getIntegerType(64)),
        {iota_dimension});
    rewriter.replaceOpWithNewOp<DynamicBroadcastInDimOp>(
        iota, result_ty, new_iota, iota.output_shape(), broadcast_attr);
    return success();
  }
};

}  // namespace

void DynamicIotaOp::getCanonicalizationPatterns(
    OwningRewritePatternList& results, MLIRContext* context) {
  results.insert<DynamicIotaIsStatic>(context);
  results.insert<DynamicIotaBroadcast>(context);
}

//===----------------------------------------------------------------------===//
// DynamicUpdateSliceOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(DynamicUpdateSliceOp op) {
  OperandRange indices = op.start_indices();
  if (indices.size() <= 1) return success();

  // Note: start_indices is constrained to Variadic<HLO_ScalarIntTensor>, so it
  // is OK to cast indices to ShapedType here.
  auto idx_tensor = indices.take_front().front().getType().cast<ShapedType>();
  Type first_elem_ty = idx_tensor.getElementType();
  Type elem_ty;

  for (auto idx : llvm::drop_begin(indices, 1)) {
    idx_tensor = idx.getType().cast<ShapedType>();
    elem_ty = idx_tensor.getElementType();

    if (first_elem_ty != elem_ty) {
      return op.emitOpError() << "start indices must have same element type "
                                 "(encountered mismatch: "
                              << first_elem_ty << " vs " << elem_ty << ")";
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// AbsOp
//===----------------------------------------------------------------------===//

LogicalResult AbsOp::inferReturnTypes(
    MLIRContext*, Optional<Location>, ValueRange operands, DictionaryAttr,
    RegionRange, SmallVectorImpl<Type>& inferredReturnTypes) {
  auto operand_ty = (*operands.begin()).getType().cast<ShapedType>();
  Type element_ty = operand_ty.getElementType();
  if (auto complex_ty = element_ty.dyn_cast<ComplexType>()) {
    element_ty = complex_ty.getElementType();
  }

  Type result_ty;
  if (operand_ty.hasRank()) {
    result_ty = RankedTensorType::get(operand_ty.getShape(), element_ty);
  } else {
    result_ty = UnrankedTensorType::get(element_ty);
  }
  inferredReturnTypes.push_back(result_ty);
  return success();
}

//===----------------------------------------------------------------------===//
// CollectivePermuteOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(CollectivePermuteOp op) {
  return mlir::hlo::VerifyCollectivePermuteSourceTargetPairs(
      op, op.source_target_pairs());
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
  auto operand_ty = getOperand().getType().cast<TensorType>();
  auto result_ty = getResult().getType().cast<TensorType>();
  if (operand_ty == result_ty) return getOperand();

  // If the result has non-static shape, a convert op is necessary to go from
  // static shape to non-static shape.
  if (!result_ty.hasStaticShape()) return {};

  // TODO(hinsu): Handle unsigned types.
  if (operand_ty.getElementType().isUnsignedInteger() ||
      result_ty.getElementType().isUnsignedInteger()) {
    return {};
  }

  // If the operand is constant, we can do the conversion now.
  if (auto elementsAttr = operands.front().dyn_cast_or_null<ElementsAttr>()) {
    return hlo::ConvertElementsAttr(elementsAttr,
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
  auto indexVal = op.index();
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
          dyn_cast_or_null<mhlo::TupleOp>(getOperand().getDefiningOp())) {
    return tupleOp.getOperand(index());
  }

  return {};
}

//===----------------------------------------------------------------------===//
// TupleOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(TupleOp op) {
  auto opType = op.getType().dyn_cast<TupleType>();
  if (!opType) return op.emitOpError("tuple op with non-tuple result");
  if (op.getNumOperands() != opType.size())
    return op.emitOpError(
        "number of operands to tuple expected to match number of types in "
        "resultant tuple type");
  for (auto it : llvm::enumerate(
           llvm::zip_first(op.getOperandTypes(), opType.getTypes()))) {
    if (std::get<0>(it.value()) != std::get<1>(it.value()))
      return op.emitOpError("has return type mismatch at ")
             << it.index() << "th value (" << std::get<0>(it.value())
             << " != " << std::get<1>(it.value()) << ")";
  }
  return success();
}

namespace {

// Pattern for unpacking and repacking the same tuple.
struct UnpackRepackSameTuple : public OpRewritePattern<TupleOp> {
  using OpRewritePattern<TupleOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TupleOp op,
                                PatternRewriter& rewriter) const override {
    if (op.val().empty()) return failure();

    Value first_element = op.val().front();
    auto first_element_op =
        dyn_cast_or_null<GetTupleElementOp>(first_element.getDefiningOp());
    if (!first_element_op || first_element_op.indexAttr().getInt() != 0)
      return failure();

    Value tuple_predecessor = first_element_op.getOperand();
    if (tuple_predecessor.getType() != op.getType()) return failure();

    for (auto element_and_idx : llvm::enumerate(op.val().drop_front(1))) {
      auto element_op = dyn_cast_or_null<GetTupleElementOp>(
          element_and_idx.value().getDefiningOp());
      if (!element_op ||
          element_op.indexAttr().getInt() != element_and_idx.index() + 1 ||
          element_op.getOperand() != tuple_predecessor)
        return failure();
    }

    rewriter.replaceOp(op, tuple_predecessor);
    return success();
  }
};

}  // namespace

void TupleOp::getCanonicalizationPatterns(OwningRewritePatternList& results,
                                          MLIRContext* context) {
  results.insert<UnpackRepackSameTuple>(context);
}

//===----------------------------------------------------------------------===//
// AllToAllOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(AllToAllOp op) {
  // If operand is ranked, size of split dimension should be a multiple of split
  // count.
  auto type = op.getOperand().getType().dyn_cast<RankedTensorType>();
  if (!type) return success();
  auto split_dim_size = type.getDimSize(op.split_dimension());
  auto split_count = op.split_count();
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
  if (!operandType) {
    // The following verification checks all depend on knowing the rank of
    // the operand. Bail out now if we don't know the rank of the operand.
    return success();
  }

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
                        "result with rank {1}",
                        dimIndex, resultRank));
    }

    if (!operandType.isDynamicDim(i)) {
      auto dimSize = operandType.getDimSize(i);
      auto resultDimSize = resultType.getDimSize(dimIndex);
      if (dimSize != 1 && dimSize != resultDimSize) {
        return op.emitOpError(
            llvm::formatv("size of operand dimension {0} ({1}) is not equal to "
                          "1 or size of result dimension {2} ({3})",
                          i, dimSize, dimIndex, resultDimSize));
      }
    }
  }

  return success();
}

OpFoldResult BroadcastInDimOp::fold(ArrayRef<Attribute> attrs) {
  auto type = getType().cast<RankedTensorType>();
  if (type == getOperand().getType()) {
    auto broadcast_values = broadcast_dimensions().getValues<int64_t>();
    if (!std::equal(broadcast_values.begin(), broadcast_values.end(),
                    llvm::seq<int64_t>(0, type.getRank()).begin())) {
      return {};
    }
    return getOperand();
  }

  // Constant fold when an operand is a splat tensor attribute.
  if (!attrs[0] || !type.hasStaticShape()) return {};
  auto splatOperandAttr = attrs[0].dyn_cast<SplatElementsAttr>();
  if (!splatOperandAttr) return {};
  // MLIR core bug (https://bugs.llvm.org/show_bug.cgi?id=46588): dense element
  // attribute iterator not implemented for complex element types.
  if (type.getElementType().isa<ComplexType>()) return {};
  return SplatElementsAttr::get(type, splatOperandAttr.getSplatValue());
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
                        "result with rank {1}",
                        dimIndex, resultRank));
    }

    auto dimSize = operandType.getDimSize(i);
    auto resultDimSize = resultType.getDimSize(dimIndex);
    // Note: verifyCompatibleShapes doesn't consider size-1 broadcasting, so we
    // add a manual check for this.
    if (dimSize != 1 && failed(verifyCompatibleShape(dimSize, resultDimSize))) {
      return op.emitOpError(
          llvm::formatv("size of operand dimension {0} ({1}) is not compatible "
                        "with size of result dimension {2} ({3})",
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

namespace {
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

class ChainedDynamicBroadcastInDimCanonicalization
    : public OpRewritePattern<DynamicBroadcastInDimOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(DynamicBroadcastInDimOp bcast,
                                PatternRewriter& rewriter) const override {
    auto preceding_bcast =
        bcast.operand().getDefiningOp<DynamicBroadcastInDimOp>();
    if (!preceding_bcast) return failure();

    // Compose broadcast dimensions.
    DenseIntElementsAttr preceding_bcast_dims =
        preceding_bcast.broadcast_dimensions();
    DenseIntElementsAttr bcast_dims = bcast.broadcast_dimensions();
    SmallVector<APInt, 4> composition;
    for (APInt preceding_dim : preceding_bcast_dims) {
      auto composed_dim = bcast_dims.getValue({preceding_dim.getZExtValue()})
                              .cast<IntegerAttr>();
      composition.push_back(composed_dim.getValue());
    }
    auto composed_bcast_dims =
        DenseIntElementsAttr::get(preceding_bcast_dims.getType(), composition);

    rewriter.replaceOpWithNewOp<DynamicBroadcastInDimOp>(
        bcast, bcast.getType(), preceding_bcast.operand(),
        bcast.output_dimensions(), composed_bcast_dims);
    return success();
  }
};
}  // namespace

void DynamicBroadcastInDimOp::getCanonicalizationPatterns(
    OwningRewritePatternList& results, MLIRContext* context) {
  results.insert<ChainedDynamicBroadcastInDimCanonicalization,
                 DynamicBroadcastInDimOpNotActuallyDynamic,
                 DynamicBroadcastToOwnShape_1, DynamicBroadcastToOwnShape_2,
                 DynamicBroadcastToOwnShape_3, DynamicBroadcastToOwnShape_4>(
      context);
}

LogicalResult DynamicBroadcastInDimOp::inferReturnTypeComponents(
    MLIRContext*, llvm::Optional<mlir::Location>, ValueRange, DictionaryAttr,
    RegionRange, llvm::SmallVectorImpl<mlir::ShapedTypeComponents>&) {
  return failure();
}

LogicalResult DynamicBroadcastInDimOp::reifyReturnTypeShapes(
    OpBuilder&, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  DynamicBroadcastInDimOp::Adaptor adaptor(operands);
  reifiedReturnShapes.push_back(adaptor.output_dimensions());
  return success();
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

LogicalResult ComplexOp::inferReturnTypes(
    MLIRContext*, Optional<Location>, ValueRange operands, DictionaryAttr,
    RegionRange, SmallVectorImpl<Type>& inferredReturnTypes) {
  auto type = operands[0].getType();
  auto element_ty = ComplexType::get(getElementTypeOrSelf(type));
  Type result_ty;
  if (auto ranked_type = type.dyn_cast<RankedTensorType>()) {
    result_ty = RankedTensorType::get(ranked_type.getShape(), element_ty);
  } else if (type.isa<UnrankedTensorType>()) {
    result_ty = UnrankedTensorType::get(element_ty);
  } else {
    result_ty = element_ty;
  }
  inferredReturnTypes.push_back(result_ty);
  return success();
}

OpFoldResult ComplexOp::fold(ArrayRef<Attribute> operands) {
  auto real_op = dyn_cast_or_null<mhlo::RealOp>(getOperand(0).getDefiningOp());
  auto imag_op = dyn_cast_or_null<mhlo::ImagOp>(getOperand(1).getDefiningOp());
  if (real_op && imag_op && real_op.getOperand() == imag_op.getOperand()) {
    return real_op.getOperand();
  }

  return {};
}

//===----------------------------------------------------------------------===//
// ImagOp
//===----------------------------------------------------------------------===//

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

LogicalResult ImagOp::inferReturnTypes(
    MLIRContext*, Optional<Location>, ValueRange operands, DictionaryAttr,
    RegionRange, SmallVectorImpl<Type>& inferredReturnTypes) {
  inferredReturnTypes.push_back(CreateRealType(operands[0].getType()));
  return success();
}

OpFoldResult ImagOp::fold(ArrayRef<Attribute> operands) {
  if (auto complex_op =
          dyn_cast_or_null<mhlo::ComplexOp>(getOperand().getDefiningOp())) {
    return complex_op.getOperand(1);
  }

  return {};
}

//===----------------------------------------------------------------------===//
// IsFiniteOp
//===----------------------------------------------------------------------===//

TensorType getSameShapeTensorType(TensorType tensor_type, Type element_type) {
  if (auto ranked_tensor_ty = tensor_type.dyn_cast<RankedTensorType>()) {
    return RankedTensorType::get(ranked_tensor_ty.getShape(), element_type);
  }
  if (auto unranked_tensor_ty = tensor_type.dyn_cast<UnrankedTensorType>()) {
    return UnrankedTensorType::get(element_type);
  }
  llvm_unreachable("unhandled type");
}

LogicalResult IsFiniteOp::inferReturnTypes(
    MLIRContext* ctx, Optional<Location>, ValueRange operands, DictionaryAttr,
    RegionRange, SmallVectorImpl<Type>& inferredReturnTypes) {
  auto arg_ty = operands.front().getType().cast<TensorType>();
  Builder b(ctx);
  inferredReturnTypes.push_back(getSameShapeTensorType(arg_ty, b.getI1Type()));
  return success();
}

//===----------------------------------------------------------------------===//
// RealOp
//===----------------------------------------------------------------------===//

LogicalResult RealOp::inferReturnTypes(
    MLIRContext*, Optional<Location>, ValueRange operands, DictionaryAttr,
    RegionRange, SmallVectorImpl<Type>& inferredReturnTypes) {
  inferredReturnTypes.push_back(CreateRealType(operands[0].getType()));
  return success();
}

OpFoldResult RealOp::fold(ArrayRef<Attribute> operands) {
  if (auto complex_op =
          dyn_cast_or_null<mhlo::ComplexOp>(getOperand().getDefiningOp())) {
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
    auto axis = op.dimension();
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

  // Find the first ranked input to determine the output rank.
  for (auto type : operands.getTypes()) {
    auto shaped_type = type.cast<ShapedType>();
    if (shaped_type.hasRank()) {
      first_type = shaped_type;
      break;
    }
  }

  // If all inputs are unranked, the result must be unranked.
  if (!first_type.hasRank()) {
    inferredReturnTypes.push_back(UnrankedTensorType::get(out_element));
    return success();
  }

  if (first_type.getRank() == 0)
    return emitOptionalError(location, "rank-0 values cannot be concatenated");

  auto out_shape = llvm::to_vector<6>(first_type.getShape());

  // Determine what the non-concatenate dimensions should be.
  for (auto type : operands.getTypes()) {
    auto shaped_ty = type.cast<ShapedType>();
    if (!shaped_ty.hasRank()) {
      continue;
    }

    for (auto it : llvm::enumerate(shaped_ty.getShape())) {
      // If a dimension is not dynamic, the output shape should match.
      if (ShapedType::isDynamic(out_shape[it.index()])) {
        out_shape[it.index()] = it.value();
      }
    }
  }

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
  auto axis = op->dimension();
  auto type = op->getType().cast<ShapedType>();

  SmallVector<T, 6> values;
  auto shape = type.getShape();

  size_t top_size = 1;
  for (int i = 0, e = axis; i < e; i++) {
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

  auto axis = dimension();
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

// Canonicalizes
// %0 = some_op(%tensor)
// %1 = "mhlo.dynamic_reshape"(%0, %shape)
//      (tensor<?xT>, tensor<1xindex>) -> tensor<?xT>
// ... uses of %1.
//
// into
//
// ... uses of %0.
// This canonicalization is only correct if the input is correct!
// TODO(b/178779691): Use a more sophisticated canonicalization that preserves
// errors in input, and still allows us to get rid of redundant reshapes.
class RemoveRedundantRank1DynamicReshape
    : public OpRewritePattern<DynamicReshapeOp> {
 public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(DynamicReshapeOp op,
                                PatternRewriter& rewriter) const override {
    auto type = op.result().getType().dyn_cast<RankedTensorType>();
    if (!type || type.getRank() != 1 || type.hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          op, "requires rank 1 shape tensor with dynamic dimension");
    }
    auto operand_type = op.operand().getType().dyn_cast<RankedTensorType>();
    if (!operand_type || operand_type.getRank() != 1 ||
        operand_type.hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          op, "requires rank 1 shape tensor with dynamic dimension");
    }
    rewriter.replaceOp(op, {op.operand()});
    return success();
  }
};

// Canonicalizes
// %0 = "mhlo.dynamic_reshape"(%tensor, %shape)
// %1 = same_operands_and_result_shape_op(%tensor)
// %2 = "mhlo.dynamic_reshape"(%1, %shape)
// ... uses of %2.
//
// into
//
// %0 = "mhlo.dynamic_reshape"(%tensor, %shape)
// %1 = same_operands_and_result_shape_op(%tensor)
// ... uses of %1.
class DynamicReshapeOpSameShapeOpResult
    : public OpRewritePattern<DynamicReshapeOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DynamicReshapeOp op,
                                PatternRewriter& rewriter) const override {
    Operation* def_op = op.operand().getDefiningOp();
    if (!def_op || !def_op->hasTrait<OpTrait::SameOperandsAndResultShape>()) {
      return failure();
    }
    Operation* input_def_op = def_op->getOperand(0).getDefiningOp();
    if (!input_def_op) {
      return failure();
    }
    auto reshape = dyn_cast<DynamicReshapeOp>(*input_def_op);
    if (reshape && reshape.output_shape() == op.output_shape()) {
      rewriter.replaceOp(op, {def_op->getResult(0)});
      return success();
    }
    return failure();
  }
};
}  // namespace

void DynamicReshapeOp::getCanonicalizationPatterns(
    OwningRewritePatternList& results, MLIRContext* context) {
  // clang-format off
  results.insert<
      DynamicReshapeOpNotActuallyDynamic,
      DynamicReshapeOpSameShapeOpResult,
      RemoveRedundantDynamicBroadcast,
      RemoveRedundantDynamicReshape,
      RemoveRedundantRank1DynamicReshape,
      ShapeOfDynamicReshape
    >(context);
  // clang-format on
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
// RealDynamicSliceOp
//===----------------------------------------------------------------------===//
// Verifies that operand rank matches start_indices/limit_indices/strides size
static LogicalResult Verify(RealDynamicSliceOp op) {
  auto input_type = op.operand().getType().dyn_cast<RankedTensorType>();
  // If operand is unranked, there is very little to verify statically.
  if (!input_type) return success();
  int input_rank = input_type.getRank();

  auto start_type = op.start_indices().getType().cast<RankedTensorType>();
  auto limit_type = op.limit_indices().getType().cast<RankedTensorType>();
  auto strides_type = op.strides().getType().cast<RankedTensorType>();

  if (input_rank != start_type.getNumElements()) {
    return op.emitOpError() << "has mismatched number of operand rank ("
                            << input_rank << ") and start_indices size ("
                            << start_type.getNumElements() << ")";
  }

  if (input_rank != limit_type.getNumElements()) {
    return op.emitOpError() << "has mismatched number of operand rank ("
                            << input_rank << ") and limit_indices size ("
                            << limit_type.getNumElements() << ")";
  }

  if (input_rank != strides_type.getNumElements()) {
    return op.emitOpError()
           << "has mismatched number of operand rank (" << input_rank
           << ") and strides size (" << strides_type.getNumElements() << ")";
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
// Logical Ops
//===----------------------------------------------------------------------===//

OpFoldResult AndOp::fold(ArrayRef<Attribute> operands) {
  if (lhs() == rhs()) return lhs();

  auto rType = getType().cast<ShapedType>();
  auto lhsVal = operands[0].dyn_cast_or_null<DenseElementsAttr>();
  auto rhsVal = operands[1].dyn_cast_or_null<DenseElementsAttr>();

  if (lhsVal && lhsVal.isSplat()) {
    if (lhsVal.getSplatValue()
            .cast<IntegerAttr>()
            .getValue()
            .isAllOnesValue()) {
      return rhs();
    }

    if (lhsVal.getSplatValue().cast<IntegerAttr>().getValue().isNullValue()) {
      return lhsVal;
    }
  }

  if (rhsVal && rhsVal.isSplat()) {
    if (rhsVal.getSplatValue()
            .cast<IntegerAttr>()
            .getValue()
            .isAllOnesValue()) {
      return lhs();
    }

    if (rhsVal.getSplatValue().cast<IntegerAttr>().getValue().isNullValue()) {
      return rhsVal;
    }
  }

  if (!rhsVal || !lhsVal) return {};

  llvm::SmallVector<APInt, 4> values;
  values.reserve(rhsVal.getNumElements());
  for (auto it : llvm::zip(rhsVal.getIntValues(), lhsVal.getIntValues())) {
    values.push_back(std::get<0>(it) & std::get<1>(it));
  }

  return DenseIntElementsAttr::get(rType, values);
}

OpFoldResult OrOp::fold(ArrayRef<Attribute> operands) {
  if (lhs() == rhs()) return lhs();

  auto rType = getType().cast<ShapedType>();
  auto lhsVal = operands[0].dyn_cast_or_null<DenseElementsAttr>();
  auto rhsVal = operands[1].dyn_cast_or_null<DenseElementsAttr>();

  if (lhsVal && lhsVal.isSplat()) {
    if (lhsVal.getSplatValue()
            .cast<IntegerAttr>()
            .getValue()
            .isAllOnesValue()) {
      return lhsVal;
    }

    if (lhsVal.getSplatValue().cast<IntegerAttr>().getValue().isNullValue()) {
      return rhs();
    }
  }

  if (rhsVal && rhsVal.isSplat()) {
    if (rhsVal.getSplatValue()
            .cast<IntegerAttr>()
            .getValue()
            .isAllOnesValue()) {
      return rhsVal;
    }

    if (rhsVal.getSplatValue().cast<IntegerAttr>().getValue().isNullValue()) {
      return lhs();
    }
  }

  if (!rhsVal || !lhsVal) return {};

  llvm::SmallVector<APInt, 4> values;
  values.reserve(rhsVal.getNumElements());
  for (auto it : llvm::zip(rhsVal.getIntValues(), lhsVal.getIntValues())) {
    values.push_back(std::get<0>(it) | std::get<1>(it));
  }

  return DenseIntElementsAttr::get(rType, values);
}

OpFoldResult XorOp::fold(ArrayRef<Attribute> operands) {
  auto rType = getType().cast<ShapedType>();
  if (lhs() == rhs()) {
    Builder builder(getContext());
    return builder.getZeroAttr(rType);
  }

  auto lhsVal = operands[0].dyn_cast_or_null<DenseElementsAttr>();
  auto rhsVal = operands[1].dyn_cast_or_null<DenseElementsAttr>();

  if (lhsVal && lhsVal.isSplat()) {
    if (lhsVal.getSplatValue().cast<IntegerAttr>().getValue().isNullValue()) {
      return rhs();
    }
  }

  if (rhsVal && rhsVal.isSplat()) {
    if (rhsVal.getSplatValue().cast<IntegerAttr>().getValue().isNullValue()) {
      return lhs();
    }
  }

  if (!rhsVal || !lhsVal) return {};

  llvm::SmallVector<APInt, 4> values;
  values.reserve(rhsVal.getNumElements());
  for (auto it : llvm::zip(rhsVal.getIntValues(), lhsVal.getIntValues())) {
    values.push_back(std::get<0>(it) ^ std::get<1>(it));
  }

  return DenseIntElementsAttr::get(rType, values);
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
  for (int i = 0, e = dimensions.size(); i < e; ++i) {
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

OpFoldResult MapOp::fold(ArrayRef<Attribute> operands) {
  mlir::Block& bb = computation().front();
  mlir::Operation& front_op = bb.front();

  auto ret_op = mlir::dyn_cast<ReturnOp>(front_op);
  if (!ret_op) return nullptr;
  if (ret_op.results().size() != 1) return nullptr;

  for (mlir::BlockArgument barg : bb.getArguments()) {
    if (barg == ret_op.results()[0]) return getOperands()[barg.getArgNumber()];
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// RecvOp
//===----------------------------------------------------------------------===//

// Checks that the result type is of the form `tuple<any_type, mhlo::token>`
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
// ReduceWindowOp
//===----------------------------------------------------------------------===//

// For reduce-window, all `inputs` need to have compatible shapes.
static LogicalResult Verify(ReduceWindowOp op) {
  if (failed(verifyCompatibleShapes(op.inputs().getTypes())))
    return op.emitOpError() << "requires same shape for all inputs";
  return success();
}

// Get the operation used for reduction applied to `result_index`th result. Its
// expected to be a binary operation that consumes `result_index`th and
// `result_index + operands().size`th arguments of the body.
Operation* ReduceWindowOp::getReductionOp(int result_index) {
  auto return_op = cast<ReturnOp>(body().front().getTerminator());
  Operation* compute_op = return_op.results()[result_index].getDefiningOp();
  if (compute_op->getNumOperands() != 2) return nullptr;
  auto arg0 = compute_op->getOperand(0).dyn_cast<BlockArgument>();
  auto arg1 = compute_op->getOperand(1).dyn_cast<BlockArgument>();
  if (!arg0 || !arg1) return nullptr;
  int arg0_num = arg0.getArgNumber();
  int arg1_num = arg1.getArgNumber();
  int other_arg_index = result_index + inputs().size();
  if (arg0_num == result_index && arg1_num == other_arg_index)
    return compute_op;
  if (arg0_num == other_arg_index && arg1_num == result_index &&
      compute_op->hasTrait<OpTrait::IsCommutative>())
    return compute_op;
  return nullptr;
}

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
                     ValueRange inputs, ValueRange init_values,
                     DenseIntElementsAttr dimensions) {
  SmallVector<Type, 1> result_ty;
  result_ty.reserve(inputs.size());

  for (Value input : inputs) {
    result_ty.push_back(
        GetReduceResultType(input.getType(), dimensions, &builder));
  }
  build(builder, state, result_ty, inputs, init_values, dimensions);
}

LogicalResult ReduceOp::fold(ArrayRef<Attribute> operands,
                             SmallVectorImpl<OpFoldResult>& results) {
  // No dimensions to reduce.
  if (dimensions().getNumElements() == 0) {
    for (Value input : this->inputs()) {
      results.push_back(input);
    }
    return success();
  }

  // If all returned values in the ReduceOp region exists outside
  // the region replace the ReduceOp with those values.
  mlir::Block& bb = this->body().front();
  SmallVector<Value> replaced_results;
  if (auto ret_op = mlir::dyn_cast<ReturnOp>(bb.back())) {
    for (Value result : ret_op.results()) {
      if (result.getParentRegion() == ret_op->getParentRegion())
        return failure();
      replaced_results.push_back(result);
    }

    results.insert(results.end(), replaced_results.begin(),
                   replaced_results.end());
    return success();
  }

  return failure();
}

// Enable constant folding to occur within the region of the ReduceOp
// by replacing block argument uses with constants if:
//  1. All the ReduceOp operands are splat constants.
//  2. The ReduceOp region consists of a single logical AND or logical OR.
// The pattern leverages the idempotent property of the AND and OR operators
// to determine the value of a reduction on splat constants. Other boolean
// operators do not have this property, and need separate patterns to resolve
// reductions of their splat constants.
struct LowerBoolSplatConstantsIntoRegion : public OpRewritePattern<ReduceOp> {
  using OpRewritePattern<ReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReduceOp op,
                                PatternRewriter& rewriter) const override {
    mlir::Block& bb = op.body().front();

    // Ensure only a compute op and return op exist and the
    // compute op is an AND or OR op.
    if (bb.getOperations().size() != 2) return failure();
    if (!mlir::isa<mhlo::AndOp, mhlo::OrOp>(bb.front())) return failure();

    // Ensure all operands are splat constants.
    SmallVector<DenseElementsAttr, 4> barg_cst_attrs;
    for (auto inp_and_barg : llvm::zip(op.getOperands(), bb.getArguments())) {
      Value inp = std::get<0>(inp_and_barg);
      BlockArgument barg = std::get<1>(inp_and_barg);
      ConstOp cst = inp.getDefiningOp<ConstOp>();
      if (!cst) return failure();

      auto cst_attr = cst.value().dyn_cast_or_null<DenseElementsAttr>();
      if (!cst_attr.isSplat()) {
        return rewriter.notifyMatchFailure(op, "Must be splat constant.");
      }

      auto barg_shaped_type = barg.getType().dyn_cast<ShapedType>();
      if (!barg_shaped_type) return failure();

      auto barg_cst_attr =
          DenseElementsAttr::get(barg_shaped_type, cst_attr.getSplatValue());
      barg_cst_attrs.push_back(barg_cst_attr);
    }

    // Create new splat constants to replace block arguments.
    for (BlockArgument barg : bb.getArguments()) {
      int arg_idx = barg.getArgNumber();
      mhlo::ConstOp new_cst = rewriter.create<mhlo::ConstOp>(
          bb.front().getLoc(), barg.getType(), barg_cst_attrs[arg_idx]);
      barg.replaceAllUsesWith(new_cst);
    }
    return success();
  }
};

void ReduceOp::getCanonicalizationPatterns(OwningRewritePatternList& results,
                                           MLIRContext* context) {
  results.insert<LowerBoolSplatConstantsIntoRegion>(context);
}
//===----------------------------------------------------------------------===//
// RngNormalOp
//===----------------------------------------------------------------------===//

LogicalResult RngNormalOp::inferReturnTypeComponents(
    MLIRContext* context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  return rngInferReturnTypeComponents(context, location, operands, attributes,
                                      regions, inferredReturnShapes);
}

//===----------------------------------------------------------------------===//
// RngUniformOp
//===----------------------------------------------------------------------===//

LogicalResult RngUniformOp::inferReturnTypeComponents(
    MLIRContext* context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  return rngInferReturnTypeComponents(context, location, operands, attributes,
                                      regions, inferredReturnShapes);
}

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(SelectOp op) {
  // TODO(jpienaar): Update to allow broadcastable and unranked inputs. This
  // corresponds to the client side HLO.
  return success();
}

OpFoldResult SelectOp::fold(ArrayRef<Attribute> operands) {
  if (on_true() == on_false()) {
    return on_true();
  }

  auto predicate = operands[0].dyn_cast_or_null<DenseIntElementsAttr>();
  if (!predicate) {
    return {};
  }

  auto predicateTy = predicate.getType().cast<ShapedType>();
  if (!predicateTy.getElementType().isInteger(1)) {
    return {};
  }

  if (predicate.isSplat()) {
    return predicate.getSplatValue<APInt>().getBoolValue() ? on_true()
                                                           : on_false();
  }

  return {};
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

LogicalResult SelectOp::inferReturnTypeComponents(
    mlir::MLIRContext*, llvm::Optional<mlir::Location>, mlir::ValueRange,
    mlir::DictionaryAttr, mlir::RegionRange,
    llvm::SmallVectorImpl<mlir::ShapedTypeComponents>&) {
  // TODO(b/168772852)
  return failure();
}

LogicalResult SelectOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  return deriveShapeFromFirstOperand(&builder, getOperation(), operands,
                                     &reifiedReturnShapes);
}

//===----------------------------------------------------------------------===//
// SetDimensionSizeOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(SetDimensionSizeOp op) {
  if (auto size = op.size().getType().dyn_cast<RankedTensorType>()) {
    if (size.getRank() != 0)
      return op.emitOpError() << "size operand should be of rank-0";
  }

  return VerifyDimAttr(op);
}

OpFoldResult SetDimensionSizeOp::fold(ArrayRef<Attribute> operands) {
  DenseElementsAttr input = operands[0].dyn_cast_or_null<DenseElementsAttr>();
  if (input) return input;

  DenseElementsAttr size = operands[1].dyn_cast_or_null<DenseElementsAttr>();
  if (!size || !size.isSplat()) return {};

  auto ty = getType().dyn_cast<RankedTensorType>();
  if (!ty) return {};

  int64_t dim_size = ty.getDimSize(dimension());
  if (dim_size == size.getSplatValue().cast<IntegerAttr>().getInt())
    return operand();
  return {};
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

OpFoldResult PadOp::fold(ArrayRef<Attribute> operands) {
  // If all padding is zero then it is an identity pad.
  auto is_zero = [](const APInt& i) { return i == 0; };
  if (llvm::all_of(edge_padding_low().getIntValues(), is_zero) &&
      llvm::all_of(edge_padding_high().getIntValues(), is_zero) &&
      llvm::all_of(interior_padding().getIntValues(), is_zero))
    return operand();

  // If any padding is negative then it isn't supported by the folder (yet).
  auto is_negative = [](const APInt& i) { return i.slt(0); };
  if (llvm::all_of(edge_padding_low().getIntValues(), is_negative) &&
      llvm::all_of(edge_padding_high().getIntValues(), is_negative) &&
      llvm::all_of(interior_padding().getIntValues(), is_negative))
    return {};

  DenseElementsAttr input = operands[0].dyn_cast_or_null<DenseElementsAttr>();
  DenseElementsAttr padding = operands[1].dyn_cast_or_null<DenseElementsAttr>();
  RankedTensorType return_type = getType().dyn_cast_or_null<RankedTensorType>();
  if (!input || !input.getType().hasRank() || !padding || !return_type ||
      !return_type.hasStaticShape())
    return {};

  // Fill the full result tensor with the padding value.
  llvm::SmallVector<Attribute, 4> result(return_type.getNumElements(),
                                         padding.getValue({}));

  auto next_index = [](llvm::SmallVector<uint64_t, 8>& index,
                       llvm::ArrayRef<int64_t> shape) {
    for (int64_t i = index.size() - 1; i >= 0; --i) {
      ++index[i];
      if (index[i] < shape[i]) return;
      index[i] = 0;
    }
  };

  // Iterate over all elements of the input tensor and copy it to the correct
  // location in the output tensor.
  llvm::SmallVector<uint64_t, 8> index(input.getType().getRank(), 0);
  uint64_t num_elements = input.getNumElements();
  for (uint64_t operand_idx = 0; operand_idx < num_elements; operand_idx++) {
    uint64_t result_idx = 0;
    uint64_t idx_multiplyer = 1;
    for (int64_t i = index.size() - 1; i >= 0; --i) {
      result_idx +=
          (edge_padding_low().getValue<int64_t>({uint64_t(i)}) +
           index[i] *
               (interior_padding().getValue<int64_t>({uint64_t(i)}) + 1)) *
          idx_multiplyer;
      idx_multiplyer *= return_type.getDimSize(i);
    }
    result[result_idx] = input.getValue(index);
    next_index(index, input.getType().getShape());
  }
  return DenseElementsAttr::get(return_type, result);
}

//===----------------------------------------------------------------------===//
// DynamicPadOp
//===----------------------------------------------------------------------===//

void DynamicPadOp::getCanonicalizationPatterns(
    OwningRewritePatternList& results, MLIRContext* context) {
  results.insert<DPadToPad>(context);
}

static LogicalResult Verify(DynamicPadOp op) {
  auto input_type = op.operand().getType().dyn_cast<RankedTensorType>();
  // If operand is unranked, there is very little to verify statically.
  if (!input_type) return success();
  int input_rank = input_type.getRank();

  auto pad_type = op.padding_value().getType().cast<RankedTensorType>();
  if (pad_type.getRank() != 0) {
    return op.emitOpError() << "padding value type should be a rank-0";
  }

  auto padding_low_type =
      op.edge_padding_low().getType().cast<RankedTensorType>();
  if (padding_low_type.getNumElements() != input_rank) {
    return op.emitOpError()
           << "edge_padding_low length(" << padding_low_type.getNumElements()
           << ") must match operand rank(" << input_rank << ").";
  }

  auto padding_high_type =
      op.edge_padding_high().getType().cast<RankedTensorType>();
  if (padding_high_type.getNumElements() != input_rank) {
    return op.emitOpError()
           << "edge_padding_high length(" << padding_high_type.getNumElements()
           << ") must match operand rank(" << input_rank << ").";
  }

  auto interior_padding_type =
      op.interior_padding().getType().cast<RankedTensorType>();
  if (interior_padding_type.getNumElements() != input_rank) {
    return op.emitOpError()
           << "edge_padding_interior length("
           << interior_padding_type.getNumElements()
           << ") must match operand rank(" << input_rank << ").";
  }

  auto output_type = op.getResult().getType().dyn_cast<RankedTensorType>();
  // If result is unranked, there is very little to verify statically.
  if (!output_type) return success();
  int output_rank = output_type.getRank();
  if (input_rank != output_rank) {
    return op.emitOpError() << "operand rank(" << input_rank
                            << ") must match result(" << output_rank << ").";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(ReshapeOp op) {
  // If the operand type is dynamically shaped there is nothing to verify.
  auto operand_ty = op.operand().getType().dyn_cast<RankedTensorType>();
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

void ReshapeOp::getCanonicalizationPatterns(OwningRewritePatternList& results,
                                            MLIRContext* context) {
  results.insert<IdentityBroadcastReshape, IdentityBroadcastInDimReshape>(
      context);
}

//===----------------------------------------------------------------------===//
// ReplicaId Op
//===----------------------------------------------------------------------===//

LogicalResult ReplicaIdOp::inferReturnTypes(
    MLIRContext* context, Optional<Location>, ValueRange operands,
    DictionaryAttr, RegionRange, SmallVectorImpl<Type>& inferredReturnTypes) {
  inferredReturnTypes.push_back(RankedTensorType::get(
      /*shape=*/{}, IntegerType::get(context, 32, IntegerType::Unsigned)));
  return success();
}

//===----------------------------------------------------------------------===//
// If Op
//===----------------------------------------------------------------------===//

static LogicalResult VerifyConditionalBranch(Operation* op, Region& region,
                                             Value operand,
                                             llvm::Twine branchName,
                                             llvm::Twine operandName) {
  mlir::Block& entryBlock = region.front();
  if (entryBlock.getNumArguments() != 1)
    return op->emitOpError()
           << branchName << " block should have single argument, but found "
           << entryBlock.getNumArguments();

  Type operandType = operand.getType();
  Type branchArgType = entryBlock.getArgument(0).getType();
  if (branchArgType != operandType)
    return op->emitOpError()
           << operandName << " type (" << operandType << ") does not match "
           << branchName << " block arg type (" << branchArgType << ")";
  TypeRange branchReturnTypes = entryBlock.getTerminator()->getOperandTypes();
  if (branchReturnTypes != op->getResultTypes())
    return op->emitOpError()
           << branchName << " returned types (" << branchReturnTypes
           << ") do not match op result types (" << op->getResultTypes() << ")";

  return success();
}

static LogicalResult Verify(IfOp op) {
  if (failed(VerifyConditionalBranch(op, op.true_branch(), op.true_arg(),
                                     /*branchName=*/"true_branch",
                                     /*operandName=*/"true_arg"))) {
    return failure();
  }

  if (failed(VerifyConditionalBranch(op, op.false_branch(), op.false_arg(),
                                     /*branchName=*/"false_branch",
                                     /*operandName=*/"false_arg"))) {
    return failure();
  }
  return success();
}

static LogicalResult InlineIfConstantCondition(IfOp ifOp,
                                               PatternRewriter& rewriter) {
  DenseIntElementsAttr pred_attr;
  if (!matchPattern(ifOp.pred(), m_Constant(&pred_attr))) return failure();

  if (pred_attr.getSplatValue<BoolAttr>().getValue()) {
    ReplaceOpWithRegion(rewriter, ifOp, ifOp.true_branch(), ifOp.true_arg());
  } else {
    ReplaceOpWithRegion(rewriter, ifOp, ifOp.false_branch(), ifOp.false_arg());
  }
  return success();
}

void IfOp::getCanonicalizationPatterns(OwningRewritePatternList& results,
                                       MLIRContext* context) {
  results.add(&InlineIfConstantCondition);
}

//===----------------------------------------------------------------------===//
// Case Op
//===----------------------------------------------------------------------===//

static LogicalResult Verify(CaseOp op) {
  auto num_branches = op.branches().size();
  if (op.branch_operands().size() != num_branches)
    return op.emitOpError() << " number of branches (" << num_branches
                            << ") does not match number of branch operands ("
                            << op.branch_operands().size() << ")";

  for (unsigned i = 0; i < num_branches; ++i)
    if (failed(VerifyConditionalBranch(
            op, op.branches()[i], op.branch_operands()[i],
            /*branchName=*/"branch " + Twine(i),
            /*operandName=*/"branch_operand " + Twine(i))))
      return failure();

  return success();
}

static LogicalResult InlineCaseConstantCondition(CaseOp caseOp,
                                                 PatternRewriter& rewriter) {
  DenseIntElementsAttr index_attr;
  if (!matchPattern(caseOp.index(), m_Constant(&index_attr))) {
    return failure();
  }
  int64_t index =
      index_attr.getSplatValue<IntegerAttr>().getValue().getSExtValue();
  // For an OOB index, the last branch is executed as the default branch:
  // https://www.tensorflow.org/xla/operation_semantics#conditional
  if (index < 0 || index >= caseOp.getNumRegions())
    index = caseOp.getNumRegions() - 1;

  Region& region = caseOp.getRegion(index);
  if (!llvm::hasSingleElement(region)) return failure();
  ReplaceOpWithRegion(rewriter, caseOp, region,
                      caseOp.branch_operands()[index]);
  return success();
}

void CaseOp::getCanonicalizationPatterns(OwningRewritePatternList& results,
                                         MLIRContext* context) {
  results.add(&InlineCaseConstantCondition);
}

//===----------------------------------------------------------------------===//
// SqrtOp
//===----------------------------------------------------------------------===//

OpFoldResult SqrtOp::fold(ArrayRef<Attribute> operands) {
  auto val = operands[0].dyn_cast_or_null<DenseElementsAttr>();
  if (!val) return {};

  auto type = getElementTypeOrSelf(getType());
  if (!type.isF32() && !type.isF64()) return {};

  auto shaped_type = getType().cast<ShapedType>();
  if (!shaped_type.hasStaticShape()) return {};

  int bit_width = type.getIntOrFloatBitWidth();
  llvm::SmallVector<APFloat, 4> values;
  values.reserve(val.getNumElements());
  for (auto it : val.getFloatValues()) {
    double value = bit_width == 32 ? it.convertToFloat() : it.convertToDouble();
    if (value < 0) return {};
    value = std::sqrt(value);
    if (bit_width == 32)
      values.emplace_back(static_cast<float>(value));
    else
      values.emplace_back(value);
  }
  return DenseFPElementsAttr::get(shaped_type, values);
}

//===----------------------------------------------------------------------===//
// UnaryOps
//===----------------------------------------------------------------------===//

template <typename Op, typename ElementType = Type, typename ValType,
          typename Convert>
static Attribute UnaryFolder(Op* op, ArrayRef<Attribute> attrs) {
  if (!attrs[0]) return {};

  DenseElementsAttr val = attrs[0].dyn_cast<DenseElementsAttr>();
  if (!val) return {};

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
  values.reserve(val.getNumElements());
  for (const auto v : val.getValues<ValType>()) {
    values.push_back(Convert()(v));
  }

  return DenseElementsAttr::get(type, values);
}

struct round {
  APFloat operator()(const APFloat& f) {
    APFloat r = f;
    r.roundToIntegral(llvm::RoundingMode::NearestTiesToAway);
    return r;
  }
};

struct logical_not {
  APInt operator()(const APInt& i) {
    return APInt(i.getBitWidth(), static_cast<uint64_t>(!i));
  }
};

template <typename FloatOrInt>
struct sign {
  APFloat compute(const APFloat& f) {
    if (f.isZero() || f.isNaN()) return f;
    double value = f.isNegative() ? -1.0 : 1.0;
    APFloat val(value);
    bool unused;
    val.convert(f.getSemantics(), APFloat::rmNearestTiesToEven, &unused);
    return val;
  }

  APInt compute(const APInt& i) {
    APInt r = i;
    if (r == 0) return r;
    if (r.isNegative()) {
      return APInt(r.getBitWidth(), -1, /*isSigned=*/true);
    }
    return APInt(r.getBitWidth(), 1, /*isSigned=*/true);
  }

  FloatOrInt operator()(const FloatOrInt& fi) { return compute(fi); }
};

#define UNARY_FOLDER(Op, Func)                                                \
  OpFoldResult Op::fold(ArrayRef<Attribute> attrs) {                          \
    if (getElementTypeOrSelf(getType()).isa<FloatType>())                     \
      return UnaryFolder<Op, FloatType, APFloat, Func<APFloat>>(this, attrs); \
    if (getElementTypeOrSelf(getType()).isa<IntegerType>())                   \
      return UnaryFolder<Op, IntegerType, APInt, Func<APInt>>(this, attrs);   \
    return {};                                                                \
  }

#define UNARY_FOLDER_INT(Op, Func)                                   \
  OpFoldResult Op::fold(ArrayRef<Attribute> attrs) {                 \
    if (getElementTypeOrSelf(getType()).isa<IntegerType>())          \
      return UnaryFolder<Op, IntegerType, APInt, Func>(this, attrs); \
    return {};                                                       \
  }

#define UNARY_FOLDER_FLOAT(Op, Func)                                 \
  OpFoldResult Op::fold(ArrayRef<Attribute> attrs) {                 \
    if (getElementTypeOrSelf(getType()).isa<FloatType>())            \
      return UnaryFolder<Op, FloatType, APFloat, Func>(this, attrs); \
    return {};                                                       \
  }

UNARY_FOLDER(NegOp, std::negate);
UNARY_FOLDER(SignOp, sign);
UNARY_FOLDER_INT(NotOp, logical_not);
UNARY_FOLDER_FLOAT(RoundOp, round);

#undef UNARY_FOLDER
#undef UNARY_FOLDER_INT
#undef UNARY_FOLDER_FLOAT

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
struct remainder : std::modulus<T> {};

template <>
struct remainder<APInt> {
  APInt operator()(const APInt& a, const APInt& b) const { return a.srem(b); }
};

template <>
struct remainder<APFloat> {
  APFloat operator()(const APFloat& a, const APFloat& b) const {
    APFloat result(a);
    result.remainder(b);
    return result;
  }
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
BINARY_FOLDER(RemOp, remainder);
BINARY_FOLDER(MaxOp, max);
BINARY_FOLDER(MinOp, min);

#undef BINARY_FOLDER

//===----------------------------------------------------------------------===//
// SliceOp
//===----------------------------------------------------------------------===//

// Returns output dimension size for slice result for the given arguments.
// Returns -1 if arguments are illegal.
static int64_t InferSliceDim(int64_t input_dim, int64_t start, int64_t end,
                             int64_t stride) {
  if (input_dim == -1 || start < 0 || start > end || end > input_dim ||
      stride == 0)
    return -1;

  return llvm::divideCeil(end - start, stride);
}

LogicalResult SliceOp::inferReturnTypes(
    MLIRContext* context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  SliceOpAdaptor slice(operands, attributes);
  // TODO(jpienaar): Update this code after refactoring verify.
  if (failed(slice.verify(location.getValueOr(UnknownLoc::get(context))))) {
    return failure();
  }

  Type ty = slice.operand().getType();
  RankedTensorType ranked_ty = ty.dyn_cast<RankedTensorType>();
  if (!ranked_ty) {
    // The operand type is unranked, so the best we can infer for the result
    // type is an unranked tensor with the same element type as the operand
    // type.
    inferredReturnTypes.assign({ty});
    return success();
  }

  ShapedType attr_ty = slice.start_indices().getType();
  if (attr_ty.getRank() != 1) {
    return emitOptionalError(location, "start_indices has rank ",
                             attr_ty.getRank(), " instead of required rank 1");
  }

  int64_t rank = ranked_ty.getRank();
  if (attr_ty.getNumElements() != rank) {
    return emitOptionalError(
        location, "the number of elements in start_indices (",
        attr_ty.getNumElements(), ") does not match the rank of the operand (",
        rank, ")");
  }

  if (!attr_ty.getElementType().isSignlessInteger(64) ||
      slice.limit_indices().getType() != attr_ty ||
      slice.strides().getType() != attr_ty) {
    // Unfortunately we can't rely on the AllTypesMatch trait for the SliceOp
    // having been verified at this point. Emit an error message that matches
    // the one that would be reported by AllTypesMatch for a more consistent
    // user experience.
    // TODO(b/171567182): Clean this up after AllTypesMatch has been refactored.
    return emitOptionalError(location,
                             "failed to verify that all of {start_indices, "
                             "limit_indices, strides} have same type");
  }

  SmallVector<int64_t, 4> start(slice.start_indices().getValues<int64_t>());
  SmallVector<int64_t, 4> limit(slice.limit_indices().getValues<int64_t>());
  SmallVector<int64_t, 4> stride_vals(slice.strides().getValues<int64_t>());

  SmallVector<int64_t, 4> shape;
  shape.reserve(rank);
  for (int64_t i = 0, e = rank; i != e; i++) {
    shape.push_back(InferSliceDim(ranked_ty.getDimSize(i), start[i], limit[i],
                                  stride_vals[i]));
  }
  inferredReturnTypes.assign(
      {RankedTensorType::get(shape, ranked_ty.getElementType())});
  return success();
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
  if (count == 0) {
    return DenseElementsAttr::get<E>(
        op->getResult().getType().cast<ShapedType>(),
        /*list=*/{});
  }

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
  auto operand_type = getOperand().getType().cast<ShapedType>();
  auto result_type = getResult().getType().cast<ShapedType>();

  if (operand_type.hasStaticShape() && result_type.hasStaticShape() &&
      (operand_type.getShape() == result_type.getShape())) {
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

    auto dimension = concat.dimension();

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

//===----------------------------------------------------------------------===//
// SortOp
//===----------------------------------------------------------------------===//

void SortOp::build(OpBuilder& builder, OperationState& state,
                   ValueRange operands, int64_t dimension, bool is_stable) {
  state.addOperands(operands);
  state.addAttribute("dimension", builder.getI64IntegerAttr(dimension));
  state.addAttribute("is_stable", builder.getBoolAttr(dimension));

  for (Value operand : operands) state.addTypes(operand.getType());

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
    int64_t cmp_dim = op.dimension();
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
                      Value rhs, StringAttr comparison_direction,
                      StringAttr compare_type) {
  auto new_type =
      UpdateResultElementType(&builder, lhs.getType(), builder.getI1Type());
  build(builder, result, new_type, lhs, rhs, comparison_direction,
        compare_type);
}

LogicalResult CompareOp::inferReturnTypeComponents(
    mlir::MLIRContext*, llvm::Optional<mlir::Location>, mlir::ValueRange,
    mlir::DictionaryAttr, mlir::RegionRange,
    llvm::SmallVectorImpl<mlir::ShapedTypeComponents>&) {
  // TODO(b/168772852)
  return failure();
}

LogicalResult CompareOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  return deriveShapeFromFirstOperand(&builder, getOperation(), operands,
                                     &reifiedReturnShapes);
}

template <typename T>
struct less : std::less<T> {};

template <>
struct less<APInt> {
  bool operator()(const APInt& a, const APInt& b) const { return a.slt(b); }
};

template <typename T>
struct less_equal : std::less_equal<T> {};

template <>
struct less_equal<APInt> {
  bool operator()(const APInt& a, const APInt& b) const { return a.sle(b); }
};

template <typename T>
struct greater : std::greater<T> {};

template <>
struct greater<APInt> {
  bool operator()(const APInt& a, const APInt& b) const { return a.sgt(b); }
};

template <typename T>
struct greater_equal : std::greater_equal<T> {};

template <>
struct greater_equal<APInt> {
  bool operator()(const APInt& a, const APInt& b) const { return a.sge(b); }
};

template <typename Op, typename ElementType, typename SrcType, typename Convert>
static Attribute CompareFolder(CompareOp op, ArrayRef<Attribute> attrs) {
  if (!attrs[0] || !attrs[1]) return {};

  DenseElementsAttr lhs = attrs[0].dyn_cast<DenseElementsAttr>();
  DenseElementsAttr rhs = attrs[1].dyn_cast<DenseElementsAttr>();
  if (!lhs || !rhs) return {};

  ShapedType operand_type =
      op.getOperand(0).getType().template cast<ShapedType>();
  if (!operand_type.hasStaticShape()) {
    return {};
  }

  if (!operand_type.getElementType().isa<ElementType>()) {
    return {};
  }

  SmallVector<bool, 6> values;
  values.reserve(lhs.getNumElements());
  for (const auto zip :
       llvm::zip(lhs.getValues<SrcType>(), rhs.getValues<SrcType>())) {
    values.push_back(Convert()(std::get<0>(zip), std::get<1>(zip)));
  }

  auto result_ty = op.getType().cast<ShapedType>();
  return DenseElementsAttr::get(result_ty, values);
}

OpFoldResult CompareOp::fold(ArrayRef<Attribute> operands) {
  auto result_ty = getType().cast<ShapedType>();
  if (!result_ty.hasStaticShape()) return {};

  auto direction = comparison_direction();
  if (lhs() == rhs() && !getElementTypeOrSelf(lhs()).isa<FloatType>()) {
    if (direction == "LE" || direction == "EQ" || direction == "GE") {
      return DenseIntElementsAttr::get(result_ty, {true});
    }
    return DenseIntElementsAttr::get(result_ty, {false});
  }

  auto op_el_type = lhs().getType().cast<ShapedType>().getElementType();
  // Fold tensor<*xi1> != false to just return tensor<*xi1>
  if (direction == "NE" && op_el_type.isInteger(1)) {
    DenseIntElementsAttr cst_attr;
    if (matchPattern(lhs(), m_Constant(&cst_attr))) {
      if (cst_attr.isSplat() && !cst_attr.getSplatValue<bool>()) {
        return rhs();
      }
    }

    if (matchPattern(rhs(), m_Constant(&cst_attr))) {
      if (cst_attr.isSplat() && !cst_attr.getSplatValue<bool>()) {
        return lhs();
      }
    }
  }

  // Fold tensor<*xi1> == True to just return tensor<*xi1>
  if (direction == "EQ" && op_el_type.isInteger(1)) {
    DenseIntElementsAttr cst_attr;
    if (matchPattern(lhs(), m_Constant(&cst_attr))) {
      if (cst_attr.isSplat() && cst_attr.getSplatValue<bool>()) {
        return rhs();
      }
    }

    if (matchPattern(rhs(), m_Constant(&cst_attr))) {
      if (cst_attr.isSplat() && cst_attr.getSplatValue<bool>()) {
        return lhs();
      }
    }
  }

  if (!operands[0] || !operands[1]) {
    return {};
  }

#define COMPARE_FOLDER(Op, comparison, Func)                                \
  if (direction == comparison) {                                            \
    if (auto folded = CompareFolder<Op, FloatType, APFloat, Func<APFloat>>( \
            *this, operands))                                               \
      return folded;                                                        \
    if (auto folded = CompareFolder<Op, IntegerType, APInt, Func<APInt>>(   \
            *this, operands))                                               \
      return folded;                                                        \
  }

  COMPARE_FOLDER(CompareOp, "EQ", std::equal_to);
  COMPARE_FOLDER(CompareOp, "NE", std::not_equal_to);
  COMPARE_FOLDER(CompareOp, "LT", less);
  COMPARE_FOLDER(CompareOp, "LE", less_equal);
  COMPARE_FOLDER(CompareOp, "GT", greater);
  COMPARE_FOLDER(CompareOp, "GE", greater_equal);
#undef COMPARE_FOLDER

  return {};
}

//===----------------------------------------------------------------------===//
// ScatterOp
//===----------------------------------------------------------------------===//

llvm::SmallVector<Attribute, 4> evaluateMhloRegion(Region& region,
                                                   ArrayRef<Attribute> inputs) {
  if (region.getNumArguments() != inputs.size()) return {};

  llvm::DenseMap<Value, Attribute> values;
  values.reserve(region.getNumArguments());
  for (auto it : llvm::zip(region.getArguments(), inputs)) {
    values.try_emplace(std::get<0>(it), std::get<1>(it));
  }

  for (auto& op : region.getOps()) {
    llvm::SmallVector<Attribute, 4> inputs;
    for (auto& operand : op.getOpOperands()) {
      inputs.push_back(values.lookup(operand.get()));
    }
    if (isa<ReturnOp>(op)) return inputs;

    llvm::SmallVector<OpFoldResult, 4> results;
    if (failed(op.fold(inputs, results))) return {};
    for (auto it : llvm::zip(op.getResults(), results)) {
      if (!std::get<1>(it).is<Attribute>()) return {};
      values.insert({std::get<0>(it), std::get<1>(it).get<Attribute>()});
    }
  }
  return {};
}

OpFoldResult ScatterOp::fold(ArrayRef<Attribute> operands) {
  auto base = operands[0].dyn_cast_or_null<DenseElementsAttr>();
  auto index = operands[1].dyn_cast_or_null<DenseIntElementsAttr>();
  auto update = operands[2].dyn_cast_or_null<DenseElementsAttr>();
  if (!base || !index || !update) return {};

  auto base_type = base.getType().dyn_cast<RankedTensorType>();
  auto index_type = index.getType().dyn_cast<RankedTensorType>();
  auto update_type = update.getType().dyn_cast<RankedTensorType>();
  if (!base_type || !index_type || !update_type) return {};

  // Add the virtual trailing dimension of size 1 if index_vector_dim equals to
  // index_type.rank.
  const int64_t index_vector_dim =
      scatter_dimension_numbers().index_vector_dim().getInt();
  if (index_vector_dim == index_type.getRank()) {
    auto index_shape = index_type.getShape().vec();
    index_shape.push_back(1);
    index_type =
        RankedTensorType::get(index_shape, index_type.getElementType());
    index = index.reshape(index_type).cast<DenseIntElementsAttr>();
  }

  // Increment the multi-dimensional index vector based on the limits for each
  // dimension specified by shape and returns false if the index rolled around
  // with true otherwise.
  auto next_index = [](llvm::SmallVector<uint64_t, 8>& index,
                       llvm::ArrayRef<int64_t> shape) {
    for (int64_t i = index.size() - 1; i >= 0; --i) {
      ++index[i];
      if (index[i] < shape[i]) return true;
      index[i] = 0;
    }
    return false;
  };

  // Iterate over all elements of the update tensor, then find the corresponding
  // value in the indices tensor to determine which location we have to update
  // in the base/result tensor.
  llvm::SmallVector<Attribute, 8> results(base.getValues<Attribute>());
  llvm::SmallVector<uint64_t, 8> update_index(update_type.getRank(), 0);
  llvm::SmallVector<uint64_t, 8> index_index;
  index_index.reserve(index_type.getRank());
  llvm::SmallVector<uint64_t, 8> base_index;
  base_index.reserve(base_type.getRank());
  do {
    // Compute the index for the slice of the indices tensor for this update
    // value.
    index_index.clear();
    if (index_vector_dim == 0) index_index.push_back(0);
    for (int64_t i = 0; i < update_index.size(); ++i) {
      if (llvm::count(scatter_dimension_numbers().update_window_dims(), i) == 0)
        index_index.push_back(update_index[i]);
      if (index_index.size() == index_vector_dim) index_index.push_back(0);
    }

    // Compute the index for the given update value in the base tensor.
    base_index.assign(base_type.getRank(), 0);
    uint64_t index_count = index_type.getShape()[index_vector_dim];
    for (uint64_t i = 0; i < index_count; ++i) {
      uint64_t operand_dim = scatter_dimension_numbers()
                                 .scatter_dims_to_operand_dims()
                                 .getValue<APInt>({i})
                                 .getSExtValue();
      index_index[index_vector_dim] = i;
      base_index[operand_dim] +=
          index.getValue<APInt>(index_index).getSExtValue();
    }
    uint64_t update_window_dim_index = 0;
    for (uint64_t i = 0; i < base_index.size(); ++i) {
      if (llvm::count(scatter_dimension_numbers().inserted_window_dims(), i))
        continue;
      base_index[i] +=
          update_index[scatter_dimension_numbers()
                           .update_window_dims()
                           .getValue<APInt>({update_window_dim_index})
                           .getSExtValue()];
      update_window_dim_index++;
    }

    // Compute the linear index for the index into the base tensor.
    int64_t linear_base_index = 0;
    int64_t linear_base_index_multiplyer = 1;
    for (int64_t i = base_index.size() - 1; i >= 0; --i) {
      // Out of bound index have backend specific behaviour so avoid folding it.
      if (base_index[i] < 0 || base_index[i] >= base_type.getShape()[i])
        return {};
      linear_base_index += base_index[i] * linear_base_index_multiplyer;
      linear_base_index_multiplyer *= base_type.getShape()[i];
    }

    // Evaluate update computation and update the value with the newly computed
    // attribute in the base tensor.
    auto lhs = DenseElementsAttr::get(
        RankedTensorType::get({}, base_type.getElementType()),
        results[linear_base_index]);
    auto rhs = DenseElementsAttr::get(
        RankedTensorType::get({}, base_type.getElementType()),
        update.getValue<Attribute>(update_index));
    auto new_value = evaluateMhloRegion(update_computation(), {lhs, rhs});
    if (new_value.size() != 1 || !new_value[0]) return {};
    results[linear_base_index] =
        new_value[0].cast<DenseElementsAttr>().getValue<Attribute>({});
  } while (next_index(update_index, update_type.getShape()));

  return DenseElementsAttr::get(base_type, results);
}

using mlir::hlo::parseWindowAttributes;
using mlir::hlo::printWindowAttributes;

}  // namespace mhlo
}  // namespace mlir

#define GET_OP_CLASSES
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.cc.inc"

namespace mlir {
namespace mhlo {

//===----------------------------------------------------------------------===//
// mhlo Dialect Interfaces
//===----------------------------------------------------------------------===//

namespace {
struct HLOInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  // Allow all call operations to be inlined.
  bool isLegalToInline(Operation* call, Operation* callable,
                       bool wouldBeCloned) const final {
    return true;
  }
  // We don't have any special restrictions on what can be inlined into
  // destination regions (e.g. while/conditional bodies). Always allow it.
  bool isLegalToInline(Region* dest, Region* src, bool wouldBeCloned,
                       BlockAndValueMapping& valueMapping) const final {
    return true;
  }
  // Operations in mhlo dialect are always legal to inline since they are
  // pure.
  bool isLegalToInline(Operation*, Region*, bool,
                       BlockAndValueMapping&) const final {
    return true;
  }
};
}  // end anonymous namespace

//===----------------------------------------------------------------------===//
// mhlo Dialect Constructor
//===----------------------------------------------------------------------===//

MhloDialect::MhloDialect(MLIRContext* context)
    : Dialect(getDialectNamespace(), context, TypeID::get<MhloDialect>()) {
  addOperations<
#define GET_OP_LIST
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.cc.inc"
      >();
  addInterfaces<HLOInlinerInterface>();
  addTypes<TokenType>();
  context->loadDialect<tensor::TensorDialect>();
}

Type MhloDialect::parseType(DialectAsmParser& parser) const {
  StringRef data_type;
  if (parser.parseKeyword(&data_type)) return Type();

  if (data_type == "token") return TokenType::get(getContext());
  parser.emitError(parser.getNameLoc()) << "unknown mhlo type: " << data_type;
  return nullptr;
}

void MhloDialect::printType(Type type, DialectAsmPrinter& os) const {
  if (type.isa<TokenType>()) {
    os << "token";
    return;
  }
  os << "<unknown mhlo type>";
}

//===----------------------------------------------------------------------===//
// Shape inference
//===----------------------------------------------------------------------===//

LogicalResult deriveShapeFromFirstOperand(
    OpBuilder* builder, Operation* op, ValueRange operands,
    SmallVectorImpl<Value>* reifiedReturnShapes) {
  Value operand = operands.front();
  ShapedType operand_type = operand.getType().dyn_cast<ShapedType>();
  if (!operand_type) {
    op->emitOpError() << "first operand is not a shaped type";
    return failure();
  }
  auto loc = op->getLoc();
  // Some users rely on the result type being a static shape.
  auto shape_type =
      RankedTensorType::get(operand_type.getRank(), builder->getIndexType());
  reifiedReturnShapes->assign(
      {builder->create<shape::ShapeOfOp>(loc, shape_type, operand)});
  return success();
}

}  // namespace mhlo
}  // namespace mlir
