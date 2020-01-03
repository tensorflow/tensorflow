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

#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <string>
#include <tuple>
#include <type_traits>

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/StandardOps/Ops.h"  // TF:local_config_mlir
#include "mlir/Dialect/Traits.h"  // TF:local_config_mlir
#include "mlir/IR/Attributes.h"  // TF:local_config_mlir
#include "mlir/IR/Builders.h"  // TF:local_config_mlir
#include "mlir/IR/Diagnostics.h"  // TF:local_config_mlir
#include "mlir/IR/DialectImplementation.h"  // TF:local_config_mlir
#include "mlir/IR/Function.h"  // TF:local_config_mlir
#include "mlir/IR/Location.h"  // TF:local_config_mlir
#include "mlir/IR/MLIRContext.h"  // TF:local_config_mlir
#include "mlir/IR/Matchers.h"  // TF:local_config_mlir
#include "mlir/IR/OpDefinition.h"  // TF:local_config_mlir
#include "mlir/IR/OpImplementation.h"  // TF:local_config_mlir
#include "mlir/IR/PatternMatch.h"  // TF:local_config_mlir
#include "mlir/IR/StandardTypes.h"  // TF:local_config_mlir
#include "mlir/IR/TypeUtilities.h"  // TF:local_config_mlir
#include "mlir/IR/Types.h"  // TF:local_config_mlir
#include "mlir/IR/Value.h"  // TF:local_config_mlir
#include "mlir/Parser.h"  // TF:local_config_mlir
#include "mlir/Support/LLVM.h"  // TF:local_config_mlir
#include "mlir/Support/LogicalResult.h"  // TF:local_config_mlir
#include "mlir/Support/STLExtras.h"  // TF:local_config_mlir
#include "mlir/Transforms/InliningUtils.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/tensor_format.h"

namespace mlir {
namespace TF {

//===----------------------------------------------------------------------===//
// TF op helper functions
//===----------------------------------------------------------------------===//

// Returns the RankedTensorType for the given operand. TensorFlow constant ops
// may have non-static shape because the shape is not propagated during constant
// folding. If the defining op for the given operand is a constant op, this
// routine uses the constant op's attribute to get the actual shape.
static RankedTensorType GetRankedTensorTypeForOperand(Value operand) {
  DenseElementsAttr attr;
  if (matchPattern(operand, m_Constant(&attr))) {
    return attr.getType().dyn_cast<RankedTensorType>();
  }
  return operand->getType().dyn_cast<RankedTensorType>();
}

// Returns true if the given `value` is of ranked float tensor type with the
// given `rank`.
static inline bool isOfRankedFloatTensorType(Value value, int rank) {
  RankedTensorType type = GetRankedTensorTypeForOperand(value);
  return type && type.getRank() == rank &&
         type.getElementType().isa<FloatType>();
}

// Returns true if the given `value` has the specified rank or has unranked
// type.
static inline bool IsOfRankOrUnranked(Value value, int64_t rank) {
  RankedTensorType type = GetRankedTensorTypeForOperand(value);
  return !type || type.getRank() == rank;
}

// Returns true if the given `value` has at least the specified rank or has
// unranked type.
static inline bool HasRankAtLeast(Value value, int64_t rank) {
  RankedTensorType type = GetRankedTensorTypeForOperand(value);
  return !type || type.getRank() >= rank;
}

// Returns true if the given `value` has at most the specified rank or has
// unranked type.
static inline bool HasRankAtMost(Value value, int64_t rank) {
  RankedTensorType type = GetRankedTensorTypeForOperand(value);
  return !type || type.getRank() <= rank;
}

// Returns true if the given pair of TensorFlow types can be cast to one
// another. In other words, a single run-time value is legal for both the types.
// For example, tensor<*xf32> and tensor<3xf32> are cast compatible.
static bool AreCastCompatible(Type a, Type b) {
  if (TensorCastOp::areCastCompatible(a, b)) return true;

  // Resource types may optionally contain subtypes information that does not
  // match. Check subtypes compatibility when possible, otherwise treat them as
  // compatible.
  auto a_or_element_type = getElementTypeOrSelf(a);
  auto b_or_element_type = getElementTypeOrSelf(b);

  auto a_kind = a_or_element_type.getKind();
  auto b_kind = b_or_element_type.getKind();

  if (a_kind == TensorFlowTypes::RESOURCE &&
      b_kind == TensorFlowTypes::RESOURCE) {
    auto a_resource_type = a_or_element_type.dyn_cast<ResourceType>();
    auto b_resource_type = b_or_element_type.dyn_cast<ResourceType>();
    bool a_has_subtype = !a_resource_type.getSubtypes().empty();
    bool b_has_subtype = !b_resource_type.getSubtypes().empty();

    if (!a_has_subtype || !b_has_subtype) return true;

    assert(a_resource_type.getSubtypes().size() <= 1 &&
           "Resource type must have at most one subtype");
    assert(b_resource_type.getSubtypes().size() <= 1 &&
           "Resource type must have at most one subtype");

    return TensorCastOp::areCastCompatible(
        a_resource_type.getSubtypes().front(),
        b_resource_type.getSubtypes().front());
  }

  // Variant types may optionally contain subtypes information that need not
  // match.  It is also not possible to compare subtypes for compatibility as
  // their interpretation depends on the ops operating on them. So, accept all
  // pairs of variant types.
  return a_kind == TensorFlowTypes::VARIANT &&
         b_kind == TensorFlowTypes::VARIANT;
}

static bool IsUnknownDimOrRank(int64_t dim_or_rank) {
  return dim_or_rank == -1;
}

// Returns the tf.Equal/tf.NotEqual result type given `x` and `y` and inputs. If
// `incompatible_shape_error` is true, reports error if `x` and `y` has
// incompatible shapes. Otherwise, returns a tensor type with unknown rank.
static Type DeduceEqualCmpOpType(Builder *builder, Location loc, Value x,
                                 Value y, BoolAttr incompatible_shape_error) {
  auto result_type =
      OpTrait::util::getBroadcastedType(x->getType(), y->getType());
  if (!result_type) {
    if (incompatible_shape_error.getValue()) {
      mlir::emitError(loc, "non-broadcastable operands");
    } else {
      return UnrankedTensorType::get(builder->getI1Type());
    }
  }

  auto ranked_type = result_type.dyn_cast<RankedTensorType>();
  if (!ranked_type) return UnrankedTensorType::get(builder->getI1Type());

  return RankedTensorType::get(ranked_type.getShape(), builder->getI1Type());
}

// Returns dimension index for the given TensorFlow axis that supports negative
// indexing.
static int64_t GetDimForAxis(int64_t axis, int64_t rank) {
  return axis >= 0 ? axis : axis + rank;
}

// Infers output type for reduction ops such as SumOp, MaxOp etc.
// TODO(b/e667204a): Move this logic to shape inference once it supports custom
// inference functions.
static Type InferReductionOpType(Value input, Value reduction_indices,
                                 BoolAttr keep_dims, Builder *builder) {
  Type input_ty = input->getType();
  Type element_ty = getElementTypeOrSelf(input_ty);

  // Output type is unranked if input type is not ranked.
  auto ranked_ty = input_ty.dyn_cast<RankedTensorType>();
  if (!ranked_ty) return UnrankedTensorType::get(element_ty);
  int64_t rank = ranked_ty.getRank();

  DenseIntElementsAttr indices;
  if (!matchPattern(reduction_indices, m_Constant(&indices))) {
    // Output type is unranked if reduction indices are not constant and reduced
    // dimensions are not kept.
    if (!keep_dims.getValue()) return UnrankedTensorType::get(element_ty);

    // Otherwise, output type has same rank as the input.
    return RankedTensorType::get(SmallVector<int64_t, 4>(rank, -1), element_ty);
  }

  int64_t num_reduce_dim = 0;
  llvm::SmallVector<bool, 4> is_reduce_dim(rank, false);
  for (APInt index : indices.getValues<APInt>()) {
    int64_t dim = GetDimForAxis(index.getSExtValue(), rank);
    // Invalid input.
    if (dim < 0 || dim >= rank) return UnrankedTensorType::get(element_ty);

    if (!is_reduce_dim[dim]) {
      is_reduce_dim[dim] = true;
      num_reduce_dim++;
    }
  }

  ArrayRef<int64_t> shape = ranked_ty.getShape();
  SmallVector<int64_t, 4> out_shape;
  out_shape.reserve(rank - (keep_dims.getValue() ? 0 : num_reduce_dim));
  for (int64_t i = 0; i < rank; ++i) {
    if (!is_reduce_dim[i])
      out_shape.push_back(shape[i]);
    else if (keep_dims.getValue())
      out_shape.push_back(1);
  }
  return RankedTensorType::get(out_shape, element_ty);
}

// Verifies that the given types are cast compatible. If not, emits appropriate
// error for the given op. If mask_one_dim is set to true, then the types are
// allowed to have one mismatching dimension. Masking one of the dimensions is
// useful for ops like Concat that requires all ranked inputs to have the same
// rank and match dimension sizes for all but one of the dimensions.
static LogicalResult VerifyTypesCompatibility(
    Operation::operand_type_range types, bool mask_one_dim, Operation *op) {
  constexpr int64_t kUninitialized = -1;
  int64_t common_rank = kUninitialized;
  llvm::SmallVector<int64_t, 4> common_dims;
  int64_t dim_to_mask = kUninitialized;

  // Initialize common_rank with rank of the first ranked type and verify that
  // following ranked types have the same rank.
  // Similarly, initialize each of the dimensions with the first type that has
  // the dimension size available and verify that all following types have the
  // same size for the dimension. However, if mask_one_dim is true, note down
  // the dimension index on the first mismatch and ignore dimension at that
  // index in following types.
  for (Type ty : types) {
    RankedTensorType ranked_ty = ty.dyn_cast<RankedTensorType>();
    if (!ranked_ty) continue;

    int64_t rank = ranked_ty.getRank();
    if (common_rank == kUninitialized) {
      common_rank = rank;
      common_dims.resize(common_rank, kUninitialized);
    } else if (common_rank != rank) {
      return op->emitError()
             << "operand type " << ranked_ty
             << " is not compatible with preceding operands; expected rank: "
             << common_rank;
    }

    for (int64_t i = 0, e = common_rank; i != e; i++) {
      if (i == dim_to_mask) continue;

      int64_t dim = ranked_ty.getDimSize(i);
      if (dim == kUninitialized) continue;

      int64_t &common_dim = common_dims[i];
      if (common_dim == kUninitialized) {
        common_dim = dim;
      } else if (common_dim != dim) {
        // If mask_one_dim is true, do not emit an error if this is the only
        // dimension with mismatches. Note down the dimension to mask it from
        // the following types.
        if (mask_one_dim && dim_to_mask == kUninitialized) {
          dim_to_mask = i;
          continue;
        }

        return op->emitError() << "operand type " << ranked_ty
                               << " is not compatible with preceding operands; "
                                  "expected dimension at index "
                               << i << ": " << common_dim;
      }
    }
  }
  return success();
}

namespace {
#include "tensorflow/compiler/mlir/tensorflow/transforms/generated_canonicalize.inc"
}  // namespace

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

void AddOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                        MLIRContext *context) {
  results.insert<AddToAddV2>(context);
}

//===----------------------------------------------------------------------===//
// AddNOp
//===----------------------------------------------------------------------===//

OpFoldResult AddNOp::fold(ArrayRef<Attribute> operands) {
  if (operands.size() == 1) return *inputs().begin();
  return {};
}

//===----------------------------------------------------------------------===//
// AddV2Op
//===----------------------------------------------------------------------===//

void AddV2Op::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                          MLIRContext *context) {
  results.insert<AddV2OfNegLeft, AddV2OfNegRight>(context);
}

//===----------------------------------------------------------------------===//
// AllOp
//===----------------------------------------------------------------------===//

// Verifies an reduction op's `input` and reduction `dims`.
static LogicalResult VerifyReductionInputAndDims(Value input, Value dims,
                                                 Location loc) {
  auto dims_type = dims->getType().dyn_cast<RankedTensorType>();
  if (!dims_type) return success();
  if (dims_type.getRank() > 1)
    return emitError(loc, "dimensions can only be 0D or 1D tensor");

  auto input_type = input->getType().dyn_cast<RankedTensorType>();
  if (!input_type) return success();
  int64_t rank = input_type.getRank();

  DenseIntElementsAttr dims_attr;
  if (!matchPattern(dims, m_Constant(&dims_attr))) return success();
  for (const auto &dim_pair : llvm::enumerate(dims_attr)) {
    int64_t cur_dim = dim_pair.value().getSExtValue();
    if (cur_dim < -rank || cur_dim >= rank)
      return emitError(loc)
             << dim_pair.index() << "-th dimension should be in the range of [-"
             << rank << ", " << rank << ")";
  }

  return success();
}

static LogicalResult Verify(AllOp op) {
  return VerifyReductionInputAndDims(op.input(), op.reduction_indices(),
                                     op.getLoc());
}

//===----------------------------------------------------------------------===//
// AnyOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(AnyOp op) {
  return VerifyReductionInputAndDims(op.input(), op.reduction_indices(),
                                     op.getLoc());
}

//===----------------------------------------------------------------------===//
// AssertOp
//===----------------------------------------------------------------------===//

namespace {

// Removes Assert with constant true predicate.
struct AssertWithTrue : public OpRewritePattern<AssertOp> {
  using OpRewritePattern<AssertOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(AssertOp op,
                                     PatternRewriter &rewriter) const override {
    ElementsAttr cst;
    if (matchPattern(op.condition(), m_Constant(&cst))) {
      if (cst.getValue<BoolAttr>({}).getValue()) {
        rewriter.eraseOp(op);
        return matchSuccess();
      }
    }
    return matchFailure();
  }
};
}  // namespace

void AssertOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                           MLIRContext *context) {
  results.insert<AssertWithTrue>(context);
}

//===----------------------------------------------------------------------===//
// BatchMatMulOp
//===----------------------------------------------------------------------===//

void BatchMatMulOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<BatchMatMulToMatMul>(context);
}

//===----------------------------------------------------------------------===//
// BatchMatMulV2Op
//===----------------------------------------------------------------------===//

void BatchMatMulV2Op::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<BatchMatMulV2ToMatMul>(context);
}

//===----------------------------------------------------------------------===//
// BiasAddOp
//===----------------------------------------------------------------------===//

// Verifies that,
// * the value and bias operands have valid ranks or are unranked.
// * Channel dimension of the value operand and length of bias matches if they
//   are not unknown.
//
static LogicalResult Verify(BiasAddOp op) {
  StringRef format = op.data_format();
  if (format == "NHWC") {
    if (!HasRankAtLeast(op.value(), 2))
      return op.emitOpError(
          "requires value operand to have rank at least two with `NHWC` data "
          "format");
  } else {
    // Op definition requires data_format to be either NHWC or NCHW.
    DCHECK_EQ(format.str(), "NCHW");
    if (!HasRankAtLeast(op.value(), 3))
      return op.emitOpError(
          "requires value operand to have rank at least three with `NCHW` data "
          "format");
  }

  if (!IsOfRankOrUnranked(op.bias(), 1))
    return op.emitOpError("requires bias operand to have rank exactly one");

  RankedTensorType value_ty =
      op.value()->getType().dyn_cast<RankedTensorType>();
  RankedTensorType bias_ty = op.bias()->getType().dyn_cast<RankedTensorType>();
  if (!bias_ty || !value_ty) return success();

  // TODO(hinsu): Leverage tensor_format.h utility in TensorFlow to compute
  // dimension indices based on format.
  int64_t feature_dim_idx = format == "NHWC" ? value_ty.getRank() - 1 : 1;
  int64_t feature_dim = value_ty.getDimSize(feature_dim_idx);
  int64_t bias_len = bias_ty.getDimSize(0);
  if (feature_dim != -1 && bias_len != -1 && feature_dim != bias_len) {
    return op.emitOpError()
           << "requires channel dimension and feature dimension to match; "
              "found "
           << feature_dim << " and " << bias_len << ", respectively";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// BiasAddGradOp
//===----------------------------------------------------------------------===//

// Verifies that,
// * the out_backprop operands have valid ranks or are unranked.
//
static LogicalResult Verify(BiasAddGradOp op) {
  StringRef format = op.data_format();
  if (format == "NHWC") {
    if (!HasRankAtLeast(op.out_backprop(), 2))
      return op.emitOpError(
          "requires out_backprop operand to have rank at least two with `NHWC` "
          "data format");
  } else {
    // Op definition requires data_format to be either NHWC or NCHW.
    DCHECK_EQ(format.str(), "NCHW");
    if (!HasRankAtLeast(op.out_backprop(), 3))
      return op.emitOpError(
          "requires out_backprop operand to have rank at least three with "
          "`NCHW` data format");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// BitcastOp
//===----------------------------------------------------------------------===//

void BitcastOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                            MLIRContext *context) {
  results.insert<BitcastSameType, BitcastNested>(context);
}

//===----------------------------------------------------------------------===//
// BroadcastToOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(BroadcastToOp op) {
  // TODO(antiagainst): check that
  // * The 'shape' input is an 1-D int tensor.
  // * Each dimension pair of the source and target shapes are either equal
  //   or one of them is one.
  return success();
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

void CastOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                         MLIRContext *context) {
  results.insert<CastSameType>(context);
}

//===----------------------------------------------------------------------===//
// ConcatOp and ConcatV2Op
//===----------------------------------------------------------------------===//

template <typename OpT,
          typename std::enable_if<llvm::is_one_of<
              OpT, ConcatOp, ConcatV2Op>::value>::type * = nullptr>
static LogicalResult Verify(OpT op) {
  // TODO(hinsu): Convert variadic length attributes to derived attributes.
  Operation::operand_range values = op.values();

  int axis_idx = std::is_same<OpT, ConcatOp>() ? 0 : 1;
  Value axis = *op.getODSOperands(axis_idx).begin();
  if (!HasRankAtMost(axis, 1)) {
    return op.emitOpError(
        "requires axis to be of scalar type (or vector type for older "
        "versions)");
  }

  return VerifyTypesCompatibility(values,
                                  /*mask_one_dim=*/true, op.getOperation());
}

//===----------------------------------------------------------------------===//
// ConcatOffsetOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(ConcatOffsetOp op) {
  if (op.N() < 2)
    return op.emitOpError() << "requires N to be at least 2, got " << op.N();

  if (op.shape().size() != op.offset().size())
    return op.emitOpError()
           << "requires sizes of shapes and offsets to be the same, got sizes "
           << op.shape().size() << " and " << op.offset().size();

  auto ranked_dim = op.concat_dim()->getType().dyn_cast<RankedTensorType>();
  if (ranked_dim && ranked_dim.getRank() != 0)
    return op.emitOpError()
           << "requires concat_dim to be a scalar, got tensor of rank "
           << ranked_dim.getRank();

  int64_t num_dims = -1;
  for (auto shape_offset_idx :
       llvm::enumerate(llvm::zip(op.shape(), op.offset()))) {
    Value shape = std::get<0>(shape_offset_idx.value());
    Value offset = std::get<1>(shape_offset_idx.value());
    const size_t idx = shape_offset_idx.index();

    if (failed(verifyCompatibleShape(shape->getType(), offset->getType())))
      return op.emitOpError() << "requires operand and result " << idx
                              << " to have compatible shapes";

    auto ranked_shape = shape->getType().dyn_cast<RankedTensorType>();
    if (!ranked_shape) continue;

    if (ranked_shape.getRank() != 1)
      return op.emitOpError() << "requires shape tensor operand " << idx
                              << " to be of rank 1, got tensor of rank "
                              << ranked_shape.getRank();

    if (!ranked_shape.hasStaticShape()) continue;

    int64_t ranked_shape_dim = ranked_shape.getDimSize(0);
    if (num_dims == -1)
      num_dims = ranked_shape_dim;
    else if (ranked_shape_dim != num_dims)
      return op.emitOpError()
             << "requires shape tensor (rank 1) operand " << idx
             << " to be of length " << num_dims
             << ", got tensor (rank 1) of length " << ranked_shape_dim;
  }

  return success();
}

LogicalResult ConcatOffsetOp::fold(ArrayRef<Attribute> operands,
                                   SmallVectorImpl<OpFoldResult> &results) {
  // ConcatOffset must have its first operand be concat_dim and at least two
  // shape tensors in variadic shapes operand.
  if (operands.size() < 3) return failure();

  // Check concat_dim is a scalar.
  auto concat_dim_attr = operands[0].dyn_cast_or_null<DenseIntElementsAttr>();
  if (!concat_dim_attr || concat_dim_attr.getType().getRank() != 0)
    return failure();

  llvm::SmallVector<DenseIntElementsAttr, 4> shapes;
  shapes.reserve(operands.size() - 1);
  for (Attribute shape : llvm::drop_begin(operands, 1))
    if (auto shape_attr = shape.dyn_cast_or_null<DenseIntElementsAttr>())
      shapes.push_back(shape_attr);
    else
      return failure();

  // Check all shapes are vectors of the same length.
  if (shapes.front().getType().getRank() != 1) return success();
  const int64_t num_dims = shapes.front().getNumElements();
  for (DenseIntElementsAttr shape : llvm::drop_begin(shapes, 1))
    if (shape.getType().getRank() != 1 || shape.getNumElements() != num_dims)
      return failure();

  // Check concat_dim is within [-num_dims, num_dims).
  int32_t concat_dim = (*concat_dim_attr.getValues<int32_t>().begin());
  if (concat_dim < 0) concat_dim += num_dims;
  if (concat_dim >= num_dims || concat_dim < 0) return failure();

  // Check all elements besides at concat_dim match across all shape tensors.
  SmallVector<int32_t, 4> shape0;
  shape0.reserve(num_dims);
  for (int32_t dim : shapes.front().getValues<int32_t>()) shape0.push_back(dim);

  for (DenseIntElementsAttr shape : llvm::drop_begin(shapes, 1)) {
    for (auto dims_and_idx : llvm::enumerate(llvm::zip(shape0, shape))) {
      if (dims_and_idx.index() == concat_dim) continue;

      if (std::get<0>(dims_and_idx.value()) !=
          std::get<1>(dims_and_idx.value()).getSExtValue())
        return failure();
    }
  }

  // Compute an exclusive cumulative sum of elements at concat_dim.
  results.reserve(shapes.size());
  SmallVector<int32_t, 4> cumulative_sum(num_dims, 0);
  RankedTensorType offset_type =
      RankedTensorType::get({num_dims}, IntegerType::get(32, getContext()));
  for (DenseIntElementsAttr shape : shapes) {
    results.push_back(DenseIntElementsAttr::get(offset_type, cumulative_sum));
    cumulative_sum[concat_dim] += shape.getValue<int32_t>(concat_dim);
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ConjOp
//===----------------------------------------------------------------------===//

void ConjOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                         MLIRContext *context) {
  results.insert<ConjNested>(context);
}

//===----------------------------------------------------------------------===//
// ConstOp
//===----------------------------------------------------------------------===//

OpFoldResult ConstOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.empty() && "constant has no operands");

  // Return the held attribute value.
  return value();
}

// Builds a constant op with the specified attribute `value`. The result
// op's type is deduced from `value`; if `value` is of scalar type,
// wraps it up with a tensor type of empty shape.
void ConstOp::build(Builder *builder, OperationState &result, Attribute value) {
  ShapedType type;
  if (auto elemAttr = value.dyn_cast<ElementsAttr>()) {
    type = elemAttr.getType();
  } else if (value.isa<BoolAttr>() || value.isa<FloatAttr>() ||
             value.isa<IntegerAttr>()) {
    // All TensorFlow types must be tensor types. In the build() method,
    // we want to provide more flexibility by allowing attributes of scalar
    // types. But we need to wrap it up with ElementsAttr to construct
    // valid TensorFlow constants.
    type = RankedTensorType::get(/*shape=*/{}, value.getType());
    value = DenseElementsAttr::get(type, value);
  }
  // TODO: support other TensorFlow specific types.
  assert(type && "unsupported attribute type for building tf.Const");
  result.types.push_back(type);
  result.addAttribute("value", value);
}

void ConstOp::build(Builder *builder, OperationState &result, Type type,
                    Attribute value) {
  // Handle the case where the type and value are already tensors.
  if (type.isa<TensorType>() && value.isa<ElementsAttr>()) {
    result.addTypes(type);
    result.addAttribute("value", value);
    return;
  }

  // Otherwise, default to the attribute builder.
  ConstOp::build(builder, result, value);
  assert(type == result.types[0] && "type mismatch in construction");
}

//===----------------------------------------------------------------------===//
// Conv2DOp and Conv3DOp
//===----------------------------------------------------------------------===//

template <typename OpT>
static LogicalResult VerifyConvOpAttributes(OpT op, int num_dims) {
  if (!IsOfRankOrUnranked(op.getResult(), num_dims))
    return op.emitOpError()
           << "requires result to be " << num_dims << "D tensor";

  auto is_not_positive = [](Attribute val) {
    return val.cast<IntegerAttr>().getValue().getSExtValue() <= 0;
  };

  int64_t strides_size = op.strides().size();
  if (strides_size != num_dims)
    return op.emitOpError() << "requires strides attribute length to be "
                            << num_dims << "; actual length " << strides_size;
  if (llvm::any_of(op.strides().getValue(), is_not_positive))
    return op.emitOpError("requires positive strides");

  int64_t dilations_size = op.strides().size();
  if (op.dilations().size() != num_dims)
    return op.emitOpError() << "requires dilations attribute length to be "
                            << num_dims << "; actual length " << dilations_size;
  if (llvm::any_of(op.dilations().getValue(), is_not_positive))
    return op.emitOpError("requires positive dilations");

  return success();
}

// Verifies that,
// * Ranks of operands and result are valid
// * Number of input channels is divisible by the number of filter input
//   channels
// * Length of explicit_paddings attribute is valid and has non negative
//   elements
// * strides and dilations attributes have positive elements
template <typename OpT, typename std::enable_if<llvm::is_one_of<
                            OpT, Conv2DOp, Conv3DOp>::value>::type * = nullptr>
static LogicalResult Verify(OpT op) {
  int num_spatial_dims = std::is_same<OpT, Conv2DOp>() ? 2 : 3;
  int num_dims = 2 + num_spatial_dims;

  if (!IsOfRankOrUnranked(op.input(), num_dims) ||
      !IsOfRankOrUnranked(op.filter(), num_dims))
    return op.emitOpError()
           << "requires operands to be " << num_dims << "D tensor";

  // EXPLICIT padding mode and the associated attribute is limited to Conv2D.
  // So, fetch attribute by string instead of the op.explicit_paddings()
  // attribute getter.
  if (op.padding() == "EXPLICIT") {
    auto paddings = op.template getAttrOfType<ArrayAttr>("explicit_paddings");
    if (!paddings)
      return op.emitOpError() << "requires attribute 'explicit_paddings' with "
                                 "'EXPLICIT' padding mode";

    int64_t paddings_size = paddings.size();
    int64_t expected_size = 2 * num_dims;

    if (paddings_size != expected_size)
      return op.emitOpError()
             << "requires explicit_paddings attribute length to be "
             << expected_size << "; actual length " << paddings_size;

    auto is_negative = [](Attribute val) {
      return val.cast<IntegerAttr>().getValue().getSExtValue() < 0;
    };
    if (llvm::any_of(paddings.getValue(), is_negative))
      return op.emitOpError("requires non negative explicit paddings");
  }

  LogicalResult verify_result = VerifyConvOpAttributes(op, num_dims);
  if (failed(verify_result)) {
    return verify_result;
  }

  int64_t input_channels = -1;
  if (auto ty = op.input()->getType().template dyn_cast<RankedTensorType>()) {
    std::string data_format = op.data_format().str();
    tensorflow::TensorFormat format;
    auto is_valid = FormatFromString(data_format, &format);
    DCHECK(is_valid) << data_format;
    int idx = tensorflow::GetTensorFeatureDimIndex(num_dims, format);
    input_channels = ty.getDimSize(idx);
  }

  int64_t filter_channels = -1;
  if (auto ty = op.filter()->getType().template dyn_cast<RankedTensorType>()) {
    int idx = tensorflow::GetFilterTensorInputChannelsDimIndex(
        num_dims, tensorflow::FORMAT_HWIO);
    filter_channels = ty.getDimSize(idx);
  }

  if (input_channels != -1 && filter_channels != -1 &&
      input_channels % filter_channels != 0)
    return op.emitOpError()
           << "requires the number of input channels to be divisible by the "
              "number of filter input channels; found "
           << input_channels << " and " << filter_channels << ", respectively";

  return success();
}

//===----------------------------------------------------------------------===//
// Conv2dBackpropInputOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(Conv2DBackpropInputOp op) {
  int num_spatial_dims = 2;
  int num_dims = 2 + num_spatial_dims;

  if (!IsOfRankOrUnranked(op.out_backprop(), num_dims) ||
      !IsOfRankOrUnranked(op.filter(), num_dims))
    return op.emitOpError()
           << "requires operands to be " << num_dims << "D tensor";

  LogicalResult verify_result = VerifyConvOpAttributes(op, num_dims);
  if (failed(verify_result)) {
    return verify_result;
  }

  return success();
}  // namespace TF

//===----------------------------------------------------------------------===//
// DivOp
//===----------------------------------------------------------------------===//

void DivOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                        MLIRContext *context) {
  results.insert<DivWithSqrtDivisor>(context);
}

//===----------------------------------------------------------------------===//
// DynamicStitchOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(DynamicStitchOp op) {
  if (op.N() < 1) return op.emitOpError("requires attribute N with value >= 1");

  if (RankedTensorType out_ty = op.getType().dyn_cast<RankedTensorType>()) {
    if (out_ty.getRank() == 0) {
      return op.emitOpError("requires non scalar output");
    }
  }

  llvm::SmallDenseSet<int64_t, 8> index_values;
  bool all_indices_const = true;
  int32_t max_index = -1;
  llvm::Optional<SmallVector<int64_t, 4>> inferred_item_shape;
  for (auto it : llvm::zip(op.indices(), op.data())) {
    Value index = std::get<0>(it);

    DenseIntElementsAttr index_attr;
    if (matchPattern(index, m_Constant(&index_attr))) {
      for (int32_t index : index_attr.getValues<int32_t>()) {
        if (index < 0)
          return op.emitOpError()
                 << "requires non-negative index values; found " << index;
        max_index = std::max(index, max_index);
        index_values.insert(index);
      }
    } else {
      all_indices_const = false;
    }

    Value data = std::get<1>(it);
    RankedTensorType index_ty = index->getType().dyn_cast<RankedTensorType>();
    RankedTensorType data_ty = data->getType().dyn_cast<RankedTensorType>();
    if (!index_ty || !data_ty) continue;

    int64_t index_rank = index_ty.getRank();
    ArrayRef<int64_t> data_shape = data_ty.getShape();
    ArrayRef<int64_t> index_shape = index_ty.getShape();
    if (failed(mlir::verifyCompatibleShape(index_shape,
                                           data_shape.take_front(index_rank))))
      return op.emitOpError() << "requires shape of data with type " << data_ty
                              << " to have prefix matching with shape of the "
                                 "corresponding index type "
                              << index_ty;

    ArrayRef<int64_t> item_shape = data_shape.drop_front(index_rank);
    if (!inferred_item_shape) {
      inferred_item_shape = llvm::to_vector<4>(item_shape);
      continue;
    }

    if (failed(mlir::verifyCompatibleShape(item_shape, *inferred_item_shape)))
      return op.emitOpError() << "has inconsistent shaped data and index "
                                 "pairs; inferred item shapes ["
                              << llvm::makeArrayRef(*inferred_item_shape)
                              << "] and [" << item_shape << "] don't match";
    for (int i = 0, e = item_shape.size(); i < e; ++i) {
      int64_t &inferred_dim = (*inferred_item_shape)[i];
      int64_t dim = item_shape[i];
      if (ShapedType::isDynamic(inferred_dim)) inferred_dim = dim;
    }
  }

  // If all indices are constants, then verify that they cover all indices in
  // the range [0, max_index] and the output type is legal.
  if (all_indices_const) {
    for (int32_t i = 0; i <= max_index; i++) {
      if (!index_values.count(i))
        return op.emitOpError() << "missing index " << i;
    }

    if (inferred_item_shape) {
      SmallVector<int64_t, 4> expected_shape;
      expected_shape.push_back(max_index + 1);
      expected_shape.append(inferred_item_shape->begin(),
                            inferred_item_shape->end());

      auto out_ty = op.getType().cast<TensorType>();
      auto expected_out_ty =
          RankedTensorType::get(expected_shape, out_ty.getElementType());

      if (!AreCastCompatible(out_ty, expected_out_ty)) {
        return op.emitOpError() << "has invalid output type; should be "
                                   "compatible with inferred type "
                                << expected_out_ty;
      }
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// EinsumOp
//===----------------------------------------------------------------------===//

// Verifies that,
// * Arity of the op is at most two.
//
// TODO(hinsu): Verify einsum equation attribute.
static LogicalResult Verify(EinsumOp op) {
  if (op.N() > 2) {
    return op.emitOpError("supports at most two operands");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// EmptyTensorListOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(EmptyTensorListOp op) {
  if (!IsOfRankOrUnranked(op.element_shape(), 0) &&
      !IsOfRankOrUnranked(op.element_shape(), 1)) {
    return op.emitOpError("requires element_shape operand to be 0D/1D tensor");
  }

  if (!IsOfRankOrUnranked(op.max_num_elements(), 0)) {
    return op.emitOpError("requires max_num_elements operand to be 0D tensor");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// EqualOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(EqualOp op) {
  // If we allow inputs to have incompatible type, then nothing to do.
  if (!op.incompatible_shape_error()) return success();

  // Otherwise, check inputs are broadcastable.
  return mlir::OpTrait::impl::verifyCompatibleOperandBroadcast(
      op.getOperation());
}

void EqualOp::build(Builder *builder, OperationState &result, Value x, Value y,
                    BoolAttr incompatible_shape_error) {
  auto result_type = DeduceEqualCmpOpType(builder, result.location, x, y,
                                          incompatible_shape_error);
  return build(builder, result, result_type, x, y, incompatible_shape_error);
}

//===----------------------------------------------------------------------===//
// ExpandDimsOp
//===----------------------------------------------------------------------===//

Type InferExpandDimsOpType(Value input, Value dim) {
  Type element_ty = input->getType().cast<TensorType>().getElementType();
  auto unranked_ty = UnrankedTensorType::get(element_ty);

  auto input_ty = input->getType().dyn_cast<RankedTensorType>();
  if (!input_ty) return unranked_ty;

  DenseIntElementsAttr dim_attr;
  if (!matchPattern(dim, m_Constant(&dim_attr)) ||
      dim_attr.getNumElements() != 1)
    return unranked_ty;
  int64_t dim_val = (*dim_attr.begin()).getSExtValue();
  int64_t input_rank = input_ty.getRank();

  if (dim_val < -input_rank - 1 || dim_val > input_rank + 1) return unranked_ty;
  if (dim_val < 0) dim_val += input_rank + 1;

  SmallVector<int64_t, 4> shape = llvm::to_vector<4>(input_ty.getShape());
  shape.insert(shape.begin() + dim_val, 1);
  return RankedTensorType::get(shape, element_ty);
}

void ExpandDimsOp::build(Builder *builder, OperationState &result, Value input,
                         Value dim) {
  return build(builder, result, InferExpandDimsOpType(input, dim), input, dim);
}

//===----------------------------------------------------------------------===//
// FakeQuantWithMinMaxArgsOp
//===----------------------------------------------------------------------===//
static LogicalResult Verify(FakeQuantWithMinMaxArgsOp op) {
  // TODO(fengliuai): moving the following to an utility method.
  const llvm::fltSemantics &semantics = op.min().getSemantics();
  float rmin, rmax;
  if (&semantics == &APFloat::IEEEsingle()) {
    rmin = op.min().convertToFloat();
    rmax = op.max().convertToFloat();
  } else {
    rmin = op.min().convertToDouble();
    rmax = op.max().convertToDouble();
  }
  // Range boundaries must be valid.
  if (rmin >= rmax) {
    return op.emitOpError("range is invalid: [" + Twine(std::to_string(rmin)) +
                          "," + Twine(std::to_string(rmax)) + "]");
  }
  int64_t num_bits = op.num_bits().getSExtValue();
  if (num_bits < 2 || num_bits > 16) {
    return op.emitOpError(
        "requires num_bits to be between 2 and 16, inclusive");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// FakeQuantWithMinMaxVarsOp
//===----------------------------------------------------------------------===//
static LogicalResult Verify(FakeQuantWithMinMaxVarsOp op) {
  if (!isOfRankedFloatTensorType(op.min(), 0))
    return op.emitOpError("requires min to be a 0d float tensor");

  if (!isOfRankedFloatTensorType(op.max(), 0))
    return op.emitOpError("requires max to be a 0d float tensor");

  int64_t num_bits = op.num_bits().getSExtValue();
  if (num_bits < 2 || num_bits > 16) {
    return op.emitOpError(
        "requires num_bits to be between 2 and 16, inclusive");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// FakeQuantWithMinMaxVarsPerChannelOp
//===----------------------------------------------------------------------===//
static LogicalResult Verify(FakeQuantWithMinMaxVarsPerChannelOp op) {
  if (!isOfRankedFloatTensorType(op.min(), 1))
    return op.emitOpError("requires min to be a 1d float tensor");

  if (!isOfRankedFloatTensorType(op.max(), 1))
    return op.emitOpError("requires max to be a 1d float tensor");

  Value inputs = op.inputs();
  if (!HasRankAtLeast(inputs, 1) ||
      inputs->getType().isa<UnrankedTensorType>()) {
    return op.emitError("requires inputs to be at least 1d float tensor");
  }

  auto inputsType = inputs->getType().cast<ShapedType>();
  int depth = inputsType.getDimSize(inputsType.getRank() - 1);
  if (op.min()->getType().cast<ShapedType>().getDimSize(0) != depth ||
      op.max()->getType().cast<ShapedType>().getDimSize(0) != depth) {
    return op.emitOpError(
        "requires min and max to have same size as last dimension of inputs");
  }
  int64_t num_bits = op.num_bits().getSExtValue();
  if (num_bits < 2 || num_bits > 16) {
    return op.emitOpError(
        "requires num_bits to be between 2 and 16, inclusive");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// FillOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(FillOp op) {
  if (!IsOfRankOrUnranked(op.dims(), 1))
    return op.emitOpError() << "requires dims to be a 1D tensor";
  if (!IsOfRankOrUnranked(op.value(), 0))
    return op.emitOpError() << "requires value to be a scalar";

  return success();
}

//===----------------------------------------------------------------------===//
// FusedBatchNormOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(FusedBatchNormOp op) {
  if (!isOfRankedFloatTensorType(op.x(), 4))
    return op.emitOpError("requires x to be a 4D float tensor");

  if (!isOfRankedFloatTensorType(op.scale(), 1))
    return op.emitOpError("requires scale to be a 1D float tensor");

  if (!isOfRankedFloatTensorType(op.offset(), 1))
    return op.emitOpError("requires offset to be a 1D float tensor");

  if (!isOfRankedFloatTensorType(op.mean(), 1))
    return op.emitOpError("requires mean to be a 1D float tensor");

  if (!isOfRankedFloatTensorType(op.variance(), 1))
    return op.emitOpError("requires variance to be a 1D float tensor");

  // TODO(antiagainst): check attributes

  return success();
}

//===----------------------------------------------------------------------===//
// GatherV2Op
//===----------------------------------------------------------------------===//

static LogicalResult Verify(GatherV2Op op) {
  int64_t batch_dims = op.batch_dims().getSExtValue();
  if (auto ty = op.indices()->getType().dyn_cast<RankedTensorType>()) {
    int64_t rank = ty.getRank();
    if (batch_dims > rank || batch_dims < -rank)
      return op.emitOpError()
             << "batch_dims (" << batch_dims << ") must be in range [" << -rank
             << ", " << rank + 1 << ")";
    if (batch_dims < 0) batch_dims += rank;
  }

  if (!HasRankAtMost(op.axis(), 1))
    return op.emitOpError("requires axis to have rank at most 1");

  DenseIntElementsAttr axis_attr;
  if (matchPattern(op.axis(), m_Constant(&axis_attr))) {
    int64_t axis = (*axis_attr.begin()).getSExtValue();
    if (auto ty = op.params()->getType().dyn_cast<RankedTensorType>()) {
      int64_t rank = ty.getRank();
      if (axis >= rank || axis < -rank)
        return op.emitOpError() << "axis (" << axis << ") must be in range ["
                                << -rank << ", " << rank << ")";
      if (axis < 0) axis += rank;
    }

    if (batch_dims >= 0 && axis >= 0 && axis < batch_dims) {
      return op.emitOpError() << "requires axis (" << axis
                              << ") to be greater than or equal to batch_dims ("
                              << batch_dims << ")";
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(IfOp op) {
  auto module = op.getParentOfType<ModuleOp>();
  auto thenFn = module.lookupSymbol<FuncOp>(op.then_branch());
  if (!thenFn)
    return op.emitOpError("then_branch refers to an undefined function : ")
           << op.then_branch();
  auto elseFn = module.lookupSymbol<FuncOp>(op.else_branch());
  if (!elseFn)
    return op.emitOpError("else_branch refers to an undefined function : ")
           << op.else_branch();
  auto thenFuncType = thenFn.getType();
  auto elseFuncType = elseFn.getType();

  // Non-conditional operands starting with the second operand are passed to
  // branches and should be pair-wise compatible with branches' inputs.
  unsigned expectedNumInputs = op.getNumOperands() - 1;
  if (thenFuncType.getNumInputs() != expectedNumInputs ||
      elseFuncType.getNumInputs() != expectedNumInputs)
    return op.emitError("branches should have " + Twine(expectedNumInputs) +
                        " inputs");

  for (unsigned i = 0; i < expectedNumInputs; ++i) {
    auto operandType = op.getOperand(i + 1)->getType().cast<TensorType>();
    auto thenInputType = thenFuncType.getInput(i).cast<TensorType>();
    if (!AreCastCompatible(operandType, thenInputType))
      return op.emitError(
          llvm::formatv("then branch input type {0} is incompatible with "
                        "operand type {1} at index {2}",
                        thenInputType, operandType, i));

    auto elseInputType = elseFuncType.getInput(i).cast<TensorType>();
    if (!AreCastCompatible(operandType, elseInputType))
      return op.emitError(
          llvm::formatv("else branch input type {0} is incompatible with "
                        "operand type {1} at index {2}",
                        elseInputType, operandType, i));

    // If branches have incompatible input types that means that no tensor can
    // serve as input to both the functions. Hence, the op is invalid.
    if (!AreCastCompatible(thenInputType, elseInputType))
      return op.emitError(llvm::formatv(
          "branches inputs have incompatible types {0} and {1} at index {2}",
          thenInputType, elseInputType, i));
  }

  // Branches' results should be pair-wise compatible with the op results.
  unsigned expectedNumResults = op.getNumResults();
  if (thenFuncType.getNumResults() != expectedNumResults ||
      elseFuncType.getNumResults() != expectedNumResults)
    return op.emitError("branches should have " + Twine(expectedNumResults) +
                        " results");

  for (unsigned i = 0; i < expectedNumResults; ++i) {
    auto resultType = op.getResult(i)->getType().cast<TensorType>();
    auto thenResultType = thenFuncType.getResult(i).cast<TensorType>();
    if (!AreCastCompatible(thenResultType, resultType))
      return op.emitError(
          llvm::formatv("then branch result type {0} is incompatible with op "
                        "result type {1} at index {2}",
                        thenResultType, resultType, i));

    auto elseResultType = elseFuncType.getResult(i).cast<TensorType>();
    if (!AreCastCompatible(elseResultType, resultType))
      return op.emitError(
          llvm::formatv("else branch result type {0} is incompatible with op "
                        "result type {1} at index {2}",
                        elseResultType, resultType, i));
  }
  return success();
}

//===----------------------------------------------------------------------===//
// InvertOp
//===----------------------------------------------------------------------===//

void InvertOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                           MLIRContext *context) {
  results.insert<InvertNested>(context);
}

//===----------------------------------------------------------------------===//
// LeakyReluOp
//===----------------------------------------------------------------------===//

OpFoldResult LeakyReluOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 1 && "leaky relu has one operand");

  // leaky_relu(x, alpha: 1) -> x
  if (alpha().convertToFloat() == 1.0f) return getOperand();

  auto calculate = [&](FloatAttr arg) {
    APFloat val = arg.getValue();
    if (val.isNegative()) val = alpha() * val;
    return FloatAttr::get(arg.getType(), val);
  };

  if (auto arg = operands[0].dyn_cast_or_null<FloatAttr>()) {
    return calculate(arg);
  } else if (auto arg = operands[0].dyn_cast_or_null<SplatElementsAttr>()) {
    if (auto elementAttr = arg.getSplatValue().dyn_cast<FloatAttr>())
      return DenseElementsAttr::get(arg.getType(), calculate(elementAttr));
  }
  return {};
}

//===----------------------------------------------------------------------===//
// LogOp
//===----------------------------------------------------------------------===//

void LogOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                        MLIRContext *context) {
  results.insert<LogOfSoftmax>(context);
}

//===----------------------------------------------------------------------===//
// LogicalNotOp
//===----------------------------------------------------------------------===//

void LogicalNotOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<LogicalNotNested, LogicalNotOfEqual, LogicalNotOfNotEqual,
                 LogicalNotOfGreater, LogicalNotOfGreaterEqual,
                 LogicalNotOfLess, LogicalNotOfLessEqual>(context);
}

//===----------------------------------------------------------------------===//
// MaxOp
//===----------------------------------------------------------------------===//

void MaxOp::build(Builder *builder, OperationState &result, Value input,
                  Value reduction_indices, BoolAttr keep_dims) {
  Type out_ty =
      InferReductionOpType(input, reduction_indices, keep_dims, builder);
  build(builder, result, out_ty, input, reduction_indices, keep_dims);
}

//===----------------------------------------------------------------------===//
// MaxPoolGradOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(MaxPoolGradOp op) {
  if (!IsOfRankOrUnranked(op.orig_input(), 4)) {
    return op.emitOpError() << "requires orig_input to be rank 4";
  }
  if (!IsOfRankOrUnranked(op.orig_output(), 4)) {
    return op.emitOpError() << "requires orig_output to be rank 4";
  }
  if (!IsOfRankOrUnranked(op.grad(), 4)) {
    return op.emitOpError() << "requires grad to be rank 4";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// NegOp
//===----------------------------------------------------------------------===//

void NegOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                        MLIRContext *context) {
  results.insert<NegNested>(context);
}

//===----------------------------------------------------------------------===//
// NotEqualOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(NotEqualOp op) {
  // If we allow inputs to have incompatible type, then nothing to do.
  if (!op.incompatible_shape_error()) return success();

  // Otherwise, check inputs are broadcastable.
  return mlir::OpTrait::impl::verifyCompatibleOperandBroadcast(
      op.getOperation());
}

void NotEqualOp::build(Builder *builder, OperationState &result, Value x,
                       Value y, BoolAttr incompatible_shape_error) {
  auto result_type = DeduceEqualCmpOpType(builder, result.location, x, y,
                                          incompatible_shape_error);
  return build(builder, result, result_type, x, y, incompatible_shape_error);
}

//===----------------------------------------------------------------------===//
// OneHotOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(OneHotOp op) {
  int64_t axis = op.axis().getSExtValue();

  auto indices_ty = op.indices()->getType().dyn_cast<RankedTensorType>();
  if (indices_ty &&
      !(axis == -1 || (axis >= 0 && axis <= indices_ty.getShape().size()))) {
    return op.emitOpError()
           << "expected axis (" << axis << ") to be -1 or between [0, "
           << indices_ty.getShape().size() << "]";
  }

  if (axis < -1) {
    return op.emitOpError() << "expected axis (" << axis
                            << ") to be -1 or between [0, rank(indices()))";
  }

  if (!IsOfRankOrUnranked(op.depth(), 0)) {
    return op.emitOpError() << "requires depth to be a scalar";
  }
  if (!IsOfRankOrUnranked(op.on_value(), 0)) {
    return op.emitOpError() << "requires on_value to be a scalar";
  }
  if (!IsOfRankOrUnranked(op.off_value(), 0)) {
    return op.emitOpError() << "requires off_value to be a scalar";
  }

  DenseIntElementsAttr depth_attr;
  if (matchPattern(op.depth(), m_Constant(&depth_attr))) {
    if (depth_attr.getType().getRank() != 0)
      return op.emitOpError() << "requires depth to be a scalar";
    int64_t depth = depth_attr.getValue<APInt>({}).getSExtValue();
    if (depth < 0) {
      return op.emitOpError() << "depth must be non-negative, got: " << depth;
    }
  }

  return success();
}

static TensorType InferOneHotOpType(Value indices, Value depth, Value on_value,
                                    Value off_value, IntegerAttr axis) {
  int64_t axis_val = axis.getInt();
  Type element_ty = on_value->getType().cast<TensorType>().getElementType();
  auto unranked_ty = UnrankedTensorType::get(element_ty);
  if (axis_val < -1) return unranked_ty;

  auto indices_ty = indices->getType().dyn_cast<RankedTensorType>();
  if (!indices_ty) return unranked_ty;

  auto shape = llvm::to_vector<2>(indices_ty.getShape());
  if (axis_val == -1) axis_val = shape.size();

  int64_t depth_val = ShapedType::kDynamicSize;
  DenseIntElementsAttr depth_attr;
  if (matchPattern(depth, m_Constant(&depth_attr)) &&
      depth_attr.getNumElements() == 1)
    depth_val = (*depth_attr.begin()).getSExtValue();
  shape.insert(shape.begin() + axis_val, depth_val);
  return RankedTensorType::get(shape, element_ty);
}

void OneHotOp::build(Builder *builder, OperationState &result, Value indices,
                     Value depth, Value on_value, Value off_value,
                     IntegerAttr axis) {
  build(builder, result,
        InferOneHotOpType(indices, depth, on_value, off_value, axis), indices,
        depth, on_value, off_value, axis);
}

//===----------------------------------------------------------------------===//
// PackOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(PackOp op) {
  // TODO(hinsu): Convert variadic length attributes to derived attributes.
  Operation::operand_range values = op.values();

  if (failed(VerifyTypesCompatibility(values,
                                      /*mask_one_dim=*/false,
                                      op.getOperation()))) {
    return failure();
  }

  int64_t inputs_rank = -1;
  for (Value value : values) {
    if (auto ty = value->getType().dyn_cast<RankedTensorType>()) {
      // Exit early as input types are verified to be compatible so all ranked
      // tensors have the same rank.
      inputs_rank = ty.getRank();
      break;
    }
  }
  if (inputs_rank == -1) return success();

  // The values can be packed along any of the dimensions between 0 and
  // inputs rank, inclusive. Also, as the negative axis values wrap around so
  // the axis value range is [-(R+1), R+1).
  int64_t range_begin = -inputs_rank - 1;  // Inclusive
  int64_t range_end = inputs_rank + 1;     // Exclusive
  int64_t axis = op.axis().getSExtValue();
  if (axis < range_begin || axis >= range_end) {
    return op.emitError() << "attribute 'axis' should be within range ["
                          << range_begin << ", " << range_end
                          << "); actual value: " << axis;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ParseExampleV2Op
//===----------------------------------------------------------------------===//

static LogicalResult Verify(ParseExampleV2Op op) {
  // NOTE(mrry): This validates properties of an op that would previously be
  // validated by the TensorFlow OpDef type checker. In addition to these
  // checks, the shape inference function for ParseExampleV2 validates the
  // consistency of the argument and result types.

  // Validate dense variadic input and output lengths.
  // NOTE(mrry): The Tdense attr is derived from dense_defaults, so we
  // do not need to validate dense_defaults.
  auto dense_types_count =
      std::distance(op.Tdense().begin(), op.Tdense().end());
  auto dense_values_count =
      std::distance(op.dense_values().begin(), op.dense_values().end());
  if (dense_values_count != dense_types_count) {
    return op.emitError() << "output 'dense_values' should have same length "
                          << "as attribute 'Tdense'";
  }

  // Validate sparse variadic output lengths.
  // NOTE(mrry): The sparse_types attr is derived from sparse_values, so we
  // do not need to validate sparse_values.
  auto sparse_types_count =
      std::distance(op.sparse_types().begin(), op.sparse_types().end());
  if (op.num_sparse() != sparse_types_count) {
    return op.emitError() << "attribute 'num_sparse' should be the same as "
                          << "the length of attribute 'sparse_types'";
  }
  if (op.sparse_indices().size() != sparse_types_count) {
    return op.emitError() << "output 'sparse_indices' should have same length "
                          << "as attribute 'sparse_types'";
  }
  if (op.sparse_shapes().size() != sparse_types_count) {
    return op.emitError() << "output 'sparse_shapes' should have same length "
                          << "as attribute 'sparse_types'";
  }

  // Validate ragged variadic output lengths.
  auto ragged_value_types_count = std::distance(op.ragged_value_types().begin(),
                                                op.ragged_value_types().end());
  auto ragged_split_types_count = std::distance(op.ragged_split_types().begin(),
                                                op.ragged_split_types().end());
  if (ragged_value_types_count != ragged_split_types_count) {
    return op.emitError() << "attribute 'ragged_value_types' should have same "
                          << "length as attribute 'ragged_split_types'";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ReciprocalOp
//===----------------------------------------------------------------------===//

void ReciprocalOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<ReciprocalNested>(context);
}

//===----------------------------------------------------------------------===//
// RandomUniformOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(RandomUniformOp op) {
  if (!IsOfRankOrUnranked(op.shape(), 1))
    return op.emitOpError("shape must be 1D tensor");
  return success();
}

//===----------------------------------------------------------------------===//
// RangeOp
//===----------------------------------------------------------------------===//

void RangeOp::build(Builder *builder, OperationState &result, Value start,
                    Value limit, Value delta) {
  assert(start->getType() == limit->getType());
  assert(start->getType() == delta->getType());
  DenseIntElementsAttr start_val;
  DenseIntElementsAttr limit_val;
  DenseIntElementsAttr delta_val;
  if (matchPattern(start, m_Constant(&start_val)) &&
      matchPattern(limit, m_Constant(&limit_val)) &&
      matchPattern(delta, m_Constant(&delta_val))) {
    auto size = llvm::APIntOps::RoundingSDiv(
        *limit_val.begin() - *start_val.begin(), *delta_val.begin(),
        llvm::APInt::Rounding::DOWN);
    return RangeOp::build(
        builder, result,
        RankedTensorType::get(
            size.getSExtValue(),
            start->getType().cast<TensorType>().getElementType()),
        start, limit, delta);
  }
  return RangeOp::build(
      builder, result,
      RankedTensorType::get(
          {-1}, start->getType().cast<TensorType>().getElementType()),
      start, limit, delta);
}
//===----------------------------------------------------------------------===//
// RankOp
//===----------------------------------------------------------------------===//

void RankOp::build(Builder *builder, OperationState &result, Value input) {
  return RankOp::build(builder, result,
                       RankedTensorType::get({}, builder->getIntegerType(32)),
                       input);
}

//===----------------------------------------------------------------------===//
// RealDivOp
//===----------------------------------------------------------------------===//

void RealDivOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                            MLIRContext *context) {
  results.insert<RealDivWithSqrtDivisor>(context);
}

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

// TODO(b/128020684): Verify the rank of the output and change to use
// m_Constant.
static LogicalResult Verify(ReshapeOp op) {
  auto shapeType = op.shape()->getType().cast<TensorType>();
  if (!shapeType.hasRank()) return success();
  if (shapeType.getRank() != 1)
    return op.emitOpError("shape must be 1D tensor");
  auto rankByShape = shapeType.getShape()[0];
  auto typeOfTensor = op.tensor()->getType().cast<TensorType>();
  // No compile time verification for unknown sized shape.
  if (rankByShape == -1 || !typeOfTensor.hasStaticShape()) return success();
  // Check values if constant shape. No compiling time verification for
  // non-constant shape.
  auto *shapeOp = op.shape()->getDefiningOp();
  if (!shapeOp) return success();
  Attribute shapeCst;
  if (auto shapeStdOp = dyn_cast<ConstantOp>(shapeOp)) {
    shapeCst = shapeStdOp.getValue();
  } else if (auto shapeTFOp = dyn_cast<ConstOp>(shapeOp)) {
    shapeCst = shapeTFOp.value();
  } else {
    return success();
  }
  auto shapeCstAttr = shapeCst.dyn_cast<ElementsAttr>();
  if (!shapeCstAttr) return op.emitOpError("shape must be a valid tensor");

  if (auto opaqueAttr = shapeCstAttr.dyn_cast<OpaqueElementsAttr>()) {
    opaqueAttr.decode(shapeCstAttr);
  }

  // We know the shape is a 1-D Tensor, then let us get the number of
  // elements it implies.
  unsigned numByShape = 1;
  unsigned unknownDimCount = 0;
  for (int i = 0, e = rankByShape; i != e; ++i) {
    auto num = shapeCstAttr.getValue<IntegerAttr>(i).getInt();
    // The dimension size value can be -1, and that the real size needs to
    // be computed so that the total size remains constant. At most one
    // component of shape can be -1.
    if (num == -1) {
      if (++unknownDimCount > 1) {
        return op.emitOpError("more than one component of shape are -1");
      }
    } else {
      numByShape *= num;
    }
  }
  auto numByTensor = typeOfTensor.getNumElements();
  // If there is one component of shape is -1, the dimension should be
  // computed so that the total size remains constant.
  if (unknownDimCount == 1) {
    if (numByTensor % numByShape != 0)
      return op.emitOpError(
          "one component of shape is -1 but couldn't infer the dimension");
    return success();
  }
  // If the elements by the tensor and implies by the shape don't match,
  // fail this static check.
  if (numByTensor != numByShape) {
    return op.emitOpError(
        "mismatch in tensor elements and shape implied elements");
  }
  return success();
}

void ReshapeOp::build(Builder *builder, OperationState &result, Value tensor,
                      Value shape) {
  auto ttype = tensor->getType().cast<ShapedType>();
  auto etype = ttype.getElementType();

  auto unranked = [builder, etype, &result, shape, tensor]() {
    return ReshapeOp::build(builder, result, UnrankedTensorType::get(etype),
                            tensor, shape);
  };

  // If tensor is unranked then we have no info about output of shape.
  if (!ttype.hasRank()) return unranked();

  DenseIntElementsAttr attr_shape;
  if (matchPattern(shape, m_Constant(&attr_shape))) {
    llvm::SmallVector<int64_t, 4> const_shape;
    const_shape.reserve(attr_shape.getNumElements());

    // Detect if reshape output shape is folded.
    bool flatten = false;
    int unknown_index = -1;
    // The product of constant shape argument excluding unknown dimension.
    int64_t product_cshape = 1;
    for (auto e : llvm::enumerate(attr_shape)) {
      int64_t val = e.value().getSExtValue();
      if (IsUnknownDimOrRank(val)) {
        if (flatten) {
          mlir::emitError(result.location)
              << "only one unknown dimension allowed";
          return;
        }
        flatten = true;
        unknown_index = e.index();
      } else {
        product_cshape *= val;
      }
      const_shape.push_back(val);
    }

    // Compute the value of the unknown dimension.
    if (flatten) {
      // Compute number of elements in tensor shape.
      auto tshape = ttype.getShape();
      int64_t product_tshape = std::accumulate(tshape.begin(), tshape.end(), 1,
                                               std::multiplies<int64_t>());
      // Set the unknown dimension such that total number of elements remain
      // constant.
      // Note: The case where the ratio is not integral, and so the total size
      // of reshape not constant, is checked in verify function.
      const_shape[unknown_index] = product_tshape / product_cshape;
    }
    return ReshapeOp::build(builder, result,
                            RankedTensorType::get(const_shape, etype), tensor,
                            shape);
  }
  return unranked();
}

//===----------------------------------------------------------------------===//
// SelectV2Op
//===----------------------------------------------------------------------===//

static Type InferSelectV2OpType(Value condition, Value e, Value t) {
  Type element_ty = e->getType().cast<TensorType>().getElementType();
  auto unranked_ty = UnrankedTensorType::get(element_ty);

  Type broadcasted_ty =
      OpTrait::util::getBroadcastedType(e->getType(), t->getType());
  if (!broadcasted_ty) return unranked_ty;

  auto cond_ranked_ty = condition->getType().dyn_cast<RankedTensorType>();
  auto broadcasted_ranked_ty = broadcasted_ty.dyn_cast<RankedTensorType>();
  if (!cond_ranked_ty || !broadcasted_ranked_ty) return unranked_ty;

  // Explicitly get broadcasted output type as element types of condition may
  // not be same as the broadcated type's element type.
  SmallVector<int64_t, 4> result_shape;
  if (!OpTrait::util::getBroadcastedShape(cond_ranked_ty.getShape(),
                                          broadcasted_ranked_ty.getShape(),
                                          result_shape))
    return unranked_ty;
  return RankedTensorType::get(result_shape, element_ty);
}

void SelectV2Op::build(Builder *builder, OperationState &result,
                       Value condition, Value e, Value t) {
  build(builder, result, InferSelectV2OpType(condition, e, t), condition, e, t);
}

//===----------------------------------------------------------------------===//
// ShapeOp
//===----------------------------------------------------------------------===//

namespace {
// Validates Shape/ShapeN/VariableShape operand and associated result types.
LogicalResult VerifyShapeOperandAndResult(Operation *op, Type operand_type,
                                          Type result_type,
                                          int variadic_idx = -1) {
  std::string variadic_idx_str =
      variadic_idx < 0 ? "" : llvm::formatv(" #{0}", variadic_idx).str();

  auto result_ranked_type = result_type.dyn_cast<RankedTensorType>();
  if (!result_ranked_type || result_ranked_type.getShape().size() != 1)
    return op->emitOpError("requires 1D type for result") << variadic_idx_str;

  auto operand_ranked_type = operand_type.dyn_cast_or_null<RankedTensorType>();
  if (operand_ranked_type) {
    // The operand is a ranked tensor.
    if (result_ranked_type.hasStaticShape() &&
        !operand_ranked_type.getShape().empty() &&
        result_ranked_type.getDimSize(0) !=
            operand_ranked_type.getShape().size())
      return op->emitOpError("requires dimension size of result")
             << variadic_idx_str << " to match rank of operand"
             << variadic_idx_str;
  } else if (result_ranked_type.hasStaticShape()) {
    // The operand is an unranked tensor, verify that the result is dynamic.
    return op->emitOpError("requires dynamic shape result")
           << variadic_idx_str << " for unranked operand" << variadic_idx_str;
  }

  Type element_type = result_ranked_type.getElementType();
  if (!element_type.isInteger(32) && !element_type.isInteger(64))
    return op->emitOpError("requires int32 or int64 return type for result")
           << variadic_idx_str;

  return success();
}
}  // anonymous namespace

static LogicalResult Verify(ShapeOp op) {
  return VerifyShapeOperandAndResult(op, op.input()->getType(), op.getType());
}

// Converts shape of the given type to attribute if it is of ranked tensor type.
// Returned attribute has integer elements of the given width.
static Attribute ConvertShapeToAttr(Type input_ty, int out_width) {
  auto ranked_ty = input_ty.dyn_cast<RankedTensorType>();
  if (!ranked_ty || !ranked_ty.hasStaticShape()) return {};

  auto shape = ranked_ty.getShape();
  int rank = shape.size();

  SmallVector<APInt, 4> dimensions;
  dimensions.reserve(rank);
  for (int i = 0; i < rank; ++i)
    dimensions.push_back(APInt(out_width, shape[i]));

  auto result_type = RankedTensorType::get(
      {rank}, IntegerType::get(out_width, input_ty.getContext()));
  return DenseElementsAttr::get(result_type, dimensions);
}

OpFoldResult ShapeOp::fold(ArrayRef<Attribute> operands) {
  int width =
      getType().cast<ShapedType>().getElementType().getIntOrFloatBitWidth();
  return ConvertShapeToAttr(getOperand()->getType(), width);
}

void ShapeOp::build(Builder *builder, OperationState &result, Value input,
                    BoolAttr use32Bit) {
  auto rankedTensorType = input->getType().dyn_cast<RankedTensorType>();
  int64_t rank = rankedTensorType ? rankedTensorType.getRank() : -1;
  auto out_type = use32Bit.getValue() ? builder->getIntegerType(32)
                                      : builder->getIntegerType(64);
  return ShapeOp::build(builder, result,
                        RankedTensorType::get({rank}, out_type), input);
}

//===----------------------------------------------------------------------===//
// ShapeNOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(ShapeNOp op) {
  const size_t num_tensors = op.N();

  if (op.getNumOperands() != num_tensors)
    return op.emitOpError() << "requires " << num_tensors << " operand(s), got "
                            << op.getNumOperands() << " operand(s)";

  if (op.getNumResults() != num_tensors)
    return op.emitOpError() << "requires " << num_tensors << " result(s), got "
                            << op.getNumResults() << " result(s)";

  for (auto i : llvm::seq<uint64_t>(0, num_tensors)) {
    auto verification = VerifyShapeOperandAndResult(
        op, op.getOperand(i)->getType(), op.getResult(i)->getType(), i);
    if (failed(verification)) return verification;
  }

  return success();
}

LogicalResult ShapeNOp::fold(ArrayRef<Attribute> operands,
                             SmallVectorImpl<OpFoldResult> &results) {
  if (getNumOperands() == 0) return success();
  int width =
      getType(0).cast<ShapedType>().getElementType().getIntOrFloatBitWidth();

  for (Type input_ty : getOperandTypes()) {
    OpFoldResult result = ConvertShapeToAttr(input_ty, width);
    if (!result) return failure();

    results.push_back(result);
  }
  return success();
}

// TODO(hinsu): Add canonicalization pattern for ShapeN ops that don't have all
// static input shapes. Replacing output values corresponding to static input
// types may enable optimizations in users of the values.

//===----------------------------------------------------------------------===//
// SizeOp
//===----------------------------------------------------------------------===//

// Verifies that,
//
// * Input type, if is a ranked tensor, has at most INT32_MAX dimensions.
//
static LogicalResult Verify(SizeOp op) {
  if (!HasRankAtMost(op.input(), std::numeric_limits<int32_t>::max()))
    return op.emitOpError(
        "requires ranked input tensor to be of rank INT32_MAX or less");

  return success();
}

//===----------------------------------------------------------------------===//
// SliceOp
//===----------------------------------------------------------------------===//

// Verifies that,
//
// - operands begin and size are 1D with the same number of elements.
// - if the input is a ranked tensor, the rank of the input equals the number
//   of elements in operands begin and size.
// - if begin are constants, 0 <= begin[i] < input_ty.getShape()[i]
//
static LogicalResult Verify(SliceOp op) {
  RankedTensorType begin_ty = GetRankedTensorTypeForOperand(op.begin());
  if (begin_ty && begin_ty.getRank() != 1) {
    return op.emitOpError() << "requires begin operand to be 1D tensor";
  }

  RankedTensorType size_ty = GetRankedTensorTypeForOperand(op.size());
  if (size_ty && size_ty.getRank() != 1) {
    return op.emitOpError() << "requires size operand to be 1D tensor";
  }

  if (!begin_ty || !size_ty || !begin_ty.hasStaticShape() ||
      !size_ty.hasStaticShape())
    return success();

  if (begin_ty.getNumElements() != size_ty.getNumElements()) {
    return op.emitOpError() << "requires begin and size operands to have the"
                               " same number of elements";
  }

  auto input_ty = op.input()->getType().dyn_cast<RankedTensorType>();
  if (input_ty && begin_ty.getNumElements() != input_ty.getRank()) {
    return op.emitOpError() << "requires number of elements in begin and size"
                               "are equal to input rank";
  }

  DenseIntElementsAttr begin_indices;
  if (matchPattern(op.begin(), m_Constant(&begin_indices))) {
    DenseIntElementsAttr slice_sizes;
    bool constant_slice_sizes =
        matchPattern(op.size(), m_Constant(&slice_sizes));
    int dim = 0;
    for (APInt raw_begin_index : begin_indices.getValues<APInt>()) {
      int64_t begin_index = raw_begin_index.getSExtValue();
      int64_t input_size = input_ty ? input_ty.getShape()[dim] : -1;
      int64_t slice_size = constant_slice_sizes
                               ? slice_sizes.getValue<APInt>(dim).getSExtValue()
                               : 0;
      if (slice_size == -1 && input_size != -1) {
        slice_size = input_size - begin_index;
      }
      if (begin_index < 0 ||
          (input_size != -1 && begin_index + slice_size > input_size)) {
        return op.emitOpError()
               << "requires 0 <= begin[i] <= begin[i] + size[i] <= Di";
      }
      ++dim;
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// SoftmaxOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(SoftmaxOp op) {
  if (!HasRankAtLeast(op.logits(), 1)) {
    return op.emitOpError("requires operand to have rank at least 1");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// SoftmaxCrossEntropyWithLogitsOp
//===----------------------------------------------------------------------===//

// Verifies that,
//
// * Input types are broadcast compatible and the broadcasted type has rank two.
//
static LogicalResult Verify(SoftmaxCrossEntropyWithLogitsOp op) {
  auto broadcasted_ty = OpTrait::util::getBroadcastedType(
                            op.features()->getType(), op.labels()->getType())
                            .dyn_cast_or_null<ShapedType>();
  if (!broadcasted_ty ||
      (broadcasted_ty.hasRank() && broadcasted_ty.getRank() != 2))
    return op.emitOpError(
        "requires features and labels to be broadcast compatible to rank two");

  return success();
}

//===----------------------------------------------------------------------===//
// SparseSoftmaxCrossEntropyWithLogitsOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(SparseSoftmaxCrossEntropyWithLogitsOp op) {
  if (!IsOfRankOrUnranked(op.features(), 2)) {
    return op.emitOpError("requires features operand of rank two");
  }
  if (!IsOfRankOrUnranked(op.labels(), 1)) {
    return op.emitOpError("requires labels operand of rank one");
  }
  auto features_ty = op.features()->getType().dyn_cast<RankedTensorType>();
  auto labels_ty = op.labels()->getType().dyn_cast<RankedTensorType>();
  if (features_ty && labels_ty) {
    int64_t features_batches = features_ty.getDimSize(0);
    int64_t labels_batches = labels_ty.getDimSize(0);
    if (!ShapedType::isDynamic(features_batches) &&
        !ShapedType::isDynamic(labels_batches) &&
        features_batches != labels_batches)
      return op.emitOpError(
          "requires features and labels with matching first dimension");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// SplitOp
//===----------------------------------------------------------------------===//

// Verifies the input and split dimension operands for tf.Split/tf.SplitV.
// Writes the split dimension's index (adjusted with input rank) via `dim_index`
// if it's a constant.
template <class Op>
LogicalResult VerifySplitInputAndSplitDim(Op op, Optional<int64_t> *dim_index) {
  *dim_index = llvm::None;

  Value split_dim = op.split_dim();
  if (auto split_dim_type = split_dim->getType().dyn_cast<RankedTensorType>())
    if (split_dim_type.getRank() != 0)
      return op.emitOpError(
          "split dimension should be an integer scalar tensor");

  // We can perform further verification if the input tensor to be split has
  // known rank and the split dimension tensor is a constant.

  auto input_type = op.value()->getType().template dyn_cast<RankedTensorType>();
  if (!input_type) return success();

  int64_t input_rank = input_type.getRank();
  if (input_rank == 0)
    return op.emitOpError("cannot split scalar input tensor");

  DenseIntElementsAttr split_dim_attr;
  if (!matchPattern(split_dim, m_Constant(&split_dim_attr))) return success();

  int64_t index = (*split_dim_attr.begin()).getSExtValue();

  if (index + input_rank < 0 || index >= input_rank) {
    return op.emitOpError("split dimension must be in range [-")
           << input_rank << ", " << input_rank << ")";
  }

  if (index < 0) index += input_rank;
  *dim_index = index;

  return success();
}

static LogicalResult Verify(SplitOp op) {
  Optional<int64_t> dim_index;
  if (failed(VerifySplitInputAndSplitDim(op, &dim_index))) return failure();
  if (!dim_index) return success();

  int64_t input_dim_size =
      op.value()->getType().cast<RankedTensorType>().getDimSize(*dim_index);
  if (input_dim_size == ShapedType::kDynamicSize) return success();

  if (input_dim_size % op.getNumResults() != 0)
    return op.emitOpError("dimension #")
           << *dim_index << " not divisible by the number of result tensors";

  return success();
}

//===----------------------------------------------------------------------===//
// SplitVOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(SplitVOp op) {
  auto split_sizes_type =
      op.size_splits()->getType().dyn_cast<RankedTensorType>();
  if (!split_sizes_type) return success();

  if (split_sizes_type.getRank() != 1 ||
      split_sizes_type.getDimSize(0) != op.getNumResults())
    return op.emitOpError("split sizes should be a 1D tensor of ")
           << op.getNumResults() << " elements";

  Optional<int64_t> dim_index = 0;
  if (failed(VerifySplitInputAndSplitDim(op, &dim_index))) return failure();
  if (!dim_index) return success();

  int64_t input_dim_size =
      op.value()->getType().cast<RankedTensorType>().getDimSize(*dim_index);
  if (input_dim_size == ShapedType::kDynamicSize) return success();

  // If split sizes come from a constant, they must sum to the dimension size
  // along split_dim, and we can have no more than one dynamic dimension.
  DenseIntElementsAttr split_sizes_attr;
  if (!matchPattern(op.size_splits(), m_Constant(&split_sizes_attr)))
    return success();

  int64_t total_dim_size = 0;  // Total dimension size assigned to splits
  llvm::Optional<int> dynamic_dim_index;

  SmallVector<int64_t, 4> split_sizes;
  split_sizes.reserve(
      split_sizes_attr.getType().cast<ShapedType>().getNumElements());

  for (auto dim : llvm::enumerate(split_sizes_attr)) {
    int64_t dim_val = dim.value().getSExtValue();
    split_sizes.push_back(dim_val);
    if (dim_val == ShapedType::kDynamicSize) {
      // We cannot have more than one dynamic dimension.
      if (dynamic_dim_index)
        return op.emitOpError(
            "cannot have more than one dynamic dimension in split sizes");
      dynamic_dim_index = dim.index();
    } else {
      total_dim_size += dim_val;
    }
  }

  if (!dynamic_dim_index && total_dim_size != input_dim_size)
    return op.emitOpError(
               "split sizes must sum up to the dimension size along split "
               "dimension, found ")
           << total_dim_size << " vs " << input_dim_size;

  if (dynamic_dim_index && total_dim_size > input_dim_size)
    return op.emitOpError(
               "split sizes must sum up to be less than or equal to the "
               "dimension size along split dimension, found ")
           << total_dim_size << " vs " << input_dim_size;

  return success();
}

//===----------------------------------------------------------------------===//
// SquareOp
//===----------------------------------------------------------------------===//

void SquareOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                           MLIRContext *context) {
  results.insert<SquareOfSub>(context);
}

//===----------------------------------------------------------------------===//
// SubOp
//===----------------------------------------------------------------------===//

void SubOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                        MLIRContext *context) {
  results.insert<SubOfNeg>(context);
}

//===----------------------------------------------------------------------===//
// SumOp
//===----------------------------------------------------------------------===//

void SumOp::build(Builder *builder, OperationState &result, Value input,
                  Value reduction_indices, BoolAttr keep_dims) {
  Type out_ty =
      InferReductionOpType(input, reduction_indices, keep_dims, builder);
  build(builder, result, out_ty, input, reduction_indices, keep_dims);
}

//===----------------------------------------------------------------------===//
// StridedSliceOp
//===----------------------------------------------------------------------===//

// Verifies that,
//
// - begin, end and strides operands are 1D and they have the same number of
//   elements. Here, the number of elements should be less than 32 to support
//   32-bit mask attributes.
// - None of the strides values are zero.
// - Ellipsis mask can have at most one bit set.

template <class OpTy>
static LogicalResult VerifyStridedSliceBase(OpTy op) {
  // Expected size for operands begin, end and strides vector operands.
  int64_t expected_size = -1;

  for (Value val : {op.begin(), op.end(), op.strides()}) {
    auto operand_ty = val->getType().dyn_cast<ShapedType>();
    if (!operand_ty || !operand_ty.hasStaticShape()) {
      // TensorFlow constant ops may have non-static shape because the shape is
      // not propagated during constant folding. If the defining op for this
      // operand is a constant op, use the constant op's attribute to get the
      // actual shape.
      DenseIntElementsAttr attr;
      if (!matchPattern(val, m_Constant(&attr))) continue;
      operand_ty = attr.getType();
    }

    if (operand_ty.getRank() != 1)
      return op.emitOpError()
             << "requires begin, end and strides to be 1D tensors";

    int64_t length = operand_ty.getDimSize(0);
    if (length == -1) continue;

    if (expected_size == -1) {
      // This op uses 32-bit masks.
      if (length >= 32)
        return op.emitOpError(
            "requires begin, end and strides operands with less than 32 "
            "elements");

      expected_size = length;
    } else if (length != expected_size) {
      return op.emitOpError() << "requires begin, end and strides to have the "
                                 "same number of elements";
    }
  }

  // If strides are constants, verify that none of the element is zero.
  DenseIntElementsAttr strides;
  if (matchPattern(op.strides(), m_Constant(&strides))) {
    if (llvm::is_contained(strides.getValues<APInt>(), 0))
      return op.emitOpError("requires non-zero strides");
  }

  // Use bit compares to ensure ellipsis_mask is 0 or a power of 2, i.e. there
  // exists only no more than one ellipsis.
  uint32_t ellipsis_mask = op.ellipsis_mask().getZExtValue();
  if (ellipsis_mask != 0 && !llvm::isPowerOf2_32(ellipsis_mask))
    return op.emitOpError("cannot have multiple ellipses");

  return success();
}

// Clamps the given `val`: returns `low` if `val` is less than `low`; returns
// `high` if `high` is less than `val`; otherwise returns `val`.
template <class T>
constexpr const T &Clamp(const T &val, const T &low, const T &high) {
  assert(!(high < low));
  return (val < low) ? low : (high < val) ? high : val;
}

// For the given `input_shape`, calculates the sliced shape using the given
// `begin`, `end`, and `stride` ranges and `begin_mask` and `end_mask` masks.
// Updates the result back to `input_shape`. At the same time, canonicalizes
// `begin`, `end`, and `strides. The calculation follows tf.StridedSlice op
// semantics.
static void CalculateSlicedShapeAndBoundRanges(
    MutableArrayRef<int64_t> input_shape, int32_t begin_mask, int32_t end_mask,
    MutableArrayRef<int64_t> begin, MutableArrayRef<int64_t> end,
    MutableArrayRef<int64_t> stride) {
  assert(input_shape.size() <= 32);  // Only 32-bit masks are supported.

  // Make sure ranges' ranks are consistent with the input.
  assert(input_shape.size() == begin.size());
  assert(input_shape.size() == end.size());
  assert(input_shape.size() == stride.size());

  for (int i = 0, e = input_shape.size(); i < e; ++i) {
    if (ShapedType::isDynamic(input_shape[i])) continue;

    int64_t dim_i = input_shape[i];
    int64_t begin_i = begin[i];
    int64_t end_i = end[i];
    int64_t stride_i = stride[i];

    // [0]: mask for begin, [1]: mask for end
    int64_t masks[] = {begin_mask & (1 << i), end_mask & (1 << i)};
    // [0]: bound for begin, [1]: bound for end
    int64_t bounds[] = {stride_i > 0 ? 0 : -1,
                        stride_i > 0 ? dim_i : dim_i - 1};

    // Canonicalizes the given range `point` (begin/end) according to the
    // current dimension. `c` means case: 0 for begin, 1 for end.
    auto canonicalize = [&](int64_t point, int c) {
      if (masks[c]) return stride_i > 0 ? bounds[c] : bounds[(c + 1) & 1];

      // Add dim as offset to negative range point.
      point = point < 0 ? dim_i + point : point;
      return Clamp(point, bounds[0], bounds[1]);
    };

    begin_i = canonicalize(begin_i, 0);
    end_i = canonicalize(end_i, 1);

    int64_t interval_len = end_i - begin_i;
    int64_t size_i = 0;
    // If internal length is zero or has different sign from stride, it's a
    // degenerated case: we are slicing nothing. Otherwise, calculate the sliced
    // size.
    if (interval_len != 0 && (interval_len < 0) == (stride_i < 0))
      size_i = (interval_len / stride_i) + (interval_len % stride_i != 0);

    input_shape[i] = size_i;
    begin[i] = begin_i;
    end[i] = end_i;
    stride[i] = stride_i;
  }
}

bool StridedSliceOp::GetSlicedBoundRanges(
    ArrayRef<int64_t> shape, SmallVectorImpl<int64_t> *begin_indices,
    SmallVectorImpl<int64_t> *end_indices, SmallVectorImpl<int64_t> *strides) {
  if (this->ellipsis_mask().getZExtValue() ||
      this->new_axis_mask().getZExtValue() ||
      this->shrink_axis_mask().getZExtValue())
    return false;  // TODO(antiagainst): support these masks

  // TODO(hinsu): Support lowering for ops with dynamic begin and end values
  // when it is possible to derive indices based on mask attributes.
  DenseIntElementsAttr begin_indices_attr, end_indices_attr, strides_attr;
  if (!matchPattern(this->begin(), m_Constant(&begin_indices_attr)) ||
      !matchPattern(this->end(), m_Constant(&end_indices_attr)) ||
      !matchPattern(this->strides(), m_Constant(&strides_attr)))
    return false;

  auto input_shape = llvm::to_vector<4>(shape);
  int rank = input_shape.size();

  begin_indices->clear();
  begin_indices->reserve(rank);
  end_indices->clear();
  end_indices->reserve(rank);
  strides->clear();
  strides->reserve(rank);

  for (const APInt &index : begin_indices_attr)
    begin_indices->push_back(index.getSExtValue());
  for (const APInt &index : end_indices_attr)
    end_indices->push_back(index.getSExtValue());
  for (const APInt &stride : strides_attr)
    strides->push_back(stride.getSExtValue());

  CalculateSlicedShapeAndBoundRanges(
      input_shape, this->begin_mask().getZExtValue(),
      this->end_mask().getZExtValue(), *begin_indices, *end_indices, *strides);
  return true;
}

//===----------------------------------------------------------------------===//
// StridedSliceGradOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(StridedSliceGradOp op) {
  auto shape_type = op.shape()->getType().dyn_cast<RankedTensorType>();
  if (shape_type && shape_type.getRank() != 1)
    return op.emitOpError("'shape' operand must be 1D tensor, but got ")
           << shape_type.getRank() << "D tensor";

  if (failed(VerifyStridedSliceBase(op))) return failure();

  // TODO(antiagainst): verify the gradient op.dy()'s shape is consistent with
  // the sliced type from StridedSlice.

  return success();
}

bool StridedSliceGradOp::GetSlicedShapeAndBoundRanges(
    SmallVectorImpl<int64_t> *shape, SmallVectorImpl<int64_t> *begin_indices,
    SmallVectorImpl<int64_t> *end_indices, SmallVectorImpl<int64_t> *strides) {
  if (this->ellipsis_mask().getZExtValue() ||
      this->new_axis_mask().getZExtValue() ||
      this->shrink_axis_mask().getZExtValue())
    return false;  // TODO(antiagainst): support these masks

  DenseIntElementsAttr shape_attr;
  DenseIntElementsAttr begin_indices_attr, end_indices_attr, strides_attr;
  if (!matchPattern(this->shape(), m_Constant(&shape_attr)) ||
      !matchPattern(this->begin(), m_Constant(&begin_indices_attr)) ||
      !matchPattern(this->end(), m_Constant(&end_indices_attr)) ||
      !matchPattern(this->strides(), m_Constant(&strides_attr)))
    return false;

  int rank = std::distance(shape_attr.begin(), shape_attr.end());

  shape->clear();
  shape->reserve(rank);
  begin_indices->clear();
  begin_indices->reserve(rank);
  end_indices->clear();
  end_indices->reserve(rank);
  strides->clear();
  strides->reserve(rank);

  for (const APInt &dim : shape_attr) shape->push_back(dim.getSExtValue());
  for (const APInt &index : begin_indices_attr)
    begin_indices->push_back(index.getSExtValue());
  for (const APInt &index : end_indices_attr)
    end_indices->push_back(index.getSExtValue());
  for (const APInt &stride : strides_attr)
    strides->push_back(stride.getSExtValue());

  CalculateSlicedShapeAndBoundRanges(*shape, this->begin_mask().getZExtValue(),
                                     this->end_mask().getZExtValue(),
                                     *begin_indices, *end_indices, *strides);
  return true;
}

//===----------------------------------------------------------------------===//
// TensorListReserveOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(TensorListReserveOp op) {
  if (!IsOfRankOrUnranked(op.element_shape(), 0) &&
      !IsOfRankOrUnranked(op.element_shape(), 1)) {
    return op.emitOpError("requires element_shape operand to be 0D/1D tensor");
  }

  if (!IsOfRankOrUnranked(op.num_elements(), 0)) {
    return op.emitOpError("requires num_elements operand to be 0D tensor");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// TensorListStackOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(TensorListStackOp op) {
  if (!IsOfRankOrUnranked(op.element_shape(), 0) &&
      !IsOfRankOrUnranked(op.element_shape(), 1)) {
    return op.emitOpError("requires element_shape operand to be 0D/1D tensor");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// TensorScatterUpdateOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(TensorScatterUpdateOp op) {
  if (!HasRankAtLeast(op.tensor(), 1))
    return op.emitOpError(
        "requires tensor operand to have at least 1 dimension");
  if (!HasRankAtLeast(op.indices(), 1))
    return op.emitOpError(
        "requires indices operand to have at least 1 dimension");
  if (!HasRankAtLeast(op.updates(), 1))
    return op.emitOpError(
        "requires updates operand to have at least 1 dimension");

  auto tensor_ty = op.tensor()->getType().dyn_cast<RankedTensorType>();
  auto indices_ty = op.indices()->getType().dyn_cast<RankedTensorType>();
  if (!tensor_ty || !indices_ty) return success();

  int64_t num_index_dims = indices_ty.getShape().back();
  if (ShapedType::isDynamic(num_index_dims)) return success();

  if (num_index_dims > tensor_ty.getRank())
    return op.emitOpError(
        "requires tensor operand with rank greater than or equal to the "
        "indices operand's last dimensions");
  return success();
}

//===----------------------------------------------------------------------===//
// TopKV2Op
//===----------------------------------------------------------------------===//

static LogicalResult Verify(TopKV2Op op) {
  if (!HasRankAtLeast(op.input(), 1))
    return op.emitOpError(
        "requires input operand to have at least 1 dimension");

  if (!IsOfRankOrUnranked(op.k(), 0))
    return op.emitOpError("requires k operand to be 0D tensor");

  return success();
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(TransposeOp op) {
  // TODO(hinsu): Verify using a custom verifier that,
  // * Transpose permutation is 1-D of size equal to the rank of the first
  //   input, if the shapes are partially known. Requires use of a more
  //   restrictive type than TF_Tensor.
  // * Result shape dimensions are possible based on the input shape.
  return success();
}

// TODO(jpienaar): perm could be optional too.
void TransposeOp::build(Builder *builder, OperationState &result, Value x,
                        Value perm) {
  auto x_type = x->getType().cast<TensorType>();
  // If value is unranked, then so is results.
  if (!x_type.hasRank())
    return TransposeOp::build(builder, result,
                              UnrankedTensorType::get(x_type.getElementType()),
                              x, perm);

  // TODO(jpienaar): Handle unknown perm case.

  // TODO(jpienaar): Extract utility function.
  auto etype = x_type.cast<ShapedType>().getElementType();
  DenseIntElementsAttr attr_shape;
  if (matchPattern(perm, m_Constant(&attr_shape))) {
    llvm::SmallVector<int64_t, 4> const_shape;
    if (attr_shape.isSplat()) {
      const_shape.assign(
          attr_shape.getNumElements(),
          x_type.getDimSize((*attr_shape.begin()).getSExtValue()));
    } else {
      const_shape.reserve(attr_shape.getNumElements());
      for (auto dim : attr_shape)
        const_shape.push_back(x_type.getDimSize(dim.getSExtValue()));
    }
    return TransposeOp::build(
        builder, result, RankedTensorType::get(const_shape, etype), x, perm);
  }
  return TransposeOp::build(builder, result, UnrankedTensorType::get(etype), x,
                            perm);
}

OpFoldResult TransposeOp::fold(ArrayRef<Attribute> operands) {
  auto const_perm = dyn_cast_or_null<TF::ConstOp>(perm()->getDefiningOp());

  if (!const_perm) {
    return {};
  }

  auto const_value = const_perm.value();

  const auto &elements = const_value.getValues<APInt>();
  for (auto it : llvm::enumerate(elements)) {
    if (it.index() != it.value()) {
      return {};
    }
  }

  return x();
}

//===----------------------------------------------------------------------===//
// TruncateDivOp
//===----------------------------------------------------------------------===//

void TruncateDivOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<TruncateDivWithSqrtDivisor>(context);
}

//===----------------------------------------------------------------------===//
// UnpackOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(UnpackOp op) {
  auto value_type = op.value()->getType().dyn_cast<RankedTensorType>();
  if (!value_type) return success();

  int64_t value_rank = value_type.getRank();
  int64_t axis = op.axis().getSExtValue();
  if (axis < -value_rank || axis >= value_rank)
    return op.emitOpError("axis attribute must be in the range of [-")
           << value_rank << ", " << value_rank << ')';

  axis = GetDimForAxis(axis, value_rank);
  int64_t dim_size = value_type.getDimSize(axis);
  if (ShapedType::isDynamic(dim_size)) return success();

  if (dim_size != op.getNumResults())
    return op.emitOpError("result count must be equal to ") << dim_size;

  return success();
}

//===----------------------------------------------------------------------===//
// Unsorted segment reduction ops
//===----------------------------------------------------------------------===//

template <class Op>
static LogicalResult VerifyUnsortedSegmentReduction(Op op) {
  if (!HasRankAtMost(op.num_segments(), 0))
    return op.emitOpError("number of segments should be a 0-D tensor");

  auto data_type = op.data()->getType().template dyn_cast<RankedTensorType>();
  auto segment_ids_type =
      op.segment_ids()->getType().template dyn_cast<RankedTensorType>();
  if (data_type && segment_ids_type) {
    if (data_type.getRank() < segment_ids_type.getRank())
      return op.emitOpError(
          "requires segment ids rank to be less than or equal to data's rank");

    int index = 0;
    for (auto shape_pair :
         llvm::zip_first(segment_ids_type.getShape(), data_type.getShape())) {
      int64_t segment_id_dim = std::get<0>(shape_pair);
      int64_t data_dim = std::get<1>(shape_pair);
      if (!ShapedType::isDynamic(segment_id_dim) &&
          !ShapedType::isDynamic(data_dim) && segment_id_dim != data_dim)
        return op.emitOpError(
                   "requires segment ids shape to be a prefix of data shape, "
                   "but dimension #")
               << index << " differs: " << segment_id_dim << " vs. "
               << data_dim;
      ++index;
    }
  }

  DenseIntElementsAttr num_segments_attr;
  if (matchPattern(op.num_segments(), m_Constant(&num_segments_attr))) {
    int64_t num_segments = (*num_segments_attr.begin()).getSExtValue();
    if (num_segments < 0)
      return op.emitOpError("num of segments cannot be negative");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// VariableShapeOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(VariableShapeOp op) {
  auto resource_operand_type = op.input()
                                   ->getType()
                                   .cast<TensorType>()
                                   .getElementType()
                                   .cast<TF::ResourceType>();
  auto subtypes = resource_operand_type.getSubtypes();
  switch (subtypes.size()) {
    case 1:
      return VerifyShapeOperandAndResult(
          op, resource_operand_type.getSubtypes().front(), op.getType());
    case 0:
      return VerifyShapeOperandAndResult(op, Type(), op.getType());
    default:
      return op.emitOpError(
          "requires resource input type to have at most 1 subtype");
  }
}

//===----------------------------------------------------------------------===//
// WhileOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(WhileOp op) {
  auto module = op.getParentOfType<ModuleOp>();
  auto condFn = module.lookupSymbol<FuncOp>(op.cond());
  auto bodyFn = module.lookupSymbol<FuncOp>(op.body());
  if (!condFn) {
    return op.emitOpError("cond refers to an undefined function : ")
           << op.cond();
  }
  if (!bodyFn) {
    return op.emitOpError("body refers to an undefined function : ")
           << op.body();
  }

  auto condFuncType = condFn.getType();
  auto bodyFuncType = bodyFn.getType();

  // Verify that the cond function has exactly one result.
  if (condFuncType.getNumResults() != 1)
    return op.emitOpError("requires cond function to have exactly one result");

  SmallVector<Type, 4> operands(op.getOperandTypes());
  SmallVector<Type, 4> results(op.getResultTypes());

  // Collect all the type lists for the op so that different pairs of type lists
  // can be compared for the compatibility.
  int numTypeLists = 5;
  std::pair<std::string, ArrayRef<Type>> typeLists[] = {
      {"operand", operands},
      {"body function result", bodyFuncType.getResults()},
      {"result", results},
      {"cond function input", condFuncType.getInputs()},
      {"body function input", bodyFuncType.getInputs()},
  };

  // A pair of type lists should be cast compatible with each other if one is
  // converted to the another for a function call or assignment or there is a
  // common source of inputs for both.  Therefore, the While op requires the
  // following pairs of type lists to be cast compatible for the tensor_cast
  // operation:
  //
  // * Operands and cond inputs to call the cond function before the
  //   first iteration.
  // * Operands and body inputs to call the body function for the first
  //   iteration if the cond functions returns True or equivalent result.
  // * Operands and results to assign cond function arguments to op results if
  //   the cond function returns False or equivalent result.
  // * All three pairs using cond inputs, body inputs and results as operand is
  //   a common source for all three.
  // * Body result and cond inputs to call the cond function for the subsequent
  //   iterations. Similarly, Body result should be compatible with body inputs
  //   and op results.
  //
  // Note that the operands and body results need not be compatible as they are
  // never converted from one to the another nor there is a common source
  // tensors.  Compatibility requirement is not transitive.

  for (int i = 0; i < numTypeLists; ++i) {
    // Skip the first pair as the While op operands and body function results
    // does not need to be compatible with each other.
    for (int j = std::max(2, i + 1); j < numTypeLists; ++j) {
      auto &a = typeLists[i];
      auto &b = typeLists[j];

      int aSize = a.second.size();
      if (aSize != b.second.size())
        return op.emitOpError(
            llvm::formatv("requires the number of {0}s to be equal to the "
                          "number of {1}s. Found {2} and {3}, respectively",
                          a.first, b.first, aSize, b.second.size()));

      for (int idx = 0; idx < aSize; ++idx) {
        auto aType = a.second[idx];
        auto bType = b.second[idx];

        if (!AreCastCompatible(aType, bType))
          return op.emitError(llvm::formatv(
              "{0} type {1} is incompatible with {2} type {3} at index {4}",
              a.first, aType, b.first, bType, idx));
      }
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// XdivyOp
//===----------------------------------------------------------------------===//

void XdivyOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                          MLIRContext *context) {
  results.insert<XdivyWithSqrtDivisor>(context);
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.cc.inc"

//===----------------------------------------------------------------------===//
// TF Dialect Interfaces
//===----------------------------------------------------------------------===//

namespace {
struct TFInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  //===--------------------------------------------------------------------===//
  // Analysis Hooks
  //===--------------------------------------------------------------------===//

  // Defines the legality of inlining TF operations.
  bool isLegalToInline(Operation *, Region *,
                       BlockAndValueMapping &) const final {
    // TODO(riverriddle) For now, enable inlining all operations. This isn't
    // correct in the face of operations that cannot be duplicated, but this
    // requires more intricate side-effect modeling.
    return true;
  }

  //===--------------------------------------------------------------------===//
  // Transformation Hooks
  //===--------------------------------------------------------------------===//

  // Attempts to materialize a conversion for a type mismatch between a call
  // from this dialect, and a callable region. This method should generate an
  // operation that takes 'input' as the only operand, and produces a single
  // result of 'resultType'. If a conversion can not be generated, nullptr
  // should be returned.
  Operation *materializeCallConversion(OpBuilder &builder, Value input,
                                       Type result_type,
                                       Location conversion_loc) const final {
    if (!result_type.isa<TensorType>() || !input->getType().isa<TensorType>())
      return nullptr;
    return builder.create<TF::CastOp>(conversion_loc, result_type, input,
                                      /*truncate=*/builder.getBoolAttr(false));
  }
};
}  // end anonymous namespace

//===----------------------------------------------------------------------===//
// TF Dialect
//===----------------------------------------------------------------------===//

TensorFlowDialect::TensorFlowDialect(MLIRContext *context)
    : Dialect(/*name=*/"tf", context) {
  addOperations<
#define GET_OP_LIST
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.cc.inc"
      >();
  addTypes<
#define HANDLE_TF_TYPE(tftype, enumerant, name) tftype##Type,
#define HANDLE_LAST_TF_TYPE(tftype, enumerant, name) tftype##Type
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.def"
      >();
  addInterfaces<TFInlinerInterface>();

  // Support unknown operations because not all TensorFlow operations are
  // registered.
  allowUnknownOperations();
}

// Parses a type registered to this dialect.
Type TensorFlowDialect::parseType(DialectAsmParser &parser) const {
  StringRef data;
  if (parser.parseKeyword(&data)) return Type();

  Location loc = parser.getEncodedSourceLoc(parser.getNameLoc());
  auto typeKind = llvm::StringSwitch<unsigned>(data)
#define HANDLE_TF_TYPE(tftype, enumerant, name) \
  .Case(name, TensorFlowTypes::enumerant)
// Custom TensorFlow types are handled separately at the end as they do partial
// match.
#define HANDLE_CUSTOM_TF_TYPE(tftype, enumerant, name)
// NOLINTNEXTLINE
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.def"
                      .StartsWith("resource", TensorFlowTypes::RESOURCE)
                      .StartsWith("variant", TensorFlowTypes::VARIANT)
                      .Default(0);
  switch (typeKind) {
    default:
      return (emitError(loc, "unknown TensorFlow type: " + data), nullptr);

#define HANDLE_TF_TYPE(tftype, enumerant, name) \
  case TensorFlowTypes::enumerant:              \
    return tftype##Type::get(getContext());
#define HANDLE_CUSTOM_TF_TYPE(tftype, enumerant, name)
// NOLINTNEXTLINE
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.def"
    case TensorFlowTypes::RESOURCE:
      return ParseResourceType(parser, loc);
    case TensorFlowTypes::VARIANT:
      return ParseVariantType(parser, loc);
  }
}

// Prints a type registered to this dialect.
void TensorFlowDialect::printType(Type ty, DialectAsmPrinter &os) const {
  assert(ty.isa<TensorFlowType>());
  switch (ty.getKind()) {
    default:
      llvm_unreachable("unexpected tensorflow type kind");
#define HANDLE_TF_TYPE(tftype, enumerant, name) \
  case TensorFlowTypes::enumerant:              \
    os << name;                                 \
    break;
#define HANDLE_CUSTOM_TF_TYPE(tftype, enumerant, name) \
  case TensorFlowTypes::enumerant:                     \
    Print##tftype##Type(ty.cast<tftype##Type>(), os);  \
    break;
// NOLINTNEXTLINE
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.def"
  }
}

namespace {
template <typename TypeWithSubtype>
Type ParseTypeWithSubtype(MLIRContext *context, DialectAsmParser &parser,
                          Location loc) {
  // Default type without inferred subtypes.
  if (failed(parser.parseOptionalLess())) return TypeWithSubtype::get(context);

  // Most types with subtypes have only one subtype.
  SmallVector<TensorType, 1> subtypes;
  do {
    TensorType tensor_ty;
    if (parser.parseType(tensor_ty)) return Type();
    subtypes.push_back(tensor_ty);
  } while (succeeded(parser.parseOptionalComma()));

  if (parser.parseGreater()) return Type();
  return TypeWithSubtype::getChecked(subtypes, context, loc);
}

template <typename TypeWithSubtype>
void PrintTypeWithSubtype(StringRef type, TypeWithSubtype ty,
                          DialectAsmPrinter &os) {
  os << type;
  ArrayRef<TensorType> subtypes = ty.getSubtypes();
  if (subtypes.empty()) return;

  os << "<";
  interleaveComma(subtypes, os);
  os << ">";
}
}  // anonymous namespace

Type TensorFlowDialect::ParseResourceType(DialectAsmParser &parser,
                                          Location loc) const {
  return ParseTypeWithSubtype<ResourceType>(getContext(), parser, loc);
}

void TensorFlowDialect::PrintResourceType(ResourceType ty,
                                          DialectAsmPrinter &os) const {
  return PrintTypeWithSubtype("resource", ty, os);
}

Type TensorFlowDialect::ParseVariantType(DialectAsmParser &parser,
                                         Location loc) const {
  return ParseTypeWithSubtype<VariantType>(getContext(), parser, loc);
}

void TensorFlowDialect::PrintVariantType(VariantType ty,
                                         DialectAsmPrinter &os) const {
  return PrintTypeWithSubtype("variant", ty, os);
}

Operation *TensorFlowDialect::materializeConstant(OpBuilder &builder,
                                                  Attribute value, Type type,
                                                  Location loc) {
  return builder.create<ConstOp>(loc, type, value);
}

}  // namespace TF
}  // namespace mlir
