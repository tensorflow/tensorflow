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

#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

#include <cstdint>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/StandardOps/Ops.h"  // TF:local_config_mlir
#include "mlir/IR/Attributes.h"  // TF:local_config_mlir
#include "mlir/IR/Builders.h"  // TF:local_config_mlir
#include "mlir/IR/Matchers.h"  // TF:local_config_mlir
#include "mlir/IR/OpImplementation.h"  // TF:local_config_mlir
#include "mlir/IR/PatternMatch.h"  // TF:local_config_mlir
#include "mlir/IR/StandardTypes.h"  // TF:local_config_mlir
#include "mlir/IR/TypeUtilities.h"  // TF:local_config_mlir
#include "mlir/Support/LLVM.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"

namespace mlir {
namespace TFL {

//===----------------------------------------------------------------------===//
// TensorFlowLiteDialect
//===----------------------------------------------------------------------===//

TensorFlowLiteDialect::TensorFlowLiteDialect(mlir::MLIRContext *context)
    : Dialect(/*name=*/"tfl", context) {
  addOperations<
#define GET_OP_LIST
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.cc.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Common support logic
//===----------------------------------------------------------------------===//

namespace {

// Returns true if the dimensions in `a` is a suffix of the ones in `b`.
// For example, dimensions {2}, {1, 2}, and {3, 1, 2} are all suffixes to
// {5, 4, 3, 1, 2}, while {1}, {5, 4}, and {1, 3, 2} are all not.
inline bool IsTrailingDimensions(ArrayRef<int64_t> a, ArrayRef<int64_t> b) {
  if (a.size() > b.size()) return false;

  return std::equal(a.rbegin(), a.rend(), b.rbegin());
}

// Returns true if it is a shaped type of f32 elements.
inline bool IsF32ShapedType(Type t) {
  if (auto shaped_type = t.dyn_cast_or_null<ShapedType>()) {
    return shaped_type.getElementType().isF32();
  }
  return false;
}

// Performs const folding `calculate` with broadcast behavior on the two
// attributes `operand1` and `operand2` and returns the result if possible.
// The two operands are expected to both be scalar values.
template <class AttrElementT,
          class ElementValueT = typename AttrElementT::ValueType,
          class CalculationT =
              llvm::function_ref<ElementValueT(ElementValueT, ElementValueT)>>
Attribute ConstFoldBinaryOpScalarScalar(Type result_type, Attribute operand1,
                                        Attribute operand2,
                                        const CalculationT &calculate) {
  auto lhs = operand1.cast<AttrElementT>();
  auto rhs = operand2.cast<AttrElementT>();

  assert(lhs.getType() == result_type && rhs.getType() == result_type &&
         "values of incompatible types should be caught by op verification");

  // TODO: Need to handle overflow/underflow cases.
  return AttrElementT::get(result_type,
                           calculate(lhs.getValue(), rhs.getValue()));
}

/// Performs const folding `calculate` with broadcast behavior on the two
/// attributes `operand1` and `operand2` and returns the result if possible.
/// This function assumes the both operands are verified to have value
/// attributes of broadcastable types.
template <class AttrElementT,
          class ElementValueT = typename AttrElementT::ValueType,
          class CalculationT =
              llvm::function_ref<ElementValueT(ElementValueT, ElementValueT)>>
Attribute ConstFoldBinaryOpDenseDense(Type result_type, DenseElementsAttr lhs,
                                      DenseElementsAttr rhs,
                                      const CalculationT &calculate) {
  auto type = result_type.cast<ShapedType>();

  if (lhs.getType() != rhs.getType()) {
    // We only support the case that one of the operand's dimensions are
    // a perfect suffix of the other.
    // TODO: support the general broadcast behavior.
    auto lhs_shape = lhs.getType().getShape();
    auto rhs_shape = rhs.getType().getShape();
    if (IsTrailingDimensions(lhs_shape, rhs_shape)) {
      if (!type.hasStaticShape()) type = rhs.getType();
    } else if (IsTrailingDimensions(rhs_shape, lhs_shape)) {
      if (!type.hasStaticShape()) type = lhs.getType();
    } else {
      return {};
    }
  } else if (!type.hasStaticShape()) {
    type = lhs.getType();
  }

  const bool rhs_is_splat = rhs.isSplat();
  const bool lhs_is_splat = lhs.isSplat();

  // If both of them are splat, compute and return.
  if (lhs_is_splat && rhs_is_splat) {
    auto element_result = AttrElementT::get(
        type.getElementType(), calculate(lhs.getSplatValue<ElementValueT>(),
                                         rhs.getSplatValue<ElementValueT>()));
    if (!element_result) return {};

    return DenseElementsAttr::get(type, element_result);
  }

  auto lhs_num_elements = lhs.getType().getNumElements();
  auto rhs_num_elements = rhs.getType().getNumElements();
  auto num_elements = std::max(lhs_num_elements, rhs_num_elements);

  // We assume the arguments have broadcast-compatible types. Make sure again.
  assert(std::max(lhs_num_elements, rhs_num_elements) == num_elements);
  assert(num_elements % std::min(lhs_num_elements, rhs_num_elements) == 0);

  SmallVector<ElementValueT, 16> lhs_old_values;
  SmallVector<ElementValueT, 16> rhs_old_values;
  if (lhs_is_splat)
    lhs_old_values.push_back(lhs.getSplatValue<ElementValueT>());
  else
    lhs_old_values = llvm::to_vector<16>(lhs.getValues<ElementValueT>());
  if (rhs_is_splat)
    rhs_old_values.push_back(rhs.getSplatValue<ElementValueT>());
  else
    rhs_old_values = llvm::to_vector<16>(rhs.getValues<ElementValueT>());

  SmallVector<ElementValueT, 16> new_values;
  new_values.reserve(num_elements);

  // Add each pair of the corresponding values in the dense elements
  // attributes.
  for (int i = 0; i < num_elements; ++i) {
    // We only support a degenerated case here: the dimensions in one operand's
    // shape is a perfect suffix to the other operand. Then conceptually it's
    // similar to broadcasting a scalar to a 1-D vector.
    // TODO: support the general broadcast behavior.
    // We are tiling the operand with less elements an integral times to match
    // the operand with more elements. We don't care which operand has less
    // elements here because we are iterating its elements in circles, which can
    // be achieved using the result index modulo the element count. For the
    // operand with more elements, since the result has the same number of
    // elements, we are only going over its elements once. The modulo operation
    // also works for that.
    int lhs_index = lhs_is_splat ? 0 : (i % lhs_num_elements);
    int rhs_index = rhs_is_splat ? 0 : (i % rhs_num_elements);

    new_values.push_back(
        calculate(lhs_old_values[lhs_index], rhs_old_values[rhs_index]));
  }

  return DenseElementsAttr::get(type, new_values);
}

/// Performs const folding `calculate` with broadcast behavior on the two
/// attributes `operand1` and `operand2` and returns the result if possible.
/// This function assumes the two operands are verified to have value
/// attributes of broadcastable types.
template <class AttrElementT,
          class ElementValueT = typename AttrElementT::ValueType,
          class CalculationT =
              llvm::function_ref<ElementValueT(ElementValueT, ElementValueT)>>
Attribute ConstFoldBinaryOp(Type result_type, Attribute operand1,
                            Attribute operand2, const CalculationT &calculate,
                            bool is_commutative) {
  if (operand1.dyn_cast_or_null<AttrElementT>()) {
    // Scalar op scalar case
    if (operand2.dyn_cast_or_null<AttrElementT>())
      return ConstFoldBinaryOpScalarScalar<AttrElementT>(result_type, operand1,
                                                         operand2, calculate);
  } else if (operand1.dyn_cast_or_null<DenseElementsAttr>() &&
             operand2.dyn_cast_or_null<DenseElementsAttr>()) {
    return ConstFoldBinaryOpDenseDense<AttrElementT>(
        result_type, operand1.cast<DenseElementsAttr>(),
        operand2.cast<DenseElementsAttr>(), calculate);
  }

  // TODO: support other attribute kinds

  return {};
}

/// Performs const folding with broadcast behavior on the two attributes in
/// `operands` and returns the result if possible.
/// Depending on the given `resultType`, either `floatCalculate` or
/// `intCalculate` is chosen to conduct the calculate.
Attribute ConstFoldBinaryOp(
    Type result_type, ArrayRef<Attribute> operands,
    llvm::function_ref<APFloat(APFloat, APFloat)> float_calculate,
    llvm::function_ref<APInt(APInt, APInt)> int_calculate,
    bool is_commutative) {
  // Note: All types are wrapped in tensor types in TFlite. E.g., f32 is
  // represented as tensor<f32>. So we are only handling tensor types here.
  auto type = result_type.dyn_cast<ShapedType>();
  if (!type) return {};

  auto elemType = type.getElementType();

  if (elemType.isa<FloatType>())
    return ConstFoldBinaryOp<FloatAttr>(result_type, operands[0], operands[1],
                                        float_calculate, is_commutative);

  if (elemType.isa<IntegerType>())
    return ConstFoldBinaryOp<IntegerAttr>(result_type, operands[0], operands[1],
                                          int_calculate, is_commutative);

  return {};
}

/// Performs const folding a attributes `operand` and returns the result if
/// possible.
/// The function currently asserts that the `result_type` to be a f32 tensor
/// type.
/// TODO: Extend this function to handle integral tensor for ops like
/// "tfl.logical_not".
Attribute ConstFoldUnaryOp(Type result_type, Attribute operand,
                           llvm::function_ref<APFloat(APFloat)> calculate) {
  assert(IsF32ShapedType(result_type));
  auto result_shape_type = result_type.cast<ShapedType>();

  if (auto dense_elements = operand.dyn_cast_or_null<DenseElementsAttr>()) {
    SmallVector<APFloat, 16> new_values;
    const int num_elements = result_shape_type.getNumElements();
    new_values.reserve(num_elements);

    for (APFloat old_value : dense_elements.getValues<APFloat>()) {
      new_values.push_back(calculate(old_value));
    }

    return DenseElementsAttr::get(result_shape_type, new_values);
  }

  return {};
}

void buildComparisonBinOp(Builder *builder, OperationState *result, Value *lhs,
                          Value *rhs) {
  auto result_type =
      OpTrait::util::getBroadcastedType(lhs->getType(), rhs->getType());
  if (!result_type)
    emitError(result->location)
        << "non-broadcastable operands: " << lhs->getType() << " and "
        << rhs->getType();
  result->addOperands({lhs, rhs});
  // Comparison binary ops always return i1 tensor.
  if (auto shaped_type = result_type.dyn_cast<ShapedType>()) {
    auto resultShape = shaped_type.getShape();
    result->types.push_back(
        builder->getTensorType(resultShape, builder->getI1Type()));
  } else {
    result->types.push_back(builder->getTensorType(builder->getI1Type()));
  }
}

void buildFusedBroadcastableBinOp(Builder *builder, OperationState *result,
                                  Value *lhs, Value *rhs,
                                  StringAttr fused_activation_function) {
  auto result_type =
      OpTrait::util::getBroadcastedType(lhs->getType(), rhs->getType());

  if (!result_type)
    emitError(result->location)
        << "non-broadcastable operands: " << lhs->getType() << " and "
        << rhs->getType();

  result->addOperands({lhs, rhs});
  result->addAttribute("fused_activation_function", fused_activation_function);
  result->types.push_back(result_type);
}

}  // end anonymous namespace

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

OpFoldResult AddOp::fold(ArrayRef<Attribute> operands) {
  // Skip fused ops for now.
  if (fused_activation_function() != "NONE") return {};
  return ConstFoldBinaryOp(
      getType(), operands, [](APFloat a, APFloat b) { return a + b; },
      [](APInt a, APInt b) { return a + b; }, getOperation()->isCommutative());
}

//===----------------------------------------------------------------------===//
// ConcatenationOp
//===----------------------------------------------------------------------===//
// TODO(ashwinm): Implement shape inference for Concatenation

//===----------------------------------------------------------------------===//
// GatherOp
//===----------------------------------------------------------------------===//

static void BuildGatherOp(Builder *builder, OperationState *result,
                          Value *params, Value *indices, IntegerAttr axis) {
  auto params_type = params->getType().cast<TensorType>();
  auto indices_type = indices->getType().cast<TensorType>();

  // If params/indices is unranked, then output is unranked.
  if (!params_type.hasRank() || !indices_type.hasRank())
    return TFL::GatherOp::build(
        builder, result, builder->getTensorType(params_type.getElementType()),
        params, indices, axis);

  int64_t params_rank = params_type.getRank();
  int64_t indices_rank = indices_type.getRank();

  // params rank is guaranteed to be at least 1.
  // Produces an output tensor with shape:
  // params.shape[:axis] + indices.shape + params.shape[axis + 1:]
  std::vector<int64_t> shape(params_type.getShape());
  int64_t axis_i = axis.getInt();

  // For neg axis values, we wrap around params, e.g. axis = -1 => params[:-1]
  if (axis_i < 0) {
    axis_i += params_rank;
  }

  // params must be atleast rank axis + 1
  if (params_rank < axis_i + 1) {
    emitError(result->location, "params must be atleast rank axis + 1");
  }

  if (indices_rank == 0) {
    // Scalar indices (output is rank(params) - 1).
    // Erase shape[axis]
    shape.erase(shape.begin() + axis_i);
  } else if (indices_rank == 1) {
    // Vector indices (output is rank(params)).
    // Copy indices.shape into params.shape[axis]
    std::copy(std::begin(indices_type.getShape()),
              std::end(indices_type.getShape()), std::begin(shape) + axis_i);
  } else {
    // Higher rank indices (output is rank(params) + rank(indices) - 1).
    shape.resize(params_rank + indices_rank - 1);
    // Copy params.shape[axis + 1: ] into shape[axis + indices_rank:]
    std::copy(std::begin(params_type.getShape()) + axis_i + 1,
              std::end(params_type.getShape()),
              std::begin(shape) + axis_i + indices_rank);

    // Copy indices.shape into params.shape[axis]
    std::copy(std::begin(indices_type.getShape()),
              std::end(indices_type.getShape()), std::begin(shape) + axis_i);
  }

  TFL::GatherOp::build(
      builder, result,
      builder->getTensorType(shape, params_type.getElementType()), params,
      indices, axis);
}

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

OpFoldResult MulOp::fold(ArrayRef<Attribute> operands) {
  // Skip fused ops for now.
  if (fused_activation_function() != "NONE") return {};
  return ConstFoldBinaryOp(
      getType(), operands, [](APFloat a, APFloat b) { return a * b; },
      [](APInt a, APInt b) { return a * b; }, getOperation()->isCommutative());
}

//===----------------------------------------------------------------------===//
// PackOp
//===----------------------------------------------------------------------===//

// TODO(b/133486129): Implement shape inference for pack

static LogicalResult Verify(PackOp op) {
  // TODO(antiagainst): Implement other checks as in
  // tensorflow/lite/kernels/pack.cc

  if (op.getOperation()->getNumOperands() != op.values_count())
    return op.emitOpError("input count should match 'values_count' attribute");

  return success();
}

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

namespace {
/// This pattern matches and merges a tfl.reshape under the following
/// condition:
/// * The input's defining op is another tfl.reshape.
// TODO(antiagainst): This pattern probably should be moved to the peephole
// category, after we have the infra for peephole passes.
struct RemoveAdjacentReshape : public RewritePattern {
  RemoveAdjacentReshape(MLIRContext *context)
      : RewritePattern(ReshapeOp::getOperationName(), 1, context) {}

  PatternMatchResult match(Operation *op) const override {
    auto thisOp = cast<ReshapeOp>(op);
    auto prevOp = thisOp.getOperand()->getDefiningOp();
    return isa_and_nonnull<ReshapeOp>(prevOp) ? matchSuccess() : matchFailure();
  }

  void rewrite(Operation *op, PatternRewriter &rewriter) const override {
    auto thisOp = cast<ReshapeOp>(op);
    auto prevOp = cast<ReshapeOp>(thisOp.getOperand()->getDefiningOp());

    // Replace
    //   %1 = "tfl.reshape"(%0)
    //   %2 = "tfl.reshape"(%1)
    // With
    //   %2 = "tfl.reshape"(%0)
    rewriter.replaceOpWithNewOp<ReshapeOp>(
        {prevOp.getResult()}, op, thisOp.getType(), prevOp.getOperand());
  }
};

}  // end anonymous namespace

OpFoldResult ReshapeOp::fold(ArrayRef<Attribute> operands) {
  // Remove identity reshape.
  if (getType() == getOperand()->getType()) return getOperand();

  // Constant folding
  assert(operands.size() == 1);
  if (auto dense_elements = operands[0].dyn_cast_or_null<DenseElementsAttr>()) {
    auto result_shape_type = getType().cast<ShapedType>();
    return dense_elements.reshape(result_shape_type);
  }

  return nullptr;
}

void ReshapeOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                            MLIRContext *context) {
  results.insert<RemoveAdjacentReshape>(context);
}

//===----------------------------------------------------------------------===//
// SubOp
//===----------------------------------------------------------------------===//

OpFoldResult SubOp::fold(ArrayRef<Attribute> operands) {
  // Skip fused ops for now.
  if (fused_activation_function() != "NONE") return {};
  return ConstFoldBinaryOp(
      getType(), operands, [](APFloat a, APFloat b) { return a - b; },
      [](APInt a, APInt b) { return a - b; }, getOperation()->isCommutative());
}

//===----------------------------------------------------------------------===//
// TopKOp
//===----------------------------------------------------------------------===//

static void BuildTopKOp(Builder *builder, OperationState *result, Value *input,
                        Value *k) {
  // Output size is only known if k is constant value. A negative dimension is
  // considered dynamic so use -1 here if k is not a constant value.
  int const_k = -1;
  ElementsAttr cst;
  if (matchPattern(k, m_Constant(&cst)))
    // These casts should all be valid due to how Tensor constants are stored.
    // TODO(jpienaar): This should use a helper function.
    const_k = cst.getValue<IntegerAttr>({}).getValue().getSExtValue();

  auto val_type = input->getType().cast<TensorType>();
  // If value is unranked, then so is results.
  if (!val_type.hasRank())
    return TFL::TopKV2Op::build(
        builder, result, builder->getTensorType(val_type.getElementType()),
        builder->getTensorType(builder->getIntegerType(32)), input, k);

  // Resultant shape is value.shape[:-1] + [k]
  std::vector<int64_t> shape(val_type.getShape());
  shape[shape.size() - 1] = const_k;
  TFL::TopKV2Op::build(
      builder, result, builder->getTensorType(shape, val_type.getElementType()),
      builder->getTensorType(shape, builder->getIntegerType(32)), input, k);
}

//===----------------------------------------------------------------------===//
// FakeQuantOp
//===----------------------------------------------------------------------===//

// Return true if the op has non-empty "minmax" attribute.
static inline bool HasValidMinMaxAttribute(Operation *op) {
  auto minmax = op->getAttrOfType<ArrayAttr>("minmax");
  return minmax && minmax.getValue().size() == 2;
}

namespace {

/// This pattern matches and remove a tfl.fake_quant if all the users of this op
/// and itself have "minmax" attribute set.
struct DropFakeQuant : public RewritePattern {
  explicit DropFakeQuant(MLIRContext *context)
      : RewritePattern(FakeQuantOp::getOperationName(), 1, context) {}

  PatternMatchResult match(Operation *op) const override {
    // We only match the op with valid "minmax" attribute.
    if (!HasValidMinMaxAttribute(op)) return matchFailure();

    // If all the users of this op have valid "minmax" attributes, it is matched
    // and can be removed.
    auto fakeQuantOp = cast<FakeQuantOp>(op);
    for (auto *operand : fakeQuantOp.getResult()->getUsers())
      if (!HasValidMinMaxAttribute(operand)) return matchFailure();

    return matchSuccess();
  }

  void rewrite(Operation *op, PatternRewriter &rewriter) const override {
    // Replace the matched FakeQuantOp by its primiary operand.
    rewriter.replaceOp(op, op->getOperand(0));
  }
};
}  // end anonymous namespace

void FakeQuantOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                              MLIRContext *context) {
  results.insert<DropFakeQuant>(context);
}

//===----------------------------------------------------------------------===//
// UnpackOp
//===----------------------------------------------------------------------===//

// TODO(b/133486129): Implement shape inference for unpack

static LogicalResult Verify(UnpackOp op) {
  // TODO(antiagainst): Implement other checks as in
  // tensorflow/lite/kernels/unpack.cc

  if (op.getOperation()->getNumResults() != op.num())
    return op.emitOpError("output count should match 'num' attribute");

  return success();
}

//===----------------------------------------------------------------------===//
// MeanOp
//===----------------------------------------------------------------------===//

// TODO(b/133854225): Implement shape inference to Mean

//===----------------------------------------------------------------------===//
// LSTMOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(LSTMOp op) {
  auto operands = op.GetStatefulOperands();
  if (operands.size() == 2 && operands[0] == 18 && operands[1] == 19) {
    return success();
  }
  return op.emitError("LSTMOp expected to have two stateful operands");
}

//===----------------------------------------------------------------------===//
// UnidirectionalSequenceLSTMOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(UnidirectionalSequenceLSTMOp op) {
  auto operands = op.GetStatefulOperands();
  if (operands.size() == 2 && operands[0] == 18 && operands[1] == 19) {
    return success();
  }
  return op.emitError(
      "UnidirectionalSequenceLSTMOp expected to have two stateful operands");
}

//===----------------------------------------------------------------------===//
// UnidirectionalSequenceRNNOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(UnidirectionalSequenceRNNOp op) {
  auto operands = op.GetStatefulOperands();
  if (operands.size() == 1 && operands[0] == 4) {
    return success();
  }
  return op.emitError(
      "UnidirectionalSequenceRNNOp expected to have one stateful operand");
}

//===----------------------------------------------------------------------===//
// SvdfOp
//===----------------------------------------------------------------------===//

static LogicalResult Verify(SVDFOp op) {
  auto operands = op.GetStatefulOperands();
  if (operands.size() == 1 && operands[0] == 4) {
    return success();
  }
  return op.emitError("SvdfOp expected to have one stateful operand");
}

//===----------------------------------------------------------------------===//
// AbsOp
//===----------------------------------------------------------------------===//

OpFoldResult AbsOp::fold(ArrayRef<Attribute> operands) {
  Type result_type = getType();
  // Only constant fold for tensor of f32 is implemented.
  if (!IsF32ShapedType(result_type)) return nullptr;

  auto compute = [](APFloat value) -> APFloat { return llvm::abs(value); };
  return ConstFoldUnaryOp(result_type, operands[0], compute);
}

//===----------------------------------------------------------------------===//
// SinOp
//===----------------------------------------------------------------------===//

OpFoldResult SinOp::fold(ArrayRef<Attribute> operands) {
  Type result_type = getType();
  // Only constant fold for tensor of f32 is implemented.
  if (!IsF32ShapedType(result_type)) return nullptr;

  auto compute = [](APFloat value) -> APFloat {
    float f = value.convertToFloat();
    float result = std::sin(f);
    return APFloat(result);
  };
  return ConstFoldUnaryOp(result_type, operands[0], compute);
}

//===----------------------------------------------------------------------===//
// CosOp
//===----------------------------------------------------------------------===//

OpFoldResult CosOp::fold(ArrayRef<Attribute> operands) {
  Type result_type = getType();
  // Only constant fold for tensor of f32 is implemented.
  if (!IsF32ShapedType(result_type)) return nullptr;

  auto compute = [](APFloat value) -> APFloat {
    float f = value.convertToFloat();
    float result = std::cos(f);
    return APFloat(result);
  };
  return ConstFoldUnaryOp(result_type, operands[0], compute);
}

//===----------------------------------------------------------------------===//
// LogOp
//===----------------------------------------------------------------------===//

OpFoldResult LogOp::fold(ArrayRef<Attribute> operands) {
  Type result_type = getType();
  // Only constant fold for tensor of f32 is implemented.
  if (!IsF32ShapedType(result_type)) return nullptr;

  auto compute = [](APFloat value) -> APFloat {
    float f = value.convertToFloat();
    float result = std::log(f);
    return APFloat(result);
  };
  return ConstFoldUnaryOp(result_type, operands[0], compute);
}

//===----------------------------------------------------------------------===//
// SqrtOp
//===----------------------------------------------------------------------===//

OpFoldResult SqrtOp::fold(ArrayRef<Attribute> operands) {
  Type result_type = getType();
  // Only constant fold for tensor of f32 is implemented.
  if (!IsF32ShapedType(result_type)) return nullptr;

  auto compute = [](APFloat value) -> APFloat {
    float f = value.convertToFloat();
    float result = std::sqrt(f);
    return APFloat(result);
  };
  return ConstFoldUnaryOp(result_type, operands[0], compute);
}

//===----------------------------------------------------------------------===//
// RsqrtOp
//===----------------------------------------------------------------------===//

OpFoldResult RsqrtOp::fold(ArrayRef<Attribute> operands) {
  Type result_type = getType();
  // Only constant fold for tensor of f32 is implemented.
  if (!IsF32ShapedType(result_type)) return nullptr;

  auto compute = [](APFloat value) -> APFloat {
    float f = value.convertToFloat();
    float result = 1.f / std::sqrt(f);
    return APFloat(result);
  };
  return ConstFoldUnaryOp(result_type, operands[0], compute);
}

//===----------------------------------------------------------------------===//
// SquareOp
//===----------------------------------------------------------------------===//

OpFoldResult SquareOp::fold(ArrayRef<Attribute> operands) {
  Type result_type = getType();
  // Only constant fold for tensor of f32 is implemented.
  if (!IsF32ShapedType(result_type)) return nullptr;

  auto compute = [](APFloat value) -> APFloat { return value * value; };
  return ConstFoldUnaryOp(result_type, operands[0], compute);
}

//===----------------------------------------------------------------------===//
// RankOp
//===----------------------------------------------------------------------===//

OpFoldResult RankOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 1);
  auto result_type = getType().cast<ShapedType>();
  if (auto elements_attr = operands[0].dyn_cast_or_null<ElementsAttr>()) {
    auto rank = static_cast<int32_t>(elements_attr.getType().getRank());
    return DenseElementsAttr::get(result_type, {rank});
  }

  // Also fold if `input` has a known rank.
  auto input_type = input()->getType().cast<ShapedType>();
  // Do not fold if rank is zero because the TFLite converter doesn't
  // distinguish between unranked input and scalar input due to b/138865275.
  // TODO(b/138865275): Remove `input_type.getRank() != 0` in the following
  // predicate and fold the op when rank is zero.
  if (input_type.hasRank() && input_type.getRank() != 0) {
    auto rank = static_cast<int32_t>(input_type.getRank());
    return DenseElementsAttr::get(result_type, {rank});
  }

  return nullptr;
}

//===----------------------------------------------------------------------===//
// ConstOp
//===----------------------------------------------------------------------===//

OpFoldResult ConstOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.empty() && "constant has no operands");

  // Return the held attribute value.
  return value();
}

//===----------------------------------------------------------------------===//
// RangeOp
//===----------------------------------------------------------------------===//

namespace {

// Compute the length of a range (1-D) tensor given `start`, `limit`, `delta`.
// Template parameter `FloatOrInt` must be standard C integer or floating-point
// types.
template <typename FloatOrInt>
int GetLengthOfRange(FloatOrInt start, FloatOrInt limit, FloatOrInt delta) {
  // Refer to the implementation in
  // tensorflow/lite/kernels/range.cc.
  return std::is_integral<FloatOrInt>::value
             ? ((std::abs(limit - start) + std::abs(delta) - 1) /
                std::abs(delta))
             : std::ceil(std::abs((limit - start) / delta));
}

// Builds a constant range tensor of `result_elem_type` elements.
// Template parameter `FloatOrIntAtrr` must be mlir::IntegerAttr or
// mlir::FloatAttr.
template <typename FloatOrIntAtrr>
DenseElementsAttr BuildConstRangeTensor(Type result_elem_type, int num_elements,
                                        FloatOrIntAtrr start_attr,
                                        FloatOrIntAtrr delta_attr) {
  using ValueType = typename FloatOrIntAtrr::ValueType;  // APInt or APFloat
  ValueType start = start_attr.getValue();
  ValueType delta = delta_attr.getValue();

  SmallVector<ValueType, 16> new_values;
  new_values.reserve(num_elements);
  ValueType new_value = start;
  for (int i = 0; i < num_elements; ++i) {
    new_values.push_back(new_value);
    new_value = new_value + delta;
  }
  // Result is always a 1-D tensor.
  auto new_result_type =
      RankedTensorType::get({num_elements}, result_elem_type);
  return DenseElementsAttr::get(new_result_type, new_values);
}
}  // namespace

OpFoldResult RangeOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 3);
  auto start_tensor = operands[0].dyn_cast_or_null<ElementsAttr>();
  auto limit_tensor = operands[1].dyn_cast_or_null<ElementsAttr>();
  auto delta_tensor = operands[2].dyn_cast_or_null<ElementsAttr>();
  if (start_tensor && limit_tensor && delta_tensor) {
    // Operands should all be scalars
    assert(start_tensor.getType().getRank() == 0 &&
           limit_tensor.getType().getRank() == 0 &&
           delta_tensor.getType().getRank() == 0);
    Type elem_type = getType().cast<ShapedType>().getElementType();
    if (elem_type.isa<IntegerType>()) {
      auto start_attr = start_tensor.getValue<IntegerAttr>({});
      auto limit_attr = limit_tensor.getValue<IntegerAttr>({});
      auto delta_attr = delta_tensor.getValue<IntegerAttr>({});
      const int num_elements = GetLengthOfRange(
          start_attr.getInt(), limit_attr.getInt(), delta_attr.getInt());
      return BuildConstRangeTensor(elem_type, num_elements, start_attr,
                                   delta_attr);
    } else if (elem_type.isa<FloatType>()) {
      auto start_attr = start_tensor.getValue<FloatAttr>({});
      auto limit_attr = limit_tensor.getValue<FloatAttr>({});
      auto delta_attr = delta_tensor.getValue<FloatAttr>({});
      const int num_elements = GetLengthOfRange(start_attr.getValueAsDouble(),
                                                limit_attr.getValueAsDouble(),
                                                delta_attr.getValueAsDouble());
      return BuildConstRangeTensor(elem_type, num_elements, start_attr,
                                   delta_attr);
    }
  }

  return nullptr;
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

namespace {

// Computes the permutation of a constant `input_tensor` according to `perm`.
// The function recursively traverses the dimensions of the output tensor in
// a row-major order and writes the value in the output tensor into
// `new_values`.
void ComputePermutation(ElementsAttr input_tensor, ArrayRef<int32_t> perm,
                        ArrayRef<int64_t> output_shape, int num_dimensions,
                        int output_axis, std::vector<uint64_t> *input_indices,
                        std::vector<Attribute> *new_values) {
  // Refer to the implementation of `Transpose` function in
  // tensorflow/lite/kernels/internal/reference/reference_ops.h
  assert(output_axis < num_dimensions);
  const int input_axis = perm[output_axis];
  for (int i = 0; i < output_shape[output_axis]; ++i) {
    // Update the input indices on `input_axis`.
    input_indices->at(input_axis) = i;
    // Write the value from `input_tensor` if it is the last axis or
    // recurse into the next axis.
    const bool is_last_axis = output_axis == num_dimensions - 1;
    if (is_last_axis) {
      new_values->push_back(input_tensor.getValue(*input_indices));
    } else {
      ComputePermutation(input_tensor, perm, output_shape, num_dimensions,
                         output_axis + 1, input_indices, new_values);
    }
  }
}

}  // namespace

OpFoldResult TransposeOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 2);
  auto input_tensor = operands[0].dyn_cast_or_null<ElementsAttr>();
  auto perm_tensor = operands[1].dyn_cast_or_null<ElementsAttr>();
  if (!input_tensor || !perm_tensor) return nullptr;

  // Do not try to fold elements attr of a quant type because
  // DenseElementsAttr does not support it.
  if (!getType().cast<ShapedType>().getElementType().isIntOrFloat())
    return nullptr;

  assert(perm_tensor.getType().getRank() == 1);
  const int num_dimensions = input_tensor.getType().getRank();
  assert(perm_tensor.getType().getNumElements() == num_dimensions);

  ArrayRef<int64_t> input_shape = input_tensor.getType().getShape();
  auto output_type = getType().cast<ShapedType>();

  SmallVector<int32_t, 4> perm;
  SmallVector<int64_t, 4> output_shape;
  for (int i = 0; i < num_dimensions; ++i) {
    perm.push_back(
        perm_tensor.getValue<IntegerAttr>({static_cast<uint64_t>(i)}).getInt());
    output_shape.push_back(input_shape[perm[i]]);

    // Check that the derived output shape matches the static shape.
    assert(!output_type.hasStaticShape() ||
           output_type.getShape()[i] == output_shape[i]);
  }

  std::vector<Attribute> new_values;
  new_values.reserve(input_tensor.getType().getNumElements());
  std::vector<uint64_t> input_indices(num_dimensions);
  ComputePermutation(input_tensor, perm, output_shape, num_dimensions,
                     /*output_axis=*/0, &input_indices, &new_values);
  auto result_type =
      RankedTensorType::get(output_shape, output_type.getElementType());
  return DenseElementsAttr::get(result_type, new_values);
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.cc.inc"

Operation *TensorFlowLiteDialect::materializeConstant(OpBuilder &builder,
                                                      Attribute value,
                                                      Type type, Location loc) {
  // If this is an opaque elements attribute or the result type doesn't match
  // the attribute type, then generate a tfl.pseudo_const.
  if (value.isa<OpaqueElementsAttr>() ||
      (value.isa<ElementsAttr>() && value.getType() != type))
    return builder.create<ConstOp>(loc, type, value.cast<ElementsAttr>());
  return nullptr;
}

}  // namespace TFL
}  // namespace mlir
