/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/reduce.h"

#include <cstdint>
#include <optional>

#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/hlo_matchers.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace mlir {
namespace odml {

// Pattern matches the following reduction function for ArgMax/ArgMin coming
// from PyTorch
// %0 = compare{GT}(%lhs_value, %rhs_value)
// %1 = select(%0, %lhs_value, %rhs_value)
// %2 = compare{EQ}(%lhs_value, %rhs_value)
// %3 = minimum(%lhs_index, %rhs_index)
// %4 = select(%0, %lhs_index, %rhs_index)
// %5 = select(%2, %3, %4)
// return %1, %5
LogicalResult MatchReduceToArgMinMaxType2(mhlo::ReduceOp reduce_op,
                                          bool is_argmax) {
  Block& body = reduce_op.getBody().front();
  if (body.getNumArguments() != 4) return failure();

  mhlo::ReturnOp return_op = dyn_cast<mhlo::ReturnOp>(body.back());
  if (!return_op || return_op.getNumOperands() != 2) return failure();

  mhlo::SelectOp value_select = llvm::dyn_cast_or_null<mhlo::SelectOp>(
      return_op.getOperand(0).getDefiningOp());
  if (!value_select || value_select.getOnTrue() != body.getArgument(0) ||
      value_select.getOnFalse() != body.getArgument(2))
    return failure();

  auto compare_direction_included =
      is_argmax ? mhlo::ComparisonDirection::GE : mhlo::ComparisonDirection::LE;
  mhlo::CompareOp value_gt = llvm::dyn_cast_or_null<mhlo::CompareOp>(
      value_select.getOperand(0).getDefiningOp());
  if (!value_gt ||
      value_gt.getComparisonDirection() != compare_direction_included ||
      value_gt.getLhs() != body.getArgument(0) ||
      value_gt.getRhs() != body.getArgument(2))
    return failure();

  mhlo::SelectOp index_select = llvm::dyn_cast_or_null<mhlo::SelectOp>(
      return_op.getOperand(1).getDefiningOp());
  if (!index_select) return failure();

  mhlo::MinOp index_select_min = llvm::dyn_cast_or_null<mhlo::MinOp>(
      index_select.getOnTrue().getDefiningOp());
  if (!index_select_min || index_select_min.getLhs() != body.getArgument(1) ||
      index_select_min.getRhs() != body.getArgument(3))
    return failure();

  mhlo::SelectOp index_select_select = llvm::dyn_cast_or_null<mhlo::SelectOp>(
      index_select.getOnFalse().getDefiningOp());
  if (!index_select_select ||
      index_select_select.getOnTrue() != body.getArgument(1) ||
      index_select_select.getOnFalse() != body.getArgument(3) ||
      index_select_select.getOperand(0).getDefiningOp() != value_gt)
    return failure();

  mhlo::CompareOp value_eq = llvm::dyn_cast_or_null<mhlo::CompareOp>(
      index_select.getOperand(0).getDefiningOp());
  if (!value_eq ||
      value_eq.getComparisonDirection() != mhlo::ComparisonDirection::EQ ||
      value_eq.getLhs() != body.getArgument(0) ||
      value_eq.getRhs() != body.getArgument(2))
    return failure();

  return success();
}

// Pattern matches the following reduction function for ArgMax/ArgMin:
// %0 = compare{GT}(%lhs_value, %rhs_value)
// %1 = compare{NE}(%lhs_value, %lhs_value)
// %2 = or(%0, %1)
// %3 = select(%2, %lhs_value, %rhs_value)
// %4 = compare{EQ}(%lhs_value, %rhs_value)
// %5 = compare{LT}(%lhs_index, %rhs_index)
// %6 = and(%4, %5)
// %7 = or(%2, %6)
// %8 = select(%7, %lhs_index, %rhs_index)
// return %3, %8
// Also note that %1 may be folded if %lhs_value is of integer types.
LogicalResult MatchReduceToArgMinMaxType1(mhlo::ReduceOp reduce_op,
                                          bool is_float, bool is_argmax) {
  Block& body = reduce_op.getBody().front();
  if (body.getNumArguments() != 4) return failure();

  mhlo::ReturnOp return_op = dyn_cast<mhlo::ReturnOp>(body.back());
  if (!return_op || return_op.getNumOperands() != 2) return failure();

  mhlo::SelectOp value_select = llvm::dyn_cast_or_null<mhlo::SelectOp>(
      return_op.getOperand(0).getDefiningOp());
  if (!value_select || value_select.getOnTrue() != body.getArgument(0) ||
      value_select.getOnFalse() != body.getArgument(2))
    return failure();

  auto compare_direction =
      is_argmax ? mhlo::ComparisonDirection::GT : mhlo::ComparisonDirection::LT;
  if (is_float) {
    mhlo::OrOp value_or = llvm::dyn_cast_or_null<mhlo::OrOp>(
        value_select.getOperand(0).getDefiningOp());
    if (!value_or) return failure();

    mhlo::CompareOp value_gt = llvm::dyn_cast_or_null<mhlo::CompareOp>(
        value_or.getLhs().getDefiningOp());
    if (!value_gt || value_gt.getComparisonDirection() != compare_direction ||
        value_gt.getLhs() != body.getArgument(0) ||
        value_gt.getRhs() != body.getArgument(2))
      return failure();

    mhlo::CompareOp value_ne = llvm::dyn_cast_or_null<mhlo::CompareOp>(
        value_or.getRhs().getDefiningOp());
    if (!value_ne ||
        value_ne.getComparisonDirection() != mhlo::ComparisonDirection::NE ||
        value_ne.getLhs() != body.getArgument(0) ||
        value_ne.getRhs() != body.getArgument(0))
      return failure();
  } else {
    mhlo::CompareOp value_gt = llvm::dyn_cast_or_null<mhlo::CompareOp>(
        value_select.getOperand(0).getDefiningOp());
    if (!value_gt || value_gt.getComparisonDirection() != compare_direction ||
        value_gt.getLhs() != body.getArgument(0) ||
        value_gt.getRhs() != body.getArgument(2))
      return failure();
  }

  mhlo::SelectOp index_select = llvm::dyn_cast_or_null<mhlo::SelectOp>(
      return_op.getOperand(1).getDefiningOp());
  if (!index_select || index_select.getOnTrue() != body.getArgument(1) ||
      index_select.getOnFalse() != body.getArgument(3))
    return failure();

  mhlo::OrOp index_or = llvm::dyn_cast_or_null<mhlo::OrOp>(
      index_select.getPred().getDefiningOp());

  if (!index_or || index_or.getLhs() != value_select.getPred())
    return failure();

  mhlo::AndOp index_and =
      llvm::dyn_cast_or_null<mhlo::AndOp>(index_or.getRhs().getDefiningOp());
  if (!index_and) return failure();

  mhlo::CompareOp value_eq = llvm::dyn_cast_or_null<mhlo::CompareOp>(
      index_and.getLhs().getDefiningOp());
  if (!value_eq ||
      value_eq.getComparisonDirection() != mhlo::ComparisonDirection::EQ ||
      value_eq.getLhs() != body.getArgument(0) ||
      value_eq.getRhs() != body.getArgument(2))
    return failure();

  mhlo::CompareOp index_lt = llvm::dyn_cast_or_null<mhlo::CompareOp>(
      index_and.getRhs().getDefiningOp());
  if (!index_lt ||
      index_lt.getComparisonDirection() != mhlo::ComparisonDirection::LT ||
      index_lt.getLhs() != body.getArgument(1) ||
      index_lt.getRhs() != body.getArgument(3))
    return failure();

  return success();
}

template <typename Reduce, typename ArgReduce, typename BooleanReduce,
          bool is_argmax>
LogicalResult ConvertReduceOpToArgMinMax<
    Reduce, ArgReduce, BooleanReduce,
    is_argmax>::matchAndRewrite(mhlo::ReduceOp reduce_op, OpAdaptor adaptor,
                                ConversionPatternRewriter& rewriter) const {
  if (reduce_op.getInputs().size() != 2) return failure();
  if (reduce_op.getDimensions().getNumElements() != 1) return failure();

  // Check that the operand init is the expected value.
  DenseElementsAttr operand_init;
  if (!matchPattern(reduce_op.getInitValues().front(),
                    m_Constant(&operand_init)))
    return failure();
  if (!IsValueInitValue(operand_init)) return failure();

  // Check that the iota init is zero.
  DenseElementsAttr iota_init;
  if (!matchPattern(reduce_op.getInitValues().back(), m_Constant(&iota_init)))
    return failure();
  if (iota_init.getValues<APInt>()[0] != 0) return failure();

  // Verify that the second argument is an Iota op along the same dimension
  // as the reduction.
  Value iota = reduce_op.getInputs().back();
  if (!MatchIota(reduce_op.getDimensions(), iota)) return failure();

  // Match the reduction computation.
  const bool is_float = mlir::isa<FloatType>(operand_init.getElementType());
  if (failed(MatchReduceToArgMinMaxType1(reduce_op, is_float, is_argmax)) &&
      failed(MatchReduceToArgMinMaxType2(reduce_op, is_argmax)))
    return rewriter.notifyMatchFailure(
        reduce_op, "Unsupported Reduce -> ArgMax/ArgMin pattern");

  Value operand = reduce_op.getInputs().front();
  int64_t axis = reduce_op.getDimensions().getValues<int64_t>()[0];

  auto dim_type = RankedTensorType::get({1}, rewriter.getI32Type());
  auto reduction_indices = rewriter.create<arith::ConstantOp>(
      reduce_op.getLoc(), dim_type,
      rewriter.getI32TensorAttr({static_cast<int32_t>(axis)}));

  // Generate a Max and an ArgMax of as the mhlo op returns both while in TF
  // we have separate ops for them. If only one of them is used then the other
  // one will be garbage collected later.
  if (!mlir::isa<ShapedType>(operand.getType())) return failure();
  auto operand_type = mlir::cast<ShapedType>(operand.getType());
  if (operand_type.getElementType().isInteger(1)) {
    // TF does not support min or max on boolean (int1) arguments.
    // Use AnyOp for MaxOp and AllOp for MinOp.
    auto tf_reduce_op = rewriter.create<BooleanReduce>(
        reduce_op.getLoc(), reduce_op->getResult(0).getType(), operand,
        reduction_indices,
        /*keep_dim=*/rewriter.getBoolAttr(false));
    auto tf_argreduce_op = rewriter.create<ArgReduce>(
        reduce_op.getLoc(), reduce_op->getResult(1).getType(), operand,
        reduction_indices);

    rewriter.replaceOp(reduce_op, {tf_reduce_op, tf_argreduce_op});
  } else {
    auto tf_reduce_op = rewriter.create<Reduce>(
        reduce_op.getLoc(), reduce_op->getResult(0).getType(), operand,
        reduction_indices,
        /*keep_dim=*/rewriter.getBoolAttr(false));

    auto tf_argreduce_op = rewriter.create<ArgReduce>(
        reduce_op.getLoc(), reduce_op->getResult(1).getType(), operand,
        reduction_indices);

    rewriter.replaceOp(reduce_op, {tf_reduce_op, tf_argreduce_op});
  }
  return success();
}

template <typename Reduce, typename ArgReduce, typename BooleanReduce>
bool ConvertReduceOpToArgMax<Reduce, ArgReduce, BooleanReduce>::
    IsValueInitValue(const DenseElementsAttr& attr) const {
  auto element_type = attr.getType().getElementType();
  if (attr.getNumElements() != 1 || !element_type.isIntOrFloat()) return false;
  if (mlir::isa<FloatType>(element_type)) {
    auto value = *attr.value_begin<APFloat>();
    return value.isNegative() && value.isInfinity();
  } else if (element_type.isInteger(1)) {
    auto value = *attr.value_begin<APInt>();
    return value.isZero();
  } else {
    auto value = *attr.value_begin<APInt>();
    return element_type.isUnsignedInteger() ? value.isMinValue()
                                            : value.isMinSignedValue();
  }
}

template <typename Reduce, typename ArgReduce, typename BooleanReduce>
bool ConvertReduceOpToArgMin<Reduce, ArgReduce, BooleanReduce>::
    IsValueInitValue(const DenseElementsAttr& attr) const {
  auto element_type = attr.getType().getElementType();
  if (attr.getNumElements() != 1 || !element_type.isIntOrFloat()) return false;
  if (mlir::isa<FloatType>(element_type)) {
    auto value = *attr.value_begin<APFloat>();
    return !value.isNegative() && value.isInfinity();
  } else if (element_type.isInteger(1)) {
    auto value = *attr.value_begin<APInt>();
    return value.isZero();
  } else {
    auto value = *attr.value_begin<APInt>();
    return element_type.isUnsignedInteger() ? value.isMaxValue()
                                            : value.isMaxSignedValue();
  }
}

// Returns true if the given reduce op can be legalized to ArgMax/ArgMin ops.
std::optional<bool> IsReduceOpLegal(mhlo::ReduceOp reduce_op) {
  if (succeeded(MatchReduceToArgMinMaxType1(reduce_op, true, true)) ||
      succeeded(MatchReduceToArgMinMaxType1(reduce_op, false, true)) ||
      succeeded(MatchReduceToArgMinMaxType1(reduce_op, true, false)) ||
      succeeded(MatchReduceToArgMinMaxType1(reduce_op, false, false)) ||
      succeeded(MatchReduceToArgMinMaxType2(reduce_op, false)) ||
      succeeded(MatchReduceToArgMinMaxType2(reduce_op, true))) {
    // If the ReduceOp matches to one of the patterns above, its illegal to have
    // it in the model after the legalization is ran, because it should have
    // been legalized to an ArgMax/ArgMin op.
    return false;
  }

  return true;
}

template class ConvertReduceOpToArgMinMax<TFL::ReduceMaxOp, TFL::ArgMaxOp,
                                          TFL::ReduceAnyOp, true>;
template class ConvertReduceOpToArgMax<TFL::ReduceMaxOp, TFL::ArgMaxOp,
                                       TFL::ReduceAnyOp>;

template class ConvertReduceOpToArgMinMax<TFL::ReduceMinOp, TFL::ArgMinOp,
                                          TFL::ReduceAllOp, false>;
template class ConvertReduceOpToArgMin<TFL::ReduceMinOp, TFL::ArgMinOp,
                                       TFL::ReduceAllOp>;

template class ConvertReduceOpToArgMinMax<TF::MaxOp, TF::ArgMaxOp, TF::AnyOp,
                                          true>;
template class ConvertReduceOpToArgMax<TF::MaxOp, TF::ArgMaxOp, TF::AnyOp>;

template class ConvertReduceOpToArgMinMax<TF::MinOp, TF::ArgMinOp, TF::AllOp,
                                          false>;
template class ConvertReduceOpToArgMin<TF::MinOp, TF::ArgMinOp, TF::AllOp>;

}  // namespace odml
}  // namespace mlir