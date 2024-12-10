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
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/hlo_matchers.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/util.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace mlir {
namespace odml {

//===------------------------------------------------------------------------===
// mhlo.reduce -> arg min/max
//===------------------------------------------------------------------------===

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

// Base class for converting mhlo::ReduceOp to TF/TFL ArgMax/ArgMin ops.
template <typename Reduce, typename ArgReduce, typename BooleanReduce,
          bool is_argmax>
class ConvertReduceOpToArgMinMax : public OpConversionPattern<mhlo::ReduceOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mhlo::ReduceOp reduce_op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final;

  virtual bool IsValueInitValue(const DenseElementsAttr& attr) const = 0;
};

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

// Base class for converting mhlo::ReduceOp to TF/TFL ArgMax/ArgMin ops.
template <typename Reduce, typename ArgReduce, typename BooleanReduce>
class ConvertReduceOpToArgMax
    : public ConvertReduceOpToArgMinMax<Reduce, ArgReduce, BooleanReduce,
                                        true> {
 public:
  using ConvertReduceOpToArgMinMax<Reduce, ArgReduce, BooleanReduce,
                                   true>::ConvertReduceOpToArgMinMax;
  bool IsValueInitValue(const DenseElementsAttr& attr) const override;
};

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

// Base class for converting mhlo::ReduceOp to TF/TFL ArgMax/ArgMin ops.
template <typename Reduce, typename ArgReduce, typename BooleanReduce>
class ConvertReduceOpToArgMin
    : public ConvertReduceOpToArgMinMax<Reduce, ArgReduce, BooleanReduce,
                                        false> {
 public:
  using ConvertReduceOpToArgMinMax<Reduce, ArgReduce, BooleanReduce,
                                   false>::ConvertReduceOpToArgMinMax;
  bool IsValueInitValue(const DenseElementsAttr& attr) const override;
};

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

//===------------------------------------------------------------------------===
// mhlo.reduce -> standard reductions
//===------------------------------------------------------------------------===

// If `value` is a splat constant, returns a success and set `splat_value`
// to the splate constant value.
// `SplatValueType` can be `APInt` or `APFloat`.
template <typename SplatValueType>
LogicalResult GetConstantSplatValue(Value value, SplatValueType& splat_value) {
  DenseElementsAttr attr;
  if (!matchPattern(value, m_Constant(&attr)) || !attr.isSplat()) {
    return failure();
  }

  splat_value = attr.getSplatValue<SplatValueType>();
  return success();
}

// Replace BinaryOp with a combination of BinaryOp and ReduceOp if the
// init value doesn't match the expectation of ReduceOp.
template <typename ReduceOp, typename BinaryOp, bool BuilderHasFAF = false>
LogicalResult rewriteNonMatchInitValue(mhlo::ReduceOp reduce_op, Value input,
                                       arith::ConstantOp reduction_indices,
                                       ConversionPatternRewriter& rewriter) {
  Value reduce_result = rewriter.create<ReduceOp>(
      reduce_op.getLoc(), reduce_op.getType(0), input, reduction_indices,
      /*keep_dim=*/rewriter.getBoolAttr(false));

  if constexpr (BuilderHasFAF) {
    rewriter.replaceOpWithNewOp<BinaryOp>(reduce_op, reduce_result,
                                          reduce_op.getInitValues()[0],
                                          rewriter.getStringAttr("NONE"));
  } else {
    rewriter.replaceOpWithNewOp<BinaryOp>(reduce_op, reduce_result.getType(),
                                          reduce_result,
                                          reduce_op.getInitValues()[0]);
  }

  return success();
}

DenseIntElementsAttr GetDimsAsI32Elements(OpBuilder& b, mhlo::ReduceOp op) {
  auto dims_attr = op.getDimensions();
  const auto n_dims = dims_attr.getNumElements();

  SmallVector<int32_t> reduce_dims;
  reduce_dims.reserve(n_dims);
  for (auto dim : dims_attr.getValues<int64_t>()) {
    reduce_dims.push_back(dim);
  }

  auto dim_type = RankedTensorType::get({n_dims}, b.getI32Type());
  return DenseIntElementsAttr::get(dim_type, reduce_dims);
}

// Cannot replace BinaryOp if the init value doesn't match the expectation of
// ReduceOp and there is no corresponding BinaryOp.
template <>
LogicalResult rewriteNonMatchInitValue<TFL::ReduceMaxOp, void>(
    mhlo::ReduceOp reduce_op, Value input, arith::ConstantOp reduction_indices,
    ConversionPatternRewriter& rewriter) {
  return failure();
}

template <>
LogicalResult rewriteNonMatchInitValue<TFL::ReduceMinOp, void>(
    mhlo::ReduceOp reduce_op, Value input, arith::ConstantOp reduction_indices,
    ConversionPatternRewriter& rewriter) {
  return failure();
}

template <>
LogicalResult rewriteNonMatchInitValue<TFL::ReduceAnyOp, void>(
    mhlo::ReduceOp reduce_op, Value input, arith::ConstantOp reduction_indices,
    ConversionPatternRewriter& rewriter) {
  return failure();
}

// Converts a mhlo.reduce op with a mlho binary operation into a tensorflow
// reduction operation. If the initial value can be ignored, then convert it
// into a single ReduceOp. Otherwise, convert it into a ReduceOp followed by
// a BinaryOp.
// For example:
//   1) A mhlo::ReduceOp on value `x` with a mhlo::AndOp and a constant initial
// value `true` is converted to a Any on value `x`.
//   2) A mhlo::ReduceOp on value `x` with a mhlo::AndOp with a non-constant
// initial value `y` is converted to a Any on value `x`, followed by a
// And with initial value `y`.
template <typename SrcBinaryOp, typename TargetReduceOp,
          typename TargetBinaryOp = void, bool BuilderHasFAF = false>
class ConvertReduce : public OpConversionPattern<mhlo::ReduceOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::ReduceOp reduce_op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    if (failed(MatchReduceOpOperand(reduce_op))) {
      return failure();
    }

    if (failed(MatchBinaryReduceFunction<SrcBinaryOp>(reduce_op.getBody()))) {
      return failure();
    }

    auto operand = reduce_op.getInputs()[0];

    //
    // build reduction dims ConstOp from attr (cast to 32bit for tflite).
    //=-----

    auto tfl_dims = GetDimsAsI32Elements(rewriter, reduce_op);
    auto tfl_dims_op =
        rewriter.create<arith::ConstantOp>(reduce_op.getLoc(), tfl_dims);

    //
    // replace with new reduce op, chaining binary op if needed.
    //=-----

    if (succeeded(MatchInitValue(reduce_op.getInitValues()[0]))) {
      rewriter.replaceOpWithNewOp<TargetReduceOp>(
          reduce_op, reduce_op.getType(0), operand, tfl_dims_op,
          /*keep_dim=*/rewriter.getBoolAttr(false));
      return success();
    }
    return rewriteNonMatchInitValue<TargetReduceOp, TargetBinaryOp,
                                    BuilderHasFAF>(reduce_op, operand,
                                                   tfl_dims_op, rewriter);
  }

 private:
  // Checks that the init value matches with the init value expected for the
  // target ReduceOp.
  virtual LogicalResult MatchInitValue(Value init_value) const = 0;

  LogicalResult MatchReduceOpOperand(mhlo::ReduceOp reduce_op) const {
    if (reduce_op.getInputs().size() != 1 ||
        reduce_op.getInitValues().size() != 1 ||
        reduce_op.getResults().size() != 1)
      return failure();

    if (!mlir::isa<RankedTensorType>(reduce_op.getInputs()[0].getType()))
      return failure();
    if (!mlir::isa<RankedTensorType>(reduce_op.getType(0))) return failure();
    return success();
  }
};

class ConvertReduceMul
    : public ConvertReduce<mhlo::MulOp, TFL::ReduceProdOp, TFL::MulOp, true> {
 public:
  using ConvertReduce::ConvertReduce;

  LogicalResult MatchInitValue(Value init_value) const override {
    auto type = mlir::cast<ShapedType>(init_value.getType()).getElementType();
    if (mlir::isa<FloatType>(type)) {
      float const_value;
      if (failed(GetConstantSplatValue<float>(init_value, const_value)) ||
          const_value != 1.0)
        return failure();
    } else if (mlir::isa<IntegerType>(type) && type.isSignlessInteger()) {
      int32_t const_value;
      if (failed(GetConstantSplatValue<int32_t>(init_value, const_value)) ||
          const_value != 1)
        return failure();
    } else {
      return failure();
    }

    return success();
  }
};

class ConvertReduceAdd
    : public ConvertReduce<mhlo::AddOp, TFL::SumOp, TFL::AddOp, true> {
 public:
  using ConvertReduce::ConvertReduce;

  LogicalResult MatchInitValue(Value init_value) const override {
    auto type = mlir::cast<ShapedType>(init_value.getType()).getElementType();
    if (mlir::isa<FloatType>(type)) {
      APFloat const_value(.0);
      if (failed(GetConstantSplatValue(init_value, const_value)) ||
          !const_value.isZero())
        return failure();
    } else if (mlir::isa<IntegerType>(type) && type.isSignlessInteger()) {
      APInt const_value;
      if (failed(GetConstantSplatValue(init_value, const_value)) ||
          !const_value.isZero())
        return failure();
    } else {
      return failure();
    }

    return success();
  }
};

class ConvertReduceMaxToReduceAny
    : public ConvertReduce<mhlo::MaxOp, TFL::ReduceAnyOp> {
 public:
  using ConvertReduce::ConvertReduce;

  LogicalResult MatchInitValue(Value init_value) const override {
    // This pattern is applicable only if the initial value is a boolean with
    // False value. Only then will it make sense to convert a
    // mhlo.reduce with mhlo.maximum reducer to a TFL.ReduceAnyOp. Because the
    // maximum value across a slice of a tensor compared to False can be viewed
    // as, checking if ANY value in the slice is True.
    auto type = mlir::cast<ShapedType>(init_value.getType()).getElementType();

    if (!mlir::isa<IntegerType>(type) || !type.isSignlessInteger() ||
        !(type.getIntOrFloatBitWidth() == 1))
      return failure();

    APInt const_value;
    if (failed(GetConstantSplatValue(init_value, const_value)) ||
        (const_value == 1))
      return failure();

    return success();
  }
};

class ConvertReduceMax : public ConvertReduce<mhlo::MaxOp, TFL::ReduceMaxOp> {
 public:
  using ConvertReduce::ConvertReduce;

  LogicalResult MatchInitValue(Value init_value) const override {
    auto type = mlir::cast<ShapedType>(init_value.getType()).getElementType();
    if (mlir::isa<FloatType>(type)) {
      APFloat const_value(.0);
      if (failed(GetConstantSplatValue(init_value, const_value)) ||
          !const_value.isInfinity() || !const_value.isNegative())
        return failure();
    } else if (mlir::isa<IntegerType>(type) && type.isSignlessInteger()) {
      // Do not handle the case where the mhlo.reduce of mhlo.maximum can be
      // legalized to TFL.ReduceAny. This can be possible if the dtype is i1
      if (type.getIntOrFloatBitWidth() == 1) return failure();
      APInt const_value;
      if (failed(GetConstantSplatValue(init_value, const_value)) ||
          !const_value.isMinSignedValue())
        return failure();
    } else {
      return failure();
    }
    return success();
  }
};

class ConvertReduceMin : public ConvertReduce<mhlo::MinOp, TFL::ReduceMinOp> {
 public:
  using ConvertReduce::ConvertReduce;

  LogicalResult MatchInitValue(Value init_value) const override {
    auto type = mlir::cast<ShapedType>(init_value.getType()).getElementType();

    if (mlir::isa<FloatType>(type)) {
      APFloat const_value(.0);
      if (failed(GetConstantSplatValue(init_value, const_value)) ||
          !const_value.isInfinity() || const_value.isNegative())
        return failure();
    } else if (mlir::isa<IntegerType>(type) && type.isSignlessInteger()) {
      APInt const_value;
      if (failed(GetConstantSplatValue(init_value, const_value)) ||
          !const_value.isMaxSignedValue())
        return failure();
    } else {
      return failure();
    }
    return success();
  }
};

class ConvertReduceAnd
    : public ConvertReduce<mhlo::AndOp, TFL::ReduceAllOp, TFL::LogicalAndOp> {
 public:
  using ConvertReduce<mhlo::AndOp, TFL::ReduceAllOp,
                      TFL::LogicalAndOp>::ConvertReduce;

  LogicalResult MatchInitValue(Value init_value) const override {
    DenseIntElementsAttr init_attr;
    if (!matchPattern(init_value, m_Constant(&init_attr)) ||
        !init_attr.getType().getElementType().isInteger(1) ||
        !init_attr.isSplat() || !init_attr.getSplatValue<BoolAttr>().getValue())
      return failure();
    return success();
  }
};

class ConvertReduceOr
    : public ConvertReduce<mhlo::OrOp, TFL::ReduceAnyOp, TFL::LogicalOrOp> {
 public:
  using ConvertReduce<mhlo::OrOp, TFL::ReduceAnyOp,
                      TFL::LogicalOrOp>::ConvertReduce;

  LogicalResult MatchInitValue(Value init_value) const override {
    DenseIntElementsAttr init_attr;
    if (!matchPattern(init_value, m_Constant(&init_attr)) ||
        !init_attr.getType().getElementType().isInteger(1) ||
        !init_attr.isSplat() || init_attr.getSplatValue<BoolAttr>().getValue())
      return failure();
    return success();
  }
};

//===------------------------------------------------------------------------===
// register patterns
//===------------------------------------------------------------------------===

// Returns false if the given reduce op can be legalized to ArgMax/ArgMin ops.
std::optional<bool> IsReduceOpLegal(mhlo::ReduceOp reduce_op) {
  if (succeeded(MatchReduceToArgMinMaxType1(reduce_op, true, true)) ||
      succeeded(MatchReduceToArgMinMaxType1(reduce_op, false, true)) ||
      succeeded(MatchReduceToArgMinMaxType1(reduce_op, true, false)) ||
      succeeded(MatchReduceToArgMinMaxType1(reduce_op, false, false)) ||
      succeeded(MatchReduceToArgMinMaxType2(reduce_op, false)) ||
      succeeded(MatchReduceToArgMinMaxType2(reduce_op, true))) {
    // If the ReduceOp matches to one of the patterns above, its illegal to
    // have
    // it in the model after the legalization is ran, because it should have
    // been legalized to an ArgMax/ArgMin op.
    return false;
  }
  return std::nullopt;
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

void PopulateReduceArgMinMaxTFPatterns(MLIRContext* ctx,
                                       RewritePatternSet& patterns) {
  using ConvertReduceOpToTfArgmax =
      ConvertReduceOpToArgMax<TF::MaxOp, TF::ArgMaxOp, TF::AnyOp>;

  using ConvertReduceOpToTfArgmin =
      ConvertReduceOpToArgMin<TF::MinOp, TF::ArgMinOp, TF::AllOp>;

  patterns.add<ConvertReduceOpToTfArgmin, ConvertReduceOpToTfArgmax>(ctx);
}

void PopulateReducePatterns(MLIRContext* ctx, RewritePatternSet& patterns,
                            ConversionTarget& target) {
  using ConvertReduceOpToTFLiteArgmax =
      ConvertReduceOpToArgMax<TFL::ReduceMaxOp, TFL::ArgMaxOp,
                              TFL::ReduceAnyOp>;

  using ConvertReduceOpToTFLiteArgmin =
      ConvertReduceOpToArgMin<TFL::ReduceMinOp, TFL::ArgMinOp,
                              TFL::ReduceAllOp>;

  patterns.add<ConvertReduceOpToTFLiteArgmax, ConvertReduceOpToTFLiteArgmin,
               ConvertReduceMul, ConvertReduceAdd, ConvertReduceMax,
               ConvertReduceMaxToReduceAny, ConvertReduceMin, ConvertReduceAnd,
               ConvertReduceOr>(ctx);

  target.addDynamicallyLegalOp<mhlo::ReduceOp>(IsReduceOpLegal);
}

}  // namespace odml
}  // namespace mlir
