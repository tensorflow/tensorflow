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

#ifndef TENSORFLOW_COMPILER_MLIR_HLO_INCLUDE_MLIR_HLO_DIALECT_MHLO_TRANSFORMS_MAP_LMHLO_TO_SCALAR_OP_H_
#define TENSORFLOW_COMPILER_MLIR_HLO_INCLUDE_MLIR_HLO_DIALECT_MHLO_TRANSFORMS_MAP_LMHLO_TO_SCALAR_OP_H_

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/IR/lhlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/map_hlo_to_lhlo_op.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/TypeUtilities.h"

namespace mlir {
namespace lmhlo {
namespace impl {

// A struct to map LhloBinaryOpTy type to the corresponding floating-point and
// integer scalar operation types.
template <typename LhloBinaryOpTy>
struct LhloToScalarOp;

template <>
struct LhloToScalarOp<lmhlo::AddOp> {
  using FOp = ::mlir::AddFOp;
  using IOp = ::mlir::AddIOp;
};
template <>
struct LhloToScalarOp<lmhlo::CompareOp> {
  using FOp = ::mlir::CmpFOp;
  using IOp = ::mlir::CmpIOp;
};
template <>
struct LhloToScalarOp<lmhlo::DivOp> {
  using FOp = ::mlir::DivFOp;
  using IOp = ::mlir::SignedDivIOp;
};
template <>
struct LhloToScalarOp<lmhlo::MulOp> {
  using FOp = ::mlir::MulFOp;
  using IOp = ::mlir::MulIOp;
};
template <>
struct LhloToScalarOp<lmhlo::RemOp> {
  using FOp = ::mlir::RemFOp;
  using IOp = ::mlir::SignedRemIOp;
};
template <>
struct LhloToScalarOp<lmhlo::SubOp> {
  using FOp = ::mlir::SubFOp;
  using IOp = ::mlir::SubIOp;
};

template <typename LhloBinaryOpTy>
struct ScalarOp {
  using FOp = typename LhloToScalarOp<LhloBinaryOpTy>::FOp;
  using IOp = typename LhloToScalarOp<LhloBinaryOpTy>::IOp;
};

// Alias for the map from LHLO binary op type to STD floating-point op type.
template <typename LhloOp>
using ScalarFOp = typename ScalarOp<LhloOp>::FOp;
// Alias for the map from LHLO binary op type to STD integer op type.
template <typename LhloOp>
using ScalarIOp = typename ScalarOp<LhloOp>::IOp;

template <typename... Args>
struct MapLhloOpToStdScalarOpImpl {
  Value operator()(Location loc, ArrayRef<Type> result_types,
                   ArrayRef<Value> args, OpBuilder* b) {
    return nullptr;
  }
};

template <typename StdScalarOp>
struct MapLhloOpToStdScalarOpImpl<StdScalarOp> {
  Value operator()(Location loc, ArrayRef<Type> result_types,
                   ArrayRef<Value> args, OpBuilder* b) {
    return b->template create<StdScalarOp>(loc, result_types, args, mlir::None);
  }
};

template <typename SupportedType, typename StdScalarOp, typename... Args>
struct MapLhloOpToStdScalarOpImpl<SupportedType, StdScalarOp, Args...> {
  Value operator()(Location loc, ArrayRef<Type> result_types,
                   ArrayRef<Value> args, OpBuilder* b) {
    Type element_type = getElementTypeOrSelf(args.front().getType());
    if (element_type.isa<SupportedType>()) {
      return b->template create<StdScalarOp>(loc, result_types, args,
                                             mlir::None);
    }
    return MapLhloOpToStdScalarOpImpl<Args...>{}(loc, result_types, args, b);
  }
};

// Inserts the computation that corresponds to the body of the loop for lowered
// LHLO unary/binary op. Returns the value for the result.
template <typename LhloOpTy>
inline Value MapLhloOpToStdScalarOp(Location loc, ArrayRef<Type> result_types,
                                    ArrayRef<Value> args, OpBuilder* b) {
  return MapLhloOpToStdScalarOpImpl<IntegerType, ScalarIOp<LhloOpTy>, FloatType,
                                    ScalarFOp<LhloOpTy>>{}(loc, result_types,
                                                           args, b);
}

template <>
inline Value MapLhloOpToStdScalarOp<lmhlo::AbsOp>(Location loc,
                                                  ArrayRef<Type> result_types,
                                                  ArrayRef<Value> args,
                                                  OpBuilder* b) {
  Type element_type = getElementTypeOrSelf(args.front().getType());
  if (element_type.isa<FloatType>()) {
    return MapLhloOpToStdScalarOpImpl<FloatType, ::mlir::AbsFOp>{}(
        loc, result_types, args, b);
  }
  if (element_type.isa<IntegerType>()) {
    // lmhlo.abs(x, result) ->  result = select((x > 0), x, sub(0, x))
    Value lhs = args[0];
    auto integer_type = element_type.dyn_cast<IntegerType>();

    Value zero_intval =
        b->create<::mlir::ConstantIntOp>(loc, 0, integer_type.getWidth());
    if (VectorType vec_type = args.front().getType().dyn_cast<VectorType>()) {
      zero_intval = b->create<::mlir::SplatOp>(loc, vec_type, zero_intval);
    }
    auto lhs_gt_zero = b->create<ScalarIOp<CompareOp>>(loc, CmpIPredicate::sge,
                                                       lhs, zero_intval);
    auto neg_val = b->create<ScalarIOp<lmhlo::SubOp>>(loc, zero_intval, lhs);
    return b->create<::mlir::SelectOp>(loc, lhs_gt_zero, lhs, neg_val);
  }
  return nullptr;
}

template <>
inline Value MapLhloOpToStdScalarOp<lmhlo::AndOp>(Location loc,
                                                  ArrayRef<Type> result_types,
                                                  ArrayRef<Value> args,
                                                  OpBuilder* b) {
  return MapLhloOpToStdScalarOpImpl<IntegerType, ::mlir::AndOp>{}(
      loc, result_types, args, b);
}

template <>
inline Value MapLhloOpToStdScalarOp<lmhlo::Atan2Op>(Location loc,
                                                    ArrayRef<Type> result_types,
                                                    ArrayRef<Value> args,
                                                    OpBuilder* b) {
  return MapLhloOpToStdScalarOpImpl<FloatType, ::mlir::Atan2Op>{}(
      loc, result_types, args, b);
}

template <typename PredicateType>
inline Optional<PredicateType> getCmpPredicate(StringRef comparison_direction) {
  return llvm::None;
}

template <>
inline Optional<CmpFPredicate> getCmpPredicate<CmpFPredicate>(
    StringRef comparison_direction) {
  return llvm::StringSwitch<Optional<CmpFPredicate>>(comparison_direction)
      .Case("EQ", CmpFPredicate::OEQ)
      .Case("NE", CmpFPredicate::ONE)
      .Case("GE", CmpFPredicate::OGE)
      .Case("GT", CmpFPredicate::OGT)
      .Case("LE", CmpFPredicate::OLE)
      .Case("LT", CmpFPredicate::OLT)
      .Default(llvm::None);
}

template <>
inline Optional<CmpIPredicate> getCmpPredicate<CmpIPredicate>(
    StringRef comparison_direction) {
  return llvm::StringSwitch<Optional<CmpIPredicate>>(comparison_direction)
      .Case("EQ", CmpIPredicate::eq)
      .Case("NE", CmpIPredicate::ne)
      .Case("GE", CmpIPredicate::sge)
      .Case("GT", CmpIPredicate::sgt)
      .Case("LE", CmpIPredicate::sle)
      .Case("LT", CmpIPredicate::slt)
      .Default(llvm::None);
}

template <typename CompareOpTy>
inline Value MapCompareOpToStdScalarOp(Location loc,
                                       StringRef comparison_direction,
                                       ArrayRef<Type> result_types,
                                       ArrayRef<Value> args, OpBuilder* b) {
  const auto& lhs = args[0];
  const auto& rhs = args[1];
  Type element_type = getElementTypeOrSelf(lhs.getType());
  if (element_type.isSignlessInteger()) {
    Optional<CmpIPredicate> predicate =
        getCmpPredicate<CmpIPredicate>(comparison_direction);
    assert(predicate.hasValue() && "expected valid comparison direction");
    return b->create<ScalarIOp<CompareOpTy>>(loc, predicate.getValue(), lhs,
                                             rhs);
  }
  if (element_type.isa<FloatType>()) {
    Optional<CmpFPredicate> predicate =
        getCmpPredicate<CmpFPredicate>(comparison_direction);
    assert(predicate.hasValue() && "expected valid comparison direction");
    return b->create<ScalarFOp<CompareOpTy>>(loc, predicate.getValue(), lhs,
                                             rhs);
  }
  return nullptr;
}

template <>
inline Value MapLhloOpToStdScalarOp<lmhlo::CopyOp>(Location loc,
                                                   ArrayRef<Type> result_types,
                                                   ArrayRef<Value> args,
                                                   OpBuilder* b) {
  return args.front();
}

template <>
inline Value MapLhloOpToStdScalarOp<lmhlo::ExpOp>(Location loc,
                                                  ArrayRef<Type> result_types,
                                                  ArrayRef<Value> args,
                                                  OpBuilder* b) {
  return MapLhloOpToStdScalarOpImpl<FloatType, ::mlir::ExpOp>{}(
      loc, result_types, args, b);
}

template <>
inline Value MapLhloOpToStdScalarOp<lmhlo::CeilOp>(Location loc,
                                                   ArrayRef<Type> result_types,
                                                   ArrayRef<Value> args,
                                                   OpBuilder* b) {
  return MapLhloOpToStdScalarOpImpl<FloatType, ::mlir::CeilFOp>{}(
      loc, result_types, args, b);
}

template <>
inline Value MapLhloOpToStdScalarOp<lmhlo::ComplexOp>(
    Location loc, ArrayRef<Type> result_types, ArrayRef<Value> args,
    OpBuilder* b) {
  return MapLhloOpToStdScalarOpImpl<CreateComplexOp>{}(loc, result_types, args,
                                                       b);
}

template <>
inline Value MapLhloOpToStdScalarOp<lmhlo::RealOp>(Location loc,
                                                   ArrayRef<Type> result_types,
                                                   ArrayRef<Value> args,
                                                   OpBuilder* b) {
  return MapLhloOpToStdScalarOpImpl<ReOp>{}(loc, result_types, args, b);
}

template <>
inline Value MapLhloOpToStdScalarOp<lmhlo::ImagOp>(Location loc,
                                                   ArrayRef<Type> result_types,
                                                   ArrayRef<Value> args,
                                                   OpBuilder* b) {
  return MapLhloOpToStdScalarOpImpl<ImOp>{}(loc, result_types, args, b);
}

template <>
inline Value MapLhloOpToStdScalarOp<lmhlo::ConvertOp>(
    Location loc, ArrayRef<Type> result_types, ArrayRef<Value> args,
    OpBuilder* b) {
  Type sourceType = getElementTypeOrSelf(args.front().getType());
  Type targetType = getElementTypeOrSelf(result_types.front());

  if (mlir::SIToFPOp::areCastCompatible(sourceType, targetType)) {
    return b->create<mlir::SIToFPOp>(loc, result_types, args, mlir::None);
  } else if (sourceType.isa<FloatType>() && targetType.isa<FloatType>()) {
    FloatType src = sourceType.cast<FloatType>();
    FloatType res = targetType.cast<FloatType>();
    if (src.getWidth() > res.getWidth()) {
      return b->create<mlir::FPTruncOp>(loc, result_types, args, mlir::None);
    } else if (src.getWidth() < res.getWidth()) {
      return b->create<mlir::FPExtOp>(loc, result_types, args, mlir::None);
    }
    // No conversion is needed for the same width floats
    return args.front();
  }
  if (sourceType.isSignlessInteger() && targetType.isSignlessInteger()) {
    IntegerType src = sourceType.cast<IntegerType>();
    IntegerType res = targetType.cast<IntegerType>();
    if (src.getWidth() > res.getWidth()) {
      return b->create<mlir::TruncateIOp>(loc, result_types, args, mlir::None);
    } else if (src.getWidth() < res.getWidth()) {
      return b->create<mlir::ZeroExtendIOp>(loc, result_types, args,
                                            mlir::None);
    }
    // No conversion is needed for the same width integers
    return args.front();
  }
  if (mlir::FPToSIOp::areCastCompatible(sourceType, targetType)) {
    return b->create<mlir::FPToSIOp>(loc, result_types, args, mlir::None);
  }
  return nullptr;
}

template <>
inline Value MapLhloOpToStdScalarOp<lmhlo::DotOp>(Location loc,
                                                  ArrayRef<Type> result_types,
                                                  ArrayRef<Value> args,
                                                  OpBuilder* b) {
  // Dot Op converter from lhlo to affine only accepts float and integer types.
  const auto& lhs = args[0];
  const auto& rhs = args[1];
  const auto& result = args[2];
  Type element_type = lhs.getType();
  if (element_type.isa<FloatType>()) {
    Value float_mul = MapLhloOpToStdScalarOpImpl<FloatType, ::mlir::MulFOp>{}(
        loc, result_types, {lhs, rhs}, b);
    return MapLhloOpToStdScalarOpImpl<FloatType, ::mlir::AddFOp>{}(
        loc, result_types, {float_mul, result}, b);
  }
  if (element_type.isa<IntegerType>()) {
    Value int_mul = MapLhloOpToStdScalarOpImpl<IntegerType, ::mlir::MulIOp>{}(
        loc, result_types, {lhs, rhs}, b);
    return MapLhloOpToStdScalarOpImpl<IntegerType, ::mlir::AddIOp>{}(
        loc, result_types, {int_mul, result}, b);
  }
  return nullptr;
}

template <>
inline Value MapLhloOpToStdScalarOp<lmhlo::CosOp>(Location loc,
                                                  ArrayRef<Type> result_types,
                                                  ArrayRef<Value> args,
                                                  OpBuilder* b) {
  return MapLhloOpToStdScalarOpImpl<FloatType, ::mlir::CosOp>{}(
      loc, result_types, args, b);
}

template <>
inline Value MapLhloOpToStdScalarOp<lmhlo::SinOp>(Location loc,
                                                  ArrayRef<Type> result_types,
                                                  ArrayRef<Value> args,
                                                  OpBuilder* b) {
  return MapLhloOpToStdScalarOpImpl<FloatType, ::mlir::SinOp>{}(
      loc, result_types, args, b);
}

template <>
inline Value MapLhloOpToStdScalarOp<lmhlo::FloorOp>(Location loc,
                                                    ArrayRef<Type> result_types,
                                                    ArrayRef<Value> args,
                                                    OpBuilder* b) {
  return MapLhloOpToStdScalarOpImpl<FloatType, ::mlir::FloorFOp>{}(
      loc, result_types, args, b);
}

template <>
inline Value MapLhloOpToStdScalarOp<lmhlo::IsFiniteOp>(
    Location loc, ArrayRef<Type> result_types, ArrayRef<Value> args,
    OpBuilder* b) {
  if (args[0].getType().isa<FloatType>()) {
    auto pos_inf = APFloat::getInf(
        args[0].getType().cast<FloatType>().getFloatSemantics());
    auto const_pos_inf =
        b->create<ConstantOp>(loc, b->getFloatAttr(args[0].getType(), pos_inf));
    Value abs_x = b->create<::mlir::AbsFOp>(loc, args[0]);
    return b->create<::mlir::CmpFOp>(loc, CmpFPredicate::ONE, abs_x,
                                     const_pos_inf);
  }
  return nullptr;
}

/// Implements the conversion of HLO op to scalar op (to use within region of a
/// linalg.generic op) for compare-select style operations like min/max.
template <typename... Args>
struct CompareSelectOpToStdScalarOp {
  static Value map(Location loc, StringRef comparison_direction,
                   ArrayRef<Type> result_types, ArrayRef<Value> args,
                   OpBuilder* b) {
    return nullptr;
  }
};

/// Specialization which allows converting to a comparison operation in standard
/// dialect with a given predicate based on the element type of the operand.
template <typename SupportedType, typename StdCompareOp, typename Predicate,
          typename... Args>
struct CompareSelectOpToStdScalarOp<SupportedType, StdCompareOp, Predicate,
                                    Args...> {
  static Value map(Location loc, StringRef comparison_direction,
                   ArrayRef<Type> result_types, ArrayRef<Value> args,
                   OpBuilder* b) {
    Type element_type = getElementTypeOrSelf(args.front().getType());
    if (element_type.isa<SupportedType>()) {
      auto predicate = getCmpPredicate<Predicate>(comparison_direction);
      assert(predicate.hasValue() && "expected valid comparison direction");
      auto cmp = b->template create<StdCompareOp>(loc, predicate.getValue(),
                                                  args[0], args[1]);
      return b->create<::mlir::SelectOp>(loc, cmp, args[0], args[1]);
    }
    return CompareSelectOpToStdScalarOp<Args...>::map(loc, comparison_direction,
                                                      result_types, args, b);
  }
};

template <>
inline Value MapLhloOpToStdScalarOp<lmhlo::LogOp>(Location loc,
                                                  ArrayRef<Type> result_types,
                                                  ArrayRef<Value> args,
                                                  OpBuilder* b) {
  return MapLhloOpToStdScalarOpImpl<FloatType, ::mlir::LogOp>{}(
      loc, result_types, args, b);
}

template <>
inline Value MapLhloOpToStdScalarOp<lmhlo::MaxOp>(Location loc,
                                                  ArrayRef<Type> result_types,
                                                  ArrayRef<Value> args,
                                                  OpBuilder* b) {
  return CompareSelectOpToStdScalarOp<
      IntegerType, ScalarIOp<lmhlo::CompareOp>, CmpIPredicate, FloatType,
      ScalarFOp<lmhlo::CompareOp>, CmpFPredicate>::map(loc, "GT", result_types,
                                                       args, b);
}

template <>
inline Value MapLhloOpToStdScalarOp<lmhlo::MinOp>(Location loc,
                                                  ArrayRef<Type> result_types,
                                                  ArrayRef<Value> args,
                                                  OpBuilder* b) {
  return CompareSelectOpToStdScalarOp<
      IntegerType, ScalarIOp<lmhlo::CompareOp>, CmpIPredicate, FloatType,
      ScalarFOp<lmhlo::CompareOp>, CmpFPredicate>::map(loc, "LT", result_types,
                                                       args, b);
}

template <>
inline Value MapLhloOpToStdScalarOp<lmhlo::NegOp>(Location loc,
                                                  ArrayRef<Type> result_types,
                                                  ArrayRef<Value> args,
                                                  OpBuilder* b) {
  Type element_type = getElementTypeOrSelf(args.front().getType());
  if (element_type.isa<FloatType>()) {
    return MapLhloOpToStdScalarOpImpl<FloatType, ::mlir::NegFOp>{}(
        loc, result_types, args, b);
  }
  if (element_type.isa<IntegerType>()) {
    // lmhlo.neg(x, result) -> result = sub(0, x)
    Value lhs = args[0];
    auto integer_type = element_type.dyn_cast<IntegerType>();

    Value zero_intval =
        b->create<::mlir::ConstantIntOp>(loc, 0, integer_type.getWidth());
    if (VectorType vec_type = args.front().getType().dyn_cast<VectorType>()) {
      zero_intval = b->create<::mlir::SplatOp>(loc, vec_type, zero_intval);
    }
    return b->create<ScalarIOp<lmhlo::SubOp>>(loc, zero_intval, lhs);
  }
  return nullptr;
}

template <>
inline Value MapLhloOpToStdScalarOp<lmhlo::NotOp>(Location loc,
                                                  ArrayRef<Type> result_types,
                                                  ArrayRef<Value> args,
                                                  OpBuilder* b) {
  Type element_type = getElementTypeOrSelf(args.front().getType());
  if (auto integer_type = element_type.dyn_cast<IntegerType>()) {
    // lmhlo.not(x) -> x ^ -1
    Value all_ones =
        b->create<::mlir::ConstantIntOp>(loc, -1, integer_type.getWidth());
    if (VectorType vec_type = args.front().getType().dyn_cast<VectorType>()) {
      all_ones = b->create<::mlir::SplatOp>(loc, vec_type, all_ones);
    }
    return b->create<::mlir::XOrOp>(loc, all_ones, args[0]);
  }
  return nullptr;
}

template <>
inline Value MapLhloOpToStdScalarOp<lmhlo::OrOp>(Location loc,
                                                 ArrayRef<Type> result_types,
                                                 ArrayRef<Value> args,
                                                 OpBuilder* b) {
  return MapLhloOpToStdScalarOpImpl<IntegerType, ::mlir::OrOp>{}(
      loc, result_types, args, b);
}

template <>
inline Value MapLhloOpToStdScalarOp<lmhlo::RsqrtOp>(Location loc,
                                                    ArrayRef<Type> result_types,
                                                    ArrayRef<Value> args,
                                                    OpBuilder* b) {
  return MapLhloOpToStdScalarOpImpl<FloatType, ::mlir::RsqrtOp>{}(
      loc, result_types, args, b);
}

template <>
inline Value MapLhloOpToStdScalarOp<lmhlo::SelectOp>(
    Location loc, ArrayRef<Type> result_types, ArrayRef<Value> args,
    OpBuilder* b) {
  return MapLhloOpToStdScalarOpImpl<::mlir::SelectOp>{}(loc, result_types, args,
                                                        b);
}

template <>
inline Value MapLhloOpToStdScalarOp<lmhlo::ShiftLeftOp>(
    Location loc, ArrayRef<Type> result_types, ArrayRef<Value> args,
    OpBuilder* b) {
  return MapLhloOpToStdScalarOpImpl<IntegerType, mlir::ShiftLeftOp>{}(
      loc, result_types, args, b);
}

template <>
inline Value MapLhloOpToStdScalarOp<lmhlo::ShiftRightArithmeticOp>(
    Location loc, ArrayRef<Type> result_types, ArrayRef<Value> args,
    OpBuilder* b) {
  return MapLhloOpToStdScalarOpImpl<IntegerType, mlir::SignedShiftRightOp>{}(
      loc, result_types, args, b);
}

template <>
inline Value MapLhloOpToStdScalarOp<lmhlo::ShiftRightLogicalOp>(
    Location loc, ArrayRef<Type> result_types, ArrayRef<Value> args,
    OpBuilder* b) {
  return MapLhloOpToStdScalarOpImpl<IntegerType, mlir::UnsignedShiftRightOp>{}(
      loc, result_types, args, b);
}

template <>
inline Value MapLhloOpToStdScalarOp<lmhlo::SignOp>(Location loc,
                                                   ArrayRef<Type> result_types,
                                                   ArrayRef<Value> args,
                                                   OpBuilder* b) {
  Type element_type = getElementTypeOrSelf(args.front().getType());
  if (auto float_type = element_type.dyn_cast<FloatType>()) {
    bool ignored;
    APFloat one_apfloat(1.0f);
    one_apfloat.convert(float_type.getFloatSemantics(),
                        APFloat::rmNearestTiesToEven, &ignored);
    Value one = b->create<mlir::ConstantFloatOp>(loc, one_apfloat, float_type);
    if (VectorType vec_type = args.front().getType().dyn_cast<VectorType>()) {
      one = b->create<::mlir::SplatOp>(loc, vec_type, one);
    }
    return b->create<::mlir::CopySignOp>(loc, result_types, one, args[0]);
  } else if (auto integer_type = element_type.dyn_cast<IntegerType>()) {
    // sign(x) = x == 0 ? 0 : ((x s>> 31) | 1)
    Value zero =
        b->create<::mlir::ConstantIntOp>(loc, 0, integer_type.getWidth());
    Value bitwidth_minus_one = b->create<::mlir::ConstantIntOp>(
        loc, integer_type.getWidth() - 1, integer_type.getWidth());
    Value one =
        b->create<::mlir::ConstantIntOp>(loc, 1, integer_type.getWidth());
    if (VectorType vec_type = args.front().getType().dyn_cast<VectorType>()) {
      zero = b->create<::mlir::SplatOp>(loc, vec_type, zero);
      bitwidth_minus_one =
          b->create<::mlir::SplatOp>(loc, vec_type, bitwidth_minus_one);
      one = b->create<::mlir::SplatOp>(loc, vec_type, one);
    }
    Value cmp =
        b->create<::mlir::CmpIOp>(loc, CmpIPredicate::eq, args[0], zero);
    Value ashr =
        b->create<::mlir::SignedShiftRightOp>(loc, args[0], bitwidth_minus_one);
    Value or_op = b->create<::mlir::OrOp>(loc, ashr, one);
    return b->create<::mlir::SelectOp>(loc, cmp, zero, or_op);
  }
  return nullptr;
}

template <>
inline Value MapLhloOpToStdScalarOp<lmhlo::SqrtOp>(Location loc,
                                                   ArrayRef<Type> result_types,
                                                   ArrayRef<Value> args,
                                                   OpBuilder* b) {
  return MapLhloOpToStdScalarOpImpl<FloatType, ::mlir::SqrtOp>{}(
      loc, result_types, args, b);
}

template <>
inline Value MapLhloOpToStdScalarOp<lmhlo::TanhOp>(Location loc,
                                                   ArrayRef<Type> result_types,
                                                   ArrayRef<Value> args,
                                                   OpBuilder* b) {
  return MapLhloOpToStdScalarOpImpl<FloatType, ::mlir::TanhOp>{}(
      loc, result_types, args, b);
}

template <>
inline Value MapLhloOpToStdScalarOp<lmhlo::XorOp>(Location loc,
                                                  ArrayRef<Type> result_types,
                                                  ArrayRef<Value> args,
                                                  OpBuilder* b) {
  return MapLhloOpToStdScalarOpImpl<IntegerType, ::mlir::XOrOp>{}(
      loc, result_types, args, b);
}

}  // namespace impl

struct HloOpToStdScalarOp {
  // Implementation for LHLO ops except lmhlo::CompareOp.
  template <typename HloOpTy, typename LhloOpTy = HloOpTy,
            typename = std::enable_if_t<
                !std::is_same<LhloOpTy, lmhlo::CompareOp>::value &&
                std::is_same<typename mhlo::HloToLhloOp<LhloOpTy>,
                             std::false_type>::value>>
  static Value map(HloOpTy op, ArrayRef<Type> result_types,
                   ArrayRef<Value> args, OpBuilder* b, unsigned i = 0) {
    return impl::MapLhloOpToStdScalarOp<LhloOpTy>(op.getLoc(), result_types,
                                                  args, b);
  }

  // Implementation for HLO ops except mhlo::CompareOp.
  template <typename HloOpTy, typename LhloOpTy = mhlo::HloToLhloOp<HloOpTy>,
            typename = std::enable_if_t<
                !std::is_same<LhloOpTy, lmhlo::CompareOp>::value &&
                !std::is_same<LhloOpTy, std::false_type>::value>>
  static Value map(HloOpTy op, ArrayRef<Type> result_types,
                   ArrayRef<Value> args, OpBuilder* b, int i = 0) {
    return impl::MapLhloOpToStdScalarOp<LhloOpTy>(op.getLoc(), result_types,
                                                  args, b);
  }

  // Implementation for lmhlo::CompareOp.
  template <typename LhloOpTy, typename = std::enable_if_t<std::is_same<
                                   LhloOpTy, lmhlo::CompareOp>::value>>
  static Value map(lmhlo::CompareOp op, ArrayRef<Type> result_types,
                   ArrayRef<Value> args, OpBuilder* b) {
    auto comparison_direction = op.comparison_direction();
    return impl::MapCompareOpToStdScalarOp<lmhlo::CompareOp>(
        op.getLoc(), comparison_direction, result_types, args, b);
  }

  // Implementation for mhlo::CompareOp.
  template <typename HloOpTy,
            typename =
                std::enable_if_t<std::is_same<HloOpTy, mhlo::CompareOp>::value>>
  static Value map(mhlo::CompareOp op, ArrayRef<Type> result_types,
                   ArrayRef<Value> args, OpBuilder* b) {
    auto comparison_direction = op.comparison_direction();
    return impl::MapCompareOpToStdScalarOp<lmhlo::CompareOp>(
        op.getLoc(), comparison_direction, result_types, args, b);
  }

  // Implementation for LHLO ops except lmhlo::CompareOp.
  template <typename LhloOpTy,
            typename = std::enable_if_t<
                !std::is_same<LhloOpTy, lmhlo::CompareOp>::value &&
                std::is_same<typename mhlo::HloToLhloOp<LhloOpTy>,
                             std::false_type>::value>>
  static Value map(Location loc, ArrayRef<Type> result_types,
                   ArrayRef<Value> args, OpBuilder* b, unsigned i = 0) {
    return impl::MapLhloOpToStdScalarOp<LhloOpTy>(loc, result_types, args, b);
  }

  // Implementation for lmhlo::CompareOp.
  template <typename LhloOpTy, typename = std::enable_if_t<std::is_same<
                                   LhloOpTy, lmhlo::CompareOp>::value>>
  static Value map(Location loc, StringRef comparison_direction,
                   ArrayRef<Type> result_types, ArrayRef<Value> args,
                   OpBuilder* b) {
    return impl::MapCompareOpToStdScalarOp<lmhlo::CompareOp>(
        loc, comparison_direction, result_types, args, b);
  }
};

}  // namespace lmhlo
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_HLO_INCLUDE_MLIR_HLO_DIALECT_MHLO_TRANSFORMS_MAP_LMHLO_TO_SCALAR_OP_H_
