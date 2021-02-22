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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/iterator_range.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/IR/lhlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/map_hlo_to_lhlo_op.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
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
  using COp = ::mlir::complex::AddOp;
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
  using COp = ::mlir::complex::SubOp;
};

// Alias for the map from LHLO binary op type to STD floating-point op type.
template <typename LhloOp>
using ScalarFOp = typename LhloToScalarOp<LhloOp>::FOp;
// Alias for the map from LHLO binary op type to STD integer op type.
template <typename LhloOp>
using ScalarIOp = typename LhloToScalarOp<LhloOp>::IOp;
// Alias for the map from LHLO binary op type to STD complex op type.
template <typename LhloOp>
using ScalarCOp = typename LhloToScalarOp<LhloOp>::COp;

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
inline Value MapLhloOpToStdScalarOp<lmhlo::AddOp>(Location loc,
                                                  ArrayRef<Type> result_types,
                                                  ArrayRef<Value> args,
                                                  OpBuilder* b) {
  return MapLhloOpToStdScalarOpImpl<IntegerType, ScalarIOp<lmhlo::AddOp>,
                                    FloatType, ScalarFOp<lmhlo::AddOp>,
                                    ComplexType, ScalarCOp<lmhlo::AddOp>>{}(
      loc, result_types, args, b);
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
  return MapLhloOpToStdScalarOpImpl<FloatType, ::mlir::math::Atan2Op>{}(
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
      .Case("NE", CmpFPredicate::UNE)
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
  return MapLhloOpToStdScalarOpImpl<FloatType, ::mlir::math::ExpOp>{}(
      loc, result_types, args, b);
}

template <>
inline Value MapLhloOpToStdScalarOp<lmhlo::Expm1Op>(Location loc,
                                                    ArrayRef<Type> result_types,
                                                    ArrayRef<Value> args,
                                                    OpBuilder* b) {
  return MapLhloOpToStdScalarOpImpl<FloatType, ::mlir::math::ExpM1Op>{}(
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
  return MapLhloOpToStdScalarOpImpl<complex::CreateOp>{}(loc, result_types,
                                                         args, b);
}

template <>
inline Value MapLhloOpToStdScalarOp<lmhlo::RealOp>(Location loc,
                                                   ArrayRef<Type> result_types,
                                                   ArrayRef<Value> args,
                                                   OpBuilder* b) {
  return MapLhloOpToStdScalarOpImpl<complex::ReOp>{}(loc, result_types, args,
                                                     b);
}

template <>
inline Value MapLhloOpToStdScalarOp<lmhlo::ImagOp>(Location loc,
                                                   ArrayRef<Type> result_types,
                                                   ArrayRef<Value> args,
                                                   OpBuilder* b) {
  return MapLhloOpToStdScalarOpImpl<complex::ImOp>{}(loc, result_types, args,
                                                     b);
}

template <>
inline Value MapLhloOpToStdScalarOp<lmhlo::ConvertOp>(
    Location loc, ArrayRef<Type> result_types, ArrayRef<Value> args,
    OpBuilder* b) {
  Type sourceType = getElementTypeOrSelf(args.front().getType());
  Type targetType = getElementTypeOrSelf(result_types.front());

  // A boolean value is considered to be unsigned when converting to
  // floating-point. Otherwise, it will become `-1`.
  if (sourceType.isInteger(/*width=*/1) &&
      mlir::UIToFPOp::areCastCompatible(sourceType, targetType)) {
    return b->create<mlir::UIToFPOp>(loc, result_types, args, mlir::None);
  } else if (mlir::SIToFPOp::areCastCompatible(sourceType, targetType)) {
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
  if (targetType.isInteger(/*width=*/1)) {
    // When casting to bool, we need to compare whether the value is equal to
    // zero.
    if (sourceType.isSignlessInteger()) {
      Value zero_intval = b->create<::mlir::ConstantIntOp>(
          loc, 0, sourceType.cast<IntegerType>().getWidth());
      if (VectorType vec_type = args.front().getType().dyn_cast<VectorType>()) {
        zero_intval = b->create<::mlir::SplatOp>(loc, vec_type, zero_intval);
      }
      return b->create<mlir::CmpIOp>(loc, CmpIPredicate::ne, args.front(),
                                     zero_intval);
    } else if (sourceType.isa<FloatType>()) {
      Value zero = b->create<ConstantOp>(loc, b->getFloatAttr(sourceType, 0.0));
      if (VectorType vec_type = args.front().getType().dyn_cast<VectorType>()) {
        zero = b->create<::mlir::SplatOp>(loc, vec_type, zero);
      }
      return b->create<mlir::CmpFOp>(loc, CmpFPredicate::UNE, args.front(),
                                     zero);
    }
  }
  if (sourceType.isSignlessInteger() && targetType.isSignlessInteger()) {
    IntegerType src = sourceType.cast<IntegerType>();
    IntegerType res = targetType.cast<IntegerType>();
    if (src.getWidth() > res.getWidth()) {
      return b->create<mlir::TruncateIOp>(loc, result_types, args, mlir::None);
    } else if (src.getWidth() == 1) {
      // Special case boolean values, so they get casted to `1` instead of `-1`.
      return b->create<mlir::ZeroExtendIOp>(loc, result_types, args,
                                            mlir::None);
    } else if (src.getWidth() < res.getWidth()) {
      return b->create<mlir::SignExtendIOp>(loc, result_types, args,
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
  return MapLhloOpToStdScalarOpImpl<FloatType, ::mlir::math::CosOp>{}(
      loc, result_types, args, b);
}

template <>
inline Value MapLhloOpToStdScalarOp<lmhlo::SinOp>(Location loc,
                                                  ArrayRef<Type> result_types,
                                                  ArrayRef<Value> args,
                                                  OpBuilder* b) {
  return MapLhloOpToStdScalarOpImpl<FloatType, ::mlir::math::SinOp>{}(
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
  return MapLhloOpToStdScalarOpImpl<FloatType, ::mlir::math::LogOp>{}(
      loc, result_types, args, b);
}

inline Value LhloAlwaysPropagateNaN(Value v, ArrayRef<Value> args, Location loc,
                                    OpBuilder* b) {
  Type element_type = getElementTypeOrSelf(args.front().getType());
  if (auto float_type = element_type.dyn_cast<FloatType>()) {
    Value isnan =
        b->create<mlir::CmpFOp>(loc, CmpFPredicate::UNO, args[0], args[1]);

    auto nan_apfloat = APFloat::getQNaN(float_type.getFloatSemantics());
    Value nan = b->create<mlir::ConstantFloatOp>(loc, nan_apfloat, float_type);
    if (VectorType vec_type = args[0].getType().dyn_cast<VectorType>()) {
      nan = b->create<::mlir::SplatOp>(loc, vec_type, nan);
    }
    v = b->create<mlir::SelectOp>(loc, isnan, nan, v);
  }
  return v;
}

template <>
inline Value MapLhloOpToStdScalarOp<lmhlo::LogisticOp>(
    Location loc, ArrayRef<Type> result_types, ArrayRef<Value> args,
    OpBuilder* b) {
  auto ty = result_types.front().cast<FloatType>();
  Value one = b->create<ConstantOp>(loc, b->getFloatAttr(ty, 1.0));
  Value x = args.front();
  Value neg_x = b->create<NegFOp>(loc, x);
  Value exp_neg_x = b->create<::mlir::math::ExpOp>(loc, neg_x);
  Value one_add_exp_neg_x = b->create<AddFOp>(loc, one, exp_neg_x);
  return b->create<DivFOp>(loc, one, one_add_exp_neg_x);
}

template <>
inline Value MapLhloOpToStdScalarOp<lmhlo::Log1pOp>(Location loc,
                                                    ArrayRef<Type> result_types,
                                                    ArrayRef<Value> args,
                                                    OpBuilder* b) {
  return MapLhloOpToStdScalarOpImpl<FloatType, ::mlir::math::Log1pOp>{}(
      loc, result_types, args, b);
}

template <>
inline Value MapLhloOpToStdScalarOp<lmhlo::MaxOp>(Location loc,
                                                  ArrayRef<Type> result_types,
                                                  ArrayRef<Value> args,
                                                  OpBuilder* b) {
  return LhloAlwaysPropagateNaN(
      CompareSelectOpToStdScalarOp<
          IntegerType, ScalarIOp<lmhlo::CompareOp>, CmpIPredicate, FloatType,
          ScalarFOp<lmhlo::CompareOp>, CmpFPredicate>::map(loc, "GT",
                                                           result_types, args,
                                                           b),
      args, loc, b);
}

template <>
inline Value MapLhloOpToStdScalarOp<lmhlo::MinOp>(Location loc,
                                                  ArrayRef<Type> result_types,
                                                  ArrayRef<Value> args,
                                                  OpBuilder* b) {
  return LhloAlwaysPropagateNaN(
      CompareSelectOpToStdScalarOp<
          IntegerType, ScalarIOp<lmhlo::CompareOp>, CmpIPredicate, FloatType,
          ScalarFOp<lmhlo::CompareOp>, CmpFPredicate>::map(loc, "LT",
                                                           result_types, args,
                                                           b),
      args, loc, b);
}

template <>
inline Value MapLhloOpToStdScalarOp<lmhlo::ClampOp>(Location loc,
                                                    ArrayRef<Type> result_types,
                                                    ArrayRef<Value> args,
                                                    OpBuilder* b) {
  assert(args.size() == 3 && "expected 3 arguments");
  Value lb = args[0];
  Value x = args[1];
  Value ub = args[2];

  // clamp(lb, x, ub) = max(min(x, ub), lb)
  Value min_x_ub =
      MapLhloOpToStdScalarOp<lmhlo::MinOp>(loc, result_types, {x, ub}, b);
  return MapLhloOpToStdScalarOp<lmhlo::MaxOp>(loc, result_types, {min_x_ub, lb},
                                              b);
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
  return MapLhloOpToStdScalarOpImpl<FloatType, ::mlir::math::RsqrtOp>{}(
      loc, result_types, args, b);
}

template <>
inline Value MapLhloOpToStdScalarOp<lmhlo::PowOp>(Location loc,
                                                  ArrayRef<Type> result_types,
                                                  ArrayRef<Value> args,
                                                  OpBuilder* b) {
  lmhlo::PowOp::Adaptor adaptor(args);
  auto lb = ImplicitLocOpBuilder(loc, *b);
  // Floating point can use std::powf
  auto result_type = result_types.front();
  if (result_type.isa<::mlir::FloatType>())
    return MapLhloOpToStdScalarOpImpl<::mlir::math::PowFOp>{}(loc, result_types,
                                                              args, b);

  assert(result_type.isa<::mlir::IntegerType>() &&
         "only float and integer `pow` is supported right now");

  // Exponentiation by squaring:
  // https://en.wikipedia.org/wiki/Exponentiation_by_squaring;
  Value neg_one = lb.create<ConstantOp>(lb.getIntegerAttr(result_type, -1));
  Value zero = lb.create<ConstantOp>(lb.getIntegerAttr(result_type, 0));
  Value one = lb.create<ConstantOp>(lb.getIntegerAttr(result_type, 1));
  Value two = lb.create<ConstantOp>(lb.getIntegerAttr(result_type, 2));
  Value step = lb.create<ConstantIndexOp>(1);
  Value lowerBound = lb.create<ConstantIndexOp>(0);
  // Everything else would overflow for any exponent > 1, as 2^64
  // is the larget possible exponent for a 64-bit integer, and
  // that's 1 << 6.
  Value upperBound = lb.create<ConstantIndexOp>(6);
  auto original_base = adaptor.lhs();
  auto original_exponent = adaptor.rhs();

  Value accum =
      lb.create<scf::ForOp>(
            lowerBound, upperBound, step,
            SmallVector<Value>({one, original_base, original_exponent}),
            [&](OpBuilder& b, Location, Value v, ValueRange iters) {
              Value accum = iters[0];
              Value base = iters[1];
              Value exponent = iters[2];

              Value condition = b.create<CmpIOp>(
                  loc, CmpIPredicate::eq,
                  b.create<::mlir::AndOp>(loc, exponent, one), one);
              Value multiplied = b.create<::mlir::MulIOp>(loc, accum, base);
              accum =
                  b.create<::mlir::SelectOp>(loc, condition, multiplied, accum);
              base = b.create<::mlir::MulIOp>(loc, base, base);
              exponent =
                  b.create<::mlir::UnsignedShiftRightOp>(loc, exponent, one);
              b.create<scf::YieldOp>(
                  loc, SmallVector<Value>({accum, base, exponent}));
            })
          .getResult(0);

  Value rhs_is_even = lb.create<CmpIOp>(
      CmpIPredicate::eq, lb.create<SignedRemIOp>(adaptor.rhs(), two), zero);
  Value rhs_is_negative =
      lb.create<CmpIOp>(CmpIPredicate::slt, adaptor.rhs(), zero);
  Value lhs_is_one = lb.create<CmpIOp>(CmpIPredicate::eq, adaptor.lhs(), one);
  Value lhs_is_neg_one =
      lb.create<CmpIOp>(CmpIPredicate::eq, adaptor.lhs(), neg_one);

  // The accum is correct when the rhs is non-negative. When rhs is
  // negative, we return 0 for integer, with the exception of lhs values of 1
  // and -1 which have integer results for negative exponents. Specifically, the
  // calulation is the following:
  //
  // - Return accum if the rhs is not negative.
  // - Return 1 or -1 depending on the parity of rhs when the lhs is -1.
  // - Return 1 if lhs is 1.
  // - Else return 0.
  Value if_lhs_is_one = lb.create<::mlir::SelectOp>(lhs_is_one, one, zero);
  Value if_lhs_is_neg_one = lb.create<::mlir::SelectOp>(
      lhs_is_neg_one, lb.create<::mlir::SelectOp>(rhs_is_even, one, neg_one),
      if_lhs_is_one);
  return lb.create<::mlir::SelectOp>(rhs_is_negative, if_lhs_is_neg_one, accum);
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
    APFloat zero_apfloat(0.0f);
    zero_apfloat.convert(float_type.getFloatSemantics(),
                         APFloat::rmNearestTiesToEven, &ignored);
    Value zero =
        b->create<mlir::ConstantFloatOp>(loc, zero_apfloat, float_type);
    if (VectorType vec_type = args.front().getType().dyn_cast<VectorType>()) {
      zero = b->create<::mlir::SplatOp>(loc, vec_type, zero);
    }
    Value ne0_i1 =
        b->create<::mlir::CmpFOp>(loc, CmpFPredicate::ONE, args[0], zero);
    Value ne0_float = b->create<::mlir::UIToFPOp>(loc, ne0_i1, zero.getType());
    Value copy_sign =
        b->create<::mlir::CopySignOp>(loc, result_types, ne0_float, args[0]);
    auto is_nan =
        b->create<::mlir::CmpFOp>(loc, CmpFPredicate::UNO, args[0], args[0]);
    return b->create<::mlir::SelectOp>(loc, is_nan, args[0], copy_sign);
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
  return MapLhloOpToStdScalarOpImpl<FloatType, ::mlir::math::SqrtOp>{}(
      loc, result_types, args, b);
}

template <>
inline Value MapLhloOpToStdScalarOp<lmhlo::SubOp>(Location loc,
                                                  ArrayRef<Type> result_types,
                                                  ArrayRef<Value> args,
                                                  OpBuilder* b) {
  return MapLhloOpToStdScalarOpImpl<IntegerType, ScalarIOp<lmhlo::SubOp>,
                                    FloatType, ScalarFOp<lmhlo::SubOp>,
                                    ComplexType, ScalarCOp<lmhlo::SubOp>>{}(
      loc, result_types, args, b);
}

template <>
inline Value MapLhloOpToStdScalarOp<lmhlo::TanhOp>(Location loc,
                                                   ArrayRef<Type> result_types,
                                                   ArrayRef<Value> args,
                                                   OpBuilder* b) {
  return MapLhloOpToStdScalarOpImpl<FloatType, ::mlir::math::TanhOp>{}(
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
