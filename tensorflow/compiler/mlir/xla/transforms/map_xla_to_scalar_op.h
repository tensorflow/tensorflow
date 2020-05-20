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

#ifndef TENSORFLOW_COMPILER_MLIR_XLA_TRANSFORMS_MAP_XLA_TO_SCALAR_OP_H_
#define TENSORFLOW_COMPILER_MLIR_XLA_TRANSFORMS_MAP_XLA_TO_SCALAR_OP_H_

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"
#include "tensorflow/compiler/mlir/xla/ir/lhlo_ops.h"
#include "tensorflow/compiler/mlir/xla/transforms/map_hlo_to_lhlo_op.h"

namespace mlir {
namespace xla_lhlo {
namespace impl {

// A struct to map LhloBinaryOpTy type to the corresponding floating-point and
// integer scalar operation types.
template <typename LhloBinaryOpTy>
struct LhloToScalarOp;

template <>
struct LhloToScalarOp<xla_lhlo::AddOp> {
  using FOp = ::mlir::AddFOp;
  using IOp = ::mlir::AddIOp;
};
template <>
struct LhloToScalarOp<xla_lhlo::CompareOp> {
  using FOp = ::mlir::CmpFOp;
  using IOp = ::mlir::CmpIOp;
};
template <>
struct LhloToScalarOp<xla_lhlo::DivOp> {
  using FOp = ::mlir::DivFOp;
  using IOp = ::mlir::SignedDivIOp;
};
template <>
struct LhloToScalarOp<xla_lhlo::MulOp> {
  using FOp = ::mlir::MulFOp;
  using IOp = ::mlir::MulIOp;
};
template <>
struct LhloToScalarOp<xla_lhlo::RemOp> {
  using FOp = ::mlir::RemFOp;
  using IOp = ::mlir::SignedRemIOp;
};
template <>
struct LhloToScalarOp<xla_lhlo::SubOp> {
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
    Type element_type = args.front().getType();
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
inline Value MapLhloOpToStdScalarOp<xla_lhlo::AbsOp>(
    Location loc, ArrayRef<Type> result_types, ArrayRef<Value> args,
    OpBuilder* b) {
  Type element_type = args.front().getType();
  if (element_type.isa<FloatType>()) {
    return MapLhloOpToStdScalarOpImpl<FloatType, ::mlir::AbsFOp>{}(
        loc, result_types, args, b);
  }
  if (element_type.isa<IntegerType>()) {
    // xla_lhlo.abs(x, result) ->  result = select((x > 0), x, sub(0, x))
    Value lhs = args[0];
    auto integer_type = element_type.dyn_cast<IntegerType>();

    auto zero_intval =
        b->create<::mlir::ConstantIntOp>(loc, 0, integer_type.getWidth());
    auto lhs_gt_zero = b->create<ScalarIOp<CompareOp>>(loc, CmpIPredicate::sge,
                                                       lhs, zero_intval);
    auto neg_val = b->create<ScalarIOp<xla_lhlo::SubOp>>(loc, zero_intval, lhs);
    return b->create<::mlir::SelectOp>(loc, lhs_gt_zero, lhs, neg_val);
  }
  return nullptr;
}

template <>
inline Value MapLhloOpToStdScalarOp<xla_lhlo::AndOp>(
    Location loc, ArrayRef<Type> result_types, ArrayRef<Value> args,
    OpBuilder* b) {
  return MapLhloOpToStdScalarOpImpl<IntegerType, ::mlir::AndOp>{}(
      loc, result_types, args, b);
}

template <typename PredicateType>
inline Optional<PredicateType> getCmpPredicate(
    StringRef xla_comparison_direction) {
  return llvm::None;
}

template <>
inline Optional<CmpFPredicate> getCmpPredicate<CmpFPredicate>(
    StringRef xla_comparison_direction) {
  return llvm::StringSwitch<Optional<CmpFPredicate>>(xla_comparison_direction)
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
    StringRef xla_comparison_direction) {
  return llvm::StringSwitch<Optional<CmpIPredicate>>(xla_comparison_direction)
      .Case("EQ", CmpIPredicate::eq)
      .Case("NE", CmpIPredicate::ne)
      .Case("GE", CmpIPredicate::sge)
      .Case("GT", CmpIPredicate::sgt)
      .Case("LE", CmpIPredicate::sle)
      .Case("LT", CmpIPredicate::slt)
      .Default(llvm::None);
}

template <typename XLACompareOpTy>
inline Value MapXlaCompareOpToStdScalarOp(Location loc,
                                          StringRef comparison_direction,
                                          ArrayRef<Type> result_types,
                                          ArrayRef<Value> args, OpBuilder* b) {
  const auto& lhs = args[0];
  const auto& rhs = args[1];
  Type element_type = lhs.getType();
  if (element_type.isSignlessInteger()) {
    Optional<CmpIPredicate> predicate =
        getCmpPredicate<CmpIPredicate>(comparison_direction);
    assert(predicate.hasValue() && "expected valid comparison direction");
    return b->create<ScalarIOp<XLACompareOpTy>>(loc, predicate.getValue(), lhs,
                                                rhs);
  }
  if (element_type.isa<FloatType>()) {
    Optional<CmpFPredicate> predicate =
        getCmpPredicate<CmpFPredicate>(comparison_direction);
    assert(predicate.hasValue() && "expected valid comparison direction");
    return b->create<ScalarFOp<XLACompareOpTy>>(loc, predicate.getValue(), lhs,
                                                rhs);
  }
  return nullptr;
}

template <>
inline Value MapLhloOpToStdScalarOp<xla_lhlo::CopyOp>(
    Location loc, ArrayRef<Type> result_types, ArrayRef<Value> args,
    OpBuilder* b) {
  return args.front();
}

template <>
inline Value MapLhloOpToStdScalarOp<xla_lhlo::ExpOp>(
    Location loc, ArrayRef<Type> result_types, ArrayRef<Value> args,
    OpBuilder* b) {
  return MapLhloOpToStdScalarOpImpl<FloatType, ::mlir::ExpOp>{}(
      loc, result_types, args, b);
}

template <>
inline Value MapLhloOpToStdScalarOp<xla_lhlo::CeilOp>(
    Location loc, ArrayRef<Type> result_types, ArrayRef<Value> args,
    OpBuilder* b) {
  return MapLhloOpToStdScalarOpImpl<FloatType, ::mlir::CeilFOp>{}(
      loc, result_types, args, b);
}

template <>
inline Value MapLhloOpToStdScalarOp<xla_lhlo::ComplexOp>(
    Location loc, ArrayRef<Type> result_types, ArrayRef<Value> args,
    OpBuilder* b) {
  return MapLhloOpToStdScalarOpImpl<CreateComplexOp>{}(loc, result_types, args,
                                                       b);
}

template <>
inline Value MapLhloOpToStdScalarOp<xla_lhlo::RealOp>(
    Location loc, ArrayRef<Type> result_types, ArrayRef<Value> args,
    OpBuilder* b) {
  return MapLhloOpToStdScalarOpImpl<ReOp>{}(loc, result_types, args, b);
}

template <>
inline Value MapLhloOpToStdScalarOp<xla_lhlo::ImagOp>(
    Location loc, ArrayRef<Type> result_types, ArrayRef<Value> args,
    OpBuilder* b) {
  return MapLhloOpToStdScalarOpImpl<ImOp>{}(loc, result_types, args, b);
}

template <>
inline Value MapLhloOpToStdScalarOp<xla_lhlo::ConvertOp>(
    Location loc, ArrayRef<Type> result_types, ArrayRef<Value> args,
    OpBuilder* b) {
  Type sourceType = args.front().getType();
  Type targetType = result_types.front();

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
inline Value MapLhloOpToStdScalarOp<xla_lhlo::CosOp>(
    Location loc, ArrayRef<Type> result_types, ArrayRef<Value> args,
    OpBuilder* b) {
  return MapLhloOpToStdScalarOpImpl<FloatType, ::mlir::CosOp>{}(
      loc, result_types, args, b);
}

template <>
inline Value MapLhloOpToStdScalarOp<xla_lhlo::SinOp>(
    Location loc, ArrayRef<Type> result_types, ArrayRef<Value> args,
    OpBuilder* b) {
  return MapLhloOpToStdScalarOpImpl<FloatType, ::mlir::SinOp>{}(
      loc, result_types, args, b);
}

/// Implements the conversion of XLA op to scalar op (to use within region of a
/// linalg.generic op) for compare-select style operations like min/max.
template <typename... Args>
struct XlaCompareSelectOpToStdScalarOp {
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
struct XlaCompareSelectOpToStdScalarOp<SupportedType, StdCompareOp, Predicate,
                                       Args...> {
  static Value map(Location loc, StringRef comparison_direction,
                   ArrayRef<Type> result_types, ArrayRef<Value> args,
                   OpBuilder* b) {
    Type element_type = args.front().getType();
    if (element_type.isa<SupportedType>()) {
      auto predicate = getCmpPredicate<Predicate>(comparison_direction);
      assert(predicate.hasValue() && "expected valid comparison direction");
      auto cmp = b->template create<StdCompareOp>(loc, predicate.getValue(),
                                                  args[0], args[1]);
      return b->create<::mlir::SelectOp>(loc, cmp, args[0], args[1]);
    }
    return XlaCompareSelectOpToStdScalarOp<Args...>::map(
        loc, comparison_direction, result_types, args, b);
  }
};

template <>
inline Value MapLhloOpToStdScalarOp<xla_lhlo::LogOp>(
    Location loc, ArrayRef<Type> result_types, ArrayRef<Value> args,
    OpBuilder* b) {
  return MapLhloOpToStdScalarOpImpl<FloatType, ::mlir::LogOp>{}(
      loc, result_types, args, b);
}

template <>
inline Value MapLhloOpToStdScalarOp<xla_lhlo::MaxOp>(
    Location loc, ArrayRef<Type> result_types, ArrayRef<Value> args,
    OpBuilder* b) {
  return XlaCompareSelectOpToStdScalarOp<
      IntegerType, ScalarIOp<xla_lhlo::CompareOp>, CmpIPredicate, FloatType,
      ScalarFOp<xla_lhlo::CompareOp>, CmpFPredicate>::map(loc, "GT",
                                                          result_types, args,
                                                          b);
}

template <>
inline Value MapLhloOpToStdScalarOp<xla_lhlo::MinOp>(
    Location loc, ArrayRef<Type> result_types, ArrayRef<Value> args,
    OpBuilder* b) {
  return XlaCompareSelectOpToStdScalarOp<
      IntegerType, ScalarIOp<xla_lhlo::CompareOp>, CmpIPredicate, FloatType,
      ScalarFOp<xla_lhlo::CompareOp>, CmpFPredicate>::map(loc, "LT",
                                                          result_types, args,
                                                          b);
}

template <>
inline Value MapLhloOpToStdScalarOp<xla_lhlo::NegOp>(
    Location loc, ArrayRef<Type> result_types, ArrayRef<Value> args,
    OpBuilder* b) {
  Type element_type = args.front().getType();
  if (element_type.isa<FloatType>()) {
    return MapLhloOpToStdScalarOpImpl<FloatType, ::mlir::NegFOp>{}(
        loc, result_types, args, b);
  }
  if (element_type.isa<IntegerType>()) {
    // xla_lhlo.neg(x, result) -> result = sub(0, x)
    Value lhs = args[0];
    auto integer_type = element_type.dyn_cast<IntegerType>();

    auto zero_intval =
        b->create<::mlir::ConstantIntOp>(loc, 0, integer_type.getWidth());
    return b->create<ScalarIOp<xla_lhlo::SubOp>>(loc, zero_intval, lhs);
  }
  return nullptr;
}

template <>
inline Value MapLhloOpToStdScalarOp<xla_lhlo::RsqrtOp>(
    Location loc, ArrayRef<Type> result_types, ArrayRef<Value> args,
    OpBuilder* b) {
  return MapLhloOpToStdScalarOpImpl<FloatType, ::mlir::RsqrtOp>{}(
      loc, result_types, args, b);
}

template <>
inline Value MapLhloOpToStdScalarOp<xla_lhlo::SelectOp>(
    Location loc, ArrayRef<Type> result_types, ArrayRef<Value> args,
    OpBuilder* b) {
  return MapLhloOpToStdScalarOpImpl<::mlir::SelectOp>{}(loc, result_types, args,
                                                        b);
}

template <>
inline Value MapLhloOpToStdScalarOp<xla_lhlo::SignOp>(
    Location loc, ArrayRef<Type> result_types, ArrayRef<Value> args,
    OpBuilder* b) {
  Type element_type = args.front().getType();
  if (element_type.isa<FloatType>()) {
    FloatType float_type = element_type.cast<FloatType>();
    APFloat const_value = float_type.isF32() ? APFloat(1.0f) : APFloat(1.0);
    Value one = b->create<mlir::ConstantFloatOp>(loc, const_value, float_type);
    return b->create<::mlir::CopySignOp>(loc, result_types, one, args[0]);
  }
  return nullptr;
}

template <>
inline Value MapLhloOpToStdScalarOp<xla_lhlo::SqrtOp>(
    Location loc, ArrayRef<Type> result_types, ArrayRef<Value> args,
    OpBuilder* b) {
  return MapLhloOpToStdScalarOpImpl<FloatType, ::mlir::SqrtOp>{}(
      loc, result_types, args, b);
}

template <>
inline Value MapLhloOpToStdScalarOp<xla_lhlo::TanhOp>(
    Location loc, ArrayRef<Type> result_types, ArrayRef<Value> args,
    OpBuilder* b) {
  return MapLhloOpToStdScalarOpImpl<FloatType, ::mlir::TanhOp>{}(
      loc, result_types, args, b);
}

}  // namespace impl

struct XlaOpToStdScalarOp {
  // Implementation for LHLO ops except xla_lhlo::CompareOp.
  template <typename XlaOpTy, typename LhloOpTy = XlaOpTy,
            typename = std::enable_if_t<
                !std::is_same<LhloOpTy, xla_lhlo::CompareOp>::value &&
                std::is_same<typename xla_hlo::HloToLhloOp<LhloOpTy>,
                             std::false_type>::value>>
  static Value map(XlaOpTy op, ArrayRef<Type> result_types,
                   ArrayRef<Value> args, OpBuilder* b, unsigned i = 0) {
    return impl::MapLhloOpToStdScalarOp<LhloOpTy>(op.getLoc(), result_types,
                                                  args, b);
  }

  // Implementation for HLO ops except xla_hlo::CompareOp.
  template <typename XlaOpTy, typename LhloOpTy = xla_hlo::HloToLhloOp<XlaOpTy>,
            typename = std::enable_if_t<
                !std::is_same<LhloOpTy, xla_lhlo::CompareOp>::value &&
                !std::is_same<LhloOpTy, std::false_type>::value>>
  static Value map(XlaOpTy op, ArrayRef<Type> result_types,
                   ArrayRef<Value> args, OpBuilder* b, int i = 0) {
    return impl::MapLhloOpToStdScalarOp<LhloOpTy>(op.getLoc(), result_types,
                                                  args, b);
  }

  // Implementation for xla_lhlo::CompareOp.
  template <typename LhloOpTy, typename = std::enable_if_t<std::is_same<
                                   LhloOpTy, xla_lhlo::CompareOp>::value>>
  static Value map(xla_lhlo::CompareOp op, ArrayRef<Type> result_types,
                   ArrayRef<Value> args, OpBuilder* b) {
    auto comparison_direction = op.comparison_direction();
    return impl::MapXlaCompareOpToStdScalarOp<xla_lhlo::CompareOp>(
        op.getLoc(), comparison_direction, result_types, args, b);
  }

  // Implementation for xla_hlo::CompareOp.
  template <typename HloOpTy, typename = std::enable_if_t<std::is_same<
                                  HloOpTy, xla_hlo::CompareOp>::value>>
  static Value map(xla_hlo::CompareOp op, ArrayRef<Type> result_types,
                   ArrayRef<Value> args, OpBuilder* b) {
    auto comparison_direction = op.comparison_direction();
    return impl::MapXlaCompareOpToStdScalarOp<xla_lhlo::CompareOp>(
        op.getLoc(), comparison_direction, result_types, args, b);
  }
};

}  // namespace xla_lhlo
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_XLA_TRANSFORMS_MAP_XLA_TO_SCALAR_OP_H_
