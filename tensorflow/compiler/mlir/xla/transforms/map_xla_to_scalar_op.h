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
#include "mlir/Dialect/StandardOps/Ops.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"
#include "tensorflow/compiler/mlir/xla/ir/lhlo_ops.h"

namespace mlir {
namespace xla_lhlo {

template <typename LHLO_BinaryOp>
struct ScalarOp;

template <>
struct ScalarOp<xla_lhlo::AddOp> {
  using FOp = ::mlir::AddFOp;
  using IOp = ::mlir::AddIOp;
};
template <>
struct ScalarOp<xla_hlo::AddOp> {
  using FOp = ::mlir::AddFOp;
  using IOp = ::mlir::AddIOp;
};
template <>
struct ScalarOp<xla_lhlo::CompareOp> {
  using FOp = ::mlir::CmpFOp;
  using IOp = ::mlir::CmpIOp;
};
template <>
struct ScalarOp<xla_lhlo::DivOp> {
  using FOp = ::mlir::DivFOp;
  using IOp = ::mlir::SignedDivIOp;
};
template <>
struct ScalarOp<xla_lhlo::MulOp> {
  using FOp = ::mlir::MulFOp;
  using IOp = ::mlir::MulIOp;
};
template <>
struct ScalarOp<xla_lhlo::RemOp> {
  using FOp = ::mlir::RemFOp;
  using IOp = ::mlir::SignedRemIOp;
};
template <>
struct ScalarOp<xla_lhlo::SubOp> {
  using FOp = ::mlir::SubFOp;
  using IOp = ::mlir::SubIOp;
};

template <typename LHLO_BinaryOp>
using ScalarFOp = typename ScalarOp<LHLO_BinaryOp>::FOp;
template <typename LHLO_BinaryOp>
using ScalarIOp = typename ScalarOp<LHLO_BinaryOp>::IOp;

template <typename LhloOp>
Value MapLhloOpToStdScalarOp(LhloOp lhlo_op, ArrayRef<Type> result_types,
                             ArrayRef<Value> args, OpBuilder* b) {
  Type element_type = args.front().getType();
  if (element_type.isa<IntegerType>()) {
    return b->template create<ScalarIOp<LhloOp>>(lhlo_op.getLoc(), result_types,
                                                 args, mlir::None);
  }
  if (element_type.isa<FloatType>()) {
    return b->template create<ScalarFOp<LhloOp>>(lhlo_op.getLoc(), result_types,
                                                 args, mlir::None);
  }
  return nullptr;
}

template <>
inline Value MapLhloOpToStdScalarOp<xla_lhlo::AbsOp>(
    xla_lhlo::AbsOp lhlo_op, ArrayRef<Type> result_types, ArrayRef<Value> args,
    OpBuilder* b) {
  Type element_type = args.front().getType();
  if (element_type.isa<FloatType>()) {
    return b->create<::mlir::AbsFOp>(lhlo_op.getLoc(), result_types, args,
                                     mlir::None);
  }
  return nullptr;
}

template <>
inline Value MapLhloOpToStdScalarOp<xla_lhlo::AndOp>(
    xla_lhlo::AndOp lhlo_op, ArrayRef<Type> result_types, ArrayRef<Value> args,
    OpBuilder* b) {
  Type element_type = args.front().getType();
  return element_type.isa<IntegerType>()
             ? b->create<::mlir::AndOp>(lhlo_op.getLoc(), result_types, args,
                                        mlir::None)
             : nullptr;
}

inline CmpFPredicate getFloatCmpPredicate(StringRef xla_comparison_direction) {
  return llvm::StringSwitch<CmpFPredicate>(xla_comparison_direction)
      .Case("EQ", CmpFPredicate::OEQ)
      .Case("NE", CmpFPredicate::ONE)
      .Case("GE", CmpFPredicate::OGE)
      .Case("GT", CmpFPredicate::OGT)
      .Case("LE", CmpFPredicate::OLE)
      .Case("LT", CmpFPredicate::OLT)
      .Default(CmpFPredicate::NumPredicates);
}

inline Optional<CmpIPredicate> getIntCmpPredicate(
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

template <>
inline Value MapLhloOpToStdScalarOp<xla_lhlo::CompareOp>(
    xla_lhlo::CompareOp lhlo_op, ArrayRef<Type> result_types,
    ArrayRef<Value> args, OpBuilder* b) {
  const auto& lhs = args[0];
  const auto& rhs = args[1];
  Type element_type = lhs.getType();
  if (element_type.isa<IntegerType>()) {
    Optional<CmpIPredicate> predicate =
        getIntCmpPredicate(lhlo_op.comparison_direction());
    assert(predicate.hasValue() && "expected valid comparison direction");
    return b->create<ScalarIOp<CompareOp>>(lhlo_op.getLoc(),
                                           predicate.getValue(), lhs, rhs);
  }
  if (element_type.isa<FloatType>()) {
    return b->create<ScalarFOp<CompareOp>>(
        lhlo_op.getLoc(), getFloatCmpPredicate(lhlo_op.comparison_direction()),
        lhs, rhs);
  }
  return nullptr;
}

template <>
inline Value MapLhloOpToStdScalarOp<xla_lhlo::CopyOp>(
    xla_lhlo::CopyOp lhlo_op, ArrayRef<Type> result_types, ArrayRef<Value> args,
    OpBuilder* b) {
  return args.front();
}

template <>
inline Value MapLhloOpToStdScalarOp<xla_lhlo::ExpOp>(
    xla_lhlo::ExpOp lhlo_op, ArrayRef<Type> result_types, ArrayRef<Value> args,
    OpBuilder* b) {
  Type element_type = args.front().getType();
  return element_type.isa<FloatType>()
             ? b->create<::mlir::ExpOp>(lhlo_op.getLoc(), result_types, args,
                                        mlir::None)
             : nullptr;
}

template <>
inline Value MapLhloOpToStdScalarOp<xla_lhlo::CeilOp>(
    xla_lhlo::CeilOp lhlo_op, ArrayRef<Type> result_types, ArrayRef<Value> args,
    OpBuilder* b) {
  Type element_type = args.front().getType();
  if (element_type.isa<FloatType>()) {
    return b->create<::mlir::CeilFOp>(lhlo_op.getLoc(), result_types, args,
                                      mlir::None);
  }
  return nullptr;
}

template <>
inline Value MapLhloOpToStdScalarOp<xla_lhlo::ConvertOp>(
    xla_lhlo::ConvertOp lhlo_op, ArrayRef<Type> result_types,
    ArrayRef<Value> args, OpBuilder* b) {
  const Type& sourceType = args.front().getType();
  const Type& targetType = result_types.front();

  if (mlir::SIToFPOp::areCastCompatible(sourceType, targetType)) {
    return b->create<mlir::SIToFPOp>(lhlo_op.getLoc(), result_types, args,
                                     mlir::None);
  } else if (sourceType.isa<FloatType>() && targetType.isa<FloatType>()) {
    FloatType src = sourceType.cast<FloatType>();
    FloatType res = targetType.cast<FloatType>();
    if (src.getWidth() > res.getWidth()) {
      return b->create<mlir::FPTruncOp>(lhlo_op.getLoc(), result_types, args,
                                        mlir::None);
    } else if (src.getWidth() < res.getWidth()) {
      return b->create<mlir::FPExtOp>(lhlo_op.getLoc(), result_types, args,
                                      mlir::None);
    }
    // No conversion is needed for the same width floats
    return args.front();
  }
  if (sourceType.isa<IntegerType>() && targetType.isa<IntegerType>()) {
    IntegerType src = sourceType.cast<IntegerType>();
    IntegerType res = targetType.cast<IntegerType>();
    if (src.getWidth() > res.getWidth()) {
      return b->create<mlir::TruncateIOp>(lhlo_op.getLoc(), result_types, args,
                                          mlir::None);
    } else if (src.getWidth() < res.getWidth()) {
      return b->create<mlir::ZeroExtendIOp>(lhlo_op.getLoc(), result_types,
                                            args, mlir::None);
    }
    // No conversion is needed for the same width integers
    return args.front();
  }
  // TODO(dfki-ehna): Add other primitive type conversions
  // if (mlir::FpToSiOp::areCastCompatible(sourceType, targetType)) {
  //   return b.create<mlir::FpToSiOp>(lhlo_op.getLoc(), result_types,
  //   args,mlir::None);
  // }

  return nullptr;
}

template <>
inline Value MapLhloOpToStdScalarOp<xla_lhlo::CosOp>(
    xla_lhlo::CosOp lhlo_op, ArrayRef<Type> result_types, ArrayRef<Value> args,
    OpBuilder* b) {
  Type element_type = args.front().getType();
  if (element_type.isa<FloatType>()) {
    return b->create<::mlir::CosOp>(lhlo_op.getLoc(), result_types, args,
                                    mlir::None);
  }
  return nullptr;
}

template <>
inline Value MapLhloOpToStdScalarOp<xla_lhlo::MaxOp>(
    xla_lhlo::MaxOp lhlo_op, ArrayRef<Type> result_types, ArrayRef<Value> args,
    OpBuilder* b) {
  const auto& lhs = args[0];
  const auto& rhs = args[1];
  Type element_type = lhs.getType();
  if (element_type.isa<IntegerType>()) {
    auto lhs_gt_rhs = b->create<ScalarIOp<CompareOp>>(
        lhlo_op.getLoc(), CmpIPredicate::sgt, lhs, rhs);
    return b->create<::mlir::SelectOp>(lhlo_op.getLoc(), lhs_gt_rhs, lhs, rhs);
  }
  if (element_type.isa<FloatType>()) {
    auto lhs_gt_rhs = b->create<ScalarFOp<CompareOp>>(
        lhlo_op.getLoc(), CmpFPredicate::OGT, lhs, rhs);
    return b->create<::mlir::SelectOp>(lhlo_op.getLoc(), lhs_gt_rhs, lhs, rhs);
  }
  return nullptr;
}

template <>
inline Value MapLhloOpToStdScalarOp<xla_lhlo::MinOp>(
    xla_lhlo::MinOp lhlo_op, ArrayRef<Type> result_types, ArrayRef<Value> args,
    OpBuilder* b) {
  const auto& lhs = args[0];
  const auto& rhs = args[1];
  Type element_type = lhs.getType();
  if (element_type.isa<IntegerType>()) {
    auto lhs_lt_rhs = b->create<ScalarIOp<CompareOp>>(
        lhlo_op.getLoc(), CmpIPredicate::slt, lhs, rhs);
    return b->create<::mlir::SelectOp>(lhlo_op.getLoc(), lhs_lt_rhs, lhs, rhs);
  }
  if (element_type.isa<FloatType>()) {
    auto lhs_lt_rhs = b->create<ScalarFOp<CompareOp>>(
        lhlo_op.getLoc(), CmpFPredicate::OLT, lhs, rhs);
    return b->create<::mlir::SelectOp>(lhlo_op.getLoc(), lhs_lt_rhs, lhs, rhs);
  }
  return nullptr;
}

template <>
inline Value MapLhloOpToStdScalarOp<xla_lhlo::NegOp>(
    xla_lhlo::NegOp lhlo_op, ArrayRef<Type> result_types, ArrayRef<Value> args,
    OpBuilder* b) {
  Type element_type = args.front().getType();
  if (element_type.isa<FloatType>()) {
    return b->create<::mlir::NegFOp>(lhlo_op.getLoc(), result_types, args,
                                     mlir::None);
  }
  return nullptr;
}

template <>
inline Value MapLhloOpToStdScalarOp<xla_lhlo::SelectOp>(
    xla_lhlo::SelectOp lhlo_op, ArrayRef<Type> result_types,
    ArrayRef<Value> args, OpBuilder* b) {
  return b->create<::mlir::SelectOp>(lhlo_op.getLoc(), result_types, args,
                                     mlir::None);
}

template <>
inline Value MapLhloOpToStdScalarOp<xla_lhlo::SignOp>(
    xla_lhlo::SignOp lhlo_op, ArrayRef<Type> result_types, ArrayRef<Value> args,
    OpBuilder* b) {
  Type element_type = args.front().getType();
  if (element_type.isa<FloatType>()) {
    FloatType float_type = element_type.cast<FloatType>();
    APFloat const_value = float_type.isF32() ? APFloat(1.0f) : APFloat(1.0);
    Value one = b->create<mlir::ConstantFloatOp>(lhlo_op.getLoc(), const_value,
                                                 float_type);
    return b->create<::mlir::CopySignOp>(lhlo_op.getLoc(), result_types, one,
                                         args[0]);
  }
  return nullptr;
}

template <>
inline Value MapLhloOpToStdScalarOp<xla_lhlo::TanhOp>(
    xla_lhlo::TanhOp lhlo_op, ArrayRef<Type> result_types, ArrayRef<Value> args,
    OpBuilder* b) {
  Type element_type = args.front().getType();
  if (element_type.isa<FloatType>()) {
    return b->create<::mlir::TanhOp>(lhlo_op.getLoc(), result_types, args,
                                     mlir::None);
  }
  return nullptr;
}

}  // namespace xla_lhlo
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_XLA_TRANSFORMS_MAP_XLA_TO_SCALAR_OP_H_
