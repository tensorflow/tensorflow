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
struct ScalarOp<xla_hlo::DivOp> {
  using FOp = ::mlir::DivFOp;
  using IOp = ::mlir::SignedDivIOp;
};
template <>
struct ScalarOp<xla_lhlo::MulOp> {
  using FOp = ::mlir::MulFOp;
  using IOp = ::mlir::MulIOp;
};
template <>
struct ScalarOp<xla_hlo::MulOp> {
  using FOp = ::mlir::MulFOp;
  using IOp = ::mlir::MulIOp;
};
template <>
struct ScalarOp<xla_lhlo::RemOp> {
  using FOp = ::mlir::RemFOp;
  using IOp = ::mlir::SignedRemIOp;
};
template <>
struct ScalarOp<xla_hlo::RemOp> {
  using FOp = ::mlir::RemFOp;
  using IOp = ::mlir::SignedRemIOp;
};
template <>
struct ScalarOp<xla_lhlo::SubOp> {
  using FOp = ::mlir::SubFOp;
  using IOp = ::mlir::SubIOp;
};
template <>
struct ScalarOp<xla_hlo::SubOp> {
  using FOp = ::mlir::SubFOp;
  using IOp = ::mlir::SubIOp;
};

template <typename LHLO_BinaryOp>
using ScalarFOp = typename ScalarOp<LHLO_BinaryOp>::FOp;
template <typename LHLO_BinaryOp>
using ScalarIOp = typename ScalarOp<LHLO_BinaryOp>::IOp;

template <typename... Args>
struct MapXlaOpToStdScalarOpImpl {
  Value operator()(Location loc, ArrayRef<Type> result_types,
                   ArrayRef<Value> args, OpBuilder* b) {
    return nullptr;
  }
};

template <typename SupportedType, typename StdScalarOp, typename... Args>
struct MapXlaOpToStdScalarOpImpl<SupportedType, StdScalarOp, Args...> {
  Value operator()(Location loc, ArrayRef<Type> result_types,
                   ArrayRef<Value> args, OpBuilder* b) {
    Type element_type = args.front().getType();
    if (element_type.isa<SupportedType>()) {
      return b->template create<StdScalarOp>(loc, result_types, args,
                                             mlir::None);
    }
    return MapXlaOpToStdScalarOpImpl<Args...>{}(loc, result_types, args, b);
  }
};

template <typename XlaOp>
inline Value MapXlaOpToStdScalarOp(XlaOp xla_op, ArrayRef<Type> result_types,
                                   ArrayRef<Value> args, OpBuilder* b) {
  return MapXlaOpToStdScalarOpImpl<IntegerType, ScalarIOp<XlaOp>, FloatType,
                                   ScalarFOp<XlaOp>>{}(xla_op.getLoc(),
                                                       result_types, args, b);
}

// TODO(ravishankarm): Find a way to reduce code-bloat in HLO and LHLO
// specialization.
template <>
inline Value MapXlaOpToStdScalarOp<xla_lhlo::AbsOp>(xla_lhlo::AbsOp xla_op,
                                                    ArrayRef<Type> result_types,
                                                    ArrayRef<Value> args,
                                                    OpBuilder* b) {
  return MapXlaOpToStdScalarOpImpl<FloatType, ::mlir::AbsFOp>{}(
      xla_op.getLoc(), result_types, args, b);
}
template <>
inline Value MapXlaOpToStdScalarOp<xla_hlo::AbsOp>(xla_hlo::AbsOp xla_op,
                                                   ArrayRef<Type> result_types,
                                                   ArrayRef<Value> args,
                                                   OpBuilder* b) {
  return MapXlaOpToStdScalarOpImpl<FloatType, ::mlir::AbsFOp>{}(
      xla_op.getLoc(), result_types, args, b);
}

template <>
inline Value MapXlaOpToStdScalarOp<xla_lhlo::AndOp>(xla_lhlo::AndOp xla_op,
                                                    ArrayRef<Type> result_types,
                                                    ArrayRef<Value> args,
                                                    OpBuilder* b) {
  return MapXlaOpToStdScalarOpImpl<IntegerType, ::mlir::AndOp>{}(
      xla_op.getLoc(), result_types, args, b);
}
template <>
inline Value MapXlaOpToStdScalarOp<xla_hlo::AndOp>(xla_hlo::AndOp xla_op,
                                                   ArrayRef<Type> result_types,
                                                   ArrayRef<Value> args,
                                                   OpBuilder* b) {
  return MapXlaOpToStdScalarOpImpl<IntegerType, ::mlir::AndOp>{}(
      xla_op.getLoc(), result_types, args, b);
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
inline Value MapXlaOpToStdScalarOp<xla_lhlo::CompareOp>(
    xla_lhlo::CompareOp xla_op, ArrayRef<Type> result_types,
    ArrayRef<Value> args, OpBuilder* b) {
  const auto& lhs = args[0];
  const auto& rhs = args[1];
  Type element_type = lhs.getType();
  if (element_type.isa<IntegerType>()) {
    Optional<CmpIPredicate> predicate =
        getIntCmpPredicate(xla_op.comparison_direction());
    assert(predicate.hasValue() && "expected valid comparison direction");
    return b->create<ScalarIOp<xla_lhlo::CompareOp>>(
        xla_op.getLoc(), predicate.getValue(), lhs, rhs);
  }
  if (element_type.isa<FloatType>()) {
    return b->create<ScalarFOp<xla_lhlo::CompareOp>>(
        xla_op.getLoc(), getFloatCmpPredicate(xla_op.comparison_direction()),
        lhs, rhs);
  }
  return nullptr;
}

template <>
inline Value MapXlaOpToStdScalarOp<xla_lhlo::CopyOp>(
    xla_lhlo::CopyOp xla_op, ArrayRef<Type> result_types, ArrayRef<Value> args,
    OpBuilder* b) {
  return args.front();
}

template <>
inline Value MapXlaOpToStdScalarOp<xla_lhlo::ExpOp>(xla_lhlo::ExpOp xla_op,
                                                    ArrayRef<Type> result_types,
                                                    ArrayRef<Value> args,
                                                    OpBuilder* b) {
  return MapXlaOpToStdScalarOpImpl<FloatType, ::mlir::ExpOp>{}(
      xla_op.getLoc(), result_types, args, b);
}
template <>
inline Value MapXlaOpToStdScalarOp<xla_hlo::ExpOp>(xla_hlo::ExpOp xla_op,
                                                   ArrayRef<Type> result_types,
                                                   ArrayRef<Value> args,
                                                   OpBuilder* b) {
  return MapXlaOpToStdScalarOpImpl<FloatType, ::mlir::ExpOp>{}(
      xla_op.getLoc(), result_types, args, b);
}

template <>
inline Value MapXlaOpToStdScalarOp<xla_lhlo::CeilOp>(
    xla_lhlo::CeilOp xla_op, ArrayRef<Type> result_types, ArrayRef<Value> args,
    OpBuilder* b) {
  return MapXlaOpToStdScalarOpImpl<FloatType, ::mlir::CeilFOp>{}(
      xla_op.getLoc(), result_types, args, b);
}
template <>
inline Value MapXlaOpToStdScalarOp<xla_hlo::CeilOp>(xla_hlo::CeilOp xla_op,
                                                    ArrayRef<Type> result_types,
                                                    ArrayRef<Value> args,
                                                    OpBuilder* b) {
  return MapXlaOpToStdScalarOpImpl<FloatType, ::mlir::CeilFOp>{}(
      xla_op.getLoc(), result_types, args, b);
}

template <>
inline Value MapXlaOpToStdScalarOp<xla_lhlo::ConvertOp>(
    xla_lhlo::ConvertOp xla_op, ArrayRef<Type> result_types,
    ArrayRef<Value> args, OpBuilder* b) {
  const Type& sourceType = args.front().getType();
  const Type& targetType = result_types.front();

  if (mlir::SIToFPOp::areCastCompatible(sourceType, targetType)) {
    return b->create<mlir::SIToFPOp>(xla_op.getLoc(), result_types, args,
                                     mlir::None);
  } else if (sourceType.isa<FloatType>() && targetType.isa<FloatType>()) {
    FloatType src = sourceType.cast<FloatType>();
    FloatType res = targetType.cast<FloatType>();
    if (src.getWidth() > res.getWidth()) {
      return b->create<mlir::FPTruncOp>(xla_op.getLoc(), result_types, args,
                                        mlir::None);
    } else if (src.getWidth() < res.getWidth()) {
      return b->create<mlir::FPExtOp>(xla_op.getLoc(), result_types, args,
                                      mlir::None);
    }
    // No conversion is needed for the same width floats
    return args.front();
  }
  if (sourceType.isa<IntegerType>() && targetType.isa<IntegerType>()) {
    IntegerType src = sourceType.cast<IntegerType>();
    IntegerType res = targetType.cast<IntegerType>();
    if (src.getWidth() > res.getWidth()) {
      return b->create<mlir::TruncateIOp>(xla_op.getLoc(), result_types, args,
                                          mlir::None);
    } else if (src.getWidth() < res.getWidth()) {
      return b->create<mlir::ZeroExtendIOp>(xla_op.getLoc(), result_types, args,
                                            mlir::None);
    }
    // No conversion is needed for the same width integers
    return args.front();
  }
  // TODO(dfki-ehna): Add other primitive type conversions
  // if (mlir::FpToSiOp::areCastCompatible(sourceType, targetType)) {
  //   return b.create<mlir::FpToSiOp>(xla_op.getLoc(), result_types,
  //   args,mlir::None);
  // }

  return nullptr;
}

template <>
inline Value MapXlaOpToStdScalarOp<xla_lhlo::CosOp>(xla_lhlo::CosOp xla_op,
                                                    ArrayRef<Type> result_types,
                                                    ArrayRef<Value> args,
                                                    OpBuilder* b) {
  return MapXlaOpToStdScalarOpImpl<FloatType, ::mlir::CosOp>{}(
      xla_op.getLoc(), result_types, args, b);
}
template <>
inline Value MapXlaOpToStdScalarOp<xla_hlo::CosOp>(xla_hlo::CosOp xla_op,
                                                   ArrayRef<Type> result_types,
                                                   ArrayRef<Value> args,
                                                   OpBuilder* b) {
  return MapXlaOpToStdScalarOpImpl<FloatType, ::mlir::CosOp>{}(
      xla_op.getLoc(), result_types, args, b);
}

template <>
inline Value MapXlaOpToStdScalarOp<xla_lhlo::MaxOp>(xla_lhlo::MaxOp xla_op,
                                                    ArrayRef<Type> result_types,
                                                    ArrayRef<Value> args,
                                                    OpBuilder* b) {
  const auto& lhs = args[0];
  const auto& rhs = args[1];
  Type element_type = lhs.getType();
  if (element_type.isa<IntegerType>()) {
    auto lhs_gt_rhs = b->create<ScalarIOp<xla_lhlo::CompareOp>>(
        xla_op.getLoc(), CmpIPredicate::sgt, lhs, rhs);
    return b->create<::mlir::SelectOp>(xla_op.getLoc(), lhs_gt_rhs, lhs, rhs);
  }
  if (element_type.isa<FloatType>()) {
    auto lhs_gt_rhs = b->create<ScalarFOp<xla_lhlo::CompareOp>>(
        xla_op.getLoc(), CmpFPredicate::OGT, lhs, rhs);
    return b->create<::mlir::SelectOp>(xla_op.getLoc(), lhs_gt_rhs, lhs, rhs);
  }
  return nullptr;
}

template <>
inline Value MapXlaOpToStdScalarOp<xla_lhlo::MinOp>(xla_lhlo::MinOp xla_op,
                                                    ArrayRef<Type> result_types,
                                                    ArrayRef<Value> args,
                                                    OpBuilder* b) {
  const auto& lhs = args[0];
  const auto& rhs = args[1];
  Type element_type = lhs.getType();
  if (element_type.isa<IntegerType>()) {
    auto lhs_lt_rhs = b->create<ScalarIOp<xla_lhlo::CompareOp>>(
        xla_op.getLoc(), CmpIPredicate::slt, lhs, rhs);
    return b->create<::mlir::SelectOp>(xla_op.getLoc(), lhs_lt_rhs, lhs, rhs);
  }
  if (element_type.isa<FloatType>()) {
    auto lhs_lt_rhs = b->create<ScalarFOp<xla_lhlo::CompareOp>>(
        xla_op.getLoc(), CmpFPredicate::OLT, lhs, rhs);
    return b->create<::mlir::SelectOp>(xla_op.getLoc(), lhs_lt_rhs, lhs, rhs);
  }
  return nullptr;
}

template <>
inline Value MapXlaOpToStdScalarOp<xla_lhlo::NegOp>(xla_lhlo::NegOp xla_op,
                                                    ArrayRef<Type> result_types,
                                                    ArrayRef<Value> args,
                                                    OpBuilder* b) {
  return MapXlaOpToStdScalarOpImpl<FloatType, ::mlir::NegFOp>{}(
      xla_op.getLoc(), result_types, args, b);
}
template <>
inline Value MapXlaOpToStdScalarOp<xla_hlo::NegOp>(xla_hlo::NegOp xla_op,
                                                   ArrayRef<Type> result_types,
                                                   ArrayRef<Value> args,
                                                   OpBuilder* b) {
  return MapXlaOpToStdScalarOpImpl<FloatType, ::mlir::NegFOp>{}(
      xla_op.getLoc(), result_types, args, b);
}

template <>
inline Value MapXlaOpToStdScalarOp<xla_lhlo::SelectOp>(
    xla_lhlo::SelectOp xla_op, ArrayRef<Type> result_types,
    ArrayRef<Value> args, OpBuilder* b) {
  return b->create<::mlir::SelectOp>(xla_op.getLoc(), result_types, args,
                                     mlir::None);
}

template <>
inline Value MapXlaOpToStdScalarOp<xla_lhlo::SignOp>(
    xla_lhlo::SignOp xla_op, ArrayRef<Type> result_types, ArrayRef<Value> args,
    OpBuilder* b) {
  Type element_type = args.front().getType();
  if (element_type.isa<FloatType>()) {
    FloatType float_type = element_type.cast<FloatType>();
    APFloat const_value = float_type.isF32() ? APFloat(1.0f) : APFloat(1.0);
    Value one = b->create<mlir::ConstantFloatOp>(xla_op.getLoc(), const_value,
                                                 float_type);
    return b->create<::mlir::CopySignOp>(xla_op.getLoc(), result_types, one,
                                         args[0]);
  }
  return nullptr;
}

template <>
inline Value MapXlaOpToStdScalarOp<xla_lhlo::TanhOp>(
    xla_lhlo::TanhOp xla_op, ArrayRef<Type> result_types, ArrayRef<Value> args,
    OpBuilder* b) {
  return MapXlaOpToStdScalarOpImpl<FloatType, ::mlir::TanhOp>{}(
      xla_op.getLoc(), result_types, args, b);
}
template <>
inline Value MapXlaOpToStdScalarOp<xla_hlo::TanhOp>(xla_hlo::TanhOp xla_op,
                                                    ArrayRef<Type> result_types,
                                                    ArrayRef<Value> args,
                                                    OpBuilder* b) {
  return MapXlaOpToStdScalarOpImpl<FloatType, ::mlir::TanhOp>{}(
      xla_op.getLoc(), result_types, args, b);
}

}  // namespace xla_lhlo
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_XLA_TRANSFORMS_MAP_XLA_TO_SCALAR_OP_H_
