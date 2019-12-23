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

#ifndef TENSORFLOW_COMPILER_MLIR_XLA_TRANSFORMS_MAP_LHLO_TO_SCALAR_OP_H_
#define TENSORFLOW_COMPILER_MLIR_XLA_TRANSFORMS_MAP_LHLO_TO_SCALAR_OP_H_

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "mlir/Dialect/StandardOps/Ops.h"  // TF:local_config_mlir
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
struct ScalarOp<xla_lhlo::SubOp> {
  using FOp = ::mlir::SubFOp;
  using IOp = ::mlir::SubIOp;
};

template <typename LHLO_BinaryOp>
using ScalarFOp = typename ScalarOp<LHLO_BinaryOp>::FOp;
template <typename LHLO_BinaryOp>
using ScalarIOp = typename ScalarOp<LHLO_BinaryOp>::IOp;

template <typename LhloOp>
Operation* MapLhloOpToStdScalarOp(LhloOp lhlo_op, ArrayRef<Type> result_types,
                                  ArrayRef<Value*> block_args, OpBuilder b) {
  Type element_type = block_args.front()->getType();
  if (element_type.isa<IntegerType>()) {
    return b.template create<ScalarIOp<LhloOp>>(lhlo_op.getLoc(), result_types,
                                                block_args, mlir::None);
  }
  if (element_type.isa<FloatType>()) {
    return b.template create<ScalarFOp<LhloOp>>(lhlo_op.getLoc(), result_types,
                                                block_args, mlir::None);
  }
  return nullptr;
}

template <>
inline Operation* MapLhloOpToStdScalarOp<xla_lhlo::MaxOp>(
    xla_lhlo::MaxOp lhlo_op, ArrayRef<Type> result_types,
    ArrayRef<Value*> block_args, OpBuilder b) {
  const auto& lhs = block_args[0];
  const auto& rhs = block_args[1];
  Type element_type = lhs->getType();
  if (element_type.isa<IntegerType>()) {
    auto lhs_gt_rhs = b.create<ScalarIOp<CompareOp>>(
        lhlo_op.getLoc(), CmpIPredicate::sgt, lhs, rhs);
    return b.create<::mlir::SelectOp>(lhlo_op.getLoc(), lhs_gt_rhs, lhs, rhs);
  }
  if (element_type.isa<FloatType>()) {
    auto lhs_gt_rhs = b.create<ScalarFOp<CompareOp>>(
        lhlo_op.getLoc(), CmpFPredicate::OGT, lhs, rhs);
    return b.create<::mlir::SelectOp>(lhlo_op.getLoc(), lhs_gt_rhs, lhs, rhs);
  }
  return nullptr;
}

template <>
inline Operation* MapLhloOpToStdScalarOp<xla_lhlo::MinOp>(
    xla_lhlo::MinOp lhlo_op, ArrayRef<Type> result_types,
    ArrayRef<Value*> block_args, OpBuilder b) {
  const auto& lhs = block_args[0];
  const auto& rhs = block_args[1];
  Type element_type = lhs->getType();
  if (element_type.isa<IntegerType>()) {
    auto lhs_lt_rhs = b.create<ScalarIOp<CompareOp>>(
        lhlo_op.getLoc(), CmpIPredicate::slt, lhs, rhs);
    return b.create<::mlir::SelectOp>(lhlo_op.getLoc(), lhs_lt_rhs, lhs, rhs);
  }
  if (element_type.isa<FloatType>()) {
    auto lhs_lt_rhs = b.create<ScalarFOp<CompareOp>>(
        lhlo_op.getLoc(), CmpFPredicate::OLT, lhs, rhs);
    return b.create<::mlir::SelectOp>(lhlo_op.getLoc(), lhs_lt_rhs, lhs, rhs);
  }
  return nullptr;
}

template <>
inline Operation* MapLhloOpToStdScalarOp<xla_lhlo::AndOp>(
    xla_lhlo::AndOp lhlo_op, ArrayRef<Type> result_types,
    ArrayRef<Value*> block_args, OpBuilder b) {
  Type element_type = block_args.front()->getType();
  return element_type.isa<IntegerType>()
             ? b.create<::mlir::AndOp>(lhlo_op.getLoc(), result_types,
                                       block_args, mlir::None)
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
inline Operation* MapLhloOpToStdScalarOp<xla_lhlo::CompareOp>(
    xla_lhlo::CompareOp lhlo_op, ArrayRef<Type> result_types,
    ArrayRef<Value*> block_args, OpBuilder b) {
  const auto& lhs = block_args[0];
  const auto& rhs = block_args[1];
  Type element_type = lhs->getType();
  if (element_type.isa<IntegerType>()) {
    Optional<CmpIPredicate> predicate =
        getIntCmpPredicate(lhlo_op.comparison_direction());
    assert(predicate.hasValue() && "expected valid comparison direction");
    return b.create<ScalarIOp<CompareOp>>(lhlo_op.getLoc(),
                                          predicate.getValue(), lhs, rhs);
  }
  if (element_type.isa<FloatType>()) {
    return b.create<ScalarFOp<CompareOp>>(
        lhlo_op.getLoc(), getFloatCmpPredicate(lhlo_op.comparison_direction()),
        lhs, rhs);
  }
  return nullptr;
}

template <>
inline Operation* MapLhloOpToStdScalarOp<xla_lhlo::SelectOp>(
    xla_lhlo::SelectOp lhlo_op, ArrayRef<Type> result_types,
    ArrayRef<Value*> block_args, OpBuilder b) {
  return b.create<::mlir::SelectOp>(lhlo_op.getLoc(), result_types, block_args,
                                    mlir::None);
}

template <>
inline Operation* MapLhloOpToStdScalarOp<xla_lhlo::ExpOp>(
    xla_lhlo::ExpOp lhlo_op, ArrayRef<Type> result_types,
    ArrayRef<Value*> block_args, OpBuilder b) {
  Type element_type = block_args.front()->getType();
  return element_type.isa<FloatType>()
             ? b.create<::mlir::ExpOp>(lhlo_op.getLoc(), result_types,
                                       block_args, mlir::None)
             : nullptr;
}

}  // namespace xla_lhlo
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_XLA_TRANSFORMS_MAP_LHLO_TO_SCALAR_OP_H_
