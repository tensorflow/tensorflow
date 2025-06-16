/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/codegen/emitters/implicit_arith_op_builder.h"

#include <cstdint>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Value.h"

namespace xla::emitters {

ImplicitArithOpBuilder::ImplicitArithOpBuilder(
    mlir::Value value, mlir::ImplicitLocOpBuilder* builder)
    : value_(value), builder_(builder) {}

mlir::Value ImplicitArithOpBuilder::value() const { return value_; }
ImplicitArithOpBuilder::operator mlir::Value() const { return value_; }

ImplicitArithOpBuilder ImplicitArithOpBuilder::operator+(int64_t rhs) const {
  return Binop<mlir::arith::AddIOp>(rhs);
}

ImplicitArithOpBuilder ImplicitArithOpBuilder::operator+(
    mlir::Value rhs) const {
  return Binop<mlir::arith::AddIOp>(rhs);
}

ImplicitArithOpBuilder ImplicitArithOpBuilder::operator-(int64_t rhs) const {
  return Binop<mlir::arith::SubIOp>(rhs);
}

ImplicitArithOpBuilder ImplicitArithOpBuilder::operator-(
    mlir::Value rhs) const {
  return Binop<mlir::arith::SubIOp>(rhs);
}

ImplicitArithOpBuilder ImplicitArithOpBuilder::operator*(int64_t rhs) const {
  return Binop<mlir::arith::MulIOp>(rhs);
}

ImplicitArithOpBuilder ImplicitArithOpBuilder::operator*(
    mlir::Value rhs) const {
  return Binop<mlir::arith::MulIOp>(rhs);
}

ImplicitArithOpBuilder ImplicitArithOpBuilder::operator&(
    mlir::Value rhs) const {
  return Binop<mlir::arith::AndIOp>(rhs);
}

ImplicitArithOpBuilder ImplicitArithOpBuilder::operator&(int64_t rhs) const {
  return Binop<mlir::arith::AndIOp>(rhs);
}

ImplicitArithOpBuilder ImplicitArithOpBuilder::operator|(
    mlir::Value rhs) const {
  return Binop<mlir::arith::OrIOp>(rhs);
}

ImplicitArithOpBuilder ImplicitArithOpBuilder::operator|(int64_t rhs) const {
  return Binop<mlir::arith::OrIOp>(rhs);
}

ImplicitArithOpBuilder ImplicitArithOpBuilder::operator^(
    mlir::Value rhs) const {
  return Binop<mlir::arith::XOrIOp>(rhs);
}

ImplicitArithOpBuilder ImplicitArithOpBuilder::operator<<(
    mlir::Value rhs) const {
  return Binop<mlir::arith::ShLIOp>(rhs);
}

ImplicitArithOpBuilder ImplicitArithOpBuilder::operator<<(int64_t rhs) const {
  return Binop<mlir::arith::ShLIOp>(rhs);
}

ImplicitArithOpBuilder ImplicitArithOpBuilder::shl(mlir::Value rhs) const {
  return Binop<mlir::arith::ShLIOp>(rhs);
}

ImplicitArithOpBuilder ImplicitArithOpBuilder::shl(int64_t rhs) const {
  return Binop<mlir::arith::ShLIOp>(rhs);
}

ImplicitArithOpBuilder ImplicitArithOpBuilder::operator>>(
    mlir::Value rhs) const {
  return Binop<mlir::arith::ShRUIOp>(rhs);
}

ImplicitArithOpBuilder ImplicitArithOpBuilder::operator>>(int64_t rhs) const {
  return Binop<mlir::arith::ShRUIOp>(rhs);
}

ImplicitArithOpBuilder ImplicitArithOpBuilder::shrui(mlir::Value rhs) const {
  return Binop<mlir::arith::ShRUIOp>(rhs);
}

ImplicitArithOpBuilder ImplicitArithOpBuilder::shrui(int64_t rhs) const {
  return Binop<mlir::arith::ShRUIOp>(rhs);
}

ImplicitArithOpBuilder ImplicitArithOpBuilder::cmp(
    mlir::arith::CmpIPredicate pred, mlir::Value rhs) const {
  return {builder_->create<mlir::arith::CmpIOp>(pred, value_, rhs), builder_};
}

ImplicitArithOpBuilder ImplicitArithOpBuilder::cmp(
    mlir::arith::CmpIPredicate pred, int64_t rhs) const {
  return cmp(pred, MakeConstant(rhs));
}

ImplicitArithOpBuilder ImplicitArithOpBuilder::operator==(
    mlir::Value rhs) const {
  return cmp(mlir::arith::CmpIPredicate::eq, rhs);
}

ImplicitArithOpBuilder ImplicitArithOpBuilder::operator==(int64_t rhs) const {
  return cmp(mlir::arith::CmpIPredicate::eq, rhs);
}

ImplicitArithOpBuilder ImplicitArithOpBuilder::operator!=(int64_t rhs) const {
  return cmp(mlir::arith::CmpIPredicate::ne, rhs);
}

ImplicitArithOpBuilder ImplicitArithOpBuilder::MakeConstant(int64_t c) const {
  return {builder_->create<mlir::arith::ConstantIntOp>(c, value_.getType()),
          builder_};
}

template <typename Op>
ImplicitArithOpBuilder ImplicitArithOpBuilder::Binop(mlir::Value rhs) const {
  return {builder_->create<Op>(value_, rhs), builder_};
}

template <typename Op>
ImplicitArithOpBuilder ImplicitArithOpBuilder::Binop(int64_t rhs) const {
  return Binop<Op>(MakeConstant(rhs));
}

}  // namespace xla::emitters
