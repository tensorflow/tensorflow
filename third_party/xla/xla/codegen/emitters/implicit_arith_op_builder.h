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

#ifndef XLA_CODEGEN_EMITTERS_IMPLICIT_ARITH_OP_BUILDER_H_
#define XLA_CODEGEN_EMITTERS_IMPLICIT_ARITH_OP_BUILDER_H_

#include <cstdint>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Value.h"

namespace xla::emitters {

// Wraps a Value to provide operator overloading for more readable building of
// integer arithmetic expressions.
class ImplicitArithOpBuilder {
 public:
  ImplicitArithOpBuilder() = default;
  ImplicitArithOpBuilder(mlir::Value value,
                         mlir::ImplicitLocOpBuilder* builder);

  mlir::Value value() const;
  operator mlir::Value() const;  // NOLINT

  // Integer addition.
  ImplicitArithOpBuilder operator+(int64_t rhs) const;
  ImplicitArithOpBuilder operator+(mlir::Value rhs) const;
  // Integer subtraction.
  ImplicitArithOpBuilder operator-(int64_t rhs) const;
  ImplicitArithOpBuilder operator-(mlir::Value rhs) const;
  // Integer multiplication.
  ImplicitArithOpBuilder operator*(int64_t rhs) const;
  ImplicitArithOpBuilder operator*(mlir::Value rhs) const;
  // Bitwise and.
  ImplicitArithOpBuilder operator&(mlir::Value rhs) const;
  ImplicitArithOpBuilder operator&(int64_t rhs) const;
  // Bitwise or.
  ImplicitArithOpBuilder operator|(mlir::Value rhs) const;
  ImplicitArithOpBuilder operator|(int64_t rhs) const;
  // Bitwise xor.
  ImplicitArithOpBuilder operator^(mlir::Value rhs) const;
  // Logical shift left.
  ImplicitArithOpBuilder operator<<(mlir::Value rhs) const;
  ImplicitArithOpBuilder operator<<(int64_t rhs) const;
  ImplicitArithOpBuilder shl(mlir::Value rhs) const;
  ImplicitArithOpBuilder shl(int64_t rhs) const;
  // Logical shift right.
  ImplicitArithOpBuilder operator>>(mlir::Value rhs) const;
  ImplicitArithOpBuilder operator>>(int64_t rhs) const;
  ImplicitArithOpBuilder shrui(mlir::Value rhs) const;
  ImplicitArithOpBuilder shrui(int64_t rhs) const;

  // Comparison operations.
  ImplicitArithOpBuilder cmp(mlir::arith::CmpIPredicate pred,
                             mlir::Value rhs) const;
  ImplicitArithOpBuilder cmp(mlir::arith::CmpIPredicate pred,
                             int64_t rhs) const;
  ImplicitArithOpBuilder operator==(mlir::Value rhs) const;
  ImplicitArithOpBuilder operator==(int64_t rhs) const;
  ImplicitArithOpBuilder operator!=(int64_t rhs) const;

  ImplicitArithOpBuilder MakeConstant(int64_t c) const;

 private:
  template <typename Op>
  ImplicitArithOpBuilder Binop(mlir::Value rhs) const;

  template <typename Op>
  ImplicitArithOpBuilder Binop(int64_t rhs) const;

 private:
  mlir::Value value_;
  mlir::ImplicitLocOpBuilder* builder_;
};

}  // namespace xla::emitters

#endif  // XLA_CODEGEN_EMITTERS_IMPLICIT_ARITH_OP_BUILDER_H_
