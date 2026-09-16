/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_MOSAIC_DIALECT_TPU_UTIL_H_
#define XLA_MOSAIC_DIALECT_TPU_UTIL_H_

#include <cstdint>
#include <optional>
#include <sstream>
#include <string>

#include "absl/log/check.h"
#include "llvm/Support/Compiler.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "xla/mosaic/dialect/tpu/tpu_dialect.h"
#include "xla/tsl/platform/statusor.h"

// TODO: Instead of CHECK_EQs, can we do something like TF_RET_CHECK but with
// MLIR diagnostics?
// e.g.
// #define MLIR_RET_CHECK_EQ(a, b, diagnostic) \
//   do { \
//     const auto a_ = a; \
//     const auto b_ = b; \
//     if (LLVM_UNLIKELY(a_ != b_)) { \
//       return diagnostic << "Check failed: " << #a << " != " << #b << "(" <<
//       a_  << " vs. " << b_ << ")"; \
//     } \
//   } while (false)

// All the macros below here are to handle the case in
// FAILUREOR_ASSIGN_OR_RETURN where the LHS is wrapped in parentheses. See a
// more detailed discussion at https://stackoverflow.com/a/62984543
#define FAILUREOR_ASSIGN_OR_RETURN_UNPARENTHESIZE_IF_PARENTHESIZED(X) \
  FAILUREOR_ASSIGN_OR_RETURN_ESCAPE(FAILUREOR_ASSIGN_OR_RETURN_EMPTY X)
#define FAILUREOR_ASSIGN_OR_RETURN_EMPTY(...) \
  FAILUREOR_ASSIGN_OR_RETURN_EMPTY __VA_ARGS__
#define FAILUREOR_ASSIGN_OR_RETURN_ESCAPE(...) \
  FAILUREOR_ASSIGN_OR_RETURN_ESCAPE_(__VA_ARGS__)
#define FAILUREOR_ASSIGN_OR_RETURN_ESCAPE_(...) \
  FAILUREOR_ASSIGN_OR_RETURN_##__VA_ARGS__
#define FAILUREOR_ASSIGN_OR_RETURN_FAILUREOR_ASSIGN_OR_RETURN_EMPTY

#define FAILUREOR_ASSIGN_OR_RETURN_IMPL(failureor, lhs, rhs)        \
  auto failureor = rhs;                                             \
  if (mlir::failed(failureor)) {                                    \
    return mlir::failure();                                         \
  }                                                                 \
  FAILUREOR_ASSIGN_OR_RETURN_UNPARENTHESIZE_IF_PARENTHESIZED(lhs) = \
      (std::move(failureor).value())

#define FAILUREOR_ASSIGN_OR_RETURN(lhs, rhs) \
  FAILUREOR_ASSIGN_OR_RETURN_IMPL(           \
      TF_STATUS_MACROS_CONCAT_NAME(failureor, __COUNTER__), lhs, rhs)

#define RETURN_IF_FAILED(...)        \
  do {                               \
    if (mlir::failed(__VA_ARGS__)) { \
      return mlir::failure();        \
    }                                \
  } while (false)

namespace mlir::tpu {

// TPU_ASSERT_* macros should be understood as an assert, i.e. use it to check
// things that should never happen. We prefer returning failure over a CHECK
// because it's easier to debug from Python (particularly from OSS where symbols
// are removed)
#define TPU_ASSERT_IMPL(stream, cond)                    \
  if (LLVM_UNLIKELY(!(cond))) {                          \
    (stream) << "Internal error: assert failed: " #cond; \
  }
#define TPU_ASSERT_CMP_IMPL(stream, lhs, rhs, cmp)                            \
  if (LLVM_UNLIKELY(!((lhs)cmp(rhs)))) {                                      \
    (stream) << "Internal error: assert failed: " #lhs " " #cmp " " #rhs " (" \
             << (lhs) << " vs. " << (rhs) << ")";                             \
    return failure();                                                         \
  }
#define TPU_ASSERT_OP(cond) TPU_ASSERT_IMPL(op.emitOpError(), cond)
#define TPU_ASSERT_CMP_OP_IMPL(lhs, rhs, cmp) \
  TPU_ASSERT_CMP_IMPL(op.emitOpError(), lhs, rhs, cmp)
#define TPU_ASSERT_EQ_OP(lhs, rhs) TPU_ASSERT_CMP_OP_IMPL(lhs, rhs, ==)
#define TPU_ASSERT_GE_OP(lhs, rhs) TPU_ASSERT_CMP_OP_IMPL(lhs, rhs, >=)
#define TPU_ASSERT_GT_OP(lhs, rhs) TPU_ASSERT_CMP_OP_IMPL(lhs, rhs, >)
#define TPU_ASSERT_LE_OP(lhs, rhs) TPU_ASSERT_CMP_OP_IMPL(lhs, rhs, <=)
#define TPU_ASSERT_LT_OP(lhs, rhs) TPU_ASSERT_CMP_OP_IMPL(lhs, rhs, <)
#define TPU_ASSERT_LOC(loc, cond) TPU_ASSERT_IMPL(mlir::emitError(loc), cond)
#define TPU_ASSERT_CMP_LOC_IMPL(loc, lhs, rhs, cmp) \
  TPU_ASSERT_CMP_IMPL(loc, lhs, rhs, cmp)
#define TPU_ASSERT_EQ_LOC(loc, lhs, rhs) \
  TPU_ASSERT_CMP_LOC_IMPL(mlir::emitError(loc), lhs, rhs, ==)
#define TPU_ASSERT_GE_LOC(loc, lhs, rhs) \
  TPU_ASSERT_CMP_LOC_IMPL(mlir::emitError(loc), lhs, rhs, >=)
#define TPU_ASSERT_GT_LOC(loc, lhs, rhs) \
  TPU_ASSERT_CMP_LOC_IMPL(mlir::emitError(loc), lhs, rhs, >)
#define TPU_ASSERT_LT_LOC(loc, lhs, rhs) \
  TPU_ASSERT_CMP_LOC_IMPL(mlir::emitError(loc), lhs, rhs, <)
#define TPU_ASSERT_LE_LOC(loc, lhs, rhs) \
  TPU_ASSERT_CMP_LOC_IMPL(mlir::emitError(loc), lhs, rhs, <=)

template <bool adjust_bool = false>
int8_t getTypeBitwidth(Type ty) {
  if (auto integer_ty = dyn_cast<IntegerType>(ty)) {
    const unsigned width = integer_ty.getWidth();
    if constexpr (adjust_bool) {
      // We store only one i1 per vreg element.
      return width == 1 ? 32 : width;
    } else {
      return width;
    }
  }
  if (isa<Float8EXMYType>(ty)) {
    return 8;
  }
  return ty.getIntOrFloatBitWidth();
}

// Returns the bitwidth of the element type. The function works for both
// scalar and vector types.
template <bool adjust_bool = false>
inline int8_t getElementTypeBitwidth(Type ty) {
  if (auto vty = dyn_cast<VectorType>(ty)) {
    return getTypeBitwidth<adjust_bool>(vty.getElementType());
  }
  return getTypeBitwidth<adjust_bool>(ty);
}

template <bool adjust_bool = false>
inline int8_t getElementTypeBitwidth(MemRefType ty) {
  return getElementTypeBitwidth<adjust_bool>(ty.getElementType());
}

// Debug only util.
template <typename T>
std::string shapeToString(const T& shape) {
  std::ostringstream os;
  os << "(";
  for (auto it = shape.begin(); it != shape.end(); ++it) {
    if (it != shape.begin()) {
      os << ",";
    }
    os << *it;
  }
  os << ")";
  return os.str();
}

// Computes the dimensions that were squeezed from the source shape to match the
// target shape. Returns the dimensions in increasing order.
FailureOr<SmallVector<int>> computeSqueezedDimsChecked(
    Operation* op, ArrayRef<int64_t> source_shape,
    ArrayRef<int64_t> target_shape);

// Determines whether the given MemRefType has the given memory space.
bool HasMemorySpace(MemRefType ty, tpu::MemorySpace space,
                    std::optional<tpu::CoreType> core_type = std::nullopt);

// Helper functions to create constants.
inline arith::ConstantOp IdxConst(int64_t idx, OpBuilder& builder,
                                  Location loc) {
  return arith::ConstantOp::create(builder, loc, builder.getIndexType(),
                                   builder.getIndexAttr(idx));
}

// Return a mod b for a, b > 0, but adjusted to return b when a mod b == 0 such
// that the result is strictly positive.
template <typename U, typename V>
auto positiveMod(U a, V b) {
  DCHECK_GT(a, 0);
  DCHECK_GT(b, 0);
  return (a - 1) % b + 1;
}

// Fills the output of `size` number of elements with the given values and their
// positions, and fills the rest with `missing`.
SmallVector<Value> fillPositions(ValueRange values, ArrayRef<int32_t> positions,
                                 int size, Value missing = nullptr);

}  // namespace mlir::tpu

#endif  // XLA_MOSAIC_DIALECT_TPU_UTIL_H_
