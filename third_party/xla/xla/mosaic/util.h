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

#ifndef XLA_MOSAIC_UTIL_H_
#define XLA_MOSAIC_UTIL_H_

#include <cstdint>
#include <optional>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

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

// Concatenates `x` and `y` after macro-expanding both.
#define MOSAIC_CONCAT_NAME_IMPL(x, y) x##y
#define MOSAIC_CONCAT_NAME(x, y) MOSAIC_CONCAT_NAME_IMPL(x, y)

namespace mlir::mosaic::internal {
template <typename T>
std::optional<T> FailureOrToOptional(mlir::FailureOr<T> value) {
  return std::optional<T>(std::move(value));
}
}  // namespace mlir::mosaic::internal

#define FAILUREOR_ASSIGN_OR_RETURN_IMPL(failureor, lhs, rhs)           \
  auto failureor = ::mlir::mosaic::internal::FailureOrToOptional(rhs); \
  if (!failureor.has_value()) {                                        \
    return mlir::failure();                                            \
  }                                                                    \
  FAILUREOR_ASSIGN_OR_RETURN_UNPARENTHESIZE_IF_PARENTHESIZED(lhs) =    \
      (std::move(*failureor))

#define FAILUREOR_ASSIGN_OR_RETURN(lhs, rhs)                                  \
  FAILUREOR_ASSIGN_OR_RETURN_IMPL(MOSAIC_CONCAT_NAME(failureor, __COUNTER__), \
                                  lhs, rhs)

#define RETURN_IF_FAILED(...)        \
  do {                               \
    if (mlir::failed(__VA_ARGS__)) { \
      return mlir::failure();        \
    }                                \
  } while (false)

namespace mlir::mosaic {

// Computes the dimensions that were squeezed from the source shape to match the
// target shape. Returns the dimensions in increasing order.
FailureOr<SmallVector<int64_t>> computeSqueezedDimsChecked(
    Operation* op, ArrayRef<int64_t> source_shape,
    ArrayRef<int64_t> target_shape);

}  // namespace mlir::mosaic

#endif  // XLA_MOSAIC_UTIL_H_
