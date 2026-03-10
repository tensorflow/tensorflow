/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_MLIR_UTILS_MATH_UTIL_H_
#define XLA_MLIR_UTILS_MATH_UTIL_H_

#include <cstdint>

#include "mlir/IR/Value.h"

namespace xla {

// The maximum meaningful alignment factor, representing an alignment of 2^62.
// Used as a sentinel for 0-values or fully aligned cases.
inline constexpr int64_t kMaxAlignment = int64_t{1} << 62;

// Returns the largest power of 2 that is known to divide 'value' at compile
// time, by recursively inspecting the arithmetic operations that define it.
// For example, if value = x* 256 + y * 64, returns 64.
//
// The 'depth' parameter is used to limit the recursion depth to prevent
// infinite loops or combinatorial explosion during traversal.
int64_t GetKnownAlignment(mlir::Value value, int depth = 0);

}  // namespace xla

#endif  // XLA_MLIR_UTILS_MATH_UTIL_H_
