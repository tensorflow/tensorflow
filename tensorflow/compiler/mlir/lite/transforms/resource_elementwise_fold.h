/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_RESOURCE_ELEMENTWISE_FOLD_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_RESOURCE_ELEMENTWISE_FOLD_H_

#include <cstdint>

#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir::TFL {

// Default upper bound on the size of a folded result that is larger than the
// sum of its constant inputs (e.g. due to broadcasting).
inline constexpr int64_t kDefaultMaxElementwiseFoldGrowthBytes =
    int64_t{64} * 1024 * 1024;  // 64 MiB

// Folds TFL elementwise ops (unary math, binary arithmetic, comparisons,
// select) and concatenation whose operands are all constants and at least one
// operand is a DenseResourceElementsAttr. Results are always materialized as
// DenseResourceElementsAttr backed by heap blobs; the computation operates
// directly on the raw buffers.
//
// An input blob is mutated in place only when that is provably unobservable:
// the blob is mutable, exactly one attribute in the module references it, the
// referencing constant op is used only by the folded op, and the operand type
// equals the result type. Otherwise a new blob is allocated. Blobs are
// released only once no attribute in the module references them.
//
// Folds that grow the constant data beyond the sum of the inputs are skipped
// when the result exceeds `max_growth_bytes`.
LogicalResult FoldResourceElementwiseOps(
    ModuleOp module,
    int64_t max_growth_bytes = kDefaultMaxElementwiseFoldGrowthBytes);

}  // namespace mlir::TFL

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_RESOURCE_ELEMENTWISE_FOLD_H_
