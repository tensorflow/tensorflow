/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_SERVICE_GPU_MODEL_AFFINE_MAP_EVALUATOR_H_
#define XLA_SERVICE_GPU_MODEL_AFFINE_MAP_EVALUATOR_H_

#include <cstdint>
#include <vector>

#include "absl/types/span.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"

namespace xla {
namespace gpu {

// Given an AffineExpr and the values for its dimensions and symbols, evaluates
// the result.
int64_t EvaluateAffineExpr(mlir::AffineExpr expr,
                           absl::Span<int64_t const> dim_values,
                           absl::Span<int64_t const> symbol_values = {});

// Given an AffineMap and the values for its dimensions and symbols, evaluates
// the results.
llvm::SmallVector<int64_t> EvaluateAffineMap(
    mlir::AffineMap affine_map, absl::Span<int64_t const> dim_values,
    absl::Span<int64_t const> symbol_values = {});

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MODEL_AFFINE_MAP_EVALUATOR_H_
