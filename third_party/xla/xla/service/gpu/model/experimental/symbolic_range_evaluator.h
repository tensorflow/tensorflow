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

#ifndef XLA_SERVICE_GPU_MODEL_EXPERIMENTAL_SYMBOLIC_RANGE_EVALUATOR_H_
#define XLA_SERVICE_GPU_MODEL_EXPERIMENTAL_SYMBOLIC_RANGE_EVALUATOR_H_

#include <cstdint>

#include "xla/service/gpu/model/experimental/symbolic_expr.h"

namespace xla {
namespace gpu {

// TODO(b/442385842): Implement Interval class
struct Interval {
  int64_t lower;
  int64_t upper;
};

class SymbolicRangeEvaluator {
 public:
  SymbolicRangeEvaluator() = default;

  Interval ComputeExpressionRange(SymbolicExpr expr) {
    // TODO(b/442385842): Implement range computation.
    return {INT64_MIN, INT64_MAX};
  }
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MODEL_EXPERIMENTAL_SYMBOLIC_RANGE_EVALUATOR_H_
