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

#ifndef XLA_SERVICE_GPU_MODEL_EXPERIMENTAL_SYMBOLIC_EXPR_SIMPLIFIER_H_
#define XLA_SERVICE_GPU_MODEL_EXPERIMENTAL_SYMBOLIC_EXPR_SIMPLIFIER_H_

#include "xla/service/gpu/model/experimental/symbolic_expr.h"
#include "xla/service/gpu/model/experimental/symbolic_map.h"
#include "xla/service/gpu/model/experimental/symbolic_range_evaluator.h"

namespace xla {
namespace gpu {

class SymbolicExprSimplifier {
 public:
  explicit SymbolicExprSimplifier(SymbolicExprContext* ctx,
                                  SymbolicRangeEvaluator* range_evaluator)
      : ctx_(ctx),
        range_evaluator_(range_evaluator),
        zero_(ctx->CreateConstant(0)) {}

  // Simplifies the map as much as possible.
  SymbolicMap Simplify(SymbolicMap map);

  // Simplifies a single expression.
  SymbolicExpr Simplify(SymbolicExpr expr);

  // Performs SymbolicExpr simplification for all constraints.
  // Returns true if simplification was performed.
  bool SimplifyConstraintExprs(SymbolicMap& map);

  // Performs range simplification for all constraints.
  // Returns true if simplification was performed.
  bool SimplifyConstraintRanges(SymbolicMap& map);

 private:
  // Attempts to simplify the expression, but doesn't attempt to simplify the
  // result further.
  SymbolicExpr SimplifyOnce(SymbolicExpr expr);

  // Splits a nested sum into a * gcd + b.
  std::tuple<SymbolicExpr /*a*/, int64_t /*gcd*/, SymbolicExpr /*b*/>
  SplitSumByGcd(SymbolicExpr sum);

  // Helpers for RewriteFloorDiv
  SymbolicExpr SimplifyModDiv(SymbolicExpr dividend, int64_t divisor);
  SymbolicExpr SimplifyDivDiv(SymbolicExpr dividend, int64_t divisor);
  SymbolicExpr SimplifySumDiv(SymbolicExpr dividend, int64_t divisor);

  // Simplifiers for different expression types.
  SymbolicExpr RewriteAdd(SymbolicExpr expr);
  SymbolicExpr RewriteMul(SymbolicExpr expr);
  SymbolicExpr RewriteFloorDiv(SymbolicExpr expr);
  SymbolicExpr RewriteCeilDiv(SymbolicExpr expr);
  SymbolicExpr RewriteMod(SymbolicExpr expr);
  SymbolicExpr RewriteMin(SymbolicExpr expr);
  SymbolicExpr RewriteMax(SymbolicExpr expr);

  [[maybe_unused]] SymbolicExprContext* ctx_;
  [[maybe_unused]] SymbolicRangeEvaluator* range_evaluator_;
  SymbolicExpr zero_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MODEL_EXPERIMENTAL_SYMBOLIC_EXPR_SIMPLIFIER_H_
