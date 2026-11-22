/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_SPMD_SHARDY_EXTENSIONS_PARTIAL_REDUCE_CUSTOM_CALL_H_
#define XLA_SERVICE_SPMD_SHARDY_EXTENSIONS_PARTIAL_REDUCE_CUSTOM_CALL_H_

#include <optional>

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LogicalResult.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace xla {
namespace sdy {

// Shardy propagation and global-to-local lowering hooks for XLA's post-HLO
// `stablehlo.custom_call @PartialReduce` (`2 * K` operands
// `[inputs..., init_values...]` -> `K` results).

// Custom-call target name for XLA's TPU partial-reduction op.
inline constexpr llvm::StringLiteral kPartialReduceTargetName = "PartialReduce";

// Builds an `OpShardingRuleAttr` for `stablehlo.custom_call @PartialReduce`:
// maps non-reduction dimensions pointwise across `0..K-1`, marks the input
// `reduction_dim` as `kPermutation`, and marks the output `reduction_dim` as
// `kNeedReplication` with `isBlocked = true`. Returns a null attribute if the
// operand/result shapes or `reduction_dim` are malformed.
mlir::sdy::OpShardingRuleAttr createPartialReduceShardingRule(
    mlir::stablehlo::CustomCallOp customCall);

// Resolves a `reduction_dim`-sharded `stablehlo.custom_call @PartialReduce` in
// global IR into a Stage 1 per-shard `@PartialReduce` (inside
// `sdy.manual_computation`), a cross-shard all-gather (`sdy.reshard`), and a
// Stage 2 fold (`stablehlo.reduce`).
//
// Return contract for `PartitioningOpInterface::resolvePermutationFactors`:
// - Returns `std::nullopt` if `shardingRule` does not identify a unique
//   `kPermutation` reduction dimension, delegating to Shardy's generic
//   replication fallback.
// - Returns `mlir::success()` when `customCall` is resolved in place (single
//   stage) or replaced by the two-stage pipeline.
// - Returns `mlir::failure()` after emitting an op error if `customCall`, its
//   `backend_config`, or its `called_computations` body is invalid.
std::optional<mlir::LogicalResult> resolvePartialReduce(
    mlir::stablehlo::CustomCallOp customCall,
    mlir::sdy::OpShardingRuleAttr shardingRule, mlir::IRRewriter& rewriter,
    const mlir::SymbolTable& symbolTable);

}  // namespace sdy
}  // namespace xla

#endif  // XLA_SERVICE_SPMD_SHARDY_EXTENSIONS_PARTIAL_REDUCE_CUSTOM_CALL_H_
