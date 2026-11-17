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

#ifndef XLA_SERVICE_SPMD_SHARDY_EXTENSIONS_CUSTOM_SPMD_PARTITIONING_H_
#define XLA_SERVICE_SPMD_SHARDY_EXTENSIONS_CUSTOM_SPMD_PARTITIONING_H_

#include <cstdint>
#include <optional>

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LogicalResult.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace xla {
namespace sdy {

// Names the custom-call target emitted by JAX `@custom_partitioning`.
inline constexpr llvm::StringLiteral kCustomSPMDPartitioningTargetName =
    "CustomSPMDPartitioning";

// Registers dialects and extensions required to deserialize, inline, and
// legalize MLIR modules returned by custom call partitioning callbacks.
void registerCustomSPMDPartitioningDialects(mlir::DialectRegistry& registry);

// Resolves a `stablehlo.custom_call` into `sdy.manual_computation` (plus
// boundary reshards and divisibility pad/slice ops) via its registered
// `xla::CustomCallPartitioner::PartitionToMlir` callback.
//
// Returns `std::nullopt` if no `CustomCallPartitioner` is registered for
// `customCall.getCallTargetName()` or if `PartitionToMlir` returns
// `std::nullopt` (allowing non-MLIR custom calls to fall through),
// `mlir::success()` when `customCall` is replaced by its partitioned
// `sdy.manual_computation` subgraph, and `mlir::failure()` after emitting an op
// error if the callback fails or returns invalid shardings or MLIR.
std::optional<mlir::LogicalResult> resolveCustomCallPartitioning(
    mlir::stablehlo::CustomCallOp customCall, mlir::IRRewriter& rewriter,
    mlir::SymbolTable& symbolTable, int64_t& nextChannelId);

}  // namespace sdy
}  // namespace xla

#endif  // XLA_SERVICE_SPMD_SHARDY_EXTENSIONS_CUSTOM_SPMD_PARTITIONING_H_
