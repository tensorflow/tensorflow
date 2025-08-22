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

#ifndef XLA_HLO_TRANSFORMS_SIMPLIFIERS_UNFLATTEN_CALL_GRAPH_H_
#define XLA_HLO_TRANSFORMS_SIMPLIFIERS_UNFLATTEN_CALL_GRAPH_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/hlo/utils/concurrency/tsl_task_executor.h"

namespace xla {

// Unflatten a call graph. This pass will find computations that are identical
// and replace them with calls to a single computation.
class UnflattenCallGraph : public HloModulePass {
 public:
  UnflattenCallGraph()
      : task_executor_(std::make_unique<xla::concurrency::TslTaskExecutor>()) {}

  absl::string_view name() const override { return "unflatten-call-graph"; }

  // Find computations that are identical and replace them with calls to a
  // single computation.
  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  // Struct to hold the result of hashing a computation.
  struct ComputationHashResult {
    uint64_t hash;
    std::string fingerprint;
    HloComputation* computation;
  };

  // Struct to hold all call instructions and called computations in a module.
  struct HloCalls {
    std::vector<HloInstruction*> calls_sites;
    absl::flat_hash_set<HloComputation*> targets;
  };

  // Iterates through all instructions in the module's computations
  // and collects all `HloInstruction`s with opcode `kCall` into 'calls_sites'
  // and all unique computations targeted by these calls into 'targets'.
  HloCalls CollectHloCalls(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads);

  // Hashes computations to produce a fingerprint and hash value.
  // Uses canonical HLO text without IDs for stable, content-based hashing.
  absl::StatusOr<std::vector<ComputationHashResult>> HashComputations(
      const absl::flat_hash_set<HloComputation*>& called_computations);

  // Verifies that computations with the same hash are identical to prevent
  // incorrect merging due to hash collisions, using progressively more
  // expensive checks.
  absl::Status ValidateComputationHashes(
      const std::vector<ComputationHashResult>& hash_results,
      const absl::flat_hash_map<uint64_t, const ComputationHashResult*>&
          hash_to_canonical);

  // Thread pool used for parallelizing computation hashing and collision
  // detection.
  std::unique_ptr<xla::concurrency::TslTaskExecutor> task_executor_;
};

}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_SIMPLIFIERS_UNFLATTEN_CALL_GRAPH_H_
