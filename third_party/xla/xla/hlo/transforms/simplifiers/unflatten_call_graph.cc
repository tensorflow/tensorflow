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

#include "xla/hlo/transforms/simplifiers/unflatten_call_graph.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/hash/hash.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/hlo/utils/concurrency/concurrency_utils.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla {

UnflattenCallGraph::HloCalls UnflattenCallGraph::CollectHloCalls(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  HloCalls calls;
  for (HloComputation* computation : module->computations(execution_threads)) {
    for (HloInstruction* instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kCall) {
        calls.calls_sites.push_back(instruction);
        calls.targets.insert(instruction->called_computations().begin(),
                             instruction->called_computations().end());
      }
    }
  }
  return calls;
}

absl::Status UnflattenCallGraph::ValidateComputationHashes(
    const std::vector<ComputationHashResult>& hash_results,
    const absl::flat_hash_map<uint64_t, const ComputationHashResult*>&
        hash_to_canonical) {
  auto validate_against_canonical =
      [&](const ComputationHashResult& candidate) -> absl::Status {
    uint64_t hash = candidate.hash;
    const ComputationHashResult* canonical = hash_to_canonical.at(hash);

    // Quick checks for common differences before full fingerprint comparison.
    if (candidate.computation->num_parameters() !=
        canonical->computation->num_parameters()) {
      return absl::InternalError(
          absl::StrCat("Hash collision detected. Hash: ", hash, "\n",
                       "Computations have different number of parameters.\n",
                       "Computation 1:\n", candidate.fingerprint, "\n",
                       "Computation 2:\n", canonical->fingerprint));
    }

    // Check that the number of instructions is the same.
    if (candidate.computation->instruction_count() !=
        canonical->computation->instruction_count()) {
      return absl::InternalError(
          absl::StrCat("Hash collision detected. Hash: ", hash, "\n",
                       "Computations have different number of instructions.\n",
                       "Computation 1:\n", candidate.fingerprint, "\n",
                       "Computation 2:\n", canonical->fingerprint));
    }

    // Check that the number of computations these two call are the same.
    if (candidate.computation->callee_computations().size() !=
        canonical->computation->callee_computations().size()) {
      return absl::InternalError(
          absl::StrCat("Hash collision detected. Hash: ", hash, "\n",
                       "Computations have different number of callees.\n",
                       "Computation 1:\n", candidate.fingerprint, "\n",
                       "Computation 2:\n", canonical->fingerprint));
    }

    // Finally check that the computations are actually identical.
    // This is the most expensive and thorough check, so we do it last.
    if (candidate.fingerprint != canonical->fingerprint) {
      return absl::InternalError(
          absl::StrCat("Hash collision detected. Hash: ", hash, "\n",
                       "Hashes are equal but computations are different.\n",
                       "Computation 1:\n", candidate.fingerprint, "\n",
                       "Computation 2:\n", canonical->fingerprint));
    }
    return absl::OkStatus();
  };

  // Validate all computations against their canonical versions in parallel.
  TF_RETURN_IF_ERROR(
      (xla::concurrency::ForEach(hash_results.begin(), hash_results.end(),
                                 validate_against_canonical, *task_executor_)));

  return absl::OkStatus();
}

absl::StatusOr<std::vector<UnflattenCallGraph::ComputationHashResult>>
UnflattenCallGraph::HashComputations(
    const absl::flat_hash_set<HloComputation*>& called_computations) {
  const HloPrintOptions print_options =
      HloPrintOptions::Canonical().set_print_ids(false);

  auto hash_computation = [&](HloComputation* computation)
      -> absl::StatusOr<ComputationHashResult> {
    std::string fingerprint = computation->ToString(print_options);
    uint64_t hash = absl::Hash<std::string>()(fingerprint);
    return ComputationHashResult{hash, std::move(fingerprint), computation};
  };

  // Hash all called computations in parallel.
  return (xla::concurrency::ForEach<ComputationHashResult>(
      called_computations.begin(), called_computations.end(), hash_computation,
      *task_executor_));
}

absl::StatusOr<bool> UnflattenCallGraph::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  XLA_SCOPED_LOGGING_TIMER_LEVEL("UnflattenCallGraph::Run", 0);
  VLOG(1) << "Running UnflattenCallGraph on module " << module->name();

  // Find all call instructions and their unique computation targets
  HloCalls calls = CollectHloCalls(module, execution_threads);
  if (calls.targets.empty()) {
    return false;
  }
  TF_ASSIGN_OR_RETURN(const std::vector<ComputationHashResult>& hash_results,
                      HashComputations(calls.targets));

  // Map computations to their hashes.
  absl::flat_hash_map<HloComputation*, uint64_t> computation_to_hash;
  computation_to_hash.reserve(calls.targets.size());
  // Map each hash to the first computation encountered with that hash
  absl::flat_hash_map<uint64_t, const ComputationHashResult*> hash_to_canonical;

  for (int i = 0; i < hash_results.size(); ++i) {
    computation_to_hash[hash_results[i].computation] = hash_results[i].hash;
    hash_to_canonical.try_emplace(hash_results[i].hash, &hash_results[i]);
  }

  // Verify there are no hash collisions.
  TF_RETURN_IF_ERROR(
      ValidateComputationHashes(hash_results, hash_to_canonical));

  bool changed = false;
  // Lambda to find the canonical computation for a given computation.
  auto get_canonical_computation = [&](HloComputation* original_called) {
    uint64_t hash = computation_to_hash.at(original_called);
    HloComputation* canonical_called = hash_to_canonical.at(hash)->computation;

    if (original_called != canonical_called) {
      VLOG(1) << "Replacing call to " << original_called->name() << " ["
              << original_called << "] with " << canonical_called->name()
              << " [" << canonical_called << "]";
      changed = true;
    }
    return canonical_called;
  };

  // Update all call sites to point to the canonical computations.
  for (HloInstruction* instruction : calls.calls_sites) {
    instruction->ReplaceCalledComputations(get_canonical_computation);
  }

  if (changed) {
    // Clean up any computations that are now no longer called.
    TF_RETURN_IF_ERROR(module->RemoveUnusedComputations());
  }

  return changed;
}
}  // namespace xla
