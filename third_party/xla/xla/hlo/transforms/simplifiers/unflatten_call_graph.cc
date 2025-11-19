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
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "highwayhash/arch_specific.h"
#include "highwayhash/hh_types.h"
#include "highwayhash/highwayhash.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/concurrency/concurrency_utils.h"
#include "xla/service/call_graph.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla {

namespace {
// Struct to hold all call instructions and called computations in a module.
struct HloCalls {
  // All callsites are guaranteed to be `kCall` instructions.
  absl::flat_hash_set<HloInstruction*> call_sites;
  absl::flat_hash_set<HloComputation*> targets;
};

// Iterates through all instructions in the module's computations
// and collects all `HloInstruction`s with opcode `kCall` into 'call_sites'
// and all unique computations targeted by these calls into 'targets'.
// It only retains call sites and targets that could potentially be duplicates
// by filtering out computations with unique properties (instruction count,
// parameter count).
HloCalls CollectHloCalls(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  std::unique_ptr<CallGraph> call_graph =
      CallGraph::Build(module, execution_threads);
  HloCalls calls;
  absl::flat_hash_map<uint64_t, uint64_t> count_num_instructions,
      count_num_params;

  for (const CallGraphNode& node : call_graph->nodes()) {
    for (const CallSite& callsite : node.callsites()) {
      if (callsite.instruction()->opcode() == HloOpcode::kCall) {
        calls.call_sites.insert(callsite.instruction());
        HloComputation* target = callsite.instruction()->to_apply();
        calls.targets.insert(target);
        ++count_num_instructions[target->instruction_count()];
        ++count_num_params[target->num_parameters()];
      }
    }
  }

  // Remove computations that cannot be duplicates: if a computation is unique
  // in terms of instruction count or parameter count it implies it cannot be
  // identical to any other computation.
  for (auto it = calls.call_sites.begin(), end = calls.call_sites.end();
       it != end;) {
    // `erase()` will invalidate `it`, so advance `it` first.
    auto copy_it = it++;
    HloComputation* computation = (*copy_it)->to_apply();
    if (count_num_instructions[computation->instruction_count()] == 1 ||
        count_num_params[computation->num_parameters()] == 1) {
      calls.targets.erase(computation);
      calls.call_sites.erase(copy_it);
    }
  }
  return calls;
}
}  // namespace

absl::StatusOr<std::vector<UnflattenCallGraph::ComputationHashResult>>
UnflattenCallGraph::HashComputations(
    const absl::flat_hash_set<HloComputation*>& called_computations) {
  auto hash_computation =
      [&](HloComputation* computation) -> ComputationHashResult {
    // Secret key used for hashing. Since we're not worried about attackers,
    // we can initialize to non-secret `openssl rand` generated values.
    static constexpr highwayhash::HHKey kHighwayHashKey = {
        0x787e1a69fdecd60b,
        0xe29d68c87b02eec8,
        0x0f6735946a0777ea,
        0x3444abc98410f39f,
    };
    highwayhash::HHStateT<HH_TARGET> state(kHighwayHashKey);
    std::string fingerprint = computation->ToString(print_options_);
    highwayhash::HHResult64 result;
    highwayhash::HighwayHashT(&state, fingerprint.data(), fingerprint.size(),
                              &result);
    return ComputationHashResult{
        result,
        std::move(fingerprint),
        computation,
    };
  };

  // Hash all called computations in parallel.
  return (xla::concurrency::ForEach<ComputationHashResult>(
      called_computations.begin(), called_computations.end(), hash_computation,
      *task_executor_));
}

absl::Status UnflattenCallGraph::ValidateComputationHashes(
    const std::vector<ComputationHashResult>& hash_results,
    const absl::flat_hash_map<uint64_t, const ComputationHashResult*>&
        hash_to_canonical) {
  auto validate_against_canonical = [&](const ComputationHashResult& result) {
    uint64_t candidate_hash = result.hash;
    const std::string& candidate_fingerprint = result.fingerprint;
    const std::string& canonical_fingerprint =
        hash_to_canonical.at(candidate_hash)->fingerprint;

    if (candidate_fingerprint != canonical_fingerprint) {
      return absl::InternalError(
          absl::StrCat("Hash collision detected. Hash: ", candidate_hash, "\n",
                       "Hashes are equal but fingerprints are different.\n",
                       "Computation 1:\n", candidate_fingerprint, "\n",
                       "Computation 2:\n", canonical_fingerprint, "\n"));
    }
    return absl::OkStatus();
  };

  // Validate all computations against their canonical versions in parallel.
  TF_RETURN_IF_ERROR(
      (xla::concurrency::ForEach(hash_results.begin(), hash_results.end(),
                                 validate_against_canonical, *task_executor_)));

  return absl::OkStatus();
}

absl::StatusOr<bool> UnflattenCallGraph::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  VLOG(1) << "Running UnflattenCallGraph on module " << module->name();

  XLA_SCOPED_LOGGING_TIMER_LEVEL(
      absl::StrCat("Ran UnflattenCallGraph on module: ", module->name()), 1);

  // Find all call instructions and their unique computation targets
  const HloCalls calls = CollectHloCalls(module, execution_threads);
  if (calls.targets.empty()) {
    return false;
  }
  TF_ASSIGN_OR_RETURN(const std::vector<ComputationHashResult> hash_results,
                      HashComputations(calls.targets));

  // Map computations to their hashes.
  // The HloComputation* keys are owned by the HloModule and are guaranteed to
  // be valid for the lifetime of this map.
  absl::flat_hash_map<HloComputation*, uint64_t> computation_to_hash;
  computation_to_hash.reserve(calls.targets.size());
  // Map each hash to the first computation encountered with that hash
  absl::flat_hash_map<uint64_t, const ComputationHashResult*> hash_to_canonical;

  for (const ComputationHashResult& result : hash_results) {
    computation_to_hash[result.computation] = result.hash;
    hash_to_canonical.try_emplace(result.hash, &result);
  }

  absl::Status validation_status =
      ValidateComputationHashes(hash_results, hash_to_canonical);
  if (!validation_status.ok()) {
    LOG(ERROR) << "UnflattenCallGraph failed validation: " << validation_status;
    return false;
  }

  bool changed = false;
  // Lambda to find the canonical computation for a given computation.
  auto get_canonical_computation = [&](const HloComputation* original_called) {
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
  for (HloInstruction* instruction : calls.call_sites) {
    instruction->ReplaceCalledComputations(get_canonical_computation);
  }

  if (changed) {
    TF_RETURN_IF_ERROR(module->RemoveUnusedComputations());
    module->CleanupComputations();
  }

  return changed;
}
}  // namespace xla
