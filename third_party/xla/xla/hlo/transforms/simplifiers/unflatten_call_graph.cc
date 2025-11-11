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
#include "absl/numeric/int128.h"
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
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/hlo/utils/concurrency/concurrency_utils.h"
#include "xla/hlo/utils/concurrency/tsl_task_executor.h"
#include "xla/printer.h"
#include "xla/service/call_graph.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "tsl/platform/fingerprint.h"

namespace xla {

namespace {
class HloComputationHasher : public Printer {
 public:
  HloComputationHasher()
      : highway_hasher_(kHighwayHashKey), farm_hash_(kFarmHashKey) {}

  void Append(const absl::AlphaNum& a) override {
    highway_hasher_.Append(a.data(), a.size());
    farm_hash_ =
        tsl::FingerprintCat64(farm_hash_, tsl::Fingerprint64(a.Piece()));
  }

  absl::uint128 ToFingerprint() {
    highwayhash::HHResult64 hh_result;
    highway_hasher_.Finalize(&hh_result);
    // Combine the HighwayHash and FarmHash results.
    return absl::MakeUint128(hh_result, farm_hash_);
  }

  static absl::uint128 StringToFingerprint(absl::string_view str) {
    highwayhash::HHStateT<HH_TARGET> state(kHighwayHashKey);
    highwayhash::HHResult64 hh_result;
    highwayhash::HighwayHashT(&state, str.data(), str.size(), &hh_result);
    uint64_t farm_result = tsl::Fingerprint64(str);
    return absl::MakeUint128(hh_result, farm_result);
  }

 private:
  highwayhash::HighwayHashCatT<HH_TARGET_PREFERRED> highway_hasher_;
  uint64_t farm_hash_;

  // Secret key used for hashing. Since we're not worried about attackers, we
  // can initialize to non-secret `openssl rand` generated values.
  static constexpr uint64_t kFarmHashKey = 0x4211c98d21f3f7f7;
  static constexpr highwayhash::HHKey kHighwayHashKey = {
      0x787e1a69fdecd60b,
      0xe29d68c87b02eec8,
      0x0f6735946a0777ea,
      0x3444abc98410f39f,
  };
};

// HighwayHasher computes a hash of an HloComputation by hashing its printed
// representation on a streaming basis (i.e. without actually stringifying).
class HloModuleHasher {
 public:
  HloModuleHasher()
      : print_options_(HloPrintOptions::Canonical()
                           .set_print_ids(false)
                           .set_print_metadata(true)
                           .set_print_backend_config(true)),
        task_executor_(std::make_unique<xla::concurrency::TslTaskExecutor>()) {}

  // Struct to hold the result of hashing a computation.
  struct ModuleHashResult {
    absl::flat_hash_map<HloComputation*, absl::uint128> computation_to_hash;
    absl::flat_hash_map<absl::uint128, HloComputation*> hash_to_canonical;
  };

  absl::StatusOr<ModuleHashResult> HashComputations(
      const absl::flat_hash_set<HloComputation*>& called_computations) {
    using HashAndComputation = std::pair<HloComputation*, absl::uint128>;

    auto hash_computation =
        [&](HloComputation* computation) -> HashAndComputation {
      HloComputationHasher printer;
      computation->Print(&printer, print_options_);
      return std::make_pair(computation, printer.ToFingerprint());
    };

    TF_ASSIGN_OR_RETURN(
        const std::vector<HashAndComputation> hashes,
        xla::concurrency::ForEach<HashAndComputation>(
            called_computations.begin(), called_computations.end(),
            hash_computation, *task_executor_));

    DCHECK_EQ(hashes.size(), called_computations.size());

    // Map each hash to the first computation encountered with that hash
    ModuleHashResult result;
    for (auto& [computation, hash] : hashes) {
      result.computation_to_hash[computation] = hash;
      result.hash_to_canonical.try_emplace(hash, computation);
    }
    return result;
  }

  absl::StatusOr<ModuleHashResult> HashComputationsWithValidation(
      const absl::flat_hash_set<HloComputation*>& called_computations) {
    struct ComputationHashResult {
      absl::uint128 hash;
      std::string fingerprint;
      HloComputation* computation;
    };

    auto hash_computation =
        [&](HloComputation* computation) -> ComputationHashResult {
      std::string fingerprint = computation->ToString(print_options_);
      return ComputationHashResult{
          HloComputationHasher::StringToFingerprint(fingerprint),
          std::move(fingerprint),
          computation,
      };
    };

    TF_ASSIGN_OR_RETURN(
        std::vector<ComputationHashResult> hash_results,
        xla::concurrency::ForEach<ComputationHashResult>(
            called_computations.begin(), called_computations.end(),
            hash_computation, *task_executor_));

    // `hash_to_canonical_string` lives only for the duration of the function.
    // Its values point to strings that live same duration in `hash_results`.
    absl::flat_hash_map<absl::uint128, absl::string_view>
        hash_to_canonical_string;
    ModuleHashResult result;
    for (ComputationHashResult& hash_result : hash_results) {
      result.computation_to_hash[hash_result.computation] = hash_result.hash;
      result.hash_to_canonical.try_emplace(hash_result.hash,
                                           hash_result.computation);
      hash_to_canonical_string.try_emplace(hash_result.hash,
                                           hash_result.fingerprint);
    }

    auto validate_against_canonical = [&](const ComputationHashResult& result) {
      absl::uint128 candidate_hash = result.hash;
      const std::string& candidate_fingerprint = result.fingerprint;
      const absl::string_view& canonical_fingerprint =
          hash_to_canonical_string.at(candidate_hash);

      if (candidate_fingerprint != canonical_fingerprint) {
        return absl::InternalError(absl::StrCat(
            "Hash collision detected. Hash: ", candidate_hash, "\n",
            "Hashes are equal but fingerprints are different.\n",
            "Computation 1:\n", candidate_fingerprint, "\n", "Computation 2:\n",
            canonical_fingerprint, "\n"));
      }
      return absl::OkStatus();
    };

    // Validate all computations against their canonical versions in parallel.
    TF_RETURN_IF_ERROR((xla::concurrency::ForEach(
        hash_results.begin(), hash_results.end(), validate_against_canonical,
        *task_executor_)));
    return result;
  }

 private:
  HloPrintOptions print_options_;
  // Thread pool used for parallelizing computation hashing and optionally
  // hash collision detection.
  std::unique_ptr<xla::concurrency::TslTaskExecutor> task_executor_;
};
// Struct to hold all call instructions and called computations in a module.
struct HloCalls {
  // All callsites are guaranteed to be `kCall` instructions.
  absl::flat_hash_set<HloInstruction*> call_sites;
  absl::flat_hash_set<HloComputation*> targets;
};

// Iterates through all instructions in the module's computations
// and collects all `HloInstruction`s with opcode `kCall` into 'calls_sites'
// and all unique computations targeted by these calls into 'targets'.
HloCalls CollectHloCalls(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module);
  HloCalls calls;
  absl::flat_hash_map<uint64_t, uint64_t> count_num_instructions;
  for (const CallGraphNode& node : call_graph->nodes()) {
    for (const CallSite& callsite : node.callsites()) {
      if (callsite.instruction()->opcode() == HloOpcode::kCall) {
        calls.call_sites.insert(callsite.instruction());
        calls.targets.insert(callsite.instruction()->to_apply());
        count_num_instructions
            [callsite.instruction()->to_apply()->instruction_count()]++;
      }
    }
  }
  // Remove computations for which there is no other computation with matching
  // number of instructions (i.e. it cannot have duplicate)
  for (auto it = calls.call_sites.begin(), end = calls.call_sites.end();
       it != end;) {
    // `erase()` will invalidate `it`, so advance `it` first.
    auto copy_it = it++;
    HloComputation* computation = (*copy_it)->to_apply();
    if (count_num_instructions[computation->instruction_count()] == 1) {
      calls.targets.erase(computation);
      calls.call_sites.erase(copy_it);
    }
  }
  return calls;
}
}  // namespace

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

  HloModuleHasher hasher;
  TF_ASSIGN_OR_RETURN(HloModuleHasher::ModuleHashResult hash_results,
                      check_hash_collision_
                          ? hasher.HashComputationsWithValidation(calls.targets)
                          : hasher.HashComputations(calls.targets));

  bool changed = false;
  // Lambda to find the canonical computation for a given computation.
  auto get_canonical_computation = [&](const HloComputation* original_called) {
    absl::uint128 hash = hash_results.computation_to_hash.at(original_called);
    HloComputation* canonical_called = hash_results.hash_to_canonical.at(hash);

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

  auto is_only_kcalled = [&](HloComputation* computation) {
    for (HloInstruction* caller : computation->caller_instructions()) {
      if (caller->opcode() != HloOpcode::kCall) {
        return false;
      }
    }
    return true;
  };

  if (changed) {
    // Clean up any computations that are now no longer called.
    for (HloComputation* computation : calls.targets) {
      // Only clean up computations that are only called by kCall instructions.
      // Leaving other calling instructions unchanged.
      if (!is_only_kcalled(computation)) {
        continue;
      }
      absl::uint128 hash = hash_results.computation_to_hash.at(computation);
      HloComputation* canonical = hash_results.hash_to_canonical.at(hash);
      if (computation != canonical) {
        TF_RETURN_IF_ERROR(module->RemoveEmbeddedComputation(computation));
      }
    }
    TF_RETURN_IF_ERROR(module->RemoveUnusedComputations());
    module->CleanupComputations();
  }

  return changed;
}
}  // namespace xla
