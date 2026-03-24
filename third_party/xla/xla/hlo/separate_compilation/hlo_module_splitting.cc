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

#include "xla/hlo/separate_compilation/hlo_module_splitting.h"

#include <cstdint>
#include <deque>
#include <memory>
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
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_clone_context.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/separate_compilation/hlo_linking_manifest.h"
#include "xla/service/compilation_environments.h"
#include "xla/service/hlo_module_config.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::separate_compilation {
namespace {

constexpr absl::string_view kEntryName = "entry";
constexpr absl::string_view kModulePrefix = "module";
constexpr absl::string_view kStubPrefix = "stub";

// Provide a name for the module containing the split.
// The name should be stable across compilations meaning that the same
// split should get the same name.
std::string GetSplitModuleName(
    absl::flat_hash_set<const HloComputation*> split) {
  // If multiple `HloComputation` elements are in a split, we have to worry
  // about their ordering when hashing, or use some ordering-invariant hash.
  CHECK(split.size() == 1) << "The current implementation only supports "
                              "singleton splits.";
  return absl::StrCat(kModulePrefix, absl::HashOf(*split.begin()));
}

std::string GetStubName(int32_t callee_index) {
  return absl::StrCat(kStubPrefix, ".", callee_index);
}

absl::StatusOr<std::unique_ptr<HloComputation>> CreateCalleeStub(
    HloComputation* callee, int32_t callee_index) {
  // Bind to keep alive for the duration of the scope.
  std::string stub_name = GetStubName(callee_index);
  auto comp_builder = HloComputation::Builder(stub_name);

  std::vector<HloInstruction*> operands;
  for (const HloInstruction* parameter : callee->parameter_instructions()) {
    TF_ASSIGN_OR_RETURN(
        HloInstruction * cloned_parameter,
        comp_builder.AddParameter(parameter->Clone(/*suffix=*/"")));
    operands.push_back(cloned_parameter);
  }

  comp_builder.AddInstruction(HloInstruction::CreateCustomCall(
      callee->root_instruction()->shape(), operands,
      /*custom_call_target=*/kStubPrefix));
  return comp_builder.Build();
}

// Returns all `kCall` instructions in the given computation.
std::vector<const HloInstruction*> CollectCallInstructions(
    const HloComputation* computation) {
  // TODO: b/419473710 - Maybe guarantee order of operand traversal?
  std::vector<const HloInstruction*> call_sites;
  for (const HloInstruction* caller : computation->MakeInstructionPostOrder()) {
    if (caller->opcode() == HloOpcode::kCall) {
      call_sites.push_back(caller);
    }
  }
  return call_sites;
}

// Composes two maps into one. The first map's value type must be second map's
// key type.
template <typename K, typename KV, typename V>
absl::StatusOr<absl::flat_hash_map<K, V>> ComposeMaps(
    const absl::flat_hash_map<K, KV>& first,
    const absl::flat_hash_map<KV, V>& second) {
  absl::flat_hash_map<K, V> result;
  for (const auto [k, kv] : first) {
    if (auto it = second.find(kv); it != second.end()) {
      result.insert({k, it->second});
    }
  }
  return result;
}

// Merges `from` into `into`. If `error_on_duplicate_key` is true, returns an
// error if any key is encountered more than once.
template <typename K, typename V>
absl::Status MergeMapInto(absl::flat_hash_map<K, V>& into,
                          const absl::flat_hash_map<K, V>& from,
                          bool error_on_duplicate_key = true) {
  for (const auto& [k, v] : from) {
    if (!into.insert({k, v}).second) {
      if (error_on_duplicate_key) {
        return absl::AlreadyExistsError("Duplicate key encountered.");
      }
    }
  }
  return absl::OkStatus();
}
}  // namespace

// Assigns entry & all kCall-ed computations to splits in stable way.
// Computations are assigned to exactly one split.
//
// Other computations, not explicitly kCall-ed, are not assigned and will be
// cloned with any explicitly assigned computation that depends on them. Note
// that this means that they might be replicated to multiple splits!
absl::StatusOr<std::vector<absl::flat_hash_set<const HloComputation*>>>
GroupComputationsForSplitting(const HloModule& module) {
  std::vector<absl::flat_hash_set<const HloComputation*>> result;

  const HloComputation* entry_computation = module.entry_computation();
  TF_RET_CHECK(entry_computation != nullptr)
      << "Module has no entry computation.";

  // Perform a BFS traversal of the graph along kCall edges.
  // All other edges are ignored.
  std::deque<const HloComputation*> computations_to_visit;
  absl::flat_hash_set<const HloComputation*> seen;

  computations_to_visit.push_back(entry_computation);
  seen.insert(entry_computation);

  while (!computations_to_visit.empty()) {
    const HloComputation* current_computation = computations_to_visit.front();
    computations_to_visit.pop_front();

    // Each reachable computation is added as a separate split.
    // If grouping is possible, this logic might change.
    result.push_back({current_computation});

    // Process callees.
    for (const HloInstruction* op : current_computation->instructions()) {
      if (op->opcode() == HloOpcode::kCall) {
        const HloComputation* callee = op->to_apply();
        TF_RET_CHECK(callee != nullptr)
            << "HloOpcode::kCall has a null callee.";
        if (!seen.contains(callee)) {
          seen.insert(callee);
          computations_to_visit.push_back(callee);
        }
      }
    }
  }
  return result;
}

absl::StatusOr<std::unique_ptr<HloModuleSplit>> CreateHloModuleSplit(
    const HloModule& module, absl::flat_hash_set<const HloComputation*> split) {
  // If multiple `HloComputation` elements are in a split, we have to worry
  // about their ordering when hashing, or use some ordering-invariant hash.
  CHECK(split.size() == 1)
      << "The current implementation supports singleton splits.";
  // TODO: b/419184359 - Revisit when we reconfigure the pipeline
  // (global->local->global). Check what data is needed by which
  // set of passes.
  std::shared_ptr<const HloModuleConfig> sub_module_config =
      module.shared_config();
  auto sub_module_env =
      std::make_unique<CompilationEnvironments>(module.comp_envs());
  auto submodule = std::make_unique<HloModule>(
      GetSplitModuleName(split), sub_module_config, std::move(sub_module_env));
  HloCloneContext clone_context(submodule.get());
  // The plan is:
  // 1. Prepare stubs as substitutions for callees.
  // 2. Clone the computation(s) with replacements of calls.
  // 3. Set the ENTRY computation.

  const HloComputation* computation = *split.begin();
  VLOG(4) << "Splitting out: " << computation->name();
  std::vector<const HloInstruction*> call_instructions =
      CollectCallInstructions(computation);
  // stub -> original callee
  absl::flat_hash_map<const HloComputation*, const HloComputation*> stub_map;
  // original computation -> split computation
  absl::flat_hash_map<const HloComputation*, const HloComputation*>
      computation_map;

  // We want to give a unique name to every call site by inserting a unique
  // stub call into every call site. However, `HloCloneContext` wants a map
  // of callee_computation -> new_computation, and this does not allow different
  // call sites to replace the same `callee_computation` with a different
  // `new_computation_2` at a different call site.
  //
  // We create a fresh stub for every call site. Since we only handle kCall
  // instructions, it can have single callee (call site). We map
  // original callee to these new stub, even though some aliasing might
  // happen. Specifically, if multiple call sites refer to the same callee
  // that callee will map to the stub for the last encountered call site.
  //
  // We tolerate this to avoid cloning the actual callees into the new module.
  // After cloning we go into the cloned computation and patch callee pointers.

  absl::flat_hash_map<const HloInstruction*, HloComputation*>
      callee_replacements;
  int32_t callee_index = 0;
  for (const HloInstruction* caller : call_instructions) {
    HloComputation* callee = caller->to_apply();
    // Skip callee that is part of the current split.
    if (split.contains(callee)) {
      // Remember the original callee, so that we can patch the call site later.
      callee_replacements[caller] = callee;
      continue;
    }
    TF_ASSIGN_OR_RETURN(std::unique_ptr<HloComputation> stub,
                        CreateCalleeStub(callee, callee_index));
    VLOG(4) << "Stubbing " << stub->name() << " --> " << callee->name() << " "
            << stub->ToString();
    HloComputation* stub_raw_ptr =
        submodule->AddComputationAndUnifyNamesAndIds(std::move(stub),
                                                     /*is_entry=*/false);
    clone_context.MapComputation(callee, stub_raw_ptr);
    callee_replacements[caller] = stub_raw_ptr;
    stub_map.insert({stub_raw_ptr, callee});
    ++callee_index;
  }

  HloComputation* entry_computation = submodule->AddEntryComputationWithLayouts(
      computation->CloneInContext(clone_context, nullptr,
                                  /*extra_parameters=*/{}, /*suffix=*/""));

  // Patch call sites.
  for (const HloInstruction* caller : call_instructions) {
    HloInstruction* mapped_call_instruction =
        clone_context.GetInstruction(caller);
    // Adjust `callee_replacement` to only contain pointer to computations in
    // the submodule. This is currently not the case because we skipped some
    // callees when stubbing and remembered pointer to their original.
    HloComputation* replacement = callee_replacements[caller];
    if (replacement->parent() != submodule.get()) {
      replacement = clone_context.GetComputation(replacement);
    }
    mapped_call_instruction->set_to_apply(callee_replacements[caller]);
  }

  entry_computation->SetAndSanitizeName(kEntryName);
  computation_map.insert({computation, entry_computation});

  VLOG(3) << submodule->ToString();
  return std::make_unique<HloModuleSplit>(
      module, std::move(submodule), std::move(stub_map),
      std::move(computation_map), std::move(call_instructions));
}

absl::StatusOr<std::unique_ptr<HloModuleSplitGroup>> CreateHloModuleSplitGroup(
    const HloModule& module) {
  absl::flat_hash_map<const HloComputation*, const HloModuleSplit*>
      computation_address_book;
  std::vector<std::unique_ptr<HloModuleSplit>> module_splits;

  // See `HloModuleSplit::stub_map`.
  absl::flat_hash_map<const HloComputation*, const HloComputation*>
      global_stub_map;
  absl::flat_hash_map<const HloComputation*, const HloComputation*>
      global_computation_map;

  TF_ASSIGN_OR_RETURN(
      std::vector<absl::flat_hash_set<const HloComputation*>> splits,
      GroupComputationsForSplitting(module));

  for (const auto& split : splits) {
    TF_ASSIGN_OR_RETURN(auto module_split, CreateHloModuleSplit(module, split));
    module_splits.push_back(std::move(module_split));
    for (const auto* original_comp : split) {
      computation_address_book.insert(
          {original_comp, module_splits.back().get()});
    }
    TF_RETURN_IF_ERROR(
        MergeMapInto(global_stub_map, module_splits.back()->stub_map));
    TF_RETURN_IF_ERROR(MergeMapInto(global_computation_map,
                                    module_splits.back()->computation_map));
  }

  if (VLOG_IS_ON(5)) {
    VLOG(5) << "Split group:";
    for (const auto& split : module_splits) {
      VLOG(5) << "Split: " << split->submodule->name();
      VLOG(5) << " Stub links:";
      for (const auto& [stub, comp] : split->stub_map) {
        VLOG(5)
            << "  " << stub->name() << " ==>> " << comp->name() << "("
            << computation_address_book[comp]->computation_map.at(comp)->name()
            << " @ " << computation_address_book[comp]->submodule->name()
            << ")";
      }
    }
  }
  // Compose at the end once all planned cloning operations are finished and
  // we know where each original computation ended up.
  TF_ASSIGN_OR_RETURN(auto stub_links,
                      ComposeMaps(global_stub_map, global_computation_map));

  HloLinkingManifest linking_manifest{
      std::move(stub_links), module.shared_config(),
      std::make_unique<CompilationEnvironments>(module.comp_envs())};

  return std::make_unique<HloModuleSplitGroup>(
      std::move(computation_address_book), std::move(module_splits),
      std::move(linking_manifest));
}

absl::StatusOr<const HloComputation*> HloModuleSplitGroup::GetClonedComputation(
    const HloComputation* original_computation) const {
  auto it = address_book.find(original_computation);
  if (it == address_book.end()) {
    return absl::NotFoundError(
        absl::StrCat("Original computation '", original_computation->name(),
                     "' not found in HloModuleSplitGroup address book."));
  }
  auto& computation_map = it->second->computation_map;
  auto it2 = computation_map.find(original_computation);
  if (it2 == computation_map.end()) {
    return absl::InternalError(absl::StrCat(
        "Original computation '", original_computation->name(),
        "' found in address book but not in computation map for its "
        "module split '",
        it->second->submodule->name(), "'."));
  }
  return it2->second;
}

}  // namespace xla::separate_compilation
