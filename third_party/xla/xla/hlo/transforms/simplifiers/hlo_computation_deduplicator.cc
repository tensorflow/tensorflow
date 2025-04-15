/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/hlo/transforms/simplifiers/hlo_computation_deduplicator.h"

#include <string>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/shape_util.h"
#include "tsl/platform/logging.h"

namespace xla {

bool HloComputationDeduplicator::ContainsLargeConstants(HloComputation* comp) {
  int total_size = 0;
  for (HloInstruction* instruction : comp->instructions()) {
    if (instruction->IsConstant()) {
      total_size += ShapeUtil::ArrayDataSize(instruction->literal().shape());
      if (total_size > 1024) {
        return true;
      }
    }
  }
  return false;
}

absl::StatusOr<bool> HloComputationDeduplicator::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  absl::flat_hash_map<std::string, HloComputation*> unique_comps;
  absl::flat_hash_map<HloComputation*, HloComputation*> replacement;

  // Options to produce a String representation that is similar to
  // HloPrintOptions::Fingerprint() but does not eliminate constants and not
  // dive into sub-computations.
  HloPrintOptions options = HloPrintOptions::Canonical();
  options.set_print_subcomputation_mode(
      HloPrintOptions::PrintSubcomputationMode::kOff);
  options.set_print_infeed_outfeed_config(false);
  options.set_print_only_essential_constants(true);
  options.set_print_operand_shape(true);
  options.set_print_ids(false);
  options.set_print_backend_config(true);
  options.set_canonicalize_computations(true);

  // This comparison function will be used to compare called subcomputations.
  // Since computations in the for-loop below are called in "PostOrder" format
  // we would have visited callees before the caller. If the callees are marked
  // as the duplicates - using the replacement map - and if the rest of the
  // instructions in computations are same then we can mark them as duplicates,
  // otherwise they both are distinct. The advantage is we do not need to dive
  // into sub-computations, thereby saving comparison time
  auto comp_eq = [&replacement](const HloComputation* a,
                                const HloComputation* b) {
    if (a->unique_id() == b->unique_id()) return true;
    if (replacement.contains(a) &&
        replacement.at(a)->unique_id() == b->unique_id()) {
      return true;
    }
    if (replacement.contains(b) &&
        replacement.at(b)->unique_id() == a->unique_id()) {
      return true;
    }
    if (replacement.contains(a) && replacement.contains(b) &&
        replacement.at(a)->unique_id() == replacement.at(b)->unique_id()) {
      return true;
    }
    return false;
  };
  for (HloComputation* comp :
       module->MakeComputationPostOrder(execution_threads)) {
    // Ignore entry computation since it is called from outside and computations
    // with large number of instructions or large-size constants due to increase
    // in time taken to stringify.
    if (comp->IsEntryComputation() || comp->instruction_count() > 128 ||
        ContainsLargeConstants(comp)) {
      continue;
    }
    // Don't deduplicate collectives and non-collectives.
    if (absl::c_any_of(
            comp->caller_instructions(), [](const HloInstruction* instr) {
              return hlo_query::IsCollectiveCommunicationOp(instr->opcode());
            })) {
      continue;
    }

    std::string comp_str = comp->ToString(options);
    auto poss_dup = unique_comps.find(comp_str);
    if (poss_dup != unique_comps.end() &&
        poss_dup->second->Equal(*comp, /* is_layout_sensitive = */ true,
                                comp_eq)) {
      VLOG(2) << "Replacing " << comp->name() << " with "
              << poss_dup->second->name();
      replacement[comp] = poss_dup->second;
    } else {
      unique_comps[std::move(comp_str)] = comp;
    }
  }
  if (mark_fusion_duplications_) {
    module->MarkFusionDuplications(replacement);
  } else {
    module->ReplaceComputations(replacement);
  }
  return !replacement.empty();
}
}  // namespace xla
