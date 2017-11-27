/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/hlo_execution_profile.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/human_readable_profile_builder.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {
HloProfileIndexMap::HloProfileIndexMap(const HloModule& module) {
  size_t current_profile_index = 0;
  for (xla::HloComputation* computation : module.MakeComputationPostOrder()) {
    InsertOrDie(&computation_to_profile_idx_, computation,
                current_profile_index++);
    for (const HloInstruction* instruction : computation->instructions()) {
      // For simplicity we track all instrutions here, but we could skip
      // non-executing instructions like constants and parameters.
      InsertOrDie(&instruction_to_profile_idx_, instruction,
                  current_profile_index++);
    }
  }
}

std::unique_ptr<HloProfilePrinter> CreateHloProfilePrinter(
    const HloProfileIndexMap& hlo_profile_index_map,
    const HloCostAnalysis& cost_analysis) {
  using HloComputationInfo = HloProfilePrinter::HloComputationInfo;
  using HloInstructionInfo = HloProfilePrinter::HloInstructionInfo;

  HloComputationInfo* computation_infos =
      new HloComputationInfo[hlo_profile_index_map.computation_count()];

  // There are two "indices" in play here.  The first one is the index of the
  // HloComputationInfo or HloInstructionInfo in the array that contains said
  // HloComputationInfo or HloInstructionInfo.  The second index is the index of
  // the HloComputationInfo or HloInstructionInfo in the profile counters array,
  // as decided by hlo_profile_index_map.  The latter index is always referred
  // to as "profile_index".

  size_t computation_index_in_static_data = 0;
  size_t max_profile_index = hlo_profile_index_map.total_count();
  for (const auto& pair : hlo_profile_index_map.computation_to_profile_idx()) {
    CHECK_LT(pair.second, max_profile_index);
    const HloComputation* computation = pair.first;
    size_t current_computation_index = computation_index_in_static_data++;
    HloComputationInfo* computation_info =
        &computation_infos[current_computation_index];

    computation_info->name = strdup(computation->name().c_str());
    computation_info->profile_index = pair.second;
    computation_info->instructions =
        new HloInstructionInfo[computation->instruction_count()];
    computation_info->instructions_size = computation->instruction_count();

    size_t instruction_index_in_static_data = 0;
    for (const HloInstruction* hlo : computation->instructions()) {
      HloProfilePrinter::HloInstructionInfo* instruction_info =
          &computation_info->instructions[instruction_index_in_static_data++];
      instruction_info->long_name = strdup(hlo->ToString().c_str());
      instruction_info->short_name =
          strdup(hlo->ToString(/*compact_operands=*/true).c_str());
      instruction_info->category = strdup(hlo->ToCategory().c_str());
      instruction_info->flop_count = cost_analysis.flop_count(*hlo);
      instruction_info->transcendental_count =
          cost_analysis.transcendental_count(*hlo);
      instruction_info->bytes_accessed = cost_analysis.bytes_accessed(*hlo);
      instruction_info->optimal_seconds = cost_analysis.optimal_seconds(*hlo);
      instruction_info->profile_index =
          hlo_profile_index_map.GetProfileIndexFor(*hlo);
      CHECK_LT(instruction_info->profile_index, max_profile_index);
    }
  }

  auto deleter = [](HloProfilePrinter::HloComputationInfo* computation_infos,
                    int64 computation_infos_size) {
    for (int64 i = 0; i < computation_infos_size; i++) {
      HloInstructionInfo* instruction_infos = computation_infos[i].instructions;
      for (int64 j = 0; j < computation_infos[i].instructions_size; j++) {
        // We can't make instruction_infos[j].long_name etc. non-const pointers
        // since they may point into static storage, so we have a const_cast
        // here.
        free(const_cast<char*>(instruction_infos[j].long_name));
        free(const_cast<char*>(instruction_infos[j].short_name));
        free(const_cast<char*>(instruction_infos[j].category));
      }
      delete[] instruction_infos;
      free(const_cast<char*>(computation_infos[i].name));
    }
    delete[] computation_infos;
  };

  return MakeUnique<HloProfilePrinter>(
      computation_infos, hlo_profile_index_map.computation_count(), deleter);
}

HloExecutionProfile::HloExecutionProfile(
    const HloProfilePrinter* hlo_profile_printer,
    const HloProfileIndexMap* hlo_profile_index_map)
    : hlo_profile_printer_(*hlo_profile_printer),
      hlo_profile_index_map_(*hlo_profile_index_map),
      profile_counters_(
          /*count*/ hlo_profile_index_map_.total_count(),
          /*value*/ 0) {}

void HloExecutionProfile::SetCyclesTakenBy(const HloInstruction* hlo,
                                           uint64 cycles_taken) {
  profile_counters_[hlo_profile_index_map_.GetProfileIndexFor(*hlo)] =
      cycles_taken;
}

uint64 HloExecutionProfile::GetCyclesTakenBy(const HloInstruction& hlo) const {
  return profile_counters_[hlo_profile_index_map_.GetProfileIndexFor(hlo)];
}

}  // namespace xla
