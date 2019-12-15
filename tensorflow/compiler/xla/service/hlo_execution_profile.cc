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

#include "absl/algorithm/container.h"
#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/service/hlo_execution_profile_data.pb.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/human_readable_profile_builder.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {
HloProfileIndexMap::HloProfileIndexMap(const HloModule& module,
                                       absl::Span<const string> extra_metrics) {
  size_t current_profile_index = 0;
  for (xla::HloComputation* computation : module.MakeComputationPostOrder()) {
    InsertOrDie(&computation_to_profile_idx_, computation,
                current_profile_index++);
    for (const HloInstruction* instruction : computation->instructions()) {
      // For simplicity we track all instructions here, but we could skip
      // non-executing instructions like constants and parameters.
      InsertOrDie(&instruction_to_profile_idx_, instruction,
                  current_profile_index++);
    }
  }
  for (const string& key : extra_metrics) {
    InsertOrDie(&extra_metric_to_profile_idx_, key, current_profile_index++);
  }
}

std::unique_ptr<HloProfilePrinterData> CreateHloProfilePrinterData(
    const HloProfileIndexMap& hlo_profile_index_map,
    const HloCostAnalysis& cost_analysis,
    const string& entry_computation_name) {
  using HloComputationInfo = HloProfilePrinterData::HloComputationInfo;
  using HloInstructionInfo = HloProfilePrinterData::HloInstructionInfo;

  size_t profile_counters_size = hlo_profile_index_map.total_count();

  std::unique_ptr<HloProfilePrinterData> profile_printer_data =
      absl::make_unique<HloProfilePrinterData>();
  profile_printer_data->set_profile_counters_size(profile_counters_size);
  profile_printer_data->mutable_computation_infos()->Reserve(
      hlo_profile_index_map.computation_count());

  const auto& computation_to_profile_idx_map =
      hlo_profile_index_map.computation_to_profile_idx();

  // computation_to_profile_idx_map's order is not deterministic so create a
  // deterministic computation_and_profile_idx_list so that we end up with a
  // deterministic HloProfilePrinterData protobuf.

  std::vector<std::pair<const HloComputation*, int64>>
      computation_and_profile_idx_list(computation_to_profile_idx_map.begin(),
                                       computation_to_profile_idx_map.end());

  // The profile indices were computed deterministically in
  // HloProfileIndexMap::HloProfileIndexMap.
  absl::c_sort(computation_and_profile_idx_list,
               [](const std::pair<const HloComputation*, int64>& left,
                  const std::pair<const HloComputation*, int64>& right) {
                 return left.second < right.second;
               });

  for (const auto& pair : computation_and_profile_idx_list) {
    CHECK_LT(pair.second, profile_counters_size);
    const HloComputation* computation = pair.first;
    HloComputationInfo* computation_info =
        profile_printer_data->add_computation_infos();

    computation_info->set_name(computation->name());
    computation_info->set_profile_index(pair.second);
    computation_info->mutable_instruction_infos()->Reserve(
        computation->instruction_count());

    for (const HloInstruction* hlo : computation->instructions()) {
      HloInstructionInfo* instruction_info =
          computation_info->add_instruction_infos();
      instruction_info->set_long_name(hlo->ToString());
      instruction_info->set_short_name(hlo->ToString(
          HloPrintOptions().set_compact_operands(true).set_print_operand_names(
              false)));
      instruction_info->set_category(hlo->ToCategory());
      instruction_info->set_flop_count(cost_analysis.flop_count(*hlo));
      instruction_info->set_transcendental_count(
          cost_analysis.transcendental_count(*hlo));
      instruction_info->set_bytes_accessed(cost_analysis.bytes_accessed(*hlo));
      instruction_info->set_optimal_seconds(
          cost_analysis.optimal_seconds(*hlo));
      instruction_info->set_profile_index(
          hlo_profile_index_map.GetProfileIndexFor(*hlo));
    }
  }

  // Add extra metrics if any.
  for (const auto& pair : hlo_profile_index_map.extra_metric_to_profile_idx()) {
    profile_printer_data->mutable_extra_metrics()->insert(
        {pair.first, pair.second});
  }

  profile_printer_data->set_entry_computation(entry_computation_name);

  return profile_printer_data;
}

HloExecutionProfile::HloExecutionProfile(
    const HloProfilePrinterData* hlo_profile_printer_data,
    const HloProfileIndexMap* hlo_profile_index_map)
    : hlo_profile_printer_data_(*hlo_profile_printer_data),
      hlo_profile_index_map_(*hlo_profile_index_map),
      profile_counters_(
          /*count=*/hlo_profile_index_map_.total_count(),
          /*value=*/0) {}

void HloExecutionProfile::SetCyclesTakenBy(const HloInstruction* hlo,
                                           uint64 cycles_taken) {
  profile_counters_[hlo_profile_index_map_.GetProfileIndexFor(*hlo)] =
      cycles_taken;
}

uint64 HloExecutionProfile::GetCyclesTakenBy(const HloInstruction& hlo) const {
  return profile_counters_[hlo_profile_index_map_.GetProfileIndexFor(hlo)];
}

HloExecutionProfileData HloExecutionProfile::ToProto() const {
  HloExecutionProfileData hlo_execution_profile_data;
  for (const auto& counter : profile_counters_) {
    hlo_execution_profile_data.add_profile_counters(counter);
  }
  *(hlo_execution_profile_data.mutable_printer_data()) =
      hlo_profile_printer_data_;
  return hlo_execution_profile_data;
}

}  // namespace xla
