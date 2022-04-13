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

#include "tensorflow/compiler/xla/service/hlo_profile_printer.h"

#include "absl/algorithm/container.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/service/human_readable_profile_builder.h"

namespace xla {
std::string PrintHloProfile(
    const HloProfilePrinterData& hlo_profile_printer_data,
    const int64_t* counters, double clock_rate_ghz) {
  using HloComputationInfo = HloProfilePrinterData::HloComputationInfo;
  using HloInstructionInfo = HloProfilePrinterData::HloInstructionInfo;

  std::string result;

  for (const auto& item : hlo_profile_printer_data.extra_metrics()) {
    absl::StrAppend(&result, "Extra metric ", item.first, ": ",
                    counters[item.second], "\n");
  }

  for (const HloComputationInfo& computation_info :
       hlo_profile_printer_data.computation_infos()) {
    const auto& instruction_infos = computation_info.instruction_infos();
    bool any_instruction_profiled = absl::c_any_of(
        instruction_infos, [&](const HloInstructionInfo& instruction_info) {
          return counters[instruction_info.profile_index()] != 0;
        });

    if (!any_instruction_profiled) {
      continue;
    }

    // Once we start using this in AOT for real, we will probably need a more
    // minimal version of HumanReadableProfileBuilder.
    HumanReadableProfileBuilder builder(
        computation_info.name(),
        hlo_profile_printer_data.entry_computation() == computation_info.name(),
        counters[computation_info.profile_index()], clock_rate_ghz);

    for (const auto& instruction_info : instruction_infos) {
      builder.AddOp(
          /*op_name=*/instruction_info.long_name(),
          /*short_name=*/instruction_info.short_name(),
          instruction_info.category(),
          counters[instruction_info.profile_index()],
          instruction_info.flop_count(),
          instruction_info.transcendental_count(),
          instruction_info.bytes_accessed(),
          instruction_info.optimal_seconds());
    }

    result += builder.ToString();
  }

  return result;
}
}  // namespace xla
