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

#include "tensorflow/compiler/xla/service/human_readable_profile_builder.h"

namespace xla {
string HloProfilePrinter::ToString(const int64* counters,
                                   double clock_rate_ghz) const {
  string result;

  for (int computation_idx = 0; computation_idx < computation_infos_size_;
       computation_idx++) {
    const HloComputationInfo& computation = computation_infos_[computation_idx];
    const HloInstructionInfo* instructions_begin = computation.instructions;
    const HloInstructionInfo* instructions_end =
        computation.instructions + computation.instructions_size;
    bool any_instruction_profiled =
        std::any_of(instructions_begin, instructions_end,
                    [&](const HloInstructionInfo& instruction_info) {
                      return counters[instruction_info.profile_index] != 0;
                    });

    if (!any_instruction_profiled) {
      continue;
    }

    // Once we start using this in AOT for real, we will probably need a more
    // minimal version of HumanReadableProfileBuilder.
    HumanReadableProfileBuilder builder(
        computation.name, counters[computation.profile_index], clock_rate_ghz);

    for (const auto* instruction = instructions_begin;
         instruction != instructions_end; instruction++) {
      builder.AddOp(
          /*op_name=*/instruction->long_name,
          /*short_name=*/instruction->short_name, instruction->category,
          counters[instruction->profile_index], instruction->flop_count,
          instruction->transcendental_count, instruction->bytes_accessed,
          instruction->optimal_seconds);
    }

    result += builder.ToString();
  }

  return result;
}

HloProfilePrinter::~HloProfilePrinter() {
  if (deleter_) {
    deleter_(computation_infos_, computation_infos_size_);
  }
}
}  // namespace xla
