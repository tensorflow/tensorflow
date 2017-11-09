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

void HloExecutionProfile::SetCyclesTakenBy(const HloInstruction* hlo,
                                           uint64 cycles_taken) {
  hlo_to_cycles_taken_[hlo] = cycles_taken;
  profiled_computations_.insert(hlo->parent());
}

uint64 HloExecutionProfile::GetCyclesTakenBy(const HloInstruction& hlo) const {
  auto iter = hlo_to_cycles_taken_.find(&hlo);
  if (iter == hlo_to_cycles_taken_.end()) {
    return 0;
  }
  return iter->second;
}

string HloExecutionProfile::ToString(
    const HloComputation& computation,
    const DeviceDescription& device_description,
    HloCostAnalysis* cost_analysis) const {
  tensorflow::Status analysis_status = computation.Accept(cost_analysis);
  if (!analysis_status.ok()) {
    return "";
  }

  HumanReadableProfileBuilder builder(computation.name(),
                                      total_cycles_executed(computation),
                                      device_description.clock_rate_ghz());
  for (const auto& item : hlo_to_cycles_taken_) {
    const HloInstruction* hlo = item.first;
    int64 cycles = item.second;

    builder.AddOp(/*op_name=*/hlo->ToString(),
                  /*short_name=*/hlo->ToString(/*compact_operands=*/true),
                  hlo->ToCategory(), cycles, cost_analysis->flop_count(*hlo),
                  cost_analysis->transcendental_count(*hlo),
                  cost_analysis->bytes_accessed(*hlo),
                  cost_analysis->seconds(*hlo));
  }
  return builder.ToString();
}

}  // namespace xla
