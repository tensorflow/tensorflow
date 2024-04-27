/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/gpu/model/gpu_cost_model_stats_collection.h"

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/model/gpu_performance_model.h"
#include "xla/service/gpu/model/gpu_performance_model_base.h"
#include "tsl/platform/status.h"

namespace xla {
namespace gpu {

absl::StatusOr<bool> GpuCostModelStatsCollection::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  // Scan all computations for fusion instructions.
  for (auto* computation : module->MakeComputationPostOrder()) {
    TF_CHECK_OK(computation->Accept(&cost_analysis_));

    for (auto* fusion_instr : computation->instructions()) {
      if (fusion_instr->opcode() != HloOpcode::kFusion) continue;

      GpuPerformanceModel::RecordEstimatedRunTime(
          fusion_instr, &cost_analysis_,
          GpuPerformanceModelOptions::ForModule(module));
    }
  }
  return false;
}

}  // namespace gpu
}  // namespace xla
