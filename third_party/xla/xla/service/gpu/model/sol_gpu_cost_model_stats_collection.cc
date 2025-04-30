/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/gpu/model/sol_gpu_cost_model_stats_collection.h"

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/model/sol_gpu_cost_model.h"
#include "xla/service/gpu/model/sol_latency_estimator.h"
#include "xla/tsl/platform/status.h"

namespace xla::gpu {

absl::StatusOr<bool> SolGpuCostModelStatsCollection::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  SolGPUCostModel::Config config = SolGPUCostModel::GetConfig(module);

  hlo_query::ForEachInstructionWithPred(
      *module,
      [](const HloInstruction* instr) {
        return hlo_query::IsCollectiveCommunicationOp(instr->opcode());
      },
      [&](HloInstruction* instr) {
        // Generate exec time for a collective.
        absl::Duration exec_time = SolLatencyEstimator::ComputeCollectiveTime(
            *instr, device_info_, shape_size_in_bytes_fn_, config);

        // Set it in the `CollectiveBackendConfig`.
        auto gpu_config = instr->backend_config<GpuBackendConfig>();
        TF_CHECK_OK(gpu_config.status()) << instr->ToString();
        auto reification_cost = gpu_config->add_reification_cost();
        *reification_cost->mutable_name() = name();
        reification_cost->set_exec_time_us(
            absl::ToDoubleMicroseconds(exec_time));
        TF_CHECK_OK(instr->set_backend_config(*gpu_config));
      });

  return false;
}

}  // namespace xla::gpu
