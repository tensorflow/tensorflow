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

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_hlo_schedule.h"
#include "xla/service/gpu/gpu_latency_hiding_scheduler.h"
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/service/gpu/model/sol_latency_estimator.h"
#include "xla/service/latency_hiding_scheduler.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {

namespace {

// Returns true if successfully set the reification cost.
bool SetReificationCost(HloInstruction* instr, double cost_us) {
  auto gpu_config = instr->backend_config<GpuBackendConfig>();
  if (!gpu_config.ok()) {
    return false;
  }
  auto reification_cost = gpu_config->add_reification_cost();
  reification_cost->set_exec_time_us(cost_us);
  reification_cost->set_name("sol");
  return instr->set_backend_config(*gpu_config).ok();
}

// Returns true if reification cost has been successfully recorded.
bool RecordReificationCost(HloInstruction& instr,
                           SolLatencyEstimator& estimator) {
  if (instr.user_count() == 1) {
    HloGraphNode from(&instr, /*original_position=*/-1);
    HloGraphNode to(instr.users()[0], /*original_position=*/-1);
    if (estimator.IsAsyncPair(from, to)) {
      return SetReificationCost(&instr, estimator.GetLatencyBetween(from, to));
    }
  }
  return SetReificationCost(&instr, estimator.NodeCost(&instr));
}

}  // namespace

absl::StatusOr<bool> SolGpuCostModelStatsCollection::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  auto cost_analysis =
      std::make_unique<GpuHloCostAnalysis>(GpuHloCostAnalysis::Options{
          shape_size_in_bytes_fn_,
          /*per_second_rates=*/{},
          /*min_latencies_seconds=*/{},
          /*count_multiple_input_accesses=*/true,
      });
  CHECK_OK(module->entry_computation()->Accept(cost_analysis.get()));
  uint64_t memory_limit =
      GetSchedulerMemoryLimit(*module, device_info_, pointer_size_);

  SchedulerConfig scheduler_config = MakeGPUSchedulerConfig(
      memory_limit,
      module->config()
          .debug_options()
          .xla_gpu_experimental_parallel_collective_overlap_limit());
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<SolLatencyEstimator> estimator,
      SolLatencyEstimator::Create(
          scheduler_config,
          std::make_unique<GpuLatencyEstimator>(pointer_size_), device_info_,
          shape_size_in_bytes_fn_, module->entry_computation(),
          std::move(cost_analysis)));

  for (HloComputation* comp : module->computations()) {
    for (HloInstruction* instr : comp->MakeInstructionPostOrder()) {
      if (!RecordReificationCost(*instr, *estimator)) {
        VLOG(2) << "Cannot record reification cost for: " << instr->ToString();
      }
    }
  }

  return false;
}

}  // namespace xla::gpu
