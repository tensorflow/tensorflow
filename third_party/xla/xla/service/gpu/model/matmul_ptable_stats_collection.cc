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

#include "xla/service/gpu/model/matmul_ptable_stats_collection.h"

#include <memory>
#include <string>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/model/hlo_op_profile.pb.h"
#include "xla/service/gpu/model/hlo_op_profiles.h"
#include "xla/service/gpu/model/matmul_interpolator.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla::gpu {

namespace {

absl::StatusOr<HloInstructionProfileList> CollectProfiles(
    const std::string& perf_table_path,
    const se::DeviceDescription& device_info) {
  DeviceHloInstructionProfiles profile;

  TF_RETURN_IF_ERROR(tsl::Env::Default()->FileExists(perf_table_path));
  TF_RETURN_IF_ERROR(tsl::ReadTextOrBinaryProto(tsl::Env::Default(),
                                                perf_table_path, &profile));
  std::string key = HloOpProfiles::GetProfileName(device_info);

  if (!profile.entries().contains(key)) {
    return absl::NotFoundError(absl::StrCat("Cannot find key: ", key));
  }
  return profile.entries().at(key);
}

ReificationCost* GetReificationCost(HloOpcode opcode,
                                    GpuBackendConfig& config) {
  if (opcode == HloOpcode::kCustomCall) {
    return config.mutable_gemm_backend_config()->add_reification_cost();
  }
  if (opcode == HloOpcode::kFusion) {
    return config.mutable_fusion_backend_config()->mutable_reification_cost();
  }
  return nullptr;
}

}  // namespace

absl::StatusOr<bool> MatmulPerfTableStatsCollection::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  TF_ASSIGN_OR_RETURN(HloInstructionProfileList profiles,
                      CollectProfiles(perf_table_path_, device_info_));
  TF_ASSIGN_OR_RETURN(std::unique_ptr<MatmulInterpolator> interpolator,
                      MatmulInterpolator::Create(profiles, device_info_));

  hlo_query::ForEachInstructionWithPred(
      *module, HloPredicateIsOp<HloOpcode::kCustomCall, HloOpcode::kFusion>,
      [&](HloInstruction* instr) {
        // Generate exec time for a matmul.
        auto estimation = interpolator->EstimatedRuntime(*instr);
        if (!estimation.has_value()) {
          VLOG(1) << "No estimation for: " << instr->ToString();
          return;
        }
        absl::Duration exec_time = *estimation;

        // Set it in the `CollectiveBackendConfig`.
        auto gpu_config = instr->backend_config<GpuBackendConfig>();
        TF_CHECK_OK(gpu_config.status())
            << "Cannot parse backend config: " << instr->ToString();
        ReificationCost* reification_cost =
            GetReificationCost(instr->opcode(), *gpu_config);
        if (reification_cost == nullptr) {
          VLOG(1) << "No reification cost for: " << instr->ToString();
          return;
        }
        reification_cost->set_exec_time_us(
            absl::ToDoubleMicroseconds(exec_time));
        *reification_cost->mutable_name() = name();
        TF_CHECK_OK(instr->set_backend_config(*gpu_config));
      });

  return false;
}

}  // namespace xla::gpu
