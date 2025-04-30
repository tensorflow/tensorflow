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
#include <optional>
#include <string>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/gpu/codegen/triton/support.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/model/gpu_dot_fusion_cost_model.h"
#include "xla/service/gpu/model/hlo_op_profile.pb.h"
#include "xla/service/gpu/model/hlo_op_profiles.h"
#include "xla/service/gpu/model/matmul_interpolator.h"
#include "xla/service/gpu/model/tiled_hlo_computation.h"
#include "xla/service/gpu/transforms/nest_gemm_fusion.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

namespace {

constexpr absl::string_view kGemmCostModelName = "gemm-cost-model";

constexpr absl::string_view kPerfTablesModelName = "perf-table-model";

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

HloDotInstruction* GetTritonGemmInstruction(const HloInstruction& dot_fusion) {
  if (!(HloPredicateIsOp<HloOpcode::kFusion>(&dot_fusion) &&
        IsTritonFusedComputation(
            *dot_fusion.fused_instructions_computation()))) {
    return nullptr;
  }

  HloInstruction* dot = hlo_query::GetFirstInstructionWithOpcode(
      *dot_fusion.fused_instructions_computation(), HloOpcode::kDot);
  if (dot == nullptr) {
    return nullptr;
  }
  return DynCast<HloDotInstruction>(dot);
}

absl::StatusOr<BlockLevelParameters> GetBlockLevelParams(
    HloDotInstruction& dot, TritonGemmConfig& config) {
  mlir::MLIRContext ctx;
  return ::xla::gpu::detail::FindBlockLevelParameters(&dot, config, &ctx);
}

absl::Status SetReificationCost(HloInstruction& instr, absl::Duration exec_time,
                                absl::string_view reification_name) {
  TF_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                      instr.backend_config<GpuBackendConfig>());
  ReificationCost* reification_cost =
      GetReificationCost(instr.opcode(), gpu_config);
  if (reification_cost == nullptr) {
    return absl::InternalError(
        absl::StrCat("Cannot add reification cost to: ", instr.ToString()));
  }
  reification_cost->set_exec_time_us(absl::ToDoubleMicroseconds(exec_time));
  *reification_cost->mutable_name() = reification_name;
  return instr.set_backend_config(gpu_config);
}

// Computes the runtime estimation via analytical GEMM cost model and adds a
// reification cost to `instr`. We do not make any constraints on what fusions
// do we add the cost to. In particular it can be the case there's a non trivial
// fusion on dot operands. As of now the analytical GEMM model does not support
// these cases so result interpretation has take this into consideration.
absl::Status MaybeRecordGemmCostModelForGemmTritonFusion(
    const se::DeviceDescription& device_info, HloInstruction& instr) {
  HloDotInstruction* dot = GetTritonGemmInstruction(instr);
  if (dot == nullptr) {
    VLOG(2) << "Cannot get triton gemm: " << instr.ToString();
    return absl::OkStatus();
  }
  auto triton_gemm_key = instr.backend_config<GpuBackendConfig>()
                             ->fusion_backend_config()
                             .triton_gemm_config();
  TF_ASSIGN_OR_RETURN(TritonGemmConfig triton_gemm_config,
                      TritonGemmConfig::FromProto(triton_gemm_key));
  TF_ASSIGN_OR_RETURN(BlockLevelParameters block_params,
                      GetBlockLevelParams(*dot, triton_gemm_config));
  TF_ASSIGN_OR_RETURN(
      absl::Duration exec_time,
      GpuDotFusionCostModel::EstimateRunTimeForDotOpWithBlockParameters(
          dot, block_params, device_info));
  return SetReificationCost(instr, exec_time, kGemmCostModelName);
}

absl::Status MaybeRecordPerfTablesForDotsAndCustomCalls(
    const se::DeviceDescription& device_info, HloInstruction& instr,
    MatmulInterpolator& interpolator) {
  if (HloPredicateIsNotOp<HloOpcode::kCustomCall, HloOpcode::kDot>(&instr)) {
    VLOG(2) << "Not a dot or custom call: " << instr.ToString();
    return absl::OkStatus();
  }
  std::optional<absl::Duration> exec_time =
      interpolator.EstimatedRuntime(instr);
  if (!exec_time.has_value()) {
    return absl::InternalError(
        absl::StrCat("Cannot estimate runtime for: ", instr.ToString()));
  }
  return SetReificationCost(instr, *exec_time, kPerfTablesModelName);
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
      *module,
      HloPredicateIsOp<HloOpcode::kCustomCall, HloOpcode::kFusion,
                       HloOpcode::kDot>,
      [&](HloInstruction* instr) {
        if (absl::Status status = MaybeRecordPerfTablesForDotsAndCustomCalls(
                device_info_, *instr, *interpolator);
            !status.ok()) {
          VLOG(1) << "Cannot record perf tables stats data: "
                  << instr->ToString() << ". Status: " << status;
        }
        if (absl::Status status = MaybeRecordGemmCostModelForGemmTritonFusion(
                device_info_, *instr);
            !status.ok()) {
          VLOG(1) << "Cannot record GEMM cost model stats data: "
                  << instr->ToString() << ". Status: " << status;
        }
      });

  return false;
}

}  // namespace xla::gpu
