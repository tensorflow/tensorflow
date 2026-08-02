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

#include "xla/backends/gpu/transforms/collectives/all_gather_optimizer.h"

#include <array>
#include <cstdint>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/transforms/collectives/gpu_collective_combiner_utils.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/shape_util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/protobuf.h"

namespace xla {
namespace gpu {
namespace {

bool NormalizeBackendConfigCopyForComparison(GpuBackendConfig& config) {
  if (!config.has_collective_backend_config() &&
      config.backend_config_case() !=
          GpuBackendConfig::BACKEND_CONFIG_NOT_SET) {
    return false;
  }

  CollectiveBackendConfig* collective_config =
      config.mutable_collective_backend_config();
  // These fields are OR-merged when building the replacement.
  collective_config->clear_is_pipelined();
  collective_config->clear_is_spmd_generated();
  return true;
}

bool HaveCompatibleCollectiveBackendConfigs(const HloInstruction& lhs,
                                            const HloInstruction& rhs) {
  auto lhs_config_or = lhs.backend_config<GpuBackendConfig>();
  auto rhs_config_or = rhs.backend_config<GpuBackendConfig>();
  if (!lhs_config_or.ok() || !rhs_config_or.ok()) {
    VLOG(2) << "Failed to parse an all-gather backend config.";
    return false;
  }

  // backend_config<T>() returns owned values. Normalize only these local
  // copies; the instructions' backend configs remain unchanged.
  GpuBackendConfig lhs_config = std::move(*lhs_config_or);
  GpuBackendConfig rhs_config = std::move(*rhs_config_or);
  if (!NormalizeBackendConfigCopyForComparison(lhs_config) ||
      !NormalizeBackendConfigCopyForComparison(rhs_config)) {
    VLOG(2) << "An all-gather has a non-collective backend config.";
    return false;
  }
  return tsl::protobuf::util::MessageDifferencer::Equals(lhs_config,
                                                         rhs_config);
}

}  // namespace

absl::StatusOr<bool> AllGatherOptimizer::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      if (!HloOpcodeIsBinaryCommutative(instruction->opcode())) {
        continue;
      }

      HloInstruction* left_op = instruction->mutable_operand(0);
      HloInstruction* right_op = instruction->mutable_operand(1);

      if (HloPredicateIsNotOp<HloOpcode::kAllGather>(right_op) ||
          HloPredicateIsNotOp<HloOpcode::kAllGather>(left_op)) {
        VLOG(2) << "Binary op's operands are not all-gather deduced types.";
        continue;
      }

      auto* left_all_gather = Cast<HloAllGatherInstruction>(left_op);
      auto* right_all_gather = Cast<HloAllGatherInstruction>(right_op);

      if (right_all_gather->constrain_layout() !=
              left_all_gather->constrain_layout() ||
          right_all_gather->use_global_device_ids() !=
              left_all_gather->use_global_device_ids() ||
          !ReplicaGroupsEqual(right_all_gather->replica_groups(),
                              left_all_gather->replica_groups())) {
        VLOG(2) << "The right and left all-gather ops are not compatible "
                   "to merge. ";
        continue;
      }

      if (!HaveCompatibleCollectiveGroupKeys(*left_all_gather,
                                             *right_all_gather)) {
        VLOG(2) << "The right and left all-gather ops belong to different "
                   "collective groups.";
        continue;
      }

      if (!HaveCompatibleCollectiveBackendConfigs(*left_all_gather,
                                                  *right_all_gather)) {
        VLOG(2) << "The right and left all-gather ops have incompatible "
                   "backend configs.";
        continue;
      }

      if (!ShapeUtil::Equal(left_all_gather->operand(0)->shape(),
                            right_all_gather->operand(0)->shape())) {
        VLOG(2) << "all-gather operands have different shapes";
        continue;
      }

      if (right_all_gather->user_count() != 1 ||
          left_all_gather->user_count() != 1) {
        VLOG(2) << "all-gather user_count > 1 ";
        continue;
      }
      auto index_in_full_shape =
          computation->AddInstruction(HloInstruction::CreateBinary(
              right_all_gather->operand(0)->shape(), instruction->opcode(),
              left_all_gather->mutable_operand(0),
              right_all_gather->mutable_operand(0)));

      int64_t all_gather_dimension =
          Cast<HloAllGatherInstruction>(right_all_gather)
              ->all_gather_dimension();

      auto combined = HloInstruction::CreateAllGather(
          left_all_gather->shape(), {index_in_full_shape}, all_gather_dimension,
          left_all_gather->device_list(),
          /*constrain_layout=*/false, left_all_gather->channel_id(),
          Cast<HloAllGatherInstruction>(left_all_gather)
              ->use_global_device_ids());
      instruction->SetupDerivedInstruction(combined.get());

      if (HasCollectiveGroupKey(*left_all_gather)) {
        CopyCollectiveGroupKey(*left_all_gather, *combined);
      } else {
        ClearCollectiveGroupKey(*combined);
      }
      combined->CopyBackendConfigFrom(left_all_gather);

      std::array<HloInstruction*, 2> all_gathers = {left_all_gather,
                                                    right_all_gather};
      RETURN_IF_ERROR(
          MergeCollectiveBackendConfig(all_gathers, combined.get()));

      RETURN_IF_ERROR(computation->ReplaceWithNewInstruction(
          instruction, std::move(combined),
          /*preserve_sharding=*/false,
          /*relay_control_dependency=*/false,
          /*remove_unused_operands=*/true,
          /*preserve_frontend_attributes=*/false));
      changed = true;
    }
  }

  return changed;
}

}  // namespace gpu
}  // namespace xla
