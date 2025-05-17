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

#include "xla/service/gpu/transforms/collective_backend_assigner.h"

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xla/backends/gpu/collectives/nvshmem_collectives.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/shape_util.h"

namespace xla {
namespace gpu {

bool CollectiveBackendAssigner::HasInternodeCommunication(
    const std::vector<ReplicaGroup>& replica_groups, int64_t num_processes) {
  absl::flat_hash_set<int64_t> nodes;
  for (const auto& group : replica_groups) {
    nodes.clear();
    for (int64_t replica_id : group.replica_ids()) {
      nodes.insert(replica_id / num_processes);
    }
    if (nodes.size() > 1) {
      VLOG(1) << "Found internode communication in replica groups";
      return true;
    }
  }
  VLOG(1) << "Found intranode communication in replica groups";
  return false;
}

bool CollectiveBackendAssigner::HasInternodeCommunication(
    const std::vector<std::pair<int64_t, int64_t>>& source_target_pairs,
    int64_t num_processes) {
  absl::flat_hash_set<int64_t> nodes;
  for (const auto& pair : source_target_pairs) {
    nodes.clear();
    nodes.insert(pair.first / num_processes);
    nodes.insert(pair.second / num_processes);
    if (nodes.size() > 1) {
      VLOG(1) << "Found internode communication in source-target pairs";
      return true;
    }
  }
  VLOG(1) << "Found intranode communication in source-target pairs";
  return false;
}

bool CollectiveBackendAssigner::HasInternodeCommunication(
    const HloInstruction& instr, int64_t num_processes) {
  if (instr.opcode() == HloOpcode::kAllReduce ||
      instr.opcode() == HloOpcode::kAllReduceStart) {
    return HasInternodeCommunication(instr.replica_groups(), num_processes);
  }
  if (instr.opcode() == HloOpcode::kCollectivePermute ||
      instr.opcode() == HloOpcode::kCollectivePermuteStart) {
    return HasInternodeCommunication(instr.source_target_pairs(),
                                     num_processes);
  }
  LOG(ERROR) << "Unsupported collective operation for internode check: "
             << instr.ToString();
  return false;
}

bool CollectiveBackendAssigner::IsCollectiveOp(const HloInstruction* instr) {
  return HloPredicateIsOp<HloOpcode::kAllReduce, HloOpcode::kAllReduceStart,
                          HloOpcode::kCollectivePermute,
                          HloOpcode::kCollectivePermuteStart>(instr);
}

int64_t CollectiveBackendAssigner::GetShapeSize(const Shape& shape) {
  int64_t size_in_bytes = 0;
  if (shape.IsTuple()) {
    for (int64_t i = 0; i < shape.tuple_shapes_size(); ++i) {
      size_in_bytes += GetShapeSize(shape.tuple_shapes(i));
    }
    return size_in_bytes;
  }
  return ShapeUtil::ByteSizeOfElements(shape);
}

// Assigns either NVSHMEM or DEFAULT as the backend for collective operations
// based on:
// 1. Communication pattern (intranode vs internode)
// 2. Message size (compared against threshold_in_bytes)
absl::StatusOr<bool> CollectiveBackendAssigner::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* comp : module->computations()) {
    for (HloInstruction* instr : comp->instructions()) {
      if (IsCollectiveOp(instr)) {
        TF_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                            instr->backend_config<GpuBackendConfig>());
        CollectiveBackendConfig& backend_config =
            *gpu_config.mutable_collective_backend_config();

        auto backend =
            !HasInternodeCommunication(
                *instr, NvshmemCollectives::Default()->num_processes()) &&
                    GetShapeSize(instr->shape()) <= threshold_in_bytes_
                ? CollectiveBackendConfig::NVSHMEM
                : CollectiveBackendConfig::DEFAULT;

        backend_config.set_backend(backend);
        VLOG(1) << "CollectiveBackendAssigner: setting backend to "
                << CollectiveBackendConfig_CollectiveBackend_Name(backend);
        TF_RETURN_IF_ERROR(instr->set_backend_config(gpu_config));
        changed = true;
      }
    }
  }

  return changed;
}

}  // namespace gpu
}  // namespace xla
