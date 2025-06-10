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

#include "xla/service/gpu/transforms/collectives/collective_backend_assigner.h"

#include <cstdint>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/transforms/collectives/collective_ops_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

bool IsCollectiveOp(const HloInstruction* instr) {
  return HloPredicateIsOp<HloOpcode::kAllReduce, HloOpcode::kAllReduceStart,
                          HloOpcode::kCollectivePermute,
                          HloOpcode::kCollectivePermuteStart>(instr);
}

int64_t GetShapeSize(const Shape& shape) {
  if (shape.IsTuple()) {
    int64_t size_in_bytes = 0;
    for (const Shape& subshape : shape.tuple_shapes()) {
      size_in_bytes += GetShapeSize(subshape);
    }
    return size_in_bytes;
  }
  return ShapeUtil::ByteSizeOfElements(shape);
}

absl::StatusOr<GPUCommunicationType> GetCommunicationType(
    const HloInstruction* instr, int num_devices_per_host,
    const se::GpuComputeCapability& gpu_version) {
  if (num_devices_per_host == -1) {
    return absl::FailedPreconditionError(
        "Could not determine number of devices per host");
  }
  return CommunicationType(num_devices_per_host,
                           *xla::Cast<HloChannelInstruction>(instr),
                           gpu_version);
}

}  // namespace

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
      if (!IsCollectiveOp(instr)) {
        continue;
      }

      TF_ASSIGN_OR_RETURN(
          GPUCommunicationType comm_type,
          GetCommunicationType(instr, num_devices_per_host_, gpu_version_));
      VLOG(1) << "CollectiveBackendAssigner: comm_type="
              << static_cast<int>(comm_type)
              << " shape_size=" << GetShapeSize(instr->shape())
              << " threshold_in_bytes_=" << threshold_in_bytes_;
      bool use_nvshmem = (num_devices_per_host_ == 1 ||
                          comm_type == GPUCommunicationType::SINGLE_HOST) &&
                         GetShapeSize(instr->shape()) < threshold_in_bytes_;
      if (!use_nvshmem) {
        continue;
      }

      TF_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                          instr->backend_config<GpuBackendConfig>());
      gpu_config.mutable_collective_backend_config()->set_backend(
          CollectiveBackendConfig::NVSHMEM);

      VLOG(1) << "CollectiveBackendAssigner: setting backend to NVSHMEM for "
              << instr->name();

      TF_RETURN_IF_ERROR(instr->set_backend_config(gpu_config));
      changed = true;
    }
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
