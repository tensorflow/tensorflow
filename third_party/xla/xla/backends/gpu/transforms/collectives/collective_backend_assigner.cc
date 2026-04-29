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

#include "xla/backends/gpu/transforms/collectives/collective_backend_assigner.h"

#include <cstdint>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/transforms/collectives/collective_ops_utils.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {
namespace {

bool IsCollectiveOp(const HloInstruction* instr) {
  return HloPredicateIsOp<HloOpcode::kAllReduce, HloOpcode::kAllReduceStart,
                          HloOpcode::kCollectivePermute,
                          HloOpcode::kCollectivePermuteStart>(instr);
}

bool IsCollectivePermuteOp(const HloInstruction* instr) {
  return HloPredicateIsOp<HloOpcode::kCollectivePermute,
                          HloOpcode::kCollectivePermuteStart>(instr);
}

bool IsAllReduceOp(const HloInstruction* instr) {
  return HloPredicateIsOp<HloOpcode::kAllReduce, HloOpcode::kAllReduceStart>(
      instr);
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
    const HloInstruction* instr, int num_visible_devices_per_process,
    const se::GpuComputeCapability& gpu_version) {
  if (num_visible_devices_per_process == -1) {
    return absl::FailedPreconditionError(
        "Could not determine number of devices per host");
  }
  return CommunicationType(num_visible_devices_per_process,
                           *xla::Cast<HloChannelInstruction>(instr),
                           gpu_version);
}

// Assigns NVSHMEM backend to eligible collective operations based on
// communication pattern and message size.
absl::StatusOr<bool> AssignNvshmemBackend(
    HloModule* module, int num_visible_devices_per_process,
    const se::GpuComputeCapability& gpu_version, int64_t threshold_in_bytes,
    int64_t slice_size) {
  bool changed = false;
  for (HloComputation* comp : module->computations()) {
    for (HloInstruction* instr : comp->instructions()) {
      if (!IsCollectiveOp(instr)) {
        continue;
      }

      ASSIGN_OR_RETURN(
          GPUCommunicationType comm_type,
          GetCommunicationType(instr, num_visible_devices_per_process,
                               gpu_version));
      int64_t shape_size = GetShapeSize(instr->shape());
      VLOG(1) << "CollectiveBackendAssigner: comm_type="
              << static_cast<int>(comm_type) << " shape_size=" << shape_size
              << " threshold_in_bytes=" << threshold_in_bytes
              << " slice_size=" << slice_size;
      bool use_nvshmem =
          (num_visible_devices_per_process == 1 ||
           comm_type == GPUCommunicationType::SINGLE_PARTITION ||
           (slice_size > 0 &&
            IsIntraNVLinkDomain(module->config(), slice_size))) &&
          (!IsAllReduceOp(instr) || shape_size < threshold_in_bytes);
      if (!use_nvshmem) {
        continue;
      }

      ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                       instr->backend_config<GpuBackendConfig>());
      gpu_config.mutable_collective_backend_config()->set_backend(
          CollectiveBackendConfig::NVSHMEM);

      VLOG(1) << "CollectiveBackendAssigner: setting backend to NVSHMEM for "
              << instr->name();

      RETURN_IF_ERROR(instr->set_backend_config(gpu_config));
      changed = true;
    }
  }
  return changed;
}

// Assigns collectives_mode for collective-permute operations based on the
// xla_gpu_collective_permute_mode debug option.
absl::StatusOr<bool> AssignCollectivePermuteMode(HloModule* module) {
  const auto mode =
      module->config().debug_options().xla_gpu_collective_permute_mode();
  if (mode == DebugOptions::COLLECTIVES_PRIVATE_MEMORY) {
    return false;
  }

  bool changed = false;
  for (HloComputation* comp : module->computations()) {
    for (HloInstruction* instr : comp->instructions()) {
      if (!IsCollectivePermuteOp(instr)) {
        continue;
      }

      ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                       instr->backend_config<GpuBackendConfig>());
      gpu_config.mutable_collective_backend_config()->set_collectives_mode(
          mode);
      RETURN_IF_ERROR(instr->set_backend_config(gpu_config));
      changed = true;
    }
  }
  return changed;
}

}  // namespace

absl::StatusOr<bool> CollectiveBackendAssigner::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;

  if (module->config().debug_options().xla_gpu_experimental_enable_nvshmem()) {
    ASSIGN_OR_RETURN(
        bool nvshmem_changed,
        AssignNvshmemBackend(module, num_visible_devices_per_process_,
                             gpu_version_, threshold_in_bytes_, slice_size_));
    changed |= nvshmem_changed;
  }

  ASSIGN_OR_RETURN(bool mode_changed, AssignCollectivePermuteMode(module));
  changed |= mode_changed;

  return changed;
}

}  // namespace gpu
}  // namespace xla
