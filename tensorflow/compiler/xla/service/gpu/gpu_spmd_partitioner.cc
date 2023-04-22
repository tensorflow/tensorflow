/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/gpu_spmd_partitioner.h"

#include <memory>

#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"

namespace xla {
namespace gpu {

Status GpuSpmdPartitioningVisitor::HandleRngGetAndUpdateState(
    HloInstruction* hlo) {
  if (hlo->sharding().HasUniqueDevice()) {
    return HandleSingleDevice(hlo);
  }

  // Tile sharding on rng-get-and-update status does not make sense, so only
  // replicated one is supported.
  TF_RET_CHECK(hlo->sharding().IsReplicated());

  // A replicated rng-get-and-update state can be schieved by just replicating
  // the instruction (since the RNG key is initialized to the compile time
  // constant value).
  auto clone =
      builder()->AddInstruction(hlo->CloneWithNewOperands(hlo->shape(), {}));
  clone->set_sharding(hlo->sharding());
  SetPartitionedHlo(
      hlo, spmd::PartitionedHlo(clone, hlo->shape(), MakePartitioningState())
               .Reshard(hlo->sharding()));
  return Status::OK();
}

std::unique_ptr<spmd::SpmdPartitioningVisitor>
GpuSpmdPartitioner::CreateVisitor(
    HloComputation* computation, int64_t num_partitions, int64_t num_replicas,
    const spmd::SPMDCollectiveOpsCreator& collective_ops_creator,
    int64* next_channel_id, spmd::SpmdLogger* logger,
    spmd::SpmdPartitionerOptions options) {
  return absl::make_unique<GpuSpmdPartitioningVisitor>(
      computation, num_partitions, num_replicas, collective_ops_creator,
      next_channel_id, logger, std::move(options), this);
}

Status GpuSpmdPartitioner::PreprocessSharding(HloModule* module) {
  // For rng-get-and-update-status with no sharding, set sharding to be
  // replicated.
  for (HloComputation* computation : module->computations()) {
    for (HloInstruction* hlo : computation->instructions()) {
      if (hlo->opcode() == HloOpcode::kRngGetAndUpdateState &&
          !hlo->has_sharding()) {
        hlo->set_sharding(HloSharding::Replicate());
      }
    }
  }
  return spmd::SpmdPartitioner::PreprocessSharding(module);
}

bool GpuSpmdPartitioner::CanSideEffectingHaveReplicatedSharding(
    const HloInstruction* hlo) {
  if (hlo->opcode() == HloOpcode::kRngGetAndUpdateState) return true;
  return spmd::SpmdPartitioner::CanSideEffectingHaveReplicatedSharding(hlo);
}

}  // namespace gpu
}  // namespace xla
