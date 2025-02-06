/* Copyright 2021 The OpenXLA Authors.

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

#include "xla/service/spmd/stateful_rng_spmd_partitioner.h"

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/service/call_graph.h"
#include "xla/service/spmd/spmd_partitioner.h"
#include "xla/status_macros.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace spmd {

absl::Status StatefulRngSpmdPartitioningVisitor::HandleRngGetAndUpdateState(
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
  return absl::OkStatus();
}

std::unique_ptr<spmd::SpmdPartitioningVisitor>
StatefulRngSpmdPartitioner::CreateVisitor(
    HloComputation* computation, int64_t num_partitions, int64_t num_replicas,
    const spmd::SPMDCollectiveOpsCreator& collective_ops_creator,
    int64_t* next_channel_id, spmd::SpmdLogger* logger,
    spmd::SpmdPartitionerOptions options, const CallGraph& call_graph) {
  return std::make_unique<StatefulRngSpmdPartitioningVisitor>(
      computation, num_partitions, num_replicas, collective_ops_creator,
      next_channel_id, logger, std::move(options), this, call_graph);
}

absl::Status StatefulRngSpmdPartitioner::PreprocessSharding(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  // For rng-get-and-update-status with no sharding, set sharding to be
  // replicated.
  for (HloComputation* computation : module->computations(execution_threads)) {
    for (HloInstruction* hlo : computation->instructions()) {
      if (hlo->opcode() == HloOpcode::kRngGetAndUpdateState &&
          !hlo->has_sharding()) {
        hlo->set_sharding(HloSharding::Replicate());
      }
    }
  }
  return spmd::SpmdPartitioner::PreprocessSharding(module, execution_threads);
}

bool StatefulRngSpmdPartitioner::CanSideEffectingHaveReplicatedSharding(
    const HloInstruction* hlo) {
  if (hlo->opcode() == HloOpcode::kRngGetAndUpdateState) return true;
  return spmd::SpmdPartitioner::CanSideEffectingHaveReplicatedSharding(hlo);
}

absl::Status StatefulRngSpmdPartitioner::HandleRotateRightWhilePreprocessing(
    HloComputation* computation) {
  if (!computation->IsWhileBodyComputation()) {
    return absl::OkStatus();
  }
  HloInstruction* while_loop = computation->WhileCallInstruction();
  TF_RET_CHECK(while_loop);
  if (computation->parent()
          ->config()
          .debug_options()
          .xla_gpu_unsafe_pipelined_loop_annotator()) {
    while_loop->set_frontend_attribute("is_pipelined_while_loop", "true");
  }
  return absl::OkStatus();
}

}  // namespace spmd
}  // namespace xla
