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

#include "xla/python/debug_callback_partitioner.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/service/spmd/spmd_partitioner.h"
#include "xla/shape.h"

namespace xla {

absl::Status DebugCallbackCustomCallPartitioner::Partition(
    spmd::SpmdPartitioningVisitor* partitioner, HloInstruction* hlo) const {
  // Cast HloInstruction to HloCustomCallInstruction.
  const HloCustomCallInstruction* custom_call =
      Cast<HloCustomCallInstruction>(hlo);

  // Initialize partitioned operands and shapes.
  const int64_t num_operands = hlo->operand_count();
  std::vector<HloInstruction*> partitioned_operands;
  partitioned_operands.reserve(num_operands);
  std::vector<Shape> partitioned_shapes_with_layout_constraints;
  partitioned_shapes_with_layout_constraints.reserve(num_operands);

  // Loop through and get partitioned operands and shapes.
  for (size_t i = 0; i < num_operands; ++i) {
    // For each operand, get partitioned hlo.
    spmd::PartitionedHlo partitioned_operand =
        partitioner->GetPartitionedHlo(hlo->operand(i));
    partitioned_operands.push_back(partitioned_operand.hlo());
    Shape partitioned_shape_with_layout_constraint =
        partitioned_operand.hlo()->shape();
    (*partitioned_shape_with_layout_constraint.mutable_layout()) =
        custom_call->operand_shapes_with_layout()[i].layout();
    partitioned_shapes_with_layout_constraints.push_back(
        partitioned_shape_with_layout_constraint);
  }

  // Create new custom call with partitioned operands.
  std::unique_ptr<HloInstruction> partitioned_instruction =
      HloInstruction::CreateCustomCall(
          hlo->shape(), partitioned_operands, custom_call->custom_call_target(),
          partitioned_shapes_with_layout_constraints, custom_call->opaque(),
          custom_call->api_version());
  auto partitioned_custom_call =
      Cast<HloCustomCallInstruction>(partitioned_instruction.get());
  partitioned_custom_call->set_custom_call_has_side_effect(
      custom_call->custom_call_has_side_effect());
  HloInstruction* partitioned_hlo = partitioner->builder()->AddInstruction(
      std::move(partitioned_instruction));
  partitioned_hlo->set_sharding(HloSharding::Replicate());

  spmd::PartitionedHlo result_partitioned =
      spmd::PartitionedHlo(partitioned_hlo, hlo->shape(),
                           partitioner->MakePartitioningState())
          .Reshard(hlo->sharding());
  partitioner->SetPartitionedHlo(hlo, result_partitioned);

  return absl::OkStatus();
}

}  // namespace xla
