/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/service/gpu/gpu_hlo_ordering.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "xla/hlo/analysis/hlo_ordering.h"
#include "xla/hlo/analysis/hlo_reachability.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/shape_util.h"

namespace xla {
namespace gpu {

ConcurrentRegionsHloOrdering::ConcurrentRegionsHloOrdering(
    const HloSchedule& schedule)
    : SequentialHloOrdering(schedule) {
  for (auto* computation : module_->MakeNonfusionComputations()) {
    predecessors_.emplace(computation, HloReachabilityMap::Build(computation));
  }
  for (const auto& [id, _] : schedule_.sequences()) {
    sorted_schedule_sequence_ids_.push_back(id);
  }
  absl::c_sort(sorted_schedule_sequence_ids_);
  Initialize();
}

namespace {

// Wether the hlo op should be considered for concurrent execution.
bool RunConcurrently(const HloInstruction* hlo) {
  // Ops that will not run any device kernels do not matter.
  if (hlo->opcode() == HloOpcode::kParameter ||
      hlo->opcode() == HloOpcode::kBitcast ||
      hlo->opcode() == HloOpcode::kGetTupleElement ||
      hlo->opcode() == HloOpcode::kTuple) {
    return true;
  }
  // Dots and convolutions are not considered for concurrent execution. This
  // avoids interference such as cache thrashing.
  if (hlo->opcode() == HloOpcode::kFusion) {
    for (HloInstruction* fused_op : hlo->fused_instructions()) {
      if (fused_op->opcode() == HloOpcode::kDot ||
          fused_op->opcode() == HloOpcode::kConvolution) {
        return false;
      }
    }
  }
  // Custom calls (cuBLAS, cuDNN, etc.) are not considered for concurrent
  // execution.
  if (hlo->opcode() == HloOpcode::kCustomCall) {
    return false;
  }
  int64_t bytes_read_written =
      ShapeUtil::ByteSizeOfElementsRecursive(hlo->shape());
  for (const HloInstruction* operand : hlo->operands()) {
    bytes_read_written +=
        ShapeUtil::ByteSizeOfElementsRecursive(operand->shape());
  }
  // Ops that read and write more than 20MB of data are not considered for
  // concurrent execution.
  return bytes_read_written <= 20000000;
}

}  // namespace

void ConcurrentRegionsHloOrdering::Initialize() {
  int64_t concurrent_region_id = 0;
  for (int64_t id : sorted_schedule_sequence_ids_) {
    const HloInstructionSequence& sequence = schedule_.sequences().at(id);
    const auto& instructions = sequence.instructions();
    for (size_t i = 0; i < instructions.size(); ++i) {
      // If the ops cannot run concurrently, start a new region.
      if (!RunConcurrently(instructions[i])) {
        concurrent_region_id++;
      }
      // If the previous ops cannot run concurrently, start a new region.
      if (RunConcurrently(instructions[i]) && i > 0 &&
          !RunConcurrently(instructions[i - 1])) {
        concurrent_region_id++;
      }
      concurrent_region_id_[instructions[i]] = concurrent_region_id;
    }
  }
}

bool ConcurrentRegionsHloOrdering::ExecutesBeforeInSameComputation(
    const HloInstruction* a, const HloInstruction* b) const {
  CHECK_EQ(a->parent(), b->parent());
  // 'a' executes before 'b' if 'a' is in the strict predecessor set of 'b',
  // even if 'a' and 'b' are in the same concurrent region.
  if (a != b && predecessors_.at(a->parent())->IsReachable(a, b)) {
    return true;
  }

  // Fall back to sequential ordering if either op is not in a concurrent
  // region.
  if (!concurrent_region_id_.contains(a) ||
      !concurrent_region_id_.contains(b)) {
    return SequentialHloOrdering::ExecutesBeforeInSameComputation(a, b);
  }

  return concurrent_region_id_.at(a) < concurrent_region_id_.at(b);
}

std::string ConcurrentRegionsHloOrdering::ToString() const {
  std::string result = "ConcurrentRegionsHloOrdering\n";
  for (const int64_t id : sorted_schedule_sequence_ids_) {
    const HloInstructionSequence& sequence = schedule_.sequences().at(id);
    for (const HloInstruction* hlo : sequence.instructions()) {
      if (!concurrent_region_id_.contains(hlo)) {
        continue;
      }
      absl::StrAppend(&result, "  ", hlo->name(),
                      "\tconcurrent_region_id=", concurrent_region_id_.at(hlo),
                      "\n");
    }
  }
  return result;
}

}  // namespace gpu
}  // namespace xla
