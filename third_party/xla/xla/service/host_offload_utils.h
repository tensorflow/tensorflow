/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_HOST_OFFLOAD_UTILS_H_
#define XLA_SERVICE_HOST_OFFLOAD_UTILS_H_

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/shape_util.h"

namespace xla {
namespace host_offload_utils {

struct InstructionAndShapeIndex {
  explicit InstructionAndShapeIndex(HloInstruction* instruction)
      : instruction(instruction) {}
  InstructionAndShapeIndex(HloInstruction* instruction, ShapeIndex shape_index)
      : instruction(instruction), shape_index(shape_index) {}
  HloInstruction* instruction;
  ShapeIndex shape_index;
  std::string ToString() const;

  template <typename H>
  static H Hash(H h, const InstructionAndShapeIndex& i) {
    h = H::combine(std::move(h), i.instruction);
    h = H::combine(std::move(h), i.shape_index);
    return std::move(h);
  }

  template <typename H>
  friend H AbslHashValue(H h, const InstructionAndShapeIndex& i) {
    return InstructionAndShapeIndex::Hash(std::move(h), i);
  }
};

bool operator==(const InstructionAndShapeIndex& lhs,
                const InstructionAndShapeIndex& rhs);

// If an instruction's user is a call, we descend into the call first.
// Eventually, a later invocation of this function while walking the graph will
// return the call itself as a successor of the ROOT instruction of the
// computation.
absl::StatusOr<std::vector<InstructionAndShapeIndex>> GetSuccessors(
    const InstructionAndShapeIndex& instruction_and_shape_index);

// If an instruction's operand is a call, return the call now. A follow up call
// of this function on that call returns the ROOT. Eventually, once the given
// instruction is a parameter, the returned predecessor will be the appropriate
// operand of the call (not the call itself, since we already returned it).
std::vector<InstructionAndShapeIndex> GetPredecessors(
    const InstructionAndShapeIndex& instruction_and_shape_index,
    std::optional<int64_t> operand_index = std::nullopt);

// Returns true if the instruction is allowed to be in the
// middle of a pure memory offload path.
bool IsValidDuringPureMemoryOffload(const HloInstruction* instruction);

// Returns true if the instruction is an async-start with host thread.
bool IsHostAsyncStart(const HloInstruction* instruction);

// Returns true if the copy is from or to host memory space.
bool IsSynchronousCopyFromOrToHost(const HloInstruction* instruction);

bool ComputeTypeIsHost(const HloInstruction* hlo_instruction);

// Sets the frontend attribute of the instruction to indicate that the
// instruction should be lowered as host compute.
void SetHostComputeFrontendAttribute(HloInstruction& host_instruction);

bool IsMoveToHostWithDynamicUpdateSlice(const HloInstruction* instr);

bool IsMoveToDeviceWithDynamicSlice(const HloInstruction* instr);

// Scans while loop body for DS/DUS, traces their index operands back to GTEs
// and marks corresponding tuple indices as dynamic variables.
absl::Status MarkDynamicVariables(HloInstruction* while_loop);

}  // namespace host_offload_utils
}  // namespace xla

#endif  // XLA_SERVICE_HOST_OFFLOAD_UTILS_H_
