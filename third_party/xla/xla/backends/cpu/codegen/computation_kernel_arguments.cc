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

#include "xla/backends/cpu/codegen/computation_kernel_arguments.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/codegen/kernel_spec.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace xla::cpu {
namespace {

struct BufferTable {
  void Add(const Shape& shape, const BufferAllocation::Slice& slice) {
    if (seen.insert({shape, slice}).second) {
      buffers.push_back({slice, shape});
    }
  }

  KernelSpec::Buffers buffers;
  absl::flat_hash_set<const HloInstruction*> seen_instructions;
  absl::flat_hash_set<std::pair<Shape, BufferAllocation::Slice>> seen;
};

struct Walk {
  const BufferAssignment& assignment;
  BufferTable arguments;
  BufferTable results;
  // Every visited instruction occurrence, in visit order.
  std::vector<const HloInstruction*> instructions;
  // Allocations of constants defined inside the walk, whose values are part of
  // the computation text.
  absl::flat_hash_set<BufferAllocation::Index> walked_constants;
};

// Adds the slices of all subshapes of `instruction` to `table`.
absl::Status AddSlices(const HloInstruction* instruction,
                       const BufferAssignment& assignment, BufferTable& table) {
  if (!table.seen_instructions.insert(instruction).second) {
    return absl::OkStatus();
  }
  return ShapeUtil::ForEachSubshapeWithStatus(
      instruction->shape(),
      [&](const Shape& subshape, const ShapeIndex& index) -> absl::Status {
        ABSL_ASSIGN_OR_RETURN(BufferAllocation::Slice slice,
                         assignment.GetUniqueSlice(instruction, index));
        if (!slice.allocation()->is_thread_local()) {
          table.Add(subshape, slice);
        }
        return absl::OkStatus();
      });
}

// Visits instructions in the order HloComputation::Print uses, so that
// computations with equal text are walked identically.
absl::Status Visit(const HloComputation& computation, Walk& walk) {
  for (const HloInstruction* instruction :
       computation.MakeInstructionPostOrder()) {
    walk.instructions.push_back(instruction);
    for (const HloInstruction* operand : instruction->operands()) {
      ABSL_RETURN_IF_ERROR(AddSlices(operand, walk.assignment, walk.arguments));
    }
    // Parameters forward the buffers of their caller.
    if (instruction->opcode() == HloOpcode::kParameter) {
      ABSL_RETURN_IF_ERROR(AddSlices(instruction, walk.assignment, walk.arguments));
    } else {
      ABSL_RETURN_IF_ERROR(AddSlices(instruction, walk.assignment, walk.results));
    }
    if (instruction->opcode() == HloOpcode::kConstant) {
      ABSL_ASSIGN_OR_RETURN(BufferAllocation::Slice slice,
                       walk.assignment.GetUniqueTopLevelSlice(instruction));
      walk.walked_constants.insert(slice.index());
    }
    for (const HloComputation* called : instruction->called_computations()) {
      if (!called->IsFusionComputation()) {
        ABSL_RETURN_IF_ERROR(Visit(*called, walk));
      }
    }
  }
  return absl::OkStatus();
}

// IrEmitter lowers this op to a module-private RNG state global, which a shared
// kernel would share and race on. Fusion bodies are included.
bool UsesRngState(const HloComputation& computation) {
  auto uses = [](const HloComputation* c) {
    return absl::c_any_of(c->instructions(), [](const HloInstruction* i) {
      return i->opcode() == HloOpcode::kRngGetAndUpdateState;
    });
  };
  return uses(&computation) ||
         absl::c_any_of(computation.MakeEmbeddedComputationsList(), uses);
}

// Returns nullopt if the kernel must not be reused.
std::optional<std::string> Fingerprint(
    const HloComputation& computation, const Walk& walk,
    const absl::flat_hash_map<BufferAllocation::Slice, int64_t>& table_index) {
  if (UsesRngState(computation)) {
    return std::nullopt;
  }
  // Constants and backend configs are baked into the emitted code.
  std::string fingerprint =
      computation.ToString(HloPrintOptions::ModuleFingerprint()
                               .set_print_only_essential_constants(false)
                               .set_print_large_constants(true)
                               .set_print_backend_config(true));

  absl::flat_hash_set<BufferAllocation::Slice> seen_slices;
  std::vector<BufferAllocation::Slice> distinct_slices;
  for (const KernelSpec::Buffers* buffers :
       {&walk.arguments.buffers, &walk.results.buffers}) {
    for (const ShapedSlice& buffer : *buffers) {
      const BufferAllocation& allocation = *buffer.slice.allocation();
      absl::StrAppend(&fingerprint, "\n",
                      ShapeUtil::HumanStringWithLayout(buffer.shape), " ",
                      buffer.slice.size(), " ", allocation.size(), " ",
                      allocation.is_constant(),
                      allocation.is_parameter_aliased_with_output());
      if (seen_slices.insert(buffer.slice).second) {
        distinct_slices.push_back(buffer.slice);
      }
    }
  }

  // Which buffer each instruction uses, so that equal tables are also used
  // in the same way. Thread-local allocations are numbered by first use, since
  // the emitter may share one alloca between slices of the same allocation.
  absl::flat_hash_map<BufferAllocation::Index, int64_t> thread_local_id;
  for (const HloInstruction* instruction : walk.instructions) {
    absl::StrAppend(&fingerprint, "\n");
    ShapeUtil::ForEachSubshape(
        instruction->shape(), [&](const Shape&, const ShapeIndex& shape_index) {
          // Visit already checked that every slice is unique.
          BufferAllocation::Slice slice =
              // NOLINTNEXTLINE
              *walk.assignment.GetUniqueSlice(instruction, shape_index);
          const BufferAllocation& allocation = *slice.allocation();
          if (auto it = table_index.find(slice); it != table_index.end()) {
            absl::StrAppend(&fingerprint, " ", it->second);
            // The literal of a constant from outside the walk is not in the
            // computation text, and the emitter may bake it into the code.
            if (allocation.is_constant() &&
                !walk.walked_constants.contains(allocation.index())) {
              absl::StrAppend(&fingerprint, ":C", allocation.index());
            }
          } else {
            // Visit adds every other slice to the table.
            CHECK(allocation.is_thread_local());
            int64_t id =
                thread_local_id
                    .try_emplace(allocation.index(), thread_local_id.size())
                    .first->second;
            if (seen_slices.insert(slice).second) {
              distinct_slices.push_back(slice);
            }
            absl::StrAppend(&fingerprint, " T", id, ":", slice.offset(), ":",
                            slice.size(), ":", allocation.size());
          }
        });
  }

  absl::StrAppend(&fingerprint, "\noverlaps:");
  for (int64_t i = 0; i < distinct_slices.size(); ++i) {
    for (int64_t j = i + 1; j < distinct_slices.size(); ++j) {
      if (distinct_slices[i].OverlapsWith(distinct_slices[j])) {
        absl::StrAppend(&fingerprint, " ", i, ",", j);
      }
    }
  }
  return fingerprint;
}

}  // namespace

absl::StatusOr<ComputationKernelArguments> ComputationKernelArguments::Create(
    const HloComputation& computation, const BufferAssignment& assignment) {
  Walk walk{assignment};
  ABSL_RETURN_IF_ERROR(Visit(computation, walk));

  // The emitter stores every argument and result by position, so later
  // entries win for slices listed more than once.
  absl::flat_hash_map<BufferAllocation::Slice, int64_t> buffer_table_index;
  int64_t index = 0;
  for (const KernelSpec::Buffers* buffers :
       {&walk.arguments.buffers, &walk.results.buffers}) {
    for (const ShapedSlice& buffer : *buffers) {
      buffer_table_index.insert_or_assign(buffer.slice, index++);
    }
  }

  std::optional<std::string> fingerprint =
      Fingerprint(computation, walk, buffer_table_index);
  return ComputationKernelArguments(
      std::move(walk.arguments.buffers), std::move(walk.results.buffers),
      std::move(buffer_table_index), std::move(fingerprint));
}

}  // namespace xla::cpu
