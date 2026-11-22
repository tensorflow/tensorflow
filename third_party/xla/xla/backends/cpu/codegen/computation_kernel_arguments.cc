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

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/codegen/kernel_spec.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/ir_emission_utils.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {
namespace {

struct BufferTable {
  void Add(const BufferAllocation::Slice& slice) {
    if (seen.insert(slice).second) {
      slices.push_back(slice);
    }
  }

  std::vector<BufferAllocation::Slice> slices;
  absl::flat_hash_set<const HloInstruction*> seen_instructions;
  absl::flat_hash_set<BufferAllocation::Slice> seen;
};

struct Region {
  BufferAllocation::Slice slice;
  int64_t first_visit = std::numeric_limits<int64_t>::max();
  bool written = false;
  int64_t parameter_index = -1;
};

struct Placement {
  int64_t region;
  int64_t offset;
};

struct Walk {
  const BufferAssignment& assignment;
  BufferTable arguments;
  BufferTable results;
  // Slices of instructions that may emit a thunk write.
  absl::flat_hash_set<BufferAllocation::Slice> defined;
  // Every visited instruction occurrence, in visit order.
  std::vector<const HloInstruction*> instructions;
  // Allocations of constants defined inside the walk, whose values are part of
  // the computation text.
  absl::flat_hash_set<BufferAllocation::Index> walked_constants;
};

// Returns whether `instruction` may emit a thunk that writes to its output
// buffer. Control flow writes come from its nested computations, which are
// walked directly.
bool DefinesBuffer(const HloInstruction& instruction) {
  switch (instruction.opcode()) {
    case HloOpcode::kCall:
    case HloOpcode::kConditional:
    case HloOpcode::kWhile:
      return false;
    default:
      return !EmitsNoThunks(instruction);
  }
}

// Adds the leaf array slices of `instruction` to `table`.
absl::Status AddSlices(const HloInstruction* instruction,
                       const BufferAssignment& assignment, BufferTable& table,
                       bool defines,
                       absl::flat_hash_set<BufferAllocation::Slice>& defined) {
  if (!table.seen_instructions.insert(instruction).second) {
    return absl::OkStatus();
  }
  return ShapeUtil::ForEachSubshapeWithStatus(
      instruction->shape(),
      [&](const Shape& subshape, const ShapeIndex& index) -> absl::Status {
        if (!subshape.IsArray()) {
          return absl::OkStatus();
        }
        ABSL_ASSIGN_OR_RETURN(BufferAllocation::Slice slice,
                         assignment.GetUniqueSlice(instruction, index));
        if (slice.empty() || slice.allocation()->is_thread_local()) {
          return absl::OkStatus();
        }
        table.Add(slice);
        if (defines) {
          defined.insert(slice);
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
      ABSL_RETURN_IF_ERROR(AddSlices(operand, walk.assignment, walk.arguments,
                                /*defines=*/false, walk.defined));
    }
    // Parameters forward the buffers of their caller.
    if (instruction->opcode() == HloOpcode::kParameter) {
      ABSL_RETURN_IF_ERROR(AddSlices(instruction, walk.assignment, walk.arguments,
                                /*defines=*/false, walk.defined));
    } else {
      ABSL_RETURN_IF_ERROR(AddSlices(instruction, walk.assignment, walk.results,
                                DefinesBuffer(*instruction), walk.defined));
    }
    if (instruction->opcode() == HloOpcode::kConstant) {
      ABSL_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
          instruction->shape(),
          [&](const Shape& subshape, const ShapeIndex& index) -> absl::Status {
            if (!subshape.IsArray()) {
              return absl::OkStatus();
            }
            ABSL_ASSIGN_OR_RETURN(
                BufferAllocation::Slice slice,
                walk.assignment.GetUniqueSlice(instruction, index));
            if (!slice.empty()) {
              walk.walked_constants.insert(slice.index());
            }
            return absl::OkStatus();
          }));
    }
    for (const HloComputation* called : instruction->called_computations()) {
      if (!called->IsFusionComputation()) {
        ABSL_RETURN_IF_ERROR(Visit(*called, walk));
      }
    }
  }
  return absl::OkStatus();
}

std::string Fingerprint(
    const HloComputation& computation, const Walk& walk,
    absl::Span<const Region* const> ordered_regions,
    const absl::flat_hash_map<BufferAllocation::Slice, Placement>& placement) {
  // Constants and backend configs are baked into the emitted code.
  std::string fingerprint =
      computation.ToString(HloPrintOptions::ModuleFingerprint()
                               .set_print_only_essential_constants(false)
                               .set_print_large_constants(true)
                               .set_print_backend_config(true));

  // Region roles and sizes decide parameter attributes.
  absl::StrAppend(&fingerprint, "\nregions:");
  for (int64_t i = 0; i < ordered_regions.size(); ++i) {
    const Region& r = *ordered_regions[i];
    absl::StrAppend(&fingerprint, "\n", i, r.written ? " W " : " A ",
                    r.slice.size());
  }

  // Which region and offset each instruction uses, so that equal region lists
  // are also used in the same way. Thread local allocations are numbered by
  // first use, so that equal computations number them identically.
  absl::StrAppend(&fingerprint, "\ninstructions:");
  absl::flat_hash_map<BufferAllocation::Index, int64_t> thread_local_id;
  for (const HloInstruction* instruction : walk.instructions) {
    absl::StrAppend(&fingerprint, "\n");
    ShapeUtil::ForEachSubshape(
        instruction->shape(),
        [&](const Shape& subshape, const ShapeIndex& shape_index) {
          if (subshape.IsTuple()) {
            absl::StrAppend(&fingerprint, " U");
            return;
          }
          if (!subshape.IsArray()) {
            absl::StrAppend(&fingerprint, " Z");
            return;
          }
          // Visit already checked that every array slice is unique.
          BufferAllocation::Slice slice =
              // NOLINTNEXTLINE
              *walk.assignment.GetUniqueSlice(instruction, shape_index);
          const BufferAllocation& allocation = *slice.allocation();
          if (allocation.is_thread_local()) {
            int64_t id =
                thread_local_id
                    .try_emplace(allocation.index(), thread_local_id.size())
                    .first->second;
            absl::StrAppend(&fingerprint, " T", id, ":", slice.offset(), ":",
                            slice.size(), ":", allocation.size());
          } else if (slice.empty()) {
            absl::StrAppend(&fingerprint, " Z");
          } else {
            // Visit adds every other slice to the table.
            const Placement& p = placement.at(slice);
            absl::StrAppend(&fingerprint, " ", p.region, "+", p.offset, ":",
                            slice.size());
            // The literal of a constant from outside the walk is not in the
            // computation text, and the emitter may bake it into the code.
            if (allocation.is_constant() &&
                !walk.walked_constants.contains(allocation.index())) {
              absl::StrAppend(&fingerprint, ":C", allocation.index());
            }
          }
        });
  }

  return fingerprint;
}

}  // namespace

std::vector<BufferAllocation::Slice>
ComputationKernelArguments::MergeOverlapping(
    absl::Span<const BufferAllocation::Slice> slices) {
  std::vector<BufferAllocation::Slice> sorted;
  sorted.reserve(slices.size());
  for (const BufferAllocation::Slice& slice : slices) {
    if (!slice.empty()) {
      sorted.push_back(slice);
    }
  }
  absl::c_sort(sorted);

  std::vector<BufferAllocation::Slice> merged;
  for (const BufferAllocation::Slice& slice : sorted) {
    if (merged.empty() || merged.back().index() != slice.index() ||
        merged.back().offset() + merged.back().size() <= slice.offset()) {
      merged.push_back(slice);
    } else {
      int64_t start = merged.back().offset();
      int64_t end =
          std::max(start + merged.back().size(), slice.offset() + slice.size());
      merged.back() = BufferAllocation::Slice(merged.back().allocation(), start,
                                              end - start);
    }
  }
  return merged;
}

absl::StatusOr<ComputationKernelArguments> ComputationKernelArguments::Create(
    const HloComputation& computation, const BufferAssignment& assignment) {
  Walk walk{assignment};
  ABSL_RETURN_IF_ERROR(Visit(computation, walk));

  absl::flat_hash_map<BufferAllocation::Slice, int64_t> first_visit;
  std::vector<BufferAllocation::Slice> all_slices;
  int64_t visit_index = 0;
  for (const std::vector<BufferAllocation::Slice>* table :
       {&walk.arguments.slices, &walk.results.slices}) {
    for (const BufferAllocation::Slice& slice : *table) {
      if (first_visit.try_emplace(slice, visit_index).second) {
        all_slices.push_back(slice);
      }
      ++visit_index;
    }
  }

  std::vector<BufferAllocation::Slice> merged = MergeOverlapping(all_slices);
  std::vector<Region> regions;
  regions.reserve(merged.size());
  for (const BufferAllocation::Slice& slice : merged) {
    regions.push_back(Region{slice});
  }

  auto find_region = [&](const BufferAllocation::Slice& slice) -> Region& {
    auto it = absl::c_upper_bound(
        regions, slice, [](const BufferAllocation::Slice& s, const Region& r) {
          if (s.index() != r.slice.index()) {
            return s.index() < r.slice.index();
          }
          return s.offset() < r.slice.offset();
        });
    --it;
    return *it;
  };

  for (const BufferAllocation::Slice& slice : all_slices) {
    Region& r = find_region(slice);
    r.first_visit = std::min(r.first_visit, first_visit.at(slice));
    if (walk.defined.contains(slice) && !slice.allocation()->is_readonly()) {
      r.written = true;
    }
  }

  std::vector<int64_t> order(regions.size());
  for (int64_t i = 0; i < regions.size(); ++i) {
    order[i] = i;
  }
  absl::c_sort(order, [&](int64_t a, int64_t b) {
    if (regions[a].written != regions[b].written) {
      return !regions[a].written;
    }
    return regions[a].first_visit < regions[b].first_visit;
  });

  KernelSpec::Buffers arguments;
  KernelSpec::Buffers results;
  std::vector<const Region*> ordered_regions(regions.size());
  for (int64_t i = 0; i < order.size(); ++i) {
    Region& r = regions[order[i]];
    r.parameter_index = i;
    ordered_regions[i] = &r;
    ShapedSlice shaped{r.slice, ShapeUtil::MakeShape(U8, {r.slice.size()})};
    if (r.written) {
      results.push_back(std::move(shaped));
    } else {
      arguments.push_back(std::move(shaped));
    }
  }

  absl::flat_hash_map<BufferAllocation::Slice, Placement> placement;
  placement.reserve(all_slices.size());
  for (const BufferAllocation::Slice& slice : all_slices) {
    const Region& r = find_region(slice);
    placement.emplace(
        slice, Placement{r.parameter_index, slice.offset() - r.slice.offset()});
  }

  std::string fingerprint =
      Fingerprint(computation, walk, ordered_regions, placement);
  return ComputationKernelArguments(std::move(arguments), std::move(results),
                                    std::move(fingerprint));
}

}  // namespace xla::cpu
