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

#include "xla/service/memory_space_assignment/algorithm.h"

#include <sys/stat.h>
#include <sys/types.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/hash/hash.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/analysis/hlo_alias_analysis.h"
#include "xla/hlo/analysis/hlo_dataflow_analysis.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/utils/hlo_live_range.h"
#include "xla/layout.h"
#include "xla/service/buffer_value.h"
#include "xla/service/call_graph.h"
#include "xla/service/computation_layout.h"
#include "xla/service/heap_simulator/allocation_block.h"
#include "xla/service/heap_simulator/heap_simulator.h"
#include "xla/service/hlo_buffer.h"
#include "xla/service/hlo_value.h"
#include "xla/service/memory_space_assignment/allocation.h"
#include "xla/service/memory_space_assignment/allocation_value.h"
#include "xla/service/memory_space_assignment/buffer_interval_comparator.h"
#include "xla/service/memory_space_assignment/cost_analysis.h"
#include "xla/service/memory_space_assignment/memory_bound_loop_optimizer.h"
#include "xla/service/memory_space_assignment/memory_space_assignment.pb.h"
#include "xla/service/memory_space_assignment/options.h"
#include "xla/service/memory_space_assignment/repacking.h"
#include "xla/service/memory_space_assignment/slice.h"
#include "xla/service/memory_space_assignment/tuning_utils.h"
#include "xla/service/memory_space_assignment/utils.h"
#include "xla/service/time_utils.h"
#include "xla/shape.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace memory_space_assignment {
namespace {

// Define a dummy chunk for chunks that will be allocated in the default memory
// space and for keeping track of number of asynchronous copies.
const HeapSimulator::Chunk kDummyChunk =
    HeapSimulator::Chunk::FromOffsetSize(-1, -1);
// For cross-program prefetched buffer, we only perform the freeing optimization
// if the buffer occupies less of the execution time ratio than this value.
const float kCrossProgramPrefetchOccupyFreeingLimit = 0.6;

int64_t GetAlignedOffset(int64_t offset, int64_t alignment) {
  return CeilOfRatio(offset, alignment) * alignment;
}

template <typename T>
std::string VectorToString(const std::vector<T>& v,
                           bool include_indices = false, int start = 0,
                           int end = std::numeric_limits<int>::max()) {
  std::vector<std::string> elements;

  for (int i = start; i < end && i < v.size(); ++i) {
    std::string prefix;
    if (include_indices) {
      prefix = absl::StrCat(i, ": ");
    }
    elements.push_back(absl::StrCat(prefix, v[i]));
  }

  return absl::StrCat("[ ", absl::StrJoin(elements, ", "), " ]");
}

bool LooksLikeAnActivation(const HloInstruction* inst, bool permissive_mode) {
  for (HloInstruction* user : inst->users()) {
    switch (user->opcode()) {
      case HloOpcode::kConvolution:
      case HloOpcode::kDot:
        if (user->operand(0) == inst) {
          return true;
        }
        break;
      case HloOpcode::kGather:
        if (user->operand(1) == inst) {
          return true;
        }
        break;
      case HloOpcode::kFusion:
        for (int i = 0; i < user->operand_count(); ++i) {
          if (user->operand(i) == inst &&
              LooksLikeAnActivation(user->fused_parameter(i),
                                    permissive_mode)) {
            return true;
          }
        }
        break;
      case HloOpcode::kBitcast:
      case HloOpcode::kBroadcast:
      case HloOpcode::kTranspose:
        if (LooksLikeAnActivation(user, permissive_mode)) {
          return true;
        }
        break;
      case HloOpcode::kCopy:
        if (user->IsFused() && (user == user->parent()->root_instruction())) {
          user = user->parent()->FusionInstruction();
          if (LooksLikeAnActivation(user, permissive_mode)) {
            return true;
          } else {
            break;
          }
        }
        return true;
      case HloOpcode::kDynamicUpdateSlice:
      case HloOpcode::kDynamicSlice:
        if (std::find(user->operands().begin() + 1, user->operands().end(),
                      inst) != user->operands().end()) {
          return true;
        }
        if (LooksLikeAnActivation(user, permissive_mode)) {
          return true;
        }
        break;
      case HloOpcode::kReduce:
        // Check init operands.
        if (std::find(user->operands().begin() + user->operand_count() / 2,
                      user->operands().end(), inst) != user->operands().end()) {
          return true;
        }
        if (LooksLikeAnActivation(user, permissive_mode)) {
          return true;
        }
        break;
      default:
        // Permissive mode assumes the tensor is not an activation when we
        // couldn't explicitly determine that it is not an activation.
        return !permissive_mode;
    }
  }
  return false;
}

// Filters out buffer uses that cannot use the cross-program prefetch due to
// aliasing with program output.
std::vector<HloUse> FindCrossProgramPrefetchUses(
    absl::Span<const HloUse> buffer_uses,
    const HloAliasAnalysis& alias_analysis) {
  std::vector<HloUse> uses;
  if (buffer_uses.empty()) {
    return uses;
  }
  const HloInstruction* root_instruction = buffer_uses.at(0)
                                               .instruction->GetModule()
                                               ->entry_computation()
                                               ->root_instruction();
  // This function returns true if the use value does not live out of the
  // module. The value lives out if it is the root or it aliases with another
  // value that lives out. We recurse to detect the latter case.
  std::function<bool(const HloUse&)> use_does_not_live_out =
      [&](const HloUse& use) {
        if (use.instruction == root_instruction &&
            (use.instruction->opcode() == HloOpcode::kTuple ||
             use.instruction->opcode() == HloOpcode::kBitcast)) {
          return false;
        }
        auto in_place_pairs =
            HloDataflowAnalysis::GetInPlaceInputOutputPairs(use.instruction);
        return absl::c_all_of(
            in_place_pairs,
            [&](const std::pair<HloOperandIndex, ShapeIndex>& in_place_pair) {
              if (in_place_pair.first.operand_number == use.operand_number &&
                  in_place_pair.first.operand_index == use.operand_index) {
                return use.instruction != root_instruction &&
                       absl::c_all_of(
                           alias_analysis.dataflow_analysis()
                               .GetUniqueValueAt(use.instruction,
                                                 in_place_pair.second)
                               .GetUses(),
                           use_does_not_live_out);
              }
              return true;
            });
      };

  absl::c_copy_if(buffer_uses, std::back_inserter(uses), use_does_not_live_out);
  return uses;
}

bool IsCrossProgramPrefetchCandidate(const HloValue& value,
                                     const HloAliasAnalysis& alias_analysis,
                                     const Options& options) {
  // Filter out values that alias with the entry computation root.
  const HloBuffer& buffer = alias_analysis.GetBufferContainingValue(value);
  const HloInstruction* root = alias_analysis.dataflow_analysis()
                                   .module()
                                   .entry_computation()
                                   ->root_instruction();
  for (const HloPosition& position : buffer.ComputePositions()) {
    if (position.instruction == root) {
      return false;
    }
  }
  std::vector<HloUse> uses =
      FindCrossProgramPrefetchUses(value.GetUses(), alias_analysis);
  return value.defining_instruction()->parent() ==
             value.defining_instruction()->GetModule()->entry_computation() &&
         value.defining_instruction()->opcode() == HloOpcode::kParameter &&
         value.index().size() <= 1 && value.shape().IsArray() &&
         !uses.empty() && options.size_fn(value) <= options.max_size_in_bytes &&
         absl::c_all_of(uses, [&](const HloUse& use) {
           const HloInstruction* inst =
               use.instruction->operand(use.operand_number);

           // Skip the LooksLikeAnActivation test since we're testing the
           // parent GTE/parameter and its children below.
           if (inst->opcode() == HloOpcode::kBitcast &&
               ((inst->operand(0)->opcode() == HloOpcode::kGetTupleElement &&
                 inst->operand(0)->operand(0)->opcode() ==
                     HloOpcode::kParameter) ||
                inst->operand(0)->opcode() == HloOpcode::kParameter)) {
             return true;
           }

           return (inst->opcode() == HloOpcode::kGetTupleElement ||
                   inst->opcode() == HloOpcode::kParameter) &&
                  !LooksLikeAnActivation(
                      inst, options.cross_program_prefetch_permissive_mode);
         });
}

bool IsUserAnnotatedCrossProgramPrefetch(const HloValue& value,
                                         const Options& options) {
  const HloInstruction* defining_instruction = value.defining_instruction();
  if (defining_instruction->parent() !=
          defining_instruction->GetModule()->entry_computation() ||
      defining_instruction->opcode() != HloOpcode::kParameter) {
    return false;
  }
  const ComputationLayout& entry_computation_layout =
      defining_instruction->GetModule()->entry_computation_layout();
  if (defining_instruction->parameter_number() >=
      entry_computation_layout.parameter_count()) {
    return false;
  }
  const Shape& shape =
      entry_computation_layout
          .parameter_layout(defining_instruction->parameter_number())
          .shape();
  return shape.has_layout() &&
         shape.layout().memory_space() == options.alternate_memory_space;
}

MsaBufferInterval CreateMsaBufferInterval(const HloBuffer& buffer,
                                          const HloValue* value,
                                          const HloLiveRange& hlo_live_range,
                                          const Options& options) {
  MsaBufferInterval interval;
  interval.buffer = value;
  interval.size = options.size_fn(*value);
  interval.start = 0;
  interval.end = hlo_live_range.schedule_end_time();
  interval.colocations = {++buffer.values().begin(), buffer.values().end()};
  interval.need_allocation = true;
  return interval;
}

struct CrossProgramPrefetches {
  std::vector<MsaBufferInterval> prefetches;
  std::vector<MsaBufferInterval> candidates;
};

CrossProgramPrefetches FindCrossProgramPrefetches(
    const HloAliasAnalysis& alias_analysis, const HloLiveRange& hlo_live_range,
    const Options& options) {
  CrossProgramPrefetches cross_program_prefetches;
  for (const HloBuffer& buffer : alias_analysis.buffers()) {
    CHECK_GE(buffer.values().size(), 1);
    const HloValue* value = buffer.values().at(0);
    MsaBufferInterval buffer_interval =
        CreateMsaBufferInterval(buffer, value, hlo_live_range, options);
    if (IsUserAnnotatedCrossProgramPrefetch(*value, options)) {
      cross_program_prefetches.prefetches.push_back(buffer_interval);
    } else if (IsCrossProgramPrefetchCandidate(*value, alias_analysis,
                                               options)) {
      cross_program_prefetches.candidates.push_back(buffer_interval);
    } else if (MemorySpaceAssignmentUtils::
                   DoesCrossProgramPrefetchBufferMatchAnyFilter(
                       options.msa_sort_order_overrides, buffer_interval)) {
      cross_program_prefetches.candidates.push_back(buffer_interval);
    }
  }

  for (auto& prefetch : cross_program_prefetches.prefetches) {
    VLOG(3) << "User annotated cross-program prefetch: "
            << prefetch.buffer->ToString();
  }

  for (auto& prefetch : cross_program_prefetches.prefetches) {
    VLOG(3) << "User annotated cross-program prefetch: "
            << prefetch.buffer->ToString();
  }

  DefaultCrossProgramPrefetchBufferIntervalComparator default_comparator(
      hlo_live_range, options.msa_sort_order_overrides);
  BufferIntervalComparator* comparator =
      (options.default_cross_program_prefetch_heuristic &&
               options.buffer_interval_comparator
           ? options.buffer_interval_comparator
           : &default_comparator);
  absl::c_sort(cross_program_prefetches.candidates,
               comparator->GetComparisonFunctor());

  VLOG(3) << "Cross-program prefetch candidates: "
          << cross_program_prefetches.candidates.size()
          << ". Sorting criteria: " << comparator->DescribeComparisonCriteria();
  for (auto& candidate : cross_program_prefetches.candidates) {
    VLOG(3) << "Cross-program prefetch candidate. Sorting criteria: "
            << comparator->CriteriaToString(candidate)
            << ". Candidate: " << candidate.buffer->ToString();
  }
  return cross_program_prefetches;
}

}  // namespace

std::string AllocationValue::ToString() const {
  std::string out = absl::StrCat("computation = ", computation()->name());
  absl::StrAppend(
      &out, (requires_contiguous_allocation_ ? " (contiguous alloc)" : ""));
  absl::StrAppend(&out, "\n position:\n");
  absl::StrAppend(&out, "  ", defining_position_.ToString(), "\n");
  absl::StrAppend(&out, " uses:\n");
  for (const Use& use : uses_) {
    absl::StrAppend(&out, "  ", use.hlo_use.ToString(), "\n");
  }
  return out;
}

std::string AllocationValue::ToShortString() const {
  return absl::StrCat("computation = ", computation()->name(),
                      ", position = ", defining_position_.ToString(),
                      ", value = ", value_->ToShortString(),
                      (requires_contiguous_allocation_ ? " (cont alloc)" : ""));
}

bool MsaAlgorithm::IsIntervalPinnedToAlternateMemory(
    const MsaBufferInterval& interval) const {
  const Shape& shape = interval.buffer->shape();
  return shape.has_layout() &&
         shape.layout().memory_space() == options_.alternate_memory_space;
}

bool MsaAlgorithm::MatchesPrefetchContext(
    const PrefetchContext& context, absl::string_view producer_name,
    ShapeIndex producer_shape_index, absl::string_view consumer_name) const {
  return context.request->use->hlo_use.instruction->name() == consumer_name &&
         context.request->allocation_value->defining_position()
                 .instruction->name() == producer_name &&
         context.request->allocation_value->defining_position().index ==
             producer_shape_index;
}

MsaAlgorithm::MsaAlgorithm(HloModule* module, AllocationSequence* allocations,
                           const Options& options,
                           const HloAliasAnalysis& alias_analysis,
                           const HloLiveRange& hlo_live_range)
    : GlobalDecreasingSizeBestFitHeap(
          options.alignment_in_bytes,
          /*type=*/kSpatial, /*buffer_interval_compare=*/nullptr,
          (options.sliced_prefetch_options.max_slices() >
                   options.sliced_prefetch_options
                       .all_slice_time_permutations_threshold()
               ? SliceTimePermutationIterator::Ty::kPreferred
               : SliceTimePermutationIterator::Ty::kAll)),
      module_(module),
      allocations_(allocations),
      options_(options),
      alias_analysis_(alias_analysis),
      hlo_live_range_(hlo_live_range),
      peak_memory_usage_(hlo_live_range.schedule_end_time() + 1) {
  // Override buffer interval compare if provided.
  auto comparison_function = GetSpatialBufferIntervalCompare();
  if (options.buffer_interval_comparator) {
    comparison_function =
        options.buffer_interval_comparator->GetComparisonFunctor();
  }

  // Prioritize pinned buffers in the buffer interval order.
  buffer_interval_compare_ =
      [this, comparison_function = std::move(comparison_function)](
          const MsaBufferInterval& a, const MsaBufferInterval& b) {
        bool is_a_pinned = IsIntervalPinnedToAlternateMemory(a);
        bool is_b_pinned = IsIntervalPinnedToAlternateMemory(b);
        if (is_a_pinned && !is_b_pinned) {
          return true;
        }
        if (!is_a_pinned && is_b_pinned) {
          return false;
        }
        return comparison_function(a, b);
      };

  call_graph_ = CallGraph::Build(&alias_analysis_.dataflow_analysis().module());

  std::vector<float> initial_resources(hlo_live_range.schedule_end_time(), 1.0);
  if (options.cost_analysis) {
    const std::vector<HloInstruction*>& flattened_instructions =
        hlo_live_range.flattened_instruction_sequence().instructions();
    for (int i = 0; i < flattened_instructions.size(); ++i) {
      const HloInstruction* inst = flattened_instructions[i];
      if (inst->opcode() == HloOpcode::kWhile ||
          inst->opcode() == HloOpcode::kConditional) {
        initial_resources[i] = 0;
      } else {
        initial_resources[i] =
            options.cost_analysis->GetInstructionElapsed(*inst);
        if (options_.use_repeated_instance_for_preferred_prefetch_time ||
            options_.memory_bound_loop_optimizer_options.enabled()) {
          uint64_t fingerprint = absl::HashOf(MsaInstructionFingerprint(inst));
          fingerprint_map_[inst] = fingerprint;
          repeated_inst_map_[fingerprint].push_back(inst);
        }
      }
      VLOG(2) << "Initial resource[" << i << "] = " << initial_resources[i]
              << " (" << inst->name() << ")";
    }
  }
  prefetch_async_copy_resource_ = AsynchronousCopyResource(initial_resources);
  eviction_async_copy_resource_ = AsynchronousCopyResource(initial_resources);
}

void MsaAlgorithm::CreateAllocationValues(
    const MsaBufferInterval& buffer_interval,
    std::vector<AllocationValue>& allocation_values) const {
  const HloValue* value = buffer_interval.buffer;
  VLOG(3) << "Creating AllocationValues";

  // Find and sort all non-trivial (excluding GTE, Tuple, and bitcast)
  // positions. We create an AllocationValue object for each non-trivial
  // position. And for each AllocationValue object, we create an
  // AllocationSequence consisting of one or more Allocation objects.The reason
  // why we exclude the trivial positions from AllocationValue is because
  // Allocation objects have special support for tuples and bitcasts.
  const absl::flat_hash_map<const HloInstruction*, int64_t>&
      instruction_schedule = hlo_live_range_.instruction_schedule();
  std::vector<HloPosition> positions;
  for (const HloPosition& position : value->positions()) {
    const HloInstruction* instruction = position.instruction;
    if (instruction->opcode() != HloOpcode::kGetTupleElement &&
        instruction->opcode() != HloOpcode::kTuple &&
        instruction->opcode() != HloOpcode::kBitcast) {
      positions.push_back(position);
    }
  }
  absl::c_stable_sort(positions,
                      [&](const HloPosition& pos1, const HloPosition& pos2) {
                        return instruction_schedule.at(pos1.instruction) <
                               instruction_schedule.at(pos2.instruction);
                      });

  // Create an AllocationValue for each non-trivial position.
  int beginning_idx = allocation_values.size();
  for (int i = 0; i < positions.size(); ++i) {
    const HloPosition& position = positions.at(i);
    allocation_values.emplace_back(value, position, buffer_interval.size);
  }

  std::vector<HloUse> uses(value->GetUses().begin(), value->GetUses().end());
  absl::c_stable_sort(uses, [&](const HloUse& use1, const HloUse& use2) {
    return instruction_schedule.at(use1.instruction) <
           instruction_schedule.at(use2.instruction);
  });

  // Associate each use with an AllocationValue. Each AllocationValue contains a
  // position and uses in the same computation. Furthermore, if the original
  // HloValue had multiple non-trivial positions in the same computation, those
  // will get their own AllocationValue as well. We split these HloValues so
  // that when we insert CopyStart/CopyDone in CopyAllocation::Process, they
  // point to the latest position. We then replace the operand of the use with
  // CopyStart/CopyDone with an operand of the latest position.
  for (const HloUse& use : uses) {
    int64_t use_time = instruction_schedule.at(use.instruction);
    HloComputation* use_computation = use.instruction->parent();

    AllocationValue* last_allocation_value = nullptr;
    for (int i = beginning_idx; i < allocation_values.size(); ++i) {
      AllocationValue* allocation_value = &allocation_values.at(i);
      if (HloDataflowAnalysis::IsAsynchronousOperationDone(
              use.instruction->opcode())) {
        if (allocation_value->defining_instruction() ==
                use.instruction->operand(0) &&
            use.operand_index == allocation_value->defining_position().index) {
          last_allocation_value = allocation_value;
        }
      } else if (!HloDataflowAnalysis::IsAsynchronousOperationStart(
                     allocation_value->defining_instruction()->opcode()) &&
                 allocation_value->computation() == use_computation &&
                 instruction_schedule.at(
                     allocation_value->defining_position().instruction) <
                     use_time) {
        last_allocation_value = allocation_value;
      }
    }
    CHECK(last_allocation_value != nullptr);
    last_allocation_value->AddUse(use, use_time);
  }

  for (int i = beginning_idx; i < allocation_values.size(); ++i) {
    AllocationValue& allocation_value = allocation_values.at(i);
    if (HloDataflowAnalysis::IsAsynchronousOperationStart(
            allocation_value.defining_instruction()->opcode())) {
      CHECK_EQ(allocation_value.uses().size(), 1);
      CHECK(HloDataflowAnalysis::IsAsynchronousOperationDone(
          allocation_value.uses().at(0).hlo_use.instruction->opcode()));
      VLOG(3) << "Mark " << allocation_value.ToShortString()
              << " to require contiguous allocation because it is an async "
                 "start operation.";
      allocation_value.set_requires_contiguous_allocation(true);
    } else if (options_.position_requires_contiguous_allocation_fn(
                   allocation_value.defining_position())) {
      VLOG(3) << "Mark " << allocation_value.ToShortString()
              << " to require contiguous allocation because of options.";
      allocation_value.set_requires_contiguous_allocation(true);
    }
    VLOG(3) << "Created allocation value: "
            << allocation_values.at(i).ToString();
  }
}

void MsaAlgorithm::FindAliases(
    std::vector<AllocationValue>* allocation_values) const {
  absl::flat_hash_map<const HloInstruction*,
                      std::vector<const AllocationValue*>>
      values_by_defining_inst;
  for (AllocationValue& value : *allocation_values) {
    values_by_defining_inst[value.defining_instruction()].push_back(&value);
  }
  auto maybe_add_alias_with_instruction = [&](const HloInstruction* instruction,
                                              AllocationValue::Use* use) {
    auto aliased_values_it = values_by_defining_inst.find(instruction);
    if (aliased_values_it != values_by_defining_inst.end()) {
      for (const AllocationValue* aliased_value : aliased_values_it->second) {
        if (absl::c_find(use->aliases, aliased_value->defining_position()) ==
            use->aliases.end()) {
          VLOG(3) << "Adding aliasing for use " << use->hlo_use.ToString()
                  << " to " << aliased_value->ToShortString();
          use->aliases.push_back(aliased_value->defining_position());
        }
      }
    }
  };

  for (AllocationValue& value : *allocation_values) {
    for (AllocationValue::Use& use : value.uses()) {
      // Find any aliases with the instruction itself (operand and output must
      // alias).
      maybe_add_alias_with_instruction(use.hlo_use.instruction, &use);

      // Find any aliases with the parameters of called computations.
      for (const HloComputation* called_computation :
           use.hlo_use.instruction->called_computations()) {
        for (const HloInstruction* parameter_instruction :
             called_computation->parameter_instructions()) {
          maybe_add_alias_with_instruction(parameter_instruction, &use);
        }
      }

      // Special case for kWhile: the root of the body computation must alias as
      // well.
      if (use.hlo_use.instruction->opcode() == HloOpcode::kWhile) {
        HloPosition root_alias{
            use.hlo_use.instruction->while_body()->root_instruction(),
            use.hlo_use.operand_index};
        VLOG(3) << "Adding while body root aliasing for use "
                << use.hlo_use.ToString() << " to " << root_alias;
        use.aliases.push_back(root_alias);
      }
    }
  }
}

void MsaAlgorithm::ExtendScopedAlternateMemoryAllocations() {
  VLOG(1) << "Starting vmem expansion";

  // Iterate through all scoped allocations and try to expand them to the
  // largest contiguous open space available.
  for (std::unique_ptr<Allocation>& allocation : *allocations_) {
    if (!allocation->is_scoped_allocation()) {
      continue;
    }

    // Find the set of nodes that are live during allocation.
    std::vector<Chunk> live_nodes = interval_tree_.ChunksOverlappingInTime(
        allocation->start_time(), allocation->end_time());
    absl::c_sort(live_nodes, [](const Chunk& a, const Chunk& b) {
      return a.offset < b.offset;
    });

    // Loop over live_nodes to compute 2 things:
    // 1. The largest contiguous free chunk (biggest_free_chunk)
    // 2. The largest chunk we can get by moving the start time of the scoped
    //    allocation earlier (i.e., to max_end_before_scoped_allocation), and
    //    the end time later (i.e., to min_offset_after_scoped_allocation).
    int64_t min_offset_after_scoped_allocation = available_heap_size();
    int64_t max_end_before_scoped_allocation = 0;
    Chunk biggest_free_chunk = Chunk::FromOffsetSize(0, 0);
    for (int i = 0; i < live_nodes.size(); ++i) {
      const Chunk& chunk = live_nodes[i];
      if (allocation->chunk().chunk_end() <= chunk.offset) {
        min_offset_after_scoped_allocation =
            std::min(min_offset_after_scoped_allocation, chunk.offset);
      }
      if (allocation->chunk().offset >= chunk.chunk_end()) {
        max_end_before_scoped_allocation =
            std::max(max_end_before_scoped_allocation, chunk.chunk_end());
      }

      Chunk next_free_chunk = Chunk::FromOffsetEnd(
          GetAlignedOffset(chunk.chunk_end(), options_.alignment_in_bytes),
          (i + 1) < live_nodes.size() ? live_nodes[i + 1].offset
                                      : available_heap_size());
      if (next_free_chunk.size > biggest_free_chunk.size) {
        biggest_free_chunk = next_free_chunk;
      }
    }

    Chunk proposed_extended_chunk =
        Chunk::FromOffsetEnd(GetAlignedOffset(max_end_before_scoped_allocation,
                                              options_.alignment_in_bytes),
                             min_offset_after_scoped_allocation);

    // Check if we should extend the boundaries of the scoped allocation or
    // move it.
    Chunk proposed_chunk = allocation->chunk();
    std::string source;
    if (proposed_extended_chunk.size > proposed_chunk.size) {
      proposed_chunk = proposed_extended_chunk;
      source = "extended";
    }
    if (biggest_free_chunk.size > proposed_chunk.size) {
      proposed_chunk = biggest_free_chunk;
      source = "free";
    }
    if (source.empty()) {
      VLOG(3) << "Could not move the scoped allocation for "
              << allocation->defining_position().ToString()
              << "; Current fragmentation: " <<
          [&]() {
            int64_t occupied_size = 0;
            for (const Chunk& chunk : live_nodes) {
              occupied_size += chunk.size;
            }
            double fragmentation =
                static_cast<double>(available_heap_size() - occupied_size) /
                static_cast<double>(available_heap_size());
            return 100.0 * fragmentation;
          }() << "%";
      continue;
    }

    VLOG(1) << "Moving the scoped allocation for "
            << allocation->defining_position().ToString() << " from "
            << allocation->chunk().ToString() << " to "
            << proposed_chunk.ToString() << " (" << source
            << "); Size increase: "
            << (100.0 *
                static_cast<double>(proposed_chunk.size -
                                    allocation->chunk().size) /
                static_cast<double>(allocation->chunk().size))
            << "%";

    // Update the allocation. We don't need to update result_.chunk_map. It's
    // not used by MSA.
    *allocation->mutable_chunk() = proposed_chunk;
    result_.UpdatedHeapSize(proposed_chunk);
  }
}

std::string MsaAlgorithm::RequiredMemoryAssignment::ToString() const {
  std::string memory_space_str =
      memory_space == MemorySpace::kDefault ? "def" : "alt";
  std::string offset_str =
      offset == nullptr ? "null" : absl::StrCat(offset->offset);

  return absl::StrCat(
      "RequiredMemoryAssignment(memory_space=", memory_space_str,
      ", time=", time, ", offset=", offset_str, ")");
}

std::vector<const MsaBufferInterval*> MsaAlgorithm::GetSortedColocatedIntervals(
    const MsaBufferInterval& interval) const {
  std::vector<const MsaBufferInterval*> colocated_intervals;
  std::vector<const MsaBufferInterval*> worklist = {&interval};
  while (!worklist.empty()) {
    const MsaBufferInterval* item = worklist.back();
    worklist.pop_back();
    colocated_intervals.push_back(item);
    for (const HloValue* buffer_colocated : item->colocations) {
      worklist.push_back(&buffer_intervals_.at(buffer_colocated));
    }
  }

  absl::c_stable_sort(colocated_intervals, [&](const MsaBufferInterval* x,
                                               const MsaBufferInterval* y) {
    return std::make_pair(x->start, x->end) < std::make_pair(y->start, y->end);
  });
  return colocated_intervals;
}

bool MsaAlgorithm::IsUseAllowedInAlternateMemory(const AllocationValue& value,
                                                 const HloUse& use) const {
  const auto& instruction_schedule = hlo_live_range_.instruction_schedule();
  if (!options_.is_use_allowed_in_alternate_mem_fn(use)) {
    return false;
  }
  if (use.instruction->opcode() == HloOpcode::kWhile) {
    HloComputation* while_body = use.instruction->while_body();

    // We don't want to allocate this buffer in alternate memory if it will be
    // evicted anyway. Find out if it has an early use or a late definition that
    // would make sense to keep it in the alternate memory.
    HloValue* parameter_value =
        &alias_analysis_.dataflow_analysis().GetUniqueValueAt(
            while_body->parameter_instruction(0), use.operand_index);
    int64_t parameter_time =
        instruction_schedule.at(while_body->parameter_instruction(0));
    int64_t root_time = instruction_schedule.at(while_body->root_instruction());
    int64_t min_use_time = root_time;
    for (const HloUse& parameter_use : parameter_value->GetUses()) {
      int64_t use_time = instruction_schedule.at(parameter_use.instruction);
      if (parameter_use.instruction->opcode() != HloOpcode::kGetTupleElement &&
          parameter_use.instruction->opcode() != HloOpcode::kTuple &&
          parameter_use.instruction->opcode() != HloOpcode::kBitcast &&
          use_time > parameter_time) {
        min_use_time = std::min(min_use_time, use_time);
      }
    }
    // If there is no use of this buffer inside the while loop, there is no need
    // to allocate it in the loop.
    if (min_use_time == root_time) {
      VLOG(4) << "While allocation not allowed in alternate memory. "
              << "use time = " << min_use_time << ", root time = " << root_time;
      return false;
    }
    const Shape& shape = parameter_value->shape();
    // Allow the buffer in alternate memory if the buffer has a short live range
    // either at the beginning or end of the while loop body.
    if (!options_.prefetch_interval_picker->CanAllocateInAlternateMemoryNoCopy(
            shape, parameter_time, min_use_time)) {
      VLOG(4) << "While allocation not allowed in alternate memory. "
              << "use time = " << min_use_time << ", root time = " << root_time;
      return false;
    }
    // Check if there is a required assignment for the while loop output.
    HloValue* while_value =
        &alias_analysis_.dataflow_analysis().GetUniqueValueAt(
            use.instruction, use.operand_index);
    int64_t while_time = instruction_schedule.at(use.instruction);
    auto existing_required_assignment =
        RequiredMemoryAssignmentAt(while_value, while_time);
    if (existing_required_assignment &&
        existing_required_assignment->memory_space == MemorySpace::kDefault) {
      VLOG(4) << "While allocation not allowed in alternate memory because "
                 "there is a required default memory assignment.";
      return false;
    }
  } else if (use.instruction->opcode() == HloOpcode::kConditional) {
    // For any use of this conditional (the same value might be passed into
    // multiple called computations), determine if the parameter->first use
    // dependency is short.
    int64_t conditional_time = instruction_schedule.at(use.instruction);
    for (const AllocationValue::Use& other_use : value.uses()) {
      if (other_use.hlo_use.instruction != use.instruction) {
        continue;
      }
      // Operand 0 is not passed into the computation.
      if (other_use.hlo_use.operand_number == 0) {
        continue;
      }
      HloComputation* called_computation =
          use.instruction->called_computations().at(
              other_use.hlo_use.operand_number - 1);
      const HloInstruction* parameter_instruction =
          called_computation->parameter_instruction(0);
      HloValue* parameter_value =
          &alias_analysis_.dataflow_analysis().GetUniqueValueAt(
              parameter_instruction, other_use.hlo_use.operand_index);
      int64_t parameter_time = instruction_schedule.at(parameter_instruction);
      int64_t min_use_time = conditional_time;
      for (const HloUse& parameter_use : parameter_value->GetUses()) {
        if (parameter_use.instruction->parent() == called_computation &&
            parameter_use.instruction->opcode() !=
                HloOpcode::kGetTupleElement &&
            parameter_use.instruction->opcode() != HloOpcode::kTuple &&
            parameter_use.instruction->opcode() != HloOpcode::kBitcast) {
          min_use_time = std::min(
              min_use_time, instruction_schedule.at(parameter_use.instruction));
        }
      }
      if (options_.prefetch_interval_picker->CanAllocateInAlternateMemoryNoCopy(
              parameter_value->shape(), parameter_time, min_use_time)) {
        VLOG(4) << "Conditional allocation allowed in alternate memory for "
                   "computation = "
                << called_computation->name()
                << ", parameter time = " << parameter_time
                << ", min use time = " << min_use_time;
        return true;
      } else {
        VLOG(4) << "Conditional allocation not allowed in alternate memory for "
                   "computation = "
                << called_computation->name()
                << ", parameter time = " << parameter_time
                << ", min use time = " << min_use_time;
      }
    }
    return false;
  }

  return true;
}

namespace {
// Columns in buffer information:
// buffer_id: int. This value can be used to match the allocation in
// allocation information.
// buffer_name: string.
// alt_mem_benefit: float. Roughly corresponds to how much the cost analysis
// thought it would be beneficial to put this in the alternate memory. The
// higher the value, the more it is memory bound.
// size: int. In bytes.
// definition_time: int. Logical time this value was defined in the schedule.
// use_times: string. This is a semicolon-separated list of integers for all
// the use times.
// use_names: string. This is a semicolon-separated list of string
// representation of uses.
// is_scoped: int. A value of 1 indicates that the buffer is a scoped
// allocation.
constexpr absl::string_view kBufferInfoColumnNames =
    "buffer_id,buffer_name,alt_mem_benefit,size,definition_time,use_times,use_"
    "names,is_scoped";
}  // namespace

void MsaAlgorithm::AppendBufferInfoDebugString(
    const MsaBufferInterval& interval, std::string* debug_str) const {
  if (debug_str->empty()) {
    // Append the column names.
    absl::StrAppend(debug_str, kBufferInfoColumnNames, "\n");
  }
  const HloBuffer& buffer =
      alias_analysis_.GetBufferContainingValue(*interval.buffer);
  const auto& instruction_schedule = hlo_live_range_.instruction_schedule();
  int64_t definition_time =
      instruction_schedule.at(interval.buffer->defining_position().instruction);
  std::vector<std::pair<int64_t, std::string>> uses;
  for (const HloValue* value : buffer.values()) {
    for (const HloUse& use : value->GetUses()) {
      uses.push_back(
          {instruction_schedule.at(use.instruction), use.ToString()});
    }
  }
  absl::c_sort(uses);
  std::vector<int64_t> use_times;
  std::vector<std::string> use_names;
  use_times.reserve(uses.size());
  use_names.reserve(uses.size());
  for (const auto& use : uses) {
    use_times.push_back(use.first);
    use_names.push_back(use.second);
  }

  absl::StrAppend(debug_str, buffer.id(), ",");
  absl::StrAppend(debug_str, "\"", interval.buffer->ToShortString(), "\",");
  auto alternate_memory_benefit =
      options_.prefetch_interval_picker->BufferIntervalAlternateMemoryBenefit(
          interval);
  absl::StrAppend(
      debug_str, alternate_memory_benefit ? *alternate_memory_benefit : 0, ",");
  absl::StrAppend(debug_str, interval.size, ",");
  absl::StrAppend(debug_str, definition_time, ",");
  absl::StrAppend(debug_str, "\"", absl::StrJoin(use_times, ";"), "\",");
  absl::StrAppend(debug_str, "\"", absl::StrJoin(use_names, ";"), "\",");
  absl::StrAppend(debug_str, "0");  // is_scoped
  absl::StrAppend(debug_str, "\n");
}

void MsaAlgorithm::AppendScopedAllocationBufferInfoDebugString(
    const HloInstruction* instruction, int64_t time, int64_t size,
    std::string& debug_str) const {
  if (debug_str.empty()) {
    // Append the column names.
    absl::StrAppend(&debug_str, kBufferInfoColumnNames, "\n");
  }
  const HloBuffer& buffer = alias_analysis_.GetUniqueBufferAt(instruction);

  // As a convention, we use negative values for scoped allocations.
  absl::StrAppend(&debug_str, -buffer.id(), ",");
  absl::StrAppend(&debug_str, "\"scoped allocation for ", instruction->name(),
                  "\",");
  absl::StrAppend(&debug_str, 0, ",");  // alt_mem_benefit
  absl::StrAppend(&debug_str, size, ",");
  absl::StrAppend(&debug_str, time, ",");
  absl::StrAppend(&debug_str, "\"\",");  // use_times
  absl::StrAppend(&debug_str, "\"\",");  // use_names
  absl::StrAppend(&debug_str, "1");      // is_scoped
  absl::StrAppend(&debug_str, "\n");
}

void MsaAlgorithm::AppendAllocationInfoDebugString(
    const Allocation& allocation, std::string& debug_str) const {
  // Columns in allocation information:
  // buffer_id: int. This value can be used the match with buffer info.
  // size: int. In bytes.
  // offset: int. In bytes.
  // start_time: int. Logical start time of the allocation.
  // end_time: int. Logical end time of the allocation.
  if (debug_str.empty()) {
    // Append the column names.
    absl::StrAppend(&debug_str, "buffer_id,size,offset,start_time,end_time\n");
  }
  if (allocation.memory_space() == MemorySpace::kAlternate) {
    const HloPosition& position = allocation.defining_position();
    const HloBuffer& buffer =
        alias_analysis_.GetUniqueBufferAt(position.instruction, position.index);
    // As a convention, we use negative values for scoped allocations.
    absl::StrAppend(
        &debug_str,
        allocation.is_scoped_allocation() ? -buffer.id() : buffer.id(), ",");
    absl::StrAppend(&debug_str, allocation.chunk().size, ",");
    absl::StrAppend(&debug_str, allocation.chunk().offset, ",");
    absl::StrAppend(&debug_str, allocation.start_time(), ",");
    absl::StrAppend(&debug_str, allocation.end_time(), "\n");
  }
}

void MsaAlgorithm::DumpDebugStringsIfEnabled() const {
  if (!options_.dump_fn) {
    return;
  }
  options_.dump_fn("bufferinfo", buffer_info_str_);
  options_.dump_fn("allocinfo", allocation_info_str_);
  options_.dump_fn("scheduleinfo", instruction_schedule_str_);
}

std::string MsaAlgorithm::OptionalRequiredMemoryAssignmentToString(
    const std::optional<MsaAlgorithm::RequiredMemoryAssignment>& assignment) {
  if (!assignment.has_value()) {
    return "null";
  }

  if (assignment->offset) {
    return absl::StrCat(MemorySpaceToString(assignment->memory_space),
                        " at offset ", assignment->offset->offset);
  }

  return MemorySpaceToString(assignment->memory_space);
}

absl::Status MsaAlgorithm::OptimizeMemoryBoundLoop(int loop_start_idx,
                                                   int loop_end_idx,
                                                   int loop_size) {
  // The MemoryBoundLoopOptimizer works with a minimum of three unrolled loop
  // iterations: previous, current, and next. So, we pick the second iteration
  // out of the loop as the current iteration.
  const int iteration_start_idx = loop_start_idx + loop_size;
  const int iteration_end_idx = iteration_start_idx + loop_size;

  TF_ASSIGN_OR_RETURN(std::unique_ptr<MemoryBoundLoopOptimizer> optimizer,
                      MemoryBoundLoopOptimizer::Create(
                          iteration_start_idx, iteration_end_idx,
                          hlo_live_range_, alias_analysis_, options_));
  optimizer->Optimize();

  // Check if this unrolled loop is in a while loop.
  const auto& instruction_sequence =
      hlo_live_range_.flattened_instruction_sequence().instructions();
  std::vector<HloInstruction*> callers = call_graph_->GetComputationCallers(
      instruction_sequence[loop_start_idx]->parent());
  const bool is_in_while_loop =
      callers.size() == 1 && callers.front()->opcode() == HloOpcode::kWhile;

  // Update the loop_optimized_allocations_map_ with the output of the
  // optimizer.
  for (MemoryBoundLoopOptimizer::LoopValue& value : optimizer->loop_values()) {
    if (value.allocations.empty() || !value.IsAllocationTypeSupported()) {
      continue;
    }
    loop_optimized_allocations_.push_back(std::move(value.allocations));
    const AllocationSequence& sequence = loop_optimized_allocations_.back();
    CHECK(!sequence.empty());
    VLOG(3) << "  alloc: " << sequence.back()->ToString();
    for (const auto& allocation : sequence) {
      // Check if the loop is in a while loop and the position needs to be
      // allocated in the default memory.
      const bool require_pos_in_default_space =
          is_in_while_loop &&
          (allocation->memory_space() == MemorySpace::kDefault ||
           allocation->is_copy_allocation());
      for (const HloUse& use : allocation->uses()) {
        const int64_t use_idx =
            hlo_live_range_.instruction_schedule().at(use.instruction) -
            iteration_start_idx;
        CHECK_GE(use_idx, 0);
        CHECK_LT(use_idx, loop_size);
        for (int64_t i = loop_start_idx + use_idx; i <= loop_end_idx;
             i += loop_size) {
          HloInstruction* repeated_inst = instruction_sequence[i];
          CHECK_EQ(use.instruction->opcode(), repeated_inst->opcode());
          CHECK_EQ(use.instruction->operand_count(),
                   repeated_inst->operand_count());
          CHECK_LT(use.operand_number, repeated_inst->operand_count());
          HloUse repeated_use{repeated_inst, use.operand_number,
                              use.operand_index};
          loop_optimized_allocations_map_[repeated_use] = {use_idx, loop_size,
                                                           allocation.get()};
          VLOG(3) << " Setting optimized allocations map. Use: "
                  << repeated_use.ToString() << " idx: " << use_idx
                  << " allocation: " << allocation->ToString();
          if (require_pos_in_default_space) {
            const HloValue& value =
                alias_analysis_.dataflow_analysis().GetUniqueValueAt(
                    repeated_inst->operand(use.operand_number),
                    use.operand_index);
            // If any of the positions is a parameter in a while loop, we add a
            // required assignment in the default memory space.
            for (const HloPosition& value_position : value.positions()) {
              if (value_position.instruction->parent() ==
                      repeated_inst->parent() &&
                  value_position.instruction->opcode() ==
                      HloOpcode::kParameter) {
                AddRequiredAssignment(value_position.instruction,
                                      value_position.index,
                                      MemorySpace::kDefault);
                break;
              }
            }
          }
        }
      }
    }
  }
  return absl::OkStatus();
}

namespace {
// A helper function to get the distance between a use and its producer (or -1
// if producer is a gte, parameter or tuple).
std::function<int(const HloInstruction*)> GetOperandDistanceFunction(
    const HloLiveRange& hlo_live_range, const HloInstruction* use_inst) {
  const int use_idx = hlo_live_range.instruction_schedule().at(use_inst);
  return [&, use_idx](const HloInstruction* operand) -> int {
    // We just use -1 for parameter, tuple, gte and constant instructions. We
    // could make this "see through" the gtes if we get too many false
    // positives.
    if (operand->opcode() == HloOpcode::kParameter ||
        operand->opcode() == HloOpcode::kTuple ||
        operand->opcode() == HloOpcode::kGetTupleElement ||
        operand->opcode() == HloOpcode::kConstant) {
      return -1;
    }
    return use_idx - hlo_live_range.instruction_schedule().at(operand);
  };
}

// A helper function to check if the operand distances of two instructions
// are compatible. This assumes `a` is scheduled loop size candidate
// instructions before `b`. The operand distances are compatible if either
// distance is -1, or if they are the same, or if they are separated by loop
// size candidate.
bool AreOperandCandidatesCompatible(int loop_size_candidate,
                                    absl::Span<const int> a_distances,
                                    absl::Span<const int> b_distances) {
  if (a_distances.size() != b_distances.size()) {
    return false;
  }
  for (int i = 0; i < a_distances.size(); ++i) {
    const int a_value = a_distances.at(i);
    const int b_value = b_distances.at(i);
    if (a_value != -1 && b_value != -1 &&
        a_value + loop_size_candidate != b_value && a_value != b_value) {
      return false;
    }
  }
  return true;
}
}  // namespace

void MsaAlgorithm::IdentifyAndOptimizeMemoryBoundLoops() {
  absl::flat_hash_map<uint64_t, int> fingerprint_schedule_map;
  const auto& instruction_sequence =
      hlo_live_range_.flattened_instruction_sequence().instructions();
  // The minimum and maximum loop sizes that we consider.
  const int kMinLoopSize = 4;
  const int kMaxLoopSize = 400;
  int optimized_loop_idx = 0;
  while (optimized_loop_idx < instruction_sequence.size()) {
    // Iterate over the flattened instruction sequence. We first try to find a
    // loop candidate where the fingerprint between two instructions matches by
    // the loop size candidate.
    int loop_size_candidate = -1;
    int loop_start_idx = -1;
    int loop_end_idx = -1;
    for (; optimized_loop_idx < instruction_sequence.size();
         ++optimized_loop_idx) {
      const HloInstruction* inst = instruction_sequence[optimized_loop_idx];
      auto fingerprint_it = fingerprint_map_.find(inst);
      if (inst->opcode() != HloOpcode::kParameter &&
          inst->opcode() != HloOpcode::kTuple &&
          inst->opcode() != HloOpcode::kGetTupleElement &&
          fingerprint_it != fingerprint_map_.end()) {
        // Find and the latest instruction with the same fingerprint as this.
        auto fingerprint_schedule_it =
            fingerprint_schedule_map.find(fingerprint_it->second);
        if (fingerprint_schedule_it != fingerprint_schedule_map.end()) {
          int distance = optimized_loop_idx - fingerprint_schedule_it->second;
          if (distance >= kMinLoopSize && distance <= kMaxLoopSize) {
            // We found two instructions with the same fingerprint. The distance
            // between the two is the loop size candidate.
            loop_size_candidate = distance;
            // Update the fingerprint map with the current loop index so that if
            // the loop size candidate doesn't find a valid loop, we can resume
            // searching from this instruction.
            fingerprint_schedule_map[fingerprint_it->second] =
                optimized_loop_idx;
            break;
          }
        }
        fingerprint_schedule_map[fingerprint_it->second] = optimized_loop_idx;
      }

      VLOG(3) << " " << optimized_loop_idx << ": "
              << instruction_sequence[optimized_loop_idx]->parent()->name()
              << " " << instruction_sequence[optimized_loop_idx]->name()
              << " fingerprint: "
              << (fingerprint_it == fingerprint_map_.end()
                      ? "none"
                      : std::to_string(fingerprint_it->second));
    }
    VLOG(3) << "Loop size candidate: " << loop_size_candidate;
    if (loop_size_candidate == -1) {
      break;
    }

    std::vector<std::vector<int>> operand_distances;

    // Scan the instructions with the candidate loop size. We try to calculate
    // the size of the loop by finding the instructions that are loop size
    // candidate apart, have the same fingerprint and compatible operand
    // distances. We start scanning the candidate loop a few instructions
    // earlier than the fingerprint identified in case the loop starts a bit
    // earlier than the fingerprint logic.
    const int kLoopScanHeadStart = 10;
    for (int i = std::max(
             0, optimized_loop_idx - loop_size_candidate - kLoopScanHeadStart);
         i < instruction_sequence.size(); ++i) {
      const HloInstruction* inst = instruction_sequence[i];
      auto fingerprint_it = fingerprint_map_.find(inst);
      auto ignore_op = [](const HloInstruction* instruction) {
        return instruction->opcode() == HloOpcode::kParameter ||
               instruction->opcode() == HloOpcode::kTuple ||
               instruction->opcode() == HloOpcode::kGetTupleElement;
      };
      // We trigger this if statement until we find the start of the loop.
      if (loop_start_idx == -1) {
        if (i > optimized_loop_idx - loop_size_candidate) {
          break;
        }
        if (ignore_op(inst) || fingerprint_it == fingerprint_map_.end()) {
          continue;
        }
        if (i + loop_size_candidate >= instruction_sequence.size()) {
          break;
        }
        const HloInstruction* candidate_inst =
            instruction_sequence[i + loop_size_candidate];
        auto candidate_fingerprint_it = fingerprint_map_.find(candidate_inst);
        if (ignore_op(candidate_inst) ||
            candidate_fingerprint_it == fingerprint_map_.end() ||
            fingerprint_it->second != candidate_fingerprint_it->second) {
          // Fingerprint mismatch.
          continue;
        }
        std::vector<int> inst_operand_distances;
        absl::c_transform(inst->operands(),
                          std::back_inserter(inst_operand_distances),
                          GetOperandDistanceFunction(hlo_live_range_, inst));
        std::vector<int> candidate_inst_operand_distances;
        absl::c_transform(
            candidate_inst->operands(),
            std::back_inserter(candidate_inst_operand_distances),
            GetOperandDistanceFunction(hlo_live_range_, candidate_inst));
        VLOG(3) << "i : " << i << " "
                << absl::StrJoin(inst_operand_distances, ", ") << " | "
                << absl::StrJoin(candidate_inst_operand_distances, ", ");
        if (!AreOperandCandidatesCompatible(loop_size_candidate,
                                            inst_operand_distances,
                                            candidate_inst_operand_distances)) {
          // Operand distance mismatch.
          continue;
        }
        // Found the start of the loop.
        loop_start_idx = i;
      }
      if (inst->parent() != instruction_sequence[loop_start_idx]->parent()) {
        VLOG(3) << "Mismatch (computation) at " << i << ": "
                << inst->parent()->name() << " vs "
                << instruction_sequence[loop_start_idx]->parent()->name();
        break;
      }
      operand_distances.push_back({});
      if (fingerprint_it == fingerprint_map_.end()) {
        continue;
      }
      absl::c_transform(inst->operands(),
                        std::back_inserter(operand_distances.back()),
                        GetOperandDistanceFunction(hlo_live_range_, inst));
      if (i >= loop_start_idx + loop_size_candidate) {
        // Verify that this still obeys the fingerprint and operand distance
        // invariants.
        const HloInstruction* prev_inst =
            instruction_sequence[i - loop_size_candidate];
        auto prev_fingerprint_it = fingerprint_map_.find(prev_inst);
        if (prev_fingerprint_it == fingerprint_map_.end()) {
          break;
        }
        if (ignore_op(inst) || ignore_op(prev_inst)) {
          if (inst->opcode() != prev_inst->opcode()) {
            VLOG(3) << "Mismatch (opcode) at " << i << ", "
                    << (i - loop_size_candidate) << ": " << inst->opcode()
                    << " vs " << prev_inst->opcode();
            break;
          }
          if (inst->operand_count() != prev_inst->operand_count()) {
            VLOG(3) << "Mismatch (# operands) at " << i << ", "
                    << (i - loop_size_candidate) << ": "
                    << inst->operand_count() << " vs "
                    << prev_inst->operand_count();
            break;
          }
        }
        if (fingerprint_it->second != prev_fingerprint_it->second) {
          VLOG(3) << "Mismatch (fp) at " << i << ", "
                  << (i - loop_size_candidate) << ": " << fingerprint_it->second
                  << " vs " << prev_fingerprint_it->second;
          break;
        }
        if (!AreOperandCandidatesCompatible(
                loop_size_candidate,
                *(operand_distances.rbegin() + loop_size_candidate),
                operand_distances.back())) {
          VLOG(3) << "Mismatch (op) at " << i << ", "
                  << (i - loop_size_candidate) << ": "
                  << absl::StrJoin(operand_distances.back(), ", ") << " vs "
                  << absl::StrJoin(
                         *(operand_distances.rbegin() + loop_size_candidate),
                         ", ");
          break;
        }
      }
      loop_end_idx = i;
    }
    float num_iterations = 0;
    if (loop_start_idx != -1) {
      num_iterations = static_cast<float>(loop_end_idx + 1 - loop_start_idx) /
                       loop_size_candidate;
    }
    VLOG(3) << "Loop start: " << loop_start_idx << " loop end: " << loop_end_idx
            << " num iterations: " << num_iterations;

    optimized_loop_idx = std::max(optimized_loop_idx, loop_end_idx) + 1;

    if (num_iterations >=
        options_.memory_bound_loop_optimizer_options.min_num_iterations()) {
      VLOG(2) << "Found valid loop. Loop start: " << loop_start_idx
              << " loop end: " << loop_end_idx
              << " num iterations: " << num_iterations;

      TF_CHECK_OK(OptimizeMemoryBoundLoop(loop_start_idx, loop_end_idx,
                                          loop_size_candidate));
    }
  }
}

bool MsaAlgorithm::IsAsyncConversionCandidate(
    const HloInstruction* instruction) const {
  bool meets_special_preconditions =
      IsAsyncConversionCopyCandidate(instruction) ||
      IsAsyncConversionSliceCandidate(instruction) ==
          AsyncConversionResult::kSuccess;
  if (!meets_special_preconditions) {
    return false;
  }

  for (auto& operand : instruction->operands()) {
    // TODO(b/374835319): relax the operand constraint to be able to cover
    // nested sync data movement cases.
    if (IsAsyncConversionCandidate(operand)) {
      VLOG(4) << "The instruction is not considered to be replaced, because it "
                 "potentially has a replaceable operand.";
      return false;
    }
    const HloValue& operand_value = alias_analysis_.dataflow_analysis()
                                        .GetValueSet(operand)
                                        .GetUniqueValue();
    if (!buffer_intervals_.at(&operand_value).need_allocation) {
      VLOG(4)
          << "The instruction is not considered to be replaced, because its "
             "operand value doesn't need an allocation.";
      return false;
    }
  }

  const HloValue& value = alias_analysis_.dataflow_analysis()
                              .GetValueSet(instruction)
                              .GetUniqueValue();
  if (!buffer_intervals_.at(&value).need_allocation) {
    VLOG(4) << "The instruction is not considered to be replaced, because its "
               "output doesn't need an allocation and it might be too late to "
               "replace this instruction.";
    return false;
  }
  if (value.IsRootOf(instruction->parent())) {
    VLOG(4) << "The instruction is not considered to be replaced, because its "
               "output value is in the root of the computation.";
    return false;
  }
  if (finalized_values_.contains(&value)) {
    VLOG(4) << "The instruction is not considered to be replaced, because its "
               "output value is in the finalized values.";
    return false;
  }
  if (buffer_intervals_.at(&value).size > available_heap_size()) {
    VLOG(4) << "The instruction is not considered to be replaced, because its "
               "output value is too large to fit in the heap.";
    return false;
  }
  // This check is here only because we skip processing the values that are not
  // allowed in alternate memory.
  if (!MemorySpaceAssignmentUtils::IsIntervalAllowedInAlternateMemory(
          buffer_intervals_.at(&value), options_.alternate_memory_space)) {
    VLOG(4) << "The instruction is not considered to be replaced, because its "
               "output value is not allowed in alternate memory.";
    return false;
  }

  for (const HloInstruction* user : instruction->users()) {
    if (HloDataflowAnalysis::IsAsynchronousOperationStart(user->opcode())) {
      VLOG(4) << "The instruction is not considered to be replaced, because "
                 "its used by an async start operation that might require "
                 "contiguous allocation.";
      return false;
    }
  }

  return true;
}

bool MsaAlgorithm::IsAsyncConversionCopyCandidate(
    const HloInstruction* instruction) const {
  if (!options_.enable_sync_copy_replacement) {
    return false;
  }
  if (instruction->opcode() != HloOpcode::kCopy) {
    return false;
  }
  if (failed_async_conversions_.contains(instruction)) {
    return false;
  }
  if (instruction->IsRoot()) {
    // Root copy is not replaceable with current implementation, because the
    // instruction has no uses
    return false;
  }
  if (instruction->operand(0)->shape() != instruction->shape()) {
    VLOG(5) << "Sync copy " << instruction->ToShortString()
            << " is not replaceable, because the operand and output shapes do "
               "not match. This could be because the copy is changing the "
               "layout. operand(0) shape: "
            << instruction->operand(0)->shape().ToString(/*print_layout=*/true)
            << ", output shape: "
            << instruction->shape().ToString(/*print_layout=*/true);
    return false;
  }
  if (instruction->shape().layout().memory_space() !=
          static_cast<int64_t>(MemorySpace::kDefault) ||
      instruction->operand(0)->shape().layout().memory_space() !=
          static_cast<int64_t>(MemorySpace::kDefault)) {
    VLOG(5) << "Sync copy " << instruction->ToShortString()
            << " is not replaceable, because the operand or output have an "
               "initial assignment.";
    return false;
  }
  return true;
}

namespace {

bool IsTrivialInstruction(const HloInstruction* instruction) {
  return instruction->opcode() == HloOpcode::kGetTupleElement ||
         instruction->opcode() == HloOpcode::kTuple ||
         instruction->opcode() == HloOpcode::kBitcast;
}

bool IsSliceLikeInstruction(const HloInstruction* instruction) {
  return instruction->opcode() == HloOpcode::kSlice ||
         instruction->opcode() == HloOpcode::kDynamicSlice;
}

}  // namespace

MsaAlgorithm::AsyncConversionResult
MsaAlgorithm::IsAsyncConversionSliceCandidate(
    const HloInstruction* instruction) const {
  if (!options_.enable_sync_slice_replacement) {
    return AsyncConversionResult::kFeatureNotEnabled;
  }
  if (failed_async_conversions_.contains(instruction)) {
    return failed_async_conversions_.at(instruction);
  }
  if (!IsSliceLikeInstruction(instruction)) {
    return AsyncConversionResult::kFailedPrecondition;
  }

  if (instruction->IsRoot()) {
    // Root slice is not replaceable with current implementation, because the
    // instruction has no uses
    return AsyncConversionResult::kFailedPrecondition;
  }

  if (!options_.is_async_slice_implemented_fn(instruction)) {
    VLOG(4) << "The sync slice is not considered to be replaced, because the "
               "async version is not implemented for "
            << instruction->ToShortString();
    return AsyncConversionResult::kFailedPrecondition;
  }

  if (instruction->shape().layout().memory_space() !=
          static_cast<int64_t>(MemorySpace::kDefault) ||
      instruction->operand(0)->shape().layout().memory_space() !=
          static_cast<int64_t>(MemorySpace::kDefault)) {
    VLOG(4) << "Sync slice " << instruction->ToShortString()
            << " is not replaceable, because the operand or output have an "
               "initial assignment.";
    return AsyncConversionResult::kFailedPrecondition;
  }
  if (instruction->shape().element_type() !=
      instruction->operand(0)->shape().element_type()) {
    VLOG(4) << "Sync slice " << instruction->ToShortString()
            << " is not replaceable because the operand and output have "
               "different element types.";
    return AsyncConversionResult::kFailedPrecondition;
  }
  return AsyncConversionResult::kSuccess;
}

std::vector<const HloValue*> MsaAlgorithm::GenerateJointProcessedValues(
    const HloValue* entrance_value) {
  std::vector<const HloValue*> worklist = {entrance_value};
  if (options_.enable_sync_copy_replacement ||
      options_.enable_sync_slice_replacement) {
    // Adds the HloValue that is related to a given instruction to the worklist
    auto add_to_worklist = [&](const HloInstruction* inst) {
      const HloValue& next_value = alias_analysis_.dataflow_analysis()
                                       .GetValueSet(inst)
                                       .GetUniqueValue();
      if (std::find(worklist.begin(), worklist.end(), &next_value) ==
          worklist.end()) {
        worklist.push_back(&next_value);
      }
    };

    for (size_t idx = 0; idx < worklist.size(); ++idx) {
      const HloValue* value = worklist.at(idx);
      // Values that are related to the current value through a sync copy use
      // are added to the worklist.
      for (const auto& use : value->GetUses()) {
        if (IsAsyncConversionCandidate(use.instruction)) {
          add_to_worklist(use.instruction);
        }
      }
      // Expand the worklist to include values that connect to the current
      // value as sync copy operands, if any.
      HloInstruction* defining_instruction = value->instruction();
      if (IsAsyncConversionCandidate(defining_instruction)) {
        // The first operand of slice like instruction (slice or dynamic-slice)
        // is the defining instruction.
        add_to_worklist(defining_instruction->operand(0));
      }
    }
    // We're sensitive to the order of the worklist.
    absl::c_stable_sort(worklist, [&](const HloValue* a, const HloValue* b) {
      return std::make_pair(buffer_intervals_.at(a).start,
                            buffer_intervals_.at(a).end) <
             std::make_pair(buffer_intervals_.at(b).start,
                            buffer_intervals_.at(b).end);
    });
    UpdateSyncDataMovementCandidatesForJointProcessedValues(worklist);
  }
  return worklist;
}

void MsaAlgorithm::UpdateSyncDataMovementCandidatesForJointProcessedValues(
    const std::vector<const HloValue*>& joint_processed_values) {
  absl::flat_hash_set<const HloInstruction*> replaceable_sync_instructions;
  absl::flat_hash_set<const HloInstruction*> do_not_touch_instructions;
  for (const HloValue* value : joint_processed_values) {
    for (const auto& use : value->GetUses()) {
      bool is_use_replaceable_sync_candidate =
          IsAsyncConversionCandidate(use.instruction);
      if (is_use_replaceable_sync_candidate &&
          !do_not_touch_instructions.contains(use.instruction)) {
        replaceable_sync_instructions.insert(use.instruction);
      }
    }
    HloInstruction* inst = value->instruction();
    bool is_inst_replaceable_sync_candidate = IsAsyncConversionCandidate(inst);
    if (is_inst_replaceable_sync_candidate &&
        !do_not_touch_instructions.contains(inst)) {
      replaceable_sync_instructions.insert(inst);
    }
  }

  sorted_async_conversion_candidates_.clear();
  sorted_async_conversion_candidates_.insert(
      sorted_async_conversion_candidates_.end(),
      replaceable_sync_instructions.begin(),
      replaceable_sync_instructions.end());
  const auto& instruction_schedule = hlo_live_range_.instruction_schedule();
  absl::c_stable_sort(sorted_async_conversion_candidates_,
                      [&instruction_schedule](const HloInstruction* a,
                                              const HloInstruction* b) {
                        return instruction_schedule.at(a) <
                               instruction_schedule.at(b);
                      });
  VLOG(3) << "Sorted pending replaceable copies: ";
  if (sorted_async_conversion_candidates_.empty()) {
    VLOG(3) << "  --Empty--";
  }
  for (size_t idx = 0; idx < sorted_async_conversion_candidates_.size();
       ++idx) {
    VLOG(3) << "  " << idx + 1 << "/"
            << sorted_async_conversion_candidates_.size() << ") "
            << sorted_async_conversion_candidates_.at(idx)->ToShortString();
  }
}

namespace {

void NicePrintAllocationValues(
    const std::vector<AllocationValue>& allocation_values, int log_level) {
  if (VLOG_IS_ON(log_level)) {
    VLOG(log_level) << "Joint-processed allocation values: ";
    if (allocation_values.empty()) {
      VLOG(log_level) << "  --Empty--";
    }
    for (int i = 0; i < allocation_values.size(); i++) {
      VLOG(log_level) << "  " << i + 1 << "/" << allocation_values.size()
                      << ") " << allocation_values.at(i).ToShortString();
    }
  }
}

}  // namespace

void MsaAlgorithm::ColorColocatedIntervalsToAlternate(
    const std::vector<const MsaBufferInterval*>& colocated_intervals) {
  for (const MsaBufferInterval* colocated_interval : colocated_intervals) {
    const HloValue* value = colocated_interval->buffer;
    // Color all of the aliased reserved buffers here because reserved
    // alternate memory allocations will not have an entry in preset
    // allocations that is normally used for coloring.
    for (auto& position : value->positions()) {
      VLOG(4) << "Coloring " << position.ToString();
      Shape* shape = ShapeUtil::GetMutableSubshape(
          position.instruction->mutable_shape(), position.index);
      CHECK(shape->IsArray())
          << "Coloring a shape that is not an array: " << position.ToString();
      shape->mutable_layout()->set_memory_space(
          options_.alternate_memory_space);
    }
  }
}

void MsaAlgorithm::CreateAllocationValuesForJointProcessedValues(
    JointAllocationProposal& proposal) {
  for (auto& interval_hlo : proposal.values) {
    auto& interval = buffer_intervals_.at(interval_hlo);

    if (finalized_values_.contains(interval_hlo)) {
      VLOG(3) << "Skip " << interval.buffer->ToShortString()
              << " because it is already processed.";
      continue;
    }

    if (!interval.need_allocation) {
      VLOG(3) << "Skip " << interval.buffer->ToShortString()
              << " because it doesn't need an allocation.";
      continue;
    }

    if (!MemorySpaceAssignmentUtils::IsIntervalAllowedInAlternateMemory(
            interval, options_.alternate_memory_space)) {
      VLOG(3) << "Skip " << interval.buffer->ToShortString()
              << " because it is not allowed in the alternate memory.";
      continue;
    }

    HloInstruction* inst = interval.buffer->instruction();
    HloModule* module = inst->GetModule();

    // Don't intra-program prefetch a cross program prefetch
    auto cross_program_prefetches = module->CrossProgramPrefetches();
    if (inst->opcode() == HloOpcode::kParameter &&
        absl::c_find_if(cross_program_prefetches, [&](auto& info) {
          return info.parameter == inst->parameter_number() &&
                 info.index == interval.buffer->index();
        }) != module->CrossProgramPrefetches().end()) {
      VLOG(3) << "Skip " << interval.buffer->ToShortString()
              << " because it is cross-program prefetched.";
      continue;
    }

    if (!options_.enable_window_prefetch &&
        interval.size > available_heap_size()) {
      const HloInstruction* defining_instruction =
          interval.buffer->instruction();
      auto may_be_replaced_by_slice_fn = [this](const HloInstruction* user) {
        return IsInstructionPendingReplacements(user) &&
               IsSliceLikeInstruction(user);
      };
      bool may_be_replaced_by_slice = std::any_of(
          defining_instruction->users().begin(),
          defining_instruction->users().end(), may_be_replaced_by_slice_fn);

      if (!may_be_replaced_by_slice) {
        VLOG(3) << "Skip " << interval.buffer->ToShortString()
                << " because the buffer is larger than the heap size.";
        continue;
      }
    }

    auto colocated_intervals = GetSortedColocatedIntervals(interval);
    if (AreIntervalsReservedInAlternateMemory(colocated_intervals)) {
      VLOG(3) << "Interval " << interval.buffer->ToShortString()
              << " is reserved in the alternate memory.";
      ColorColocatedIntervalsToAlternate(colocated_intervals);
      continue;
    }

    if (!ConsumeFuel("memory_space_assignment", [&] {
          return absl::StrCat("Ran out of fuel at buffer: ",
                              colocated_intervals[0]->buffer->ToShortString());
        })) {
      continue;
    }

    if (options_.dump_fn != nullptr || VLOG_IS_ON(3)) {
      // Only fill buffer_info_str_ if needed.
      AppendBufferInfoDebugString(interval, &buffer_info_str_);
    }

    CreateAllocationValuesFromColocatedIntervals(colocated_intervals,
                                                 proposal.allocation_values);
    proposal.colocated_intervals.push_back(colocated_intervals);
  }
  // Order allocation values so that when read and write sequences are
  // different, we're sure the allocation value corresponding to the read
  // sequence is processed before the written sequence's allocation value.
  // We move the allocation values that have their defining instruction in the
  // sync conversion list to the end to be processed last.
  std::stable_partition(proposal.allocation_values.begin(),
                        proposal.allocation_values.end(),
                        [this](AllocationValue& allocation_value) {
                          return !IsInstructionPendingReplacements(
                              allocation_value.defining_instruction());
                        });

  NicePrintAllocationValues(proposal.allocation_values, /*log_level=*/3);
}

MsaAlgorithm::JointAllocationProposal MsaAlgorithm::GetJointProposal(
    MsaBufferInterval& interval) {
  JointAllocationProposal proposal;
  proposal.values = GenerateJointProcessedValues(interval.buffer);
  if (VLOG_IS_ON(3)) {
    VLOG(3) << "Joint-processed values for " << interval.buffer->ToShortString()
            << ": ";
    for (size_t idx = 0; idx < proposal.values.size(); ++idx) {
      const HloValue* hlo_value = proposal.values.at(idx);
      VLOG(3) << "  " << idx + 1 << "/" << proposal.values.size() << ") "
              << hlo_value->ToShortString();
    }
  }

  CreateAllocationValuesForJointProcessedValues(proposal);
  return proposal;
}

bool MsaAlgorithm::RepackAllocationsIncludeConvertedSyncMemOp() {
  for (RepackAllocationBlock& allocation_block : repack_allocation_blocks_) {
    if (allocation_block.allocation->is_copy_allocation()) {
      if (dynamic_cast<CopyAllocation*>(allocation_block.allocation)
              ->sync_mem_op()) {
        return true;
      }
    }
    if (allocation_block.allocation->is_sliced_copy_allocation()) {
      if (dynamic_cast<SlicedCopyAllocation*>(allocation_block.allocation)
              ->sync_mem_op()) {
        return true;
      }
    }
  }
  return false;
}

namespace {

// Fixes the AllocationSequence after post-allocation transformation:
//  1. Remove the allocations with to_be_removed instructions as the defining
//     positions.
//  2. Update the vector of uses for all allocations according to the
//     update_use_map.
// Note that to_be_removed instructions will later be removed from the module
// during SimplifyGraph() call in memory_space_assignment.cc
void FixAllocationSequenceAfterPostAllocationTransformation(
    AllocationSequence* allocations,
    const PostAllocationTransformationUpdate& transformation_info) {
  VLOG(3) << "Fixing AllocationSequence after post-allocation transformation";

  // (1)
  allocations->erase(
      std::remove_if(
          allocations->begin(), allocations->end(),
          [transformation_info](const std::unique_ptr<Allocation>& allocation) {
            return std::find(transformation_info.to_be_removed.begin(),
                             transformation_info.to_be_removed.end(),
                             allocation->defining_position().instruction) !=
                   transformation_info.to_be_removed.end();
          }),
      allocations->end());

  // (2)
  for (auto& allocation : *allocations) {
    std::vector<HloUse> uses_to_update;
    for (const HloUse& use : allocation->uses()) {
      for (const auto& [old_use, new_use] :
           transformation_info.update_use_map) {
        if (use == old_use) {
          uses_to_update.push_back(old_use);
          break;  // found the use, no need to keep searching update_use_map
        }
      }
    }

    // Perform update uses
    if (!uses_to_update.empty()) {
      for (const HloUse& old_use : uses_to_update) {
        const HloUse& new_use = transformation_info.update_use_map.at(old_use);
        allocation->RemoveUse(old_use);
        allocation->AddUse(new_use);
      }
    }
  }
}

// Verifies that the operands_in_alternate_memory_map is consistent with the
// allocations in the AllocationSequence.
bool VerifyOperandsInAlternateMemoryMap(
    const AllocationSequence* allocations,
    const absl::flat_hash_map<const HloInstruction*,
                              absl::flat_hash_set<std::pair<int, ShapeIndex>>>&
        operands_in_alternate_memory_map) {
  absl::flat_hash_map<const HloInstruction*,
                      absl::flat_hash_set<std::pair<int, ShapeIndex>>>
      reference_map;
  for (const std::unique_ptr<xla::memory_space_assignment::Allocation>&
           allocation : *allocations) {
    if (allocation->is_in_alternate_mem()) {
      for (const HloUse& use : allocation->uses()) {
        reference_map[use.instruction].insert(
            std::make_pair(use.operand_number, use.operand_index));
      }
    }
  }
  return reference_map == operands_in_alternate_memory_map;
}

// GetAsyncCopyElapsed with a default value.
float CopyResourceForShape(const Options& options, const Shape& shape) {
  return options.cost_analysis
             ? options.cost_analysis->GetAsyncCopyElapsed(shape)
             : 0.1;
}

absl::StatusOr<MemorySpace> GetMemorySpaceEnum(const int64_t memory_space,
                                               const Options& options) {
  if (memory_space == options.alternate_memory_space) {
    return MemorySpace::kAlternate;
  }
  if (memory_space == options.default_memory_space) {
    return MemorySpace::kDefault;
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Invalid memory space: ", memory_space));
}

}  // namespace

absl::Status MsaAlgorithm::ProcessColoredBuffers() {
  for (const auto& buffer_coloring : options_.buffer_colorings) {
    HloPosition position;
    HloUse use;
    bool use_colored = false;
    int64_t time_of_coloring;
    if (std::holds_alternative<HloUse>(
            buffer_coloring.buffer_position_or_use)) {
      use_colored = true;
      use = std::get<HloUse>(buffer_coloring.buffer_position_or_use);
      position =
          HloPosition{use.instruction->mutable_operand(use.operand_number),
                      use.operand_index};
      time_of_coloring =
          hlo_live_range_.instruction_schedule().at(use.instruction);
    } else {
      position = std::get<HloPosition>(buffer_coloring.buffer_position_or_use);
      time_of_coloring =
          hlo_live_range_.instruction_schedule().at(position.instruction);
    }
    const int64_t memory_space = buffer_coloring.memory_space;
    HloValue& value = alias_analysis_.dataflow_analysis().GetUniqueValueAt(
        position.instruction, position.index);
    TF_ASSIGN_OR_RETURN(const MemorySpace memory_space_enum,
                        GetMemorySpaceEnum(memory_space, options_));
    if (memory_space_enum == MemorySpace::kDefault) {
      default_memory_coloring_requirements_[value.defining_position()]
          .push_back(time_of_coloring);
      continue;
    }
    CHECK(memory_space_enum == MemorySpace::kAlternate);
    MsaBufferInterval interval =
        MsaBufferInterval{/*buffer=*/nullptr,
                          /*size=*/buffer_intervals_.at(&value).size,
                          /*start=*/time_of_coloring,
                          /*end=*/time_of_coloring,
                          /*colocations=*/{},
                          /*need_allocation=*/true};
    Chunk chunk_candidate = FindChunkCandidate(interval);
    if (chunk_candidate.chunk_end() > available_heap_size()) {
      if (use_colored) {
        return FailedPrecondition(
            "%s",
            absl::StrCat(
                "Too many buffers are colored in the alternate memory. Could "
                "not reserve alternate memory for colored operand of "
                "instruction ",
                use.instruction->name(), " at operand index ",
                use.operand_number, " shape index ",
                use.operand_index.ToString()));
      }
      return FailedPrecondition(
          "%s",
          absl::StrCat(
              "Too many buffers are colored in the alternate memory. Could not "
              "reserve alternate memory for colored output of instruction ",
              position.instruction->name(), " at shape index ",
              position.index.ToString()));
    }
    std::vector<std::unique_ptr<ReservedAllocation>>& reserved_allocations =
        reserved_allocations_for_alt_mem_colorings_[value.defining_position()];
    reserved_allocations.push_back(std::make_unique<ReservedAllocation>(
        value.defining_position(), chunk_candidate, time_of_coloring));
    CommitChunk(interval, chunk_candidate);
    // We need to add an allocation block to the repack_allocation_blocks_ so
    // repacking can account for the reserved memory.
    repack_allocation_blocks_.push_back(MakeRepackAllocationBlock(
        time_of_coloring, time_of_coloring, chunk_candidate.size,
        chunk_candidate.offset,
        static_cast<int64_t>(repack_allocation_blocks_.size()),
        reserved_allocations.back().get()));
    repack_allocation_blocks_.back().next_colocated =
        &(repack_allocation_blocks_.back());
  }
  return absl::OkStatus();
}

absl::StatusOr<HeapSimulator::Result<HloValue>> MsaAlgorithm::Finish() {
  // Note: Memory Space Assignment creates a HeapSimulator and passes an
  // MsaAlgorithm object to it. buffer_intervals_ is populated by calling the
  // Alloc(), Free() and ShareWith() methods on the MsaAlgorithm object in
  // HeapSimulator.
  if (options_.autotuning_config.has_value()) {
    CHECK_EQ((*options_.autotuning_config).size(), buffer_intervals_.size());
  }
  VLOG(1) << "Slicing is "
          << (options_.sliced_prefetch_options.max_slices() >= 2 ? "enabled"
                                                                 : "disabled");

  AllocateReservedScopedAllocations();
  std::vector<MsaBufferInterval> sorted_buffer_intervals =
      GetSortedBufferIntervals();
  memory_space_assignment::CustomizeSortedBufferInterval(
      options_.autotuning_config, sorted_buffer_intervals);

  // Calculate the memory pressure for the buffers that can be assigned in the
  // alternate memory.
  memory_pressure_ = 0;
  VLOG(5) << [&]() {
    std::string s("Sorted MsaBufferInterval order.");
    if (options_.buffer_interval_comparator) {
      absl::StrAppend(
          &s, " Pre-autotuning sort criteria: ",
          options_.buffer_interval_comparator->DescribeComparisonCriteria());
    }
    return s;
  }();
  for (auto& interval : sorted_buffer_intervals) {
    if (!interval.need_allocation ||
        !MemorySpaceAssignmentUtils::IsIntervalAllowedInAlternateMemory(
            interval, options_.alternate_memory_space) ||
        interval.size > available_heap_size()) {
      continue;
    }
    VLOG(5) << [&]() {
      std::string s("SortedBufferInterval.");
      if (options_.buffer_interval_comparator) {
        absl::StrAppend(
            &s, " Criteria: ",
            options_.buffer_interval_comparator->CriteriaToString(interval));
      }
      absl::StrAppend(&s, " Buffer: ", interval.buffer->ToShortString());
      return s;
    }();
    memory_pressure_ += interval.size;
  }
  VLOG(1) << "Memory pressure = " << memory_pressure_;

  CrossProgramPrefetches cross_program_prefetches =
      FindCrossProgramPrefetches(alias_analysis_, hlo_live_range_, options_);
  // Return error if cross program prefetch is disabled and user has requested
  // cross program prefetch.
  if (!options_.enable_cross_program_prefetch &&
      !cross_program_prefetches.prefetches.empty()) {
    return absl::FailedPreconditionError(
        "Cross program prefetch is disabled but user has requested cross "
        "program prefetch.");
  }

  // Return error if number of user requested cross program prefetches is
  // greater than the maximum number of cross program prefetches allowed.
  if (cross_program_prefetches.prefetches.size() >
      options().max_cross_program_prefetches) {
    return absl::FailedPreconditionError(
        "Number of user requested cross program prefetches is greater than the "
        "maximum number of cross program prefetches allowed.");
  }

  // Allocate user requested cross program prefetches first.
  for (auto& prefetch : cross_program_prefetches.prefetches) {
    HloModule* module = prefetch.buffer->instruction()->GetModule();
    AllocateCrossProgramPrefetchBuffer(module, prefetch);
  }
  if (options_.enable_cross_program_prefetch) {
    for (auto& prefetch_candidate : cross_program_prefetches.candidates) {
      HloModule* module = prefetch_candidate.buffer->instruction()->GetModule();
      if (0 <= options().max_cross_program_prefetches &&
          options().max_cross_program_prefetches <=
              module->CrossProgramPrefetches().size()) {
        break;
      }
      AllocateCrossProgramPrefetchBuffer(module, prefetch_candidate);
    }
  }

  VLOG(1) << "Assigning buffers to alternate memory. Max heap size = "
          << options_.max_size_in_bytes;

  AddInputAndOutputRequiredAssignments();
  TF_RETURN_IF_ERROR(ProcessColoredBuffers());

  if (VLOG_IS_ON(3) || options_.dump_fn != nullptr) {
    VLOG(3) << "Flattened instruction sequence:";
    const auto& instruction_sequence =
        hlo_live_range_.flattened_instruction_sequence().instructions();
    absl::StrAppend(&instruction_schedule_str_, "time,instruction_name\n");
    for (int i = 0; i < instruction_sequence.size(); ++i) {
      VLOG(3) << " " << i << ": " << instruction_sequence[i]->parent()->name()
              << " " << instruction_sequence[i]->name();
      absl::StrAppend(&instruction_schedule_str_, i, ",",
                      instruction_sequence[i]->name(), "\n");
    }
  }

  if (options_.memory_bound_loop_optimizer_options.enabled()) {
    IdentifyAndOptimizeMemoryBoundLoops();
  }

  for (const auto& interval : sorted_buffer_intervals) {
    auto colocated_intervals = GetSortedColocatedIntervals(interval);
    if (AreIntervalsReservedInAlternateMemory(colocated_intervals)) {
      // Increment the reserved part of alternate memory so that it is not
      // available for other buffers.
      reserved_in_bytes_ += options_.size_fn(*interval.buffer);
    }
  }
  VLOG(2) << "Total reserved bytes = " << reserved_in_bytes_;
  for (MsaBufferInterval& interval : sorted_buffer_intervals) {
    VLOG(3) << "Processing buffer: " << interval.buffer->ToString();
    if (finalized_values_.contains(interval.buffer)) {
      VLOG(3) << "Skip entrance interval" << interval.buffer->ToShortString()
              << " because it is already processed.";
      continue;
    }

    JointAllocationProposal proposal = GetJointProposal(interval);
    if (proposal.allocation_values.empty()) {
      VLOG(3) << "No allocation values for these joint-processed values.";
      continue;
    }
    // Retry allocating this value with larger limits if allocation fails.
    bool repacked = false;
    for (int retry_number = 0; retry_number < options_.max_retries;
         retry_number++) {
      for (auto& colocated_intervals : proposal.colocated_intervals) {
        AddRequiredAssignmentsForColocatedIntervals(colocated_intervals);
      }
      options_.prefetch_interval_picker->SetRetryNumber(retry_number);
      TF_ASSIGN_OR_RETURN(
          AllocationResult result,
          AllocateAllocationValues(absl::MakeSpan(proposal.allocation_values)));
      VLOG(2) << "Allocation result = " << ResultToString(result);
      VLOG(4)
          << "Non-finalized allocations after processing allocation values:";
      for (int allocation_value_idx = 0;
           allocation_value_idx < proposal.allocation_values.size();
           ++allocation_value_idx) {
        auto& allocation_value =
            proposal.allocation_values.at(allocation_value_idx);
        VLOG(4) << "  " << allocation_value.ToShortString();
        for (auto& allocation : *allocation_value.allocation_sequence()) {
          VLOG(4) << "    " << allocation->ToString();
        }
      }
      if (result_is(result, AllocationResult::kFailSyncDataMoveReplacement)) {
        if (!options_.enable_sync_copy_replacement &&
            !options_.enable_sync_slice_replacement) {
          return absl::FailedPreconditionError(
              "Allocation result is "
              "AllocationResult::kFailSyncCopyReplacement, but "
              "no sync replacement is enabled.");
        }

        UncommitPendingChunks(absl::MakeSpan(proposal.allocation_values));
        proposal = GetJointProposal(interval);
        if (proposal.allocation_values.empty()) {
          VLOG(3)
              << "No allocation values found in the updated joint-processed "
                 "values. Moving on to the next set of joint-processed values.";
          break;
        }
        --retry_number;

      } else if (result_requires_uncommit(result)) {
        UncommitPendingChunks(absl::MakeSpan(proposal.allocation_values));
        VLOG(2) << "Couldn't allocate. Retry number " << retry_number;
        if (retry_number > 0 && !sorted_async_conversion_candidates_.empty()) {
          failed_async_conversions_[sorted_async_conversion_candidates_.at(0)] =
              AsyncConversionResult::kFailedGaveUp;
          VLOG(2) << "Giving the allocation another chance by dropping one "
                     "async conversion candidate.";
          proposal = GetJointProposal(interval);
          --retry_number;
        }
      } else if ((result_is(result, AllocationResult::kFailOutOfMemory) ||
                  options_.repack_after_every_allocation) &&
                 num_repacks_ < options_.max_repacks && !repacked &&
                 !RepackAllocationsIncludeConvertedSyncMemOp()) {
        UncommitPendingChunks(absl::MakeSpan(proposal.allocation_values));
        ++num_repacks_;
        repacked = true;
        if (!options_.repacker) {
          return absl::FailedPreconditionError("Repacker not provided.");
        }

        std::vector<AllocationBlock*> repack_allocation_blocks;
        ExportAllocationsForRepacking(repack_allocation_blocks);
        VLOG(2) << "Repacking.";
        auto repack_status =
            options_.repacker->Repack(absl::MakeSpan(repack_allocation_blocks));
        CHECK_EQ(repack_status.status(), absl::OkStatus());
        VLOG(2) << "Repack complete. Modified = " << *repack_status;
        // For debug and testing purpose, also update allocations if
        // repack_after_every_allocation is on.
        if (*repack_status || options_.repack_after_every_allocation) {
          ImportRepackedAllocations();
          --retry_number;
        }
        if (*repack_status) {
          ++num_repacks_successful_;
        }
      } else {
        // Check if any of the allocation sites are inefficient. If so, get rid
        // of the pending allocation, require all of the inefficient sites in
        // the default memory, and perform allocation again.
        std::vector<HloPositionOrUse> inefficient_sites = {};
        if (sorted_async_conversion_candidates_.empty()) {
          inefficient_sites =
              GetInefficientAllocationSites(proposal.allocation_values);
        }
        if (!inefficient_sites.empty()) {
          UncommitPendingChunks(absl::MakeSpan(proposal.allocation_values));
          for (const HloPositionOrUse& site : inefficient_sites) {
            // To avoid a livelock situation, we commit the required assignments
            // right away. Otherwise, reallocation can find alternate memory
            // allocations at other sites, which can also be inefficient.
            std::visit(
                [this](const auto& site) {
                  VLOG(3) << "Inefficient site: " << site.ToString();
                  AddRequiredAssignment(site, MemorySpace::kDefault,
                                        /*offset=*/nullptr,
                                        /*add_to_pending=*/false);
                },
                site);
          }
          --retry_number;
          continue;
        }

        FinalizeAllocations(absl::MakeSpan(proposal.allocation_values));
        break;
      }
    }
    // Keep track of the processed values to prevent double-processing in future
    // joint-processed intervals.
    for (auto& value : proposal.values) {
      finalized_values_.insert(value);
    }
  }

  CHECK(VerifyOperandsInAlternateMemoryMap(allocations_,
                                           operands_in_alternate_memory_map_))
      << "operands_in_alternate_memory_map_ is not consistent with the "
         "finalizied allocations.";

  if (options_.repack_after_every_allocation) {
    if (!options_.repacker) {
      return absl::FailedPreconditionError("Repacker cannot be null.");
    }

    if (RepackAllocationsIncludeConvertedSyncMemOp()) {
      return absl::InternalError(
          "Repacking is not supported yet when there are converted sync mem "
          "ops.");
    }

    std::vector<AllocationBlock*> repack_allocation_blocks;
    ExportAllocationsForRepacking(repack_allocation_blocks);
    VLOG(2) << "Final Repacking.";
    auto repack_status =
        options_.repacker->Repack(absl::MakeSpan(repack_allocation_blocks));
    if (!repack_status.ok()) {
      return repack_status.status();
    }
    VLOG(2) << "Final Repack complete. Modified = " << *repack_status;
  }

  if (options_.dump_fn != nullptr || VLOG_IS_ON(3)) {
    for (auto& allocation : *allocations_) {
      // Only fill allocation_info_str_ if needed.
      AppendAllocationInfoDebugString(*allocation, allocation_info_str_);
    }
  }

  VLOG(1) << "Repack summary: " << num_repacks_successful_
          << " succeeded out of " << num_repacks_;

  VLOG(3) << "Debug buffer info: ";
  XLA_VLOG_LINES(3, buffer_info_str_);
  VLOG(3) << "Debug allocation info: ";
  XLA_VLOG_LINES(3, allocation_info_str_);
  DumpDebugStringsIfEnabled();

  if (VLOG_IS_ON(3)) {
    VLOG(3) << "Sync copy replacement summary: ";
    VLOG(3) << "\tnumber of successful async conversion: "
            << successful_async_conversion_set_.size();
    VLOG(3) << "\tnumber of failed async conversion: "
            << failed_async_conversions_.size();
    for (const HloInstruction* inst : successful_async_conversion_set_) {
      VLOG(3) << "Successful copy replacement: " << inst->ToString();
    }
    for (auto& failure : failed_async_conversions_) {
      VLOG(3) << "Failed copy replacement: " << failure.first->ToString()
              << ", reason: " << int(failure.second);
    }
  }

  // Run post allocation transformation and fix the allocation sequence if
  // needed.
  if (options_.post_allocation_transformation_fn) {
    auto has_in_place_user = [](HloInstruction* instr) {
      for (HloInstruction* user : instr->users()) {
        auto alias_pairs =
            HloDataflowAnalysis::GetInPlaceInputOutputPairs(user);
        for (const auto& [operand_index, output_index] : alias_pairs) {
          if (user->operand(operand_index.operand_number) == instr) {
            return true;
          }
        }
      }
      return false;
    };
    VLOG(3) << "Running post allocation transformation on module";
    for (HloComputation* comp : alias_analysis_.dataflow_analysis()
                                    .module()
                                    .MakeNonfusionComputations()) {
      for (HloInstruction* instr : comp->MakeInstructionPostOrder()) {
        // If the operand is in alternate memory, we don't run the
        // post-allocation transformation.
        auto operand_it = operands_in_alternate_memory_map_.find(instr);
        if (operand_it != operands_in_alternate_memory_map_.end()) {
          continue;
        }

        // If the instruction is a successful async conversion, we don't run the
        // post-allocation transformation.
        if (successful_async_conversion_set_.contains(instr)) {
          continue;
        }

        // If any of the operands of the instruction has an in-place user, we
        // don't run the post-allocation transformation.
        bool in_place_user = false;
        for (HloInstruction* operand : instr->operands()) {
          // We don't care about users of constants.
          if (operand->opcode() == HloOpcode::kConstant) {
            continue;
          }
          if (has_in_place_user(operand)) {
            in_place_user = true;
            break;
          }
        }
        if (in_place_user) {
          continue;
        }

        VLOG(3) << "Running post allocation transformation on: \n"
                << instr->ToString();
        TF_ASSIGN_OR_RETURN(PostAllocationTransformationUpdate changes,
                            options_.post_allocation_transformation_fn(instr));
        if (!changes.to_be_removed.empty()) {
          VLOG(3) << "Post allocation transformation info: \n"
                  << changes.ToString();
          FixAllocationSequenceAfterPostAllocationTransformation(allocations_,
                                                                 changes);
        }
      }
    }
  }

  if (options_.expanded_scoped_alternate_memory_mode ==
      ExpandedScopedAlternateMemoryMode::ENABLED) {
    ExtendScopedAlternateMemoryAllocations();
  }

  HeapSimulator::Result<HloValue> result;
  result.heap_size = result_.heap_size;
  result.heap_results.emplace_back(std::move(result_));
  return result;
}

namespace {

// Convert a tuple HloUse to its equivalent HloPosition.
HloPosition TupleUseToPosition(const HloUse& use) {
  CHECK_EQ(use.instruction->opcode(), HloOpcode::kTuple);
  ShapeIndex index = use.operand_index;
  index.push_front(use.operand_number);
  return {use.instruction, index};
}

// Returns the memory space of the defining position of an Allocation object.
MemorySpace GetDefiningPositionMemorySpace(const Allocation& allocation) {
  if (!allocation.is_copy_like_allocation()) {
    return allocation.memory_space();
  }
  if (allocation.memory_space() == MemorySpace::kDefault) {
    return MemorySpace::kAlternate;
  }
  return MemorySpace::kDefault;
}

}  // namespace

std::vector<std::vector<const Allocation*>>
MsaAlgorithm::GetLinkedAllocationsInAlternateMemory(
    absl::Span<const AllocationValue> allocation_values) const {
  std::vector<std::vector<const Allocation*>> linked_allocations;
  // A map from position to index into linked_allocations.
  absl::flat_hash_map<HloPosition, int> link_id_map;
  // Iterate over the allocation values. Find Allocation objects across the
  // allocation values that are part of the same linked allocation group. We
  // define a linked allocation group as Allocation objects that have aliased
  // positions or uses. An example would be an Allocation object that has a
  // dynamic-update-slice use and another Allocation object that has the same
  // dynamic-update-slice as its defining position.
  for (const AllocationValue& allocation_value : allocation_values) {
    absl::flat_hash_map<HloUse, std::vector<HloPosition>> aliases;
    for (const AllocationValue::Use& allocation_value_use :
         allocation_value.uses()) {
      if (!allocation_value_use.aliases.empty()) {
        aliases[allocation_value_use.hlo_use] = allocation_value_use.aliases;
      }
    }
    for (const auto& allocation : *allocation_value.allocation_sequence()) {
      MemorySpace position_memory_space =
          GetDefiningPositionMemorySpace(*allocation);
      if (allocation->memory_space() == MemorySpace::kDefault &&
          position_memory_space == MemorySpace::kDefault) {
        // This is just a regular allocation in the default memory, skip.
        continue;
      }
      int link_id = -1;
      // For every position and use in the alternate memory space, check if
      // there is already a linked allocation group, and if so, use that link
      // index.
      if (position_memory_space == MemorySpace::kAlternate) {
        auto link_id_map_it = link_id_map.find(allocation->defining_position());
        if (link_id_map_it != link_id_map.end()) {
          link_id = link_id_map_it->second;
        }
      }
      if (allocation->memory_space() == MemorySpace::kAlternate) {
        for (const HloUse& use : allocation->uses()) {
          if (use.instruction->opcode() == HloOpcode::kTuple) {
            auto link_id_map_it = link_id_map.find(TupleUseToPosition(use));
            if (link_id_map_it != link_id_map.end()) {
              if (link_id != -1 && link_id != link_id_map_it->second) {
                // We found multiple link indices for the given allocation. We
                // merge the two linked allocation groups in that case.
                int old_link_id = link_id_map_it->second;
                if (old_link_id < link_id) {
                  std::swap(link_id, old_link_id);
                }
                absl::c_copy(linked_allocations[old_link_id],
                             std::back_inserter(linked_allocations[link_id]));
                linked_allocations[old_link_id].clear();
                for (auto it = link_id_map.begin(); it != link_id_map.end();
                     ++it) {
                  if (it->second == old_link_id) {
                    it->second = link_id;
                  }
                }
              }
              link_id = link_id_map_it->second;
            }
          }
        }
      }
      if (link_id == -1) {
        // Create a new linked allocation group if we couldn't find one.
        link_id = linked_allocations.size();
        linked_allocations.push_back({allocation.get()});
      } else {
        linked_allocations[link_id].push_back(allocation.get());
      }
      // Propagate the link index to all of the aliases of uses in the alternate
      // memory.
      if (allocation->memory_space() == MemorySpace::kAlternate) {
        for (const HloUse& use : allocation->uses()) {
          auto alias_it = aliases.find(use);
          if (alias_it != aliases.end()) {
            for (const HloPosition& aliased_position : alias_it->second) {
              link_id_map[aliased_position] = link_id;
            }
          }
        }
      }
    }
  }

  linked_allocations.erase(
      std::remove_if(
          linked_allocations.begin(), linked_allocations.end(),
          [](const auto& allocations) { return allocations.empty(); }),
      linked_allocations.end());

  if (VLOG_IS_ON(3)) {
    for (int i = 0; i < linked_allocations.size(); ++i) {
      VLOG(3) << "  Link id = " << i;
      for (const Allocation* allocation : linked_allocations[i]) {
        VLOG(3) << "    " << allocation->ToString();
      }
    }
  }
  return linked_allocations;
}

std::vector<HloPositionOrUse> MsaAlgorithm::GetInefficientAllocationSites(
    absl::Span<const AllocationValue> allocation_values) const {
  // The logic below is used mostly for testing, allowing a test case to inject
  // some custom logic for this method.
  if (options_.get_inefficient_allocation_sites_fn) {
    std::vector<HloPosition> defining_positions;
    defining_positions.reserve(allocation_values.size());
    for (const AllocationValue& value : allocation_values) {
      defining_positions.push_back(value.defining_position());
    }
    return options_.get_inefficient_allocation_sites_fn(
        absl::MakeSpan(defining_positions));
  }

  if (!options_.cost_analysis ||
      options_.inefficient_use_to_copy_ratio == 0.0) {
    return {};
  }

  int64_t size = 0;
  if (!allocation_values.empty()) {
    size = allocation_values.at(0).size();
  }

  if (VLOG_IS_ON(3)) {
    VLOG(3) << "Inefficient allocation calculations:";
    for (const AllocationValue& allocation_value : allocation_values) {
      for (const auto& allocation : *allocation_value.allocation_sequence()) {
        VLOG(3) << "  Allocation: " << allocation->ToString();
        if (!allocation->is_copy_like_allocation()) {
          const HloPosition& defining_position =
              allocation->defining_position();
          int64_t accessed = options_.cost_analysis->OutputBytesAccessed(
              *defining_position.instruction, defining_position.index);
          VLOG(3) << "    pos: " << defining_position.ToString()
                  << ", accessed: " << accessed << " / " << size;
        }
        for (const HloUse& use : allocation->uses()) {
          int64_t accessed = options_.cost_analysis->OperandBytesAccessed(
              *use.instruction, use.operand_number, use.operand_index);
          VLOG(3) << "    use: " << use.ToString() << ", accessed: " << accessed
                  << " / " << size;
        }
      }
    }
  }

  std::vector<std::vector<const Allocation*>> linked_allocations =
      GetLinkedAllocationsInAlternateMemory(allocation_values);
  std::vector<HloPositionOrUse> inefficient_sites;
  for (const std::vector<const Allocation*>& allocation_group :
       linked_allocations) {
    // For all of allocation in the linked allocation group, calculate the total
    // use bytes in alternate memory and async copy bytes. If the ratio between
    // the two is below inefficient_use_to_copy_ratio, add all of the
    // participating allocation sites into inefficient_sites.
    VLOG(3) << "  AllocationGroup:";
    int64_t copy_bytes = 0;
    int64_t use_bytes = 0;
    for (const Allocation* allocation : allocation_group) {
      VLOG(3) << "    Allocation: " << allocation->ToString();
      MemorySpace position_memory_space =
          GetDefiningPositionMemorySpace(*allocation);
      if (allocation->is_copy_like_allocation()) {
        copy_bytes += size;
      }
      if (position_memory_space == MemorySpace::kAlternate) {
        use_bytes += options_.cost_analysis->OutputBytesAccessed(
            *allocation->defining_position().instruction,
            allocation->defining_position().index);
      }
      if (allocation->memory_space() == MemorySpace::kAlternate) {
        for (const HloUse& use : allocation->uses()) {
          use_bytes += options_.cost_analysis->OperandBytesAccessed(
              *use.instruction, use.operand_number, use.operand_index);
        }
      }
    }
    VLOG(3) << "      use bytes: " << use_bytes
            << ", copy bytes: " << copy_bytes;
    if (options_.inefficient_use_to_copy_ratio * copy_bytes > use_bytes) {
      for (const Allocation* allocation : allocation_group) {
        MemorySpace position_memory_space =
            GetDefiningPositionMemorySpace(*allocation);
        if (position_memory_space == MemorySpace::kAlternate) {
          if (!allocation->is_copy_like_allocation()) {
            inefficient_sites.push_back(allocation->defining_position());
          }
        }
        if (allocation->memory_space() == MemorySpace::kAlternate) {
          for (const HloUse& use : allocation->uses()) {
            inefficient_sites.push_back(use);
          }
        }
      }
    }
  }
  return inefficient_sites;
}

void MsaAlgorithm::AddRequiredAssignmentsForColocatedIntervals(
    absl::Span<const MsaBufferInterval* const> colocated_intervals) {
  // TODO(berkin): For now, place the phi values due to conditionals in
  // default memory.
  for (const MsaBufferInterval* colocated_interval : colocated_intervals) {
    const HloValue* value = colocated_interval->buffer;
    for (const auto& position : value->positions()) {
      if (position.instruction->opcode() == HloOpcode::kConditional) {
        VLOG(3) << "Adding required assignment for condition output: "
                << value->ToShortString();
        AddRequiredAssignment(position.instruction, position.index,
                              MemorySpace::kDefault);
        for (const HloComputation* called_computation :
             position.instruction->called_computations()) {
          AddRequiredAssignment(called_computation->root_instruction(),
                                position.index, MemorySpace::kDefault);
        }
      }
    }
  }
}

void MsaAlgorithm::CreateAllocationValuesFromColocatedIntervals(
    absl::Span<const MsaBufferInterval* const> colocated_intervals,
    std::vector<AllocationValue>& allocation_values) {
  std::vector<AllocationValue> new_allocation_values;
  // Create AllocationValues for all the colocated intervals.
  for (const auto& colocated_interval : colocated_intervals) {
    CreateAllocationValues(*colocated_interval, new_allocation_values);
  }
  // Go through the AllocationValues and delete the ones that have the identical
  // defining instruction and use instructions. This is useful for async
  // operations that can read and write to the same buffer, e.g., in-place
  // asynchronous collective permute. The AllocationValues that corresponds to
  // collective-permute-start{0} (the input) and collective-permute-start{1}
  // (the output) refer to the same buffer by definition (since they are created
  // from colocated intervals). If we don't delete one of these buffers, then
  // when we try to allocate the AllocationValue, we would think they overlap.
  auto create_instruction_vector = [](const AllocationValue& allocation_value) {
    std::vector<const HloInstruction*> instruction_vector;
    instruction_vector.push_back(allocation_value.defining_instruction());
    for (const AllocationValue::Use& use : allocation_value.uses()) {
      instruction_vector.push_back(use.hlo_use.instruction);
    }
    return instruction_vector;
  };
  for (int i = 0; i < new_allocation_values.size() - 1; ++i) {
    for (int j = i + 1; j < new_allocation_values.size(); ++j) {
      const AllocationValue& allocation_value_1 = new_allocation_values[i];
      const AllocationValue& allocation_value_2 = new_allocation_values[j];
      if (create_instruction_vector(allocation_value_1) ==
          create_instruction_vector(allocation_value_2)) {
        VLOG(3) << "Allocation values " << allocation_value_1.ToShortString()
                << " and " << allocation_value_2.ToShortString()
                << " are equivalent, deleting the second one.";
        new_allocation_values.erase(new_allocation_values.begin() + j);
        --j;
      }
    }
  }

  FindAliases(&new_allocation_values);
  absl::c_move(new_allocation_values, std::back_inserter(allocation_values));
}

void MsaAlgorithm::MaybeSplitAllocationValues(
    absl::Span<AllocationValue> allocation_values) {
  if (options_.determine_split_dimension_fn == nullptr ||
      options_.shape_size_fn == nullptr ||
      options_.init_split_tree_fn == nullptr) {
    return;
  }

  std::vector<std::optional<SplitConfig>> results;

  for (AllocationValue& allocation_value : allocation_values) {
    std::optional<SplitConfig> result = options_.determine_split_dimension_fn(
        *allocation_value.value(), &instruction_to_split_dims_);
    results.push_back(std::move(result));
  }
  for (int i = 0; i < results.size(); ++i) {
    if (results[i] != results[0]) {
      VLOG(3) << "Skipping splitting joint allocation values with different "
                 "split choices: "
              << allocation_values[0].ToShortString() << " -> "
              << (results[0].has_value() ? results[0]->ToString() : "nullopt")
              << " vs " << allocation_values[i].ToShortString() << " -> "
              << (results[i].has_value() ? results[i]->ToString() : "nullopt");
      return;
    }
  }

  for (int i = 0; i < allocation_values.size(); ++i) {
    auto& allocation_value = allocation_values[i];
    HloInstruction* defining_instruction =
        allocation_value.value()->defining_instruction();
    auto& result = results[i];
    if (!instruction_to_split_dims_.contains(defining_instruction)) {
      instruction_to_split_dims_[allocation_value.value()
                                     ->defining_instruction()] =
          options_.init_split_tree_fn(defining_instruction, nullptr);
    }
    int64_t* mutable_element =
        instruction_to_split_dims_[defining_instruction].mutable_element(
            allocation_value.value()->defining_position().index);
    if (!result.has_value()) {
      *mutable_element = options_.replicated_split_dimension;
      VLOG(4) << "Splitting allocation value: "
              << allocation_value.ToShortString() << ": kReplicated.";
      continue;
    }
    // TODO(b/382592216): Delay this assignment until after the AllocationValue
    //  actually gets an alternate memory allocation.
    *mutable_element = result->dimension();
    Shape new_shape = allocation_value.value()->shape();
    if (new_shape.has_layout() &&
        new_shape.layout().split_configs().size() == 0) {
      new_shape.mutable_layout()->add_split_configs(result.value());
    }
    allocation_value.set_split_shape(new_shape);
    int64_t shape_size = options_.shape_size_fn(new_shape);

    VLOG(4) << "Splitting allocation value: "
            << allocation_value.ToShortString() << ": " << result->ToString();
    allocation_value.set_size(shape_size);
  }
}

bool MsaAlgorithm::RequiresNoCopyAlternateMemAllocation(
    AllocationValue& allocation_value) const {
  return allocation_value.value()->shape().has_layout() &&
         allocation_value.value()->shape().layout().memory_space() ==
             options_.alternate_memory_space;
}

void MsaAlgorithm::AssignDefaultMemIfNotAllowedInAlternateMem(
    AllocationValue& allocation_value, int64_t definition_time) {
  if (!options_.is_position_allowed_in_alternate_mem_fn(
          allocation_value.defining_position())) {
    if (RequiresNoCopyAlternateMemAllocation(allocation_value)) {
      LOG(WARNING) << "The value " << allocation_value.value()->ToShortString()
                   << " is pre-colored for alternate memory but the position "
                   << allocation_value.defining_position().ToString()
                   << " is not allowed in the alternate memory. Respecting the "
                      "color "
                      "but this may break things later in compilation.";
    } else {
      AddRequiredAssignment(allocation_value.value(),
                            allocation_value.defining_instruction(),
                            MemorySpace::kDefault, definition_time);
    }
  }
}

std::vector<AllocationSegmentContext>
MsaAlgorithm::GenerateAllocationSegmentContexts(
    absl::Span<AllocationValue>& allocation_values,
    absl::flat_hash_map<const HloInstruction*, std::vector<size_t>>&
        value_indices_by_sync_inst,
    int allocation_value_idx) const {
  AllocationValue& allocation_value =
      allocation_values.at(allocation_value_idx);
  std::vector<AllocationSegmentContext> uses_work_list;
  for (int primary_use_idx = 0;
       primary_use_idx < allocation_value.uses().size(); ++primary_use_idx) {
    AllocationValue::Use& primary_use =
        allocation_value.uses().at(primary_use_idx);
    if (!IsInstructionPendingReplacements(primary_use.hlo_use.instruction)) {
      uses_work_list.push_back({&allocation_value.uses(), primary_use_idx,
                                allocation_value_idx, false});
    } else {
      uses_work_list.push_back({&allocation_value.uses(), primary_use_idx,
                                allocation_value_idx, true});
      for (auto sync_destination_idx :
           value_indices_by_sync_inst.at(primary_use.hlo_use.instruction)) {
        AllocationValue& sync_destination =
            allocation_values.at(sync_destination_idx);
        if (sync_destination.defining_instruction() ==
            primary_use.hlo_use.instruction) {
          VLOG(3) << "Adding secondary uses related to allocation value "
                  << sync_destination.ToShortString()
                  << " to uses worklist, because the allocation value is "
                     "defined at the copy use instruction output.";
          for (int secondary_use_id = 0;
               secondary_use_id < sync_destination.uses().size();
               ++secondary_use_id) {
            // This is an important line
            sync_destination.uses().at(secondary_use_id).sync_mem_op_operand =
                primary_use.hlo_use.instruction;
            int allocation_value_to_update_idx = sync_destination_idx;
            uses_work_list.push_back({&sync_destination.uses(),
                                      secondary_use_id,
                                      allocation_value_to_update_idx, false});
          }
        } else {
          VLOG(3) << "Skipping secondary uses related to allocation value "
                  << sync_destination.ToShortString()
                  << ", because the allocation value is not defined at the "
                     "copy use instruction "
                     "output.";
        }
      }
    }
  }
  // Sort uses according to their use time
  std::sort(uses_work_list.begin(), uses_work_list.end(),
            [](const auto& a, const auto& b) {
              return a.uses->at(a.use_idx).time < b.uses->at(b.use_idx).time;
            });
  VLOG(3) << "Uses work list:";
  for (int i = 0; i < uses_work_list.size(); i++) {
    auto [uses, use_idx, allocation_value_to_update_idx,
          only_extend_existing_allocation] = uses_work_list.at(i);
    VLOG(3) << "  " << i + 1 << "/" << uses_work_list.size() << ") "
            << uses->at(use_idx).hlo_use.ToString();
  }
  if (uses_work_list.empty()) {
    VLOG(3) << "  --Empty--";
  }
  return uses_work_list;
}

absl::StatusOr<AllocationResult> MsaAlgorithm::AllocateAllocationValues(
    absl::Span<AllocationValue> allocation_values) {
  const auto& instruction_schedule = hlo_live_range_.instruction_schedule();
  absl::flat_hash_map<const HloInstruction*, std::vector<size_t>>
      value_indices_by_sync_inst;
  for (size_t idx = 0; idx < allocation_values.size(); ++idx) {
    const HloInstruction* inst =
        allocation_values.at(idx).defining_instruction();
    if (IsInstructionPendingReplacements(inst)) {
      value_indices_by_sync_inst[inst].push_back(idx);
    }
  }

  // Extract all use times
  std::vector<int64_t> all_use_times;
  for (const AllocationValue& allocation_value : allocation_values) {
    for (const auto& use : allocation_value.uses()) {
      all_use_times.push_back(use.time);
    }
  }
  absl::c_sort(all_use_times);
  for (int i = 0; i < all_use_times.size(); ++i) {
    VLOG(3) << "all_use_times[" << i << "] = " << all_use_times[i];
  }

  MaybeSplitAllocationValues(allocation_values);

  // Data structure to contain the preferred offset for a given computation.
  // We ensure that the same offset will be allocated outside the while loop
  // as well as inside the while loop.
  absl::flat_hash_map<const HloComputation*, AliasedOffset*>
      preferred_offset_for_computation;
  absl::flat_hash_map<const AllocationValue*, AliasedOffset*>
      preferred_offset_for_allocation_value;
  absl::flat_hash_map<const AllocationValue*, int64_t>
      definition_time_for_allocation_value;
  AllocationResult result = AllocationResult::kSuccess;
  for (int alloc_value_idx = 0; alloc_value_idx < allocation_values.size();
       ++alloc_value_idx) {
    auto& allocation_value = allocation_values.at(alloc_value_idx);
    VLOG(3) << alloc_value_idx + 1 << "/" << allocation_values.size()
            << ") Allocating allocation value: "
            << allocation_value.ToShortString();

    if (IsInstructionPendingReplacements(
            allocation_value.defining_instruction())) {
      VLOG(3) << "Skip allocating allocation value "
              << allocation_value.ToShortString();
      continue;
    }

    VLOG(4) << "require_no_copy_alternate_mem_allocation = "
            << RequiresNoCopyAlternateMemAllocation(allocation_value);
    if (RequiresNoCopyAlternateMemAllocation(allocation_value) &&
        allocation_value.size() > available_heap_size()) {
      VLOG(3) << "Skip " << allocation_value.value()->ToShortString()
              << " because the buffer is larger than the heap size.";
      continue;
    }

    const AllocationValue::Use* previous_use = nullptr;
    auto uses_work_list = GenerateAllocationSegmentContexts(
        allocation_values, value_indices_by_sync_inst, alloc_value_idx);

    // Iterate over the uses.
    for (auto& entry : uses_work_list) {
      const AllocationValue::Use& use = entry.uses->at(entry.use_idx);
      AllocationValue& allocation_value_to_update =
          allocation_values.at(entry.allocation_value_to_update_idx);
      std::string extension_only_hint_str =
          entry.only_extend_existing_allocation ? " (extension only): " : ": ";
      VLOG(3) << "Working on use" << extension_only_hint_str
              << use.hlo_use.ToString()
              << ", allocation value: " << allocation_value.ToShortString()
              << ", updates allocation value: "
              << allocation_value_to_update.ToShortString();

      if (!definition_time_for_allocation_value.contains(
              &allocation_value_to_update)) {
        definition_time_for_allocation_value[&allocation_value_to_update] =
            hlo_live_range_.instruction_schedule().at(
                allocation_value_to_update.defining_instruction());
        AssignDefaultMemIfNotAllowedInAlternateMem(
            allocation_value_to_update, definition_time_for_allocation_value.at(
                                            &allocation_value_to_update));
      }

      if (!preferred_offset_for_allocation_value.contains(
              &allocation_value_to_update)) {
        auto preferred_offset_it = preferred_offset_for_computation.find(
            allocation_value_to_update.computation());
        if (preferred_offset_it != preferred_offset_for_computation.end()) {
          preferred_offset_for_allocation_value[&allocation_value_to_update] =
              preferred_offset_it->second;
        } else {
          preferred_offset_for_allocation_value[&allocation_value_to_update] =
              nullptr;
        }
      }
      preferred_offset_for_allocation_value[&allocation_value_to_update] =
          UpdatePreferredOffsetForUse(use,
                                      preferred_offset_for_allocation_value.at(
                                          &allocation_value_to_update));
      AllocationRequest request = CreateAllocationRequest(
          allocation_value, allocation_value_to_update, use, previous_use,
          preferred_offset_for_allocation_value.at(&allocation_value_to_update),
          definition_time_for_allocation_value.at(&allocation_value_to_update),
          RequiresNoCopyAlternateMemAllocation(allocation_value_to_update),
          all_use_times, entry.only_extend_existing_allocation,
          allocation_values.subspan(0, alloc_value_idx));
      if (options_.allocation_request_modifier_testing_fn) {
        options_.allocation_request_modifier_testing_fn(request);
      }
      // Bitcasts don't define buffers and don't directly consume buffers.
      // Skip allocating buffers for bitcast uses (unless they are the root
      // instruction). The uses that feed from bitcasts will be handled
      // specially.
      if (use.hlo_use.instruction->opcode() != HloOpcode::kBitcast ||
          use.hlo_use.instruction ==
              use.hlo_use.instruction->parent()->root_instruction()) {
        UpdateRequestWithAlternateMemoryColoringRequirements(request);
        UpdateRequestWithDefaultMemoryColoringRequirements(request);
        AllocationResult allocate_segment_result = AllocateSegment(request);
        VLOG(2) << "AllocateSegment result: "
                << ResultToString(allocate_segment_result);
        result_mark(allocate_segment_result, result);
        if (options_.allocation_result_modifier_testing_fn) {
          options_.allocation_result_modifier_testing_fn(request, result);
        }
        if (request.require_copy_allocation) {
          auto allocation_sequence =
              allocation_value_to_update.mutable_allocation_sequence();
          auto it = std::find_if(
              allocation_sequence->begin(), allocation_sequence->end(),
              [&](const std::unique_ptr<
                  xla::memory_space_assignment::Allocation>& allocation_ptr) {
                if (allocation_ptr->is_copy_allocation()) {
                  auto copy_allocation =
                      dynamic_cast<const CopyAllocation*>(allocation_ptr.get());
                  return copy_allocation &&
                         (copy_allocation->copy_done_schedule_before() <=
                          request.required_copy_allocation_latest_time) &&
                         (copy_allocation->sync_mem_op() ==
                          request.required_copy_allocation_for) &&
                         (!request.required_copy_for_slice ||
                          (request.required_copy_for_slice &&
                           !copy_allocation->cross_program_prefetch_index()
                                .has_value()));
                }
                if (allocation_ptr->is_sliced_copy_allocation()) {
                  auto sliced_copy_allocation =
                      dynamic_cast<const SlicedCopyAllocation*>(
                          allocation_ptr.get());
                  return sliced_copy_allocation &&
                         (sliced_copy_allocation->earliest_available_time() <=
                          request.required_copy_allocation_latest_time) &&
                         (sliced_copy_allocation->sync_mem_op() ==
                          request.required_copy_allocation_for) &&
                         !request.required_copy_for_slice;
                }
                return false;
              });

          if (result_requires_uncommit(result) ||
              it == allocation_sequence->end()) {
            VLOG(3) << "No async copy allocation found by the end of "
                       "segment allocation. "
                       "Sync copy replacement has failed. Fall back to the "
                       "normal mode.";
            failed_async_conversions_[request.required_copy_allocation_for] =
                AsyncConversionResult::kFailedSatisfyingConstraints;
            result_mark(AllocationResult::kFailSyncDataMoveReplacement, result);
            result_mark(AllocationResult::kFailRequiresUncommit, result);
          } else {
            bool has_correct_use = false;
            for (auto& alloc_use : (*it)->uses()) {
              if (alloc_use == request.use->hlo_use) {
                has_correct_use = true;
                break;
              }
            }
            if (!has_correct_use) {
              VLOG(3) << "No async copy allocation found by the end of "
                         "segment allocation with the correct use. "
                         "Sync copy replacement has failed. Fall back to the "
                         "normal mode.";
              failed_async_conversions_[request.required_copy_allocation_for] =
                  AsyncConversionResult::kFailedPrecondition;
              result_mark(AllocationResult::kFailSyncDataMoveReplacement,
                          result);
              result_mark(AllocationResult::kFailRequiresUncommit, result);
            } else {
              not_finalized_async_conversions_.push_back(
                  request.required_copy_allocation_for);
              VLOG(3) << "Replacing "
                      << request.required_copy_allocation_for->ToShortString()
                      << " with " << (*it)->ToString();
            }
          }
        }
        if (request.require_no_copy_alternate_mem_allocation &&
            result != AllocationResult::kSuccess) {
          absl::Status failed_precondition = FailedPrecondition(
              "The value defined at %s requires allocation in the alternate "
              "memory, which could not be satisfied. This typically happens "
              "because more pinned buffers are live than the alternate memory "
              "capacity.",
              allocation_value.defining_instruction()->ToString());
          LOG(ERROR) << failed_precondition;
          return failed_precondition;
        }
        if (result_requires_uncommit(result)) {
          // If the allocation finding failed (e.g., due to running out of
          // asynchronous copies), then fall back to allocating the buffer
          // entirely in the default memory.
          return result;
        }

        // If there are multiple uses, they can try using the memory
        // allocation already at the alternate memory.
        definition_time_for_allocation_value[&allocation_value_to_update] =
            instruction_schedule.at(use.hlo_use.instruction);
        previous_use = &use;
      }
      if (entry.only_extend_existing_allocation) {
        continue;
      }
      const auto use_time = request.end_time;
      UpdateAllocationRequirementForUseAliases(allocation_value_to_update, use,
                                               use_time);
      MaybeCreateMirroredParentAllocationForWhileUse(
          allocation_value_to_update, use, use_time, allocation_values,
          preferred_offset_for_computation);
    }
  }

  if (!VerifyAllConversionsAreSuccessful()) {
    result_mark(AllocationResult::kFailSyncDataMoveReplacement, result);
    result_mark(AllocationResult::kFailRequiresUncommit, result);
  }

  return result;
}

bool MsaAlgorithm::VerifyAllConversionsAreSuccessful() {
  for (const HloInstruction* instruction :
       sorted_async_conversion_candidates_) {
    if (absl::c_find(not_finalized_async_conversions_, instruction) ==
        not_finalized_async_conversions_.end()) {
      if (!failed_async_conversions_.contains(instruction)) {
        failed_async_conversions_[instruction] =
            AsyncConversionResult::kFailedNotProcessed;
        VLOG(3) << "Async conversion failed for "
                << instruction->ToShortString()
                << " because its operand or user was not processed.";
      }
      return false;
    }
  }
  return true;
}

AliasedOffset* MsaAlgorithm::UpdatePreferredOffsetForUse(
    const AllocationValue::Use& use, AliasedOffset* preferred_offset) const {
  // Assign the required assignment offset as a preferred offset.
  std::optional<RequiredMemoryAssignment> required_assignment =
      AliasedRequiredAssignmentForUse(use);
  if (required_assignment &&
      required_assignment->memory_space == MemorySpace::kAlternate) {
    if (preferred_offset) {
      CHECK_EQ(preferred_offset, required_assignment->offset);
    } else {
      preferred_offset = required_assignment->offset;
      VLOG(3) << "Setting preferred offset due to required assignment for use: "
              << preferred_offset->offset;
    }
  }
  return preferred_offset;
}

AllocationRequest MsaAlgorithm::CreateAllocationRequest(
    AllocationValue& allocation_value,
    AllocationValue& allocation_value_to_update,
    const AllocationValue::Use& use, const AllocationValue::Use* previous_use,
    AliasedOffset* preferred_offset, int64_t definition_time,
    bool require_no_copy_alternate_mem_allocation,
    const std::vector<int64_t>& all_use_times,
    bool only_extend_existing_allocation,
    absl::Span<AllocationValue> processed_allocation_values) {
  const HloUse& hlo_use = use.hlo_use;
  const auto& instruction_schedule = hlo_live_range_.instruction_schedule();
  bool require_copy_allocation = false;
  int64_t required_copy_allocation_latest_time = 0;
  HloInstruction* required_copy_allocation_for = nullptr;
  bool required_copy_for_slice = false;
  std::optional<int64_t> earliest_prefetch_time = std::nullopt;
  if (use.sync_mem_op_operand &&
      IsInstructionPendingReplacements(use.sync_mem_op_operand)) {
    required_copy_allocation_for = use.sync_mem_op_operand;
    require_copy_allocation = true;
    required_copy_for_slice =
        (IsAsyncConversionSliceCandidate(use.sync_mem_op_operand) ==
         AsyncConversionResult::kSuccess);
    // The async copy allocation can be delayed until the earliest time at which
    // the value is used in a position or the earliest use time of the updated
    // allocation value. We find the minimum of these two times.
    int64_t min_time =
        GetCorrectedUseTime(allocation_value.defining_instruction());
    int64_t earliest_position_time = std::numeric_limits<int64_t>::max();
    for (auto& position : allocation_value.value()->positions()) {
      auto position_time = GetCorrectedUseTime(position.instruction);
      if (position_time > min_time) {
        earliest_position_time =
            std::min(earliest_position_time, position_time);
      }
    }
    int64_t earliest_use_time = std::numeric_limits<int64_t>::max();
    for (auto& secondary_use : allocation_value_to_update.uses()) {
      if (!IsTrivialInstruction(secondary_use.hlo_use.instruction) ||
          secondary_use.hlo_use.instruction ==
              use.hlo_use.instruction->parent()->root_instruction()) {
        earliest_use_time = std::min(
            earliest_use_time, GetCorrectedUseTime(secondary_use.hlo_use));
      }
    }
    required_copy_allocation_latest_time =
        std::min(earliest_use_time, earliest_position_time);
    // We need to make sure that the copy allocation is scheduled before the
    // controlled successor of the sync mem op.
    for (const HloInstruction* control_successor :
         required_copy_allocation_for->control_successors()) {
      int64_t successor_time = instruction_schedule.at(control_successor);
      if (successor_time < required_copy_allocation_latest_time) {
        VLOG(3) << "Updating the required replacement async mem op allocation "
                   "latest time from "
                << required_copy_allocation_latest_time << " to "
                << successor_time << ", because of control successor "
                << control_successor->ToString();
        required_copy_allocation_latest_time = successor_time;
      }
    }

    // Make sure that the earliest prefetch time is after the schedule time of
    // the latest operand of the sync mem op.
    for (const HloInstruction* operand :
         required_copy_allocation_for->operands()) {
      int64_t operand_time = instruction_schedule.at(operand);
      earliest_prefetch_time =
          std::max(earliest_prefetch_time.value_or(-1), operand_time);
    }
  }

  int64_t use_time = instruction_schedule.at(hlo_use.instruction);
  bool allow_no_copy_alternate_mem_allocation = true;
  bool allow_prefetch = true;
  bool prefer_no_copy_alternate_mem_allocation = false;
  // TODO(b/318886791):  Rename boundary variables (here and other places)
  // like `latest_prefetch_time` and `earliest_prefetch_time` indicate
  // whether they are exclusive or inclusive boundaries.
  int64_t latest_prefetch_time = use_time;

  // Control flow  calls include kWhile, kCall, and kConditional opcodes.
  bool is_sequential_call =
      (GetInstructionCallContext(hlo_use.instruction->opcode()) ==
       CallContext::kControlFlow);
  if (is_sequential_call) {
    for (const HloComputation* called_computation :
         hlo_use.instruction->called_computations()) {
      const HloLiveRange::TimeBound& computation_span =
          hlo_live_range_.computation_span_times().at(called_computation);
      latest_prefetch_time =
          std::min(computation_span.start - 1, latest_prefetch_time);
    }
    use_time = GetCorrectedUseTime(hlo_use);
  }

  // Add a required assignment in default memory if the use not allowed in
  // alternate memory.
  if (!IsUseAllowedInAlternateMemory(allocation_value_to_update, hlo_use)) {
    if (require_no_copy_alternate_mem_allocation) {
      LOG(WARNING) << "The value "
                   << allocation_value_to_update.value()->ToShortString()
                   << " is pre-colored for alternate memory but the use "
                   << hlo_use.ToString()
                   << " is not allowed in the alternate memory. Respecting the "
                      "color but this may break things later in compilation.";
    } else {
      AddRequiredAssignment(allocation_value_to_update.value(),
                            hlo_use.instruction, MemorySpace::kDefault,
                            use_time);
    }
  } else if (previous_use != nullptr) {
    // We allow buffers in alternate memory that are passed into
    // conditionals to give up their alternate memory allocation inside the
    // called computation. This means that if a conditional operator has an
    // alternate memory allocation, subsequent uses cannot use the same
    // alternate memory allocation in order not to clobber data. So we force
    // default memory allocation for these subsequent uses.
    if (previous_use->hlo_use.instruction->opcode() ==
            HloOpcode::kConditional &&
        previous_use->hlo_use.instruction != hlo_use.instruction) {
      allow_no_copy_alternate_mem_allocation = false;
      earliest_prefetch_time =
          instruction_schedule.at(previous_use->hlo_use.instruction);
      VLOG(3) << "Previous use (" << previous_use->hlo_use.ToString()
              << ") of use (" << hlo_use.ToString()
              << ") is a conditional, so this use will need to evict. "
              << "Earliest prefetch time = " << *earliest_prefetch_time;
    }
  }

  AllocationRequest request;
  // Bitcasts don't define buffers and don't directly consume buffers. Skip
  // allocating buffers for bitcast uses (unless they are the root
  // instruction). The uses that feed from bitcasts will be handled
  // specially.
  if (hlo_use.instruction->opcode() != HloOpcode::kBitcast ||
      hlo_use.instruction ==
          hlo_use.instruction->parent()->root_instruction()) {
    std::optional<int64_t> preferred_prefetch_time = std::nullopt;
    auto loop_optimized_allocation_it =
        loop_optimized_allocations_map_.find(use.hlo_use);
    if (loop_optimized_allocation_it != loop_optimized_allocations_map_.end()) {
      const LoopOptimizedAllocationInfo& loop_optimized_allocation_info =
          loop_optimized_allocation_it->second;
      const Allocation* allocation =
          loop_optimized_allocation_info.loop_optimized_allocation;
      VLOG(3) << "Found optimized allocation for " << use.hlo_use.ToString()
              << " (loop idx: " << loop_optimized_allocation_info.use_index
              << "): " << allocation->ToString();
      if (require_no_copy_alternate_mem_allocation) {
        if (allocation->is_copy_allocation() ||
            allocation->memory_space() == MemorySpace::kDefault) {
          LOG(WARNING) << "Optimized allocation could not be applied "
                          "because the tensor is pre-colored, allocation: "
                       << allocation->ToString();
        }
      } else if (allocation->is_copy_allocation()) {
        allow_no_copy_alternate_mem_allocation = true;
        const CopyAllocation* copy_allocation =
            static_cast<const CopyAllocation*>(allocation);
        int64_t effective_copy_start_time =
            copy_allocation->copy_start_schedule_after();
        if (copy_allocation->copy_start_schedule_after() ==
                loop_optimized_allocation_info.loop_size - 1 &&
            copy_allocation->copy_done_schedule_before() == 0) {
          effective_copy_start_time = -loop_optimized_allocation_info.loop_size;
        } else if (copy_allocation->copy_start_schedule_after() + 1 >=
                   copy_allocation->copy_done_schedule_before()) {
          effective_copy_start_time -= loop_optimized_allocation_info.loop_size;
        }
        preferred_prefetch_time =
            hlo_live_range_.instruction_schedule().at(hlo_use.instruction) -
            loop_optimized_allocation_info.use_index +
            effective_copy_start_time;
        VLOG(3) << "Prefer prefetch at " << *preferred_prefetch_time
                << " (effective: " << effective_copy_start_time << ")";
      } else if (allocation->memory_space() == MemorySpace::kDefault) {
        allow_prefetch = false;
        allow_no_copy_alternate_mem_allocation = false;
        VLOG(3) << "Disallowing alternate memory allocation.";
      } else {
        CHECK(allocation->memory_space() == MemorySpace::kAlternate);
        prefer_no_copy_alternate_mem_allocation = true;
        VLOG(3) << "Prefer no-copy alternate memory allocation.";
      }
    }

    // TODO(mehrdadk): Remove this code once we have a better way to find
    // repeated instructions.
    if (/* DISABLES CODE */ (false)) {
      const std::vector<const HloInstruction*>* repeated_insts =
          GetRepeatedInstructionList(hlo_use.instruction);
      if (repeated_insts) {
        for (int i = 0; i < repeated_insts->size(); ++i) {
          const HloInstruction* repeated = repeated_insts->at(i);
          VLOG(4) << "Repeated instruction for use: " << repeated->name() << " "
                  << hlo_live_range_.instruction_schedule().at(repeated);
          if (repeated == hlo_use.instruction && i > 0) {
            const HloInstruction* prev_repeated = repeated_insts->at(i - 1);
            if (prev_repeated->parent() == hlo_use.instruction->parent()) {
              preferred_prefetch_time =
                  hlo_live_range_.instruction_schedule().at(prev_repeated) + 1;
              VLOG(3) << "Found a previous repeated (" << prev_repeated->name()
                      << ") at " << (*preferred_prefetch_time - 1)
                      << ". Setting preferred prefetch time = "
                      << *preferred_prefetch_time;
            }
          }
        }
      }
    }

    int64_t live_range_start_time = (earliest_prefetch_time.has_value()
                                         ? earliest_prefetch_time.value()
                                         : std::min(definition_time, use_time));
    auto overridden_preferred_prefetch_time =
        MemorySpaceAssignmentUtils::GetOverriddenPreferredPrefetchTime(
            options_.preferred_prefetch_overrides, allocation_value.size(),
            hlo_use, instruction_schedule, live_range_start_time,
            latest_prefetch_time);
    TF_CHECK_OK(overridden_preferred_prefetch_time.status());
    if (overridden_preferred_prefetch_time.value().has_value()) {
      LOG(INFO) << "Overriding preferred prefetch for "
                << hlo_use.instruction->name() << " operand number "
                << hlo_use.operand_number << " operand index "
                << hlo_use.operand_index.ToString() << " size "
                << allocation_value.size() << " live range ("
                << live_range_start_time << ", " << latest_prefetch_time
                << ") from "
                << (preferred_prefetch_time.has_value()
                        ? preferred_prefetch_time.value()
                        : -1)
                << " to " << overridden_preferred_prefetch_time.value().value();
      preferred_prefetch_time = overridden_preferred_prefetch_time.value();
    }

    // Rarely, (e.g., when conditional true and false parameters are the
    // same), definition time can be the time of the conditional and use
    // time is the parameter use, which is less.
    request.inclusive_start_time = std::min(definition_time, use_time);
    request.latest_prefetch_time = latest_prefetch_time;
    request.size = allocation_value_to_update.size();
    request.prefer_no_copy_alternate_mem_allocation =
        prefer_no_copy_alternate_mem_allocation;
    request.allow_no_copy_alternate_mem_allocation =
        allow_no_copy_alternate_mem_allocation;
    request.allow_prefetch = allow_prefetch;
    request.require_no_copy_alternate_mem_allocation =
        require_no_copy_alternate_mem_allocation;
    request.earliest_prefetch_time = earliest_prefetch_time;
    request.preferred_prefetch_time = preferred_prefetch_time;
    request.preferred_offset = preferred_offset;
    request.use = &use;
    request.allocation_value = &allocation_value;
    request.all_use_times = all_use_times;
    request.require_copy_allocation = require_copy_allocation;
    request.required_copy_allocation_latest_time =
        required_copy_allocation_latest_time;
    request.required_copy_allocation_for = required_copy_allocation_for;
    request.required_copy_for_slice = required_copy_for_slice;
    request.allocation_value_to_update = &allocation_value_to_update;
  }

  request.end_time = use_time;
  request.only_extend_existing_allocation = only_extend_existing_allocation;
  request.processed_allocation_values = processed_allocation_values;
  return request;
}

void MsaAlgorithm::UpdateAllocationRequirementForUseAliases(
    const AllocationValue& allocation_value, const AllocationValue::Use& use,
    int64_t use_time) {
  Allocation* aliased_allocation =
      GetLiveAllocationAt(*allocation_value.allocation_sequence(), use_time);
  VLOG(4) << "Aliased allocation at time " << use_time << ": "
          << (aliased_allocation ? aliased_allocation->ToString()
                                 : "couldn't find the aliased allocation");

  for (const HloPosition& aliased_position : use.aliases) {
    AddAliasedRequiredAssignment(aliased_position.instruction,
                                 aliased_position.index, aliased_allocation);
  }
}

void MsaAlgorithm::MaybeCreateMirroredParentAllocationForWhileUse(
    const AllocationValue& allocation_value, const AllocationValue::Use& use,
    int64_t use_time, absl::Span<AllocationValue> allocation_values,
    absl::flat_hash_map<const HloComputation*, AliasedOffset*>&
        preferred_offset_for_computation) {
  const HloUse& hlo_use = use.hlo_use;

  if (hlo_use.instruction->opcode() != HloOpcode::kWhile) return;

  Allocation* aliased_allocation =
      GetLiveAllocationAt(*allocation_value.allocation_sequence(), use_time);
  if (aliased_allocation->memory_space() != MemorySpace::kAlternate) return;

  const auto& instruction_schedule = hlo_live_range_.instruction_schedule();
  if (options_.enable_while_redundant_eviction_elimination &&
      absl::c_find_if(allocation_value.value()->positions(),
                      [&hlo_use](const HloPosition& position) {
                        return position.instruction == hlo_use.instruction &&
                               position.index == hlo_use.operand_index;
                      }) != allocation_value.value()->positions().end()) {
    auto allocation_sequence = allocation_value.allocation_sequence();
    auto prev_allocation_in_default_mem_it = std::find_if(
        allocation_sequence->rbegin(), allocation_sequence->rend(),
        [&](const auto& allocation) {
          return allocation->memory_space() == MemorySpace::kDefault &&
                 allocation->defining_position() ==
                     allocation_value.defining_position();
        });
    if (prev_allocation_in_default_mem_it != allocation_sequence->rend()) {
      VLOG(3) << "Found a prev allocation in default mem for while use: "
              << (*prev_allocation_in_default_mem_it)->ToString();
      auto body_allocation_value_it =
          absl::c_find_if(allocation_values, [&](const AllocationValue& value) {
            return value.computation() == hlo_use.instruction->while_body() &&
                   value.defining_instruction()->opcode() ==
                       HloOpcode::kParameter;
          });
      CHECK_NE(body_allocation_value_it, allocation_values.end());
      VLOG(3) << "Body allocation value: "
              << body_allocation_value_it->ToShortString();
      int64_t body_parameter_time = instruction_schedule.at(
          body_allocation_value_it->defining_instruction());
      body_allocation_value_it->mutable_allocation_sequence()->push_back(
          std::make_unique<ParentAllocation>(
              **prev_allocation_in_default_mem_it, hlo_use.instruction,
              body_allocation_value_it->defining_position(),
              body_parameter_time));
      VLOG(3) << "Created: "
              << body_allocation_value_it->allocation_sequence()
                     ->back()
                     ->ToString();

      auto after_while_allocation_value_it =
          absl::c_find_if(allocation_values, [&](const AllocationValue& value) {
            return value.defining_instruction() == hlo_use.instruction;
          });
      CHECK_NE(after_while_allocation_value_it, allocation_values.end());
      VLOG(3) << "After while allocation value: "
              << after_while_allocation_value_it->ToShortString();
      int64_t while_time = instruction_schedule.at(hlo_use.instruction);
      after_while_allocation_value_it->mutable_allocation_sequence()->push_back(
          std::make_unique<MirroredAllocation>(
              **prev_allocation_in_default_mem_it, while_time));
      VLOG(3) << "Created: "
              << after_while_allocation_value_it->allocation_sequence()
                     ->back()
                     ->ToString();
    }
  }
  // Special case for while loops since the root offset must agree with
  // other offsets: remember the preferred offset for the while loop body.
  preferred_offset_for_computation[hlo_use.instruction->while_body()] =
      GetAliasedOffset(*aliased_allocation);
}

bool operator<(const AsynchronousCopy& a, const AsynchronousCopy& b) {
  return a.AsTuple() < b.AsTuple();
}

bool operator==(const AsynchronousCopy& a, const AsynchronousCopy& b) {
  return a.AsTuple() == b.AsTuple();
}

bool operator!=(const AsynchronousCopy& a, const AsynchronousCopy& b) {
  return a.AsTuple() != b.AsTuple();
}

void AsynchronousCopyOrdering::AddCopy(const AsynchronousCopy& copy) {
  auto it = ranges_.find({copy.exclusive_start_time, copy.end_time});
  if (it != ranges_.end()) {
    CHECK_EQ(it->first.exclusive_start_time, copy.exclusive_start_time);
    CHECK(it->second.insert(copy).second);
  } else {
    ranges_[{copy.exclusive_start_time, copy.end_time}] = {copy};
  }
}

void AsynchronousCopyOrdering::RemoveCopy(const AsynchronousCopy& copy) {
  auto copy_it = ranges_.find({copy.exclusive_start_time, copy.end_time});
  CHECK(copy_it != ranges_.end());
  CHECK_EQ(copy_it->first.exclusive_start_time, copy.exclusive_start_time);
  CHECK_EQ(copy_it->second.erase(copy), 1);
  if (copy_it->second.empty()) {
    ranges_.erase(copy_it);
  }
}

bool AsynchronousCopyOrdering::ViolatesOrdering(int64_t exclusive_start_time,
                                                int64_t end_time) const {
  // We allow identical start and end times. It is enough to check for just the
  // start time in case we find a match in ranges_ because the found value will
  // either be identical to {start_time, estimated_end_time} (and this doesn't
  // violate) or its start_time will be smaller and estimated_end_time will be
  // larger (this violates).
  auto copy_it = ranges_.find({exclusive_start_time, end_time});
  if (copy_it != ranges_.end() &&
      copy_it->first.exclusive_start_time != exclusive_start_time) {
    VLOG(4) << "Violates ordering: (" << exclusive_start_time << ", "
            << end_time << ") and (" << copy_it->first.exclusive_start_time
            << ", " << copy_it->first.end_time << ")";
    return true;
  }
  return false;
}

bool AsynchronousCopyResource::ConsumeResource(
    int64_t exclusive_start_time, int64_t end_time, int64_t resource,
    std::vector<std::pair<int64_t, int64_t>>* delay_changes,
    int64_t resource_to_free) {
  // Cache the pointers to the arrays to avoid the overhead of `operator[]`
  // size checks in hardened libc++.
  //
  // NOTE: Do not modify the vectors `initial_resources_` or `delay_` in this
  // function, otherwise the pointers will become dangling.
  int64_t* initial_resources_scaled_ptr = initial_resources_scaled_.data();
  int64_t* delay_ptr = delay_.data();

  std::list<AsynchronousCopy>::iterator current_copy = async_copies_.end();
  // In order to propagate the resource to the next scheduled copy, we iterate
  // over the copies in start time order until we either find enough free
  // resource (and return true), or find out that we don't have enough free
  // resource (and return false).
  while (true) {
    // resource is modified below. We save its initial value for logging below.
    const int64_t amount_requested = resource;

    VLOG(3) << "Consume resource: start time_exclusive = "
            << exclusive_start_time << ", end time = " << end_time
            << ", resource = " << resource << ", delay = "
            << delay_[ExclusiveToInclusiveStartTime(exclusive_start_time)]
            << ", free = " << resource_to_free;
    VLOG(5) << "Available resources: "
            << VectorToString(
                   GetCurrentResources(), /*include_indices=*/true,
                   ExclusiveToInclusiveStartTime(exclusive_start_time),
                   end_time);

    // Nothing to do if we're not adding or removing any resources.
    if (resource == 0 && resource_to_free == 0) {
      return true;
    }

    // For the async copy we're adding, check the delay_ array to see how much
    // this copy would have to be delayed because of an earlier copy that wasn't
    // finished when this copy starts.
    if (current_copy == async_copies_.end()) {
      resource +=
          delay_ptr[ExclusiveToInclusiveStartTime(exclusive_start_time)];
    }

    // Find the copy that is right after this one. If there are leftover
    // resources by the time the next copy starts, the next copy will be pushed
    // further later in time.
    std::list<AsynchronousCopy>::iterator next_copy = async_copies_.end();
    if (current_copy != async_copies_.end()) {
      next_copy = std::next(current_copy);
    } else {
      auto async_copy_time_it =
          async_copy_time_map_.upper_bound(exclusive_start_time);
      if (async_copy_time_it != async_copy_time_map_.end()) {
        next_copy = async_copy_time_it->second;
      }
    }

    // Check if this copy will push the next copy later in time (or if removing
    // the resource, check if the removal of this copy move the next copy
    // earlier in time).
    std::optional<int64_t> delay_for_next_copy = std::nullopt;
    int64_t resource_freed = 0;
    for (int64_t time = ExclusiveToInclusiveStartTime(exclusive_start_time);
         time < end_time && resource != 0; ++time) {
      int64_t initial_resource_scaled = initial_resources_scaled_ptr[time];
      // Iterate over the logical times that this copy spans. Note that the
      // start and end time ranges are exclusive.
      int64_t used_resource = std::min(resource, initial_resource_scaled);
      if (next_copy != async_copies_.end() &&
          next_copy->exclusive_start_time ==
              InclusiveToExclusiveStartTime(time)) {
        // This is the time where the next copy begins. If the resource is
        // non-zero at this point, the copy didn't finish by the time the next
        // copy started, so the next copy would need to be pushed later in time.
        delay_for_next_copy = resource;
        resource_to_free -= resource_freed;
      }
      if (!delay_for_next_copy.has_value()) {
        // Update the delay_ vector and resource_freed variable with the amount
        // that was freed when removing the copy.
        int64_t old_delay = delay_ptr[time];
        int64_t old_resource =
            std::max<int64_t>(0, initial_resource_scaled - old_delay);
        int64_t new_delay = std::max<int64_t>(0, resource - resource_to_free);
        int64_t new_resource =
            std::max<int64_t>(0, initial_resource_scaled - new_delay);
        resource_freed += std::max<int64_t>(0, new_resource - old_resource);
        delay_ptr[time] = new_delay;
        if (delay_changes) {
          delay_changes->emplace_back(time, old_delay);
        }
      }
      // Update the resource with the used amount in this logical time.
      resource -= used_resource;
    }

    // If resource isn't satisfied by the end, we didn't have enough
    // resources.
    if (resource > 0) {
      VLOG(3) << "Doesn't have enough resource; requested resource = "
              << amount_requested << "; leftover resources = " << resource;
      return false;
    }

    if (!delay_for_next_copy.has_value()) {
      return true;
    }
    // If this copy overlapped with another one, we run for another iteration
    // with the next copy  with the amount of resource that needs to be added or
    // removed.
    exclusive_start_time = next_copy->exclusive_start_time;
    end_time = next_copy->end_time;
    resource =
        *delay_for_next_copy + GetScaledIntegerResource(next_copy->resource);
    current_copy = next_copy;
  }
}

void AsynchronousCopyResource::AddCopy(const AsynchronousCopy& copy) {
  CHECK(ConsumeResource(copy.exclusive_start_time, copy.end_time,
                        GetScaledIntegerResource(copy.resource)));

  // Find the iterator for the copy that would be right after this copy and put
  // this copy right before it in async_copies_.
  auto async_copy_time_it =
      async_copy_time_map_.upper_bound(copy.exclusive_start_time);
  auto insertion_it = (async_copy_time_it == async_copy_time_map_.end())
                          ? async_copies_.end()
                          : async_copy_time_it->second;
  auto inserted_it = async_copies_.insert(insertion_it, copy);
  // If this copy is the first copy we have seen with the start time, add the
  // inserted iterator into async_copy_time_map_ for fast lookups. Note that
  // async_copy_time_map_ always points to the very first copy with the same
  // start index. If there are multiple asynchronous copies that have the same
  // start time, the memory space assignment algorithm schedules them in the
  // same order that AddCopy was called.
  if (async_copy_time_map_.find(copy.exclusive_start_time) ==
      async_copy_time_map_.end()) {
    async_copy_time_map_[copy.exclusive_start_time] = inserted_it;
  }
}

void AsynchronousCopyResource::RemoveCopy(const AsynchronousCopy& copy) {
  // The ConsumeResource method can only correctly remove the last copy that
  // starts at a given start time. So if the copy that is requested to be
  // removed is not the last copy for this start time, we need to temporarily
  // remove later copies that has the same start time and then add them back one
  // by one. To do this, we first find the iterator that points to the earliest
  // copy after this start time. We then decrement this iterator and temporarily
  // remove the copies until we find the copy we actually want to remove. After
  // we remove the copy that we actually want to remove, we add back the
  // temporarily removed copies one by one in the same order.
  auto async_copy_time_it =
      async_copy_time_map_.upper_bound(copy.exclusive_start_time);
  auto copy_it = (async_copy_time_it == async_copy_time_map_.end())
                     ? async_copies_.end()
                     : async_copy_time_it->second;
  CHECK(copy_it != async_copies_.begin());
  --copy_it;

  std::list<AsynchronousCopy> copies_to_add_back;
  auto prev_copy_it = copy_it;
  for (; *copy_it != copy; copy_it = prev_copy_it) {
    CHECK(copy_it != async_copies_.begin());
    CHECK_EQ(copy_it->exclusive_start_time, copy.exclusive_start_time);
    copies_to_add_back.push_front(*copy_it);
    VLOG(4) << "RemoveCopy found a copy to temporarily remove and add back: "
            << copy_it->exclusive_start_time << " " << copy_it->end_time << " "
            << copy_it->resource;
    prev_copy_it = std::prev(copy_it);
    RemoveCopy(copy_it);
  }
  CHECK(*copy_it == copy);
  RemoveCopy(copy_it);

  for (const AsynchronousCopy& copy_to_add_back : copies_to_add_back) {
    AddCopy(copy_to_add_back);
  }
}

void AsynchronousCopyResource::RemoveCopy(
    std::list<AsynchronousCopy>::iterator& copy_it) {
  // This method works only for the latest copy for the given start time.
  CHECK(std::next(copy_it) == async_copies_.end() ||
        std::next(copy_it)->exclusive_start_time >
            copy_it->exclusive_start_time);
  CHECK(ConsumeResource(
      copy_it->exclusive_start_time, copy_it->end_time,
      /*resource=*/0,
      /*delay_changes=*/nullptr,
      /*resource_to_free=*/GetScaledIntegerResource(copy_it->resource)));
  // If the copy to be removed is the value pointed by async_copy_time_map_, we
  // make the next copy with the same start time to be pointed by
  // async_copy_time_map_. If there are no such copies, we remove the key for
  // this copy start time.
  int64_t exclusive_start_time = copy_it->exclusive_start_time;
  auto async_copy_time_it = async_copy_time_map_.find(exclusive_start_time);
  if (copy_it == async_copy_time_it->second) {
    if (std::next(copy_it) != async_copies_.end() &&
        std::next(copy_it)->exclusive_start_time == exclusive_start_time) {
      async_copy_time_it->second = std::next(copy_it);
    } else {
      async_copy_time_map_.erase(async_copy_time_it);
    }
  }
  async_copies_.erase(copy_it);
}

bool AsynchronousCopyResource::HasEnoughResource(int64_t exclusive_start_time,
                                                 int64_t end_time,
                                                 float resource) {
  std::vector<std::pair<int64_t, int64_t>> delay_changes;
  delay_changes.reserve(delay_.size());
  bool result =
      ConsumeResource(exclusive_start_time, end_time,
                      GetScaledIntegerResource(resource), &delay_changes);
  // Apply the delay changes in reverse order. This ensures that the original
  // value of each delay is restored.
  if (!delay_changes.empty()) {
    for (int64_t i = delay_changes.size() - 1; i >= 0; --i) {
      const auto& [time, delay] = delay_changes[i];
      delay_[time] = delay;
    }
  }
  return result;
}

bool AsynchronousCopyResource::HasEnoughResourceMultiCheck(
    const std::vector<ResourceSpec>& specs) {
  delay_changes_.resize(0);
  delay_changes_.reserve(delay_.size());
  bool result = absl::c_all_of(specs, [&](const ResourceSpec& spec) {
    return ConsumeResource(spec.exclusive_start_time, spec.end_time,
                           GetScaledIntegerResource(spec.resource),
                           &delay_changes_);
  });
  // Apply the delay changes in reverse order. This ensures that the original
  // value of each delay is restored.
  if (!delay_changes_.empty()) {
    for (int64_t i = delay_changes_.size() - 1; i >= 0; --i) {
      const auto& [time, delay] = delay_changes_[i];
      delay_[time] = delay;
    }
  }
  return result;
}

namespace {

// A convenience struct for use in the implementation of
// AsynchronousCopyResource::Dump().
struct CopyResourceDumpData {
  float initial_resource;
  float delay;
  float available;
  std::vector<int> overlapping_copies;
};

}  // namespace

std::string AsynchronousCopyResource::Dump(
    int64_t start_time, int64_t end_time,
    MemorySpace memory_space_filter) const {
  std::vector<float> available = GetCurrentResources();
  std::vector<CopyResourceDumpData> time_dump_data;
  for (int i = start_time; i < end_time; ++i) {
    time_dump_data.push_back({
        initial_resources_[i],
        GetDescaledFloatResource(delay_[i]),
        available[i],
        /*overlapping_copies=*/{},
    });
  }

  std::vector<std::string> lines;
  lines.push_back(absl::StrCat("AsynchronousCopyResource::Dump(start_time: ",
                               start_time, ", end_time: ", end_time, ")"));
  for (const AsynchronousCopy& copy : async_copies_) {
    if (copy.destination != memory_space_filter) {
      continue;
    }
    int64_t overlap_start = std::max(start_time, copy.exclusive_start_time);
    int64_t overlap_end = std::min(end_time, copy.end_time);
    if (overlap_start < overlap_end) {
      lines.push_back(absl::StrCat(
          "copy(id: ", copy.id,
          ", exclusive_start: ", copy.exclusive_start_time,
          ", end: ", copy.end_time, ", resource: ", copy.resource, ")"));
    }
    for (int i = overlap_start; i < overlap_end; ++i) {
      time_dump_data[i - start_time].overlapping_copies.push_back(copy.id);
    }
  }

  std::vector<size_t> col_sizes;
  std::vector<std::vector<std::string>> rows;
  rows.push_back({"time", "initial", "delay", "avail", "overlapping copies"});
  for (absl::string_view col : rows.front()) {
    col_sizes.push_back(col.size());
  }
  for (int i = 0; i < time_dump_data.size(); ++i) {
    rows.push_back({absl::StrCat(i + start_time),
                    absl::StrCat(time_dump_data[i].initial_resource),
                    absl::StrCat(time_dump_data[i].delay),
                    absl::StrCat(time_dump_data[i].available),
                    absl::StrJoin(time_dump_data[i].overlapping_copies, ",")});
    for (int j = 0; j < rows.back().size(); ++j) {
      col_sizes[j] = std::max(col_sizes[j], rows.back()[j].size());
    }
  }
  for (const std::vector<std::string>& row : rows) {
    std::string line;
    std::string sep;
    for (int i = 0; i < col_sizes.size(); ++i) {
      absl::StrAppend(&line, sep, row[i]);
      sep = std::string(col_sizes[i] + 2 - row[i].size(), ' ');
    }
    lines.push_back(line);
  }

  return absl::StrJoin(lines, "\n");
}

AliasedOffset* MsaAlgorithm::GetAliasedOffset(const Allocation& allocation) {
  auto aliased_offset_it = aliased_offset_map_.find(&allocation);
  CHECK(aliased_offset_it != aliased_offset_map_.end());
  return aliased_offset_it->second;
}

void MsaAlgorithm::CreateOrAddToAliasedOffset(const Allocation& allocation,
                                              AliasedOffset* aliased_offset) {
  CHECK(allocation.memory_space() == MemorySpace::kAlternate);
  CHECK(!aliased_offset_map_.contains(&allocation));
  if (!aliased_offset) {
    aliased_offsets_.push_back({allocation.chunk().offset});
    aliased_offset = &aliased_offsets_.back();
  }
  CHECK_EQ(allocation.chunk().offset, aliased_offset->offset);
  CHECK(aliased_offset->allocations.insert(&allocation).second);
  aliased_offset_map_[&allocation] = aliased_offset;
}

/*static*/ Allocation* MsaAlgorithm::GetLiveAllocationAt(
    const AllocationSequence& allocations, int64_t time) {
  for (auto allocation_it = allocations.rbegin();
       allocation_it != allocations.rend(); ++allocation_it) {
    // The use case of GetLiveAllocationAt is to find the allocation that
    // corresponds to the full buffer. Window prefetched allocations allocates
    // only partial buffers, so we want to skip them.
    if ((*allocation_it)->start_time() <= time &&
        (*allocation_it)->end_time() >= time &&
        !(*allocation_it)->is_window_prefetched_allocation()) {
      return allocation_it->get();
    }
  }
  return nullptr;
}

namespace {

void SetDefaultMemorySpace(const HloValue* value, const Options& options) {
  for (auto& position : value->positions()) {
    Shape* shape = ShapeUtil::GetMutableSubshape(
        position.instruction->mutable_shape(), position.index);
    if (!shape->has_layout() ||
        shape->layout().memory_space() != options.alternate_memory_space) {
      continue;
    }
    shape->mutable_layout()->set_memory_space(options.default_memory_space);
  }
  HloModule* module = value->defining_instruction()->GetModule();
  module->mutable_config().SetComputationLayoutIfExists(
      module->entry_computation()->ComputeProgramShape());
}

}  // namespace

void MsaAlgorithm::AllocateCrossProgramPrefetchBuffer(
    HloModule* module, const MsaBufferInterval& prefetch_candidate) {
  Chunk chunk_candidate = FindChunkCandidate(prefetch_candidate);
  if (chunk_candidate.chunk_end() > available_heap_size()) {
    VLOG(3) << "Could not allocate preferred memory for cross program prefetch";
    return;
  }

  const HloValue* buffer = prefetch_candidate.buffer;
  int64_t parameter = buffer->instruction()->parameter_number();
  int cross_program_prefetch_index = module->CrossProgramPrefetches().size();
  SetDefaultMemorySpace(buffer, options_);
  module->AddCrossProgramPrefetch(parameter, buffer->index());

  AllocationSequence allocations;
  allocations.push_back(std::make_unique<PinnedAllocation>(
      buffer->defining_position(), MemorySpace::kDefault, kDummyChunk,
      prefetch_candidate.start, prefetch_candidate.end));

  // Find the earliest use.
  const auto& instruction_schedule = hlo_live_range_.instruction_schedule();
  auto uses = FindCrossProgramPrefetchUses(buffer->GetUses(), alias_analysis_);
  CHECK_GE(uses.size(), 1);
  auto use_schedule_compare = [&](const HloUse& lhs, const HloUse& rhs) {
    return instruction_schedule.at(lhs.instruction) <
           instruction_schedule.at(rhs.instruction);
  };
  auto first_use = absl::c_min_element(uses, use_schedule_compare);
  int64_t latest_prefetch_time =
      instruction_schedule.at(first_use->instruction);

  // Find the latest use time.
  int64_t last_use_time = instruction_schedule.at(
      absl::c_max_element(uses, use_schedule_compare)->instruction);
  for (const HloValue* colocation : prefetch_candidate.colocations) {
    auto colocation_uses = colocation->GetUses();
    if (!colocation_uses.empty()) {
      last_use_time = std::max(
          last_use_time,
          instruction_schedule.at(
              absl::c_max_element(colocation_uses, use_schedule_compare)
                  ->instruction));
    }
  }

  int64_t end_of_program_prefetch_end_time = instruction_schedule.size();
  int64_t end_of_program_inclusive_prefetch_start_time =
      options_.prefetch_interval_picker->PreferredPrefetchStartTime(
          buffer->defining_position().shape(), last_use_time,
          end_of_program_prefetch_end_time, end_of_program_prefetch_end_time);
  VLOG(2) << "last use time = " << last_use_time
          << ", end-of-program inclusive prefetch start time = "
          << end_of_program_inclusive_prefetch_start_time;
  float total_execution_time =
      options_.prefetch_interval_picker->GetLogicalIntervalElapsed(
          0, instruction_schedule.size());
  float buffer_occupied_time =
      options_.prefetch_interval_picker->GetLogicalIntervalElapsed(
          end_of_program_inclusive_prefetch_start_time,
          end_of_program_prefetch_end_time);
  if (options_.cost_analysis) {
    buffer_occupied_time = std::max(buffer_occupied_time,
                                    options_.cost_analysis->GetAsyncCopyElapsed(
                                        buffer->defining_position().shape()));
  }
  buffer_occupied_time +=
      options_.prefetch_interval_picker->GetLogicalIntervalElapsed(
          0, last_use_time);
  float buffer_occupied_ratio = buffer_occupied_time / total_execution_time;
  VLOG(2) << "Total execution time = " << total_execution_time
          << ", buffer occupied time = " << buffer_occupied_time
          << ", buffer occupied ratio = " << buffer_occupied_ratio;
  // Freeing buffer only makes sense if the buffer will be free for a
  // substantial time. Only perform this optimization if the ratio is below the
  // limit, and if the memory pressure is above the alternate memory size.
  bool free_buffer =
      (options_.enable_cross_program_prefetch_freeing &&
       memory_pressure_ > options_.max_size_in_bytes &&
       buffer_occupied_ratio < kCrossProgramPrefetchOccupyFreeingLimit &&
       end_of_program_inclusive_prefetch_start_time > last_use_time &&
       end_of_program_inclusive_prefetch_start_time <
           end_of_program_prefetch_end_time);
  int64_t cross_program_prefetch_end_time =
      free_buffer ? last_use_time : prefetch_candidate.end;

  AddAsyncCopyOrOtherMemOp(
      *allocations.back(), MemorySpace::kAlternate, chunk_candidate,
      /*exclusive_start_time=*/
      InclusiveToExclusiveStartTime(prefetch_candidate.start),
      cross_program_prefetch_end_time, latest_prefetch_time, &allocations,
      /*aliased_offset=*/nullptr,
      /*resource=*/0.0, cross_program_prefetch_index);

  absl::c_for_each(uses, [&](auto& use) { allocations.back()->AddUse(use); });
  AliasedOffset* cross_program_prefetch_offset =
      GetAliasedOffset(*allocations.back());

  if (free_buffer) {
    VLOG(2) << "Adding an end-of-program prefetch for freed "
               "cross-program-prefetched buffer.";
    AddAsyncCopyOrOtherMemOp(
        *allocations.front(), MemorySpace::kAlternate, chunk_candidate,
        /*exclusive_start_time=*/
        InclusiveToExclusiveStartTime(
            end_of_program_inclusive_prefetch_start_time),
        end_of_program_prefetch_end_time, end_of_program_prefetch_end_time,
        &allocations, cross_program_prefetch_offset,
        /*resource=*/0.0);
    CHECK_EQ(cross_program_prefetch_offset->offset,
             allocations.back()->chunk().offset);
  }

  const int allocations_initial_size = allocations_->size();
  for (auto& allocation : allocations) {
    if (allocation->memory_space() == MemorySpace::kAlternate) {
      MsaBufferInterval buffer_interval;
      buffer_interval.start = allocation->start_time();
      buffer_interval.end = allocation->end_time();
      buffer_interval.size = allocation->chunk().size;
      buffer_interval.buffer = prefetch_candidate.buffer;
      AddToPendingChunks(buffer_interval, chunk_candidate);
      for (const HloUse& use : allocation->uses()) {
        operands_in_alternate_memory_map_[use.instruction].insert(
            std::make_pair(use.operand_number, use.operand_index));
      }
    }
    allocations_->push_back(std::move(allocation));
  }

  // Add a repack allocation block for the Allocation objects in alternate
  // memory.
  std::vector<AllocationBlock*> colocations;
  for (int i = allocations_initial_size; i < allocations_->size(); ++i) {
    const auto& allocation = allocations_->at(i);
    if (allocation->memory_space() == MemorySpace::kAlternate) {
      repack_allocation_blocks_.push_back(MakeRepackAllocationBlock(
          allocation->start_time(), allocation->end_time(),
          allocation->chunk().size, allocation->chunk().offset,
          static_cast<int64_t>(repack_allocation_blocks_.size()),
          allocation.get()));
      colocations.push_back(&repack_allocation_blocks_.back());
    }
  }
  for (int i = 0; i < colocations.size() - 1; ++i) {
    colocations[i]->next_colocated = colocations[i + 1];
  }
  if (!colocations.empty()) {
    colocations.back()->next_colocated = colocations.front();
  }

  ClearPendingChunks();
}

void MsaAlgorithm::AllocateReservedScopedAllocations() {
  const std::vector<HloInstruction*>& instruction_sequence =
      hlo_live_range_.flattened_instruction_sequence().instructions();
  for (BreadthFirstMidpointIterator it(0, instruction_sequence.size() - 1);
       !it.End(); it.Next()) {
    HloInstruction* instruction = instruction_sequence[it.value()];
    int64_t reserved_scoped_memory =
        std::min(options_.reserved_scoped_memory_fn(
                     instruction, /*operands_in_alternate_memory=*/{},
                     /*outputs_in_alternate_memory=*/{}),
                 options_.max_size_in_bytes);
    if (reserved_scoped_memory == 0) {
      continue;
    }
    AllocateScopedAllocation(instruction, /*is_post_module=*/false,
                             reserved_scoped_memory, it.value());
  }
  // If requested, make all scoped allocations to colocate with each other so
  // that when we repack, all scoped allocations get the same offsets. Since
  // they will all have the same scoped memory addresses, this increases the
  // opportunity to deduplicate different ops.  However, this may hurt the
  // memory packing efficiency.
  if (options_.allocate_reserved_scoped_memory_at_same_offset) {
    for (auto allocation_block_it = repack_allocation_blocks_.begin();
         allocation_block_it != repack_allocation_blocks_.end() &&
         std::next(allocation_block_it) != repack_allocation_blocks_.end();
         ++allocation_block_it) {
      allocation_block_it->next_colocated = &*std::next(allocation_block_it);
    }
    if (!repack_allocation_blocks_.empty()) {
      repack_allocation_blocks_.back().next_colocated =
          &repack_allocation_blocks_.front();
    }
  }

  // Allocate post-module scoped allocation if requested. It never needs to be
  // colocated with other scoped allocations.
  if (options_.post_module_scoped_alternate_memory_size_in_bytes > 0) {
    AllocateScopedAllocation(
        /*instruction=*/module_->entry_computation()->root_instruction(),
        /*is_post_module=*/true,
        options_.post_module_scoped_alternate_memory_size_in_bytes,
        hlo_live_range_.schedule_end_time());
  }

  ClearPendingChunks();
}

void MsaAlgorithm::AllocateScopedAllocation(HloInstruction* instruction,
                                            bool is_post_module, int64_t size,
                                            int64_t time) {
  VLOG(1) << "Allocate reserved scoped memory at " << time << " ("
          << (is_post_module ? "<post-module>" : instruction->name())
          << "): " << size;
  MsaBufferInterval interval;
  interval.buffer = nullptr;
  interval.size = size;
  interval.start = time;
  interval.end = time;
  interval.need_allocation = true;
  Chunk chunk_candidate = FindChunkCandidate(interval, /*preferred_offset=*/0);
  CHECK_EQ(chunk_candidate.offset, 0);
  AddToPendingChunks(interval, chunk_candidate);

  if (options_.dump_fn != nullptr || VLOG_IS_ON(3)) {
    AppendScopedAllocationBufferInfoDebugString(instruction, time, size,
                                                buffer_info_str_);
  }

  allocations_->push_back(std::make_unique<ScopedAllocation>(
      chunk_candidate, time, instruction, is_post_module));

  repack_allocation_blocks_.push_back(MakeRepackAllocationBlock(
      time, time, size,
      /*initial_offset=*/0,
      static_cast<int64_t>(repack_allocation_blocks_.size()),
      allocations_->back().get()));
  repack_allocation_blocks_.back().next_colocated =
      &repack_allocation_blocks_.back();
}

int64_t MsaAlgorithm::GetCorrectedUseTime(
    const HloInstruction* instruction) const {
  const absl::flat_hash_map<const HloInstruction*, int64_t>& schedule =
      hlo_live_range_.instruction_schedule();
  if (instruction->opcode() == HloOpcode::kWhile) {
    // Given an example while loop and flattened schedule (logical times shown
    // on the left):
    //
    // 0:  a = ...
    // 1:  ...
    //     cond {
    // 2:   p = param(0)
    // 3:   ...
    //     }
    //     body {
    // 4:   p = param(0)
    // 5:   ...
    // 6:   ROOT ...
    //     }
    // 7:  w = while(a), body=body, cond=cond
    //
    // When processing "a" (time 0) and its while use (time 7), we update the
    // interval to time 0-4. This is so that the remaining interval (5-6) can be
    // allocated separately and this buffer doesn't waste alternate memory space
    // within the while loop body.
    HloComputation* while_body = instruction->while_body();
    // We require while body ROOTs to be the last in the schedule.
    CHECK_EQ(schedule.at(while_body->root_instruction()) + 1,
             schedule.at(instruction))
        << "While body ROOTs need to be the last in the schedule! "
           "Please run RootInstructionSinker.";
    // The corrected use is the parameter time. This is so that we can decide on
    // alternate memory allocations within the while loop body when we look at
    // uses within the while loop body.
    return schedule.at(while_body->parameter_instruction(0));
  }
  if (instruction->opcode() == HloOpcode::kConditional) {
    // The corrected use time is the earliest parameter of the called
    // computations.
    int64_t use_time = std::numeric_limits<int64_t>::max();
    for (const HloComputation* called_computation :
         instruction->called_computations()) {
      use_time = std::min(
          use_time, schedule.at(called_computation->parameter_instruction(0)));
    }
    return use_time;
  }
  // Otherwise, just return the time of the use instruction.
  return hlo_live_range_.instruction_schedule().at(instruction);
}

int64_t MsaAlgorithm::GetCorrectedUseTime(const HloUse& use) const {
  return GetCorrectedUseTime(use.instruction);
}

std::optional<MsaAlgorithm::RequiredMemoryAssignment>
MsaAlgorithm::RequiredMemoryAssignmentAt(const HloValue* buffer,
                                         int64_t time) const {
  auto required_assignment_it = required_assignments_.find(buffer);
  std::optional<RequiredMemoryAssignment> required_assignment_at_time;
  if (required_assignment_it != required_assignments_.end()) {
    for (const RequiredMemoryAssignment& required_assignment :
         required_assignment_it->second) {
      if (required_assignment.time == time) {
        // Sanity check that there is only one required at time.
        CHECK(!required_assignment_at_time)
            << buffer->ToShortString() << " at time " << time;
        required_assignment_at_time = required_assignment;
      }
    }
  }
  return required_assignment_at_time;
}

std::optional<MsaAlgorithm::RequiredMemoryAssignment>
MsaAlgorithm::AliasedRequiredAssignmentForUse(
    const AllocationValue::Use& use) const {
  std::optional<RequiredMemoryAssignment> required_assignment;
  for (const HloPosition& position : use.aliases) {
    const HloValue* value =
        &alias_analysis_.dataflow_analysis().GetUniqueValueAt(
            position.instruction, position.index);
    int64_t time =
        hlo_live_range_.instruction_schedule().at(position.instruction);
    std::optional<RequiredMemoryAssignment> required_assignment_for_alias =
        RequiredMemoryAssignmentAt(value, time);
    if (required_assignment == std::nullopt) {
      required_assignment = required_assignment_for_alias;
    } else {
      CHECK(required_assignment_for_alias == std::nullopt ||
            required_assignment->equals_ignoring_time(
                *required_assignment_for_alias));
    }
  }
  return required_assignment;
}

void MsaAlgorithm::AddAliasedRequiredAssignment(
    const HloInstruction* instruction, ShapeIndex index,
    const Allocation* aliased_allocation) {
  AliasedOffset* offset = nullptr;
  if (aliased_allocation->memory_space() == MemorySpace::kAlternate) {
    offset = GetAliasedOffset(*aliased_allocation);
  }
  AddRequiredAssignment(instruction, index, aliased_allocation->memory_space(),
                        offset);
}

void MsaAlgorithm::AddRequiredAssignment(const HloValue* value,
                                         const HloInstruction* instruction,
                                         MemorySpace memory_space, int64_t time,
                                         AliasedOffset* offset,
                                         bool add_to_pending) {
  // Check for existing required assignment at this time and make sure it is the
  // same as this if there is one.
  auto existing_required_assignment = RequiredMemoryAssignmentAt(value, time);
  if (existing_required_assignment) {
    CHECK(memory_space == existing_required_assignment->memory_space)
        << "inst = " << instruction->ToString() << " at " << time;
    CHECK((!offset && !existing_required_assignment->offset) ||
          offset == existing_required_assignment->offset);
    VLOG(3) << "Not adding required assignment because there is one already: "
            << value->ToShortString() << " at " << time << " at "
            << (memory_space == MemorySpace::kDefault ? "def" : "alt");
  } else {
    VLOG(3) << "Adding required assignment: " << value->ToShortString()
            << " at " << time << " at "
            << (memory_space == MemorySpace::kDefault ? "def" : "alt");
    RequiredMemoryAssignment required_assignment{memory_space, time, offset};
    required_assignments_[value].push_back(required_assignment);
    if (add_to_pending) {
      pending_required_assignments_.push_back({value, required_assignment});
    }
  }
}

void MsaAlgorithm::AddRequiredAssignment(const HloInstruction* instruction,
                                         ShapeIndex index,
                                         MemorySpace memory_space,
                                         AliasedOffset* offset,
                                         bool add_to_pending) {
  const HloValue* value =
      &alias_analysis_.dataflow_analysis().GetUniqueValueAt(instruction, index);
  int64_t instruction_time =
      hlo_live_range_.instruction_schedule().at(instruction);
  AddRequiredAssignment(value, instruction, memory_space, instruction_time,
                        offset, add_to_pending);
}

void MsaAlgorithm::AddRequiredAssignment(const HloPosition& position,
                                         MemorySpace memory_space,
                                         AliasedOffset* offset,
                                         bool add_to_pending) {
  AddRequiredAssignment(position.instruction, position.index, memory_space,
                        offset, add_to_pending);
}

void MsaAlgorithm::AddRequiredAssignment(const HloUse& use,
                                         MemorySpace memory_space,
                                         AliasedOffset* offset,
                                         bool add_to_pending) {
  const HloValue* value = &alias_analysis_.dataflow_analysis().GetUniqueValueAt(
      use.instruction->operand(use.operand_number), use.operand_index);
  int64_t instruction_time = GetCorrectedUseTime(use);
  AddRequiredAssignment(value, use.instruction, memory_space, instruction_time,
                        offset, add_to_pending);
}

void MsaAlgorithm::AddInputAndOutputRequiredAssignments() {
  // Go through the parameters, outputs, and constants and pin them to the
  // corresponding memory by adding a required assignment.
  const HloModule& module = alias_analysis_.dataflow_analysis().module();
  const auto& instruction_schedule = hlo_live_range_.instruction_schedule();
  HloComputation* entry_computation = module.entry_computation();
  for (HloInstruction* parameter_instruction :
       entry_computation->parameter_instructions()) {
    int64_t parameter_instruction_time =
        instruction_schedule.at(parameter_instruction);
    ShapeUtil::ForEachSubshape(
        parameter_instruction->shape(),
        [&](const Shape& subshape, const ShapeIndex& index) {
          MemorySpace memory_space = MemorySpace::kDefault;
          if (subshape.has_layout() && subshape.layout().memory_space() ==
                                           options_.alternate_memory_space) {
            memory_space = MemorySpace::kAlternate;
          }
          for (const HloBuffer* buffer :
               alias_analysis_.ComputeBuffersAt(parameter_instruction, index)) {
            for (const HloValue* value : buffer->values()) {
              VLOG(3) << "Adding required assignment for parameter value = "
                      << value->ToShortString()
                      << " time = " << parameter_instruction_time << " space = "
                      << (memory_space == MemorySpace::kDefault ? "def"
                                                                : "alt");
              AddRequiredAssignment(value, parameter_instruction, memory_space,
                                    parameter_instruction_time,
                                    /*offset=*/nullptr,
                                    /*add_to_pending=*/false);
            }
          }
        });
  }
  HloInstruction* root_instruction = entry_computation->root_instruction();
  int64_t root_instruction_time = instruction_schedule.at(root_instruction);
  ShapeUtil::ForEachSubshape(
      root_instruction->shape(),
      [&](const Shape& subshape, const ShapeIndex& index) {
        MemorySpace memory_space = MemorySpace::kDefault;
        if (subshape.has_layout() && subshape.layout().memory_space() ==
                                         options_.alternate_memory_space) {
          memory_space = MemorySpace::kAlternate;
        }
        for (const HloBuffer* buffer :
             alias_analysis_.ComputeBuffersAt(root_instruction, index)) {
          for (const HloValue* value : buffer->values()) {
            VLOG(3) << "Adding required assignment for output value = "
                    << value->ToShortString()
                    << " time = " << root_instruction_time << " space = "
                    << (memory_space == MemorySpace::kDefault ? "def" : "alt");
            AddRequiredAssignment(value, root_instruction, memory_space,
                                  root_instruction_time,
                                  /*offset=*/nullptr, /*add_to_pending=*/false);
          }
        }
      });

  for (const HloComputation* computation : module.MakeNonfusionComputations()) {
    for (HloInstruction* instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kConstant) {
        auto constant_instruction_it = instruction_schedule.find(instruction);
        if (constant_instruction_it == instruction_schedule.end()) {
          continue;
        }
        int64_t constant_instruction_time = constant_instruction_it->second;
        ShapeUtil::ForEachLeafShape(
            instruction->shape(),
            [&](const Shape& /*sub_shape*/, const ShapeIndex& index) {
              for (const HloBuffer* buffer :
                   alias_analysis_.ComputeBuffersAt(instruction, index)) {
                for (const HloValue* value : buffer->values()) {
                  VLOG(3) << "Adding required assignment for constant value = "
                          << value->ToShortString()
                          << " time = " << constant_instruction_time
                          << " space = def";
                  AddRequiredAssignment(value, instruction,
                                        MemorySpace::kDefault,
                                        constant_instruction_time,
                                        /*offset=*/nullptr,
                                        /*add_to_pending=*/false);
                }
              }
            });
      }
    }
  }

  // Go through all of the values and pin them to the default memory if they are
  // not allowed on the alternate memory.
  for (const HloValue* value : alias_analysis_.dataflow_analysis().values()) {
    if (!options_.is_allowed_in_alternate_mem_fn(*value)) {
      // We won't find the instruction in the schedule if it's inside a fusion.
      // If so, just skip.
      auto instruction_time_it =
          instruction_schedule.find(value->instruction());
      if (instruction_time_it == instruction_schedule.end()) {
        continue;
      }
      int64_t instruction_time = instruction_time_it->second;
      auto& required_assignments = required_assignments_[value];
      // Check if there is an existing matching required assignment (e.g.
      // inserted by the logic above) and if so ensure it requires a default
      // memory allocation.
      auto matching_assignment = absl::c_find_if(
          required_assignments,
          [&](const RequiredMemoryAssignment& required_assignment) {
            return required_assignment.time == instruction_time;
          });
      if (matching_assignment != required_assignments.end()) {
        CHECK(matching_assignment->memory_space == MemorySpace::kDefault)
            << "Mismatch in required assignments at time " << instruction_time
            << " value: " << value->ToString();
      } else {
        VLOG(3) << "Adding required assignment: " << value->ToShortString()
                << " at " << instruction_time << " at def";
        required_assignments.push_back(
            {MemorySpace::kDefault, instruction_time});
      }
    }
  }
}

bool MsaAlgorithm::AreIntervalsReservedInAlternateMemory(
    absl::Span<const MsaBufferInterval* const> colocated_intervals) const {
  auto is_position_in_alternate_memory = [&](const HloPosition& position) {
    const Shape& shape = position.shape();
    return shape.has_layout() &&
           shape.layout().memory_space() == options_.alternate_memory_space;
  };

  const HloModule& module = alias_analysis_.dataflow_analysis().module();
  const HloComputation* entry_computation = module.entry_computation();
  const HloInstruction* root_instruction =
      entry_computation->root_instruction();
  for (const MsaBufferInterval* colocated_interval : colocated_intervals) {
    const HloValue* value = colocated_interval->buffer;
    if (value->defining_instruction()->opcode() == HloOpcode::kParameter &&
        value->defining_instruction()->parent() == entry_computation &&
        is_position_in_alternate_memory(value->defining_position())) {
      return true;
    }

    for (const HloPosition& position : value->positions()) {
      if (position.instruction == root_instruction &&
          is_position_in_alternate_memory(position)) {
        return true;
      }
    }
  }
  return false;
}

const std::vector<const HloInstruction*>*
MsaAlgorithm::GetRepeatedInstructionList(
    const HloInstruction* instruction) const {
  const auto fingerprint_it = fingerprint_map_.find(instruction);
  if (fingerprint_it == fingerprint_map_.end()) {
    return nullptr;
  }
  const auto repeated_insts_it =
      repeated_inst_map_.find(fingerprint_it->second);
  CHECK(repeated_insts_it != repeated_inst_map_.end());
  return &repeated_insts_it->second;
}

void MsaAlgorithm::UpdateReservedScopedAllocationSize() {
  // Check all instructions, if their operands/outputs have been placed in
  // alternate memory, update their scoped allocation size.
  VLOG(2) << "Update scoped allocation size before repacking.";
  const auto& instruction_sequence =
      hlo_live_range_.flattened_instruction_sequence().instructions();
  absl::flat_hash_map<int64_t, int64_t> reserved_scoped_memory_map;
  for (int i = 0; i < instruction_sequence.size(); ++i) {
    const HloInstruction* instruction = instruction_sequence[i];
    absl::flat_hash_set<std::pair<int, ShapeIndex>>
        empty_operand_in_alternate_memory;
    auto find_operands_in_alternate_memory_it =
        operands_in_alternate_memory_map_.find(instruction);
    const absl::flat_hash_set<std::pair<int, ShapeIndex>>&
        operands_in_alternate_memory =
            find_operands_in_alternate_memory_it ==
                    operands_in_alternate_memory_map_.end()
                ? empty_operand_in_alternate_memory
                : find_operands_in_alternate_memory_it->second;
    absl::flat_hash_set<ShapeIndex> empty_output_in_alternate_memory;
    auto find_outputs_in_alternate_memory_it =
        outputs_in_alternate_memory_map_.find(instruction);
    const absl::flat_hash_set<ShapeIndex>& outputs_in_alternate_memory =
        find_outputs_in_alternate_memory_it ==
                outputs_in_alternate_memory_map_.end()
            ? empty_output_in_alternate_memory
            : find_outputs_in_alternate_memory_it->second;
    reserved_scoped_memory_map[i] = options_.reserved_scoped_memory_fn(
        instruction, operands_in_alternate_memory, outputs_in_alternate_memory);
  }
  // Update scoped allocation sizes.
  for (RepackAllocationBlock& allocation_block : repack_allocation_blocks_) {
    Allocation* allocation = allocation_block.allocation;
    if (allocation->is_scoped_allocation()) {
      int64_t time = allocation->start_time();
      peak_memory_usage_[time] +=
          (reserved_scoped_memory_map[time] - allocation->chunk().size);
      allocation_block.size = reserved_scoped_memory_map[time];
      allocation->mutable_chunk()->size = reserved_scoped_memory_map[time];
    }
  }
}

void MsaAlgorithm::ExportAllocationsForRepacking(
    std::vector<AllocationBlock*>& allocations) {
  using SliceDetail = SlicedCopyAllocation::SliceDetail;

  if (options_.reduce_scoped_memory_limit) {
    UpdateReservedScopedAllocationSize();
  }

  for (RepackAllocationBlock& allocation_block : repack_allocation_blocks_) {
    allocation_block.original_slice_data = std::nullopt;
    allocation_block.repacked_slice_data = std::nullopt;

    if (!allocation_block.allocation->is_sliced_copy_allocation()) {
      allocations.push_back(&allocation_block);
      continue;
    }

    SlicedCopyAllocation* allocation =
        dynamic_cast<SlicedCopyAllocation*>(allocation_block.allocation);
    std::vector<const SliceDetail*> slice_details_sorted_by_offset;
    slice_details_sorted_by_offset.reserve(
        allocation->slice_details_sorted_by_start_time().size());
    for (const SliceDetail& slice_detail :
         allocation->slice_details_sorted_by_start_time()) {
      slice_details_sorted_by_offset.push_back(&slice_detail);
    }
    absl::c_stable_sort(slice_details_sorted_by_offset,
                        [](const SliceDetail* lhs, const SliceDetail* rhs) {
                          return lhs->slice_decision.chunk.offset <
                                 rhs->slice_decision.chunk.offset;
                        });

    // Since this is a sliced allocation, construct SlicedAllocationData to
    // attach to the AllocationBlock.
    SlicedAllocationData original_slice_data;
    for (const SliceDetail* slice_detail : slice_details_sorted_by_offset) {
      CHECK_EQ(slice_detail->copy_start_after_time,
               slice_detail->slice_decision.exclusive_start_time);
      original_slice_data.slices_sorted_by_offset.push_back(AllocatedSlice{
          slice_detail->slice_decision.chunk.size,
          slice_detail->slice_decision.chunk.offset,
          /*inclusive_start_time=*/
          ExclusiveToInclusiveStartTime(
              slice_detail->slice_decision.exclusive_start_time)});
    }

    allocation_block.original_slice_data = std::move(original_slice_data);
    allocations.push_back(&allocation_block);
  }
}

void MsaAlgorithm::ImportRepackedAllocations() {
  interval_tree_.Clear();
  for (RepackAllocationBlock& allocation_block : repack_allocation_blocks_) {
    if (allocation_block.allocation->is_sliced_copy_allocation()) {
      ImportRepackedSlicedAllocation(allocation_block);
      continue;
    }
    ImportRepackedNonSlicedAllocation(allocation_block);
  }
}

void MsaAlgorithm::ImportRepackedNonSlicedAllocation(
    RepackAllocationBlock& block) {
  Allocation* allocation = block.allocation;
  int64_t original_offset = block.initial_offset;
  int64_t repacked_offset = block.offset;

  // Update the Allocation, AllocationBlock, and interval_tree_.
  allocation->set_offset(repacked_offset);
  block.initial_offset = repacked_offset;
  block.offset = -1;
  interval_tree_.Add(
      block.inclusive_start_time, block.end_time,
      HeapSimulator::Chunk::FromOffsetSize(repacked_offset, block.size));

  VLOG(3) << "Repacking move. offset: " << original_offset << " -> "
          << repacked_offset << "; size: " << block.size
          << "; Allocation: " << allocation->ToString();
}

void MsaAlgorithm::ImportRepackedSlicedAllocation(
    RepackAllocationBlock& block) {
  using SlicedCopyAllocation = memory_space_assignment::SlicedCopyAllocation;
  using SliceDetail = SlicedCopyAllocation::SliceDetail;

  CHECK_OK(AreRepackedSlicesValid(block));

  SlicedCopyAllocation* allocation =
      dynamic_cast<SlicedCopyAllocation*>(block.allocation);
  CHECK(block.allocation->is_sliced_copy_allocation());
  int64_t original_offset = block.initial_offset;
  int64_t repacked_offset = block.offset;
  std::vector<int64_t> original_slice_offsets =
      allocation->SliceOffsetsSortedByStartTime();

  // Update the Allocation, AllocationBlock, and interval_tree_.
  allocation->set_offset(repacked_offset);
  if (block.repacked_slice_data.has_value()) {
    allocation->ImportRepackedSliceData(*block.repacked_slice_data);
  } else {
    allocation->AddDiffToAllSliceOffsets(repacked_offset - original_offset);
  }
  block.initial_offset = repacked_offset;
  block.offset = -1;
  // Note, in a non-repacking setting, we would have reworked the chunks as
  // described in
  // MsaAlgorithm::PrefetchContext::SlicedSolution::slices_for_pending_chunks.
  // Doing so was for the benefit of MsaAlgorithm::pending_chunks_. However,
  // pending_chunks_ are cleared before repacking, when UncommitPendingChunks()
  // is called. Thus, we don't need to worry about modifying the chunks here.
  for (const SliceDetail& slice_detail :
       allocation->slice_details_sorted_by_start_time()) {
    interval_tree_.Add(
        /*start=*/
        ExclusiveToInclusiveStartTime(slice_detail.copy_start_after_time),
        block.end_time, slice_detail.slice_decision.chunk);
  }

  VLOG(3) << "Repacking move. offset: " << original_offset << " -> "
          << repacked_offset << "; size: " << block.size << "; " <<
      [&]() {
        std::vector<int64_t> new_slice_offsets =
            allocation->SliceOffsetsSortedByStartTime();
        CHECK_EQ(original_slice_offsets.size(), new_slice_offsets.size());
        std::vector<std::string> offset_moves;
        offset_moves.reserve(original_slice_offsets.size());
        for (int i = 0; i < original_slice_offsets.size(); ++i) {
          offset_moves.push_back(absl::StrCat(original_slice_offsets[i], " -> ",
                                              new_slice_offsets[i]));
        }
        return absl::StrCat("slice_offsets: [",
                            absl::StrJoin(offset_moves, ", "), "]");
      }()
          << "; Allocation: " << allocation->ToString();
}

absl::Status MsaAlgorithm::AreRepackedSlicesValid(
    const RepackAllocationBlock& block) {
  if (!block.repacked_slice_data.has_value()) {
    return absl::OkStatus();
  }
  if (!block.original_slice_data.has_value()) {
    return InvalidArgumentStrCat(
        "Repacked sliced allocation has repacked slice data but not original "
        "slice data.");
  }
  int64_t num_slices =
      block.original_slice_data->slices_sorted_by_offset.size();
  if (num_slices != block.repacked_slice_data->slices_sorted_by_offset.size()) {
    return InvalidArgumentStrCat(
        "Repacked sliced allocation has ", num_slices,
        " slices but repacking has data for ",
        block.repacked_slice_data->slices_sorted_by_offset.size(), " slices.");
  }

  // Ensure that the slice size to start time mapping has not changed. If it
  // changes, its invalidates MSA's internal state, e.g., the peak_memory_usage_
  // data structure.
  std::vector<std::pair<int64_t, int64_t>> original_size_to_time_mapping;
  original_size_to_time_mapping.reserve(num_slices);
  for (const AllocatedSlice& slice :
       block.original_slice_data->slices_sorted_by_offset) {
    original_size_to_time_mapping.push_back(
        std::make_pair(slice.size, slice.inclusive_start_time));
  };
  absl::c_sort(original_size_to_time_mapping);
  std::vector<std::pair<int64_t, int64_t>> repacked_size_to_time_mapping;
  repacked_size_to_time_mapping.reserve(num_slices);
  for (const AllocatedSlice& slice :
       block.repacked_slice_data->slices_sorted_by_offset) {
    repacked_size_to_time_mapping.push_back(
        std::make_pair(slice.size, slice.inclusive_start_time));
  };
  absl::c_sort(repacked_size_to_time_mapping);
  if (original_size_to_time_mapping != repacked_size_to_time_mapping) {
    return InvalidArgumentStrCat(
        "Repacked slices do not preserve the initial slice size-start time "
        "mappings.");
  }

  return absl::OkStatus();
}

void MsaAlgorithm::UncommitPendingChunks(
    absl::Span<AllocationValue> allocation_values) {
  // Clear the allocation sequence of the allocation values so that in case we
  // retry allocation after uncommitting.
  for (AllocationValue& allocation_value : allocation_values) {
    allocation_value.mutable_allocation_sequence()->clear();
  }
  for (const auto& interval_and_chunk : pending_chunks_) {
    const MsaBufferInterval& interval = interval_and_chunk.first;
    const Chunk& chunk = interval_and_chunk.second;
    VLOG(3) << "Uncommitting: [" << interval.start << ", " << interval.end
            << "] off = " << chunk.offset << " size = " << chunk.size;
    for (int i = interval.start; i <= interval.end; ++i) {
      peak_memory_usage_[i] -= chunk.size;
      CHECK_GE(peak_memory_usage_[i], 0)
          << "Peak memory usage at " << i
          << " is below zero after uncommitting. " << interval.start << "-"
          << interval.end << " : [" << chunk.offset << ", " << chunk.size
          << "]";
    }
    interval_tree_.Remove(interval.start, interval.end, chunk);
  }
  for (const AsynchronousCopy& async_copy : pending_async_copies_) {
    if (async_copy.destination == MemorySpace::kAlternate) {
      prefetch_interval_tree_.Remove(
          /*start=*/
          ExclusiveToInclusiveStartTime(async_copy.exclusive_start_time),
          async_copy.end_time, kDummyChunk);
      prefetch_async_copy_resource_.RemoveCopy(async_copy);
      if (options_.enforce_prefetch_fifo_order) {
        async_copy_ordering_.RemoveCopy(async_copy);
      }
    } else {
      eviction_interval_tree_.Remove(
          /*start=*/
          ExclusiveToInclusiveStartTime(async_copy.exclusive_start_time),
          async_copy.end_time, kDummyChunk);
      eviction_async_copy_resource_.RemoveCopy(async_copy);
    }
  }
  for (const auto& value_and_required_assignment :
       pending_required_assignments_) {
    auto& required_assignment_vector =
        required_assignments_[value_and_required_assignment.first];
    const RequiredMemoryAssignment& required_assignment =
        value_and_required_assignment.second;
    VLOG(3) << "Removing required assignment: "
            << (required_assignment.memory_space == MemorySpace::kDefault
                    ? "def"
                    : "alt")
            << " time = " << required_assignment.time << " off = "
            << (required_assignment.offset ? required_assignment.offset->offset
                                           : -1);
    for (auto it = required_assignment_vector.begin();
         it != required_assignment_vector.end(); ++it) {
      if (*it == value_and_required_assignment.second) {
        required_assignment_vector.erase(it);
        break;
      }
    }
  }
  ClearPendingChunks();
}

void MsaAlgorithm::FinalizeAllocations(
    absl::Span<AllocationValue> allocation_values) {
  VLOG(3) << "Finalized allocations:";
  for (const HloInstruction* copy_inst : sorted_async_conversion_candidates_) {
    successful_async_conversion_set_.insert(copy_inst);
  }
  not_finalized_async_conversions_.clear();
  std::vector<std::pair<const AliasedOffset*, std::vector<Allocation*>>>
      colocation_vector;
  absl::flat_hash_map<const AliasedOffset*, size_t> offset_to_index;
  for (int allocation_value_idx = 0;
       allocation_value_idx < allocation_values.size();
       ++allocation_value_idx) {
    auto& allocation_value = allocation_values.at(allocation_value_idx);
    VLOG(3) << "  " << allocation_value_idx + 1 << "/"
            << allocation_values.size() << ") "
            << allocation_value.ToShortString();
    for (auto& allocation : *allocation_value.mutable_allocation_sequence()) {
      VLOG(3) << "    " << allocation->ToString();
      if (allocation->memory_space() == MemorySpace::kAlternate &&
          allocation_value.mutable_split_shape().has_value()) {
        allocation->set_split_shape(allocation_value.mutable_split_shape());
      }
      if ((allocation->memory_space() == MemorySpace::kAlternate) &&
          (!allocation->is_scoped_allocation())) {
        for (const HloUse& use : allocation->uses()) {
          operands_in_alternate_memory_map_[use.instruction].insert(
              std::make_pair(use.operand_number, use.operand_index));
        }
        if (!allocation->is_copy_like_allocation()) {
          outputs_in_alternate_memory_map_[allocation->defining_position()
                                               .instruction]
              .insert(allocation->defining_position().index);
        }
      }
      allocations_->push_back(std::move(allocation));
      Allocation* inserted_allocation = allocations_->back().get();
      if (inserted_allocation->memory_space() == MemorySpace::kAlternate) {
        auto* aliased_offset = GetAliasedOffset(*inserted_allocation);
        auto [it, inserted] =
            offset_to_index.emplace(aliased_offset, colocation_vector.size());
        if (inserted) {
          colocation_vector.emplace_back(aliased_offset,
                                         std::vector{inserted_allocation});
        } else {
          size_t index = it->second;
          colocation_vector[index].second.push_back(inserted_allocation);
        }
      }
    }
  }
  // The allocations that have the same AliasedOffset need to be colocated.
  // Export these to repack_allocation_blocks_ so that we can repack them to
  // reduce fragmentation.
  for (auto& colocation : colocation_vector) {
    std::vector<AllocationBlock*> colocations;
    for (Allocation* colocated_allocation : colocation.second) {
      repack_allocation_blocks_.push_back(MakeRepackAllocationBlock(
          colocated_allocation->start_time(), colocated_allocation->end_time(),
          colocated_allocation->chunk().size,
          colocated_allocation->chunk().offset,
          static_cast<int64_t>(repack_allocation_blocks_.size()),
          colocated_allocation));
      colocations.push_back(&repack_allocation_blocks_.back());
    }
    for (int i = 0; i < colocations.size() - 1; ++i) {
      colocations[i]->next_colocated = colocations[i + 1];
    }
    if (!colocations.empty()) {
      colocations.back()->next_colocated = colocations.front();
    }
  }
  ClearPendingChunks();
}

void MsaAlgorithm::ClearPendingChunks() {
  pending_chunks_.clear();
  pending_async_copies_.clear();
  pending_required_assignments_.clear();
  aliased_offset_map_.clear();
  aliased_offsets_.clear();
}

bool MsaAlgorithm::IsInstructionPendingReplacements(
    const HloInstruction* instruction) const {
  return std::find(sorted_async_conversion_candidates_.begin(),
                   sorted_async_conversion_candidates_.end(),
                   instruction) != sorted_async_conversion_candidates_.end();
}

void MsaAlgorithm::AddToPendingChunks(const MsaBufferInterval& buffer_interval,
                                      const Chunk& chunk_candidate) {
  VLOG(3) << "Committing chunk: " << buffer_interval.start << "-"
          << buffer_interval.end << " : " << chunk_candidate.ToString();
  pending_chunks_.emplace_back(buffer_interval, chunk_candidate);
  for (int i = buffer_interval.start; i <= buffer_interval.end; ++i) {
    peak_memory_usage_[i] += chunk_candidate.size;
    CHECK_LE(peak_memory_usage_[i], options_.max_size_in_bytes)
        << "Peak memory usage at " << i
        << " exceeds the max size of alternate memory. "
        << buffer_interval.start << "-" << buffer_interval.end << " : "
        << chunk_candidate.ToString();
  }
  CommitChunk(buffer_interval, chunk_candidate);
}

std::optional<int> MsaAlgorithm::FindEarliestExclusiveTimeToSatisfyPeakMemory(
    int exclusive_start_time, int end_time, int64_t size) const {
  std::optional<int> earliest_time_exclusive = std::nullopt;
  for (int time_inclusive = ExclusiveToInclusiveEndTime(end_time);
       time_inclusive > exclusive_start_time; --time_inclusive) {
    if (peak_memory_usage_[time_inclusive] + size <=
        options_.max_size_in_bytes) {
      earliest_time_exclusive = InclusiveToExclusiveStartTime(time_inclusive);
    } else {
      break;
    }
  }

  return earliest_time_exclusive;
}

std::string MsaAlgorithm::SingleFailureResultToString(
    const AllocationResult& result) {
  switch (result) {
    case AllocationResult::kSuccess:
      return "Success";
    case AllocationResult::kFailOutOfMemory:
      return "FailOutOfMemory";
    case AllocationResult::kFailPrevAllocationNotInAlternateMem:
      return "FailPrevAllocationNotInAlternateMem";
    case AllocationResult::kFailLiveRangeTooLong:
      return "FailLiveRangeTooLong";
    case AllocationResult::kFailLiveRangeTooShort:
      return "FailLiveRangeTooShort";
    case AllocationResult::kFailOutOfAsyncCopies:
      return "FailOutOfAsyncCopies";
    case AllocationResult::kFailViolatesAsyncCopyResource:
      return "FailViolatesAsyncCopyResource";
    case AllocationResult::kFailRequiresUncommit:
      return "FailRequiresUncommit";
    case AllocationResult::kAllSlicesHaveTheSameStartTime:
      return "AllSlicesHaveTheSameStartTime";
    case AllocationResult::kFailConflictingPreferredOffsets:
      return "FailConflictingPreferredOffsets";
    case AllocationResult::kFailSyncDataMoveReplacement:
      return "FailSyncDataMoveReplacement";
    default:
      return "UnknownResult";
  }
}

std::string MsaAlgorithm::ResultToString(const AllocationResult& result) {
  if (result == AllocationResult::kSuccess) {
    return "Success";
  }
  std::string result_str = "";
  for (int failure_order = 0; failure_order < 16; ++failure_order) {
    AllocationResult failure_value =
        static_cast<AllocationResult>(1 << failure_order);
    if (result_is(result, static_cast<AllocationResult>(failure_value))) {
      result_str += (SingleFailureResultToString(failure_value) + " | ");
    }
  }
  result_str = result_str.substr(0, result_str.size() - 3);
  return result_str;
}

void MsaAlgorithm::CheckAndUpdateForDualLiveAllocationValues(
    const std::optional<RequiredMemoryAssignment>&
        required_memory_assignment_at_start,
    AllocationRequest& request) {
  if (!request.allocation_value->requires_contiguous_allocation()) {
    return;
  }
  if (!required_memory_assignment_at_start.has_value()) {
    return;
  }
  if (required_memory_assignment_at_start->memory_space !=
      MemorySpace::kAlternate) {
    return;
  }
  // Go through previous allocations, for the same HloValue, and check if they
  // have already allocated alternate memory at the beginning of the current
  // AllocationValue, such that we are required to use the same heap offset.
  std::vector<Allocation*> overlapping_allocations;
  Chunk required_chunk = Chunk::FromOffsetSize(
      required_memory_assignment_at_start->offset->offset, request.size);
  for (const AllocationValue& processed_allocation_value :
       request.processed_allocation_values) {
    for (const std::unique_ptr<Allocation>& allocation :
         *processed_allocation_value.allocation_sequence()) {
      if (allocation->is_in_alternate_mem() &&
          allocation->start_time() <= request.inclusive_start_time &&
          request.inclusive_start_time <= allocation->end_time() &&
          allocation->chunk() == required_chunk) {
        overlapping_allocations.push_back(allocation.get());
      }
    }
  }
  absl::c_sort(overlapping_allocations,
               [](const Allocation* a, const Allocation* b) {
                 return a->start_time() < b->start_time();
               });
  int64_t chunk_start_time = request.inclusive_start_time;
  for (const Allocation* allocation : overlapping_allocations) {
    chunk_start_time = std::max(chunk_start_time, allocation->end_time() + 1);
  }

  // Note, we don't have to set request.preferred_offset, or do anything special
  // to handle aliasing. This is done for us. Specifically, before calling
  // CheckAndUpdateForDualLiveAllocationValues(), AllocateSegment() inserts a
  // PinnedAllocation with no associated heap chunk, at the beginning of
  // request.allocation_value. It aliases that PinnedAllocation with any
  // overlapping allocations calculated above. In
  // AllocateInAlternateMemoryNoCopy(), we will find that PinnedAllocation and
  // realize we need to use the same alternate memory offset.
  request.no_copy_chunk_inclusive_start_time = chunk_start_time;
  VLOG(3) << "Setting the no-copy chunk (inc) start time to "
          << chunk_start_time;
}

void MsaAlgorithm::ReleaseReservedAllocationForAlternateMemoryColorings(
    ReservedAllocation* reserved_allocation) {
  // We check if the reserved chunk is still reserved because this might
  // be a retry of the same allocation request and the chunk might have
  // been released in the previous attempt.
  if (!reserved_allocation->is_chunk_reserved_in_interval_tree()) {
    return;
  }
  // Release the reserved chunk from the interval tree.
  CHECK(interval_tree_.Remove(reserved_allocation->start_time(),
                              reserved_allocation->end_time(),
                              reserved_allocation->chunk()));
  reserved_allocation->chunk_freed_in_interval_tree();
  // Remove the allocation from the repack_allocation_blocks_ list.
  auto it = std::remove_if(
      repack_allocation_blocks_.begin(), repack_allocation_blocks_.end(),
      [reserved_allocation](
          const RepackAllocationBlock& repack_allocation_block) {
        return repack_allocation_block.allocation == reserved_allocation;
      });
  size_t original_size = repack_allocation_blocks_.size();
  repack_allocation_blocks_.erase(it, repack_allocation_blocks_.end());
  CHECK_EQ(original_size - repack_allocation_blocks_.size(), 1);
}

void MsaAlgorithm::FreeAlternateMemoryColoringReservedAllocations(
    AllocationRequest& request) {
  if (!request.require_start_colored_in_alternate_memmory &&
      !request.require_end_colored_in_alternate_memory) {
    return;
  }
  const HloPosition& defining_position =
      request.allocation_value->defining_position();
  auto reserved_allocations_it =
      reserved_allocations_for_alt_mem_colorings_.find(defining_position);
  CHECK(reserved_allocations_it !=
        reserved_allocations_for_alt_mem_colorings_.end());

  int64_t inclusive_start_time = request.inclusive_start_time;
  int64_t use_time = request.end_time;
  for (std::unique_ptr<ReservedAllocation>& reserved_allocation_ptr :
       reserved_allocations_it->second) {
    if (request.require_start_colored_in_alternate_memmory &&
        reserved_allocation_ptr->start_time() == inclusive_start_time) {
      ReleaseReservedAllocationForAlternateMemoryColorings(
          reserved_allocation_ptr.get());
    }
    if (request.require_end_colored_in_alternate_memory &&
        reserved_allocation_ptr->end_time() == use_time) {
      ReleaseReservedAllocationForAlternateMemoryColorings(
          reserved_allocation_ptr.get());
    }
  }
}

void MsaAlgorithm::UpdateRequestWithAlternateMemoryColoringRequirements(
    AllocationRequest& request) {
  if (!request.allocation_value) {
    return;
  }
  const HloPosition& defining_position =
      request.allocation_value->defining_position();
  int64_t definition_time =
      hlo_live_range_.instruction_schedule().at(defining_position.instruction);

  int64_t inclusive_start_time = request.inclusive_start_time;
  int64_t use_time = request.end_time;

  auto reserved_allocations_it =
      reserved_allocations_for_alt_mem_colorings_.find(defining_position);
  if (reserved_allocations_it !=
      reserved_allocations_for_alt_mem_colorings_.end()) {
    for (std::unique_ptr<ReservedAllocation>& reserved_allocation_ptr :
         reserved_allocations_it->second) {
      if (inclusive_start_time == definition_time &&
          reserved_allocation_ptr->start_time() == definition_time) {
        request.require_start_colored_in_alternate_memmory = true;
      }
      if (reserved_allocation_ptr->end_time() == use_time) {
        request.require_end_colored_in_alternate_memory = true;
      }
    }
  }
}

void MsaAlgorithm::UpdateRequestWithDefaultMemoryColoringRequirements(
    AllocationRequest& request) {
  if (!request.allocation_value) {
    return;
  }
  const HloPosition& defining_position =
      request.allocation_value->defining_position();
  int64_t definition_time =
      hlo_live_range_.instruction_schedule().at(defining_position.instruction);

  int64_t inclusive_start_time = request.inclusive_start_time;
  int64_t use_time = request.end_time;

  auto default_memory_colorings_it =
      default_memory_coloring_requirements_.find(defining_position);
  if (default_memory_colorings_it !=
      default_memory_coloring_requirements_.end()) {
    for (int64_t coloring_time : default_memory_colorings_it->second) {
      if (inclusive_start_time == definition_time &&
          coloring_time == definition_time) {
        request.require_start_colored_in_default_memory = true;
      }
      if (coloring_time == use_time) {
        request.require_end_colored_in_default_memory = true;
      }
    }
  }
}

AllocationResult MsaAlgorithm::AllocateSegment(AllocationRequest& request) {
  auto allocation_sequence =
      request.allocation_value->mutable_allocation_sequence();
  // inclusive_start_time == end_time is a special case where the value is
  // consumed multiple times by the same instruction. We can just find the
  // previous allocation and use that allocation.
  if (request.inclusive_start_time == request.end_time) {
    Allocation* allocation = GetLiveAllocationAt(
        *request.allocation_value_to_update->mutable_allocation_sequence(),
        request.end_time);
    CHECK_NE(allocation, nullptr);
    allocation->AddUse(request.use->hlo_use);
    return AllocationResult::kSuccess;
  }

  const HloPosition& defining_position =
      request.allocation_value->defining_position();
  VLOG(2) << "Finding allocation for "
          << request.allocation_value->ToShortString() << " ["
          << request.inclusive_start_time << ", " << request.end_time
          << "] latest prefetch = " << request.latest_prefetch_time
          << " last use = " << request.allocation_value->uses().back().time
          << " use = " << request.use->hlo_use.ToString()
          << ". Size = " << request.size
          << ", def pos = " << defining_position.ToString();
  if (request.require_no_copy_alternate_mem_allocation) {
    VLOG(2) << "Requiring alternate memory allocation.";
  }
  CHECK_LE(request.inclusive_start_time, request.end_time);
  if (VLOG_IS_ON(3) && options_.cost_analysis) {
    const HloPosition& defining_position =
        request.allocation_value->defining_position();
    const HloUse& use = request.use->hlo_use;
    VLOG(3) << "Definition benefit = "
            << options_.cost_analysis->GetAlternateMemoryBenefit(
                   request.allocation_value->defining_position())
            << " use benefit = "
            << options_.cost_analysis->GetAlternateMemoryBenefit(
                   request.use->hlo_use);
    VLOG(3) << "Definition bytes accessed = "
            << options_.cost_analysis->OutputBytesAccessed(
                   *defining_position.instruction, defining_position.index)
            << ", use bytes accessed = "
            << options_.cost_analysis->OperandBytesAccessed(
                   *use.instruction, use.operand_number, use.operand_index);
  }

  // There could be a requirement to pin this buffer to default memory either
  // because it is a parameter or an output.  If the buffer is a parameter, then
  // we're allowed to prefetch. If the use expects the output to be in default
  // memory, we cannot prefetch it because if we did, it would be in alternate
  // memory instead.
  std::optional<RequiredMemoryAssignment> required_assignment_at_start =
      RequiredMemoryAssignmentAt(request.allocation_value->value(),
                                 request.inclusive_start_time);
  std::optional<MemorySpace> required_memory_space_at_start;
  if (required_assignment_at_start) {
    required_memory_space_at_start = required_assignment_at_start->memory_space;
  }
  // Find required assignment both for the use and its aliases. If they are both
  // non-nullopt, then make sure they require the same assignment.
  std::optional<RequiredMemoryAssignment> required_assignment_at_end =
      RequiredMemoryAssignmentAt(request.allocation_value_to_update->value(),
                                 request.end_time);
  std::optional<RequiredMemoryAssignment> aliased_required_assignment_at_end =
      AliasedRequiredAssignmentForUse(*request.use);
  if (required_assignment_at_end != aliased_required_assignment_at_end) {
    if (required_assignment_at_end == std::nullopt) {
      required_assignment_at_end = aliased_required_assignment_at_end;
    } else {
      CHECK(aliased_required_assignment_at_end == std::nullopt ||
            aliased_required_assignment_at_end->equals_ignoring_time(
                *required_assignment_at_end));
    }
  }
  std::optional<MemorySpace> required_memory_space_at_end;
  if (required_assignment_at_end) {
    required_memory_space_at_end = required_assignment_at_end->memory_space;
  }

  if (required_assignment_at_start) {
    bool needs_required_allocation = true;
    if (!allocation_sequence->empty()) {
      auto prev_allocation_it = std::find_if(
          allocation_sequence->rbegin(), allocation_sequence->rend(),
          [&](const auto& allocation) {
            return allocation->memory_space() == required_memory_space_at_start;
          });
      if (prev_allocation_it != allocation_sequence->rend()) {
        (*prev_allocation_it)->Extend(request.inclusive_start_time);
        needs_required_allocation = false;
      }
    }
    if (needs_required_allocation) {
      std::optional<Chunk> aliased_chunk = std::nullopt;
      if (required_assignment_at_start->memory_space ==
          MemorySpace::kAlternate) {
        aliased_chunk = Chunk::FromOffsetSize(
            required_assignment_at_start->offset->offset, request.size);
      }
      allocation_sequence->push_back(std::make_unique<PinnedAllocation>(
          defining_position, required_assignment_at_start->memory_space,
          aliased_chunk, request.inclusive_start_time,
          request.inclusive_start_time));
      if (required_assignment_at_start->memory_space ==
          MemorySpace::kAlternate) {
        CreateOrAddToAliasedOffset(*allocation_sequence->back(),
                                   required_assignment_at_start->offset);
      }
    }
  }

  VLOG(3)
      << "Required memory assignment at start: "
      << OptionalRequiredMemoryAssignmentToString(required_assignment_at_start)
      << "; required memory assignment at end: "
      << OptionalRequiredMemoryAssignmentToString(required_assignment_at_end);

  FreeAlternateMemoryColoringReservedAllocations(request);

  AllocationResult allocation_result = AllocationResult::kSuccess;
  // First try keeping the allocation entirely in the alternate memory.
  if (!request.require_start_colored_in_default_memory &&
      !request.require_end_colored_in_default_memory &&
      required_memory_space_at_start != MemorySpace::kDefault &&
      required_memory_space_at_end != MemorySpace::kDefault &&
      request.allow_no_copy_alternate_mem_allocation &&
      !request.require_copy_allocation) {
    CheckAndUpdateForDualLiveAllocationValues(required_assignment_at_start,
                                              request);
    allocation_result = AllocateInAlternateMemoryNoCopy(request);
    if (allocation_result == AllocationResult::kSuccess) {
      return AllocationResult::kSuccess;
    }
    // If we required alternate memory allocation, return on failure.
    if (request.require_no_copy_alternate_mem_allocation) {
      return allocation_result;
    }
  }

  CHECK(!request.require_no_copy_alternate_mem_allocation);

  if (request.require_start_colored_in_alternate_memmory) {
    // Since no-copy-allocation failed, continuous allocation is not possible in
    // the alternate memory.
    CHECK(!request.allocation_value->requires_contiguous_allocation());
    allocation_result = ForceAlternateMemoryAllocationForMinTime(request);
    // Allocation for short live range should succeed since we released a
    // reserved chunk from the interval tree.
    CHECK(allocation_result == AllocationResult::kSuccess);
  }

  auto prev_allocation_it = allocation_sequence->rbegin();
  // Find a previous allocation that is in the default memory space (not
  // necessarily the very last allocation).
  auto prev_allocation_in_default_mem_it =
      std::find_if(allocation_sequence->rbegin(), allocation_sequence->rend(),
                   [&](const auto& allocation) {
                     return allocation->memory_space() == MemorySpace::kDefault;
                   });

  if (!request.allocation_value->requires_contiguous_allocation()) {
    if (prev_allocation_in_default_mem_it == allocation_sequence->rend() &&
        prev_allocation_it != allocation_sequence->rend() &&
        (*prev_allocation_it)->memory_space() == MemorySpace::kAlternate &&
        (*prev_allocation_it)->defining_position() == defining_position) {
      // If there was an allocation for this HloValue that was in the alternate
      // memory space, we also need to perform an eviction.
      AllocationResult eviction_result = Evict(
          request,
          /*force_evict=*/request.require_start_colored_in_alternate_memmory);
      if (eviction_result != AllocationResult::kSuccess) {
        // A non-success eviction requires us to uncommit previous allocations.
        return result_mark(AllocationResult::kFailRequiresUncommit,
                           eviction_result);
      }
      prev_allocation_in_default_mem_it = allocation_sequence->rbegin();
    } else if (prev_allocation_in_default_mem_it ==
               allocation_sequence->rend()) {
      allocation_sequence->push_back(std::make_unique<PinnedAllocation>(
          defining_position, MemorySpace::kDefault,
          /*chunk=*/std::nullopt, request.inclusive_start_time,
          request.end_time));
      prev_allocation_in_default_mem_it = allocation_sequence->rbegin();
    }
  } else if (prev_allocation_in_default_mem_it == allocation_sequence->rend()) {
    VLOG(3) << "Allocation requires contiguous allocation, but it wasn't "
               "possible to find one.";
    CHECK(!request.require_start_colored_in_default_memory);
    CHECK(!request.require_end_colored_in_default_memory);
    return result_mark(AllocationResult::kFailRequiresUncommit,
                       allocation_result);
  }

  CHECK(prev_allocation_in_default_mem_it != allocation_sequence->rend());
  CHECK((*prev_allocation_in_default_mem_it)->memory_space() ==
        MemorySpace::kDefault);

  // If the allocation value requires a contiguous allocation but has a memory
  // space mismatch between the start and end required assignments, then we need
  // to uncommit.
  if (request.allocation_value->requires_contiguous_allocation() &&
      required_memory_space_at_start.has_value() &&
      required_memory_space_at_end.has_value() &&
      required_memory_space_at_start != required_memory_space_at_end) {
    VLOG(3) << "Allocation requires contiguous allocation but has memory space "
               "mismatch.";
    return result_mark(AllocationResult::kFailRequiresUncommit,
                       allocation_result);
  }

  // Finally, try to prefetch the buffer into alternate memory.
  if (request.allow_prefetch &&
      !request.allocation_value->requires_contiguous_allocation() &&
      !request.only_extend_existing_allocation &&
      required_memory_space_at_end != MemorySpace::kDefault &&
      !request.require_end_colored_in_default_memory) {
    if (request.require_copy_allocation && !request.required_copy_for_slice) {
      auto it = std::find_if(
          allocation_sequence->begin(), allocation_sequence->end(),
          [&](const std::unique_ptr<xla::memory_space_assignment::Allocation>&
                  allocation_ptr) {
            auto copy_allocation =
                dynamic_cast<const CopyAllocation*>(allocation_ptr.get());
            return copy_allocation &&
                   copy_allocation->copy_done_schedule_before() <=
                       request.required_copy_allocation_latest_time;
          });
      if (it == allocation_sequence->end()) {
        int64_t latest_prefetch_time =
            std::min(request.latest_prefetch_time,
                     request.required_copy_allocation_latest_time);
        VLOG(3) << "Updating the latest prefetch time from "
                << request.latest_prefetch_time << " to "
                << latest_prefetch_time
                << ", because this use requires a copy allocation before "
                << request.required_copy_allocation_latest_time;
        request.latest_prefetch_time = latest_prefetch_time;
      }
    }
    AllocationResult prefetch_result =
        Prefetch(request, **prev_allocation_in_default_mem_it, nullptr,
                 /*force_prefetch=*/
                 request.require_end_colored_in_alternate_memory);
    if (prefetch_result == AllocationResult::kSuccess) {
      if (request.preferred_prefetch_time) {
        // Warn if the prefetch time picked doesn't match the preferred prefetch
        // time.
        CHECK(!request.allocation_value_to_update->allocation_sequence()
                   ->empty());
        const Allocation* allocation =
            request.allocation_value_to_update->allocation_sequence()
                ->back()
                .get();
        int64_t prefetch_time = 0;
        if (allocation->is_copy_allocation()) {
          prefetch_time = static_cast<const CopyAllocation*>(allocation)
                              ->copy_start_schedule_after();
        } else if (allocation->is_sliced_copy_allocation()) {
          prefetch_time = static_cast<const SlicedCopyAllocation*>(allocation)
                              ->slice_details_sorted_by_start_time()
                              .front()
                              .copy_start_after_time;
        } else {
          LOG(FATAL) << "Prefetch allocation are expected to be "
                        "CopyAllocations or SlicedCopyAllocations.";
        }
        if (prefetch_time != *request.preferred_prefetch_time) {
          VLOG(1) << "Scheduled prefetch time (" << prefetch_time
                  << ") doesn't match the preferred prefetch time ("
                  << *request.preferred_prefetch_time
                  << "): " << request.use->hlo_use.ToString();
        }
      }
      return AllocationResult::kSuccess;
    }
    // Warn if there was a preferred prefetch time but we couldn't actually
    // prefetch.
    if (request.preferred_prefetch_time) {
      VLOG(1) << "The request has a preferred prefetch time ("
              << *request.preferred_prefetch_time
              << ") which could not be satisfied: "
              << request.use->hlo_use.ToString();
    }
    result_mark(prefetch_result, allocation_result);
  }

  CHECK(!request.require_end_colored_in_alternate_memory);

  // If the end assignment was required to be in alternate memory but that
  // wasn't possible, then this allocation is invalid.
  if (required_memory_space_at_end == MemorySpace::kAlternate) {
    return result_mark(AllocationResult::kFailRequiresUncommit,
                       allocation_result);
  }

  // If the start assignment was required to be in alternate memory and the
  // buffer needs a contiguous assignment, we couldn't satisfy this requirement
  // and must abort.
  if (required_memory_space_at_start == MemorySpace::kAlternate &&
      request.allocation_value->requires_contiguous_allocation()) {
    return result_mark(AllocationResult::kFailRequiresUncommit,
                       allocation_result);
  }

  // If a copy wasn't inserted, then add this use to the latest allocation in
  // default memory.
  (*prev_allocation_in_default_mem_it)->Extend(request.end_time);
  (*prev_allocation_in_default_mem_it)->AddUse(request.use->hlo_use);

  // If the buffer is placed in default memory, we can try window prefetching
  // it, which will try to prefetch only a window worth of data to alternate
  // memory.
  WindowPrefetch(request, **prev_allocation_in_default_mem_it);
  return allocation_result;
}

void MsaAlgorithm::AddAsyncCopyForWindowPrefetch(
    Allocation& prev_allocation, HloUse use, const Chunk& chunk,
    int64_t exclusive_start_time, int64_t inclusive_end_time,
    AllocationSequence* allocations, AliasedOffset* aliased_offset,
    float resource, const WindowPrefetchedAllocation::Options& options) {
  allocations->push_back(std::make_unique<WindowPrefetchedAllocation>(
      prev_allocation, use, chunk, exclusive_start_time, inclusive_end_time,
      options));

  RegisterAsyncCopy(MemorySpace::kAlternate, exclusive_start_time,
                    inclusive_end_time, allocations, aliased_offset, resource,
                    /*cross_program_prefetch_index=*/std::nullopt);
}

void MsaAlgorithm::AddAsyncCopyOrOtherMemOp(
    Allocation& prev_allocation, MemorySpace memory_space,
    std::optional<Chunk> chunk, int64_t exclusive_start_time, int64_t end_time,
    int64_t copy_done_schedule_before_time, AllocationSequence* allocations,
    AliasedOffset* aliased_offset, float resource,
    std::optional<int> cross_program_prefetch_index,
    HloInstruction* sync_mem_op) {
  VLOG(3) << "Copy to "
          << (memory_space == MemorySpace::kDefault ? "default" : "alternate")
          << " memory in (" << exclusive_start_time << ", "
          << copy_done_schedule_before_time << "), keeping until " << end_time
          << ", estimated copy resource is " << resource;
  CHECK_LT(exclusive_start_time, copy_done_schedule_before_time);

  allocations->push_back(std::make_unique<CopyAllocation>(
      prev_allocation, memory_space, chunk, exclusive_start_time,
      copy_done_schedule_before_time, end_time, cross_program_prefetch_index,
      sync_mem_op));

  RegisterAsyncCopy(memory_space, exclusive_start_time,
                    copy_done_schedule_before_time, allocations, aliased_offset,
                    resource, cross_program_prefetch_index);
}

void MsaAlgorithm::RegisterAsyncCopy(
    MemorySpace memory_space, int64_t exclusive_start_time,
    int64_t copy_done_schedule_before_time, AllocationSequence* allocations,
    AliasedOffset* aliased_offset, float resource,
    std::optional<int> cross_program_prefetch_index) {
  // Register the additional async copy with the interval tree to keep track of
  // the limit at any given time.
  pending_async_copies_.push_back({exclusive_start_time,
                                   copy_done_schedule_before_time, resource,
                                   memory_space, next_async_copy_id_++});
  if (memory_space == MemorySpace::kAlternate) {
    prefetch_interval_tree_.Add(
        /*start=*/
        ExclusiveToInclusiveStartTime(exclusive_start_time),
        copy_done_schedule_before_time, kDummyChunk);
    prefetch_async_copy_resource_.AddCopy(pending_async_copies_.back());
    if (options_.enforce_prefetch_fifo_order) {
      async_copy_ordering_.AddCopy(pending_async_copies_.back());
    }
    CreateOrAddToAliasedOffset(*allocations->back(), aliased_offset);
  } else {
    eviction_interval_tree_.Add(
        /*start=*/
        ExclusiveToInclusiveStartTime(exclusive_start_time),
        copy_done_schedule_before_time, kDummyChunk);
    eviction_async_copy_resource_.AddCopy(pending_async_copies_.back());
  }
}

namespace {

// Computes a string that can be used for logging/debugging. For each slice, the
// string includes:
// - When the slice starts
// - When the slice copy must complete
// - When the allocation for the slice ends
// - An estimation of how much copy resource the slice consumes
std::string SliceTimesAndCopyResourcesToString(
    const std::vector<SliceDecision>& slice_decisions, int64_t prefetch_end,
    int64_t allocation_end) {
  std::vector<std::string> slice_strings;
  slice_strings.reserve(slice_decisions.size());

  for (const auto& slice_decision : slice_decisions) {
    std::vector<std::string> details;
    details.push_back(absl::StrCat(slice_decision.exclusive_start_time));
    details.push_back(absl::StrCat(prefetch_end));
    details.push_back(absl::StrCat(allocation_end));
    details.push_back(absl::StrCat(slice_decision.copy_resource_consumed));

    slice_strings.push_back(
        absl::StrCat("(", absl::StrJoin(details, ", "), ")"));
  }

  return absl::StrCat(
      "Slices(copy_start_time, copy_done_by_time, allocation_end, "
      "estimated_copy_resource) = [",
      absl::StrJoin(slice_strings, ", "), "]");
}

}  // namespace

void MsaAlgorithm::AddAsyncSlicesForPrefetch(
    const Allocation& prev_allocation, AllocationSequence* allocations,
    AliasedOffset* aliased_offset,
    const std::vector<SliceDecision>& slice_decisions_sorted_by_start_time,
    int64_t prefetch_end_time, int64_t allocation_end_time,
    HloInstruction* sync_mem_op) {
  VLOG(3) << "Sliced copy to alternate memory. "
          << SliceTimesAndCopyResourcesToString(
                 slice_decisions_sorted_by_start_time, prefetch_end_time,
                 allocation_end_time);
  CHECK(absl::c_all_of(
      slice_decisions_sorted_by_start_time, [&](const auto& slice_decision) {
        return slice_decision.exclusive_start_time < prefetch_end_time;
      }));

  allocations->push_back(std::make_unique<SlicedCopyAllocation>(
      prev_allocation, MemorySpace::kAlternate,
      slice_decisions_sorted_by_start_time, prefetch_end_time,
      allocation_end_time, options_.sliced_prefetch_options,
      options_.get_equivalent_s8_shape_fn, sync_mem_op));

  // Register the additional async copy with the interval tree to keep track of
  // the limit at any given time.
  for (const auto& slice_decision : slice_decisions_sorted_by_start_time) {
    pending_async_copies_.push_back(
        {slice_decision.exclusive_start_time, prefetch_end_time,
         slice_decision.copy_resource_consumed, MemorySpace::kAlternate,
         next_async_copy_id_++});
    prefetch_interval_tree_.Add(slice_decision.exclusive_start_time,
                                prefetch_end_time, kDummyChunk);
    prefetch_async_copy_resource_.AddCopy(pending_async_copies_.back());
    if (options_.enforce_prefetch_fifo_order) {
      async_copy_ordering_.AddCopy(pending_async_copies_.back());
    }
  }
  CreateOrAddToAliasedOffset(*allocations->back(), aliased_offset);
}

bool MsaAlgorithm::ViolatesMaximumOutstandingAsyncCopies(
    int64_t inclusive_start_time, int64_t end_time, bool is_prefetch,
    int64_t extra_async_copy_limit, int64_t num_additional_copies) const {
  if (options_.max_outstanding_prefetches < 0 && is_prefetch) {
    return false;
  }
  if (options_.max_outstanding_evictions < 0 && !is_prefetch) {
    return false;
  }

  // Count the prefetches/evictions in the interval tree for the given interval.
  if (is_prefetch) {
    int64_t num_prefetches = prefetch_interval_tree_.NumChunksOverlappingInTime(
                                 inclusive_start_time, end_time) +
                             num_additional_copies;
    return num_prefetches >=
           options_.max_outstanding_prefetches + extra_async_copy_limit;
  } else {
    int64_t num_evictions = eviction_interval_tree_.NumChunksOverlappingInTime(
                                inclusive_start_time, end_time) +
                            num_additional_copies;
    return num_evictions >=
           options_.max_outstanding_evictions + extra_async_copy_limit;
  }
}

AllocationResult MsaAlgorithm::ForceAlternateMemoryAllocationForMinTime(
    const AllocationRequest& request) {
  CHECK(request.allocation_value->allocation_sequence()->empty());

  MsaBufferInterval alternate_mem_interval = MsaBufferInterval{
      /*buffer=*/request.allocation_value->value(),
      /*size=*/request.size,
      /*start=*/request.inclusive_start_time,
      /*end=*/request.inclusive_start_time,
      /*colocations=*/{},
      /*need_allocation=*/true,
  };

  Chunk chunk_candidate = FindChunkCandidate(alternate_mem_interval);

  if (chunk_candidate.chunk_end() > available_heap_size()) {
    return AllocationResult::kFailOutOfMemory;
  }

  const HloPosition& defining_position =
      request.allocation_value->defining_position();
  request.allocation_value->mutable_allocation_sequence()->push_back(
      std::make_unique<PinnedAllocation>(
          defining_position, MemorySpace::kAlternate, chunk_candidate,
          alternate_mem_interval.start, alternate_mem_interval.end));
  // Since we did not use request.preferred_offset, we pass nullptr to
  // CreateOrAddToAliasedOffset.
  CreateOrAddToAliasedOffset(
      *request.allocation_value->allocation_sequence()->back(),
      /*aliased_offset=*/nullptr);
  return AllocationResult::kSuccess;
}

AllocationResult MsaAlgorithm::AllocateInAlternateMemoryNoCopy(
    const AllocationRequest& request) {
  Allocation* prev_allocation = nullptr;
  bool can_eliminate_copy = false;
  if (request.allocation_value->allocation_sequence()->empty()) {
    // There hasn't been any allocations for this interval so far. We can
    // eliminate copy if the value can be placed in the alternate memory.
    can_eliminate_copy = options_.is_allowed_in_alternate_mem_fn(
        *request.allocation_value->value());
  } else {
    // If there has been a previous allocation, we can eliminate the copy if the
    // previous allocation was also in the alternate memory.
    prev_allocation =
        request.allocation_value->allocation_sequence()->back().get();
    can_eliminate_copy =
        (prev_allocation->memory_space() == MemorySpace::kAlternate &&
         !prev_allocation->is_window_prefetched_allocation());
  }

  if (!can_eliminate_copy) {
    VLOG(3) << "Can't eliminate copy.";
    return AllocationResult::kFailPrevAllocationNotInAlternateMem;
  }

  const HloPosition& defining_position =
      request.allocation_value->defining_position();
  // If prefer_no_copy_alternate_mem_allocation is true, bypass the live range
  // duration checks.
  if (!request.require_no_copy_alternate_mem_allocation &&
      !request.prefer_no_copy_alternate_mem_allocation &&
      !options_.prefetch_interval_picker->CanAllocateInAlternateMemoryNoCopy(
          defining_position.shape(), request.inclusive_start_time,
          request.end_time)) {
    VLOG(3) << "Live range is too long.";
    return AllocationResult::kFailLiveRangeTooLong;
  }

  MsaBufferInterval alternate_mem_interval;
  alternate_mem_interval.buffer = request.allocation_value->value();
  alternate_mem_interval.size = request.size;
  alternate_mem_interval.end = request.end_time;
  alternate_mem_interval.start = request.inclusive_start_time;

  // Prefer the offset that was previously used for the previous allocation.
  AliasedOffset* preferred_offset = nullptr;
  if (prev_allocation != nullptr) {
    preferred_offset = GetAliasedOffset(*prev_allocation);
    // If there is a previous allocation, set the start time one after the end
    // of the previous allocation's end.
    alternate_mem_interval.start = prev_allocation->end_time() + 1;
    if (request.no_copy_chunk_inclusive_start_time.has_value()) {
      alternate_mem_interval.start =
          *request.no_copy_chunk_inclusive_start_time;
    }
  }

  if (request.preferred_offset) {
    // If there is a preferred offset provided in the request and if it doesn't
    // match the previous allocation, this request cannot be satisfied.
    if (preferred_offset && request.preferred_offset != preferred_offset) {
      VLOG(3) << "Cannot perform no-copy allocation due to mismatch: "
                 "preferred_offset = "
              << preferred_offset->offset << ", request.preferred_offset = "
              << request.preferred_offset->offset;
      return AllocationResult::kFailConflictingPreferredOffsets;
    }
    preferred_offset = request.preferred_offset;
  }

  VLOG(3) << "We can eliminate copy to alternate memory. Preferred offset = "
          << (preferred_offset ? preferred_offset->offset : -1);
  // In case there are additional uses after this use, we rely on the last use
  // time to try to reserve a chunk in the heap simulator. This is to prevent
  // the following scenario:
  //
  //                            +-------+
  //                           /         \
  //                   Producer--->Use1   +-->Use2
  //                       +---------+---------+
  // New buffer:           |         |         |
  //                       +---------+---------+
  //
  //                                     +-----------+
  // Current heap:                       | offset: 0 |
  //           --------------------------+-----------+------
  //
  // Because we allocate buffers greedily, Producer to Use1 segment first, and
  // then Use1 to Use2 segment, it is possible to allocate the first segment at
  // an offset that is available for the first segment (e.g. offset 0) but not
  // for the entire live range. This can result in unnecessary copies. By using
  // the last use time, we try to find an allocation that is available for the
  // entire Producer to Use2 range.
  std::optional<Chunk> chunk_candidate = FindBestChunkCandidate(
      request, preferred_offset, &alternate_mem_interval);
  // Check if the new heap size fits within limits. Also ensure if a
  // preferred offset was provided, that offset was used.
  if (chunk_candidate) {
    VLOG(3) << "Keep the buffer in alternate memory. Offset = "
            << chunk_candidate->offset << ", size = " << chunk_candidate->size
            << ", heap_size = " << result_.UpdatedHeapSize(*chunk_candidate)
            << ", prefetch picker = "
            << options_.prefetch_interval_picker->ToNoCopyDebugString(
                   defining_position.shape(),
                   /*start_time=*/
                   InclusiveToExclusiveStartTime(request.inclusive_start_time),
                   request.end_time);
    AddToPendingChunks(alternate_mem_interval, *chunk_candidate);

    // If there was a previous allocation, the buffer location is the
    // same as the previous. Otherwise, it is the operand.
    if (prev_allocation != nullptr &&
        (prev_allocation->is_copy_like_allocation() ||
         prev_allocation->defining_position() == defining_position)) {
      prev_allocation->Extend(request.end_time);
    } else {
      request.allocation_value->mutable_allocation_sequence()->push_back(
          std::make_unique<PinnedAllocation>(
              defining_position, MemorySpace::kAlternate, chunk_candidate,
              request.inclusive_start_time, request.end_time));
      CreateOrAddToAliasedOffset(
          *request.allocation_value->allocation_sequence()->back(),
          preferred_offset);
    }
    if (!request.only_extend_existing_allocation) {
      request.allocation_value->allocation_sequence()->back()->AddUse(
          request.use->hlo_use);
    }
    return AllocationResult::kSuccess;
  }
  if (request.prefer_no_copy_alternate_mem_allocation) {
    VLOG(1) << "Preferred no-copy allocation, but this was not possible: "
            << request.use->hlo_use.ToString();
  }
  return AllocationResult::kFailOutOfMemory;
}

AllocationResult MsaAlgorithm::Evict(const AllocationRequest& request,
                                     bool force_evict) {
  CHECK_GT(request.allocation_value->allocation_sequence()->size(), 0);
  Allocation* prev_allocation =
      request.allocation_value->allocation_sequence()->back().get();
  // We do not ever expect an Evict() to be immediately proceeded by a prefetch.
  // If that case ever occurs, the eviction_exclusive_start_time below will be
  // calculated incorrectly, as it will need to come after the prefetch finishes
  // coping data.
  CHECK(!prev_allocation->is_copy_like_allocation())
      << "Evict has been given copy-like previous allocation.\nEvict "
         "candidate:\n"
      << request.allocation_value->ToString() << "\nPrevious allocation:\n"
      << prev_allocation->ToString();

  // The previous allocation's inclusive start time is the eviction's exclusive
  // start time to ensure that the value is created before we start copying
  // back to default memory.
  int64_t eviction_exclusive_start_time = prev_allocation->start_time();
  int64_t eviction_end_time = prev_allocation->end_time();
  CHECK(eviction_exclusive_start_time <= eviction_end_time);

  int64_t preferred_eviction_end_time = std::max(
      options_.prefetch_interval_picker->PreferredEvictionEndTime(
          request.allocation_value_to_update->defining_position().shape(),
          eviction_exclusive_start_time, request.end_time),
      eviction_end_time);
  // Evictions must complete by the time of this use.
  preferred_eviction_end_time =
      std::min(preferred_eviction_end_time, request.latest_prefetch_time);

  MsaBufferInterval eviction_mem_interval;
  eviction_mem_interval.buffer = request.allocation_value->value();
  // When replacing an sync slice, the size of the original allocation_value
  // matters instead of the queuing_allocation_value
  // TODO(mehrdadk): separate the request size for src and dst
  // AllocationSequence
  eviction_mem_interval.size =
      std::max(request.allocation_value->size(), request.size);
  // Try to reserve a buffer from the end of the previous allocation to the
  // preferred eviction end time.
  eviction_mem_interval.start = eviction_end_time + 1;
  eviction_mem_interval.end = preferred_eviction_end_time;
  int64_t preferred_offset = prev_allocation->chunk().offset;
  VLOG(3) << "Considering eviction after" << eviction_exclusive_start_time
          << ", with preferred end time = " << eviction_mem_interval.end;

  for (; eviction_mem_interval.end > eviction_end_time;
       --eviction_mem_interval.end) {
    Chunk chunk_candidate =
        FindChunkCandidate(eviction_mem_interval, preferred_offset);
    if (chunk_candidate.offset == preferred_offset) {
      AddToPendingChunks(eviction_mem_interval, chunk_candidate);
      break;
    }
  }
  eviction_end_time = eviction_mem_interval.end;

  VLOG(3) << "Evicting buffer at " << prev_allocation->chunk().offset << " ("
          << eviction_exclusive_start_time << ", " << eviction_end_time << ")";

  float eviction_resource =
      options_.cost_analysis
          ? options_.cost_analysis->GetAsyncCopyElapsed(
                request.allocation_value->defining_position().shape())
          : 0.1;

  bool eviction_interval_too_short =
      (eviction_exclusive_start_time == eviction_end_time);
  bool eviction_violates_resource =
      !eviction_async_copy_resource_.HasEnoughResource(
          eviction_exclusive_start_time, eviction_end_time, eviction_resource);
  if (eviction_violates_resource) {
    // If we're in the last retry, set resource to 0.
    if (options_.prefetch_interval_picker->retry_number() ==
        options_.max_retries - 1) {
      VLOG(3) << "Violates resource in last retry, setting resource = 0";
      eviction_resource = 0;
    }
    eviction_violates_resource =
        !eviction_async_copy_resource_.HasEnoughResource(
            eviction_exclusive_start_time, eviction_end_time,
            eviction_resource);
  }
  bool eviction_violates_outstanding_copies =
      ViolatesMaximumOutstandingAsyncCopies(
          /*inclusive_start_time=*/ExclusiveToInclusiveStartTime(
              eviction_exclusive_start_time),
          eviction_end_time,
          /*is_prefetch=*/false);

  // See if this interval would violate the asynchronous copy limit.
  if (!eviction_interval_too_short && !eviction_violates_outstanding_copies &&
      !eviction_violates_resource) {
    prev_allocation->Extend(eviction_end_time);
    AddAsyncCopyOrOtherMemOp(
        *prev_allocation, MemorySpace::kDefault,
        /*chunk=*/std::nullopt, eviction_exclusive_start_time,
        prev_allocation->end_time(), eviction_end_time,
        request.allocation_value->mutable_allocation_sequence(),
        /*aliased_offset=*/nullptr, eviction_resource);
    return AllocationResult::kSuccess;
  }

  if (eviction_violates_outstanding_copies) {
    VLOG(3) << "This violates the maximum async copies.";
  } else if (eviction_violates_resource) {
    VLOG(3) << "This violates resource.";
  } else {
    VLOG(3) << "Eviction interval is too short ("
            << eviction_exclusive_start_time << ", " << eviction_end_time
            << ").";
  }

  if (force_evict) {
    VLOG(3) << "Forcing evicting.";
    AddAsyncCopyOrOtherMemOp(
        *prev_allocation, MemorySpace::kDefault,
        /*chunk=*/std::nullopt, prev_allocation->end_time() - 1,
        request.end_time, prev_allocation->end_time() + 1,
        request.allocation_value->mutable_allocation_sequence(),
        /*aliased_offset=*/nullptr, 0);
    return AllocationResult::kSuccess;
  }

  // If the eviction couldn't be scheduled, then fail. This buffer will be
  // kept in the default memory.
  VLOG(3) << "Bailing: Could not evict " << request.use->hlo_use.ToString()
          << " because we hit the limit of maximum asynchronous copies "
          << "between ("
          << hlo_live_range_.flattened_instruction_sequence()
                 .instructions()[eviction_exclusive_start_time]
          << ", "
          << hlo_live_range_.flattened_instruction_sequence()
                 .instructions()[eviction_end_time]
          << ")";
  return AllocationResult::kFailOutOfAsyncCopies;
}

int64_t MsaAlgorithm::FindPrefetchEndTime(
    const AllocationRequest& request, int64_t earliest_prefetch_time) const {
  return request.latest_prefetch_time;
}

namespace {

// A debugging/logging method for describing a sliced solution.
std::string DescribeSlicedBufferMove(
    const std::vector<SliceDecision>& slice_decisions,
    const MsaAlgorithm::HeapResult& heap_result,
    const MsaAlgorithm::Chunk& full_chunk,
    absl::string_view prefetch_picker_debug_string) {
  std::vector<std::string> slice_strings;
  slice_strings.reserve(slice_decisions.size());

  for (const auto& slice_decision : slice_decisions) {
    slice_strings.push_back(absl::StrCat(
        "(", slice_decision.exclusive_start_time, ", ",
        slice_decision.chunk.offset, ", ", slice_decision.chunk.size, ")"));
  }

  return absl::StrCat(
      "Moving buffer to alternate memory in slices. Slices(start_time, offset, "
      "size) = [",
      absl::StrJoin(slice_strings, ", "),
      "]. Heap size = ", heap_result.UpdatedHeapSize(full_chunk),
      ". Prefetch picker = ", prefetch_picker_debug_string);
}

}  // namespace

AllocationResult MsaAlgorithm::WindowPrefetch(
    const AllocationRequest& request,
    Allocation& prev_allocation_in_default_mem) {
  if (!options_.enable_window_prefetch) {
    return AllocationResult::kSuccess;
  }

  const HloUse use = request.use->hlo_use;
  VLOG(3) << "Considering window prefetch for use=" << use.ToString();

  // Get the window prefetch details for this use.
  WindowPrefetchDetail details =
      options_.window_prefetch_detail_fn(use.instruction);
  for (const WindowPrefetchDetail::WindowDetail& window : details.windows()) {
    if (window.operand() != use.operand_number) {
      continue;
    }

    // Construct the options needed for creating the window prefetch allocation.
    WindowPrefetchedAllocation::Options options;
    options.bytes = window.size();
    options.alternate_memory_space = options_.alternate_memory_space;
    options.notify_operand_appended_fn = options_.notify_operand_appended_fn;

    // Construct the request for prefetching the content of the window.
    AllocationRequest window_prefetch_request = request;
    window_prefetch_request.window_prefetch_options = &options;
    window_prefetch_request.size = window.size();
    int64_t end_time = request.end_time;
    window_prefetch_request.end_time = end_time;
    std::vector<int64_t> all_use_times = {end_time};
    window_prefetch_request.all_use_times = all_use_times;

    if (options_.window_prefetch_mode == WindowPrefetchMode::kWindowPrefetch) {
      // Window prefetch mode
      const Shape shape = ShapeUtil::MakeShape(U8, {window.size()});
      Prefetch(window_prefetch_request, prev_allocation_in_default_mem, &shape);
    } else {
      // Window exposure mode, we only need to find a chunk for the window
      // buffer.
      CHECK(options_.window_prefetch_mode ==
            WindowPrefetchMode::kWindowExposure);
      MsaBufferInterval alternate_mem_interval;
      alternate_mem_interval.buffer =
          request.allocation_value_to_update->value();
      alternate_mem_interval.size = window.size();
      alternate_mem_interval.end = end_time;
      alternate_mem_interval.start = end_time;
      std::optional<Chunk> candidate_chunk = FindBestChunkCandidate(
          window_prefetch_request, /*preferred_offset=*/nullptr,
          &alternate_mem_interval);
      if (candidate_chunk.has_value()) {
        AddToPendingChunks(alternate_mem_interval, *candidate_chunk);

        AllocationSequence* allocation_sequence =
            request.allocation_value_to_update->mutable_allocation_sequence();
        allocation_sequence->push_back(
            std::make_unique<WindowPrefetchedAllocation>(
                prev_allocation_in_default_mem, use, *candidate_chunk,
                end_time - 1, end_time, options));
        CreateOrAddToAliasedOffset(*allocation_sequence->back(),
                                   /*aliased_offset=*/nullptr);
        allocation_sequence->back()->AddUse(use);
      }
    }
  }
  return AllocationResult::kSuccess;
}

AllocationResult MsaAlgorithm::Prefetch(
    const AllocationRequest& request,
    Allocation& prev_allocation_in_default_mem, const Shape* shape,
    bool force_prefetch) {
  AllocationResult result = PrefetchWithResourceConstraints(
      request, prev_allocation_in_default_mem, shape);
  if (result == AllocationResult::kSuccess || !force_prefetch) {
    return result;
  }

  // We require the buffer in alternate memory for the use, but, prefetching
  // failed, so we force a prefetch and point the use to the prefetch
  // allocation.
  MsaBufferInterval alternate_mem_interval = MsaBufferInterval{
      /*buffer=*/request.allocation_value->value(),
      /*size=*/request.size,
      /*start=*/request.end_time,
      /*end=*/request.end_time,
      /*colocations=*/{},
      /*need_allocation=*/true,
  };

  Chunk chunk_candidate = FindChunkCandidate(alternate_mem_interval);

  if (chunk_candidate.chunk_end() > available_heap_size()) {
    return AllocationResult::kFailOutOfMemory;
  }

  AddAsyncCopyOrOtherMemOp(
      prev_allocation_in_default_mem, MemorySpace::kAlternate, chunk_candidate,
      request.end_time - 1, request.end_time, request.end_time,
      request.allocation_value->mutable_allocation_sequence(),
      /*aliased_offset=*/nullptr, 0.0f,
      /*cross_program_prefetch_index=*/std::nullopt,
      request.required_copy_allocation_for);
  request.allocation_value->allocation_sequence()->back()->AddUse(
      request.use->hlo_use);
  return AllocationResult::kSuccess;
}

AllocationResult MsaAlgorithm::PrefetchWithResourceConstraints(
    const AllocationRequest& request,
    Allocation& prev_allocation_in_default_mem, const Shape* shape) {
  // Try partially placing the buffer in the alternate space. The time that is
  // overlapped will be used to asynchronously copy the buffer from the
  // default memory to the alternate memory.
  //
  //                      start                 end
  //                      time                  time
  //                      X---------------------X
  // Alternate:                          +------+
  // Default:             +---------------------+
  //                                     ^      ^
  //                                   Copy    Copy
  //                                   Start   Done

  VLOG(5) << "Considering prefetch of "
          << request.allocation_value->defining_instruction()->ToString()
          << (request.preferred_offset
                  ? absl::StrCat(", with a preferred offset of ",
                                 request.preferred_offset->offset, ".")
                  : "");
  PrefetchContext context;
  context.request = &request;
  context.prev_allocation_in_default_mem = &prev_allocation_in_default_mem;
  // If the request has window prefetch options, it is called from window
  // prefetch.
  context.window_prefetch = (request.window_prefetch_options != nullptr);
  CHECK(!context.window_prefetch || options_.enable_window_prefetch);

  // Create a SliceProposal and WorkingIntervals.
  SetupPrefetchWorkingIntervalsAndSliceProposal(context);

  // Compute some additional preliminaries
  AllocationResult init_result = InitializePrefetchIntervalPicker(context);
  if (init_result != AllocationResult::kSuccess) {
    return init_result;
  }
  AllocationResult check_result = EnsureSomeSpatialPrefetchFitExists(context);
  if (check_result != AllocationResult::kSuccess) {
    return check_result;
  }
  const HloUse& use = request.use->hlo_use;
  if (shape != nullptr) {
    context.full_shape = shape;
  } else {
    context.full_shape = &ShapeUtil::GetSubshape(
        use.instruction->operand(use.operand_number)->shape(),
        use.operand_index);
  }
  // While uses might be allowed to have additional outstanding prefetches.
  context.extra_async_copy_limit =
      use.instruction->opcode() == HloOpcode::kWhile
          ? options_.while_use_extra_outstanding_prefetch_limit
          : 0;

  // If the request is for a sync mem op conversion to async, we may allow for
  // more async copies.
  if (context.request->require_copy_allocation) {
    context.extra_async_copy_limit +=
        options_.extend_async_copies_limit_for_sync_mem_op_conversion;
  }

  // Loop over potential prefetch starting times. At the selected start time, we
  // check if we have enough resources and memory for a sliced version of the
  // request and a non-sliced version of the request. We return the first sliced
  // solution that we find. We fallback to the first unsliced solution we find,
  // if we are unable to find a sliced solution.
  AllocationResult result = AllocationResult::kSuccess;
  while (!options_.prefetch_interval_picker->Done()) {
    // Get the prefetch start time from the interval picker.
    context.exclusive_prefetch_start_time =
        options_.prefetch_interval_picker->Next();
    CHECK_LT(context.exclusive_prefetch_start_time, context.prefetch_end_time);
    if (context.exclusive_out_of_mem_start.has_value() &&
        context.exclusive_prefetch_start_time <=
            *context.exclusive_out_of_mem_start) {
      VLOG(4) << "This would OOM (cached).";
      return AllocationResult::kFailOutOfMemory;
    }

    if (context.slice_proposal_collection) {
      VLOG(5) << "Trying sliced solution.";
      // Check if a sliced solution fits.
      AllocationResult sliced_result =
          CheckPrefetchFit(/*for_sliced_solution=*/true, context);
      if (sliced_result == AllocationResult::kSuccess) {
        // Break out of the loop and use the sliced solution.
        CHECK(context.sliced_solution);
        break;
      } else if (sliced_result !=
                 AllocationResult::kAllSlicesHaveTheSameStartTime) {
        result_mark(sliced_result, result);
      }
    }

    // If we don't already have an unsliced solution, check the current fit.
    if (!context.unsliced_solution) {
      VLOG(5) << "Trying unsliced solution.";
      AllocationResult unsliced_result =
          CheckPrefetchFit(/*for_sliced_solution=*/false, context);
      if (unsliced_result != AllocationResult::kSuccess) {
        result_mark(unsliced_result, result);
      } else if (!context.slice_proposal_collection) {
        // We found an unsliced solution and there is no slice proposal, so
        // break out of the loop and use the unsliced solution.
        CHECK(context.unsliced_solution);
        break;
      }
    }
  }

  // Check if we found any solutions.
  if (context.sliced_solution && !context.request->required_copy_for_slice) {
    CHECK(!context.sliced_solution->slices_for_pending_chunks.empty());
    VLOG(3) << DescribeSlicedBufferMove(
        context.sliced_solution->slice_decisions_sorted_by_start_time, result_,
        context.sliced_solution->slices_for_pending_chunks.back().second,
        context.sliced_solution->prefetch_picker_debug_string);

    for (const auto& interval_chunk_pair :
         context.sliced_solution->slices_for_pending_chunks) {
      AddToPendingChunks(interval_chunk_pair.first, interval_chunk_pair.second);
    }
    AddAsyncSlicesForPrefetch(
        *context.prev_allocation_in_default_mem,
        context.request->allocation_value_to_update
            ->mutable_allocation_sequence(),
        context.request->preferred_offset,
        context.sliced_solution->slice_decisions_sorted_by_start_time,
        context.prefetch_end_time, context.request->end_time,
        context.request->required_copy_allocation_for);
    context.request->allocation_value_to_update->allocation_sequence()
        ->back()
        ->AddUse(context.request->use->hlo_use);
    return AllocationResult::kSuccess;
  }
  if (context.unsliced_solution) {
    VLOG(3) << "Move the buffer to alternate memory after time "
            << InclusiveToExclusiveStartTime(
                   context.unsliced_solution_intervals.full.start)
            << ". Offset = "
            << context.unsliced_solution->chunk_candidate.offset
            << ", size = " << context.unsliced_solution->chunk_candidate.size
            << ", heap_size = "
            << result_.UpdatedHeapSize(
                   context.unsliced_solution->chunk_candidate)
            << ", prefetch picker = "
            << context.unsliced_solution->prefetch_picker_debug_string;
    AddToPendingChunks(context.unsliced_solution_intervals.full,
                       context.unsliced_solution->chunk_candidate);
    if (context.window_prefetch) {
      AddAsyncCopyForWindowPrefetch(
          *context.prev_allocation_in_default_mem, request.use->hlo_use,
          context.unsliced_solution->chunk_candidate,
          context.unsliced_solution_intervals.full.start - 1,
          context.prefetch_end_time,
          context.request->allocation_value_to_update
              ->mutable_allocation_sequence(),
          context.request->preferred_offset,
          context.unsliced_solution->prefetch_resource,
          *context.request->window_prefetch_options);
    } else {
      AddAsyncCopyOrOtherMemOp(
          *context.prev_allocation_in_default_mem, MemorySpace::kAlternate,
          context.unsliced_solution->chunk_candidate,
          context.unsliced_solution_intervals.full.start - 1,
          context.request->end_time, context.prefetch_end_time,
          context.request->allocation_value_to_update
              ->mutable_allocation_sequence(),
          context.request->preferred_offset,
          context.unsliced_solution->prefetch_resource,
          /*cross_program_prefetch_index=*/std::nullopt,
          context.request->required_copy_allocation_for);
      context.prev_allocation_in_default_mem->Extend(
          context.request->latest_prefetch_time);
    }

    request.allocation_value_to_update->allocation_sequence()->back()->AddUse(
        request.use->hlo_use);
    return AllocationResult::kSuccess;
  }

  // If we didn't consider any prefetch intervals, then the live range was too
  // short.
  return (result == AllocationResult::kSuccess
              ? AllocationResult::kFailLiveRangeTooShort
              : result);
}

void MsaAlgorithm::GenerateSliceProposal(PrefetchContext& context) const {
  if (context.request->required_copy_for_slice) {
    VLOG(5) << "Not slicing " << context.request->use->hlo_use
            << " because slicing a slice instruction is not supported yet.";
    return;
  }

  if (options_.sliced_prefetch_options.max_slices() < 2) {
    return;
  }
  auto log_prefix = [&]() {
    return absl::StrCat(
        "Slice request(options = ",
        options_.sliced_prefetch_options.ShortDebugString(), "; shape = ",
        context.prev_allocation_in_default_mem->defining_position()
            .shape()
            .ToString(),
        ")");
  };

  if (context.request->size < options_.sliced_prefetch_options.min_bytes()) {
    VLOG(5) << "Not slicing " << log_prefix() << " because the request size "
            << context.request->size
            << " is smaller than the min configured size of "
            << options_.sliced_prefetch_options.min_bytes();
    return;
  }

  auto status_or_proposal = options_.propose_slice_fn(
      context.prev_allocation_in_default_mem->defining_position().shape(),
      options_.sliced_prefetch_options);
  if (!status_or_proposal.ok()) {
    VLOG(2) << log_prefix() << " failed: " << status_or_proposal.status();
    return;
  }

  if (status_or_proposal.value().size() < 2) {
    VLOG(2) << log_prefix() << ". No slices proposed.";
    return;
  }

  VLOG(6) << log_prefix() << ". Slice proposal = ["
          << absl::StrJoin(status_or_proposal.value(), ", ",
                           [](std::string* out, const SliceProposal& proposal) {
                             absl::StrAppend(out, proposal.ToString());
                           })
          << "]";

  context.slice_proposal_collection = std::move(status_or_proposal.value());
}

void MsaAlgorithm::SetupPrefetchWorkingIntervalsAndSliceProposal(
    PrefetchContext& context) const {
  // Setup the full WorkingIntervals for the sliced and unsliced solutions.
  // Future code will adjust the start and end times.
  context.sliced_solution_intervals.full = MsaBufferInterval{
      context.request->allocation_value_to_update->value(),
      /*size=*/context.request->size,
      /*start=*/-1,
      /*end=*/context.request->end_time,
      /*colocations=*/{},
      /*need_allocation=*/true,
  };
  context.unsliced_solution_intervals.full =
      context.sliced_solution_intervals.full;

  // Attempt to generate a slice proposal.
  if (!context.window_prefetch) {
    GenerateSliceProposal(context);
  }

  // Setup the full SlicedBufferIntervals for the sliced and unsliced solutions.
  // If there is no slice proposal, we will not try a sliced solution. In such a
  // case, we do not populate context.sliced_solution_intervals.
  if (context.slice_proposal_collection) {
    context.sliced_solution_intervals.sliced =
        std::make_unique<SlicedBufferInterval>(
            SlicedBufferInterval::CreateMutableInterval(
                context.sliced_solution_intervals.full));
    std::vector<int64_t> sizes;
    sizes.reserve(context.slice_proposal_collection->size());
    for (const SliceProposal& single_slice_proposal :
         *context.slice_proposal_collection) {
      sizes.push_back(single_slice_proposal.slice_size);
    }
    context.sliced_solution_intervals.sliced->Slice(sizes);
  }
  context.unsliced_solution_intervals.sliced =
      std::make_unique<SlicedBufferInterval>(
          SlicedBufferInterval::CreateMutableInterval(
              context.unsliced_solution_intervals.full));
}

AllocationResult MsaAlgorithm::InitializePrefetchIntervalPicker(
    PrefetchContext& context) {
  int64_t earliest_exclusive_prefetch_time =
      context.prev_allocation_in_default_mem->earliest_available_time();
  if (context.request->earliest_prefetch_time) {
    earliest_exclusive_prefetch_time =
        std::max(earliest_exclusive_prefetch_time,
                 *context.request->earliest_prefetch_time);
  }
  context.prefetch_end_time =
      FindPrefetchEndTime(*context.request, earliest_exclusive_prefetch_time);

  // As a compile time optimization, use the peak memory usage to filter out
  // allocation times that would push us to OOM.
  std::optional<int> earliest_exclusive_non_oom_prefetch_time =
      FindEarliestExclusiveTimeToSatisfyPeakMemory(
          earliest_exclusive_prefetch_time, context.prefetch_end_time,
          context.request->size);
  if (!earliest_exclusive_non_oom_prefetch_time) {
    VLOG(3) << "Any prefetch in range (" << earliest_exclusive_prefetch_time
            << ", " << context.prefetch_end_time << ") for size "
            << context.request->size << " would go out of memory.";
    return AllocationResult::kFailOutOfMemory;
  }
  if (!context.slice_proposal_collection) {
    // We can only perform this optimization if we are not slicing.
    // earliest_non_oom_prefetch_time lets us know the first time the entire
    // buffer will fit, but we may be able to start slices before that time. So,
    // we leave earliest_prefetch_time at its initial value.
    VLOG(4) << "After peak memory check, prefetch range is ("
            << *earliest_exclusive_non_oom_prefetch_time << ", "
            << context.prefetch_end_time
            << "). Original earliest prefetch time is "
            << earliest_exclusive_prefetch_time;
    earliest_exclusive_prefetch_time =
        *earliest_exclusive_non_oom_prefetch_time;
  }
  std::optional<int64_t> preferred_prefetch_time =
      context.request->preferred_prefetch_time;
  if (preferred_prefetch_time) {
    preferred_prefetch_time =
        std::max(*preferred_prefetch_time, earliest_exclusive_prefetch_time);
  }
  options_.prefetch_interval_picker->Begin(
      context.request->use->hlo_use, earliest_exclusive_prefetch_time,
      context.prefetch_end_time, preferred_prefetch_time);
  VLOG(3) << "Trying prefetch picker = "
          << options_.prefetch_interval_picker->ToDebugString();

  return AllocationResult::kSuccess;
}

AllocationResult MsaAlgorithm::EnsureSomeSpatialPrefetchFitExists(
    PrefetchContext& context) const {
  SlicedBufferInterval* interval =
      (context.slice_proposal_collection
           ? context.sliced_solution_intervals.sliced.get()
           : context.unsliced_solution_intervals.sliced.get());

  // Note, UpdateInclusiveSliceStartTimes() will correctly update start times
  // for both sliced and unsliced solutions.
  interval->UpdateExclusiveSliceStartTimes(
      std::vector<int64_t>(interval->num_slices(),
                           options_.prefetch_interval_picker->latest_time()));
  std::vector<Chunk> chunk_candidates = FindBestChunkCandidates(
      *context.request, context.request->preferred_offset, interval);
  if (chunk_candidates.empty()) {
    VLOG(3) << "The latest prefetch (" << interval->full_buffer_interval().start
            << ", " << context.request->end_time
            << ") cannot find valid chunks. Giving up.";
    return AllocationResult::kFailOutOfMemory;
  }

  return AllocationResult::kSuccess;
}

namespace {

// Returns the copy resources needed for the specified slice proposal
// collection, in descending order.
std::vector<float> GetCopyResourcesSortedDescending(
    const Options& options,
    const SliceProposalCollection& slice_proposal_collection) {
  std::vector<float> copy_resources;
  copy_resources.reserve(slice_proposal_collection.size());
  for (const SliceProposal& proposal : slice_proposal_collection) {
    copy_resources.push_back(
        CopyResourceForShape(options, proposal.slice_shape));
  }
  absl::c_sort(copy_resources);
  return copy_resources;
}

// Returns true if we would have enough async copy resources to copy each
// specified slice.
bool DoWeHaveEnoughCopyResource(
    const std::vector<int64_t>& slice_start_times, int64_t prefetch_end_time,
    const std::vector<float>& copy_resource_per_slice,
    AsynchronousCopyResource& async_copy_resource) {
  CHECK_EQ(slice_start_times.size(), copy_resource_per_slice.size());

  std::vector<AsynchronousCopyResource::ResourceSpec> specs;
  specs.reserve(slice_start_times.size());

  // Note, the HasEnoughResourceMultiCheck() below is sensitive to this order.
  // The specs must be in slice start time order because that's the order
  // they'll be added to prefetch_async_copy_resource_ in
  // AddAsyncSlicesForPrefetch(), if the solution is selected.
  static const float kSlicedCopyResourceInflation = 1.8;
  for (int i = 0; i < slice_start_times.size(); ++i) {
    float original_copy_resource = copy_resource_per_slice[i];
    float new_copy_resource = original_copy_resource;
    if (slice_start_times.size() > 1) {
      // This is a hack that makes us more conservative about using sliced
      // prefetching vs unsliced prefetching.
      new_copy_resource = original_copy_resource * kSlicedCopyResourceInflation;
      VLOG(5)
          << "Inflating required copy resources DoWeHaveEnoughCopyResource() "
             "slice check from "
          << original_copy_resource << " to " << new_copy_resource;
    }
    specs.push_back(
        {slice_start_times[i], prefetch_end_time, new_copy_resource});
  }

  auto specs_to_string = [&specs]() {
    return absl::StrCat(
        "[ ",
        absl::StrJoin(specs, ", ",
                      [](std::string* out,
                         const AsynchronousCopyResource::ResourceSpec& spec) {
                        absl::StrAppend(out, "{exclusive start: ",
                                        spec.exclusive_start_time,
                                        ", end: ", spec.end_time,
                                        ", resource: ", spec.resource, "}");
                      }),
        " ]");
  };

  VLOG(5) << "Checking for enough copy resources for: " << specs_to_string();
  if (!async_copy_resource.HasEnoughResourceMultiCheck(specs)) {
    VLOG(4) << "Not enough copy resources for " << specs_to_string();
    return false;
  }
  return true;
}

// We compute a map from indices in chunk_candidates to indices in a
// SliceProposalCollection. Since the indices of chunk_candidates correspond to
// slice start times order, and SliceProposalCollections are always sorted in
// offset order, the mapping allows us to get the sizing details of a slice at a
// specific slice time.
absl::flat_hash_map<int64_t, int64_t> GetCandidateToProposalIndexMap(
    const std::vector<MsaAlgorithm::Chunk>& chunk_candidates) {
  std::vector<std::pair<int64_t, int64_t>> sorted_offset_candidate_index_pairs;
  sorted_offset_candidate_index_pairs.reserve(chunk_candidates.size());
  for (int64_t chunk_candidate_index = 0;
       chunk_candidate_index < chunk_candidates.size();
       ++chunk_candidate_index) {
    sorted_offset_candidate_index_pairs.push_back(std::make_pair(
        chunk_candidates[chunk_candidate_index].offset, chunk_candidate_index));
  }
  absl::c_sort(sorted_offset_candidate_index_pairs);

  absl::flat_hash_map<int64_t, int64_t> candidate_to_proposal_index_map;
  for (int64_t offset_index = 0;
       offset_index < sorted_offset_candidate_index_pairs.size();
       ++offset_index) {
    int64_t chunk_candidate_index =
        sorted_offset_candidate_index_pairs[offset_index].second;
    candidate_to_proposal_index_map[chunk_candidate_index] = offset_index;
  }

  return candidate_to_proposal_index_map;
}

}  // namespace

AllocationResult MsaAlgorithm::CheckPrefetchFit(bool for_sliced_solution,
                                                PrefetchContext& context) {
  SlicedBufferInterval* sliced_buffer_interval =
      context.GetMutableWorkingIntervals(for_sliced_solution).sliced.get();

  if (for_sliced_solution) {
    CHECK(context.slice_proposal_collection);
    CHECK_EQ(context.slice_proposal_collection->size(),
             sliced_buffer_interval->num_slices());
  }

  // Update the prefetch start time in our working solution.
  std::vector<int64_t> exclusive_slice_start_times =
      SlicedPrefetchStartTimePicker::Pick(
          sliced_buffer_interval->num_slices(),
          context.exclusive_prefetch_start_time, context.prefetch_end_time,
          [&](int64_t exclusive_start_time,
              int64_t exclusive_end_time) -> float {
            return options_.prefetch_interval_picker->GetLogicalIntervalElapsed(
                exclusive_start_time, exclusive_end_time);
          },
          [&](int64_t lhs_time, int64_t rhs_time) -> bool {
            return hlo_live_range_.flattened_instruction_sequence()
                       .instructions()[lhs_time]
                       ->parent() ==
                   hlo_live_range_.flattened_instruction_sequence()
                       .instructions()[rhs_time]
                       ->parent();
          });
  CHECK_EQ(sliced_buffer_interval->num_slices(),
           exclusive_slice_start_times.size());
  sliced_buffer_interval->UpdateExclusiveSliceStartTimes(
      exclusive_slice_start_times);
  VLOG(4) << AlternateMemoryAllocationAttemptToString(for_sliced_solution,
                                                      context);

  // Check if all slices have the same start time. If so, we might as well
  // resort to a full copy.
  if (for_sliced_solution &&
      absl::c_all_of(
          exclusive_slice_start_times, [&](int64_t slice_start_time) {
            return slice_start_time == exclusive_slice_start_times.front();
          })) {
    return AllocationResult::kAllSlicesHaveTheSameStartTime;
  }

  // Check that we have enough copy resource for the prefetching.
  std::vector<float> copy_resource_per_slice_sorted_by_start_time;
  // If there is a preferred prefetch time due to a loop optimized allocation,
  // we already keep track of the prefetch resources there, so skip tracking
  // resources here.
  if (context.request->preferred_prefetch_time) {
    copy_resource_per_slice_sorted_by_start_time =
        std::vector<float>(exclusive_slice_start_times.size(), 0.0);
  } else if (for_sliced_solution) {
    // In a sliced setting, we don't yet know when each slice will be
    // prefetched. Given the proposed slice times, the most conservative copy
    // resource check we can make is to assume that larger slices are started
    // at earlier times, i.e., they have more time to complete. That is the
    // check we will make here. Once, we've decided when each slice will be
    // prefetched, we can do an exact check below.
    //
    // We start by computing the amount of copy resources needed for each slice,
    // if larger slices are started at earlier times.
    copy_resource_per_slice_sorted_by_start_time =
        GetCopyResourcesSortedDescending(options_,
                                         *context.slice_proposal_collection);
  } else {
    copy_resource_per_slice_sorted_by_start_time.push_back(
        CopyResourceForShape(options_, *context.full_shape));
  }
  CHECK_EQ(sliced_buffer_interval->num_slices(),
           copy_resource_per_slice_sorted_by_start_time.size());

  if (!DoWeHaveEnoughCopyResource(exclusive_slice_start_times,
                                  context.prefetch_end_time,
                                  copy_resource_per_slice_sorted_by_start_time,
                                  prefetch_async_copy_resource_)) {
    return AllocationResult::kFailViolatesAsyncCopyResource;
  }

  // Check if the copies we would add for the prefetch would violate copy
  // ordering.
  if (options_.enforce_prefetch_fifo_order &&
      absl::c_any_of(exclusive_slice_start_times,
                     [&](int64_t slice_start_time) {
                       return async_copy_ordering_.ViolatesOrdering(
                           slice_start_time, context.prefetch_end_time);
                     })) {
    VLOG(4) << "This would violate asynchronous copy ordering.";
    return AllocationResult::kFailViolatesAsyncCopyResource;
  }

  // Check if the copies we would add for the prefetch violate the maximum
  // number of outstanding async copies.
  for (int i = 0; i < exclusive_slice_start_times.size(); ++i) {
    if (ViolatesMaximumOutstandingAsyncCopies(
            exclusive_slice_start_times[i], context.prefetch_end_time,
            /*is_prefetch=*/true, context.extra_async_copy_limit, i)) {
      VLOG(4) << "This would violate the outstanding async copy limit.";
      return AllocationResult::kFailOutOfAsyncCopies;
    }
  }

  // Check if we can find a place in alternate memory for the prefetch.
  std::vector<Chunk> chunk_candidates = FindBestChunkCandidates(
      *context.request, context.request->preferred_offset,
      sliced_buffer_interval);
  CHECK(chunk_candidates.empty() ||
        chunk_candidates.size() == sliced_buffer_interval->num_slices());
  std::string prefetch_picker_debug_string;
  if (VLOG_IS_ON(4)) {
    prefetch_picker_debug_string =
        options_.prefetch_interval_picker->ToDebugString();
  }
  if (for_sliced_solution && !chunk_candidates.empty()) {
    // We're trying a sliced solution. So, if FindBestChunkCandidates() found a
    // solution, each slice should have its own chunk candidate.
    CHECK_EQ(chunk_candidates.size(), sliced_buffer_interval->num_slices());
    // We need a mapping from chunks in chunk_candidates to slice proposals in
    // context.slice_proposal_context.
    absl::flat_hash_map<int64_t, int64_t> candidate_to_proposal_index_map =
        GetCandidateToProposalIndexMap(chunk_candidates);

    // Create slice decisions, sorted by time.
    std::vector<SliceDecision> slice_decisions_sorted_by_start_time;
    for (int64_t slice_time = 0;
         slice_time < sliced_buffer_interval->num_slices(); ++slice_time) {
      const SliceProposal& proposal = context.slice_proposal_collection->at(
          candidate_to_proposal_index_map[slice_time]);
      copy_resource_per_slice_sorted_by_start_time[slice_time] =
          CopyResourceForShape(options_, proposal.slice_shape);
      slice_decisions_sorted_by_start_time.push_back(SliceDecision{
          chunk_candidates[slice_time], exclusive_slice_start_times[slice_time],
          proposal, copy_resource_per_slice_sorted_by_start_time[slice_time]});
    }

    // Check that we have enough copy resources for all the slice decisions.
    if (!DoWeHaveEnoughCopyResource(
            exclusive_slice_start_times, context.prefetch_end_time,
            copy_resource_per_slice_sorted_by_start_time,
            prefetch_async_copy_resource_)) {
      return AllocationResult::kFailViolatesAsyncCopyResource;
    }

    // Construct MsaBufferInterval-Chunk pairs that are appropriate for pending
    // chunks, as described in PrefetchContext::SlicedSolution.
    std::vector<std::pair<MsaBufferInterval, Chunk>> slices_for_pending_chunks;
    slices_for_pending_chunks.reserve(sliced_buffer_interval->num_slices());
    Chunk final_chunk = Chunk::FromOffsetSize(
        absl::c_min_element(
            chunk_candidates,
            [](const Chunk& a, const Chunk& b) { return a.offset < b.offset; })
            ->offset,
        absl::c_accumulate(
            chunk_candidates, 0,
            [](int64_t sum, const Chunk& chunk) { return sum + chunk.size; }));
    MsaBufferInterval final_buffer_interval{
        context.request->allocation_value->value(),
        /*size=*/final_chunk.size,
        /*start=*/
        ExclusiveToInclusiveStartTime(exclusive_slice_start_times.back()),
        /*end=*/context.request->end_time,
        /*colocations=*/
        sliced_buffer_interval->full_buffer_interval().colocations,
        /*need_allocation=*/true};
    for (int64_t slice_time = 0;
         slice_time < sliced_buffer_interval->num_slices(); ++slice_time) {
      const Chunk& chunk = chunk_candidates[slice_time];
      int64_t inclusive_start_time = ExclusiveToInclusiveStartTime(
          exclusive_slice_start_times[slice_time]);
      if (inclusive_start_time ==
          ExclusiveToInclusiveStartTime(exclusive_slice_start_times.back())) {
        // This and the following chunks will be merged into the final chunk.
        // Note, it's possible for more than one slice to start at the same
        // time.
        break;
      }
      CHECK_LT(inclusive_start_time, ExclusiveToInclusiveStartTime(
                                         exclusive_slice_start_times.back()));
      slices_for_pending_chunks.push_back(std::make_pair(
          MsaBufferInterval{
              context.request->allocation_value->value(),
              /*size=*/chunk.size,
              /*start=*/inclusive_start_time,
              /*end=*/exclusive_slice_start_times.back(),
              // We only use the final_buffer_interval for colocations because
              // slices start at different offsets, and the colocation
              // infrastructure expects all colocated buffers to start at the
              // same offset.
              /*colocations=*/{},
              /*need_allocation=*/true,
          },
          chunk));
    }
    slices_for_pending_chunks.push_back(
        std::make_pair(final_buffer_interval, final_chunk));

    context.sliced_solution = PrefetchContext::SlicedSolution{
        std::move(slice_decisions_sorted_by_start_time),
        std::move(slices_for_pending_chunks),
        prefetch_picker_debug_string,
    };
    return AllocationResult::kSuccess;
  } else if (!chunk_candidates.empty()) {
    // We're trying an unsliced solution. So, if FindBestChunkCandidates() found
    // a solution, there must be only 1 chunk for it.
    CHECK_EQ(chunk_candidates.size(), 1);
    CHECK_EQ(copy_resource_per_slice_sorted_by_start_time.size(), 1);
    context.unsliced_solution = PrefetchContext::UnslicedSolution{
        chunk_candidates.front(),
        copy_resource_per_slice_sorted_by_start_time.front(),
        prefetch_picker_debug_string,
    };
    return AllocationResult::kSuccess;
  }

  // Mark the out of memory start with the prefetch start time so that we don't
  // explore prefetch start times earlier than this point. If a sliced prefetch
  // doesn't fit at a given time, an unsliced prefetch will not fit either.
  // Thus, if we are considering a sliced prefetch for the current request,
  // we can only update out_of_mem_start when we check with slices.
  if (for_sliced_solution || !context.slice_proposal_collection) {
    CHECK_GT(exclusive_slice_start_times.size(), 0);
    context.exclusive_out_of_mem_start = std::max(
        context.exclusive_out_of_mem_start ? *context.exclusive_out_of_mem_start
                                           : -1,
        exclusive_slice_start_times.front());
  }

  VLOG(4) << "Out of memory.";
  return AllocationResult::kFailOutOfMemory;
}

std::string MsaAlgorithm::AlternateMemoryAllocationAttemptToString(
    bool for_sliced_solution, const PrefetchContext& context) const {
  const SlicedBufferInterval* sliced_buffer_interval =
      context.GetWorkingIntervals(for_sliced_solution).sliced.get();

  std::vector<std::string> slice_times;
  std::vector<int64_t> estimated_slice_prefetch_end_times;

  for (int i = 0; i < sliced_buffer_interval->num_slices(); ++i) {
    slice_times.push_back(absl::StrCat(
        "[", sliced_buffer_interval->IntervalForMakeFreeChunks(i).start, ", ",
        sliced_buffer_interval->full_buffer_interval().end, ")"));
    if (context.slice_proposal_collection) {
      estimated_slice_prefetch_end_times.push_back(
          options_.prefetch_interval_picker->EstimatedPrefetchEndTime(
              context.slice_proposal_collection->at(i).slice_shape,
              sliced_buffer_interval->IntervalForMakeFreeChunks(i).start,
              context.prefetch_end_time));
    } else {
      estimated_slice_prefetch_end_times.push_back(
          options_.prefetch_interval_picker->EstimatedPrefetchEndTime(
              *context.full_shape,
              sliced_buffer_interval->IntervalForMakeFreeChunks(i).start,
              context.prefetch_end_time));
    }
  }

  return absl::StrCat(
      "Trying alternate memory allocation. Slice times = { ",
      absl::StrJoin(slice_times, ", "), " }. Estimated prefetch end times = { ",
      absl::StrJoin(estimated_slice_prefetch_end_times, ", "), " }");
}

std::optional<MsaAlgorithm::Chunk> MsaAlgorithm::FindBestChunkCandidate(
    const AllocationRequest& request, const AliasedOffset* preferred_offset,
    MsaBufferInterval* alternate_mem_interval) const {
  SlicedBufferInterval sliced_buffer_interval =
      SlicedBufferInterval::CreateMutableInterval(*alternate_mem_interval);
  std::vector<Chunk> chunks = FindBestChunkCandidates(request, preferred_offset,
                                                      &sliced_buffer_interval);
  CHECK_LE(chunks.size(), 1);
  if (chunks.empty()) {
    return std::nullopt;
  }
  return chunks[0];
}

std::vector<MsaAlgorithm::Chunk> MsaAlgorithm::FindBestChunkCandidates(
    const AllocationRequest& request, const AliasedOffset* preferred_offset,
    SlicedBufferInterval* alternate_mem_interval) const {
  int64_t end_time = request.end_time;
  if (!preferred_offset) {
    // First find the earliest use that is the same or later than the end time.
    const auto& use_times = request.all_use_times;
    auto use_time_it = absl::c_lower_bound(use_times, end_time);
    CHECK(use_time_it != use_times.end());
    int64_t earliest_use = *use_time_it;
    auto earliest_use_it = use_time_it;

    // Then find the latest use that can be allocated contiguously without
    // copies.
    const Shape& shape =
        request.allocation_value_to_update->defining_position().shape();
    for (;
         (use_time_it + 1) != use_times.end() &&
         options_.prefetch_interval_picker->CanAllocateInAlternateMemoryNoCopy(
             shape, *use_time_it, *(use_time_it + 1));
         ++use_time_it) {
    }
    CHECK(use_time_it != use_times.end());
    int64_t latest_contiguous_use_time = *use_time_it;

    // Find chunks that are as long living as possible.
    std::vector<Chunk> last_chunk_candidates;
    int64_t latest_matching_use = std::numeric_limits<int64_t>::min();
    (void)std::lower_bound(
        earliest_use_it, std::next(use_time_it), -1, [&](int64_t use, int64_t) {
          alternate_mem_interval->UpdateEndTime(use);
          std::vector<Chunk> chunk_candidates =
              FindChunkCandidates(*alternate_mem_interval);
          int64_t candidates_end =
              absl::c_max_element(chunk_candidates, [](const Chunk& c1,
                                                       const Chunk& c2) {
                return c1.chunk_end() < c2.chunk_end();
              })->chunk_end();
          if (candidates_end <= available_heap_size()) {
            if (use > latest_matching_use) {
              last_chunk_candidates = std::move(chunk_candidates);
              latest_matching_use = use;
            }
            return true;
          }
          return false;
        });
    if (!last_chunk_candidates.empty()) {
      VLOG(3) << "FindBestChunkCandidates earliest use = " << earliest_use
              << ", latest contiguous use = " << latest_contiguous_use_time
              << ", use with available mem = " << latest_matching_use
              << ", offsets = { "
              << absl::StrJoin(last_chunk_candidates, ", ",
                               [](std::string* out, const Chunk& c) {
                                 absl::StrAppend(out, c.offset);
                               })
              << " }";
    }
    alternate_mem_interval->UpdateEndTime(end_time);
    return last_chunk_candidates;
  }
  // If a preferred offset is given, try to find an allocation at that offset
  // only.
  alternate_mem_interval->UpdateEndTime(end_time);
  std::vector<Chunk> chunk_candidates =
      FindChunkCandidates(*alternate_mem_interval, preferred_offset->offset);
  int64_t candidates_start =
      absl::c_min_element(chunk_candidates, [](const Chunk& c1,
                                               const Chunk& c2) {
        return c1.offset < c2.offset;
      })->offset;

  if (candidates_start == preferred_offset->offset) {
    return chunk_candidates;
  }

  return {};
}

}  // namespace memory_space_assignment
}  // namespace xla
