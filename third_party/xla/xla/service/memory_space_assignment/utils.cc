/* Copyright 2020 The OpenXLA Authors.

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

#include "xla/service/memory_space_assignment/utils.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <random>
#include <string>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "highwayhash/arch_specific.h"
#include "highwayhash/hh_types.h"
#include "highwayhash/highwayhash.h"
#include "re2/re2.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/hlo/utils/hlo_live_range.h"
#include "xla/service/heap_simulator/heap_simulator.h"
#include "xla/service/hlo_value.h"
#include "xla/service/memory_space_assignment/memory_space_assignment.pb.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace memory_space_assignment {
namespace {

// The default seed used by HloRandomFilter when no seed is given.
const int64_t kRandomFilterDefaultSeed = 1234;

}  // namespace

bool MemorySpaceAssignmentUtils::IsValueAllowedInAlternateMemory(
    const HloValue* value, int64_t alternate_memory_space) {
  // If the buffer is a tuple, don't use this algorithm for now. The buffers
  // that are pointed to by the tuple will still use this algorithm.  Because
  // tuples are cheap to place in the alternate memory (they are just pointers)
  // we don't need to use prefetch/evict logic.
  if (value->shape().IsTuple()) {
    VLOG(4) << "Keeping value " << value->ToShortString()
            << " in default mem because it is a tuple.";
    return false;
  }

  // Don't place scalars in the alternate memory.
  if (ShapeUtil::IsEffectiveScalar(value->shape())) {
    VLOG(4) << "Keeping value " << value->ToShortString()
            << " in default mem because it is a scalar.";
    return false;
  }

  // TODO(berkin): Not allocating add-dependencies either since they need to be
  // treated specially. We should revisit this later.
  for (const HloPosition& position : value->positions()) {
    if (position.instruction->opcode() == HloOpcode::kAddDependency) {
      VLOG(4) << "Keeping value " << value->ToShortString()
              << " in default mem because it has a "
              << "add-dependency position.";
      return false;
    }
  }

  // Send and Recv HLOs return a request identifier. These should not be
  // allocated in the alternate memory.
  for (const HloPosition& position : value->positions()) {
    if ((position.instruction->opcode() == HloOpcode::kSend ||
         position.instruction->opcode() == HloOpcode::kRecv) &&
        DynCast<HloSendRecvInstruction>(position.instruction)
            ->is_host_transfer()) {
      // TODO(berkin): Host transfers using alternate memory space doesn't seem
      // to work at the moment.
      VLOG(4) << "Keeping value " << value->ToShortString()
              << " in default mem because it is a send/recv buffer used for "
                 "host transfer.";
      return false;
    }

    // If the tensor is pre-colored to a memory space that is neither the
    // default (0) nor the alternate, disallow it from the alternate memory
    // space.
    int64_t memory_space = 0;
    if (position.shape().has_layout()) {
      memory_space = position.shape().layout().memory_space();
    }
    if (memory_space != 0 && memory_space != alternate_memory_space) {
      VLOG(4) << "Value " << value->ToShortString()
              << " not allowed in the alternate memory space due to existing "
                 "memory space: "
              << memory_space;
      return false;
    }
  }

  return true;
}

bool MemorySpaceAssignmentUtils::IsIntervalAllowedInAlternateMemory(
    const GlobalDecreasingSizeBestFitHeap<HloValue>::BufferInterval& interval,
    int64_t alternate_memory_space) {
  return IsValueAllowedInAlternateMemory(interval.buffer,
                                         alternate_memory_space) &&
         absl::c_all_of(interval.colocations,
                        [alternate_memory_space](const HloValue* value) {
                          return IsValueAllowedInAlternateMemory(
                              value, alternate_memory_space);
                        });
}

bool MemorySpaceAssignmentUtils::DoesUseMatchFilter(
    const HloOperandFilter& filter, const HloUse& hlo_use,
    int64_t operand_size) {
  // The order of checks is such that the most expensive checks are done last.
  if (filter.has_size_gte() && operand_size < filter.size_gte()) {
    return false;
  }
  if (filter.has_size_lte() && operand_size > filter.size_lte()) {
    return false;
  }
  if (filter.has_operand_number() &&
      hlo_use.operand_number != filter.operand_number()) {
    return false;
  }
  if (filter.has_tuple_index() &&
      hlo_use.operand_index != ShapeIndex(filter.tuple_index().index().begin(),
                                          filter.tuple_index().index().end())) {
    return false;
  }
  if (filter.has_instruction_name_regex() &&
      !RE2::FullMatch(hlo_use.instruction->name(),
                      filter.instruction_name_regex())) {
    return false;
  }
  if (filter.has_instruction_regex() &&
      !RE2::FullMatch(hlo_use.instruction->ToString(),
                      filter.instruction_regex())) {
    return false;
  }
  return true;
}

double GetInstructionUniformRandom(const HloInstruction& instruction,
                                   int64_t seed) {
  // We use the instruction Fingerprint because it's robust across different
  // runs and compilations, as well as because it depends only on the
  // isomorphic computation graph of the instruction rather than the variable
  // names. We would like to make identical decisions for identical
  // instructions.
  //
  // See HloPrintOptions::Fingerprint() for details.
  //
  // If in the future that turns out to not be good enough, we can also pass an
  // option in the proto to choose different string representations of the
  // instructions - for example, the name or "instruction.ToString()" if we want
  // a finer grained perturbation, or alternatively the
  // "instruction.SignatureString()" if we want a coarser perturbation.
  const std::string& instruction_identifier =
      instruction.ToString(HloPrintOptions::Fingerprint());
  std::string instruction_seed_str = absl::StrCat(
      instruction_identifier.size(), ":", instruction_identifier, ":", seed);

  // We use highwayhash as our hashing function, because it is "Strong",
  // relatively fast, and "all (64/128/256 bit) variants of HighwayHash frozen,
  // i.e. unchanging forever" which is good for test stability and repeatability
  // of experiments.

  // PI in hex
  static constexpr highwayhash::HHKey hash_key = {
      0x3243F6A8885A308Dull,
      0x313198A2E0370734ull,
      0x4A4093822299F31Dull,
      0x0082EFA98EC4E6C8ull,
  };

  highwayhash::HHStateT<HH_TARGET> state(hash_key);
  highwayhash::HHResult64 seed_result;
  highwayhash::HighwayHashT(&state, instruction_seed_str.data(),
                            instruction_seed_str.size(), &seed_result);

  // Assuming highwayhash is a "good" hash and uniformly distributed across all
  // 64 bits, we could probably skip the step of going through the random
  // library to generate our uniform [0...1) random.
  //
  // We add this anyway to make it clear we're looking for a uniform random
  // number.
  std::mt19937_64 gen(seed_result);
  return std::uniform_real_distribution<double>()(gen);
}

bool MemorySpaceAssignmentUtils::DoesInstructionMatchRandomFilter(
    const HloPositionMatcher& filter, const HloInstruction& instruction) {
  if (!filter.has_hlo_random_filter()) {
    // This HloPositionMatcher doesn't have a random filter. We return that we
    // aren't blocking this instruction.
    return true;
  }
  const auto& hlo_random_filter = filter.hlo_random_filter();
  double selection_range_begin = 0.;
  if (hlo_random_filter.has_selection_range_begin()) {
    selection_range_begin = hlo_random_filter.selection_range_begin();
  }
  double selection_range_end = 1.;
  if (hlo_random_filter.has_selection_range_end()) {
    selection_range_end = hlo_random_filter.selection_range_end();
  }
  int64_t seed = kRandomFilterDefaultSeed;
  if (hlo_random_filter.has_seed()) {
    seed = hlo_random_filter.seed();
  }
  double rnd = GetInstructionUniformRandom(instruction, seed);
  return rnd >= selection_range_begin && rnd < selection_range_end;
}

bool MemorySpaceAssignmentUtils::DoesPositionMatchFilter(
    const HloPositionMatcher& filter,
    const MsaBufferInterval& buffer_interval) {
  // The order of checks is such that the most expensive checks are done last.
  HloInstruction* instruction = buffer_interval.buffer->instruction();
  if (filter.has_size_gte() && filter.size_gte() > buffer_interval.size) {
    return false;
  }
  if (filter.has_size_lte() && filter.size_lte() < buffer_interval.size) {
    return false;
  }
  if (filter.has_tuple_index() &&
      buffer_interval.buffer->index() !=
          ShapeIndex(filter.tuple_index().index().begin(),
                     filter.tuple_index().index().end())) {
    return false;
  }
  return DoesInstructionMatchFilter(filter, *instruction) &&
         DoesBufferIntervalMatchHloUseFilter(filter, buffer_interval) &&
         DoesInstructionMatchRandomFilter(filter, *instruction);
}

bool MemorySpaceAssignmentUtils::DoesInstructionMatchFilter(
    const HloPositionMatcher& filter, const HloInstruction& instruction) {
  if (filter.has_instruction_name_regex() &&
      !RE2::FullMatch(instruction.name(), filter.instruction_name_regex())) {
    return false;
  }
  if (filter.has_instruction_regex() &&
      !RE2::FullMatch(instruction.ToString(), filter.instruction_regex())) {
    return false;
  }
  return true;
}

bool MemorySpaceAssignmentUtils::DoesBufferIntervalMatchHloUseFilter(
    const HloPositionMatcher& filter,
    const MsaBufferInterval& buffer_interval) {
  if (!filter.has_hlo_use_filter()) {
    return true;
  }
  for (const HloUse& use : buffer_interval.buffer->GetUses()) {
    if (DoesUseMatchFilter(filter.hlo_use_filter(), use,
                           buffer_interval.size)) {
      return true;
    }
  }
  return false;
}

absl::StatusOr<xla::HloLiveRange::LogicalTime>
MemorySpaceAssignmentUtils::GetScheduleTimeFromInstructionMatcher(
    const HloPositionMatcher& position_matcher,
    const absl::flat_hash_map<const xla::HloInstruction*,
                              xla::HloLiveRange::LogicalTime>& schedule) {
  for (auto schedule_entry : schedule) {
    if (DoesInstructionMatchFilter(position_matcher, *schedule_entry.first)) {
      return schedule_entry.second;
    }
  }
  return NotFound("Reference instruction %s was not found in the schedule.",
                  position_matcher.DebugString());
}

absl::StatusOr<std::optional<int64_t>>
MemorySpaceAssignmentUtils::GetPrefetchTimeByEagerness(
    float prefetch_eagerness, int64_t earliest_prefetch_time,
    int64_t latest_prefetch_time) {
  CHECK_GE(prefetch_eagerness, 0.0);
  CHECK_LE(prefetch_eagerness, 1.0);
  if (earliest_prefetch_time > latest_prefetch_time) {
    return static_cast<std::optional<int64_t>>(std::nullopt);
  }
  return static_cast<std::optional<int64_t>>(
      earliest_prefetch_time +
      (latest_prefetch_time - earliest_prefetch_time) * prefetch_eagerness);
}

absl::StatusOr<std::optional<int64_t>>
MemorySpaceAssignmentUtils::GetPrefetchTimeAfterInstruction(
    const HloPositionMatcher& after_instruction,
    const absl::flat_hash_map<const xla::HloInstruction*,
                              xla::HloLiveRange::LogicalTime>& schedule) {
  TF_ASSIGN_OR_RETURN(
      auto reference_instruction_time,
      GetScheduleTimeFromInstructionMatcher(after_instruction, schedule));
  return static_cast<std::optional<int64_t>>(reference_instruction_time);
}

absl::StatusOr<std::optional<int64_t>>
MemorySpaceAssignmentUtils::GetPrefetchTimeBeforeInstruction(
    const HloPositionMatcher& before_instruction,
    const absl::flat_hash_map<const xla::HloInstruction*,
                              xla::HloLiveRange::LogicalTime>& schedule) {
  TF_ASSIGN_OR_RETURN(
      auto reference_instruction_time,
      GetScheduleTimeFromInstructionMatcher(before_instruction, schedule));
  return static_cast<std::optional<int64_t>>(reference_instruction_time - 1);
}
absl::StatusOr<std::optional<int64_t>>
MemorySpaceAssignmentUtils::GetPrefetchTime(
    const PreferredPrefetchOverrideOptions& override_options,
    int64_t earliest_prefetch_time, int64_t latest_prefetch_time,
    const absl::flat_hash_map<const HloInstruction*, HloLiveRange::LogicalTime>&
        instruction_schedule) {
  switch (override_options.options_case()) {
    case PreferredPrefetchOverrideOptions::kPrefetchEagerness:
      return GetPrefetchTimeByEagerness(override_options.prefetch_eagerness(),
                                        earliest_prefetch_time,
                                        latest_prefetch_time);
    case PreferredPrefetchOverrideOptions::kAfterInstruction:
      return GetPrefetchTimeAfterInstruction(
          override_options.after_instruction(), instruction_schedule);
    case PreferredPrefetchOverrideOptions::kBeforeInstruction:
      return GetPrefetchTimeBeforeInstruction(
          override_options.before_instruction(), instruction_schedule);
    case PreferredPrefetchOverrideOptions::OPTIONS_NOT_SET:
      break;
  }
  return static_cast<absl::StatusOr<std::optional<int64_t>>>(std::nullopt);
}

absl::StatusOr<std::optional<int64_t>>
MemorySpaceAssignmentUtils::GetOverriddenPreferredPrefetchTime(
    const PreferredPrefetchOverrides& preferred_prefetch_overrides,
    int64_t operand_size, const HloUse& hlo_use,
    const absl::flat_hash_map<const HloInstruction*, HloLiveRange::LogicalTime>&
        instruction_schedule,
    int64_t earliest_prefetch_time, int64_t latest_prefetch_time) {
  for (const auto& override : preferred_prefetch_overrides.overrides()) {
    if (!MemorySpaceAssignmentUtils::DoesUseMatchFilter(
            override.hlo_operand_filter(), hlo_use, operand_size)) {
      continue;
    }
    VLOG(3) << "Config match for instruction " << hlo_use.instruction->name()
            << " operand number " << hlo_use.operand_number << " operand index "
            << hlo_use.operand_index.ToString() << " size " << operand_size
            << " live range (" << earliest_prefetch_time << ", "
            << latest_prefetch_time << ")";
    TF_ASSIGN_OR_RETURN(
        auto prefetch_time,
        GetPrefetchTime(override.override_options(), earliest_prefetch_time,
                        latest_prefetch_time, instruction_schedule));
    if (prefetch_time.has_value() &&
        prefetch_time.value() >= earliest_prefetch_time &&
        prefetch_time.value() <= latest_prefetch_time) {
      return prefetch_time;
    }
  }
  return static_cast<absl::StatusOr<std::optional<int64_t>>>(std::nullopt);
}

bool MemorySpaceAssignmentUtils::DoesCrossProgramPrefetchBufferMatchAnyFilter(
    const MsaSortOrderOverrides& sort_order_overrides,
    const MsaBufferInterval& buffer_interval) {
  return GetBufferIntervalOverridePriority(
             sort_order_overrides, buffer_interval,
             /*is_cross_program_prefetch=*/true) < 0;
}

int64_t MemorySpaceAssignmentUtils::GetBufferIntervalOverridePriority(
    const MsaSortOrderOverrides& msa_sort_order_overrides,
    const MsaBufferInterval& buffer_interval, bool is_cross_program_prefetch) {
  if (msa_sort_order_overrides.overrides_size() == 0) {
    return 0;
  }
  for (int64_t i = 0; i < msa_sort_order_overrides.overrides_size(); ++i) {
    const auto& override = msa_sort_order_overrides.overrides(i);
    if (is_cross_program_prefetch &&
        (!override.has_apply_to_cross_program_prefetches() ||
         !override.apply_to_cross_program_prefetches())) {
      continue;
    }
    if (!MemorySpaceAssignmentUtils::DoesPositionMatchFilter(
            override.hlo_position_matcher(), buffer_interval)) {
      continue;
    }
    VLOG(3) << "Override Sort Order Config " << i << " matches "
            << buffer_interval.buffer->instruction()->ToString();
    switch (override.override_options().options_case()) {
      case MsaSortOrderOverrideOptions::kAssignFirst:
        return std::numeric_limits<int64_t>::lowest() + i;
      case MsaSortOrderOverrideOptions::kAssignLast:
        return std::numeric_limits<int64_t>::max() - i;
      case MsaSortOrderOverrideOptions::kAssignValue:
        return override.override_options().assign_value();
      case MsaSortOrderOverrideOptions::OPTIONS_NOT_SET:
        continue;
    }
  }
  return 0;
}
}  // namespace memory_space_assignment
}  // namespace xla
