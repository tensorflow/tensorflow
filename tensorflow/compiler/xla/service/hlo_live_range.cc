/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/hlo_live_range.h"

#include "absl/strings/str_format.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor.h"
#include "tensorflow/compiler/xla/service/hlo_value.h"

namespace xla {
/*static*/
StatusOr<std::unique_ptr<HloLiveRange>> HloLiveRange::Run(
    const HloSchedule& schedule, const HloAliasAnalysis& alias_analysis,
    const HloComputation* computation, bool module_scoped_analysis) {
  std::unique_ptr<HloLiveRange> hlo_live_range(
      new HloLiveRange(schedule, alias_analysis, module_scoped_analysis));
  hlo_live_range->schedule_end_time_ =
      hlo_live_range->FlattenSchedule(*computation, 0);
  hlo_live_range->CalculateBufferStartEndMap();
  hlo_live_range->NormalizeAliasedBuffers();
  return std::move(hlo_live_range);
}

void HloLiveRange::NormalizeAliasedBuffers() {
  for (const HloBuffer& hlo_buffer : alias_analysis_.buffers()) {
    std::vector<const HloValue*> aliased_buffers;
    for (const HloValue* hlo_value : hlo_buffer.values()) {
      if (buffer_live_ranges_.contains(hlo_value)) {
        aliased_buffers.push_back(hlo_value);
      }
    }
    absl::c_sort(
        aliased_buffers, [&](const HloValue* value1, const HloValue* value2) {
          const TimeBound& live_range1 = buffer_live_ranges_.at(value1);
          const TimeBound& live_range2 = buffer_live_ranges_.at(value2);

          return std::forward_as_tuple(live_range1.start, live_range1.end) <
                 std::forward_as_tuple(live_range2.start, live_range2.end);
        });

    for (int64_t i = 0; i + 1 < aliased_buffers.size(); ++i) {
      const HloValue* value1 = aliased_buffers[i];
      const HloValue* value2 = aliased_buffers[i + 1];
      TimeBound& live_range1 = buffer_live_ranges_[value1];
      TimeBound& live_range2 = buffer_live_ranges_[value2];
      if (live_range1.start == live_range2.start) {
        // If value1 has the same start time as value2, make value1 disappear
        // by setting the end time same as start time:
        //
        // Before:
        // +----+           value1
        // +----------+     value2
        //
        // After:
        // +                value1
        // +----------+     value2
        //
        // Note that only when heap simulator runs before copy insertion can
        // this happen where one instruction defines multiple aliased buffers
        // -- This is illegle to execute and can be fixed by copy insertion
        // later.
        live_range1.end = live_range2.end;
        continue;
      }

      if (live_range1.end < live_range2.start) {
        continue;
      }

      if (live_range1.end > live_range2.end) {
        live_range2.end = live_range1.end;
      }
      live_range1.end = live_range2.start - 1;
    }
  }
}

// FlattenSchedule walks through the computation and tracks down the ordinal
// number of each instruction in the schedule.
int64_t HloLiveRange::FlattenSchedule(const HloComputation& computation,
                                      int64_t start_time) {
  if (!schedule_.is_computation_scheduled(&computation)) {
    total_order_scheduled_ = false;
    return start_time;
  }

  const HloInstructionSequence& instruction_sequence =
      schedule_.sequence(&computation);
  int64_t time = start_time;
  for (HloInstruction* instruction : instruction_sequence.instructions()) {
    if (module_scoped_analysis_) {
      // Recurse into sub computations if running with module scoped analysis
      // mode.
      if (instruction->opcode() == HloOpcode::kCall ||
          instruction->opcode() == HloOpcode::kConditional) {
        for (const HloComputation* called_computation :
             instruction->called_computations()) {
          time = FlattenSchedule(*called_computation, time);
        }
      }
      if (instruction->opcode() == HloOpcode::kWhile) {
        time = FlattenSchedule(*instruction->while_condition(), time);
        time = FlattenSchedule(*instruction->while_body(), time);
      }
    }
    if (instruction_schedule_.count(instruction) != 0) {
      continue;
    }
    instruction_schedule_.insert({instruction, time++});
    flattened_instruction_sequence_.push_back(instruction);
  }
  computation_span_times_.try_emplace(&computation,
                                      TimeBound{start_time, time});
  DCHECK_EQ(instruction_schedule_.size(),
            flattened_instruction_sequence_.size());
  DCHECK_LE(instruction_schedule_.size(), time);
  return time;
}

void HloLiveRange::CalculateBufferStartEndMap() {
  for (const HloValue* value : alias_analysis_.dataflow_analysis().values()) {
    // Ignore buffers that are not defined.
    if (instruction_schedule_.count(value->defining_instruction()) == 0) {
      continue;
    }

    int64_t buffer_start_time = instruction_schedule_[value->instruction()];

    int64_t buffer_end_time = -1;
    for (const HloUse& use : value->uses()) {
      const HloInstruction* used = use.instruction;
      // As an optimization, we deem a while's init value's live range ends as
      // soon as the loop body starts. This optimization is only applicable in
      // module scoped mode.
      if (module_scoped_analysis_ && used->opcode() == HloOpcode::kWhile) {
        // The current live range is at the end of the while, move it to the
        // beginning of the body.
        used = used->while_body()->parameter_instruction(0);
        VLOG(1) << "Moved value " << value->ToShortString()
                << " to while param: " << used->ToString();
      }
      if (instruction_schedule_.count(used) == 0) {
        // We didn't track the instruction `used`. This happens when we do
        // computation scope (versus module scope) heap simulation and when
        // the used instruction is outside of the computation being simulated.
        continue;
      }
      buffer_end_time = std::max(buffer_end_time, instruction_schedule_[used]);
    }

    // Parameters are defined at the beginning of the computation. This prevents
    // any instruction that's scheduled before the parameter clobbers the
    // parameter's buffer.
    if (value->instruction()->opcode() == HloOpcode::kParameter) {
      const HloComputation* computation = value->instruction()->parent();
      auto it = computation_span_times_.find(computation);
      if (it != computation_span_times_.end()) {
        buffer_start_time = std::min(buffer_start_time, it->second.start);
      }
    }

    if (buffer_end_time == -1) {
      buffer_end_time = buffer_start_time;
    }

    HloPosition end_position;
    int64_t max_end_time = 0;
    for (const HloPosition& position : value->positions()) {
      if (instruction_schedule_[position.instruction] >= max_end_time) {
        max_end_time = instruction_schedule_[value->instruction()];
        end_position = position;
      }
      const HloComputation* position_comp = position.instruction->parent();
      // If this instruction lives out, the live range of the instruction
      // should be extended to the end of the computation.
      if (position.instruction == position_comp->root_instruction()) {
        auto it = computation_span_times_.find(position_comp);
        if (it == computation_span_times_.end()) {
          continue;
        }
        if (buffer_end_time < it->second.end) {
          buffer_end_time = it->second.end;
          end_position = position;
        }
      }
    }

    const HloModule* module = value->instruction()->parent()->parent();

    // Readonly entry parameters (parameters that don't alias) live across whole
    // computation.
    if (value->instruction()->opcode() == HloOpcode::kParameter &&
        value->instruction()->parent() == module->entry_computation() &&
        !module->input_output_alias_config().ParameterHasAlias(
            value->instruction()->parameter_number(), value->index())) {
      buffer_end_time = schedule_end_time_;
    }

    CHECK(buffer_start_time <= buffer_end_time)
        << buffer_start_time << ", " << buffer_end_time
        << value->instruction()->ToString();

    auto& live_range = buffer_live_ranges_[value];
    live_range.start = buffer_start_time;
    live_range.end = buffer_end_time;
    live_range.end_position = end_position;
  }
}

int64_t HloLiveRange::ComputePeakMemoryMoment() const {
  std::vector<std::tuple<int64_t /*time*/, bool /*is_end*/, const HloValue*>>
      events;
  for (const HloValue* value : alias_analysis_.dataflow_analysis().values()) {
    auto it = buffer_live_ranges_.find(value);
    if (it != buffer_live_ranges_.end()) {
      events.emplace_back(it->second.start, false, value);
      events.emplace_back(it->second.end + 1, true, value);
    }
  }
  std::sort(events.begin(), events.end());

  int64_t memory_usage = 0;
  int64_t peak_usage = 0;
  absl::optional<int64_t> peak_time;
  for (const auto& event : events) {
    int64_t time;
    bool is_end;
    const HloValue* value;
    std::tie(time, is_end, value) = event;
    auto buffer_size = ShapeUtil::ByteSizeOf(value->instruction()->shape(), 8);
    if (is_end) {
      memory_usage -= buffer_size;
    } else {
      memory_usage += buffer_size;
    }
    if (peak_usage < memory_usage) {
      peak_usage = memory_usage;
      peak_time = time;
    }
  }
  return peak_time.value_or(0);
}

std::string HloLiveRange::ToString() const {
  std::string output;
  absl::StrAppendFormat(&output, "HloLiveRange (max %d):\n",
                        schedule_end_time_);
  absl::StrAppendFormat(&output, "  InstructionSequence:\n");
  auto& instructions = flattened_instruction_sequence().instructions();
  for (int64_t i = 0; i < instructions.size(); ++i) {
    absl::StrAppendFormat(&output, "    %d:%s\n", i, instructions[i]->name());
  }

  absl::StrAppendFormat(&output, "  BufferLiveRange:\n");

  for (const HloValue* value : alias_analysis_.dataflow_analysis().values()) {
    auto it = buffer_live_ranges_.find(value);
    if (it != buffer_live_ranges_.end()) {
      absl::StrAppendFormat(
          &output, "    %s%s:%d-%d\n", value->instruction()->name(),
          value->index().ToString(), it->second.start, it->second.end);
    }
  }

  int64_t peak_moment = ComputePeakMemoryMoment();

  absl::StrAppendFormat(&output, "  Live ranges at %lld (peak):\n",
                        peak_moment);
  for (const HloValue* value : alias_analysis_.dataflow_analysis().values()) {
    auto it = buffer_live_ranges_.find(value);
    if (it != buffer_live_ranges_.end()) {
      if (it->second.start <= peak_moment && peak_moment <= it->second.end) {
        int64_t bytes = ShapeUtil::ByteSizeOf(value->instruction()->shape(), 8);
        absl::StrAppendFormat(&output, "    %s: %lld bytes\n",
                              value->instruction()->name(), bytes);
      }
    }
  }

  return output;
}

}  // namespace xla
