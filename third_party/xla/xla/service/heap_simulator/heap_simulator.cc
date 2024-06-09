/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/heap_simulator/heap_simulator.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <ostream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/utils/hlo_live_range.h"
#include "xla/map_util.h"
#include "xla/service/heap_simulator/allocation_block.h"
#include "xla/service/hlo_value.h"
#include "xla/service/memory_space_assignment/repacking.h"
#include "xla/service/time_utils.h"
#include "xla/util.h"

namespace xla {

using absl::flat_hash_map;
using absl::flat_hash_set;

bool IsOdd(int x) { return (x % 2) == 1; }

bool IsEven(int x) { return (x % 2) == 0; }

HeapSimulator::Chunk HeapSimulator::Chunk::FromOffsetEnd(int64_t offset,
                                                         int64_t end) {
  return FromOffsetSize(offset, end - offset);
}

HeapSimulator::Chunk HeapSimulator::Chunk::FromOffsetSize(int64_t offset,
                                                          int64_t size) {
  return Chunk(offset, size);
}

std::string HeapSimulator::Chunk::ToString() const {
  return absl::StrCat("[", offset, ",", chunk_end(), ")");
}

bool HeapSimulator::Chunk::OverlapsWith(Chunk other_chunk) const {
  CHECK_NE(size, 0);
  CHECK_NE(other_chunk.size, 0);
  return offset < other_chunk.chunk_end() && other_chunk.offset < chunk_end();
}

std::ostream& operator<<(std::ostream& stream,
                         const HeapSimulator::Chunk& chunk) {
  stream << chunk.ToString();
  return stream;
}

/*static*/
absl::StatusOr<int64_t> HeapSimulator::MinimumMemoryForModule(
    const HloSchedule& schedule,
    const LogicalBuffer::SizeFunction& size_function) {
  if (schedule.empty()) {
    return 0;
  }
  const HloModule* module = schedule.module();

  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloAliasAnalysis> alias_analysis,
                      HloAliasAnalysis::Run(module));

  // The absolute minimum memory required for a given sequence of instructions
  // is determined by the sequence of Alloc and Free calls on a simulated heap,
  // ignoring fragmentation. We run the heap simulation on the whole module,
  // rather than summing each computation, since it gives us a better lower
  // bound, by minimizing the liveness of sub-computations.
  TF_ASSIGN_OR_RETURN(
      HeapSimulator::Result<HloValue> result,
      HeapSimulator::Run(std::make_unique<NoFragmentationStatsHeap<HloValue>>(),
                         *module, schedule, *alias_analysis, size_function));
  return result.heap_size;
}

/*static*/
absl::StatusOr<int64_t> HeapSimulator::MinimumMemoryForComputation(
    const HloComputation& computation, const HloInstructionSequence& sequence,
    const HloAliasAnalysis& alias_analysis,
    const LogicalBuffer::SizeFunction& size_function,
    const absl::flat_hash_map<const HloComputation*, int64_t>*
        memory_by_computation) {
  TF_ASSIGN_OR_RETURN(
      HeapSimulator::Result<HloValue> result,
      HeapSimulator::Run(std::make_unique<NoFragmentationStatsHeap<HloValue>>(),
                         computation, sequence, alias_analysis, size_function,
                         HeapSimulator::Options(), memory_by_computation));
  return result.heap_size;
}

absl::StatusOr<int64_t> HeapSimulator::MinimumMemoryForComputation(
    const HloComputation& computation, const HloInstructionSequence& sequence,
    const HloAliasAnalysis& alias_analysis,
    const LogicalBuffer::SizeFunction& size_function,
    const HloSchedule* schedule) {
  TF_ASSIGN_OR_RETURN(
      HeapSimulator::Result<HloValue> result,
      HeapSimulator::Run(std::make_unique<NoFragmentationStatsHeap<HloValue>>(),
                         computation, sequence, alias_analysis, size_function,
                         schedule, HeapSimulator::Options()));
  return result.heap_size;
}

/*static*/
absl::StatusOr<HeapSimulator::Result<HloValue>> HeapSimulator::Run(
    std::unique_ptr<HeapAlgorithm<HloValue>> algorithm, const HloModule& module,
    const HloSchedule& schedule, const HloAliasAnalysis& alias_analysis,
    const BufferValue::SizeFunction& size_fn, const Options& options) {
  HeapSimulator heap(std::move(algorithm), size_fn, options, &schedule);
  const HloComputation* entry_computation = module.entry_computation();
  const HloInstructionSequence& instruction_sequence =
      schedule.sequence(entry_computation);
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloLiveRange> hlo_live_range,
      HloLiveRange::Run(schedule, alias_analysis, entry_computation));
  TF_RETURN_IF_ERROR(heap.RunComputation(*entry_computation,
                                         instruction_sequence, alias_analysis,
                                         hlo_live_range.get()));
  return heap.Finish();
}

/*static*/
absl::StatusOr<HeapSimulator::Result<HloValue>> HeapSimulator::Run(
    std::unique_ptr<HeapAlgorithm<HloValue>> algorithm,
    const HloComputation& computation,
    const HloInstructionSequence& instruction_sequence,
    const HloAliasAnalysis& alias_analysis,
    const BufferValue::SizeFunction& size_fn, const Options& options,
    const absl::flat_hash_map<const HloComputation*, int64_t>*
        memory_by_computation) {
  HeapSimulator heap(std::move(algorithm), size_fn, options,
                     /*schedule=*/nullptr, memory_by_computation);
  HloSchedule schedule(computation.parent());
  schedule.set_sequence(&computation, instruction_sequence);
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloLiveRange> hlo_live_range,
                      HloLiveRange::Run(schedule, alias_analysis, &computation,
                                        /*module_scoped_analysis=*/false));
  TF_RETURN_IF_ERROR(heap.RunComputation(computation, instruction_sequence,
                                         alias_analysis, hlo_live_range.get()));
  return heap.Finish();
}

/*static*/
absl::StatusOr<HeapSimulator::Result<HloValue>> HeapSimulator::Run(
    std::unique_ptr<HeapAlgorithm<HloValue>> algorithm,
    const HloComputation& computation,
    const HloInstructionSequence& instruction_sequence,
    const HloAliasAnalysis& alias_analysis,
    const BufferValue::SizeFunction& size_fn, const HloSchedule* schedule,
    const Options& options) {
  HeapSimulator heap(std::move(algorithm), size_fn, options,
                     /*schedule=*/schedule, nullptr);
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloLiveRange> hlo_live_range,
      HloLiveRange::Run(*schedule, alias_analysis, &computation));
  TF_RETURN_IF_ERROR(heap.RunComputation(computation, instruction_sequence,
                                         alias_analysis, hlo_live_range.get()));
  return heap.Finish();
}

// Runs a heap simulation for the given 'computation', assuming the given
// 'instruction_sequence'.
absl::Status HeapSimulator::RunComputation(
    const HloComputation& computation,
    const HloInstructionSequence& instruction_sequence,
    const HloAliasAnalysis& alias_analysis, HloLiveRange* hlo_live_range) {
  XLA_VLOG_LINES(1, computation.parent()->ToString());
  XLA_VLOG_LINES(2, computation.ToString());

  VLOG(1) << hlo_live_range->ToString();

  HloDataflowAnalysis& dataflow_analysis = alias_analysis.dataflow_analysis();

  // Record the buffer define/free event for each time step. We free all
  // remaining buffers (entry parameter, etc) after the program has finished
  // running, so we set the size of to program_end_time + 1.
  std::vector<std::vector<const HloValue*>> buffers_defined(
      hlo_live_range->schedule_end_time() + 1);
  std::vector<std::vector<const HloValue*>> buffers_freed(
      hlo_live_range->schedule_end_time() + 1);

  // values_to_assign tracks the HloValues that we need to assign a buffer to.
  // Note that we only need to assign a buffer to a value when both of the
  // following conditions are met:
  //
  // - The user specifically asks us to assign a buffer to a set of HloValues,
  // and the value is in the set. If the user don't provide such a set, by
  // default we assign buffer to all HloValues.
  //
  // - If the instruction is in a nested call of the current computation, only
  // assign a buffer if we are doing global heap simulation.
  std::vector<const HloValue*> values_to_assign;
  values_to_assign.reserve(dataflow_analysis.values().size());

  auto& buffer_live_ranges = hlo_live_range->buffer_live_ranges();

  for (const HloValue* value : dataflow_analysis.values()) {
    // Ignore buffers that are not tracked.
    if (!buffer_live_ranges.contains(value)) {
      continue;
    }
    if (IgnoreBuffer(value)) {
      continue;
    }

    values_to_assign.push_back(value);
  }

  absl::c_sort(values_to_assign,
               [&](const HloValue* value1, const HloValue* value2) {
                 const auto& live_range1 = buffer_live_ranges.at(value1);
                 const auto& live_range2 = buffer_live_ranges.at(value2);
                 return std::forward_as_tuple(live_range1.start,
                                              live_range1.end, value1->id()) <
                        std::forward_as_tuple(live_range2.start,
                                              live_range2.end, value2->id());
               });

  // For each value that we need to assign a buffer to, add the define and free
  // events.
  for (const HloValue* value : values_to_assign) {
    auto live_range = buffer_live_ranges.at(value);
    buffers_defined[live_range.start].push_back(value);
    buffers_freed[live_range.end].push_back(value);
  }

  // All HloValues in a hlo buffer should be allocated to the same address. This
  // map tracks the first value that got allocated in a buffer.
  absl::flat_hash_map<const HloBuffer*, const HloValue*> first_allocated_value;

  VLOG(1) << "Program time" << hlo_live_range->schedule_end_time();

  // Populate buffer sizes with the maximum size of the constituent HloValues.
  for (const HloBuffer& buffer : alias_analysis.buffers()) {
    int64_t size = 0;
    for (const HloValue* value : buffer.values()) {
      size = std::max(size, size_fn_(*value));
    }
    for (const HloValue* value : buffer.values()) {
      buffer_sizes_[value] = size;
    }
  }

  // Go through each step in the program and replay each buffer define and free
  // events.
  for (int64_t i = 0; i < hlo_live_range->schedule_end_time() + 1; ++i) {
    VLOG(1) << "Time step: " << i;

    for (const HloValue* value : buffers_defined[i]) {
      bool shared = false;
      VLOG(1) << "Start buffer: " << value->ToShortString();
      const HloBuffer* hlo_buffer =
          &alias_analysis.GetBufferContainingValue(*value);
      if (first_allocated_value.count(hlo_buffer) != 0) {
        // We've already assigned an address for another value in this HloBuffer
        // (HloBuffer holds several aliased HloValues). All values in a buffer
        // should be assigned the same address. Find the one that's already
        // allocated and reuse its address.
        ShareBuffer(value, first_allocated_value[hlo_buffer],
                    value->instruction());
        VLOG(1) << "  ShareWith"
                << first_allocated_value[hlo_buffer]->ToShortString();
        continue;
      }
      if (options_.may_reuse_operand_buffers &&
          hlo_buffer->values().size() == 1) {
        // We don't support sharing an aliased buffer
        // (hlo_buffer->values().size() > 1) with its operand.
        for (const HloInstruction* operand : value->instruction()->operands()) {
          const HloValueSet operand_value_set =
              dataflow_analysis.GetValueSet(operand);
          for (const HloValue* operand_value : operand_value_set.values()) {
            const HloBuffer* operand_buffer =
                &alias_analysis.GetBufferContainingValue(*operand_value);
            if (operand_buffer->values().size() > 1) {
              continue;
            }
            auto it = buffer_live_ranges.find(operand_value);
            if (it == buffer_live_ranges.end()) {
              continue;
            }

            auto& operand_live_range = it->second;

            auto& user_live_range = buffer_live_ranges[value];

            // Can only share buffers that are about to be freed.
            if (operand_live_range.end != i) {
              continue;
            }

            if (IgnoreBuffer(operand_value)) {
              continue;
            }

            if (!absl::c_linear_search(buffers_freed[i], operand_value)) {
              // If the operand buffer is not being freed (either because it has
              // existing users, or it has been reused by other buffers), don't
              // consider the operand as a candidate of buffer sharing.
              continue;
            }

            // The instruction that defines the operand value can be different
            // from the actual operand, if directly passing the defining
            // instruction into "CanShareOperandBufferWithUser" it creates a
            // check failure. The first condition guards against that case.
            if (value->instruction()->IsUserOf(operand_value->instruction()) &&
                value->instruction()->opcode() != HloOpcode::kCopy &&
                dataflow_analysis.CanShareOperandBufferWithUser(
                    operand_value->instruction(), operand_value->index(),
                    value->instruction(), value->index())) {
              // Remove the operand buffer right before sharing (allocating) a
              // new one.
              Free(operand_value, operand_value->instruction());
              buffers_freed[i].erase(
                  std::remove(buffers_freed[i].begin(), buffers_freed[i].end(),
                              operand_value),
                  buffers_freed[i].end());
              ShareBuffer(value, operand_value, value->instruction());
              // The live range of the operand buffer is now extended to the end
              // of the current instruction.
              operand_live_range.end = user_live_range.end;
              VLOG(1) << "Sharing " << value->ToShortString() << " with "
                      << operand_value->ToShortString()
                      << ", size:" << size_fn_(*value);
              shared = true;
              break;
            }
          }
          if (shared) {
            break;
          }
        }
      }
      if (!shared) {
        Alloc(value, value->instruction());
        first_allocated_value[hlo_buffer] = value;
      }
    }

    if (!buffers_freed[i].empty()) {
      VLOG(1) << "Free Buffer: ";
    }
    for (const HloValue* value : buffers_freed[i]) {
      VLOG(1) << "  " << value->ToShortString();

      Free(value, value->instruction());
    }
  }
  return absl::OkStatus();
}

HeapSimulator::HeapSimulator(
    std::unique_ptr<HeapAlgorithm<HloValue>> algorithm,
    const BufferValue::SizeFunction& size_fn, const Options& options,
    const HloSchedule* schedule,
    const absl::flat_hash_map<const HloComputation*, int64_t>*
        memory_by_computation)
    : no_fragmentation_stats_(
          std::make_unique<NoFragmentationStatsHeap<HloValue>>()),
      algorithm_(std::move(algorithm)),
      size_fn_(size_fn),
      options_(options),
      schedule_(schedule),
      memory_by_computation_(memory_by_computation) {
  debug_trace_.set_whole_module_simulation(schedule_ != nullptr);
}

HeapSimulator::~HeapSimulator() {}

bool HeapSimulator::IgnoreBuffer(const HloValue* buffer) const {
  // Buffers for constants are ignored unless the alloc_constants option is
  // set. Also ignore buffers that we're not meant to assign.
  //
  // TODO(b/32248867): For consistency, constants should get allocations.
  if (!options_.alloc_constants &&
      buffer->instruction()->opcode() == HloOpcode::kConstant) {
    return true;
  }
  return options_.buffers_to_assign != nullptr &&
         !options_.buffers_to_assign->contains(buffer);
}

// Alloc always calls the underlying heap algorithm.
void HeapSimulator::Alloc(const HloValue* buffer,
                          const HloInstruction* instruction) {
  CHECK(!allocated_buffers_.contains(buffer))
      << "Alloc called on allocated buffer: " << *buffer;
  CHECK(!freed_buffers_.contains(buffer))
      << "Alloc called on freed buffer: " << *buffer;

  allocated_buffers_.insert(buffer);
  const int64_t size = GetBufferSize(buffer);
  algorithm_->Alloc(buffer, size);
  no_fragmentation_stats_->Alloc(buffer, size);
  FillDebugTrace(HeapSimulatorTrace::Event::ALLOC, buffer, instruction,
                 nullptr);
}

// Free calls the underlying algorithm for non-shared buffers, and for shared
// buffers whose group liveness has expired.  Shared group liveness is tracked
// by maintaining a refcount; the Free call on the last buffer in the group
// causes Free to be called on the underlying algorithm.
void HeapSimulator::Free(const HloValue* buffer,
                         const HloInstruction* instruction) {
  const int64_t size = GetBufferSize(buffer);
  algorithm_->Free(buffer, size);
  no_fragmentation_stats_->Free(buffer, size);
  FillDebugTrace(HeapSimulatorTrace::Event::FREE, buffer, instruction, nullptr);
}

// ShareBuffer associates buffers with their SharedGroup in shared_buffers_.
// The 'buffer' must be a non-allocated, non-freed buffer, just like in calls
// to Alloc.  The 'shared' buffer must be a previously allocated or shared
// buffer. Both 'buffer' and 'shared' will be associated with the same
// SharedGroup.
void HeapSimulator::ShareBuffer(const HloValue* buffer, const HloValue* shared,
                                const HloInstruction* instruction) {
  algorithm_->ShareWith(buffer, shared, GetBufferSize(shared));
  no_fragmentation_stats_->ShareWith(buffer, shared, GetBufferSize(shared));
  FillDebugTrace(HeapSimulatorTrace::Event::SHARE_WITH, buffer, instruction,
                 shared);
}

int64_t HeapSimulator::GetBufferSize(const HloValue* buffer) const {
  auto it = buffer_sizes_.find(buffer);
  CHECK(it != buffer_sizes_.end());
  return it->second;
}

absl::StatusOr<HeapSimulator::Result<HloValue>> HeapSimulator::Finish() {
  TF_ASSIGN_OR_RETURN(Result<HloValue> result, algorithm_->Finish());

  // Post-process the result to add chunks for shared buffers.  An empty chunk
  // map means that either no buffers were allocated, or the heap was only
  // collecting statistics, e.g. NoFragmentationStatsHeap.
  size_t total_chunk_count = absl::c_accumulate(
      result.heap_results, static_cast<size_t>(0),
      [&](size_t lhs, const HeapResult<HloValue>& rhs) -> size_t {
        return lhs + rhs.chunk_map.size();
      });
  if (total_chunk_count != 0) {
    // If we were told to assign specific buffers, make sure we've assigned
    // exactly that many buffers.
    if (options_.buffers_to_assign != nullptr) {
      CHECK_EQ(options_.buffers_to_assign->size(), total_chunk_count);
    }
  }

  // Fragmentation is the difference between the actual and ideal sizes.
  TF_ASSIGN_OR_RETURN(const Result<HloValue> no_frag_result,
                      no_fragmentation_stats_->Finish());
  result.fragmentation_size = result.heap_size - no_frag_result.heap_size;

  // Copy the debug trace we collected to the final result.
  result.debug_trace.Swap(&debug_trace_);

  return result;
}

void HeapSimulator::FillDebugTrace(HeapSimulatorTrace::Event::Kind kind,
                                   const HloValue* buffer,
                                   const HloInstruction* instruction,
                                   const HloValue* share_with_canonical) {
  HeapSimulatorTrace::Event* event = debug_trace_.add_events();
  event->set_kind(kind);
  event->set_buffer_id(buffer->id());
  *event->mutable_computation_name() =
      std::string(instruction->parent()->name());
  *event->mutable_instruction_name() = std::string(instruction->name());
  if (kind == HeapSimulatorTrace::Event::SHARE_WITH) {
    CHECK(share_with_canonical != nullptr);
    event->set_share_with_canonical_id(share_with_canonical->id());
  } else {
    CHECK(share_with_canonical == nullptr);
  }
}

template <typename BufferType>
void NoFragmentationStatsHeap<BufferType>::Alloc(const BufferType* buffer,
                                                 int64_t size) {
  current_heap_size_ += size;
  if (current_heap_size_ > max_heap_size_) {
    max_heap_size_ = current_heap_size_;
  }
}

template <typename BufferType>
void NoFragmentationStatsHeap<BufferType>::AccountForSubcomputationMemory(
    const HloInstruction* instruction, int64_t alloc_size_by_instruction,
    const absl::flat_hash_map<const HloComputation*, int64_t>&
        memory_by_computation) {
  // We only count the memory usage of the largest subcomputation, instead of
  // adding them all, because subcomputations won't execute in parallel.
  int64_t max_subcomputation_bytes = 0;
  for (const auto* c : instruction->called_computations()) {
    auto it = memory_by_computation.find(c);
    if (it != memory_by_computation.end()) {
      int64_t subcomputation_bytes = it->second;
      if (subcomputation_bytes > max_subcomputation_bytes) {
        max_subcomputation_bytes = subcomputation_bytes;
      }
    }
  }
  if (max_subcomputation_bytes > 0 &&
      (instruction->opcode() == HloOpcode::kWhile ||
       instruction->opcode() == HloOpcode::kCall ||
       instruction->opcode() == HloOpcode::kConditional)) {
    // The output buffer of while/call/conditional is always aliased with the
    // output buffer of the root instruction in the body. Don't double count.
    max_subcomputation_bytes -= alloc_size_by_instruction;
  }
  max_heap_size_ =
      std::max(max_heap_size_, current_heap_size_ + max_subcomputation_bytes);
}

template <typename BufferType>
void NoFragmentationStatsHeap<BufferType>::Free(const BufferType* buffer,
                                                int64_t size) {
  current_heap_size_ -= size;
}

template <typename BufferType>
absl::StatusOr<HeapSimulator::Result<BufferType>>
NoFragmentationStatsHeap<BufferType>::Finish() {
  // The result.chunk_map is empty, since we only collect stats, and don't
  // actually compute chunk assignments.
  Result result;
  result.heap_size = max_heap_size_;
  return result;
}

template <typename BufferType>
GlobalDecreasingSizeBestFitHeap<BufferType>::GlobalDecreasingSizeBestFitHeap(
    int64_t alignment, Type type, BufferIntervalCompare buffer_interval_compare,
    SliceTimePermutationIterator::Ty slice_time_permutation_iterator_type)
    : alignment_(alignment),
      slice_time_permutation_iteration_type_(
          slice_time_permutation_iterator_type) {
  if (type == kTemporal) {
    buffer_interval_compare_ = GetTemporalBufferIntervalCompare();
    CHECK(buffer_interval_compare == nullptr);
  } else if (type == kSpatial) {
    buffer_interval_compare_ = GetSpatialBufferIntervalCompare();
    CHECK(buffer_interval_compare == nullptr);
  } else {
    CHECK(type == kCustom);
    CHECK(buffer_interval_compare != nullptr);
    buffer_interval_compare_ = buffer_interval_compare;
  }
}

template <typename BufferType>
typename GlobalDecreasingSizeBestFitHeap<BufferType>::BufferIntervalCompare
GlobalDecreasingSizeBestFitHeap<BufferType>::GetTemporalBufferIntervalCompare()
    const {
  return LessThanByKey([this](const BufferInterval& x) {
    int64_t x_end = x.end;
    for (auto colocation : GetTransitiveColocations(x)) {
      x_end = std::max(x_end, buffer_intervals_.at(colocation).end);
    }
    // Sort by duration (descending), size (descending), buffer (ascending).
    return std::make_tuple(x.start - x_end, -x.size, std::cref(*x.buffer));
  });
}

template <typename BufferType>
SliceTimePermutationIterator::Ty GlobalDecreasingSizeBestFitHeap<
    BufferType>::slice_time_permutation_iterator_type() const {
  return slice_time_permutation_iteration_type_;
}

template <typename BufferType>
/*static*/ typename GlobalDecreasingSizeBestFitHeap<
    BufferType>::BufferIntervalCompare
GlobalDecreasingSizeBestFitHeap<BufferType>::GetSpatialBufferIntervalCompare() {
  return LessThanByKey([](const BufferInterval& x) {
    // Sort by size (descending), duration (descending), buffer (ascending).
    return std::make_tuple(-x.size, x.start - x.end, std::cref(*x.buffer));
  });
}

template <typename BufferType>
void GlobalDecreasingSizeBestFitHeap<BufferType>::Alloc(
    const BufferType* buffer, int64_t size) {
  // Degenerate case: 0-sized buffers are always allocated at offset 0.
  if (size == 0) {
    result_.chunk_map.emplace(buffer, Chunk::FromOffsetSize(0, 0));
    return;
  }

  auto emplace_result = buffer_intervals_.emplace(
      buffer, BufferInterval{buffer, size, current_time_, -1, {}, true});
  CHECK(emplace_result.second);
  ++current_time_;
}

template <typename BufferType>
void GlobalDecreasingSizeBestFitHeap<BufferType>::ShareWith(
    const BufferType* buffer, const BufferType* share_with, int64_t size) {
  // Degenerate case: 0-sized buffers are always allocated at offset 0.
  if (size == 0) {
    result_.chunk_map.emplace(buffer, Chunk::FromOffsetSize(0, 0));
    return;
  }
  CHECK_NE(buffer_intervals_.count(share_with), 0);
  buffer_intervals_[share_with].colocations.push_back(buffer);
  auto emplace_result = buffer_intervals_.emplace(
      buffer, BufferInterval{buffer, size, current_time_, -1, {}, false});
  CHECK(emplace_result.second);
  ++current_time_;
}

template <typename BufferType>
absl::flat_hash_set<const BufferType*>
GlobalDecreasingSizeBestFitHeap<BufferType>::GetTransitiveColocations(
    const BufferInterval& interval) const {
  absl::flat_hash_set<const BufferType*> result;
  std::vector<const BufferInterval*> worklist = {&interval};
  while (!worklist.empty()) {
    const BufferInterval* item = worklist.back();
    worklist.pop_back();
    for (const BufferType* buffer_colocated : item->colocations) {
      if (result.insert(buffer_colocated).second) {
        worklist.push_back(&buffer_intervals_.at(buffer_colocated));
      }
    }
  }

  return result;
}

template <typename BufferType>
void GlobalDecreasingSizeBestFitHeap<BufferType>::Free(const BufferType* buffer,
                                                       int64_t size) {
  // Degenerate case: 0-sized buffers are always allocated at offset 0.
  if (size == 0) {
    return;
  }
  BufferInterval& buffer_interval = FindOrDie(buffer_intervals_, buffer);
  CHECK_EQ(buffer_interval.buffer, buffer);
  CHECK_EQ(buffer_interval.size, size);
  CHECK_EQ(buffer_interval.end, -1);
  if (buffer_interval.end != -1) {
    return;
  }
  buffer_interval.end = current_time_;
  ++current_time_;
}

using Chunk = HeapSimulator::Chunk;

void BufferIntervalTree::Add(int64_t start, int64_t end, const Chunk& chunk) {
  node_storage_.emplace_back(BufferIntervalTreeNode{
      start, end, end, chunk,
      /*left=*/nullptr, /*right=*/nullptr, /*parent=*/nullptr});
  if (root_ == nullptr) {
    root_ = &node_storage_.back();
    // This is root.
    return;
  }

  BufferIntervalTreeNode* parent = root_;
  while (true) {
    parent->subtree_end = std::max(parent->subtree_end, end);
    if (parent->start > start) {
      if (parent->left == nullptr) {
        parent->left = &node_storage_.back();
        node_storage_.back().parent = parent;
        return;
      }
      parent = parent->left;
    } else {
      if (parent->right == nullptr) {
        parent->right = &node_storage_.back();
        node_storage_.back().parent = parent;
        return;
      }
      parent = parent->right;
    }
  }
}

bool BufferIntervalTree::Remove(int64_t start, int64_t end,
                                const Chunk& chunk) {
  BufferIntervalTreeNode* to_delete = root_;
  while (to_delete != nullptr) {
    if (to_delete->start == start && to_delete->end == end &&
        to_delete->chunk.offset == chunk.offset) {
      break;
    }
    if (start < to_delete->start) {
      to_delete = to_delete->left;
    } else {
      to_delete = to_delete->right;
    }
  }
  if (to_delete == nullptr) {
    // Nothing to delete.
    return false;
  }
  // Found the node to be deleted, enter deletion sequence.

  // Recursively traverse the parents of node and fix up the `subtree_end`
  // invariant of a node. Recursive lambda need an explicit
  // std::function declaration.
  std::function<void(BufferIntervalTreeNode*)> fix_up =
      [&](BufferIntervalTreeNode* node) {
        if (node == nullptr) {
          return;
        }
        node->subtree_end = node->end;
        if (node->left) {
          node->subtree_end =
              std::max(node->subtree_end, node->left->subtree_end);
        }
        if (node->right) {
          node->subtree_end =
              std::max(node->subtree_end, node->right->subtree_end);
        }
        // Recursively go up.
        fix_up(node->parent);
      };

  if (to_delete->right == nullptr) {
    // to_delete has no right child, simply move up left child of to_delete if
    // any.
    //
    // Turn:
    //      parent
    //       /
    // to_delete
    //  /      \
    // left    nullptr
    //
    // Into:
    //      parent
    //      /
    //    left
    if (root_ == to_delete) {
      // Deleting root is simply resetting root;
      root_ = to_delete->left;
      return true;
    }

    if (to_delete == to_delete->parent->left) {
      // to_delete is left child of parent.
      to_delete->parent->left = to_delete->left;
    }
    if (to_delete == to_delete->parent->right) {
      // to_delete is right child of parent.
      to_delete->parent->right = to_delete->left;
    }
    // Rewire parent to the node being moved up.
    if (to_delete->left) {
      to_delete->left->parent = to_delete->parent;
    }
    // Fix up starting from subroot.
    fix_up(to_delete);
  } else {
    // 1. Find left-most node of the right subtree, promote it to the position
    // of to_delete.
    BufferIntervalTreeNode* to_promote = to_delete->right;
    while (to_promote->left != nullptr) {
      // Go to left-most subtree.
      to_promote = to_promote->left;
    }

    // 2. Copy the content of `to_promote` to `to_delete`.
    to_delete->start = to_promote->start;
    to_delete->end = to_promote->end;
    // This is incorrect but we will fix this up later in the `fix_up`
    // procedure.
    to_delete->subtree_end = to_promote->subtree_end;
    to_delete->chunk = to_promote->chunk;
    auto to_promote_parent = to_promote->parent;
    // 3. Move the right child of `to_promote` up if there is any.
    //
    // Turn
    //
    // to_delete
    //         \
    //        to_promote_parent
    //         /
    //    to_promote
    //          \
    //          right
    // into
    //
    // to_promote
    //         \
    //         to_promote_parent
    //         /
    //      right
    if (to_promote_parent->left == to_promote) {
      to_promote_parent->left = to_promote->right;
    } else {
      to_promote_parent->right = to_promote->right;
    }
    if (to_promote->right) {
      // Set correct parent.
      to_promote->right->parent = to_promote_parent;
    }
    // 4. Recursive fix up the `subtree_end` starting from
    // `to_promote_parent`.
    fix_up(to_promote_parent);
  }
  // Don't free the entry in node_storage_ until we free the entire tree.
  return true;
}

std::vector<Chunk> BufferIntervalTree::ChunksOverlappingInTime(
    int64_t start, int64_t end) const {
  std::vector<Chunk> result;
  if (root_ == nullptr) {
    return result;
  }
  std::vector<const BufferIntervalTreeNode*> visiting_stack;
  visiting_stack.push_back(root_);
  while (!visiting_stack.empty()) {
    const BufferIntervalTreeNode* top = visiting_stack.back();
    visiting_stack.pop_back();
    if (start > top->subtree_end) {
      continue;
    }
    if (top->left != nullptr) {
      visiting_stack.push_back(top->left);
    }
    if (top->start <= end && top->end >= start) {
      result.push_back(top->chunk);
    }
    if (end < top->start) {
      continue;
    }
    if (top->right != nullptr) {
      visiting_stack.push_back(top->right);
    }
  }
  return result;
}

template <typename BufferType>
std::string
GlobalDecreasingSizeBestFitHeap<BufferType>::BufferInterval::ToString() const {
  return absl::StrCat("{ ",  //
                      "buffer: {", (buffer ? buffer->ToString() : "null"),
                      "}, ",                                          //
                      "size: ", size, ", ",                           //
                      "start: ", start, ", ",                         //
                      "end: ", end, ", ",                             //
                      "num_colocations: ", colocations.size(), ", ",  //
                      "need_allocation: ", need_allocation,           //
                      " }");
}

template <typename BufferType>
const  // NOLINT(readability-const-return-type)
    typename GlobalDecreasingSizeBestFitHeap<BufferType>::SlicedBufferInterval
    GlobalDecreasingSizeBestFitHeap<BufferType>::SlicedBufferInterval::
        CreateConstInterval(const BufferInterval& full_buffer_interval) {
  return SlicedBufferInterval(full_buffer_interval);
}

template <typename BufferType>
typename GlobalDecreasingSizeBestFitHeap<BufferType>::SlicedBufferInterval
GlobalDecreasingSizeBestFitHeap<BufferType>::SlicedBufferInterval::
    CreateMutableInterval(BufferInterval& full_buffer_interval) {
  return SlicedBufferInterval(full_buffer_interval, &full_buffer_interval);
}

template <typename BufferType>
void GlobalDecreasingSizeBestFitHeap<BufferType>::SlicedBufferInterval::Slice(
    absl::Span<const int64_t> slice_sizes_sorted_by_offset) {
  if (slice_sizes_sorted_by_offset.empty()) {
    slice_sizes_sorted_by_offset_ = {full_buffer_interval_.size};
    make_free_chunks_intervals_ = {full_buffer_interval_};
    return;
  }

  const int64_t min_slice_size =
      *absl::c_min_element(slice_sizes_sorted_by_offset);
  slice_sizes_sorted_by_offset_ = std::vector<int64_t>(
      slice_sizes_sorted_by_offset.begin(), slice_sizes_sorted_by_offset.end());

  size_t num_slices = slice_sizes_sorted_by_offset.size();
  make_free_chunks_intervals_.clear();
  make_free_chunks_intervals_.reserve(num_slices);

  int64_t size_total = 0;
  absl::InlinedVector<const BufferType*, 2> empty_colocations;
  for (int i = 0; i < num_slices; ++i) {
    int64_t new_size = slice_sizes_sorted_by_offset[i];
    size_total += new_size;
    make_free_chunks_intervals_.push_back(BufferInterval{
        full_buffer_interval_.buffer,
        /*size=*/
        (i == num_slices - 1 ? full_buffer_interval_.size : min_slice_size),
        /*start=*/0,
        /*end=*/full_buffer_interval_.end,
        /*colocations=*/
        (i == num_slices - 1 ? full_buffer_interval_.colocations
                             : empty_colocations),
        full_buffer_interval_.need_allocation});
  }

  CHECK_EQ(size_total, full_buffer_interval_.size)
      << " slice sizes: {" << absl::StrJoin(slice_sizes_sorted_by_offset, ", ")
      << "};";
}

template <typename BufferType>
void GlobalDecreasingSizeBestFitHeap<BufferType>::SlicedBufferInterval::
    UpdateExclusiveSliceStartTimes(
        const std::vector<int64_t>& exclusive_start_times) {
  std::vector<int64_t> inclusive_start_times = exclusive_start_times;
  absl::c_for_each(inclusive_start_times,
                   [](int64_t& t) { t = ExclusiveToInclusiveStartTime(t); });
  UpdateInclusiveSliceStartTimes(inclusive_start_times);
}

template <typename BufferType>
void GlobalDecreasingSizeBestFitHeap<BufferType>::SlicedBufferInterval::
    UpdateInclusiveSliceStartTimes(
        const std::vector<int64_t>& inclusive_start_times) {
  CHECK_EQ(inclusive_start_times.size(), num_slices());
  CHECK(mutable_full_buffer_interval_ != nullptr);
  mutable_full_buffer_interval_->start = inclusive_start_times.front();
  for (size_t slice_time = 0; slice_time < num_slices(); ++slice_time) {
    make_free_chunks_intervals_[slice_time].start =
        inclusive_start_times[slice_time];
    if (slice_time != num_slices() - 1) {
      make_free_chunks_intervals_[slice_time].end =
          ExclusiveToInclusiveEndTime(inclusive_start_times[slice_time + 1]);
    } else {
      make_free_chunks_intervals_[slice_time].end = full_buffer_interval_.end;
    }
  }
}

template <typename BufferType>
void GlobalDecreasingSizeBestFitHeap<
    BufferType>::SlicedBufferInterval::UpdateEndTime(int64_t end_time) {
  CHECK(mutable_full_buffer_interval_ != nullptr);
  mutable_full_buffer_interval_->end = end_time;
  make_free_chunks_intervals_.back().end = end_time;
}

template <typename BufferType>
const typename GlobalDecreasingSizeBestFitHeap<BufferType>::BufferInterval&
GlobalDecreasingSizeBestFitHeap<
    BufferType>::SlicedBufferInterval::full_buffer_interval() const {
  return full_buffer_interval_;
}

template <typename BufferType>
const std::vector<int64_t>& GlobalDecreasingSizeBestFitHeap<
    BufferType>::SlicedBufferInterval::SliceSizesSortedByOffset() const {
  return slice_sizes_sorted_by_offset_;
}

template <typename BufferType>
std::vector<int64_t> GlobalDecreasingSizeBestFitHeap<
    BufferType>::SlicedBufferInterval::inclusive_start_times() const {
  std::vector<int64_t> inclusive_start_times;
  inclusive_start_times.reserve(num_slices());
  for (const BufferInterval& buffer_interval : make_free_chunks_intervals_) {
    inclusive_start_times.push_back(buffer_interval.start);
  }

  return inclusive_start_times;
}

template <typename BufferType>
const typename GlobalDecreasingSizeBestFitHeap<BufferType>::BufferInterval&
GlobalDecreasingSizeBestFitHeap<BufferType>::SlicedBufferInterval::
    IntervalForMakeFreeChunks(int64_t slice_time) const {
  CHECK_LT(slice_time, num_slices());
  return make_free_chunks_intervals_[slice_time];
}

template <typename BufferType>
GlobalDecreasingSizeBestFitHeap<BufferType>::SlicedBufferInterval::
    SlicedBufferInterval(const BufferInterval& full_buffer_interval,
                         BufferInterval* mutable_full_buffer_interval)
    : full_buffer_interval_(full_buffer_interval),
      mutable_full_buffer_interval_(mutable_full_buffer_interval) {
  // Start with 1 slice. Slice() will initialize the remaining data members.
  Slice({});
}

template <typename BufferType>
std::string GlobalDecreasingSizeBestFitHeap<
    BufferType>::SlicedBufferInterval::ToString() const {
  return absl::StrCat(
      "{ full_buffer_interval: ", full_buffer_interval_.ToString(), ", ",
      "MakeFreeChunks intervals: { ",
      absl::StrJoin(make_free_chunks_intervals_, ", ",
                    [](std::string* out, const BufferInterval& interval) {
                      absl::StrAppend(out, interval.ToString());
                    }),
      " }, ", "slize_sizes_sorted_by_offsets: { ",
      absl::StrJoin(slice_sizes_sorted_by_offset_, ", "), " } }");
}

namespace {

// A class that indicates if a permutation of starting slice times is valid. See
// SliceTimePermutationIterator for the meaning of slice time permutations.
//
// In non-repacking scenarios, all slices are valid. In repacking scenarios,
// a permutation is invalid if it does not maintain the mapping between slice
// times and slice sizes of the original placement.
class SliceTimePermutationValidator {
 public:
  explicit SliceTimePermutationValidator(
      const SlicedAllocationData* original_slices)
      : original_num_slices_(original_slices ? original_slices->num_slices()
                                             : 0) {
    if (original_num_slices_ <= 0) {
      return;
    }
    slice_time_to_inclusive_schedule_time_ =
        original_slices->SortedInclusiveStartTimes();
    absl::c_sort(slice_time_to_inclusive_schedule_time_);

    original_slice_sizes_and_start_times_pairwise_sorted_.reserve(
        original_num_slices_);
    for (const AllocatedSlice& slice :
         original_slices->slices_sorted_by_offset) {
      original_slice_sizes_and_start_times_pairwise_sorted_.push_back(
          std::make_pair(slice.size, slice.inclusive_start_time));
    }
    absl::c_sort(original_slice_sizes_and_start_times_pairwise_sorted_);

    sizes_sorted_by_offset_ = original_slices->SizesSortedByOffset();
  }

  bool IsValid(absl::Span<const int64_t> permutation) {
    if (original_num_slices_ <= 0) {
      return true;
    }

    // Compute the slice size to slice start time mapping proposed by the
    // permutation.
    std::vector<std::pair<int64_t, int64_t>>
        proposed_slice_sizes_and_start_times_pairwise_sorted;
    proposed_slice_sizes_and_start_times_pairwise_sorted.reserve(
        original_num_slices_);
    CHECK_EQ(sizes_sorted_by_offset_.size(), original_num_slices_);
    CHECK_EQ(permutation.size(), original_num_slices_);
    for (int i = 0; i < original_num_slices_; ++i) {
      proposed_slice_sizes_and_start_times_pairwise_sorted.push_back(
          std::make_pair(
              sizes_sorted_by_offset_[i],
              slice_time_to_inclusive_schedule_time_[permutation[i]]));
    }
    absl::c_sort(proposed_slice_sizes_and_start_times_pairwise_sorted);

    bool allowed = (original_slice_sizes_and_start_times_pairwise_sorted_ ==
                    proposed_slice_sizes_and_start_times_pairwise_sorted);
    VLOG(3) << [&]() {
      auto export_pair = [](std::string* out,
                            const std::pair<int64_t, int64_t>& p) {
        absl::StrAppend(out, "<", p.first, ", ", p.second, ">");
      };
      return absl::StrCat(
          "Slice permutation ", (allowed ? "allowed" : "disallowed"),
          ". Original slice <size, start_time> mapping: ",
          absl::StrJoin(original_slice_sizes_and_start_times_pairwise_sorted_,
                        ", ", export_pair),
          ". Proposed mapping: ",
          absl::StrJoin(proposed_slice_sizes_and_start_times_pairwise_sorted,
                        ", ", export_pair),
          ".");
    }();

    return allowed;
  }

 private:
  int64_t original_num_slices_;

  // The original allocation mapping from slice times to schedule times.
  std::vector<int64_t> slice_time_to_inclusive_schedule_time_;

  std::vector<std::pair<int64_t, int64_t>>
      original_slice_sizes_and_start_times_pairwise_sorted_;

  std::vector<int64_t> sizes_sorted_by_offset_;
};

// A manager class that tracks if we've already observed an equivalent
// permutation. See the description of SliceTimePermutationIterator for a
// definition of permutation equivalence.
class ObservedPermutationManager {
 public:
  explicit ObservedPermutationManager(
      absl::Span<const int64_t> inclusive_start_times) {
    slice_time_to_inclusive_start_time_ = std::vector<int64_t>(
        inclusive_start_times.begin(), inclusive_start_times.end());
    absl::c_sort(slice_time_to_inclusive_start_time_);
  }

  // Returns true if an equivalent permutation was already seen. If false is
  // returned, we track that we've now observed permutation.
  bool Insert(absl::Span<const int64_t> permutation) {
    std::vector<int64_t> permutation_inclusive_start_times;
    permutation_inclusive_start_times.reserve(permutation.size());
    for (int64_t slice_time : permutation) {
      permutation_inclusive_start_times.push_back(
          slice_time_to_inclusive_start_time_[slice_time]);
    }

    return observed_inclusive_start_time_permutation_
        .insert(permutation_inclusive_start_times)
        .second;
  }

  void Clear() { observed_inclusive_start_time_permutation_.clear(); }

 protected:
  std::vector<int64_t> slice_time_to_inclusive_start_time_;
  absl::flat_hash_set<std::vector<int64_t>>
      observed_inclusive_start_time_permutation_;
};

// A SliceTimePermutationIterator that iterates over all valid (see
// SliceTimePermutationValidator for more details) permutations of slice times.
class SliceTimeAllPermutationIterator : public SliceTimePermutationIterator {
 public:
  explicit SliceTimeAllPermutationIterator(int64_t num_slices)
      : num_slices_(num_slices), permutation_(num_slices, 0) {}

  ~SliceTimeAllPermutationIterator() override = default;

  void Begin() override {
    done_ = (num_slices_ <= 0);

    for (int64_t i = 0; i < num_slices_; ++i) {
      permutation_[i] = i;
    }
  }

  bool Done() const override { return done_; }

  void Next() override {
    if (Done()) {
      return;
    }
    done_ = !absl::c_next_permutation(permutation_);
  }

  absl::Span<const int64_t> Get() const override { return permutation_; }

 private:
  SliceTimeAllPermutationIterator() = default;

  int64_t num_slices_;
  bool done_ = true;
  std::vector<int64_t> permutation_;
};

// A SliceTimePermutationIterator that iterates over "preferred" shapes, as
// described in SliceTimePermutationIterator::Ty::kPreferred. When we have
// original sliced allocation data available (from a repack), before
// generating preferred permutation, we fix the slice time of any slice whose
// size is different from the first slice. We fix the slice time for such slices
// to their slice times in the original sliced data. Doing so avoids generating
// invalid permutations (as defined in SliceTimePermutationIterator).
//
// Note, in repacking situations, we don't know the exact slice time that each
// slice was assigned. We only know the inclusive start time of each slice.
// This gives us the slice time, except in cases where 2 slices have the same
// inclusive slice time. We choose to break such ties using offset, which is
// fine because it doesn't hurt performance.
class SliceTimePreferredPermutationIterator
    : public SliceTimePermutationIterator {
 public:
  SliceTimePreferredPermutationIterator(
      int64_t num_slices,
      const SlicedAllocationData* original_sliced_allocation)
      : num_slices_(num_slices),
        fixed_permutation_values_(num_slices, false),
        permutation_(num_slices, 0) {
    // In the body of the constructor we need to:
    // - If original_sliced_allocation is specified, we update
    //   fixed_permutation_values_ and permutation_ accordingly
    // - Initialize slice_times_available_for_permutation_.

    if (!original_sliced_allocation) {
      // If there are no original slice times, then any slice time can appear
      // at any permutation index.
      slice_times_available_for_permutation_.reserve(num_slices_);
      for (int64_t slice_time = 0; slice_time < num_slices_; ++slice_time) {
        slice_times_available_for_permutation_.push_back(slice_time);
      }
      return;
    }

    absl::flat_hash_map<const AllocatedSlice*, int64_t>
        slice_to_slice_time_map =
            BuildSliceToSliceTimeMap(original_sliced_allocation);
    const AllocatedSlice* first_slice = nullptr;
    if (!original_sliced_allocation->slices_sorted_by_offset.empty()) {
      first_slice =
          &original_sliced_allocation->slices_sorted_by_offset.front();
    }
    for (int offset_index = 0; offset_index < num_slices_; ++offset_index) {
      CHECK(first_slice);
      const AllocatedSlice& slice =
          original_sliced_allocation->slices_sorted_by_offset[offset_index];
      if (slice.size != first_slice->size) {
        fixed_permutation_values_[offset_index] = true;
        permutation_[offset_index] = slice_to_slice_time_map[&slice];
        continue;
      }
      slice_times_available_for_permutation_.push_back(
          slice_to_slice_time_map[&slice]);
    }
    absl::c_sort(slice_times_available_for_permutation_);
  }

  ~SliceTimePreferredPermutationIterator() override = default;

  void Begin() override {
    permutation_type_ = NextPermutationType(PermutationType::kUninitialized);
    SetUpPermutationForCurrentType();
  }

  bool Done() const override {
    return permutation_type_ == PermutationType::kDone;
  }

  void Next() override {
    permutation_type_ = NextPermutationType(permutation_type_);
    SetUpPermutationForCurrentType();
  }

  absl::Span<const int64_t> Get() const override { return permutation_; }

 private:
  enum class PermutationType {
    kUninitialized,
    // space
    //   ^
    //   |             +--+
    //   |          +--+  |
    //   |       +--+     |
    //   |    +--+        |
    //   | +--+           |
    //   | +--------------+
    //   +------------------> time
    kSmallerOffsetSmallerSliceTime,
    // space
    //   ^
    //   | +--------------+
    //   | +--+           |
    //   |    +--+        |
    //   |       +--+     |
    //   |          +--+  |
    //   |             +--+
    //   +------------------> time
    kSmallerOffsetLargerSliceTime,
    // space
    //   ^
    //   |             +--+
    //   |       +-----+  |
    //   | +-----+        |
    //   | +--+           |
    //   |    +-----+     |
    //   |          +-----+
    //   +------------------> time
    kDistributeSmallSliceTimesAroundMiddleOffset,
    kDone,
  };

  SliceTimePreferredPermutationIterator() = default;

  // Increments from one PermutationType to the next. Note, we skip some
  // PermutationTypes if the number of slices is small enough to make some
  // PermutationTypes generate the same permutation.
  PermutationType NextPermutationType(PermutationType ty) {
    switch (ty) {
      case PermutationType::kUninitialized:
        if (num_slices_ <= 0) {
          return PermutationType::kDone;
        }
        return PermutationType::kSmallerOffsetSmallerSliceTime;
      case PermutationType::kSmallerOffsetSmallerSliceTime:
        if (num_slices_ <= 1) {
          return PermutationType::kDone;
        }
        return PermutationType::kSmallerOffsetLargerSliceTime;
      case PermutationType::kSmallerOffsetLargerSliceTime:
        if (num_slices_ <= 2) {
          return PermutationType::kDone;
        }
        return PermutationType::kDistributeSmallSliceTimesAroundMiddleOffset;
      case PermutationType::kDistributeSmallSliceTimesAroundMiddleOffset:
      case PermutationType::kDone:
        return PermutationType::kDone;
    }
  }

  // Maps slices in original_sliced_allocation to their slice time.
  //
  // REQUIRES:
  // - original_sliced_allocation may not be null
  absl::flat_hash_map<const AllocatedSlice*, int64_t> BuildSliceToSliceTimeMap(
      const SlicedAllocationData* original_sliced_allocation) {
    CHECK(original_sliced_allocation);
    std::vector<const AllocatedSlice*> slice_time_to_slice;
    slice_time_to_slice.reserve(num_slices_);
    for (const AllocatedSlice& slice :
         original_sliced_allocation->slices_sorted_by_offset) {
      slice_time_to_slice.push_back(&slice);
    }
    absl::c_sort(slice_time_to_slice, [](const AllocatedSlice* lhs,
                                         const AllocatedSlice* rhs) {
      return std::make_tuple(lhs->inclusive_start_time, lhs->offset) <
             std::make_tuple(rhs->inclusive_start_time, rhs->offset);
    });

    absl::flat_hash_map<const AllocatedSlice*, int64_t> map;
    for (int slice_time = 0; slice_time < slice_time_to_slice.size();
         ++slice_time) {
      map[slice_time_to_slice[slice_time]] = slice_time;
    }

    return map;
  }

  // Builds permutation_ according to permutation_type_.
  //
  // REQUIRES:
  // - permutation_type_ != kUninitialized
  void SetUpPermutationForCurrentType() {
    CHECK(permutation_type_ != PermutationType::kUninitialized);
    if (Done()) {
      return;
    }

    int permutation_index = NextAvailablePermutationIndex(-1);

    for (int i = slice_times_available_for_permutation_.size() - 1; i >= 0;
         --i) {
      if (permutation_type_ == PermutationType::kSmallerOffsetLargerSliceTime ||
          (permutation_type_ ==
               PermutationType::kDistributeSmallSliceTimesAroundMiddleOffset &&
           IsOdd(i))) {
        CHECK_LT(permutation_index, permutation_.size());
        permutation_[permutation_index] =
            slice_times_available_for_permutation_[i];
        permutation_index = NextAvailablePermutationIndex(permutation_index);
      }
    }
    for (int i = 0; i < slice_times_available_for_permutation_.size(); ++i) {
      if (permutation_type_ ==
              PermutationType::kSmallerOffsetSmallerSliceTime ||
          (permutation_type_ ==
               PermutationType::kDistributeSmallSliceTimesAroundMiddleOffset &&
           IsEven(i))) {
        CHECK_LT(permutation_index, permutation_.size());
        permutation_[permutation_index] =
            slice_times_available_for_permutation_[i];
        permutation_index = NextAvailablePermutationIndex(permutation_index);
      }
    }
    CHECK_EQ(permutation_index, permutation_.size());
  }

  // Increments permutation_index. We skip over indices with fixed slice times.
  int NextAvailablePermutationIndex(int permutation_index) {
    do {
      ++permutation_index;
    } while (permutation_index < permutation_.size() &&
             fixed_permutation_values_[permutation_index]);
    return permutation_index;
  }

  int64_t num_slices_;
  // For each value in permutation, indicates if it has a fixed value tied to
  // a sliced allocation before repacking. If fixed_permutation_values[i] is
  // true, permutation_[i] holds the fixed slice time for the slice with the
  // ith smallest offset.
  std::vector<bool> fixed_permutation_values_;
  // Slice times that are available for permutation. A slice time is not
  // available for permutation if we have to fix it to an offset to generate
  // valid permutations, due to repacking.
  std::vector<int64_t> slice_times_available_for_permutation_;
  // The current type of permutation we are generating.
  PermutationType permutation_type_ = PermutationType::kUninitialized;
  // The permutation pertaining to permutation_type_.
  std::vector<int64_t> permutation_;
};

// A ComposedSliceTimePermutationIterator uses a base_iterator to generate
// permutations. However, it only returns valid permutations, for which we
// have not already emitted an equivalent permutation.
class ComposedSliceTimePermutationIterator
    : public SliceTimePermutationIterator {
 public:
  ComposedSliceTimePermutationIterator(
      SliceTimePermutationValidator validator,
      ObservedPermutationManager seen_manager,
      std::unique_ptr<SliceTimePermutationIterator> base_iterator)
      : validator_(std::move(validator)),
        seen_(std::move(seen_manager)),
        base_iterator_(std::move(base_iterator)) {}

  ~ComposedSliceTimePermutationIterator() override = default;

  void Begin() override { NextImpl(/*initialize=*/true); }

  bool Done() const override { return base_iterator_->Done(); }

  void Next() override { NextImpl(/*initialize=*/false); }

  absl::Span<const int64_t> Get() const override {
    return base_iterator_->Get();
  }

 private:
  void NextImpl(bool initialize) {
    if (initialize) {
      seen_.Clear();
      base_iterator_->Begin();
    }

    if (Done()) {
      return;
    }

    if (!initialize) {
      base_iterator_->Next();
    }

    // Keep advancing if we're not done, and the permutation is invalid or an
    // equivalent permutation has already been observed.
    while (!Done() && (!validator_.IsValid(Get()) || !seen_.Insert(Get()))) {
      base_iterator_->Next();
    }
  }

  SliceTimePermutationValidator validator_;
  ObservedPermutationManager seen_;
  std::unique_ptr<SliceTimePermutationIterator> base_iterator_;
};

}  // namespace

std::unique_ptr<SliceTimePermutationIterator>
SliceTimePermutationIterator::CreateForNewAllocation(
    Ty ty, absl::Span<const int64_t> inclusive_slice_start_times) {
  switch (ty) {
    case Ty::kAll:
      return std::make_unique<ComposedSliceTimePermutationIterator>(
          SliceTimePermutationValidator(/*original_slices=*/nullptr),
          ObservedPermutationManager(inclusive_slice_start_times),
          std::make_unique<SliceTimeAllPermutationIterator>(
              inclusive_slice_start_times.size()));
    case Ty::kPreferred:
      return std::make_unique<ComposedSliceTimePermutationIterator>(
          SliceTimePermutationValidator(/*original_slices=*/nullptr),
          ObservedPermutationManager(inclusive_slice_start_times),
          std::make_unique<SliceTimePreferredPermutationIterator>(
              inclusive_slice_start_times.size(),
              /*original_sliced_allocation=*/nullptr));
  }
}

std::unique_ptr<SliceTimePermutationIterator>
SliceTimePermutationIterator::CreateForRepack(
    Ty ty, const SlicedAllocationData* original_sliced_allocation) {
  // Repacking defaults to 1 slice in the absence of slicing data.
  int64_t num_slices = 1;
  if (original_sliced_allocation) {
    num_slices = original_sliced_allocation->num_slices();
  }

  std::vector<int64_t> inclusive_start_times;
  if (original_sliced_allocation) {
    inclusive_start_times =
        original_sliced_allocation->SortedInclusiveStartTimes();
  } else {
    // We don't actually know the first inclusive start time, but the actual
    // values don't matter, just their uniqueness within
    // inclusive_start_times. So, for a single slice, which is how we
    // treat any repacked allocation without slice data, any start time will
    // work.
    inclusive_start_times.push_back(0);
  }

  switch (ty) {
    case Ty::kAll:
      return std::make_unique<ComposedSliceTimePermutationIterator>(
          SliceTimePermutationValidator(original_sliced_allocation),
          ObservedPermutationManager(inclusive_start_times),
          std::make_unique<SliceTimeAllPermutationIterator>(num_slices));
    case Ty::kPreferred:
      return std::make_unique<ComposedSliceTimePermutationIterator>(
          SliceTimePermutationValidator(original_sliced_allocation),
          ObservedPermutationManager(inclusive_start_times),
          std::make_unique<SliceTimePreferredPermutationIterator>(
              num_slices, original_sliced_allocation));
  }
}

template <typename BufferType>
std::string GlobalDecreasingSizeBestFitHeap<
    BufferType>::SlicedAllocationFinder::FreeChunkPiece::ToString() const {
  return absl::StrCat("{ dimensions: ", dimensions.ToString(), ", free at: t",
                      earliest_free_slice_time, " }");
}

template <typename BufferType>
std::string GlobalDecreasingSizeBestFitHeap<
    BufferType>::SlicedAllocationFinder::FreeChunkRoot::ToString() const {
  return absl::StrCat(
      "{ chunk: ", chunk.ToString(), ", pieces: { ",
      absl::StrJoin(
          pieces.rbegin(), pieces.rend(), ", ",
          [](std::string* out, const auto& offset_sliced_free_chunk_pair) {
            absl::StrAppend(out,
                            offset_sliced_free_chunk_pair.second.ToString());
          }),
      " } }");
}

template <typename BufferType>
GlobalDecreasingSizeBestFitHeap<BufferType>::SlicedAllocationFinder::
    FreeChunkRoot::FreeChunkRoot(const Chunk& free_chunk,
                                 int64_t free_chunk_slice_time)
    : chunk(free_chunk),
      pieces({{free_chunk.offset, {free_chunk_slice_time, free_chunk}}}) {}

template <typename BufferType>
void GlobalDecreasingSizeBestFitHeap<BufferType>::SlicedAllocationFinder::
    FreeChunkRoot::Update(const Chunk& free_chunk,
                          int64_t free_chunk_slice_time) {
  VLOG(4) << "Updating root " << chunk.ToString() << " with "
          << free_chunk.ToString() << ", free at t" << free_chunk_slice_time;

  // Iterate through all pieces that might overlap with free_chunk.
  std::vector<FreeChunkPiece> new_pieces;
  for (auto it = pieces.lower_bound(free_chunk.chunk_end() - 1);
       it != pieces.end() &&
       it->second.dimensions.chunk_end() >= free_chunk.offset;) {
    const FreeChunkPiece& piece = it->second;
    if (!free_chunk.OverlapsWith(piece.dimensions) ||
        free_chunk_slice_time != piece.earliest_free_slice_time - 1) {
      ++it;
      continue;
    }

    // If free_chunk overlaps with piece and the slice time of piece is 1 before
    // that of free_chunk, replace piece with up to 3 new pieces:
    // - new_piece0: the part of piece that spatially comes before free_chunk
    // - new_piece1: the part of free_chunk that overlaps with piece
    // - new_piece2: the part of piece that spatially comes after free_chunk

    if (free_chunk.offset > piece.dimensions.offset) {
      FreeChunkPiece new_piece0(
          {piece.earliest_free_slice_time,
           Chunk::FromOffsetEnd(
               piece.dimensions.offset,
               std::min(free_chunk.offset, piece.dimensions.chunk_end()))});
      new_pieces.push_back(new_piece0);
    }

    FreeChunkPiece new_piece1(
        {free_chunk_slice_time,
         Chunk::FromOffsetEnd(
             std::max(free_chunk.offset, piece.dimensions.offset),
             std::min(free_chunk.chunk_end(), piece.dimensions.chunk_end()))});
    new_pieces.push_back(new_piece1);

    if (free_chunk.chunk_end() < piece.dimensions.chunk_end()) {
      FreeChunkPiece new_piece2(
          {piece.earliest_free_slice_time,
           Chunk::FromOffsetEnd(free_chunk.chunk_end(),
                                piece.dimensions.chunk_end())});
      new_pieces.push_back(new_piece2);
    }
    it = pieces.erase(it);
  }

  for (auto it = new_pieces.begin(); it != new_pieces.end(); ++it) {
    pieces.insert({it->dimensions.offset, *it});
  }

  VLOG(4) << "Root after update: " << ToString();
}

namespace {

// Code for rendering time_by_chunks as ascii art. Since this is for debugging,
// we only render ascii art of certain dimensions.
constexpr int64_t kMaxRenderOffset = 200;
constexpr int64_t kMaxRenderSliceTime = 9;
std::string RenderTimeByFreeChunks(
    const std::vector<std::vector<Chunk>>& time_by_chunks) {
  if (time_by_chunks.size() - 1 > kMaxRenderSliceTime) {
    return "too many time slices to render";
  }

  std::vector<std::string> time_by_memory_units;
  for (int i = 0; i < time_by_chunks.size(); ++i) {
    // Populate each row with Xs to start.
    time_by_memory_units.push_back(std::string(kMaxRenderOffset + 1, 'X'));

    for (const Chunk& chunk : time_by_chunks[i]) {
      if (chunk.chunk_end() > kMaxRenderOffset) {
        return "largest offset is too large to render";
      }
      for (int j = chunk.offset; j < chunk.chunk_end(); ++j) {
        // Overwrite X with a space if memory_unit j is free at slice time i.
        time_by_memory_units[i][j] = ' ';
      }
    }
  }

  // Create the final ascii art lines.
  std::vector<std::string> lines;
  lines.push_back("   ^");
  for (int i = time_by_memory_units.size() - 1; i >= 0; --i) {
    lines.push_back(absl::StrCat("t", i, " |", time_by_memory_units[i]));
  }
  std::string yaxis = "   +";
  for (int i = 0; i < kMaxRenderOffset + 1; ++i) {
    if (i % 10 == 0) {
      yaxis += "!";
      continue;
    }
    if (i % 5 == 0) {
      yaxis += "|";
      continue;
    }
    yaxis += "-";
  }
  lines.push_back(absl::StrCat(yaxis, ">"));
  lines.push_back("         space");

  return absl::StrJoin(lines, "\n");
}

}  // namespace

template <typename BufferType>
GlobalDecreasingSizeBestFitHeap<BufferType>::SlicedAllocationFinder::
    SlicedAllocationFinder(
        absl::Span<const FreeChunks> free_chunks_per_slice_time,
        std::vector<int64_t> sorted_slice_sizes, int64_t max_colocation_size,
        int64_t preferred_offset, int64_t alignment,
        std::unique_ptr<SliceTimePermutationIterator>
            slice_time_permutation_iterator,
        absl::AnyInvocable<bool(int64_t) const> is_offset_allowed)
    : sorted_slice_sizes_(std::move(sorted_slice_sizes)),
      slice_size_sum_(std::accumulate(sorted_slice_sizes_.begin(),
                                      sorted_slice_sizes_.end(),
                                      static_cast<int64_t>(0))),
      max_colocation_size_(max_colocation_size),
      preferred_offset_(preferred_offset),
      alignment_(alignment),
      slice_time_permutation_iterator_(
          std::move(slice_time_permutation_iterator)),
      is_offset_allowed_(std::move(is_offset_allowed)) {
  CHECK_EQ(sorted_slice_sizes_.size(), free_chunks_per_slice_time.size())
      << "We expect a data structure explaining the free chunks at each slice "
         "time.";
  CHECK(!free_chunks_per_slice_time.empty())
      << "Even an unsliced allocation is expected to have a list of free "
         "chunks at slice time t0.";

  if (VLOG_IS_ON(1)) {
    // Create a 2d vector where each row represents a slice time and each
    // column represents a free chunk at that slice time.
    std::vector<std::vector<Chunk>> time_by_chunks;
    for (int64_t i = 0; i < free_chunks_per_slice_time.size(); ++i) {
      std::vector<Chunk> chunks;
      for (const auto& free_chunk : free_chunks_per_slice_time[i]) {
        chunks.push_back(
            Chunk::FromOffsetEnd(free_chunk.first, free_chunk.second));
      }
      time_by_chunks.push_back(chunks);
    }

    LOG(INFO) << "Initial free space:\n"
              << RenderTimeByFreeChunks(time_by_chunks);
  }

  if (max_colocation_size_ < slice_size_sum_) {
    // If max_colocation_size was specified as -1 (or some other incorrect
    // value), set it to the sum of the real slices.
    max_colocation_size_ = slice_size_sum_;
  }

  // Build free_chunks_.
  //
  // Start by initializing FreeChunkRoots at LatestSliceTime().
  for (const std::pair<const int64_t, int64_t>& free_chunk_pair :
       free_chunks_per_slice_time.back()) {
    Chunk free_chunk =
        Chunk::FromOffsetEnd(free_chunk_pair.first, free_chunk_pair.second);
    if (free_chunk.size == 0) {
      continue;
    }
    CHECK_GT(free_chunk.size, 0);

    free_chunks_.insert(
        {free_chunk_pair.first, FreeChunkRoot(free_chunk, LatestSliceTime())});
  }
  // For slice times < LatestSliceTime(), slice the space of each root according
  // to when each subset of that root space is available.
  for (int64_t free_chunk_slice_time = LatestSliceTime() - 1;
       free_chunk_slice_time >= EarliestSliceTime(); --free_chunk_slice_time) {
    // Note, free_chunks_ and free_chunks_per_slice_time[] are sorted in
    // descending order of free chunk offsets. We simultaneously iterate through
    // the 2 data structures, increasing the iterator for whichever one points
    // to the greater chunk position.
    auto it = free_chunks_.begin();
    for (const std::pair<const int64_t, int64_t>& free_chunk_pair :
         free_chunks_per_slice_time[free_chunk_slice_time]) {
      Chunk free_chunk =
          Chunk::FromOffsetEnd(free_chunk_pair.first, free_chunk_pair.second);

      if (free_chunk.size == 0) {
        continue;
      }
      CHECK_GT(free_chunk.size, 0);

      // Increment it while all of free_chunk < all of it.
      for (; it != free_chunks_.end() &&
             free_chunk.chunk_end() - 1 < it->second.chunk.offset;
           ++it) {
      }

      if (it == free_chunks_.end()) {
        // free_chunk (and everything remaining in
        // free_chunks_per_slice_time[free_chunk_slice_time]) spatially come
        // before everything in free_chunks_.
        break;
      }

      // At this point, free_chunk and it overlap OR all of it < all of
      // free_chunk. For example, the following diagram illustrates the
      // relationship between the position of it and the possible positions for
      // free_chunk (fc below):
      //
      //           [---- it ----)
      //       [-fc-)..................................[-fc-)... ->

      // While free_chunk and it overlap, keep iterating it and updating
      // the root at it.

      // We restore it to 1 before its last value (in the loop) because at the
      // end of the loop it no longer overlaps with free_chunk, and it - 1
      // may overlap with the next free_chunk as well.
      auto previous_it = it;
      for (; it != free_chunks_.end() &&
             it->second.chunk.OverlapsWith(free_chunk);
           previous_it = it, ++it) {
        FreeChunkRoot& root = it->second;
        root.Update(free_chunk, free_chunk_slice_time);
      }
      it = previous_it;
    }
  }

  VLOG(2) << "Initial candidates:\n" << FreeChunksToAsciiArt();
  VLOG(2) << "SlicedAllocationFinder:\n" << ToString();
}

template <typename BufferType>
std::string GlobalDecreasingSizeBestFitHeap<
    BufferType>::SlicedAllocationFinder::FreeChunksToAsciiArt() const {
  auto it = free_chunks_.begin();
  if (it == free_chunks_.end()) {
    return "no candidate data";
  }
  int64_t final_offset = it->second.chunk.chunk_end();

  if (LatestSliceTime() > kMaxRenderSliceTime ||
      final_offset > kMaxRenderOffset) {
    return "candidates too large to render";
  }

  std::vector<std::vector<Chunk>> time_by_chunks;
  for (int64_t i = EarliestSliceTime(); i <= LatestSliceTime(); ++i) {
    time_by_chunks.push_back({});
  }

  for (const std::pair<const int64_t, FreeChunkRoot>& offset_root_pair :
       free_chunks_) {
    for (const std::pair<const int64_t, FreeChunkPiece>& offset_piece_pair :
         offset_root_pair.second.pieces) {
      for (int64_t slice_time =
               offset_piece_pair.second.earliest_free_slice_time;
           slice_time <= LatestSliceTime(); ++slice_time) {
        time_by_chunks[slice_time].push_back(
            offset_piece_pair.second.dimensions);
      }
    }
  }

  return RenderTimeByFreeChunks(time_by_chunks);
}

template <typename BufferType>
std::string GlobalDecreasingSizeBestFitHeap<
    BufferType>::SlicedAllocationFinder::ToString() const {
  std::vector<std::string> lines;

  lines.push_back(absl::StrCat("slices:              { ",
                               absl::StrJoin(sorted_slice_sizes_, ", "), " }"));
  lines.push_back(absl::StrCat("max_colocation_size: ", max_colocation_size_));
  lines.push_back(absl::StrCat("preferred_offset:    ", preferred_offset_));
  lines.push_back("free chunks:");
  int i = 0;
  for (auto it = free_chunks_.rbegin(); it != free_chunks_.rend(); ++it) {
    lines.push_back(absl::StrCat("  chunk ", i, ": ", it->second.ToString()));
    ++i;
  }

  return absl::StrJoin(lines, "\n");
}

template <typename BufferType>
typename GlobalDecreasingSizeBestFitHeap<
    BufferType>::SlicedAllocationFinder::ChunksSortedBySliceTime
GlobalDecreasingSizeBestFitHeap<BufferType>::SlicedAllocationFinder::Find()
    const {
  // Check if we can place the fully allocated buffer at the preferred offset
  if (preferred_offset_ >= 0) {
    ChunksSortedBySliceTime chunks = FindForOffset(preferred_offset_);
    if (!chunks.empty()) {
      VLOG(1) << "SlicedAllocationFinder found chunks: " << "{ "
              << absl::StrJoin(chunks, ", ", absl::StreamFormatter()) << " }";
      return chunks;
    }
  }

  // Find the smallest overall chunk that fits the allocation request
  std::vector<const FreeChunkRoot*> root_heap;
  for (auto it = free_chunks_.rbegin(); it != free_chunks_.rend(); ++it) {
    root_heap.push_back(&it->second);
  }
  auto heap_cmp = [](const FreeChunkRoot* lhs, const FreeChunkRoot* rhs) {
    if (lhs->chunk.size != rhs->chunk.size) {
      return lhs->chunk.size > rhs->chunk.size;
    }
    return lhs->chunk.offset > rhs->chunk.offset;
  };
  auto heap_next = [&]() -> const FreeChunkRoot* {
    if (root_heap.empty()) {
      return nullptr;
    }
    absl::c_pop_heap(root_heap, heap_cmp);
    const FreeChunkRoot* root = root_heap.back();
    root_heap.pop_back();
    return root;
  };
  absl::c_make_heap(root_heap, heap_cmp);
  // Each call to heap_next() gives us the next smallest root.
  for (const FreeChunkRoot* root = heap_next(); root != nullptr;
       root = heap_next()) {
    VLOG(3) << "SlicedAllocationFinder::Find() searching " << root->ToString();
    ChunksSortedBySliceTime chunks = FindInRoot(*root);
    if (!chunks.empty()) {
      VLOG(1) << "SlicedAllocationFinder found chunks: " << "{ "
              << absl::StrJoin(chunks, ", ", absl::StreamFormatter()) << " }";
      return chunks;
    }
  }

  LOG(ERROR) << "We did not find a place for our sliced allocation. This "
                "should not happen because MSA operates on an infinitely "
                "sized heap.";
  return {};
}

template <typename BufferType>
typename GlobalDecreasingSizeBestFitHeap<
    BufferType>::SlicedAllocationFinder::ChunksSortedBySliceTime
GlobalDecreasingSizeBestFitHeap<
    BufferType>::SlicedAllocationFinder::FindForOffset(int64_t offset) const {
  VLOG(3) << "SlicedAllocationFinder::FindForOffset() searching offset "
          << offset;
  auto it = free_chunks_.lower_bound(offset);
  if (it != free_chunks_.end()) {
    const FreeChunkRoot* root = &it->second;
    ChunksSortedBySliceTime chunks = FindInRoot(*root, offset);
    if (!chunks.empty()) {
      VLOG(3) << "SlicedAllocationFinder found chunks at " << offset << ": "
              << "{ " << absl::StrJoin(chunks, ", ", absl::StreamFormatter())
              << " }";
      return chunks;
    }
  }

  return {};
}

template <typename BufferType>
absl::Status GlobalDecreasingSizeBestFitHeap<BufferType>::
    SlicedAllocationFinder::DoesPermutationFit(
        absl::Span<const int64_t> permutation_of_slice_times,
        const FreeChunkRoot& root, int64_t offset) const {
  absl::Status result =
      DoesPermutationFitImpl(permutation_of_slice_times, root, offset);
  VLOG(3) << "SlicedAllocationFinder::DoesPermutationFit\n"
          << "  permutation of slice times: [ "
          << absl::StrJoin(permutation_of_slice_times, ",") << " ]\n"
          << "  offset: " << offset << "\n"
          << "  root: " << root.ToString() << "\n"
          << "  -> " << result;
  return result;
}

template <typename BufferType>
absl::Status GlobalDecreasingSizeBestFitHeap<BufferType>::
    SlicedAllocationFinder::DoesPermutationFitImpl(
        absl::Span<const int64_t> permutation_of_slice_times,
        const FreeChunkRoot& root, int64_t offset) const {
  if (permutation_of_slice_times.size() != sorted_slice_sizes_.size()) {
    return InvalidArgumentStrCat(
        sorted_slice_sizes_.size(), " slices times expected in permutation. ",
        permutation_of_slice_times.size(), " specified.");
  }
  if (offset >= root.chunk.chunk_end()) {
    return FailedPrecondition(
        "%s", absl::StrCat("Free chunk root ", root.chunk.ToString(),
                           " does not overlap with offset ", offset, "."));
  }
  if (offset + max_colocation_size_ > root.chunk.chunk_end()) {
    return FailedPrecondition(
        "%s", absl::StrCat("Not enough space to fit enitre allocation [",
                           offset, ", ", offset + max_colocation_size_,
                           ") in free chunk root ", root.chunk.ToString()));
  }
  if (!is_offset_allowed_(offset)) {
    return FailedPrecondition(
        "%s", absl::StrCat("We are not permitted to place an allocation at ",
                           "offset ", offset, "."));
  }

  auto piece_fwd_it = root.pieces.lower_bound(offset);
  if (piece_fwd_it == root.pieces.end()) {
    return FailedPrecondition(
        "%s", absl::StrCat("Offset ", offset, " comes before free chunk root ",
                           root.chunk.ToString()));
  }
  ++piece_fwd_it;
  auto piece_reverse_it = std::make_reverse_iterator(piece_fwd_it);
  auto at_pieces_end = [&](auto it) { return it == root.pieces.rend(); };
  size_t slice_index = 0;
  auto out_of_slices = [&](size_t index) { return index > LatestSliceTime(); };

  // Check to see if the slices will fit in pieces, starting at
  // piece_reverse_it.
  int64_t amount_of_current_slice_consumed = 0;
  int64_t current_offset = offset;
  while (!at_pieces_end(piece_reverse_it) && !out_of_slices(slice_index)) {
    int64_t current_slice_time = permutation_of_slice_times[slice_index];
    int64_t current_slice_size = sorted_slice_sizes_[slice_index];
    int64_t remaining_in_slice =
        current_slice_size - amount_of_current_slice_consumed;

    int64_t current_piece_time =
        piece_reverse_it->second.earliest_free_slice_time;
    int64_t remaining_in_piece =
        piece_reverse_it->second.dimensions.chunk_end() - current_offset;

    int64_t amount_to_consume =
        std::min(remaining_in_slice, remaining_in_piece);

    if (current_piece_time > current_slice_time) {
      // The current piece is not free far enough back in time to support the
      // current slice.
      return FailedPrecondition(
          "%s",
          absl::StrCat("At slice time t", current_slice_time, ", slice ",
                       slice_index, " does not fit at offset ", current_offset,
                       " in root ", root.chunk.ToString()));
    }

    if (remaining_in_slice >= remaining_in_piece) {
      ++piece_reverse_it;
      amount_of_current_slice_consumed += amount_to_consume;
    }
    if (remaining_in_slice <= remaining_in_piece) {
      ++slice_index;
      amount_of_current_slice_consumed = 0;
    }

    current_offset += amount_to_consume;
  }

  if (!out_of_slices(slice_index)) {
    return InternalStrCat("Ran out of space in root ", root.chunk.ToString(),
                          " to fit slice permutation; however, we should "
                          "have caught such a condition earlier.");
  }

  return absl::OkStatus();
}

// Future opportunities:
// 1) Potential optimization: We don't have to try every offset in
//    [root.chunk.offset, root.chunk.chunk_end()). If a permutation doesn't fit
//    at offset, it won't fit at offset + 1, unless the geometry of the free
//    space changes at offset + 1. If we carefully choose which offsets to try,
//    we don't have to try them all.
// 2) Potential tuning: We don't have a specific way to prioritize 1 permutation
//    or 1 offset over another. For example, it is likely better to place an
//    allocation at the beginning or the end of a root, to minimize
//    fragmentation. In the future, we may want to prioritize such
//    considerations.
template <typename BufferType>
typename GlobalDecreasingSizeBestFitHeap<
    BufferType>::SlicedAllocationFinder::ChunksSortedBySliceTime
GlobalDecreasingSizeBestFitHeap<BufferType>::SlicedAllocationFinder::FindInRoot(
    const FreeChunkRoot& root,
    std::optional<int64_t> only_try_this_offset) const {
  int64_t first_offset = root.chunk.offset;
  int64_t last_end = root.chunk.chunk_end();
  if (only_try_this_offset.has_value()) {
    first_offset = *only_try_this_offset;
    last_end = *only_try_this_offset + max_colocation_size_;
    if (*only_try_this_offset % alignment_ != 0) {
      return {};
    }
  } else if (first_offset % alignment_ != 0) {
    first_offset = first_offset + (alignment_ - (first_offset % alignment_));
  }
  CHECK_EQ(first_offset % alignment_, 0);
  for (int64_t offset = first_offset; offset + max_colocation_size_ <= last_end;
       offset += alignment_) {
    for (slice_time_permutation_iterator_->Begin();
         !slice_time_permutation_iterator_->Done();
         slice_time_permutation_iterator_->Next()) {
      if (DoesPermutationFit(slice_time_permutation_iterator_->Get(), root,
                             offset)
              .ok()) {
        return PermutationToChunks(slice_time_permutation_iterator_->Get(),
                                   offset);
      }
    }

    // Optimization: We can skip checking other offsets if the root
    // represents the same space at all slice times. In such a case, if we
    // don't fit at the first offset, we won't fit at any offset.
    if (root.pieces.size() == 1) {
      break;
    }
  }

  return {};
}

template <typename BufferType>
typename GlobalDecreasingSizeBestFitHeap<
    BufferType>::SlicedAllocationFinder::ChunksSortedBySliceTime
GlobalDecreasingSizeBestFitHeap<BufferType>::SlicedAllocationFinder::
    PermutationToChunks(absl::Span<const int64_t> permutation_of_slice_times,
                        int64_t offset) const {
  ChunksSortedBySliceTime chunks(permutation_of_slice_times.size() + 1,
                                 Chunk::FromOffsetSize(-1, 1));
  int64_t current_offset = offset;
  for (int64_t slice_index = 0; slice_index <= LatestSliceTime();
       ++slice_index) {
    int64_t size = sorted_slice_sizes_[slice_index];
    chunks[permutation_of_slice_times[slice_index]] =
        Chunk::FromOffsetSize(current_offset, size);
    current_offset += size;
  }
  chunks.back() = Chunk::FromOffsetSize(
      current_offset, max_colocation_size_ - (current_offset - offset));

  DCHECK(std::all_of(chunks.begin(), chunks.end(), [](const Chunk& chunk) {
    return chunk.offset >= 0 && chunk.size >= 0;
  }));

  return chunks;
}

template <typename BufferType>
absl::StatusOr<HeapSimulator::Result<BufferType>>
GlobalDecreasingSizeBestFitHeap<BufferType>::Finish() {
  std::vector<BufferInterval> sorted_buffer_intervals =
      GetSortedBufferIntervals();

  for (auto& buffer_interval : sorted_buffer_intervals) {
    if (!buffer_interval.need_allocation) {
      continue;
    }

    // This implementation of the heap algorithm does not have a notion of
    // maximum heap size, so it just commits.
    CommitChunk(buffer_interval, FindChunkCandidate(buffer_interval));
  }
  VLOG(1) << "result heap_size: " << result_.heap_size;
  Result result;
  result.heap_size = result_.heap_size;
  result.heap_results.emplace_back(result_);
  return result;
}

template <typename BufferType>
std::vector<
    typename GlobalDecreasingSizeBestFitHeap<BufferType>::BufferInterval>
GlobalDecreasingSizeBestFitHeap<BufferType>::GetSortedBufferIntervals() const {
  std::vector<BufferInterval> sorted_buffer_intervals;
  sorted_buffer_intervals.reserve(buffer_intervals_.size());
  for (auto& entry : buffer_intervals_) {
    sorted_buffer_intervals.push_back(entry.second);
  }
  absl::c_sort(sorted_buffer_intervals, buffer_interval_compare_);

  return sorted_buffer_intervals;
}

template <typename BufferType>
typename GlobalDecreasingSizeBestFitHeap<BufferType>::Chunk
GlobalDecreasingSizeBestFitHeap<BufferType>::FindChunkCandidate(
    const GlobalDecreasingSizeBestFitHeap::BufferInterval& buffer_interval,
    int64_t preferred_offset) const {
  const SlicedBufferInterval sliced_buffer_interval =
      SlicedBufferInterval::CreateConstInterval(buffer_interval);
  std::vector<Chunk> chunks =
      FindChunkCandidates(sliced_buffer_interval, preferred_offset);
  CHECK_EQ(chunks.size(), 1);
  return chunks[0];
}

template <typename BufferType>
typename GlobalDecreasingSizeBestFitHeap<BufferType>::FreeChunks
GlobalDecreasingSizeBestFitHeap<BufferType>::MakeFreeChunks(
    const BufferInterval& buffer_interval, int64_t max_colocation_size) const {
  // Map free chunk offsets -> ends.
  // We use `greater` for the comparison so that we can use `lower_bound` to
  // find the largest key less than or equal to the lookup value.
  FreeChunks free_chunks{
      {0, INT64_MAX}};  // Initialize with "infinite" free memory.

  // Subtract chunks that are in use from the free chunks.
  auto subtract_used_chunks = [&](const std::vector<Chunk>& used_chunks) {
    for (const Chunk& used_chunk : used_chunks) {
      // Find the free chunks containing the start and end of the used chunk.
      auto it_end = free_chunks.lower_bound(used_chunk.chunk_end());
      if (it_end == free_chunks.end()) continue;
      auto it_start = free_chunks.lower_bound(used_chunk.offset);

      // Store original free chunk end, in case `it_start == it_end`.
      int64_t free_chunk_end = it_end->second;

      // Subtract from free chunk containing start of used range, removing if it
      // becomes too small for the buffer.
      if (it_start != free_chunks.end()) {
        if (used_chunk.offset - it_start->first >= buffer_interval.size) {
          it_start->second = std::min(it_start->second, used_chunk.offset);
        } else {
          ++it_start;  // Increment iterator so that this entry is erased
                       // below.
        }
      }

      // Erase from the start chunk (possibly inclusive) to the end chunk
      // (always inclusive). We iterate from end to start, as the map is in
      // reverse order.
      free_chunks.erase(it_end, it_start);

      // Create a new free chunk after the used chunk, if it is large enough.
      int64_t chunk_end_aligned = RoundUpTo(used_chunk.chunk_end(), alignment_);
      if (free_chunk_end - chunk_end_aligned >= max_colocation_size) {
        CHECK(free_chunks.insert({chunk_end_aligned, free_chunk_end}).second);
      }
    }
  };

  subtract_used_chunks(interval_tree_.ChunksOverlappingInTime(
      buffer_interval.start, buffer_interval.end));

  for (const BufferType* colocation :
       GetTransitiveColocations(buffer_interval)) {
    const BufferInterval& interval = buffer_intervals_.at(colocation);
    VLOG(1) << "  Alias size " << interval.size << ", start " << interval.start
            << ", end " << interval.end << " " << interval.buffer->ToString();

    subtract_used_chunks(
        interval_tree_.ChunksOverlappingInTime(interval.start, interval.end));
  }

  return free_chunks;
}

template <typename BufferType>
std::vector<typename GlobalDecreasingSizeBestFitHeap<BufferType>::Chunk>
GlobalDecreasingSizeBestFitHeap<BufferType>::FindChunkCandidates(
    const SlicedBufferInterval& sliced_buffer_interval,
    int64_t preferred_offset) const {
  VLOG(1) << "Finding chunks for sliced buffer interval: "
          << sliced_buffer_interval.ToString();

  int64_t max_colocation_size =
      GetMaxColocationSize(sliced_buffer_interval.full_buffer_interval());
  auto chunks =
      CreateSlicedAllocationFinder(
          sliced_buffer_interval, max_colocation_size, preferred_offset,
          SliceTimePermutationIterator::CreateForNewAllocation(
              slice_time_permutation_iteration_type_,
              sliced_buffer_interval.inclusive_start_times()))
          .Find();
  return PostProcessFindChunkCandidatesResult(sliced_buffer_interval,
                                              std::move(chunks));
}

template <typename BufferType>
int64_t GlobalDecreasingSizeBestFitHeap<BufferType>::GetMaxColocationSize(
    const BufferInterval& buffer_interval) const {
  int64_t max_colocation_size = buffer_interval.size;
  for (const BufferType* colocation :
       GetTransitiveColocations(buffer_interval)) {
    max_colocation_size =
        std::max(max_colocation_size, buffer_intervals_.at(colocation).size);
  }

  return max_colocation_size;
}

template <typename BufferType>
typename GlobalDecreasingSizeBestFitHeap<BufferType>::SlicedAllocationFinder
GlobalDecreasingSizeBestFitHeap<BufferType>::CreateSlicedAllocationFinder(
    const SlicedBufferInterval& sliced_interval, int64_t max_colocation_size,
    int64_t preferred_offset,
    std::unique_ptr<SliceTimePermutationIterator>
        slice_time_permutation_iterator,
    absl::AnyInvocable<bool(int64_t) const> is_offset_allowed) const {
  // Build up a list of free chunks for each slice time.
  std::vector<FreeChunks> free_chunks_per_slice_time;
  free_chunks_per_slice_time.reserve(sliced_interval.num_slices());
  for (int slice_time = 0; slice_time < sliced_interval.num_slices() - 1;
       ++slice_time) {
    // We don't need to account for colocation until the last slice time, in
    // which we've allocated all the slices. So we set max_colocation_size to
    // -1.
    free_chunks_per_slice_time.push_back(
        MakeFreeChunks(sliced_interval.IntervalForMakeFreeChunks(slice_time),
                       /*max_colocation_size=*/-1));
  }
  // We account for colocation size in the last slice time, where we've
  // allocated all the slices.
  free_chunks_per_slice_time.push_back(MakeFreeChunks(
      sliced_interval.IntervalForMakeFreeChunks(sliced_interval.num_slices() -
                                                1),
      max_colocation_size));

  return SlicedAllocationFinder(
      free_chunks_per_slice_time, sliced_interval.SliceSizesSortedByOffset(),
      max_colocation_size, preferred_offset, alignment_,
      std::move(slice_time_permutation_iterator), std::move(is_offset_allowed));
}

template <typename BufferType>
std::vector<typename GlobalDecreasingSizeBestFitHeap<BufferType>::Chunk>
GlobalDecreasingSizeBestFitHeap<BufferType>::
    PostProcessFindChunkCandidatesResult(
        const SlicedBufferInterval& sliced_interval,
        std::vector<Chunk> chunks) const {
  if (chunks.empty()) {
    return {};
  }
  CHECK_EQ(chunks.size(), sliced_interval.num_slices() + 1);
  // The extra chunk is to ensure that colocations of larger sizes can fit.
  // However, we don't need that extra space for the buffer for which we found
  // chunks.
  chunks.pop_back();

  return chunks;
}

template <typename BufferType>
void GlobalDecreasingSizeBestFitHeap<BufferType>::CommitChunk(
    const GlobalDecreasingSizeBestFitHeap<BufferType>::BufferInterval&
        buffer_interval,
    GlobalDecreasingSizeBestFitHeap<BufferType>::Chunk chunk) {
  CHECK_EQ(chunk.size, buffer_interval.size);
  result_.heap_size = result_.UpdatedHeapSize(chunk);
  interval_tree_.Add(buffer_interval.start, buffer_interval.end, chunk);
  for (auto colocation : GetTransitiveColocations(buffer_interval)) {
    auto colocation_interval = buffer_intervals_[colocation];
    // Create a colocation chunk with the same offset but with the correct size
    // of the colocated interval in case the colocations are of different sizes.
    Chunk colocation_chunk =
        Chunk::FromOffsetSize(chunk.offset, colocation_interval.size);
    result_.heap_size = result_.UpdatedHeapSize(colocation_chunk);
    interval_tree_.Add(colocation_interval.start, colocation_interval.end,
                       colocation_chunk);
    AddToChunkMap(colocation, colocation_chunk);
  }

  AddToChunkMap(buffer_interval.buffer, chunk);
}

template <typename BufferType>
void GlobalDecreasingSizeBestFitHeap<BufferType>::AddToChunkMap(
    const BufferType* buffer, Chunk chunk) {
  const auto emplace_result = result_.chunk_map.emplace(buffer, chunk);
  DCHECK(emplace_result.second);
}

absl::StatusOr<HeapSimulator::Result<HloValue>>
ConstrainedGlobalDecreasingSizeBestFitHeap::Finish() {
  std::vector<BufferInterval> sorted_buffer_vec = GetSortedBufferIntervals();
  // Convert into std::list so that erase() is O(1).
  std::list<BufferInterval> sorted_buffer_intervals(sorted_buffer_vec.begin(),
                                                    sorted_buffer_vec.end());

  // Use do-while here, because we need to create 1 heap in `multi_heap_result`
  // even if `sorted_buffer_intervals` is empty.
  Result multi_heap_result;
  do {
    // Place buffers into the currently processed heap as many as possible.
    for (auto it = sorted_buffer_intervals.begin();
         it != sorted_buffer_intervals.end();) {
      BufferInterval buffer_interval = *it;
      if (!buffer_interval.need_allocation) {
        it = sorted_buffer_intervals.erase(it);
        continue;
      }
      if (buffer_interval.size > size_limit_per_heap_) {
        LOG(WARNING) << "Alloc buffer size " << buffer_interval.size
                     << " larger than the per-heap size limit "
                     << size_limit_per_heap_;
      }

      Chunk chunk_candidate = FindChunkCandidate(buffer_interval);
      if (chunk_candidate.chunk_end() <= size_limit_per_heap_ ||
          // Commit the chunk as long as the heap is empty. We do this because
          // we want the size constraint to be soft, meaning that results are
          // successfully generated even if there are some buffer sizes larger
          // than the given constraint size.
          result_.heap_size == 0) {
        CommitChunk(buffer_interval, chunk_candidate);
        it = sorted_buffer_intervals.erase(it);
        continue;
      }

      ++it;
    }
    // Collect the result from the currently processed heap and reset the heap
    // states.
    multi_heap_result.heap_size += result_.heap_size;
    multi_heap_result.heap_results.push_back(std::move(result_));
    result_ = {};
    interval_tree_ = {};
  } while (!sorted_buffer_intervals.empty());

  VLOG(1) << "Number of heaps produced = "
          << multi_heap_result.heap_results.size();
  return multi_heap_result;
}

template <typename BufferType>
absl::StatusOr<HeapSimulator::Result<BufferType>>
ChooseBestHeapAlgorithm<BufferType>::Finish() {
  DCHECK(!algorithms_.empty());
  std::vector<Result> results(algorithms_.size());
  int64_t min_size = INT64_MAX;
  int min_size_index = -1;
  for (int i = 0; i < algorithms_.size(); ++i) {
    TF_ASSIGN_OR_RETURN(results[i], algorithms_[i]->Finish());
    if (results[i].heap_size < min_size) {
      min_size = results[i].heap_size;
      min_size_index = i;
    }
  }

  DCHECK_GE(min_size_index, 0);
  return results[min_size_index];
}

template class GlobalDecreasingSizeBestFitHeap<HloValue>;
template class GlobalDecreasingSizeBestFitHeap<AllocationBlock>;
template class ChooseBestHeapAlgorithm<HloValue>;

}  // namespace xla
