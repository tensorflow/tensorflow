/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/heap_simulator.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/xla/comparison_util.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_schedule.h"
#include "tensorflow/compiler/xla/hlo/utils/hlo_live_range.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/memory_space_assignment_repacking.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {

using absl::flat_hash_map;
using absl::flat_hash_set;

bool HeapSimulator::Chunk::OverlapsWith(Chunk other_chunk) const {
  CHECK_NE(size, 0);
  CHECK_NE(other_chunk.size, 0);
  return offset < other_chunk.chunk_end() && other_chunk.offset < chunk_end();
}

/*static*/
StatusOr<int64_t> HeapSimulator::MinimumMemoryForModule(
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
StatusOr<int64_t> HeapSimulator::MinimumMemoryForComputation(
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

StatusOr<int64_t> HeapSimulator::MinimumMemoryForComputation(
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
StatusOr<HeapSimulator::Result<HloValue>> HeapSimulator::Run(
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
StatusOr<HeapSimulator::Result<HloValue>> HeapSimulator::Run(
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
StatusOr<HeapSimulator::Result<HloValue>> HeapSimulator::Run(
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
Status HeapSimulator::RunComputation(
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
  return OkStatus();
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
  const int64_t size = size_fn_(*buffer);
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
  const int64_t size = size_fn_(*buffer);
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
  algorithm_->ShareWith(buffer, shared, size_fn_(*shared));
  no_fragmentation_stats_->ShareWith(buffer, shared, size_fn_(*shared));
  FillDebugTrace(HeapSimulatorTrace::Event::SHARE_WITH, buffer, instruction,
                 shared);
}

HeapSimulator::Result<HloValue> HeapSimulator::Finish() {
  Result<HloValue> result = algorithm_->Finish();

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
  const Result<HloValue> no_frag_result = no_fragmentation_stats_->Finish();
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
HeapSimulator::Result<BufferType>
NoFragmentationStatsHeap<BufferType>::Finish() {
  // The result.chunk_map is empty, since we only collect stats, and don't
  // actually compute chunk assignments.
  Result result;
  result.heap_size = max_heap_size_;
  return result;
}

template <typename BufferType>
GlobalDecreasingSizeBestFitHeap<BufferType>::GlobalDecreasingSizeBestFitHeap(
    int64_t alignment, Type type)
    : alignment_(alignment) {
  if (type == kTemporal) {
    buffer_interval_compare_ = GetTemporalBufferIntervalCompare();
  } else {
    CHECK(type == kSpatial);
    buffer_interval_compare_ = GetSpatialBufferIntervalCompare();
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
    result_.chunk_map.emplace(buffer, Chunk{0, 0});
    return;
  }

  auto emplace_result = buffer_intervals_.emplace(
      buffer, BufferInterval{buffer, size, current_time_, -1, {}, true});
  DCHECK(emplace_result.second);
  ++current_time_;
}

template <typename BufferType>
void GlobalDecreasingSizeBestFitHeap<BufferType>::ShareWith(
    const BufferType* buffer, const BufferType* share_with, int64_t size) {
  // Degenerate case: 0-sized buffers are always allocated at offset 0.
  if (size == 0) {
    result_.chunk_map.emplace(buffer, Chunk{0, 0});
    return;
  }
  DCHECK_NE(buffer_intervals_.count(share_with), 0);
  buffer_intervals_[share_with].colocations.push_back(buffer);
  auto emplace_result = buffer_intervals_.emplace(
      buffer, BufferInterval{buffer, size, current_time_, -1, {}, false});
  DCHECK(emplace_result.second);
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
  DCHECK_EQ(buffer_interval.buffer, buffer);
  DCHECK_EQ(buffer_interval.size, size);
  DCHECK_EQ(buffer_interval.end, -1);
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
      // Deleting root is simply reseting root;
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
HeapSimulator::Result<BufferType>
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
  VLOG(1) << "Finding chunks for buffer: "
          << buffer_interval.buffer->ToString();
  VLOG(1) << "Size " << buffer_interval.size << ", start "
          << buffer_interval.start << ", end " << buffer_interval.end;
  // Get all colocated buffers and gather all interferenced chunks.
  //
  // Imagine that we've already allocated three chunks : a, b and c.  And now
  // we want to allocate d. Since e is colocated with d, we have to allocate
  // chunks for them together at the same address. To do this, we first gather
  // all chunks that overlap with d and e on the time dimension, in this case
  // the overlapped chunks are a and b (c doesn't overlap with either of d and
  // e), then find create a new chunk that doesn't overlap with a and b on the
  // space dimension.
  //
  // space
  //   ^
  //   |+--d---+      +---e---+
  //   |
  //   |+---+  +---------------+  +-------+
  //   ||   |  |               |  |       |
  //   ||   |  |               |  |       |
  //   |+-a-+  +-------b-------+  +---c---+
  //   ----------------------------------------> time

  // TODO(b/275905276): when slicing, build free_chunks for each consecutive
  // slice time, where slice time is logical time.

  // Map free chunk offsets -> ends.
  // We use `greater` for the comparison so that we can use `lower_bound` to
  // find the largest key less than or equal to the lookup value.
  absl::btree_map<int64_t, int64_t, std::greater<int64_t>> free_chunks{
      {0, INT64_MAX}};  // Initialize with "infinite" free memory.

  // Find the max size of interval across its colocations and use this value to
  // determine whether the buffer will fit in the heap.
  int64_t max_colocation_size = buffer_interval.size;
  for (const BufferType* colocation :
       GetTransitiveColocations(buffer_interval)) {
    max_colocation_size =
        std::max(max_colocation_size, buffer_intervals_.at(colocation).size);
  }

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
          ++it_start;  // Increment iterator so that this entry is erased below.
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

  // TODO(b/275905276): when slicing, merge the free_chunks for each slice time.
  // The end result should be a list of free chunks in which buffer_interval
  // not only fits in each free chunk, but the slices of buffer interval can
  // be allocated according to their requirements.
  // Try to find a large enough free chunk containing the preferred offset.
  Chunk chunk{preferred_offset, max_colocation_size};
  auto it = (preferred_offset < 0) ? free_chunks.end()
                                   : free_chunks.lower_bound(preferred_offset);
  if (it == free_chunks.end() || (it->second < chunk.chunk_end())) {
    // Otherwise, find the smallest free chunk. In the case of a tie, prefer the
    // smallest offset. We ensure above that all of the free chunks are large
    // enough to store the buffer.
    chunk.offset = absl::c_min_element(free_chunks, [](auto a, auto b) {
                     return std::forward_as_tuple(a.second - a.first, a.first) <
                            std::forward_as_tuple(b.second - b.first, b.first);
                   })->first;
  }
  return chunk;
}

template <typename BufferType>
void GlobalDecreasingSizeBestFitHeap<BufferType>::CommitChunk(
    const GlobalDecreasingSizeBestFitHeap<BufferType>::BufferInterval&
        buffer_interval,
    GlobalDecreasingSizeBestFitHeap<BufferType>::Chunk chunk) {
  // Update the maximum heap size according to the one determined by the chunk
  // candidate. In case of colocations of different sizes, the chunk size
  // returned is the maximum of all colocations, so use this value to update the
  // heap size.
  result_.heap_size = result_.UpdatedHeapSize(chunk);
  // Now, update the chunk size to the actual size of the buffer interval.
  chunk.size = buffer_interval.size;
  interval_tree_.Add(buffer_interval.start, buffer_interval.end, chunk);
  for (auto colocation : GetTransitiveColocations(buffer_interval)) {
    auto colocation_interval = buffer_intervals_[colocation];
    // Create a colocation chunk with the same offset but with the correct size
    // of the colocated interval in case the colocations are of different sizes.
    Chunk colocation_chunk{chunk.offset, colocation_interval.size};
    AddToChunkMap(colocation, colocation_chunk);
    interval_tree_.Add(colocation_interval.start, colocation_interval.end,
                       colocation_chunk);
  }

  AddToChunkMap(buffer_interval.buffer, chunk);
}

template <typename BufferType>
void GlobalDecreasingSizeBestFitHeap<BufferType>::AddToChunkMap(
    const BufferType* buffer, Chunk chunk) {
  const auto emplace_result = result_.chunk_map.emplace(buffer, chunk);
  DCHECK(emplace_result.second);
}

HeapSimulator::Result<HloValue>
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
HeapSimulator::Result<BufferType>
ChooseBestHeapAlgorithm<BufferType>::Finish() {
  DCHECK(!algorithms_.empty());
  std::vector<Result> results(algorithms_.size());
  int64_t min_size = INT64_MAX;
  int min_size_index = -1;
  for (int i = 0; i < algorithms_.size(); ++i) {
    results[i] = algorithms_[i]->Finish();
    if (results[i].heap_size < min_size) {
      min_size = results[i].heap_size;
      min_size_index = i;
    }
  }

  DCHECK_GE(min_size_index, 0);
  return results[min_size_index];
}

template class GlobalDecreasingSizeBestFitHeap<HloValue>;
template class GlobalDecreasingSizeBestFitHeap<
    MemorySpaceAssignmentRepacker::AllocationBlock>;
template class ChooseBestHeapAlgorithm<HloValue>;

}  // namespace xla
