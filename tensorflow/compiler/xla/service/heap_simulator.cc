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
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/hlo_live_range.h"
#include "tensorflow/compiler/xla/service/hlo_schedule.h"
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
StatusOr<int64> HeapSimulator::MinimumMemoryForModule(
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
      HeapSimulator::Result result,
      HeapSimulator::Run(absl::make_unique<NoFragmentationStatsHeap>(), *module,
                         schedule, *alias_analysis, size_function));
  return result.heap_size;
}

/*static*/
StatusOr<int64> HeapSimulator::MinimumMemoryForComputation(
    const HloComputation& computation, const HloInstructionSequence& sequence,
    const HloAliasAnalysis& alias_analysis,
    const LogicalBuffer::SizeFunction& size_function,
    const absl::flat_hash_map<const HloComputation*, int64>*
        memory_by_computation) {
  TF_ASSIGN_OR_RETURN(
      HeapSimulator::Result result,
      HeapSimulator::Run(absl::make_unique<NoFragmentationStatsHeap>(),
                         computation, sequence, alias_analysis, size_function,
                         HeapSimulator::Options(), memory_by_computation));
  return result.heap_size;
}

StatusOr<int64> HeapSimulator::MinimumMemoryForComputation(
    const HloComputation& computation, const HloInstructionSequence& sequence,
    const HloAliasAnalysis& alias_analysis,
    const LogicalBuffer::SizeFunction& size_function,
    const HloSchedule* schedule) {
  TF_ASSIGN_OR_RETURN(
      HeapSimulator::Result result,
      HeapSimulator::Run(absl::make_unique<NoFragmentationStatsHeap>(),
                         computation, sequence, alias_analysis, size_function,
                         schedule, HeapSimulator::Options()));
  return result.heap_size;
}

/*static*/
StatusOr<HeapSimulator::Result> HeapSimulator::Run(
    std::unique_ptr<HeapAlgorithm> algorithm, const HloModule& module,
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
StatusOr<HeapSimulator::Result> HeapSimulator::Run(
    std::unique_ptr<HeapAlgorithm> algorithm, const HloComputation& computation,
    const HloInstructionSequence& instruction_sequence,
    const HloAliasAnalysis& alias_analysis,
    const BufferValue::SizeFunction& size_fn, const Options& options,
    const absl::flat_hash_map<const HloComputation*, int64>*
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
StatusOr<HeapSimulator::Result> HeapSimulator::Run(
    std::unique_ptr<HeapAlgorithm> algorithm, const HloComputation& computation,
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

  for (const HloValue* value : dataflow_analysis.values()) {
    // Ignore buffers that are not tracked.
    if (hlo_live_range->instruction_schedule().count(
            value->defining_instruction()) == 0) {
      continue;
    }
    if (IgnoreBuffer(value)) {
      continue;
    }
    values_to_assign.push_back(value);
  }

  auto& buffer_live_ranges = hlo_live_range->buffer_live_ranges();

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
  for (int64 i = 0; i < hlo_live_range->schedule_end_time() + 1; ++i) {
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
  return Status::OK();
}

HeapSimulator::HeapSimulator(
    std::unique_ptr<HeapAlgorithm> algorithm,
    const BufferValue::SizeFunction& size_fn, const Options& options,
    const HloSchedule* schedule,
    const absl::flat_hash_map<const HloComputation*, int64>*
        memory_by_computation)
    : no_fragmentation_stats_(absl::make_unique<NoFragmentationStatsHeap>()),
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
  const int64 size = size_fn_(*buffer);
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
  const int64 size = size_fn_(*buffer);
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

HeapSimulator::Result HeapSimulator::Finish() {
  Result result = algorithm_->Finish();

  // Post-process the result to add chunks for shared buffers.  An empty chunk
  // map means that either no buffers were allocated, or the heap was only
  // collecting statistics, e.g. NoFragmentationStatsHeap.
  if (!result.chunk_map.empty()) {
    // If we were told to assign specific buffers, make sure we've assigned
    // exactly that many buffers.
    if (options_.buffers_to_assign != nullptr) {
      CHECK_EQ(options_.buffers_to_assign->size(), result.chunk_map.size());
    }
  }

  // Fragmentation is the difference between the actual and ideal sizes.
  const Result no_frag_result = no_fragmentation_stats_->Finish();
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
  event->set_computation_name(instruction->parent()->name());
  event->set_instruction_name(instruction->name());
  if (kind == HeapSimulatorTrace::Event::SHARE_WITH) {
    CHECK(share_with_canonical != nullptr);
    event->set_share_with_canonical_id(share_with_canonical->id());
  } else {
    CHECK(share_with_canonical == nullptr);
  }
}

void NoFragmentationStatsHeap::Alloc(const HloValue* buffer, int64 size) {
  current_heap_size_ += size;
  if (current_heap_size_ > max_heap_size_) {
    max_heap_size_ = current_heap_size_;
  }
}

void NoFragmentationStatsHeap::AccountForSubcomputationMemory(
    const HloInstruction* instruction, int64 alloc_size_by_instruction,
    const absl::flat_hash_map<const HloComputation*, int64>&
        memory_by_computation) {
  // We only count the memory usage of the largest subcomputation, instead of
  // adding them all, because subcomputations won't execute in parallel.
  int64 max_subcomputation_bytes = 0;
  for (const auto* c : instruction->called_computations()) {
    auto it = memory_by_computation.find(c);
    if (it != memory_by_computation.end()) {
      int64 subcomputation_bytes = it->second;
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

void NoFragmentationStatsHeap::Free(const HloValue* buffer, int64 size) {
  current_heap_size_ -= size;
}

HeapSimulator::Result NoFragmentationStatsHeap::Finish() {
  // The result.chunk_map is empty, since we only collect stats, and don't
  // actually compute chunk assignments.
  Result result;
  result.heap_size = max_heap_size_;
  return result;
}

GlobalDecreasingSizeBestFitHeap::GlobalDecreasingSizeBestFitHeap(
    int64 alignment, Type type)
    : alignment_(alignment) {
  if (type == kTemporal) {
    buffer_interval_compare_ = GetTemporalBufferIntervalCompare();
  } else {
    CHECK(type == kSpatial);
    buffer_interval_compare_ = GetSpatialBufferIntervalCompare();
  }
}

GlobalDecreasingSizeBestFitHeap::BufferIntervalCompare
GlobalDecreasingSizeBestFitHeap::GetTemporalBufferIntervalCompare() const {
  return [&](const BufferInterval& x, const BufferInterval& y) {
    int64 x_end = x.end;
    for (auto colocation : GetTransitiveColocations(x)) {
      x_end = std::max(x_end, buffer_intervals_.at(colocation).end);
    }

    int64 y_end = y.end;
    for (auto colocation : GetTransitiveColocations(y)) {
      y_end = std::max(y_end, buffer_intervals_.at(colocation).end);
    }

    if (x_end - x.start != y_end - y.start) {
      return x_end - x.start > y_end - y.start;
    }

    if (x.size != y.size) {
      return x.size > y.size;
    }
    return x.buffer->id() < y.buffer->id();
  };
}

/*static*/ GlobalDecreasingSizeBestFitHeap::BufferIntervalCompare
GlobalDecreasingSizeBestFitHeap::GetSpatialBufferIntervalCompare() {
  return [&](const BufferInterval& x, const BufferInterval& y) {
    if (x.size != y.size) {
      return x.size > y.size;
    }
    if (x.end - x.start != y.end - y.start) {
      return x.end - x.start > y.end - y.start;
    }
    return x.buffer->id() < y.buffer->id();
  };
}

void GlobalDecreasingSizeBestFitHeap::Alloc(const HloValue* buffer,
                                            int64 size) {
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

void GlobalDecreasingSizeBestFitHeap::ShareWith(const HloValue* buffer,
                                                const HloValue* share_with,
                                                int64 size) {
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

absl::flat_hash_set<const HloValue*>
GlobalDecreasingSizeBestFitHeap::GetTransitiveColocations(
    const BufferInterval& interval) const {
  absl::flat_hash_set<const HloValue*> result;
  std::vector<const BufferInterval*> worklist = {&interval};
  while (!worklist.empty()) {
    const BufferInterval* item = worklist.back();
    worklist.pop_back();
    for (const HloValue* buffer_colocated : item->colocations) {
      result.insert(buffer_colocated);
      worklist.push_back(&buffer_intervals_.at(buffer_colocated));
    }
  }

  return result;
}

void GlobalDecreasingSizeBestFitHeap::Free(const HloValue* buffer, int64 size) {
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

void BufferIntervalTree::Add(int64 start, int64 end, const Chunk& chunk) {
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

bool BufferIntervalTree::Remove(int64 start, int64 end, const Chunk& chunk) {
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
    int64 start, int64 end) const {
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

HeapSimulator::Result GlobalDecreasingSizeBestFitHeap::Finish() {
  std::vector<BufferInterval> sorted_buffer_intervals =
      GetSortedBufferIntervals();

  for (auto& buffer_interval : sorted_buffer_intervals) {
    if (!buffer_interval.need_allocation) {
      continue;
    }

    ChunkCandidate chunk_candidate = FindChunkCandidate(buffer_interval);
    // This implementation of the heap algorithm does not have a notion of
    // maximum heap size, so it just commits.
    CommitChunk(buffer_interval, chunk_candidate);
  }
  VLOG(1) << "result heap_size: " << result_.heap_size;
  return result_;
}

std::vector<GlobalDecreasingSizeBestFitHeap::BufferInterval>
GlobalDecreasingSizeBestFitHeap::GetSortedBufferIntervals() const {
  std::vector<BufferInterval> sorted_buffer_intervals;
  for (auto& entry : buffer_intervals_) {
    sorted_buffer_intervals.push_back(entry.second);
  }
  absl::c_sort(sorted_buffer_intervals, buffer_interval_compare_);

  return sorted_buffer_intervals;
}

GlobalDecreasingSizeBestFitHeap::ChunkCandidate
GlobalDecreasingSizeBestFitHeap::FindChunkCandidate(
    const GlobalDecreasingSizeBestFitHeap::BufferInterval& buffer_interval,
    int64 preferred_offset) const {
  VLOG(1) << "Finding chunks for buffer: "
          << buffer_interval.buffer->ToString();
  VLOG(1) << "Size " << buffer_interval.size << ", start "
          << buffer_interval.start << ", end " << buffer_interval.end;
  auto chunks_overlapping_in_time = interval_tree_.ChunksOverlappingInTime(
      buffer_interval.start, buffer_interval.end);
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
  for (auto colocation : GetTransitiveColocations(buffer_interval)) {
    auto colocation_interval = buffer_intervals_.at(colocation);
    auto colocation_overlapping = interval_tree_.ChunksOverlappingInTime(
        colocation_interval.start, colocation_interval.end);
    VLOG(1) << "  Alias size " << colocation_interval.size << ", start "
            << colocation_interval.start << ", end " << colocation_interval.end
            << " " << colocation_interval.buffer->ToString();
    chunks_overlapping_in_time.insert(chunks_overlapping_in_time.end(),
                                      colocation_overlapping.begin(),
                                      colocation_overlapping.end());
  }
  absl::c_sort(chunks_overlapping_in_time, [](const Chunk& x, const Chunk& y) {
    return x.offset < y.offset;
  });

  // Find the minimum free chunk that can hold this buffer.
  ChunkCandidate chunk_candidate{Chunk{-1, INT64_MAX}, result_.heap_size};
  Chunk& min_fit_chunk = chunk_candidate.chunk;
  int64 preferred_chunk_end = preferred_offset + buffer_interval.size;
  auto use_free_chunk_if_smaller = [&](int64 free_offset, int64 free_size) {
    if (free_size < buffer_interval.size) {
      return;
    }

    // If a preferred offset is provided, pick that offset.
    if (free_offset <= preferred_offset &&
        free_offset + free_size >= preferred_chunk_end) {
      min_fit_chunk = {preferred_offset, buffer_interval.size};
    } else if (free_offset + free_size == result_.heap_size &&
               free_offset <= preferred_offset) {
      // If the free offset is at the very end and if the preferred offset lies
      // in this, pick the preferred offset and grow the heap.
      min_fit_chunk = {preferred_offset, buffer_interval.size};
      chunk_candidate.heap_size = preferred_chunk_end;
    }

    // Pick the min-fit chunk only if we didn't have a preferred offset or a
    // chunk at the preferred offset hasn't been found.
    if ((preferred_offset < 0 || min_fit_chunk.offset != preferred_offset) &&
        free_size < min_fit_chunk.size) {
      min_fit_chunk = {free_offset, free_size};
    }
  };

  int64 offset = 0;
  for (auto& chunk : chunks_overlapping_in_time) {
    if (offset < chunk.offset) {
      use_free_chunk_if_smaller(offset, chunk.offset - offset);
    }
    offset = std::max(offset, RoundUpToNearest(chunk.chunk_end(), alignment_));
  }
  use_free_chunk_if_smaller(offset, result_.heap_size - offset);
  // When preferred offset is provided and the preferred offset is larger than
  // the current heap size, simply use the preferred offset provided.
  if (result_.heap_size <= preferred_offset) {
    chunk_candidate.heap_size = preferred_chunk_end;
    min_fit_chunk = {preferred_offset, buffer_interval.size};
  }

  if (min_fit_chunk.offset == -1) {
    // Increase the heap size to fit in the last free chunk.
    chunk_candidate.heap_size = offset + buffer_interval.size;
    min_fit_chunk = {offset, buffer_interval.size};
  }

  min_fit_chunk.size = buffer_interval.size;
  return chunk_candidate;
}

void GlobalDecreasingSizeBestFitHeap::CommitChunk(
    const GlobalDecreasingSizeBestFitHeap::BufferInterval& buffer_interval,
    GlobalDecreasingSizeBestFitHeap::ChunkCandidate chunk_candidate) {
  // Update the maximum heap size according to the one determined by the chunk
  // candidate.
  result_.heap_size = chunk_candidate.heap_size;
  interval_tree_.Add(buffer_interval.start, buffer_interval.end,
                     chunk_candidate.chunk);
  for (auto colocation : GetTransitiveColocations(buffer_interval)) {
    AddToChunkMap(colocation, chunk_candidate.chunk);
    auto colocation_interval = buffer_intervals_[colocation];
    interval_tree_.Add(colocation_interval.start, colocation_interval.end,
                       chunk_candidate.chunk);
  }

  AddToChunkMap(buffer_interval.buffer, chunk_candidate.chunk);
}

void GlobalDecreasingSizeBestFitHeap::AddToChunkMap(const HloValue* buffer,
                                                    Chunk chunk) {
  const auto emplace_result = result_.chunk_map.emplace(buffer, chunk);
  DCHECK(emplace_result.second);
}

HeapSimulator::Result ChooseBestHeapAlgorithm::Finish() {
  DCHECK(!algorithms_.empty());
  std::vector<Result> results(algorithms_.size());
  int64 min_size = INT64_MAX;
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

}  // namespace xla
