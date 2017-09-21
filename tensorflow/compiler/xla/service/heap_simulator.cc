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

#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/liveness_util.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {

using tensorflow::gtl::FlatMap;
using tensorflow::gtl::FlatSet;

namespace {

// Returns the set of buffers that may be sources of all operands of the given
// instruction.  The returned buffers are guaranteed to have no duplicates, and
// to be sorted in a deterministic order.
std::vector<const LogicalBuffer*> UniqueOperandSourceBuffers(
    const HloInstruction* instruction,
    const TuplePointsToAnalysis& points_to_analysis) {
  std::vector<const LogicalBuffer*> buffers;
  for (const HloInstruction* operand : instruction->operands()) {
    points_to_analysis.GetPointsToSet(operand).ForEachElement(
        [&](const ShapeIndex& /*index*/,
            const PointsToSet::BufferList& points_to) {
          buffers.insert(buffers.end(), points_to.begin(), points_to.end());
        });
  }

  // Sort and then remove duplicates from buffers.
  std::sort(buffers.begin(), buffers.end(),
            [](const LogicalBuffer* a, const LogicalBuffer* b) {
              return a->id() < b->id();
            });
  buffers.erase(std::unique(buffers.begin(), buffers.end(),
                            [](const LogicalBuffer* a, const LogicalBuffer* b) {
                              return a->id() == b->id();
                            }),
                buffers.end());
  return buffers;
}

}  // namespace

/*static*/
StatusOr<HeapSimulator::Result> HeapSimulator::Run(
    std::unique_ptr<HeapAlgorithm> algorithm, const HloModule& module,
    const SequentialHloOrdering::HloModuleSequence& module_sequence,
    const TuplePointsToAnalysis& points_to_analysis,
    const LogicalBuffer::SizeFunction& size_fn,
    const FlatSet<const LogicalBuffer*>* buffers_to_assign) {
  HeapSimulator heap(std::move(algorithm), size_fn, buffers_to_assign,
                     &module_sequence);
  const HloComputation* entry_computation = module.entry_computation();
  const std::vector<const HloInstruction*>& instruction_sequence =
      FindOrDie(module_sequence, entry_computation);
  TF_RETURN_IF_ERROR(heap.RunComputation(
      *entry_computation, instruction_sequence, points_to_analysis));
  return heap.Finish();
}

/*static*/
StatusOr<HeapSimulator::Result> HeapSimulator::Run(
    std::unique_ptr<HeapAlgorithm> algorithm, const HloComputation& computation,
    const std::vector<const HloInstruction*>& instruction_sequence,
    const TuplePointsToAnalysis& points_to_analysis,
    const LogicalBuffer::SizeFunction& size_fn,
    const FlatSet<const LogicalBuffer*>* buffers_to_assign) {
  HeapSimulator heap(std::move(algorithm), size_fn, buffers_to_assign,
                     /*module_sequence=*/nullptr);
  TF_RETURN_IF_ERROR(heap.RunComputation(computation, instruction_sequence,
                                         points_to_analysis));
  return heap.Finish();
}

// Runs a heap simulation for the given 'computation', assuming the given
// 'instruction_sequence'.
Status HeapSimulator::RunComputation(
    const HloComputation& computation,
    const std::vector<const HloInstruction*>& instruction_sequence,
    const TuplePointsToAnalysis& points_to_analysis) {
  // The goal here is to minimize memory usage, assuming the given sequential
  // ordering of instructions.  The strategy is to walk through the instruction
  // sequence, calling Alloc and Free on the underlying heap algorithm.  The
  // heap algorithm takes care of packing and reducing fragmentation.
  //
  // 'live_buffers' tracks the liveness of each buffer that we assign, by
  // associating it with a set of HloInstructions that need to be visited.  When
  // the set becomes empty, the buffer is no longer used, and can be freed.
  FlatMap<const LogicalBuffer*, FlatSet<const HloInstruction*>> live_buffers;

  const HloInstruction* root = computation.root_instruction();
  auto output_source_buffers =
      points_to_analysis.GetPointsToSet(root).CreateFlattenedSet();

  std::vector<const LogicalBuffer*> dead_buffers_to_free;
  std::vector<const LogicalBuffer*> operand_buffers_to_free;
  for (const HloInstruction* instruction : instruction_sequence) {
    const TuplePointsToAnalysis::BufferDefinitionVector&
        buffers_defined_by_instruction =
            points_to_analysis.GetBuffersDefinedByInstruction(instruction);

    // Initialize live_buffers for each buffer that we're going to assign.  The
    // set of instructions that need to be visited contains all users of all
    // aliases.  The alias itself is not necessary; if it has users, the users
    // are necessarily scheduled after the alias.  And if it has no users, it is
    // either a dead value or an output, both of which are handled below.
    //
    // We ignore control dependencies here. The reasoning is that the control
    // dependencies have already been accounted for in the ordering of the given
    // 'instruction_sequence', and should not otherwise artificially extend the
    // lifetime of buffers that aren't already connected by a data dependency.
    dead_buffers_to_free.clear();
    for (const LogicalBuffer* buffer : buffers_defined_by_instruction) {
      if (IgnoreBuffer(buffer)) {
        continue;
      }
      FlatSet<const HloInstruction*>* live_set = nullptr;
      for (const BufferAlias& alias :
           points_to_analysis.GetBufferAliases(*buffer)) {
        const std::vector<HloInstruction*>& users =
            alias.instruction()->users();
        if (!users.empty()) {
          if (live_set == nullptr) {
            live_set = &live_buffers[buffer];
          }
          live_set->insert(users.begin(), users.end());
        }
      }

      // Add a nullptr sentry to ensure entry parameters and output source
      // buffers are not freed until the very end.
      const bool entry_parameter =
          &computation == computation.parent()->entry_computation() &&
          buffer->instruction()->opcode() == HloOpcode::kParameter;
      const bool output = output_source_buffers.count(buffer) > 0;
      if (entry_parameter || output) {
        live_buffers[buffer].insert(nullptr);
      }

      // If the buffer has no users and isn't an entry parameter or output, it
      // must be a dead value.
      if (live_buffers.count(buffer) == 0) {
        dead_buffers_to_free.push_back(buffer);
      }
    }

    // Update live_buffers to indicate we've visited this instruction; this is
    // the inverse of the initialization logic.  We erase this instruction from
    // all source buffers of all operands of this instruction.  Buffers that
    // have no instructions left to visit are moved from live_buffers to
    // operand_buffers_to_free.
    operand_buffers_to_free.clear();
    for (const LogicalBuffer* operand_buffer :
         UniqueOperandSourceBuffers(instruction, points_to_analysis)) {
      if (IgnoreBuffer(operand_buffer)) {
        continue;
      }
      auto it = live_buffers.find(operand_buffer);
      FlatSet<const HloInstruction*>* live_set = &it->second;
      live_set->erase(instruction);
      if (live_set->empty()) {
        live_buffers.erase(it);
        operand_buffers_to_free.push_back(operand_buffer);
      }
    }

    // Allocate buffers defined by this instruction.  This is the latest point
    // that we can allocate; right before the buffer is first used.  This must
    // happen before dead or operand buffers are freed; the instruction reads
    // the operand buffers to produce its output.
    //
    // INVARIANT: Either Alloc or ShareBuffer will be called for each buffer
    // that we should assign.
    for (const LogicalBuffer* buffer : buffers_defined_by_instruction) {
      if (IgnoreBuffer(buffer)) {
        continue;
      }

      // Check whether the buffer can share with one of its operands; we can
      // save memory by sharing the buffer, rather than allocating a new one.
      // We can only share with the operand buffer if it is about to be freed;
      // we must be the last user of the buffer.
      bool shared = false;
      for (const LogicalBuffer* operand_buffer : operand_buffers_to_free) {
        if (buffer->instruction()->IsUserOf(operand_buffer->instruction()) &&
            buffer->instruction()->opcode() != HloOpcode::kCopy &&
            CanShareOperandBufferWithUser(
                operand_buffer->instruction(), operand_buffer->index(),
                buffer->instruction(), buffer->index(), points_to_analysis)) {
          ShareBuffer(buffer, operand_buffer, instruction);
          shared = true;
          break;
        }
      }

      if (!shared) {
        Alloc(buffer, instruction);
      }
    }

    // If the whole module is sequential, we can save memory by running the
    // heap-simulation for sub-computations inline. E.g. the buffers for the
    // condition and body of a kWhile instruction are only live for the duration
    // of the instruction itself.
    //
    // The order that the sub-computations are simulated does not affect
    // correctness; since the whole module is sequential, we know that the
    // sub-computations will never be run concurrently.
    if (module_sequence_ != nullptr) {
      if (instruction->opcode() == HloOpcode::kCall ||
          instruction->opcode() == HloOpcode::kWhile) {
        for (const HloComputation* called_computation :
             instruction->called_computations()) {
          const std::vector<const HloInstruction*>& called_sequence =
              FindOrDie(*module_sequence_, called_computation);
          TF_RETURN_IF_ERROR(RunComputation(
              *called_computation, called_sequence, points_to_analysis));
        }
      }

      // Other sub-computations (e.g. Map, Reduce, ...) are skipped; they are
      // assigned "thread-local" allocations, meaning their buffers are not
      // allocated up-front at the beginning of the computation.
    }

    // Free buffers that are no longer live.  This is the earliest point that we
    // can de-allocate; right after the last use of the buffer.
    for (const LogicalBuffer* buffer : dead_buffers_to_free) {
      Free(buffer, instruction);
    }
    for (const LogicalBuffer* buffer : operand_buffers_to_free) {
      Free(buffer, instruction);
    }
  }

  // Any remaining live buffers must be entry parameters or output source
  // buffers, which had a nullptr sentry added.  Free them now.
  for (const auto& buffer_pending : live_buffers) {
    const LogicalBuffer* buffer = buffer_pending.first;
    const FlatSet<const HloInstruction*>& pending = buffer_pending.second;
    CHECK_EQ(pending.size(), 1) << *buffer;
    CHECK(*pending.begin() == nullptr) << *buffer;
    Free(buffer, root);
  }

  return Status::OK();
}

HeapSimulator::HeapSimulator(
    std::unique_ptr<HeapAlgorithm> algorithm,
    const LogicalBuffer::SizeFunction& size_fn,
    const FlatSet<const LogicalBuffer*>* buffers_to_assign,
    const SequentialHloOrdering::HloModuleSequence* module_sequence)
    : no_fragmentation_stats_(MakeUnique<NoFragmentationStatsHeap>()),
      algorithm_(std::move(algorithm)),
      size_fn_(size_fn),
      buffers_to_assign_(buffers_to_assign),
      module_sequence_(module_sequence) {
  debug_trace_.set_whole_module_simulation(module_sequence_ != nullptr);
}

HeapSimulator::~HeapSimulator() {}

bool HeapSimulator::IgnoreBuffer(const LogicalBuffer* buffer) const {
  // Buffers for constants are ignored, as with BufferAssigner.  Also ignore
  // buffers that we're not meant to assign.
  //
  // TODO(b/32248867): For consistency, constants should get allocations.
  return buffer->instruction()->opcode() == HloOpcode::kConstant ||
         (buffers_to_assign_ != nullptr &&
          buffers_to_assign_->count(buffer) == 0);
}

// Alloc always calls the underlying heap algorithm.
void HeapSimulator::Alloc(const LogicalBuffer* buffer,
                          const HloInstruction* instruction) {
  CHECK(allocated_buffers_.count(buffer) == 0)
      << "Alloc called on allocated buffer: " << *buffer;
  CHECK(freed_buffers_.count(buffer) == 0)
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
void HeapSimulator::Free(const LogicalBuffer* buffer,
                         const HloInstruction* instruction) {
  auto shared_it = shared_buffers_.find(buffer);
  if (shared_it != shared_buffers_.end()) {
    std::shared_ptr<SharedGroup> group = shared_it->second;
    --group->refcount;
    if (group->refcount > 0) {
      return;
    }
    CHECK_EQ(group->refcount, 0)
        << "Free caused negative refcount on shared buffer: " << *buffer;
    buffer = group->canonical;
  }

  CHECK(allocated_buffers_.count(buffer) > 0)
      << "Free called on non-allocated buffer: " << *buffer;
  CHECK(freed_buffers_.count(buffer) == 0)
      << "Free called on freed buffer: " << *buffer;

  freed_buffers_.insert(buffer);
  const int64 size = size_fn_(*buffer);
  algorithm_->Free(buffer, size);
  no_fragmentation_stats_->Free(buffer, size);

  FillDebugTrace(HeapSimulatorTrace::Event::FREE, buffer, instruction, nullptr);
}

// ShareBuffer associates buffers with their SharedGroup in shared_buffers_.
// The 'buffer' must be a non-allocated, non-freed buffer, just like in calls to
// Alloc.  The 'shared' buffer must be a previously allocated or shared buffer.
// Both 'buffer' and 'shared' will be associated with the same SharedGroup.
void HeapSimulator::ShareBuffer(const LogicalBuffer* buffer,
                                const LogicalBuffer* shared,
                                const HloInstruction* instruction) {
  CHECK_LE(size_fn_(*buffer), size_fn_(*shared))
      << "ShareBuffer oversized buffer" << *buffer << " shared: " << *shared;
  CHECK(allocated_buffers_.count(buffer) == 0)
      << "ShareBuffer called on allocated buffer: " << *buffer;
  CHECK(freed_buffers_.count(buffer) == 0)
      << "ShareBuffer called on freed buffer: " << *buffer;
  CHECK(freed_buffers_.count(shared) == 0)
      << "ShareBuffer called on freed shared buffer: " << *shared;

  const LogicalBuffer* canonical = nullptr;
  auto shared_it = shared_buffers_.find(shared);
  if (shared_it != shared_buffers_.end()) {
    // The 'shared' buffer already has a group; it might be the canonical, but
    // also might not be.  Just add 'buffer' to the existing group.
    std::shared_ptr<SharedGroup> group = shared_it->second;
    canonical = group->canonical;
    ++group->refcount;
    shared_buffers_.emplace(buffer, group);
  } else {
    // The 'shared' buffer doesn't have a group; it must be the canonical.  Add
    // both 'buffer' and 'shared' to a new group.
    CHECK(allocated_buffers_.count(shared) > 0)
        << "ShareBuffer called on non-allocated shared buffer: " << *shared;
    auto group = std::make_shared<SharedGroup>();
    canonical = shared;
    group->canonical = canonical;
    group->refcount = 2;
    shared_buffers_.emplace(buffer, group);
    shared_buffers_.emplace(shared, group);
  }

  FillDebugTrace(HeapSimulatorTrace::Event::SHARE_WITH, buffer, instruction,
                 canonical);
}

HeapSimulator::Result HeapSimulator::Finish() {
  Result result = algorithm_->Finish();

  // Post-process the result to add chunks for shared buffers.  An empty chunk
  // map means that either no buffers were allocated, or the heap was only
  // collecting statistics, e.g. NoFragmentationStatsHeap.
  if (!result.chunk_map.empty()) {
    for (const auto& share_pair : shared_buffers_) {
      const LogicalBuffer* buffer = share_pair.first;
      std::shared_ptr<SharedGroup> group = share_pair.second;
      if (buffer != group->canonical) {
        // The canonical must already exist in the chunk_map, since we called
        // Alloc(canonical) on the underlying algorithm.  Add non-canonical
        // chunks with the same offset as the canonical.
        Chunk chunk = FindOrDie(result.chunk_map, group->canonical);
        chunk.size = size_fn_(*buffer);
        result.chunk_map.emplace(buffer, chunk);
      }
    }
    // If we were told to assign specific buffers, make sure we've assigned
    // exactly that many buffers.
    if (buffers_to_assign_ != nullptr) {
      CHECK_EQ(buffers_to_assign_->size(), result.chunk_map.size());
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
                                   const LogicalBuffer* buffer,
                                   const HloInstruction* instruction,
                                   const LogicalBuffer* share_with_canonical) {
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

void NoFragmentationStatsHeap::Alloc(const LogicalBuffer* buffer, int64 size) {
  current_heap_size_ += size;
  if (current_heap_size_ > max_heap_size_) {
    max_heap_size_ = current_heap_size_;
  }
}

void NoFragmentationStatsHeap::Free(const LogicalBuffer* buffer, int64 size) {
  current_heap_size_ -= size;
}

HeapSimulator::Result NoFragmentationStatsHeap::Finish() {
  // The result.chunk_map is empty, since we only collect stats, and don't
  // actually compute chunk assignments.
  Result result;
  result.heap_size = max_heap_size_;
  return result;
}

void DecreasingSizeRunsHeap::Alloc(const LogicalBuffer* buffer, int64 size) {
  SetMode(kAlloc);
  run_.emplace_back(Op{buffer, size});
}

void DecreasingSizeRunsHeap::Free(const LogicalBuffer* buffer, int64 size) {
  CHECK(mode_ != kInit) << "Free called on empty heap: " << *buffer;
  SetMode(kFree);
  run_.emplace_back(Op{buffer, size});
}

HeapSimulator::Result DecreasingSizeRunsHeap::Finish() {
  CallAndDrainRun();
  return algorithm_->Finish();
}

void DecreasingSizeRunsHeap::SetMode(Mode mode) {
  if (mode_ != mode) {
    CallAndDrainRun();
    mode_ = mode;
  }
}

void DecreasingSizeRunsHeap::CallAndDrainRun() {
  if (mode_ == kInit) {
    CHECK(run_.empty());
    return;
  }

  // Call ops in the run sorted by decreasing size, breaking ties by buffer id.
  std::sort(run_.begin(), run_.end(), [](const Op& a, const Op& b) {
    if (a.size != b.size) {
      return a.size > b.size;
    }
    return a.buffer->id() < b.buffer->id();
  });
  for (const Op& op : run_) {
    if (mode_ == kAlloc) {
      algorithm_->Alloc(op.buffer, op.size);
    } else {
      algorithm_->Free(op.buffer, op.size);
    }
  }
  run_.clear();
}

void LazyBestFitHeap::Alloc(const LogicalBuffer* buffer, int64 size) {
  // Degenerate case: 0-sized buffers are always allocated at offset 0.
  if (size == 0) {
    result_.chunk_map.emplace(buffer, Chunk{0, 0});
  }

  // First try to allocate from the best-fitting free chunk.
  auto best_fit_it = free_.lower_bound(Chunk{0, size});
  while (best_fit_it != free_.end()) {
    // Account for alignment.
    const Chunk best = *best_fit_it;
    const int64 new_offset = RoundUpToNearest(best.offset, alignment_);
    const int64 new_end = new_offset + size;
    if (new_end > best.chunk_end()) {
      // We don't fit after accounting for alignment.
      ++best_fit_it;
      continue;
    }
    // The buffer is allocated a chunk out of the best-fitting free chunk.
    free_.erase(best_fit_it);
    result_.chunk_map.emplace(buffer, Chunk{new_offset, size});
    // Add remaining portions of the best-fitting free chunk back into free_.
    AddFreeChunk(best.offset, new_offset - best.offset);
    AddFreeChunk(new_end, best.chunk_end() - new_end);
    return;
  }

  // The buffer doesn't completely fit into any existing free chunk.  If the
  // last free chunk is adjacent to the end of the heap, allocate the buffer
  // re-using that space, increasing the heap size.
  //
  // Allocating the buffer now causes the heap to grow by less than the buffer
  // size, whereas if we allocated lazily in Free, the heap would grow by
  // exactly the buffer size.  However it's still a greedy heuristical approach;
  // we might have ended up with a tighter packing by being lazy here.
  //
  // In theory we could also check if we could re-use space from the first free
  // chunk and grow the heap at the front, and choose whether to grow from the
  // front or back based on the amount of re-use.  But that's more complicated,
  // and these are all heuristics anyways, so it isn't implemented.
  for (auto it = free_.begin(); it != free_.end(); ++it) {
    if (it->chunk_end() == result_.heap_size) {
      // Account for alignment in the last free chunk.
      const Chunk last = *it;
      const int64 new_offset = RoundUpToNearest(last.offset, alignment_);
      if (new_offset >= last.chunk_end()) {
        // There's no point in using the last free chunk if alignment causes us
        // to skip over it anyways.
        break;
      }
      // The buffer is allocated a chunk that includes the last free chunk.
      free_.erase(it);
      result_.chunk_map.emplace(buffer, Chunk{new_offset, size});
      // Add remaining portion of the last free chunk back into free_.
      AddFreeChunk(last.offset, new_offset - last.offset);
      // Grow the heap.
      const int64 new_end = new_offset + size;
      CHECK_GT(new_end, result_.heap_size);
      CHECK_LT(new_end, result_.heap_size + size);
      result_.heap_size = new_end;
      return;
    }
  }

  // Otherwise lazily allocate the buffer in Free.
  result_.chunk_map.emplace(buffer, Chunk{kLazyAllocOffset, size});
}

void LazyBestFitHeap::Free(const LogicalBuffer* buffer, int64 size) {
  auto alloc_it = result_.chunk_map.find(buffer);
  CHECK(alloc_it != result_.chunk_map.end())
      << "Free called on non-allocated buffer: " << *buffer;
  Chunk* alloc = &alloc_it->second;
  CHECK_EQ(alloc->size, size) << "Free with mismatched sizes: " << *buffer;
  if (alloc->offset != kLazyAllocOffset) {
    // The buffer was already allocated in Alloc, do a normal free.
    AddFreeChunk(alloc->offset, alloc->size);
  } else {
    // This buffer is lazily allocated, so we *can not* allocate out of existing
    // free chunks, since that might cause interference between buffers.  The
    // buffer is allocated by growing the heap, accounting for alignment.
    alloc->offset = RoundUpToNearest(result_.heap_size, alignment_);
    const int64 new_end = alloc->chunk_end();
    AddFreeChunk(result_.heap_size, new_end - result_.heap_size);
    CHECK_GT(new_end, result_.heap_size);
    CHECK_GE(new_end, result_.heap_size + alloc->size);
    result_.heap_size = new_end;
  }
}

void LazyBestFitHeap::AddFreeChunk(int64 offset, int64 size) {
  if (size <= 0) {
    return;
  }

  // Coalesce the chunk with adjacent free chunks on either side.  We must
  // remove the free chunks from free_, since it's ordered by size.
  Chunk chunk{offset, size};
  for (auto it = free_.begin(); it != free_.end();) {
    if (it->chunk_end() == chunk.offset || it->offset == chunk.chunk_end()) {
      chunk.offset = std::min(chunk.offset, it->offset);
      chunk.size += it->size;
      it = free_.erase(it);
    } else {
      ++it;
    }
  }

  // This is the only place we add free chunks to free_.  It maintains the
  // invariant that all free chunks are disjoint and non-adjacent.
  free_.emplace(chunk);
}

HeapSimulator::Result LazyBestFitHeap::Finish() {
  if (!free_.empty()) {
    // When Finish is called, all calls to Alloc must have had corresponding
    // calls to Free, which will result in a single free chunk [0, heap_size).
    CHECK_EQ(free_.size(), 1);
    CHECK_EQ(free_.begin()->offset, 0);
    CHECK_EQ(free_.begin()->size, result_.heap_size);
  }
  return result_;
}

}  // namespace xla
