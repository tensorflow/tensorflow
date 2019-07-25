/* Copyright 2019 Google LLC. All Rights Reserved.

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

#include "tensorflow/lite/experimental/ruy/trmul.h"

#include <cstring>

#include "profiling/instrumentation.h"
#include "tensorflow/lite/experimental/ruy/allocator.h"
#include "tensorflow/lite/experimental/ruy/block_map.h"
#include "tensorflow/lite/experimental/ruy/common.h"
#include "tensorflow/lite/experimental/ruy/opt_set.h"
#include "tensorflow/lite/experimental/ruy/side_pair.h"
#include "tensorflow/lite/experimental/ruy/spec.h"
#include "tensorflow/lite/experimental/ruy/thread_pool.h"
#include "tensorflow/lite/experimental/ruy/trace.h"

namespace ruy {

namespace {

struct TrMulTask final : Task {
  TrMulTask(TrMulParams* params_, const BlockMap& block_map_,
            std::atomic<int>* atomic_block_id_, int thread_id_,
            SidePair<std::atomic<bool>*> packed_,
            TuningResolver* tuning_resolver_, Allocator* local_allocator_,
            Trace* trace_)
      : params(params_),
        block_map(block_map_),
        atomic_block_id(atomic_block_id_),
        thread_id(thread_id_),
        packed(packed_),
        tuning_resolver(tuning_resolver_),
        local_allocator(local_allocator_),
        trace(trace_) {}

  void Run() override {
    TraceRecordThreadStart(thread_id, trace);

    // Local indicators of packedness to avoid the overhead of atomic ops.
    SidePair<bool*> local_packed{nullptr, nullptr};

    for (Side side : {Side::kLhs, Side::kRhs}) {
      if (packed[side]) {
        const int size = NumBlocksPerSide(side, block_map);
        local_allocator->Allocate(size, &local_packed[side]);
        memset(local_packed[side], 0, size * sizeof(bool));
      }
    }

    const int num_blocks = NumBlocks(block_map);

    const Tuning tuning = tuning_resolver->Resolve();

    TraceRecordThreadLoopStart(thread_id, trace);

    SidePair<int> block;
    SidePair<int> start;
    SidePair<int> end;

    // Each thread starts by initially reserving the block whose id
    // is the thread id.
    int block_id = thread_id;
    TraceRecordBlockReserved(thread_id, block_id, trace);

    while (block_id < num_blocks) {
      // Reserve the next block to handle. In order to hide the latency
      // (typically comparable to an access to the level of data cache that
      // is shared among CPU cores, e.g. 60 cycles on an ARM CPU as of 2019)
      // of this atomic operation, we structure this code so as to avoid
      // immediately depending on the `next_n` result.
      const int next_block_id =
          atomic_block_id->fetch_add(1, std::memory_order_relaxed);
      TraceRecordBlockReserved(thread_id, next_block_id, trace);
      // Get coordinates of the current block to handle, in "block space".
      GetBlockByIndex(block_map, block_id, &block);
      // Get coordinates of the current block to handle, in matrix space.
      GetBlockMatrixCoords(block_map, block, &start, &end);
      TraceRecordBlockCoordsComputed(block_id, trace);
      // Maybe pack the current LHS/RHS block, if not already packed.
      for (Side side : {Side::kLhs, Side::kRhs}) {
        EnsurePacked(side, block_id, local_packed, block, start, end, tuning);
      }
      // Actually do matrix multiplication work
      params->RunKernel(tuning, start, end);
      TraceRecordBlockFinished(block_id, trace);
      // Move on to the next block as obtained by the atomic increment
      // at the start of this while loop iteration.
      block_id = next_block_id;
    }

    local_allocator->FreeAll();

    TraceRecordThreadEnd(thread_id, trace);
  }

 private:
  void EnsurePacked(Side side, int block_id, const SidePair<bool*> local_packed,
                    const SidePair<int>& block, const SidePair<int>& start,
                    const SidePair<int>& end, Tuning tuning) {
    // If two threads concurrently hit the same block to pack,
    // we allow them to concurrently pack it, writing the same packed matrix
    // data to the same location. That is considered worth it to avoid
    // having one thread blocked on another one. Avoiding that is considered
    // important especially on mobile, where there can be large speed
    // discrepancy between threads, e.g. if different threads are scheduled
    // on CPU cores of different types (big/little), different clock speed,
    // different contention with other processes.
    if (local_packed[side] && !local_packed[side][block[side]]) {
      if (!packed[side][block[side]].load(std::memory_order_acquire)) {
        params->RunPack(side, tuning, start, end);
        TraceRecordBlockPacked(side, block_id, trace);
        packed[side][block[side]].store(true, std::memory_order_release);
      }
      local_packed[side][block[side]] = true;
    }
  }

  TrMulParams* params;
  const BlockMap& block_map;
  std::atomic<int>* atomic_block_id;
  int thread_id;
  SidePair<std::atomic<bool>*> packed;
  TuningResolver* tuning_resolver;
  Allocator* local_allocator;
  Trace* trace;
};

void AllocatePMatrix(Allocator* allocator, PMatrix* packed) {
  packed->data = allocator->AllocateBytes(DataSize(*packed));
  packed->sums = allocator->AllocateBytes(SumsSize(*packed));
}

int GetThreadCount(Context* context, int rows, int cols, int depth) {
  // Empirically determined rule for reasonable number of
  // threads to use. This is proportional to the number of arithmetic ops
  // in this Mul (product of the 3 sizes).
  int guess = (std::uint64_t(rows) * cols * depth) >> 13;
  return clamp(guess, 1, context->max_num_threads);
}

LoopStructure GetLoopStructure(int thread_count, int rows, int cols, int depth,
                               int cache_friendly_traversal_threshold) {
  if (thread_count == 1 &&
      (rows + cols) * depth < cache_friendly_traversal_threshold) {
    return LoopStructure::kSimple;
  }
  return LoopStructure::kGeneral;
}

}  // namespace

void TrMul(TrMulParams* params, Context* context) {
  gemmlowp::ScopedProfilingLabel label("TrMul");

  PMatrix& packed_lhs = params->packed[Side::kLhs];
  PMatrix& packed_rhs = params->packed[Side::kRhs];
  DMatrix& lhs = params->src[Side::kLhs];
  DMatrix& rhs = params->src[Side::kRhs];

  const int rows = lhs.layout.cols;
  const int cols = rhs.layout.cols;
  const int depth = lhs.layout.rows;

  int thread_count = GetThreadCount(context, rows, cols, depth);
  const auto loop_structure =
      GetLoopStructure(thread_count, rows, cols, depth,
                       params->cache_friendly_traversal_threshold);
  Allocator* allocator = context->GetMainAllocator();

  // Allocate packed matrices
  for (Side side : {Side::kLhs, Side::kRhs}) {
    if (!params->is_prepacked[side]) {
      AllocatePMatrix(allocator, &params->packed[side]);
    }
  }

  // Case of running this TrMul as a simple loop.
  // This is a good place to start reading this function: all the rest
  // of this function is just an optimized, but functionally equivalent,
  // version of that.
  if (loop_structure == LoopStructure::kSimple) {
    gemmlowp::ScopedProfilingLabel label_simple("TrMulImpl, simple loop");
    Tuning tuning = context->GetMainThreadTuning();

    const SidePair<int> origin{0, 0};
    const SidePair<int> rounded_dims{packed_lhs.layout.cols,
                                     packed_rhs.layout.cols};
    for (Side side : {Side::kLhs, Side::kRhs}) {
      if (!params->is_prepacked[side]) {
        params->RunPack(side, tuning, origin, rounded_dims);
      }
    }
    params->RunKernel(tuning, origin, rounded_dims);

    allocator->FreeAll();
    return;
  }

  gemmlowp::ScopedProfilingLabel label_general("TrMulImpl, general case");

  auto* trace = NewTraceOrNull(&context->tracing, rows, depth, cols);
  TraceRecordStart(trace);

  // Initialize block map.
  BlockMap block_map;
  MakeBlockMap(packed_lhs.layout.cols, packed_rhs.layout.cols, depth,
               packed_lhs.layout.kernel.cols, packed_rhs.layout.kernel.cols,
               packed_lhs.data_type.size, packed_rhs.data_type.size,
               params->cache_friendly_traversal_threshold, &block_map);

  // Initialize per-thread state.
  thread_count = clamp(thread_count, 1, NumBlocks(block_map));
  context->EnsureNPerThreadStates(thread_count);
  for (auto& per_thread_state : context->per_thread_states) {
    per_thread_state->tuning_resolver.SetTuning(context->explicit_tuning);
  }

  // Allocate and initialize atomic values tracking already-packed blocks.
  SidePair<std::atomic<bool>*> packed{nullptr, nullptr};
  for (Side side : {Side::kLhs, Side::kRhs}) {
    if (!params->is_prepacked[side]) {
      const int size = NumBlocksPerSide(side, block_map);
      allocator->Allocate(size, &packed[side]);
      for (int i = 0; i < size; i++) {
        packed[side][i].store(false, std::memory_order_relaxed);
      }
    }
  }

  // Create the atomic block id, allocate it using Allocator so that
  // we get the alignment ensuring that it sits alone in its exclusives
  // reservation granule.
  std::atomic<int>* atomic_block_id;
  allocator->Allocate(1, &atomic_block_id);

  // Create task objects.
  TrMulTask* tasks;
  allocator->Allocate(thread_count, &tasks);

  atomic_block_id->store(thread_count);

  for (int i = 0; i < thread_count; i++) {
    new (tasks + i) TrMulTask(params, block_map, atomic_block_id, i, packed,
                              &context->per_thread_states[i]->tuning_resolver,
                              &context->per_thread_states[i]->allocator, trace);
  }

  // Do the computation.
  TraceRecordExecute(trace);
  TraceStartRecordingBlockAndThreadFields(block_map, thread_count, trace);

  context->workers_pool.Execute(thread_count, tasks);

  // Finish up.
  for (int i = 0; i < thread_count; i++) {
    tasks[i].~TrMulTask();
  }

  TraceRecordEnd(trace);

  allocator->FreeAll();
}

}  // namespace ruy
