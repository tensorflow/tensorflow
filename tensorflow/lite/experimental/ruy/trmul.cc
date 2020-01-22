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

#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "tensorflow/lite/experimental/ruy/allocator.h"
#include "tensorflow/lite/experimental/ruy/block_map.h"
#include "tensorflow/lite/experimental/ruy/check_macros.h"
#include "tensorflow/lite/experimental/ruy/common.h"
#include "tensorflow/lite/experimental/ruy/internal_matrix.h"
#include "tensorflow/lite/experimental/ruy/matrix.h"
#include "tensorflow/lite/experimental/ruy/opt_set.h"
#include "tensorflow/lite/experimental/ruy/profiler/instrumentation.h"
#include "tensorflow/lite/experimental/ruy/side_pair.h"
#include "tensorflow/lite/experimental/ruy/size_util.h"
#include "tensorflow/lite/experimental/ruy/spec.h"
#include "tensorflow/lite/experimental/ruy/thread_pool.h"
#include "tensorflow/lite/experimental/ruy/trace.h"
#include "tensorflow/lite/experimental/ruy/tune.h"

namespace ruy {

namespace {

enum class PackingStatus : std::uint8_t { kNotStarted, kInProgress, kFinished };

struct TrMulTask final : Task {
  TrMulTask(TrMulParams* params_, const BlockMap& block_map_,
            std::atomic<int>* atomic_block_id_, int thread_id_,
            bool need_atomics_,
            SidePair<std::atomic<PackingStatus>*> packing_status_,
            TuningResolver* tuning_resolver_, Allocator* local_allocator_,
            Trace* trace_)
      : params(params_),
        block_map(block_map_),
        atomic_block_id(atomic_block_id_),
        thread_id(thread_id_),
        need_atomics(need_atomics_),
        packing_status(packing_status_),
        tuning_resolver(tuning_resolver_),
        local_allocator(local_allocator_),
        trace(trace_),
        local_packed{nullptr, nullptr} {}

  void Run() override {
    TraceRecordThreadStart(thread_id, trace);

    for (Side side : {Side::kLhs, Side::kRhs}) {
      if (!params->is_prepacked[side]) {
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
      // Maybe pack the current LHS/RHS block, if not already packed.
      EnsurePacked(block, start, end, tuning);
      // Actually do matrix multiplication work
      params->RunKernel(tuning, start, end);
      TraceRecordBlockFinished(thread_id, block_id, trace);
      // Move on to the next block as obtained by the atomic increment
      // at the start of this while loop iteration.
      block_id = next_block_id;
    }

    local_allocator->FreeAll();

    TraceRecordThreadEnd(thread_id, trace);
  }

 private:
  // Tries to pack a block, without blocking.
  // If the block was already packed, returns true.
  // If the block was not started packing, packs it and returns true.
  // If the block was being packed by another thread, returns false.
  bool TryPack(Side side, int block, int start, int end, Tuning tuning) {
    if (params->is_prepacked[side]) {
      return true;
    }
    if (!local_packed[side][block]) {
      if (need_atomics) {
        // Explanation of this compare_exchange_strong operation:
        // This atomically performs all of the following:
        // 1. Read `status` with "acquire" memory order.
        //    * That this read uses "acquire" is because both memory orders
        //      specified have "acquire" as their read-component.
        // 2. Compare (bitwise) with `exchanged_status`.
        // 3. If equal, stores the value kInProgress to `status` with "release"
        //    memory order, and returns true, so we take this 'if' branch.
        //    * That this store uses "release" is because of the _rel part in
        //      memory_order_acq_rel passed as the first memory order argument.
        // 4. If not equal, stores the loaded value of `status` to
        //    `exchanged_status` with "relaxed" semantics, and returns false,
        //    so we take the 'else' branch.
        //    * That this store uses "relaxed" is because the second memory
        //      order argument, memory_order_acquire, implies no particular
        //      store semantics. "relaxed" is acceptable here because this
        //      stores to a local stack variable.
        //
        // Rationale for compare_exchange_strong as opposed to
        // compare_exchange_weak:
        // The spurious-failure case with compare_exchange_weak will actually
        // happen a lot here, because the atomic 'status' bytes are stored
        // contiguously in arrays and neighboring values will be accessed
        // by multiple threads concurrently. On a typical ARM CPU, an exclusives
        // reservation granule is 64 bytes, so a lot of false-sharing may
        // happen. Using compare_exchange_weak would thus result in often having
        // TryPack return 'false' when it could instead have done the packing
        // work and returned 'true'. Heuristically, that is not a good thing.
        // Moreover, this changes the TryPack contract, loosening it and making
        // it harder for the caller to reason about. Finally, the overhead of
        // atomic operations is mitigated by the enclosing check on
        // local_packed, so maybe the overhead of compare_exchange_strong isn't
        // such a problem. But we don't really know for sure, that would be
        // interesting to experiment more with.
        PackingStatus exchanged_status = PackingStatus::kNotStarted;
        std::atomic<PackingStatus>& status = packing_status[side][block];
        if (status.compare_exchange_strong(
                exchanged_status, PackingStatus::kInProgress,
                std::memory_order_acq_rel, std::memory_order_acquire)) {
          // In this branch, the status was kNotStarted and we just atomically
          // changed it to kInProgress as we are about to handle the packing
          // ourselves.
          params->RunPack(side, tuning, start, end);
          TraceRecordBlockPacked(thread_id, side, block, trace);
          status.store(PackingStatus::kFinished, std::memory_order_release);
        } else if (exchanged_status == PackingStatus::kInProgress) {
          // Another thread is currently packing this block.
          return false;
        }
        RUY_DCHECK(status.load(std::memory_order_acquire) ==
                   PackingStatus::kFinished);
      } else {
        // Single-threaded case: no need for expensive atomics, local_packed
        // is the truth already.
        params->RunPack(side, tuning, start, end);
        TraceRecordBlockPacked(thread_id, side, block, trace);
      }
      local_packed[side][block] = true;
    }
    return true;
  }

  // Ensures that both the LHS and RHS blocks required by the specified block
  // are packed. In the event that they are already being packed on another
  // threads, this function may perform the packing of some other block while
  // waiting for that other thread to finish packing the requested block.
  void EnsurePacked(const SidePair<int>& block, const SidePair<int>& start,
                    const SidePair<int>& end, Tuning tuning) {
#if RUY_OPT_ENABLED(RUY_OPT_PACK_AHEAD)
    SidePair<int> next_runahead_block{block[Side::kLhs] + 1,
                                      block[Side::kRhs] + 1};
    Side next_runahead_side = Side::kLhs;
#endif
    while (true) {
      bool both_sides_packed = true;
      for (Side side : {Side::kLhs, Side::kRhs}) {
        both_sides_packed &=
            TryPack(side, block[side], start[side], end[side], tuning);
      }
      if (both_sides_packed) {
        break;
      }
#if RUY_OPT_ENABLED(RUY_OPT_PACK_AHEAD)
      const Side runahead_side = next_runahead_side;
      const int runahead_block = next_runahead_block[runahead_side];
      next_runahead_side =
          next_runahead_side == Side::kLhs ? Side::kRhs : Side::kLhs;
      if (runahead_block >= NumBlocksPerSide(runahead_side, block_map)) {
        continue;
      }
      int runahead_block_start, runahead_block_end;
      GetBlockMatrixCoords(runahead_side, block_map, runahead_block,
                           &runahead_block_start, &runahead_block_end);
      TryPack(runahead_side, runahead_block, runahead_block_start,
              runahead_block_end, tuning);
      next_runahead_block[runahead_side] = runahead_block + 1;
#endif
    }
  }

  TrMulParams* params;
  const BlockMap& block_map;
  std::atomic<int>* atomic_block_id;
  int thread_id;
  bool need_atomics;
  SidePair<std::atomic<PackingStatus>*> packing_status;
  TuningResolver* tuning_resolver;
  Allocator* local_allocator;
  Trace* trace;

  // Local indicators of packedness to avoid the overhead of atomic ops.
  SidePair<bool*> local_packed;
};

void AllocatePMatrix(Allocator* allocator, PMatrix* packed) {
  packed->data = allocator->AllocateBytes(DataSize(*packed));
  packed->sums = allocator->AllocateBytes(SumsSize(*packed));
}

int GetThreadCount(Context* context, int rows, int cols, int depth) {
#if RUY_PLATFORM(EMSCRIPTEN)
  // b/139927184, std::thread constructor raises exception
  return 1;
#endif
  // Empirically determined rule for reasonable number of
  // threads to use. This is proportional to the number of arithmetic ops
  // in this Mul (product of the 3 sizes).
  static constexpr int kDivisorLog2 = 15;
  const int guess_log2 = std::max(
      0, ceil_log2(rows) + ceil_log2(cols) + ceil_log2(depth) - kDivisorLog2);
  return std::min(1 << guess_log2, context->max_num_threads);
}

LoopStructure GetLoopStructure(int tentative_thread_count, int rows, int cols,
                               int depth,
                               int cache_friendly_traversal_threshold) {
  if (tentative_thread_count == 1) {
    // If we are in the GEMV case or the size is below the
    // threshold, stay with the simple loop structure.
    if ((cols == 1) ||
        (rows + cols) * depth < cache_friendly_traversal_threshold) {
      return LoopStructure::kSimple;
    }
  }
  return LoopStructure::kGeneral;
}

}  // namespace

void TrMul(TrMulParams* params, Context* context) {
  profiler::ScopeLabel label(
      "TrMul (Path=0x%x, max_num_threads=%d, is_prepacked=(%d,%d))",
      static_cast<int>(params->path), context->max_num_threads,
      params->is_prepacked[Side::kLhs], params->is_prepacked[Side::kRhs]);

  PMatrix& packed_lhs = params->packed[Side::kLhs];
  PMatrix& packed_rhs = params->packed[Side::kRhs];
  DMatrix& lhs = params->src[Side::kLhs];
  DMatrix& rhs = params->src[Side::kRhs];

  const int rows = lhs.layout.cols;
  const int cols = rhs.layout.cols;
  const int depth = lhs.layout.rows;

  const int tentative_thread_count = GetThreadCount(context, rows, cols, depth);
  const auto loop_structure =
      GetLoopStructure(tentative_thread_count, rows, cols, depth,
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
    profiler::ScopeLabel label_simple("TrMulImpl, simple loop");
    Tuning tuning = context->GetMainThreadTuning();

    const SidePair<int> origin{0, 0};
    const SidePair<int> rounded_dims{packed_lhs.layout.cols,
                                     packed_rhs.layout.cols};
    for (Side side : {Side::kLhs, Side::kRhs}) {
      if (!params->is_prepacked[side]) {
        params->RunPack(side, tuning, origin[side], rounded_dims[side]);
      }
    }
    params->RunKernel(tuning, origin, rounded_dims);

    allocator->FreeAll();
    return;
  }

  profiler::ScopeLabel label_general("TrMulImpl, general case");

  auto* trace = NewTraceOrNull(&context->tracing, rows, depth, cols);
  TraceRecordStart(trace);

  // Initialize block map.
  BlockMap block_map;
  MakeBlockMap(packed_lhs.layout.cols, packed_rhs.layout.cols, depth,
               packed_lhs.layout.kernel.cols, packed_rhs.layout.kernel.cols,
               packed_lhs.data_type.size, packed_rhs.data_type.size,
               tentative_thread_count, params->path,
               params->cache_friendly_traversal_threshold, &block_map);

  // Initialize per-thread state.
  const int thread_count = block_map.thread_count;
  const bool need_atomics = thread_count > 1;
  context->EnsureNPerThreadStates(thread_count);
  for (auto& per_thread_state : context->per_thread_states) {
    per_thread_state->tuning_resolver.SetTuning(context->explicit_tuning);
  }

  // In the need_atomics case, allocate and initialize atomic values tracking
  // the packing status of blocks.
  SidePair<std::atomic<PackingStatus>*> packing_status{nullptr, nullptr};
  if (need_atomics) {
    for (Side side : {Side::kLhs, Side::kRhs}) {
      if (!params->is_prepacked[side]) {
        const int size = NumBlocksPerSide(side, block_map);
        allocator->Allocate(size, &packing_status[side]);
        for (int i = 0; i < size; i++) {
          packing_status[side][i].store(PackingStatus::kNotStarted,
                                        std::memory_order_relaxed);
        }
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
    new (tasks + i) TrMulTask(params, block_map, atomic_block_id, i,
                              need_atomics, packing_status,
                              &context->per_thread_states[i]->tuning_resolver,
                              &context->per_thread_states[i]->allocator, trace);
  }

  // Do the computation.
  TraceRecordExecute(block_map, trace);
  context->workers_pool.Execute(thread_count, tasks);

  // Finish up.
  for (int i = 0; i < thread_count; i++) {
    tasks[i].~TrMulTask();
  }

  allocator->FreeAll();
  TraceRecordEnd(trace);
}

}  // namespace ruy
