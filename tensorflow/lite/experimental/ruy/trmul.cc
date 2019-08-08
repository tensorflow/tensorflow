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
#include "tensorflow/lite/experimental/ruy/thread_pool.h"
#include "tensorflow/lite/experimental/ruy/trace.h"

namespace ruy {

namespace {

struct TrMulTask final : Task {
  TrMulTask(TrMulParams* params_, const BlockMap& block_map_,
            std::atomic<std::uint32_t>* atomic_n_, std::uint32_t thread_id_,
            std::atomic<bool>* lhs_packed_, std::atomic<bool>* rhs_packed_,
            TuningResolver* tuning_resolver_, Allocator* local_allocator_,
            Trace* trace_)
      : params(params_),
        block_map(block_map_),
        atomic_n(atomic_n_),
        thread_id(thread_id_),
        lhs_packed(lhs_packed_),
        rhs_packed(rhs_packed_),
        tuning_resolver(tuning_resolver_),
        local_allocator(local_allocator_),
        trace(trace_) {}

  void Run() override {
    TraceRecordThreadStart(thread_id, trace);

    std::uint16_t num_blocks_of_rows = NumBlocksOfRows(block_map);
    std::uint16_t num_blocks_of_cols = NumBlocksOfCols(block_map);
    std::uint32_t num_blocks = NumBlocks(block_map);

    bool* local_lhs_packed = nullptr;
    bool* local_rhs_packed = nullptr;

    if (lhs_packed) {
      local_allocator->Allocate(num_blocks_of_rows, &local_lhs_packed);
      memset(local_lhs_packed, 0, num_blocks_of_rows * sizeof(bool));
    }
    if (rhs_packed) {
      local_allocator->Allocate(num_blocks_of_cols, &local_rhs_packed);
      memset(local_rhs_packed, 0, num_blocks_of_cols * sizeof(bool));
    }

    const Tuning tuning = tuning_resolver->Resolve();

    TraceRecordThreadLoopStart(thread_id, trace);

    std::uint16_t block_r, block_c;
    int start_r, start_c, end_r, end_c;
    std::uint16_t next_block_r, next_block_c;
    int next_start_r, next_start_c, next_end_r, next_end_c;

    std::uint32_t n = thread_id;
    std::uint32_t next_n;

    GetBlockByIndex(block_map, n, &block_r, &block_c);
    TraceRecordBlockReserved(thread_id, n, trace);
    GetBlockMatrixCoords(block_map, block_r, block_c, &start_r, &start_c,
                         &end_r, &end_c);
    TraceRecordBlockCoordsComputed(n, trace);

    while (n < num_blocks) {
      // Get index of next block to handle
      next_n = atomic_n->fetch_add(1, std::memory_order_relaxed);
      // If we actually got a next block to handle (not already at end)
      if (next_n < num_blocks) {
        // Get coords of that next block to handle
        // TODO(benoitjacob): is this whole next_* business worth it?
        // The idea was to have more independent things to do in parallel
        // (Pack+Kernel on current block while we resolve next block atomic
        // index and coords) but we don't seem to be actually taking advantage
        // of this unless the compiler is doing a really good code-reordering
        // job here, which is made unlikely by the conditional enclosing this.
        GetBlockByIndex(block_map, next_n, &next_block_r, &next_block_c);
        TraceRecordBlockReserved(thread_id, next_n, trace);
        GetBlockMatrixCoords(block_map, next_block_r, next_block_c,
                             &next_start_r, &next_start_c, &next_end_r,
                             &next_end_c);
        TraceRecordBlockCoordsComputed(next_n, trace);
      }
      // Maybe pack the current LHS block, if not already packed.
      // Note that if two threads concurrently hit the same LHS block to pack,
      // we allow them to concurrently pack it, writing the same packed matrix
      // data to the same location. That is considered worth it to avoid
      // having one thread blocked on another one. Avoiding that is considered
      // important especially on mobile, where there can be large speed
      // discrepancy between threads, e.g. if different threads are scheduled
      // on CPU cores of different types (big/little), different clock speed,
      // different contention with other processes.
      if (local_lhs_packed && !local_lhs_packed[block_r]) {
        if (!lhs_packed[block_r].load(std::memory_order_acquire)) {
          params->LhsRunPack(tuning, start_r, end_r);
          TraceRecordBlockPackedLhs(n, trace);
          local_lhs_packed[block_r] = true;
          lhs_packed[block_r].store(true, std::memory_order_release);
        }
      }
      // Maybe pack the current RHS block. Same comments as above for LHS.
      if (local_rhs_packed && !local_rhs_packed[block_c]) {
        if (!rhs_packed[block_c].load(std::memory_order_acquire)) {
          params->RhsRunPack(tuning, start_c, end_c);
          TraceRecordBlockPackedRhs(n, trace);
          local_rhs_packed[block_c] = true;
          rhs_packed[block_c].store(true, std::memory_order_release);
        }
      }
      // Actually do matrix multiplication work
      params->RunKernel(tuning, start_r, start_c, end_r, end_c);
      TraceRecordBlockFinished(n, trace);
      n = next_n;
      block_r = next_block_r;
      block_c = next_block_c;
      start_r = next_start_r;
      start_c = next_start_c;
      end_r = next_end_r;
      end_c = next_end_c;
    }

    local_allocator->FreeAll();

    TraceRecordThreadEnd(thread_id, trace);
  }

 private:
  TrMulParams* params;
  const BlockMap& block_map;
  std::atomic<std::uint32_t>* atomic_n;
  std::uint32_t thread_id;
  std::atomic<bool>* lhs_packed;
  std::atomic<bool>* rhs_packed;
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

LoopStructure GetLoopStructure(int thread_count, int rows, int cols,
                               int depth) {
  if (thread_count == 1 &&
      (rows + cols) * depth < kCacheFriendlyLoopThreshold) {
    return LoopStructure::kSimple;
  }
  return LoopStructure::kGeneral;
}

}  // namespace

void TrMul(TrMulParams* params, Context* context) {
  gemmlowp::ScopedProfilingLabel label("TrMul");

  PMatrix& packed_lhs = params->packed_lhs;
  PMatrix& packed_rhs = params->packed_rhs;
  DMatrix& lhs = params->lhs;
  DMatrix& rhs = params->rhs;

  const int rows = lhs.layout.cols;
  const int cols = rhs.layout.cols;
  const int depth = lhs.layout.rows;
  const int rows_rounded_up = packed_lhs.layout.cols;
  const int cols_rounded_up = packed_rhs.layout.cols;

  int thread_count = GetThreadCount(context, rows, cols, depth);
  const auto loop_structure = GetLoopStructure(thread_count, rows, cols, depth);
  Allocator* allocator = context->GetMainAllocator();

  if (!params->lhs_is_prepacked) {
    AllocatePMatrix(allocator, &packed_lhs);
  }
  if (!params->rhs_is_prepacked) {
    AllocatePMatrix(allocator, &packed_rhs);
  }

  if (loop_structure == LoopStructure::kSimple) {
    gemmlowp::ScopedProfilingLabel label_simple("TrMulImpl, simple loop");
    Tuning tuning = context->GetMainThreadTuning();

    if (!params->lhs_is_prepacked) {
      params->LhsRunPack(tuning, 0, rows_rounded_up);
    }
    if (!params->rhs_is_prepacked) {
      params->RhsRunPack(tuning, 0, cols_rounded_up);
    }
    params->RunKernel(tuning, 0, 0, rows_rounded_up, cols_rounded_up);

    allocator->FreeAll();
    return;
  }

  gemmlowp::ScopedProfilingLabel label_general("TrMulImpl, general case");

  auto* trace = NewTraceOrNull(&context->tracing, rows, depth, cols);
  TraceRecordStart(trace);

  // Initialize block map.
  BlockMap block_map;
  MakeBlockMap(rows_rounded_up, cols_rounded_up, depth,
               packed_lhs.layout.kernel.cols, packed_rhs.layout.kernel.cols,
               packed_lhs.data_type.size, packed_rhs.data_type.size,
               &block_map);
  std::uint16_t num_blocks_of_rows = NumBlocksOfRows(block_map);
  std::uint16_t num_blocks_of_cols = NumBlocksOfCols(block_map);
  std::uint32_t num_blocks = NumBlocks(block_map);
  RUY_DCHECK_EQ(num_blocks, num_blocks_of_rows * num_blocks_of_cols);

  // Initialize per-thread state.
  thread_count = clamp(thread_count, 1, num_blocks);
  context->EnsureNPerThreadStates(thread_count);
  for (auto& per_thread_state : context->per_thread_states) {
    per_thread_state->tuning_resolver.SetTuning(context->explicit_tuning);
  }

  // Allocate memory.
  std::atomic<bool>* lhs_packed = nullptr;
  if (!params->lhs_is_prepacked) {
    allocator->Allocate(num_blocks_of_rows, &lhs_packed);
  }
  std::atomic<bool>* rhs_packed = nullptr;
  if (!params->rhs_is_prepacked) {
    allocator->Allocate(num_blocks_of_cols, &rhs_packed);
  }
  std::atomic<std::uint32_t>* atomic_n;
  allocator->Allocate(1, &atomic_n);
  TrMulTask* tasks;
  allocator->Allocate(thread_count, &tasks);

  // Initialize allocated data.
  if (lhs_packed != nullptr) {
    for (int i = 0; i < num_blocks_of_rows; i++) {
      lhs_packed[i].store(false, std::memory_order_release);
    }
  }
  if (rhs_packed != nullptr) {
    for (int i = 0; i < num_blocks_of_cols; i++) {
      rhs_packed[i].store(false, std::memory_order_release);
    }
  }
  atomic_n->store(thread_count);

  for (int i = 0; i < thread_count; i++) {
    new (tasks + i)
        TrMulTask(params, block_map, atomic_n, i, lhs_packed, rhs_packed,
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
