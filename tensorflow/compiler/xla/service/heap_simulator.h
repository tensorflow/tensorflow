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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HEAP_SIMULATOR_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HEAP_SIMULATOR_H_

#include <memory>
#include <set>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/service/buffer_value.h"
#include "tensorflow/compiler/xla/service/buffer_value_containers.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_ordering.h"
#include "tensorflow/compiler/xla/service/tuple_points_to_analysis.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/lib/gtl/flatset.h"

namespace xla {

// Returns the minimum memory required to compute an HLO module where all
// computations have been scheduled (represented by the given module_sequence),
// assuming no fragmentation.
StatusOr<int64> MinimumMemoryForModule(
    const SequentialHloOrdering::HloModuleSequence& module_sequence,
    const LogicalBuffer::SizeFunction& size_function);

// Returns the minimum memory required to compute the given computation,
// assuming no fragmentation.
StatusOr<int64> MinimumMemoryForComputation(
    const HloComputation& computation,
    const std::vector<const HloInstruction*>& sequence,
    const TuplePointsToAnalysis& points_to_analysis,
    const LogicalBuffer::SizeFunction& size_function);

// Forward declare classes defined below.
class HeapAlgorithm;

// HeapSimulator assigns buffer offsets by running a simulation of a regular
// memory heap with Alloc and Free calls.  It only works for completely
// sequential instruction sequences.  Unlike regular heaps, we have the
// advantage that the sequence of Alloc and Free calls is known up-front; we
// don't need to return the assignment of buffer offsets until the very end.
class HeapSimulator {
 public:
  // Chunk represents a contiguous piece of memory.  Each BufferValue will be
  // associated with a chunk in the assignment result.
  struct Chunk {
    int64 offset;
    int64 size;

    int64 chunk_end() const { return offset + size; }
  };

  // Result represents the result of the heap simulation.
  struct Result {
    // The assignment of buffers to chunks.
    tensorflow::gtl::FlatMap<const BufferValue*, Chunk> chunk_map;

    // The total size in bytes of the heap, containing all assigned chunks.
    int64 heap_size = 0;

    // The total size in bytes of heap fragmentation.
    int64 fragmentation_size = 0;

    // A trace of heap simulation events.
    HeapSimulatorTrace debug_trace;
  };

  // The different options to be passed to the Run() APIs.
  struct Options {
    Options()
        : may_reuse_operand_buffers(true),
          alloc_constants(false),
          buffers_to_assign(nullptr) {}

    // Whether a buffer about to be Free()-ed, can be recycled for a new born
    // one, hence collapsing Free()+Alloc() calls (default true).
    bool may_reuse_operand_buffers;
    // Whether to issue Alloc() and Free() calls for constants (default false).
    bool alloc_constants;
    // If 'buffers_to_assign' is provided, only those buffers are assigned
    // offsets, otherwise all buffers defined by the instructions are assigned.
    const BufferValueFlatSet* buffers_to_assign;
  };

  // Run the heap simulation with the given algorithm, assuming the given
  // module_sequence, which must contain a topologically-consistent total
  // ordering of all instructions within each computation. The result is invalid
  // if instructions are not run in exactly this sequence.
  //
  // Running heap simulation on the whole module tends to save memory, compared
  // to running on a per-computation basis, since we can re-use buffer space for
  // called sub-computations.
  //
  static StatusOr<Result> Run(
      std::unique_ptr<HeapAlgorithm> algorithm, const HloModule& module,
      const SequentialHloOrdering::HloModuleSequence& module_sequence,
      const TuplePointsToAnalysis& points_to_analysis,
      const BufferValue::SizeFunction& size_fn,
      const Options& options = Options());

  // Same as above, but runs on a single computation. The 'instruction_sequence'
  // must contain a topologically-consistent total ordering of all instructions
  // in the computation. The result is invalid if instructions are not run in
  // exactly this sequence.
  static StatusOr<Result> Run(
      std::unique_ptr<HeapAlgorithm> algorithm,
      const HloComputation& computation,
      const std::vector<const HloInstruction*>& instruction_sequence,
      const TuplePointsToAnalysis& points_to_analysis,
      const BufferValue::SizeFunction& size_fn,
      const Options& options = Options());

 private:
  // If 'module_sequence' is non-null, it is used to find kCall and kWhile
  // sub-computations, and the heap simulation for those sub-computations will
  // be run recursively. I.e. the simulation is run over the whole module.
  HeapSimulator(
      std::unique_ptr<HeapAlgorithm> algorithm,
      const BufferValue::SizeFunction& size_fn, const Options& options,
      const SequentialHloOrdering::HloModuleSequence* module_sequence);
  ~HeapSimulator();

  Status RunComputation(
      const HloComputation& computation,
      const std::vector<const HloInstruction*>& instruction_sequence,
      const TuplePointsToAnalysis& points_to_analysis);

  bool IgnoreBuffer(const BufferValue* buffer) const;
  void Alloc(const BufferValue* buffer, const HloInstruction* instruction);
  void Free(const BufferValue* buffer, const HloInstruction* instruction);
  void ShareBuffer(const BufferValue* buffer, const BufferValue* shared,
                   const HloInstruction* instruction);
  Result Finish();

  void FillDebugTrace(HeapSimulatorTrace::Event::Kind kind,
                      const BufferValue* buffer,
                      const HloInstruction* instruction,
                      const BufferValue* shared_with_canonical);

  const std::unique_ptr<HeapAlgorithm> no_fragmentation_stats_;
  const std::unique_ptr<HeapAlgorithm> algorithm_;
  const BufferValue::SizeFunction size_fn_;
  const Options options_;
  const SequentialHloOrdering::HloModuleSequence* module_sequence_;

  // In addition to Alloc and Free, the heap simulator exposes a concept of
  // buffer sharing.  When ShareBuffer is called, instead of allocating new
  // space for the buffer, it associates the buffer with a previously allocated
  // (or shared) buffer.  Each group of mutually-shared buffers points to a
  // single SharedGroup instance, which is a shared control block.
  //
  // This forced buffer sharing is hidden from the underlying heap algorithm,
  // which only sees a regular Alloc call on the canonical buffer.  The
  // corresponding Free call is delayed until the liveness of all shared buffers
  // in the group has expired, which is tracked via the refcount.  The results
  // are post-processed in Finish to add chunks for shared buffers.
  //
  // The shared_buffers_ map associates each shared buffer (including the
  // canonical) to its SharedGroup control block.
  struct SharedGroup {
    const BufferValue* canonical = nullptr;
    int64 refcount = 0;
  };
  tensorflow::gtl::FlatMap<const BufferValue*, std::shared_ptr<SharedGroup>>
      shared_buffers_;

  // Hold some sets for error-checking the sequence of Alloc and Free calls.
  tensorflow::gtl::FlatSet<const BufferValue*> allocated_buffers_;
  tensorflow::gtl::FlatSet<const BufferValue*> freed_buffers_;

  // Debugging information filled in while the heap simulator runs.
  HeapSimulatorTrace debug_trace_;
};

// Abstract base class describing a heap simulation algorithm that assigns
// offsets to buffers.  A sequence of Alloc / Free calls will be made, with the
// same semantics as a regular memory heap.  Finish will be called at the end to
// collect the simulation results.
class HeapAlgorithm {
 public:
  using Chunk = HeapSimulator::Chunk;
  using Result = HeapSimulator::Result;

  virtual ~HeapAlgorithm() = default;

  // Alloc allocates a buffer of 'size' bytes.
  virtual void Alloc(const BufferValue* buffer, int64 size) = 0;

  // Free de-allocates a previously allocated buffer.
  virtual void Free(const BufferValue* buffer, int64 size) = 0;

  // Finish collects the buffer offset assignment results.  Free may only be
  // called once, after the Alloc and Free calls.
  virtual Result Finish() = 0;
};

// NoFragmentationStatsHeap computes the heap size assuming no fragmentation;
// this is the absolute minimum size for a given instruction sequence.  The
// result.chunk_map returned in Finish is always empty, since we only collect
// stats, and don't actually compute chunk assignments.
class NoFragmentationStatsHeap : public HeapAlgorithm {
 public:
  NoFragmentationStatsHeap() = default;
  ~NoFragmentationStatsHeap() override = default;

  void Alloc(const BufferValue* buffer, int64 size) override;
  void Free(const BufferValue* buffer, int64 size) override;
  Result Finish() override;

 private:
  int64 current_heap_size_ = 0;
  int64 max_heap_size_ = 0;
};

// DecreasingSizeRunsHeap collects runs of Alloc and Free calls, sorts them by
// decreasing size, and delegates the actual calls to another heap algorithm.
// This greedy heuristic tends to reduce fragmentation for all algorithms.
class DecreasingSizeRunsHeap : public HeapAlgorithm {
 public:
  DecreasingSizeRunsHeap(std::unique_ptr<HeapAlgorithm> algorithm)
      : algorithm_(std::move(algorithm)) {}
  ~DecreasingSizeRunsHeap() override {}

  void Alloc(const BufferValue* buffer, int64 size) override;
  void Free(const BufferValue* buffer, int64 size) override;
  Result Finish() override;

 private:
  // A single Alloc or Free operation that we've buffered in run_.
  struct Op {
    const BufferValue* buffer;
    int64 size;
  };

  // Current collection mode; kInit means no ops have been collected yet.
  enum Mode { kInit, kAlloc, kFree };

  void SetMode(Mode mode);
  void CallAndDrainRun();

  const std::unique_ptr<HeapAlgorithm> algorithm_;
  std::vector<Op> run_;
  Mode mode_ = kInit;
};

// LazyBestFitHeap is a variant of the traditional best-fit heap.  This is a
// greedy heuristic, based on the idea that delaying offset assignment helps
// reduce fragmentation.  Here's an example of a "bad" offset assignment, where
// a tiny buffer A prevents adjacent free chunks from being coalesced:
//    BAD: |  free  |A|  free  |
// If we could have delayed the assignment of A, we might have ended up with:
//   GOOD: |      free       |A|
//
// In general it's actually hard to say whether GOOD is better than BAD; the
// heuristic we use is we try to leave large contiguous chunks free, and we try
// to avoid growing the overall heap size unless necessary.
//
// Just like regular best-fit, in Alloc we look for the smallest free chunk that
// fits the requested size.  Unlike regular best-fit, we postpone offset
// assignment for buffers that cannot re-use existing free chunks (and force us
// to grow the heap); these buffers are "lazily" assigned offsets in Free.
class LazyBestFitHeap : public HeapAlgorithm {
 public:
  LazyBestFitHeap(int64 alignment) : alignment_(alignment) {}
  ~LazyBestFitHeap() override {}

  void Alloc(const BufferValue* buffer, int64 size) override;
  void Free(const BufferValue* buffer, int64 size) override;
  Result Finish() override;

 private:
  // Sentry value used to indicate a chunk that wasn't assigned an offset in
  // Alloc, and will instead be assigned an offset in Free.
  enum { kLazyAllocOffset = -1 };

  struct OrderChunkByIncreasingSize {
    bool operator()(const Chunk& a, const Chunk& b) const {
      if (a.size != b.size) return a.size < b.size;
      return a.offset < b.offset;
    }
  };

  void AddFreeChunk(int64 offset, int64 size);

  const int64 alignment_;
  Result result_;

  // Maintain the set of free chunks, ordered by increasing size.
  std::set<Chunk, OrderChunkByIncreasingSize> free_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HEAP_SIMULATOR_H_
