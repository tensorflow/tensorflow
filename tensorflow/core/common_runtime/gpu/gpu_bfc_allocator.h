/* Copyright 2015 Google Inc. All Rights Reserved.

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

#ifndef TENSORFLOW_COMMON_RUNTIME_GPU_GPU_BFC_ALLOCATOR_H_
#define TENSORFLOW_COMMON_RUNTIME_GPU_GPU_BFC_ALLOCATOR_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/common_runtime/gpu/gpu_allocator_retry.h"
#include "tensorflow/core/common_runtime/gpu/visitable_allocator.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// A GPU memory allocator that implements a 'best-fit with coalescing'
// algorithm.  This is essentially a very simple version of Doug Lea's
// malloc (dlmalloc).
//
// The goal of this allocator is to support defragmentation via
// coalescing.  One assumption we make is that the process using this
// allocator owns pretty much all of the GPU memory, and that nearly
// all requests to allocate GPU memory go through this interface.
class GPUBFCAllocator : public VisitableAllocator {
 public:
  // 'device_id' refers to the StreamExecutor ID of the device within
  // the process and must reference a valid ID in the process.
  explicit GPUBFCAllocator(int device_id, size_t total_memory);
  ~GPUBFCAllocator() override;

  string Name() override { return "gpu_bfc"; }
  void* AllocateRaw(size_t alignment, size_t num_bytes) override;
  void* AllocateRaw(size_t alignment, size_t num_bytes,
                    const AllocationAttributes& allocation_attr) override;
  void DeallocateRaw(void* ptr) override;

  void AddAllocVisitor(Visitor visitor) override;

  // Does nothing, because gpu memory is never freed.
  void AddFreeVisitor(Visitor visitor) override {}

  bool TracksAllocationSizes() override;

  size_t RequestedSize(void* ptr) override;

  size_t AllocatedSize(void* ptr) override;

  int64 AllocationId(void* ptr) override;

 private:
  struct Bin;

  void MaybeInitialize() EXCLUSIVE_LOCKS_REQUIRED(lock_);

  void* AllocateRawInternal(size_t alignment, size_t num_bytes,
                            bool dump_log_on_failure);
  void DeallocateRawInternal(void* ptr);

  // A ChunkHandle is an index into the chunks_ vector in GPUBFCAllocator
  // kInvalidChunkHandle means an invalid chunk
  typedef int ChunkHandle;
  static const int kInvalidChunkHandle = -1;

  typedef int BinNum;
  static const int kInvalidBinNum = -1;
  static const int kNumBins = 21;

  // Chunks point to GPU memory.  Their prev/next pointers form a
  // doubly-linked list of addresses sorted by GPU base address that
  // must be contiguous.  Chunks contain information about whether
  // they are in use or whether they are free, and contain a pointer
  // to the bin they are in.
  struct Chunk {
    size_t size = 0;  // Full size of GPU buffer.

    // We sometimes give chunks that are larger than needed to reduce
    // fragmentation.  requested_size keeps track of what the client
    // actually wanted so we can understand whether our splitting
    // strategy is efficient.
    size_t requested_size = 0;

    // allocation_id is set to -1 when the chunk is not in use. It is assigned a
    // value greater than zero before the chunk is returned from
    // AllocateRaw, and this value is unique among values assigned by
    // the parent allocator.
    int64 allocation_id = -1;
    void* ptr = nullptr;  // pointer to granted GPU subbuffer.

    // If not kInvalidChunkHandle, the memory referred to by 'prev' is directly
    // preceding the memory used by this chunk.  E.g., It should start
    // at 'ptr - prev->size'
    ChunkHandle prev = kInvalidChunkHandle;

    // If not kInvalidChunkHandle, the memory referred to by 'next' is directly
    // following the memory used by this chunk.  E.g., It should be at
    // 'ptr + size'
    ChunkHandle next = kInvalidChunkHandle;

    // What bin are we in?
    BinNum bin_num = kInvalidBinNum;

    bool in_use() { return allocation_id != -1; }

    string DebugString(GPUBFCAllocator* a, bool recurse) {
      string dbg;
      strings::StrAppend(&dbg, "  Size: ", strings::HumanReadableNumBytes(size),
                         " | Requested Size: ",
                         strings::HumanReadableNumBytes(requested_size),
                         " | in_use: ", in_use());
      if (recurse && prev != GPUBFCAllocator::kInvalidChunkHandle) {
        Chunk* p = a->ChunkFromHandle(prev);
        strings::StrAppend(&dbg, ", prev: ", p->DebugString(a, false));
      }
      if (recurse && next != GPUBFCAllocator::kInvalidChunkHandle) {
        Chunk* n = a->ChunkFromHandle(next);
        strings::StrAppend(&dbg, ", next: ", n->DebugString(a, false));
      }
      return dbg;
    }
  };
  // A Bin is a collection of similar-sized free chunks.
  struct Bin {
    // All chunks in this bin have >= bin_size memory.
    size_t bin_size = 0;

    struct ChunkComparator {
      explicit ChunkComparator(GPUBFCAllocator* allocator)
          : allocator_(allocator) {}
      // Sort first by size and then use pointer address as a tie breaker.
      bool operator()(const ChunkHandle ha, const ChunkHandle hb) const {
        const Chunk* a = allocator_->ChunkFromHandle(ha);
        const Chunk* b = allocator_->ChunkFromHandle(hb);
        if (a->size != b->size) {
          return a->size < b->size;
        }
        return a->ptr < b->ptr;
      }

     private:
      GPUBFCAllocator* allocator_;  // The parent allocator
    };

    typedef std::set<ChunkHandle, ChunkComparator> FreeChunkSet;
    // List of free chunks within the bin, sorted by chunk size.
    // Chunk * not owned.
    FreeChunkSet free_chunks;
    Bin(GPUBFCAllocator* allocator, size_t bs)
        : bin_size(bs), free_chunks(ChunkComparator(allocator)) {}
  };
  class PtrToChunkMap {
   public:
    explicit PtrToChunkMap(size_t memory_size, size_t min_allocation_size)
        : base_ptr_(nullptr), memory_size_(memory_size) {
      CHECK_EQ(min_allocation_size, 256);  // Otherwise shifts need to
                                           // be adjusted in this class
      const size_t n_handles =
          (memory_size + min_allocation_size - 1) / min_allocation_size;
      handles_ = new ChunkHandle[n_handles];
      for (size_t i = 0; i < n_handles; i++) {
        handles_[i] = kInvalidChunkHandle;
      }
    }
    ~PtrToChunkMap() { delete[] handles_; }

    void set_base_ptr(void* base_ptr) {
      CHECK(base_ptr_ == nullptr);
      base_ptr_ = base_ptr;
    }

    ChunkHandle get_handle(const void* p) const {
      return handles_[IndexFor(p)];
    }
    void set_handle(const void* p, ChunkHandle h) { handles_[IndexFor(p)] = h; }
    void erase(const void* p) { set_handle(p, kInvalidChunkHandle); }

   private:
    int IndexFor(const void* p) const {
      std::uintptr_t p_int = reinterpret_cast<std::uintptr_t>(p);
      std::uintptr_t base_int = reinterpret_cast<std::uintptr_t>(base_ptr_);
      DCHECK_GE(p_int, base_int);
      DCHECK_LT(p_int, base_int + memory_size_);
      return static_cast<int>(
          ((p_int - base_int) >>
           8));  // Shift by 8 because min_allocation_size is 256
    }
    void* base_ptr_;
    size_t memory_size_;
    // Array of size "memory_size / min_allocation_size".  It is
    // indexed by (p-base) / min_allocation_size, contains ChunkHandle
    // for the memory allocation represented by "p"
    ChunkHandle* handles_;
  };

  void SplitChunk(ChunkHandle h, size_t num_bytes)
      EXCLUSIVE_LOCKS_REQUIRED(lock_);
  void Merge(ChunkHandle h, ChunkHandle h2) EXCLUSIVE_LOCKS_REQUIRED(lock_);
  void FreeAndMaybeCoalesce(ChunkHandle h) EXCLUSIVE_LOCKS_REQUIRED(lock_);
  void InsertFreeChunkIntoBin(ChunkHandle h) EXCLUSIVE_LOCKS_REQUIRED(lock_);
  void RemoveFreeChunkIterFromBin(Bin::FreeChunkSet* free_chunks,
                                  const Bin::FreeChunkSet::iterator& c)
      EXCLUSIVE_LOCKS_REQUIRED(lock_);
  void RemoveFreeChunkFromBin(ChunkHandle h) EXCLUSIVE_LOCKS_REQUIRED(lock_);
  void DeleteChunk(ChunkHandle h) EXCLUSIVE_LOCKS_REQUIRED(lock_);

  void DumpMemoryLog(size_t num_bytes) EXCLUSIVE_LOCKS_REQUIRED(lock_);

  ChunkHandle AllocateChunk() EXCLUSIVE_LOCKS_REQUIRED(lock_);
  void DeallocateChunk(ChunkHandle h) EXCLUSIVE_LOCKS_REQUIRED(lock_);

  Chunk* ChunkFromHandle(ChunkHandle h) EXCLUSIVE_LOCKS_REQUIRED(lock_);

  GPUAllocatorRetry retry_helper_;

  // Structures immutable after construction
  const int device_id_;
  // The base pointer where all the GPU memory begins.
  void* base_ptr_ = nullptr;
  size_t gpu_memory_size_ = 0;
  inline int Log2FloorNonZero(uint64 n) {
#if defined(__GNUC__)
    return 63 ^ __builtin_clzll(n);
#else
    int r = 0;
    while (n > 0) {
      r++;
      n >>= 1;
    }
    return r;
#endif
  }

  // Map from bin size to Bin
  Bin* BinFromIndex(BinNum index) {
    return reinterpret_cast<Bin*>(&(bins_space_[index * sizeof(Bin)]));
  }
  size_t BinNumToSize(BinNum index) {
    return static_cast<size_t>(256) << index;
  }
  BinNum BinNumForSize(size_t bytes) {
    uint64 v = std::max<size_t>(bytes, 256) >> 8;
    int b = std::min(kNumBins - 1, Log2FloorNonZero(v));
    return b;
  }
  Bin* BinForSize(size_t bytes) { return BinFromIndex(BinNumForSize(bytes)); }

  char bins_space_[sizeof(Bin) * kNumBins];

  perftools::gputools::StreamExecutor* stream_exec_;  // Not owned.

  // Structures mutable after construction
  mutable mutex lock_;
  PtrToChunkMap ptr_to_chunk_map_ GUARDED_BY(lock_);

  std::vector<Chunk> chunks_;
  ChunkHandle free_chunks_list_;  // Ptr to head of linked list of free Chunks

  // Called once on each region, ASAP.
  std::vector<Visitor> region_visitors_;

  // Counter containing the next unique identifier to assign to a
  // newly-created chunk.
  int64 next_allocation_id_ GUARDED_BY(lock_);

  TF_DISALLOW_COPY_AND_ASSIGN(GPUBFCAllocator);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMMON_RUNTIME_GPU_GPU_BFC_ALLOCATOR_H_
