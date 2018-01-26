/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMMON_RUNTIME_BFC_ALLOCATOR_H_
#define TENSORFLOW_COMMON_RUNTIME_BFC_ALLOCATOR_H_

#include <array>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/common_runtime/allocator_retry.h"
#include "tensorflow/core/common_runtime/visitable_allocator.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {

// A memory allocator that implements a 'best-fit with coalescing'
// algorithm.  This is essentially a very simple version of Doug Lea's
// malloc (dlmalloc).
//
// The goal of this allocator is to support defragmentation via
// coalescing.  One assumption we make is that the process using this
// allocator owns pretty much all of the memory, and that nearly
// all requests to allocate memory go through this interface.
class BFCAllocator : public VisitableAllocator {
 public:
  // Takes ownership of sub_allocator.
  BFCAllocator(SubAllocator* sub_allocator, size_t total_memory,
               bool allow_growth, const string& name);
  ~BFCAllocator() override;

  string Name() override { return name_; }
  void* AllocateRaw(size_t alignment, size_t num_bytes) override;
  void* AllocateRaw(size_t alignment, size_t num_bytes,
                    const AllocationAttributes& allocation_attr) override;
  void DeallocateRaw(void* ptr) override;

  void AddAllocVisitor(Visitor visitor) override;

  // Does nothing, because memory is never freed.
  void AddFreeVisitor(Visitor visitor) override {}

  bool TracksAllocationSizes() override;

  size_t RequestedSize(void* ptr) override;

  size_t AllocatedSize(void* ptr) override;

  int64 AllocationId(void* ptr) override;

  void GetStats(AllocatorStats* stats) override;

  void ClearStats() override;

 private:
  struct Bin;

  void* AllocateRawInternal(size_t alignment, size_t num_bytes,
                            bool dump_log_on_failure);
  void DeallocateRawInternal(void* ptr);

  // A ChunkHandle is an index into the chunks_ vector in BFCAllocator
  // kInvalidChunkHandle means an invalid chunk
  typedef size_t ChunkHandle;
  static const int kInvalidChunkHandle = -1;

  typedef int BinNum;
  static const int kInvalidBinNum = -1;
  static const int kNumBins = 21;

  // Chunks point to memory.  Their prev/next pointers form a
  // doubly-linked list of addresses sorted by base address that
  // must be contiguous.  Chunks contain information about whether
  // they are in use or whether they are free, and contain a pointer
  // to the bin they are in.
  struct Chunk {
    size_t size = 0;  // Full size of buffer.

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
    void* ptr = nullptr;  // pointer to granted subbuffer.

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

    bool in_use() const { return allocation_id != -1; }

    string DebugString(BFCAllocator* a,
                       bool recurse) NO_THREAD_SAFETY_ANALYSIS {
      string dbg;
      strings::StrAppend(&dbg, "  Size: ", strings::HumanReadableNumBytes(size),
                         " | Requested Size: ",
                         strings::HumanReadableNumBytes(requested_size),
                         " | in_use: ", in_use());
      if (recurse && prev != BFCAllocator::kInvalidChunkHandle) {
        Chunk* p = a->ChunkFromHandle(prev);
        strings::StrAppend(&dbg, ", prev: ", p->DebugString(a, false));
      }
      if (recurse && next != BFCAllocator::kInvalidChunkHandle) {
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
      explicit ChunkComparator(BFCAllocator* allocator)
          : allocator_(allocator) {}
      // Sort first by size and then use pointer address as a tie breaker.
      bool operator()(const ChunkHandle ha,
                      const ChunkHandle hb) const NO_THREAD_SAFETY_ANALYSIS {
        const Chunk* a = allocator_->ChunkFromHandle(ha);
        const Chunk* b = allocator_->ChunkFromHandle(hb);
        if (a->size != b->size) {
          return a->size < b->size;
        }
        return a->ptr < b->ptr;
      }

     private:
      BFCAllocator* allocator_;  // The parent allocator
    };

    typedef std::set<ChunkHandle, ChunkComparator> FreeChunkSet;
    // List of free chunks within the bin, sorted by chunk size.
    // Chunk * not owned.
    FreeChunkSet free_chunks;
    Bin(BFCAllocator* allocator, size_t bs)
        : bin_size(bs), free_chunks(ChunkComparator(allocator)) {}
  };

  static const size_t kMinAllocationBits = 8;
  static const size_t kMinAllocationSize = 1 << kMinAllocationBits;

  // AllocationRegion maps pointers to ChunkHandles for a single
  // contiguous memory region.
  //
  // This class is thread-compatible.
  class AllocationRegion {
   public:
    AllocationRegion(void* ptr, size_t memory_size)
        : ptr_(ptr),
          memory_size_(memory_size),
          end_ptr_(
              static_cast<void*>(static_cast<char*>(ptr_) + memory_size_)) {
      DCHECK_EQ(0, memory_size % kMinAllocationSize);
      const size_t n_handles =
          (memory_size + kMinAllocationSize - 1) / kMinAllocationSize;
      handles_ = new ChunkHandle[n_handles];
      for (size_t i = 0; i < n_handles; i++) {
        handles_[i] = kInvalidChunkHandle;
      }
    }

    AllocationRegion() {}

    ~AllocationRegion() { delete[] handles_; }

    AllocationRegion(AllocationRegion&& other) { Swap(other); }

    AllocationRegion& operator=(AllocationRegion&& other) {
      Swap(other);
      return *this;
    }

    void* ptr() const { return ptr_; }
    void* end_ptr() const { return end_ptr_; }
    size_t memory_size() const { return memory_size_; }
    ChunkHandle get_handle(const void* p) const {
      return handles_[IndexFor(p)];
    }
    void set_handle(const void* p, ChunkHandle h) { handles_[IndexFor(p)] = h; }
    void erase(const void* p) { set_handle(p, kInvalidChunkHandle); }

   private:
    void Swap(AllocationRegion& other) {
      std::swap(ptr_, other.ptr_);
      std::swap(memory_size_, other.memory_size_);
      std::swap(end_ptr_, other.end_ptr_);
      std::swap(handles_, other.handles_);
    }

    int IndexFor(const void* p) const {
      std::uintptr_t p_int = reinterpret_cast<std::uintptr_t>(p);
      std::uintptr_t base_int = reinterpret_cast<std::uintptr_t>(ptr_);
      DCHECK_GE(p_int, base_int);
      DCHECK_LT(p_int, base_int + memory_size_);
      return static_cast<int>(((p_int - base_int) >> kMinAllocationBits));
    }

    // Metadata about the allocation region.
    void* ptr_ = nullptr;
    size_t memory_size_ = 0;
    void* end_ptr_ = nullptr;

    // Array of size "memory_size / kMinAllocationSize".  It is
    // indexed by (p-base) / kMinAllocationSize, contains ChunkHandle
    // for the memory allocation represented by "p"
    ChunkHandle* handles_ = nullptr;

    TF_DISALLOW_COPY_AND_ASSIGN(AllocationRegion);
  };

  // RegionManager aggregates one or more "AllocationRegions" and provides
  // a layer of indirection from pointers to the underlying ChunkHandle,
  // allowing allocation across multiple discontiguous memory regions.
  //
  // This class is thread-compatible.
  class RegionManager {
   public:
    RegionManager() {}
    ~RegionManager() {}

    void AddAllocationRegion(void* ptr, size_t memory_size) {
      // Insert sorted by end_ptr
      auto entry =
          std::upper_bound(regions_.begin(), regions_.end(), ptr, &Comparator);
      regions_.insert(entry, AllocationRegion(ptr, memory_size));
    }

    ChunkHandle get_handle(const void* p) const {
      return RegionFor(p)->get_handle(p);
    }

    void set_handle(const void* p, ChunkHandle h) {
      return MutableRegionFor(p)->set_handle(p, h);
    }
    void erase(const void* p) { return MutableRegionFor(p)->erase(p); }

    const std::vector<AllocationRegion>& regions() const { return regions_; }

   private:
    static bool Comparator(const void* ptr, const AllocationRegion& other) {
      return ptr < other.end_ptr();
    }

    AllocationRegion* MutableRegionFor(const void* p) {
      return const_cast<AllocationRegion*>(RegionFor(p));
    }

    const AllocationRegion* RegionFor(const void* p) const {
      auto entry =
          std::upper_bound(regions_.begin(), regions_.end(), p, &Comparator);

      if (entry != regions_.end()) {
        return &(*entry);
      }

      LOG(FATAL) << "Could not find Region for " << p;
      return nullptr;
    }

   private:
    std::vector<AllocationRegion> regions_;
  };

  // Returns 'bytes' rounded up to the next highest kMinAllocationSize.
  size_t RoundedBytes(size_t bytes);

  // Try to add a new memory region that can satisfy an allocation of
  // 'rounded_bytes' bytes.  Returns true on success and false on
  // failure.
  bool Extend(size_t rounded_bytes) EXCLUSIVE_LOCKS_REQUIRED(lock_);

  // Returns a pointer to an underlying allocated chunk of size
  // 'rounded_bytes'.
  void* FindChunkPtr(BinNum bin_num, size_t rounded_bytes, size_t num_bytes)
      EXCLUSIVE_LOCKS_REQUIRED(lock_);

  // Splits the chunk specified by 'h' into two chunks, one at least
  // of size 'num_bytes'.
  void SplitChunk(ChunkHandle h, size_t num_bytes)
      EXCLUSIVE_LOCKS_REQUIRED(lock_);

  // Merges the two chunk handles.  Requires that the chunks are
  // contiguous in their allocation.
  void Merge(ChunkHandle h, ChunkHandle h2) EXCLUSIVE_LOCKS_REQUIRED(lock_);

  // Frees the memory represented by 'h', coalescing the chunk if
  // possible.
  void FreeAndMaybeCoalesce(ChunkHandle h) EXCLUSIVE_LOCKS_REQUIRED(lock_);

  // Adds the chunk 'h' to the proper free bin.
  void InsertFreeChunkIntoBin(ChunkHandle h) EXCLUSIVE_LOCKS_REQUIRED(lock_);

  // Removes the free chunk pointed to by 'c' from the set free_chunks.
  void RemoveFreeChunkIterFromBin(Bin::FreeChunkSet* free_chunks,
                                  const Bin::FreeChunkSet::iterator& c)
      EXCLUSIVE_LOCKS_REQUIRED(lock_);

  // Removes a free chunk from the bin.
  void RemoveFreeChunkFromBin(ChunkHandle h) EXCLUSIVE_LOCKS_REQUIRED(lock_);

  // Removes the chunk metadata represented by 'h'.
  void DeleteChunk(ChunkHandle h) EXCLUSIVE_LOCKS_REQUIRED(lock_);

  string RenderOccupancy() EXCLUSIVE_LOCKS_REQUIRED(lock_);
  void DumpMemoryLog(size_t num_bytes) EXCLUSIVE_LOCKS_REQUIRED(lock_);

  ChunkHandle AllocateChunk() EXCLUSIVE_LOCKS_REQUIRED(lock_);
  void DeallocateChunk(ChunkHandle h) EXCLUSIVE_LOCKS_REQUIRED(lock_);

  Chunk* ChunkFromHandle(ChunkHandle h) EXCLUSIVE_LOCKS_REQUIRED(lock_);

  // Information about a Bin that is useful for debugging.
  struct BinDebugInfo {
    size_t total_bytes_in_use = 0;
    size_t total_bytes_in_bin = 0;
    size_t total_requested_bytes_in_use = 0;
    size_t total_chunks_in_use = 0;
    size_t total_chunks_in_bin = 0;
  };

  // Computes and returns a BinDebugInfo for each Bin.
  std::array<BinDebugInfo, kNumBins> get_bin_debug_info()
      EXCLUSIVE_LOCKS_REQUIRED(lock_);

  AllocatorRetry retry_helper_;

  // Structures immutable after construction
  size_t memory_limit_ = 0;

  inline int Log2FloorNonZeroSlow(uint64 n) {
    int r = 0;
    while (n > 0) {
      r++;
      n >>= 1;
    }
    return r - 1;
  }

  // Returns floor(log2(n)).
  inline int Log2FloorNonZero(uint64 n) {
#if defined(__GNUC__)
    return 63 ^ __builtin_clzll(n);
#elif defined(PLATFORM_WINDOWS)
    unsigned long index;
    _BitScanReverse64(&index, n);
    return index;
#else
    return Log2FloorNonZeroSlow(n);
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
    uint64 v = std::max<size_t>(bytes, 256) >> kMinAllocationBits;
    int b = std::min(kNumBins - 1, Log2FloorNonZero(v));
    return b;
  }
  Bin* BinForSize(size_t bytes) { return BinFromIndex(BinNumForSize(bytes)); }

  char bins_space_[sizeof(Bin) * kNumBins];

  // The size of the current region allocation.
  size_t curr_region_allocation_bytes_;

  // The total number of allocated bytes by the allocator.
  size_t total_region_allocated_bytes_ = 0;

  // An indicator that expansion of a region has hit the limits
  // of the available memory.
  bool started_backpedal_ = false;

  std::unique_ptr<SubAllocator> suballocator_;
  string name_;

  // Structures mutable after construction
  mutable mutex lock_;
  RegionManager region_manager_ GUARDED_BY(lock_);

  std::vector<Chunk> chunks_ GUARDED_BY(lock_);

  // Pointer to head of linked list of free Chunks
  ChunkHandle free_chunks_list_ GUARDED_BY(lock_);

  // Called once on each region, ASAP.
  std::vector<Visitor> region_visitors_ GUARDED_BY(lock_);

  // Counter containing the next unique identifier to assign to a
  // newly-created chunk.
  int64 next_allocation_id_ GUARDED_BY(lock_);

  // Stats.
  AllocatorStats stats_ GUARDED_BY(lock_);

  friend class GPUBFCAllocatorPrivateMethodsTest;
  TF_DISALLOW_COPY_AND_ASSIGN(BFCAllocator);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMMON_RUNTIME_BFC_ALLOCATOR_H_
