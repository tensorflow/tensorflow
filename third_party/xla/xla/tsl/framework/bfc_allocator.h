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

#ifndef XLA_TSL_FRAMEWORK_BFC_ALLOCATOR_H_
#define XLA_TSL_FRAMEWORK_BFC_ALLOCATOR_H_

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/casts.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/tsl/framework/allocator.h"
#include "xla/tsl/framework/allocator_retry.h"
#include "xla/tsl/framework/scoped_allocation_trace.h"
#include "xla/tsl/framework/shared_counter.h"
#include "xla/tsl/lib/core/bits.h"
#include "xla/tsl/platform/logging.h"
#include "tsl/platform/numbers.h"

namespace tensorflow {
class MemoryDump;
}
namespace tsl {
using tensorflow::MemoryDump;

// A memory allocator that implements best-fit with coalescing (BFC), a
// simple dlmalloc-style allocator for arenas where most allocations go through
// this interface.
//
// See prior art: https://gee.cs.oswego.edu/dl/html/malloc.html
//
// High-level model:
//
// - Backing memory comes from the SubAllocator as AllocationRegions. With
//   Options::allow_growth=true the allocator grows by adding regions up to
//   total_memory; with Options::allow_growth=false it reserves one fixed region
//   during construction. stats_.bytes_reserved tracks bytes held from the
//   SubAllocator, while stats_.bytes_in_use tracks bytes currently live for
//   clients.
//
// - Each AllocationRegion is represented as an ordered sequence of Chunks that
//   cover the region without gaps. This is boundary-tag-style bookkeeping: the
//   allocator can find physically adjacent chunks and coalesce neighboring free
//   chunks, even though the metadata lives in Chunk objects instead of literal
//   dlmalloc headers/trailers. A Chunk is either entirely in use or entirely
//   free. Allocations split free chunks when needed, and frees coalesce
//   adjacent free chunks to repair fragmentation.
//
// - Free chunks are indexed by size-class Bins. Each Bin stores ChunkHandles in
//   a FreeChunkSet ordered by chunk size and then address. Allocation starts in
//   the smallest viable bin, scans upward, and uses the smallest fitting chunk.
//   Allocated chunks are never in a Bin.
//
// - AllocationAttributes::allocation_end controls placement. Without spatial
//   partitioning all requests use AllocationEnd::kLower, and ordinary free
//   chunks stay in ChunkTag::kLower, which is classic BFC behavior.
//
// - With Options::enable_spatial_partitioning=true, which requires
//   Options::allow_growth=false, the fixed address range is split into
//   lower-end ownership, one central gap, and upper-end ownership.
//   AllocationEnd::kLower requests grow upward, and AllocationEnd::kUpper
//   requests grow downward. ChunkTag records ownership: kLower and kUpper for
//   allocated chunks and same-tag interior holes, and kCentralGap for the
//   central gap. The central gap is tracked by central_gap_ instead of being
//   inserted into a Bin. Each end first reuses binned holes with its own tag,
//   then carves from the central gap. This keeps each end's placements
//   independent of activity from the opposite end except when lower and upper
//   allocations exhaust the central gap.
//
class BFCAllocator : public Allocator {
 public:
  struct Options {
    bool allow_growth = true;

    // If true, the allocator may sleep for a period of time when it can't
    // fulfill an allocation request, in the hopes that another thread will free
    // up memory in the meantime.
    //
    // If false, the allocator will never sleep, even if
    // AllocationAttributes::attr_retry_on_failure is true.
    bool allow_retry_on_failure = true;

    // Whether the allocator will deallocate free regions to avoid OOM due to
    // memory fragmentation.
    bool garbage_collection = false;

    // Controls when a chunk should be split, if its size exceeds the requested
    // allocation size.
    double fragmentation_fraction = 0;

    // If true, the allocator spatially partitions a single pre-allocated
    // address range by serving requests from either end. AllocationEnd::kLower
    // requests grow up from the low address; AllocationEnd::kUpper requests
    // grow down from the high address; a central gap sits in between:
    //
    //   low address                                      high address
    //   |------------------------------------------------------------|
    //   | lower-end owned --->   central gap   <--- upper-end owned |
    //   |------------------------------------------------------------|
    //
    // The split is fully dynamic with no hard boundary: a request carves from
    // the central gap or reuses a free hole of its OWN tag, but never the
    // other end's tagged interior holes. When a buffer at either end of the
    // central gap is freed it rejoins the gap, growing it, and adjacent holes
    // with the same tag cascade back in turn -- so e.g. allocating 100% lower,
    // freeing it, then allocating 100% upper is fully supported. The only
    // failure is true exhaustion: lower and upper meeting with no gap left.
    //
    // Because neither end ever carves the other's interior holes, each end's
    // placement is a pure function of that end's request sequence and is never
    // perturbed by activity from the opposite end, except when lower and upper
    // allocations exhaust the central gap. That makes offsets reproducible
    // across processes that issue the same requests for that end in the same
    // order, e.g. symmetric collective buffers across ranks.
    //
    // Requires allow_growth=false (a single fixed address range).
    bool enable_spatial_partitioning = false;
  };

  BFCAllocator(std::unique_ptr<SubAllocator> sub_allocator, size_t total_memory,
               const std::string& name, const Options& opts);

  ~BFCAllocator() override;

  std::string Name() override { return name_; }

  static constexpr size_t kMinAllocationBits = 8;
  static constexpr size_t kMinAllocationSize = 1 << kMinAllocationBits;

  void* AllocateRaw(size_t alignment, size_t num_bytes) override {
    return AllocateRaw(alignment, num_bytes, AllocationAttributes());
  }

  void* AllocateRaw(size_t alignment, size_t num_bytes,
                    const AllocationAttributes& allocation_attr) override;

  void DeallocateRaw(void* ptr) override;

  bool TracksAllocationSizes() const override;

  size_t RequestedSize(const void* ptr) const override;

  size_t AllocatedSize(const void* ptr) const override;

  int64_t AllocationId(const void* ptr) const override;

  std::optional<AllocatorStats> GetStats() override;

  bool ClearStats() override;

  void SetTimingCounter(SharedCounter* sc) { timing_counter_ = sc; }

  void SetSafeFrontier(uint64_t count) override;

  AllocatorMemoryType GetMemoryType() const override;

  bool ShouldRecordOpName() const { return true; }

  MemoryDump RecordMemoryMap();

 private:
  struct Bin;

  void* AllocateRawInternal(size_t alignment, size_t num_bytes,
                            bool dump_log_on_failure,
                            uint64_t freed_before_count,
                            AllocationEnd allocation_end);

  void* AllocateRawInternalWithRetry(
      size_t alignment, size_t num_bytes,
      const AllocationAttributes& allocation_attr);

  void DeallocateRawInternal(void* ptr);

  // Chunks whose freed_at_count is later than the safe frontier value are kept
  // on a special list and not subject to merging immediately upon being freed.
  //
  // This function sweeps that list looking for Chunks whose timestamp is now
  // safe. When found their freed_at_count is set to 0 and we attempt to merge
  // them with their neighbors.
  //
  // If required_bytes > 0 then this function is being called in the context of
  // a need for this many bytes that could not be satisfied without merging
  // unsafe chunks, so we go ahead and merge the unsafe chunks too, just up to
  // the point that a free chunk of required_bytes is produced.  Note that
  // unsafe merged chunks adopt the most conservative timestamp from their
  // constituents so they're only useful for allocations not requiring a
  // particular timestamp.
  bool MergeTimestampedChunks(size_t required_bytes)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // Return the largest binned free chunk. Free chunks are sorted by size (and
  // then address) in a bin.
  size_t LargestBinnedFreeChunk() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  size_t LargestBinnedFreeChunk(AllocationEnd allocation_end)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // Return the largest free chunk, including the central gap when spatial
  // partitioning is enabled.
  size_t LargestFreeChunk() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // Add TraceMe (in memory allocation and deallocation) for memory stats
  // profiling. The chunk_ptr is passed to get information such as address,
  // chunk size and requested_size.
  void AddTraceMe(absl::string_view traceme_name, const void* ptr)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // Overloaded AddTraceMe function with chunk information.
  void AddTraceMe(absl::string_view traceme_name, const void* chunk_ptr,
                  int64_t req_bytes, int64_t alloc_bytes)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // A ChunkHandle is an index into the chunks_ vector in BFCAllocator
  // kInvalidChunkHandle means an invalid chunk
  using ChunkHandle = size_t;
  static constexpr ChunkHandle kInvalidChunkHandle = SIZE_MAX;

  using BinNum = int;
  static constexpr int kInvalidBinNum = -1;
  // The following means that the largest bin'd chunk size is 256 << 21 = 512MB.
  static constexpr int kNumBins = 21;

  // Tag describing a chunk's ownership state. Spatial partitioning keeps three
  // contiguous spans by address:
  //
  //   [ kLower (grows up) ][ kCentralGap ][ kUpper (grows down) ]
  //
  // A request may carve from the contiguous kCentralGap span or reuse a
  // free hole with its OWN tag, but never the other end's tagged holes. This
  // keeps each end's offsets a pure function of that end's request sequence.
  // The split between lower-end, central-gap, and upper-end spans is fully
  // dynamic with no hard boundary: when a boundary chunk is freed it rejoins
  // the central gap, growing it, and adjacent same-tag holes cascade back in
  // turn. So e.g. allocating 100% kLower, freeing it, then allocating 100%
  // kUpper is supported -- the freed lower space cascades back into one
  // central gap that the upper end can then consume.
  enum class ChunkTag : uint8_t {
    kCentralGap,  // The single central gap between lower-end and upper-end
                  // ownership. Either end may carve from it.
    kLower,  // Lower-end-owned: in use, or a free hole reusable only by the
             // lower end until it rejoins the gap.
    kUpper,  // Upper-end-owned: in use, or a free hole reusable only by the
             // upper end until it rejoins the gap.
  };

  template <typename Sink>
  friend void AbslStringify(Sink& sink, ChunkTag tag);

  // The tag owned by an allocation from `allocation_end`.
  static ChunkTag ChunkTagOf(AllocationEnd allocation_end) {
    return allocation_end == AllocationEnd::kUpper ? ChunkTag::kUpper
                                                   : ChunkTag::kLower;
  }

  // A Chunk points to a piece of memory that's either entirely free or entirely
  // in use by one user memory allocation.
  //
  // An AllocationRegion's memory is split up into one or more disjoint Chunks,
  // which together cover the whole region without gaps.  Chunks participate in
  // a doubly-linked list, and the prev/next pointers point to the physically
  // adjacent chunks.
  //
  // Since a chunk cannot be partially in use, we may need to split a free chunk
  // in order to service a user allocation.  We always merge adjacent free
  // chunks.
  //
  // Chunks contain information about whether they are in use or whether they
  // are free, and contain a pointer to the bin they are in.
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
    int64_t allocation_id = -1;
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

    // Optional count when this chunk was most recently made free.
    uint64_t freed_at_count = 0;

    // Ownership state for this chunk (see ChunkTag). A chunk in the central
    // gap is kCentralGap; interior free holes keep their tag until they
    // rejoin the gap.
    ChunkTag tag = ChunkTag::kCentralGap;

    // Snapshot of the thread-local allocation annotation stack captured when
    // this chunk became in-use. Cleared when the chunk is freed.
    std::optional<ScopedAllocationTrace::Snapshot> allocation_annotation;

    bool in_use() const { return allocation_id != -1; }

#ifdef TENSORFLOW_MEM_DEBUG
    // optional debugging info
    const char* op_name = nullptr;
    uint64_t step_id = 0;
    int64_t action_count = 0;
#endif

    std::string DebugString(BFCAllocator* a, bool recurse)
        ABSL_EXCLUSIVE_LOCKS_REQUIRED(a->mutex_) {
      std::string dbg;
      absl::StrAppend(
          &dbg, "  Size: ", strings::HumanReadableNumBytes(size),
          " | Requested Size: ", strings::HumanReadableNumBytes(requested_size),
          " | in_use: ", in_use(), " | bin_num: ", bin_num);
      if (recurse && prev != BFCAllocator::kInvalidChunkHandle) {
        Chunk* p = a->ChunkFromHandle(prev);
        absl::StrAppend(&dbg, ", prev: ", p->DebugString(a, false));
      }
      if (recurse && next != BFCAllocator::kInvalidChunkHandle) {
        Chunk* n = a->ChunkFromHandle(next);
        absl::StrAppend(&dbg, ", next: ", n->DebugString(a, false));
      }
#ifdef TENSORFLOW_MEM_DEBUG
      absl::StrAppend(&dbg, ", for: ", op_name ? op_name : "UNKNOWN",
                      ", stepid: ", step_id, ", last_action: ", action_count);
#endif
      return dbg;
    }
  };

  // A Bin is a collection of similar-sized free chunks.
  // Allocated chunks are never in a Bin.
  struct Bin {
    // All chunks in this bin have >= bin_size memory.
    size_t bin_size = 0;

    class ChunkComparator {
     public:
      explicit ChunkComparator(BFCAllocator* allocator)
          : allocator_(allocator) {}
      // Sort first by size and then use pointer address as a tie breaker.
      bool operator()(const ChunkHandle ha, const ChunkHandle hb) const
          ABSL_NO_THREAD_SAFETY_ANALYSIS {
        const Chunk* a = allocator_->ChunkFromHandle(ha);
        const Chunk* b = allocator_->ChunkFromHandle(hb);
        if (a->size != b->size) {
          return a->size < b->size;
        }
        return a->ptr < b->ptr;
      }

     private:
      BFCAllocator* allocator_;  // The parent allocator.
    };

    using FreeChunkSet = absl::btree_set<ChunkHandle, ChunkComparator>;
    // List of free chunks within the bin, sorted by chunk size.
    FreeChunkSet free_chunks;
    Bin(BFCAllocator* allocator, size_t bs)
        : bin_size(bs), free_chunks(ChunkComparator(allocator)) {}
  };

  // BFCAllocator allocates memory into a collection of disjoint
  // AllocationRegions.  Each AllocationRegion corresponds to one call to
  // SubAllocator::Alloc().  (Actually, if a subsequent call to
  // SubAllocator::Alloc() returns another region immediately adjacent to the
  // last, it will be used to extend the first AllocationRegion, not create a
  // separate one.)
  //
  // An AllocationRegion contains one or more Chunks, covering all of its
  // memory.  Its primary job is to map pointers to ChunkHandles.
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
      handles_.resize(n_handles, kInvalidChunkHandle);
    }

    AllocationRegion() = default;
    AllocationRegion(AllocationRegion&& other) { Swap(&other); }
    AllocationRegion& operator=(AllocationRegion&& other) {
      Swap(&other);
      return *this;
    }

    void* ptr() const { return ptr_; }
    void* end_ptr() const { return end_ptr_; }
    size_t memory_size() const { return memory_size_; }
    void extend(size_t size) {
      memory_size_ += size;
      DCHECK_EQ(0, memory_size_ % kMinAllocationSize);

      end_ptr_ = static_cast<void*>(static_cast<char*>(end_ptr_) + size);
      const size_t n_handles =
          (memory_size_ + kMinAllocationSize - 1) / kMinAllocationSize;
      handles_.resize(n_handles, kInvalidChunkHandle);
    }
    ChunkHandle get_handle(const void* p) const {
      return handles_[IndexFor(p)];
    }
    void set_handle(const void* p, ChunkHandle h) { handles_[IndexFor(p)] = h; }
    void erase(const void* p) { set_handle(p, kInvalidChunkHandle); }

   private:
    void Swap(AllocationRegion* other) {
      std::swap(ptr_, other->ptr_);
      std::swap(memory_size_, other->memory_size_);
      std::swap(end_ptr_, other->end_ptr_);
      std::swap(handles_, other->handles_);
    }

    size_t IndexFor(const void* p) const {
      uintptr_t p_int = absl::bit_cast<uintptr_t>(p);
      uintptr_t base_int = absl::bit_cast<uintptr_t>(ptr_);
      DCHECK_GE(p_int, base_int);
      DCHECK_LT(p_int, base_int + memory_size_);
      return static_cast<size_t>(((p_int - base_int) >> kMinAllocationBits));
    }

    // Metadata about the allocation region.
    void* ptr_ = nullptr;
    size_t memory_size_ = 0;
    void* end_ptr_ = nullptr;

    // Array of size "memory_size / kMinAllocationSize".  It is
    // indexed by (p-base) / kMinAllocationSize, contains ChunkHandle
    // for the memory allocation represented by "p"
    std::vector<ChunkHandle> handles_;

    AllocationRegion(const AllocationRegion&) = delete;
    void operator=(const AllocationRegion&) = delete;
  };

  // RegionManager aggregates one or more "AllocationRegions" and provides
  // a layer of indirection from pointers to the underlying ChunkHandle,
  // allowing allocation across multiple discontiguous memory regions.
  //
  // This class is thread-compatible.
  class RegionManager {
   public:
    RegionManager() = default;
    ~RegionManager() = default;

    void AddAllocationRegion(void* ptr, size_t memory_size) {
      // Insert sorted by end_ptr.
      auto entry =
          std::upper_bound(regions_.begin(), regions_.end(), ptr, &Comparator);
      regions_.insert(entry, AllocationRegion(ptr, memory_size));
    }

    // Adds an allocation region for the given ptr and size, potentially
    // extending a region if ptr matches the end_ptr of an existing region.
    // If a region is extended, returns a pointer to the extended region so that
    // the BFC allocator can reason about chunkification.
    AllocationRegion* AddOrExtendAllocationRegion(void* ptr,
                                                  size_t memory_size) {
      // Insert sorted by end_ptr.
      auto entry =
          std::upper_bound(regions_.begin(), regions_.end(), ptr, &Comparator);
      // Check if can be coalesced with preceding region.
      if (entry != regions_.begin()) {
        auto preceding_region = entry - 1;
        if (preceding_region->end_ptr() == ptr) {
          if (VLOG_IS_ON(1)) {
            LOG(INFO) << "Extending region " << preceding_region->ptr()
                      << " of "
                      << strings::HumanReadableNumBytes(
                             preceding_region->memory_size())
                      << "  by " << strings::HumanReadableNumBytes(memory_size)
                      << " bytes";
          }
          preceding_region->extend(memory_size);
          return &*preceding_region;
        }
      }
      VLOG(1) << "Inserting new region " << ptr << " of "
              << strings::HumanReadableNumBytes(memory_size);
      regions_.insert(entry, AllocationRegion(ptr, memory_size));
      return nullptr;
    }

    std::vector<AllocationRegion>::iterator RemoveAllocationRegion(
        std::vector<AllocationRegion>::iterator it) {
      return regions_.erase(it);
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
  static size_t RoundedBytes(size_t bytes);

  // Returns the first aligned address at or above 'ptr'. Alignment must be a
  // power of two.
  static uintptr_t AlignUp(uintptr_t ptr, size_t alignment);

  // Returns the last aligned address at or below 'ptr'. Alignment must be a
  // power of two.
  static uintptr_t AlignDown(uintptr_t ptr, size_t alignment);

  // Bytes to skip at the low end of a free chunk so the allocation starts
  // aligned. The padding is rounded so it can be represented as a Chunk when
  // split from the allocation.
  static size_t LowEndAlignmentPadding(uintptr_t chunk_start, size_t alignment);

  // Start address for an allocation carved from the high end of a free chunk.
  // Returns an address below `chunk_start` if the allocation cannot fit.
  static uintptr_t HighEndAlignedStart(uintptr_t chunk_start, size_t chunk_size,
                                       size_t rounded_bytes, size_t alignment);

  // Try to add a new memory region that can satisfy an allocation of
  // 'rounded_bytes' bytes.  Returns true on success and false on
  // failure.
  bool Extend(size_t alignment, size_t rounded_bytes)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // Deallocate free regions to give back the memory to suballocator, so that
  // we can reallocate a larger region.  The main use scenario of this function
  // is when OOM happens but we have free regions and the sum of sizes of free
  // regions and unallocated bytes is larger than the requested size, implying
  // (external) memory fragmentation.  Returns true if any free regions are
  // found and freed; false otherwise.
  bool DeallocateFreeRegions(size_t rounded_bytes);

  // Helper function to deallocate regions.
  void DeallocateRegions(const absl::flat_hash_set<void*>& region_ptrs)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // Returns a pointer to an underlying allocated chunk of size
  // 'rounded_bytes' aligned to 'alignment', served from 'allocation_end'.
  void* FindChunkPtr(BinNum bin_num, size_t rounded_bytes, size_t num_bytes,
                     size_t alignment, uint64_t freed_before,
                     AllocationEnd allocation_end)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // Best-fit scan restricted to binned interior holes owned by
  // 'allocation_end'. Returns the user pointer, or nullptr if no same-tag hole
  // fits.
  void* FindTaggedChunkPtr(BinNum bin_num, size_t rounded_bytes,
                           size_t num_bytes, size_t alignment,
                           uint64_t freed_before, AllocationEnd allocation_end)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // Carves from the central gap. In spatial partitioning mode the gap is
  // tracked directly by central_gap_ instead of being inserted into bins.
  void* FindChunkPtrInCentralGap(size_t rounded_bytes, size_t num_bytes,
                                 size_t alignment, uint64_t freed_before,
                                 AllocationEnd allocation_end)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // Carves an allocation of 'num_bytes' (rounded to 'rounded_bytes') out of the
  // free chunk 'h', which must already have been removed from its free
  // structure. The low variant grows up from the chunk's low address (the
  // default); the high variant grows down from the chunk's high address. Both
  // return the user pointer.
  void* AllocateChunkFromLowEnd(ChunkHandle h, size_t rounded_bytes,
                                size_t num_bytes, size_t alignment)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  void* AllocateChunkFromHighEnd(ChunkHandle h, size_t rounded_bytes,
                                 size_t num_bytes, size_t alignment)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // Marks 'chunk' in use and updates allocation stats. Common tail of the two
  // AllocateChunkFrom*End helpers.
  void FinishChunkAllocation(Chunk* chunk, size_t num_bytes)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // Splits the chunk specified by 'h' into two chunks, one at least
  // of size 'num_bytes'.
  void SplitChunk(ChunkHandle h, size_t num_bytes)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // Tag of the free chunk formed by merging two adjacent free neighbors:
  // the common tag if both holes have the same tag (an interior hole keeps
  // its end), otherwise kCentralGap -- so a hole merging with the central gap,
  // or lower and upper holes becoming adjacent, yields space reusable by either
  // end.
  ChunkTag MergedChunkTag(ChunkTag a, ChunkTag b) const;

  // Merges the two chunk handles.  Requires that the chunks are
  // contiguous in their allocation.
  void MergeChunks(ChunkHandle h, ChunkHandle h2)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // Adds the chunk 'h' to the free data structure. Spatial partitioning
  // keeps the single central gap out of the bins and bins only lower/upper
  // interior holes; classic BFC inserts every free chunk into a size bin.
  void InsertFreeChunk(ChunkHandle h) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // Removes the chunk 'h' from the free data structure.
  void RemoveFreeChunk(ChunkHandle h) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // Reclassifies a just-freed lower/upper boundary chunk as kCentralGap when it
  // is no longer interior to its tag.
  void ReturnBoundaryChunkToGap(ChunkHandle h)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // Removes the free chunk pointed to by 'c' from the set free_chunks.
  void RemoveFreeChunkIterFromBin(Bin::FreeChunkSet* free_chunks,
                                  const Bin::FreeChunkSet::iterator& c)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // Removes a free chunk from the bin.
  void RemoveFreeChunkFromBin(ChunkHandle h)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  void MaybeRemoveFreeChunkFromBin(ChunkHandle h)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // Removes the chunk metadata represented by 'h'.
  void DeleteChunk(ChunkHandle h) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  std::string RenderOccupancy() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  void DumpMemoryLog(size_t num_bytes, AllocationEnd allocation_end)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  tensorflow::MemoryDump RecordMemoryMapInternal()
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  void MaybeWriteMemoryMap() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  ChunkHandle AllocateChunk() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  void DeallocateChunk(ChunkHandle h) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  Chunk* ChunkFromHandle(ChunkHandle h) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  const Chunk* ChunkFromHandle(ChunkHandle h) const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  void MarkFree(ChunkHandle h) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  ChunkHandle TryToCoalesce(ChunkHandle h, bool ignore_freed_at)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // Fragmentation is calculated as the reverse ratio of the largest free chunk
  // size over total free memory, and returns a value within [0, 1].
  double GetFragmentation() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

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
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  AllocatorRetry retry_helper_;

  // Structures immutable after construction
  size_t memory_limit_ = 0;

  // Maximum bytes a chunk may exceed the requested size before it is split, to
  // bound internal fragmentation. Derived from Options::fragmentation_fraction
  // and memory_limit_ once at construction.
  int64_t max_internal_fragmentation_bytes_ = 0;

  // Map from bin size to Bin
  Bin* BinFromIndex(BinNum index) {
    return reinterpret_cast<Bin*>(&(bins_space_[index * sizeof(Bin)]));
  }
  size_t BinNumToSize(BinNum index) {
    return static_cast<size_t>(256) << index;
  }
  BinNum BinNumForSize(size_t bytes) {
    uint64_t v = std::max<size_t>(bytes, 256) >> kMinAllocationBits;
    int b = std::min(kNumBins - 1, tsl::Log2Floor64(v));
    return b;
  }
  Bin* BinForSize(size_t bytes) { return BinFromIndex(BinNumForSize(bytes)); }

  char bins_space_[sizeof(Bin) * kNumBins];

  const Options opts_;

  // Tag assigned to newly-created free chunks. Classic BFC keeps ordinary
  // free chunks in kLower; spatial partitioning starts each fixed region as
  // the kCentralGap span.
  const ChunkTag free_chunk_tag_;

  // The size of the current region allocation.
  size_t curr_region_allocation_bytes_;

  // Whether the allocator will coalesce adjacent sub allocator provided
  // AllocationRegions. This may be disabled if discrete sub allocator
  // regions can't be treated as contiguous (e.g. if the allocation refers to
  // device visible memory which is not adjacent to the other region in the
  // device's address space).
  const bool coalesce_regions_;

  std::unique_ptr<SubAllocator> sub_allocator_;
  std::string name_;
  SharedCounter* timing_counter_ = nullptr;
  std::deque<ChunkHandle> timestamped_chunks_;

  std::atomic<uint64_t> safe_frontier_ = {0};

  // Structures mutable after construction
  mutable absl::Mutex mutex_;
  RegionManager region_manager_ ABSL_GUARDED_BY(mutex_);

  std::vector<Chunk> chunks_ ABSL_GUARDED_BY(mutex_);

  // Head of a singly-linked list of unused Chunk metadata slots in chunks_.
  // The list reuses Chunk::next while the slot is inactive.
  ChunkHandle unused_chunk_handle_head_ ABSL_GUARDED_BY(mutex_);

  // The single central gap in spatial partitioning mode. It is not present in
  // any Bin; lower/upper interior free holes remain binned.
  ChunkHandle central_gap_ ABSL_GUARDED_BY(mutex_) = kInvalidChunkHandle;

  // Counter containing the next unique identifier to assign to a
  // newly-created chunk.
  int64_t next_allocation_id_ ABSL_GUARDED_BY(mutex_);

  // Stats.
  AllocatorStats stats_ ABSL_GUARDED_BY(mutex_);

#ifdef TENSORFLOW_MEM_DEBUG
  int64_t action_counter_ ABSL_GUARDED_BY(mutex_);
#define MEM_DEBUG_SIZE_HISTORY_SIZE 4096
  int64_t size_history_[MEM_DEBUG_SIZE_HISTORY_SIZE];
#endif

  friend class GPUBFCAllocatorPrivateMethodsTest;
  friend class GPUBFCAllocatorPrivateMethodsTest_SubAllocatorSpecific;
  BFCAllocator(const BFCAllocator&) = delete;
  void operator=(const BFCAllocator&) = delete;
};

//===----------------------------------------------------------------------===//
// Stringification of enums.
//===----------------------------------------------------------------------===//

template <typename Sink>
void AbslStringify(Sink& sink, BFCAllocator::ChunkTag tag) {
  switch (tag) {
    case BFCAllocator::ChunkTag::kCentralGap:
      sink.Append("central_gap");
      return;
    case BFCAllocator::ChunkTag::kLower:
      sink.Append("lower");
      return;
    case BFCAllocator::ChunkTag::kUpper:
      sink.Append("upper");
      return;
  }
}

}  // namespace tsl

#endif  // XLA_TSL_FRAMEWORK_BFC_ALLOCATOR_H_
