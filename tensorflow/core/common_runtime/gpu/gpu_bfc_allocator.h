#ifndef TENSORFLOW_COMMON_RUNTIME_GPU_GPU_BFC_ALLOCATOR_H_
#define TENSORFLOW_COMMON_RUNTIME_GPU_GPU_BFC_ALLOCATOR_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/stream_executor/stream_executor.h"
#include "tensorflow/core/common_runtime/gpu/gpu_allocator_retry.h"
#include "tensorflow/core/common_runtime/gpu/visitable_allocator.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/platform/thread_annotations.h"

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
  void DeallocateRaw(void* ptr) override;

  void AddAllocVisitor(Visitor visitor) override;

  // Does nothing, because gpu memory is never freed.
  void AddFreeVisitor(Visitor visitor) override {}

  bool TracksAllocationSizes() override;

  size_t RequestedSize(void* ptr) override;

  size_t AllocatedSize(void* ptr) override;

 private:
  struct Bin;

  void* AllocateRawInternal(size_t alignment, size_t num_bytes,
                            bool dump_log_on_failure);
  void DeallocateRawInternal(void* ptr);

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

    bool in_use = false;
    void* ptr = nullptr;  // pointer to granted GPU subbuffer.

    // If not null, the memory referred to by 'prev' is directly
    // preceding the memory used by this chunk.  E.g., It should start
    // at 'ptr - prev->size'
    Chunk* prev = nullptr;

    // If not null, the memory referred to by 'next' is directly
    // following the memory used by this chunk.  E.g., It should be at
    // 'ptr + size'
    Chunk* next = nullptr;

    // What bin are we in?
    Bin* bin = nullptr;

    string DebugString(bool recurse) {
      string dbg;
      strings::StrAppend(&dbg, "  Size: ", strings::HumanReadableNumBytes(size),
                         " | Requested Size: ",
                         strings::HumanReadableNumBytes(requested_size),
                         " | in_use: ", in_use);
      if (recurse && prev) {
        strings::StrAppend(&dbg, ", prev: ", prev->DebugString(false));
      }
      if (recurse && next) {
        strings::StrAppend(&dbg, ", next: ", next->DebugString(false));
      }
      return dbg;
    }
  };

  Chunk* AllocateNewChunk(size_t num_bytes);
  void SplitChunk(Chunk* c, size_t num_bytes);
  void Merge(Chunk* c1, Chunk* c2);
  void FreeAndMaybeCoalesce(Chunk* c);
  void InsertFreeChunkIntoBin(Chunk* c);
  void RemoveFreeChunkFromBin(Chunk* c);
  void DeleteChunk(Chunk* c);

  void DumpMemoryLog(size_t num_bytes);

  // A Bin is a collection of similar-sized free chunks.
  struct Bin {
    // All chunks in this bin have >= bin_size memory.
    size_t bin_size = 0;

    struct ChunkComparator {
      // Sort first by size and then use pointer address as a tie breaker.
      bool operator()(const Chunk* a, const Chunk* b) const {
        if (a->size != b->size) {
          return a->size < b->size;
        }
        return a->ptr < b->ptr;
      }
    };

    // List of free chunks within the bin, sorted by chunk size.
    // Chunk * not owned.
    std::set<Chunk*, ChunkComparator> free_chunks;

    explicit Bin(size_t bs) : bin_size(bs) {}
  };

  GPUAllocatorRetry retry_helper_;

  // Structures immutable after construction
  const int device_id_;
  // The base pointer where all the GPU memory begins.
  void* base_ptr_ = nullptr;
  size_t gpu_memory_size_ = 0;

  // Map from bin size to Bin
  // After construction, the bin map is never resized.
  std::map<size_t, Bin*> bins_;

  perftools::gputools::StreamExecutor* stream_exec_;  // Not owned.

  // Structures mutable after construction
  mutable mutex lock_;
  // Chunk * owned.
  std::unordered_map<void*, Chunk*> ptr_to_chunk_map_;

  // Called once on each region, ASAP.
  std::vector<Visitor> region_visitors_;

  TF_DISALLOW_COPY_AND_ASSIGN(GPUBFCAllocator);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMMON_RUNTIME_GPU_GPU_BFC_ALLOCATOR_H_
