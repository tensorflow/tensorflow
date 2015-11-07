#ifndef TENSORFLOW_COMMON_RUNTIME_GPU_GPU_REGION_ALLOCATOR_H_
#define TENSORFLOW_COMMON_RUNTIME_GPU_GPU_REGION_ALLOCATOR_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/stream_executor/stream_executor.h"
#include "tensorflow/core/common_runtime/gpu/gpu_allocator_retry.h"
#include "tensorflow/core/common_runtime/gpu/visitable_allocator.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/platform/thread_annotations.h"

namespace tensorflow {

class GPURegionAllocator : public VisitableAllocator {
 public:
  // 'device_id' must be a valid device on the machine.
  //
  // total_bytes is how many bytes this allocator should allocate up
  // to.  This may be less than the total available.
  explicit GPURegionAllocator(int device_id, size_t total_bytes);
  ~GPURegionAllocator() override;

  string Name() override { return "gpu_region"; }
  void* AllocateRaw(size_t alignment, size_t num_bytes) override;
  void DeallocateRaw(void* ptr) override;
  void AddAllocVisitor(Visitor visitor) override;
  // Does nothing, because regions are never freed.
  void AddFreeVisitor(Visitor visitor) override {}

  bool TracksAllocationSizes() override;
  size_t RequestedSize(void* ptr) override;
  size_t AllocatedSize(void* ptr) override;

 private:
  // A Chunk is the header on a single piece of memory given back
  // in response to an AllocateRaw() call.
  struct Chunk {
    char* ptr;               // pointer to granted GPU buffer.
    size_t size;             // Full size of GPU buffer.
    size_t bytes_allocated;  // Bytes asked for by client.
    bool in_use;
    Chunk* prev;  // Used for chaining in pool.
    Chunk* next;
    Chunk()
        : ptr(nullptr),
          size(0),
          bytes_allocated(0),
          in_use(false),
          prev(nullptr),
          next(nullptr) {}
  };

  // A Pool is a collection of same-sized Chunks.
  struct Pool {
    int num_chunks;             // total chunks in this pool
    int num_free;               // total free chunks in this pool
    int64 cumulative_malloced;  // number of chunks malloced so far
    int64 cumulative_freed;     // number of chunks freed so far

    // double-linked ring of chunks; all free chunks precede all
    // granted chunks
    Chunk* first;
    Chunk* last;
    Pool()
        : num_chunks(0),
          num_free(0),
          cumulative_malloced(0),
          cumulative_freed(0),
          first(nullptr),
          last(nullptr) {}

    string ToString() const {
      return strings::StrCat("chunks: ", num_chunks, " free: ", num_free,
                             " cumulative malloc: ", cumulative_malloced,
                             " cumulative freed: ", cumulative_freed);
    }
  };

  // A Region is a single area of GPU memory that has been
  // reserved by this class and carved up into Chunks.
  struct Region {
    char* ptr;   // base GPU ptr
    char* next;  // frontier of unused part of region
    size_t size;
    Region() : ptr(nullptr), size(0) {}
  };

  // Calculate size of chunk for an allocation of this size.
  // Min chunk size is 16, for alignment.
  // For larger sizes, we round up somewhat so there are fewer
  // size-specific pools.
  static size_t ChunkSize(size_t bytes);

  void* AllocateRawInternal(size_t alignment, size_t num_bytes,
                            bool dump_log_on_failure);
  void DeallocateRawInternal(void* ptr);

  bool ExpandPool(Pool* p, size_t chunk_size, size_t requested_size,
                  bool dump_log_on_failure) EXCLUSIVE_LOCKS_REQUIRED(lock_);

  // Inspects region maps and crashes with debug information if there
  // are any memory leaks as detected by the region allocator.
  void CheckForMemoryLeaks() LOCKS_EXCLUDED(lock_);

  void DumpMemoryLog() EXCLUSIVE_LOCKS_REQUIRED(lock_);

  perftools::gputools::StreamExecutor* stream_exec_;  // Not owned.

  typedef std::unordered_map<size_t, Pool> PoolMap;
  typedef std::unordered_map<void*, Chunk*> ChunkMap;

  GPUAllocatorRetry retry_helper_;
  mutable mutex lock_;
  PoolMap pools_ GUARDED_BY(lock_);

  // Owns regions.
  std::vector<Region*> regions_ GUARDED_BY(lock_);

  // Maps from GPU ptr to Chunk owning it.
  //
  // Owns chunks.
  ChunkMap chunk_map_ GUARDED_BY(lock_);

  // Called once on each region, ASAP.
  std::vector<Visitor> region_visitors_ GUARDED_BY(lock_);

  const int device_id_;

  // Total amount of memory (in bytes) available to this Allocator
  const size_t total_bytes_;

  // Total amount of memory allocated to regions.
  size_t allocated_memory_ = 0;

  size_t region_size_ = 0;

  TF_DISALLOW_COPY_AND_ASSIGN(GPURegionAllocator);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMMON_RUNTIME_GPU_GPU_REGION_ALLOCATOR_H_
