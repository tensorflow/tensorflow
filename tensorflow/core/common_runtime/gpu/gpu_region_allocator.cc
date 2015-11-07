#include "tensorflow/core/common_runtime/gpu/gpu_region_allocator.h"

//#include "base/commandlineflags.h"
#include "tensorflow/stream_executor/multi_platform_manager.h"
#include "tensorflow/core/common_runtime/gpu/gpu_allocator_retry.h"
#include "tensorflow/core/common_runtime/gpu/gpu_init.h"
#include "tensorflow/core/lib/core/bits.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/port.h"

#if defined(PLATFORM_GOOGLE)
DEFINE_bool(brain_gpu_region_allocator_heap_check_on_destruction, true,
            "If true, the CUDA gpu manager checks that all allocated "
            "memory through the GPU memory pool implementation has been "
            "freed.");

DEFINE_int64(brain_gpu_region_allocator_region_size, 0,
             "If > 0, sets the default chunk-size allocatable from GPU memory. "
             "Else defaults to entire GPU memory.");

#else
bool FLAGS_brain_gpu_region_allocator_heap_check_on_destruction = true;
tensorflow::int64 FLAGS_brain_gpu_region_allocator_region_size = 0;
#endif

namespace gpu = ::perftools::gputools;

namespace tensorflow {

GPURegionAllocator::GPURegionAllocator(int device_id, size_t total_bytes)
    : device_id_(device_id), total_bytes_(total_bytes) {
  // Get a pointer to the stream_executor for this device
  stream_exec_ = GPUMachineManager()->ExecutorForDevice(device_id).ValueOrDie();

  // Set the region size based on explicit user request, or based on
  // total GPU capacity.
  if (FLAGS_brain_gpu_region_allocator_region_size > 0) {
    region_size_ = FLAGS_brain_gpu_region_allocator_region_size;
  } else {
    region_size_ = static_cast<size_t>(total_bytes_);
  }

  LOG(INFO) << "Setting region size to " << region_size_;
}

GPURegionAllocator::~GPURegionAllocator() {
  if (FLAGS_brain_gpu_region_allocator_heap_check_on_destruction) {
    CheckForMemoryLeaks();
  }

  gtl::STLDeleteValues(&chunk_map_);

  for (auto r : regions_) {
    gpu::DeviceMemoryBase gpu_ptr{r->ptr};
    stream_exec_->Deallocate(&gpu_ptr);
    delete r;
  }
}

void* GPURegionAllocator::AllocateRaw(size_t alignment, size_t num_bytes) {
  static const int64 kMaxMillisToWait = 10000;  // 10 seconds
  return retry_helper_.AllocateRaw(
      [this](size_t a, size_t nb, bool v) {
        return AllocateRawInternal(a, nb, v);
      },
      kMaxMillisToWait, alignment, num_bytes);
}

void* GPURegionAllocator::AllocateRawInternal(size_t alignment,
                                              size_t num_bytes,
                                              bool dump_log_on_failure) {
  if (num_bytes == 0) {
    LOG(ERROR) << "tried to allocate 0 bytes";
    return nullptr;
  }
  size_t chunk_size = ChunkSize(num_bytes);

  VLOG(2) << "chunk_size " << chunk_size << " from num_bytes "
          << strings::HumanReadableNumBytes(num_bytes);
  mutex_lock l(lock_);
  Pool* pool = &pools_[chunk_size];
  if (pool->num_free == 0) {
    if (!ExpandPool(pool, chunk_size, num_bytes, dump_log_on_failure)) {
      if (dump_log_on_failure) {
        LOG(WARNING) << "Out of GPU memory, see memory state dump above";
      }
      return nullptr;
    }
  }
  CHECK_LT(0, pool->num_free);
  CHECK(pool->first);
  CHECK(pool->last);
  Chunk* c = pool->first;
  CHECK(c);
  CHECK(!c->in_use);

  c->in_use = true;
  // Move c to the back of the queue.
  if (c->next != nullptr) {
    pool->first = c->next;
    pool->first->prev = nullptr;
    c->next = nullptr;
  }

  if (pool->last != c) {
    pool->last->next = c;
    c->prev = pool->last;
    pool->last = c;
  }
  pool->num_free--;
  pool->cumulative_malloced++;

  void* rv = c->ptr;
  c->bytes_allocated = num_bytes;

  VLOG(2) << "new ptr " << rv;
  return rv;
}

void GPURegionAllocator::DeallocateRaw(void* ptr) {
  retry_helper_.DeallocateRaw([this](void* p) { DeallocateRawInternal(p); },
                              ptr);
}

void GPURegionAllocator::DeallocateRawInternal(void* ptr) {
  VLOG(2) << "DeallocateRaw: " << ptr;
  if (ptr == nullptr) {
    LOG(ERROR) << "tried to deallocate nullptr";
    return;
  }

  mutex_lock l(lock_);
  ChunkMap::const_iterator iter = chunk_map_.find(ptr);
  CHECK(iter != chunk_map_.end());

  Chunk* c = iter->second;
  VLOG(2) << "chunk of size " << c->size << " at " << c;

  Pool* pool = &(pools_[c->size]);
  // Move chunk to head of queue, and mark free.
  DCHECK(c->in_use);
  c->in_use = false;
  if (c->prev) c->prev->next = c->next;
  if (c->next) c->next->prev = c->prev;
  if (pool->first == c) pool->first = c->next;
  if (pool->last == c) pool->last = c->prev;
  c->next = pool->first;
  c->prev = nullptr;
  if (c->next) c->next->prev = c;
  pool->first = c;
  if (pool->last == nullptr) pool->last = c;
  pool->num_free++;
  pool->cumulative_freed++;
}

bool GPURegionAllocator::ExpandPool(Pool* pool, size_t chunk_size,
                                    size_t requested_size,
                                    bool dump_log_on_failure) {
  VLOG(1) << "ExpandPool of " << chunk_size << " from " << pool->num_chunks
          << " current members";
  DCHECK_NE(0, chunk_size);
  // If chunk_size is < 4096, double the pool size.  Otherwise
  // just increase by one.
  int num_chunks = pool->num_chunks;
  if (num_chunks == 0) {
    if (chunk_size > 4096) {
      num_chunks = 1;
    } else {
      num_chunks = 4096 / chunk_size;
    }
  }
  // For larger chunks, limit the amount of expansion.
  size_t aggregate_size = num_chunks * chunk_size;
  if (aggregate_size > (1 << 20)) {
    num_chunks = static_cast<int>(
        std::max(static_cast<size_t>(1), (1 << 20) / chunk_size));
  }
  while (num_chunks > 0) {
    Region* r = (regions_.empty() ? nullptr : regions_.back());
    if (r == nullptr ||
        (((r->ptr + r->size) - r->next) < static_cast<int64>(chunk_size))) {
      // Current region is not large enough to accommodate another chunk.
      while (r == nullptr || (((r->ptr + r->size) - r->next) <
                              static_cast<int64>(chunk_size))) {
        // Get another region.
        size_t this_region_size = std::max(region_size_, chunk_size);

        // Check if we would exceed our limit.
        if (allocated_memory_ + this_region_size > total_bytes_) {
          if (dump_log_on_failure) DumpMemoryLog();
          return false;
        }

        // Perform the allocation, still checking that the allocator
        // has not run out of memory.
        gpu::DeviceMemory<char> gpu_mem =
            stream_exec_->AllocateArray<char>(this_region_size);
        if (gpu_mem == nullptr) {
          if (dump_log_on_failure) DumpMemoryLog();
          return false;
        }

        // We never release memory once expanded.
        allocated_memory_ += this_region_size;

        Region* nr = new Region;
        nr->ptr = static_cast<char*>(gpu_mem.opaque());

        if (VLOG_IS_ON(2)) {
          int64 free_bytes;
          int64 total_bytes;
          if (stream_exec_->DeviceMemoryUsage(&free_bytes, &total_bytes)) {
            VLOG(2) << "free " << free_bytes << " total " << total_bytes;
          } else {
            // Note: stream_exec call also logs internally on failure.
            VLOG(2) << "could not retrieve memory usage";
          }
        }
        VLOG(1) << "new Region of size " << this_region_size << " at "
                << static_cast<void*>(nr->ptr) << " on device " << device_id_;
        r = nr;
        r->size = this_region_size;
        r->next = r->ptr;
        regions_.push_back(r);

        for (auto visitor : region_visitors_) {
          visitor(r->ptr, r->size);
        }
      }
    } else {
      // Allocate a new chunk and push on front of Pool.
      Chunk* c = new Chunk;
      c->ptr = r->next;
      chunk_map_[c->ptr] = c;
      c->size = chunk_size;
      r->next += chunk_size;
      c->next = pool->first;
      if (c->next != nullptr) c->next->prev = c;
      pool->first = c;
      if (pool->last == nullptr) pool->last = c;
      pool->num_chunks++;
      pool->num_free++;
      --num_chunks;
    }
  }

  return true;
}

void GPURegionAllocator::CheckForMemoryLeaks() {
  std::vector<string> errors;
  mutex_lock l(lock_);  // could use reader lock
  for (auto pool_map : pools_) {
    const Pool& p = pool_map.second;
    Chunk* curr_chunk = p.first;
    while (curr_chunk != nullptr) {
      if (curr_chunk->in_use) {
        errors.push_back(
            strings::StrCat("Unfreed chunk of size ", curr_chunk->size));
      }
      curr_chunk = curr_chunk->next;
    }
  }
  if (!errors.empty()) {
    LOG(FATAL) << "GPU Memory leaks:\n" << str_util::Join(errors, "\n");
  }
}

// Since there's no merging of chunks once allocated, we want to
// maximize their reusablity (which argues for fewer, larger sizes),
// while minimizing waste (which argues for tight-fitting sizes).
//
// The smallest unit of allocation is 256 bytes.
// NOTE(tucker): akrizhevsky says that nvidia's memory manager always
// aligns to 256 bytes, and doing so results in significant speedup.
//
// Up to 2^16 bytes we only allocate in powers of 2.
//
// Above that, we pick a max-waste which is the largest power
// of 2 <= 1/16 of the requested size, then round up to the nearest
// multiple of max_waste.
//
// static
size_t GPURegionAllocator::ChunkSize(size_t bytes) {
  if (bytes <= 256) {
    return 256;
  } else if (bytes <= (1 << 16)) {
    return 1uLL << Log2Ceiling64(bytes);
  } else {
    // 1/16th of requested size
    size_t max_waste = 1uLL << (Log2Ceiling64(bytes) - 4);
    return (bytes + max_waste) & (~(max_waste - 1));
  }
}

void GPURegionAllocator::AddAllocVisitor(Visitor visitor) {
  VLOG(1) << "AddVisitor";
  mutex_lock l(lock_);
  region_visitors_.push_back(visitor);
  for (auto region : regions_) {
    visitor(region->ptr, region->size);
  }
}

void GPURegionAllocator::DumpMemoryLog() {
  size_t region_bytes = 0;
  for (auto r : regions_) {
    region_bytes += r->size;
  }
  size_t chunk_bytes = 0;
  std::vector<size_t> chunk_sizes;
  for (auto i : pools_) {
    chunk_sizes.push_back(i.first);
  }
  std::sort(chunk_sizes.begin(), chunk_sizes.end());
  for (auto i : chunk_sizes) {
    int32 chunks_in_use = 0;
    const Pool& p = pools_[i];
    chunk_bytes += i * p.num_chunks;

    if (p.num_chunks > 0) {
      // Iterate backwards (allocated chunks are last).
      Chunk* curr_chunk = p.last;
      while (curr_chunk != nullptr) {
        if (curr_chunk->in_use) {
          ++chunks_in_use;
        }
        curr_chunk = curr_chunk->prev;
        if (curr_chunk == p.first) {
          break;
        }
      }
    }

    LOG(INFO) << "Chunk size: " << i << " ("
              << strings::HumanReadableNumBytes(i) << ") Pool: " << p.ToString()
              << "\nNumber of chunks: " << p.num_chunks
              << ", in_use chunks: " << chunks_in_use;
  }

  LOG(INFO) << "Aggregate Region Memory: " << region_bytes << " ("
            << strings::HumanReadableNumBytes(region_bytes) << ")";
  LOG(INFO) << "Aggregate Chunk Memory: " << chunk_bytes << " ("
            << strings::HumanReadableNumBytes(chunk_bytes) << ")";
}

bool GPURegionAllocator::TracksAllocationSizes() { return true; }

size_t GPURegionAllocator::RequestedSize(void* ptr) {
  mutex_lock l(lock_);
  auto it = chunk_map_.find(ptr);
  CHECK(it != chunk_map_.end())
      << "Asked for requested size of pointer we never allocated: " << ptr;
  auto c = it->second;
  return c->bytes_allocated;
}

size_t GPURegionAllocator::AllocatedSize(void* ptr) {
  mutex_lock l(lock_);
  auto it = chunk_map_.find(ptr);
  CHECK(it != chunk_map_.end())
      << "Asked for allocated size of pointer we never allocated: " << ptr;
  auto c = it->second;
  return c->size;
}

}  // namespace tensorflow
