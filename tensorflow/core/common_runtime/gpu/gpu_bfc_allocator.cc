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

#include "tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.h"

#include "tensorflow/core/common_runtime/gpu/gpu_allocator_retry.h"
#include "tensorflow/core/common_runtime/gpu/gpu_init.h"
#include "tensorflow/core/lib/core/bits.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/types.h"

namespace gpu = ::perftools::gputools;

namespace tensorflow {

GPUBFCAllocator::GPUBFCAllocator(int device_id, size_t total_memory)
    : device_id_(device_id),
      ptr_to_chunk_map_(total_memory, 256),
      free_chunks_list_(kInvalidChunkHandle),
      next_allocation_id_(1) {
  // Get a pointer to the stream_executor for this device
  stream_exec_ = GPUMachineManager()->ExecutorForDevice(device_id).ValueOrDie();

  // Allocate the requested amount of memory.
  gpu_memory_size_ = total_memory;

  // Create a bunch of bins of various good sizes.

  // We create bins to fit all possible ranges that cover the
  // gpu_memory_size_ starting from allocations up to 256 bytes to
  // allocations up to (and including) the memory limit.
  for (BinNum b = 0; b < kNumBins; b++) {
    size_t bin_size = BinNumToSize(b);
    LOG(INFO) << "Creating bin of max chunk size "
              << strings::HumanReadableNumBytes(bin_size);
    new (BinFromIndex(b)) Bin(this, bin_size);
    CHECK_EQ(BinForSize(bin_size), BinFromIndex(b));
    CHECK_EQ(BinForSize(bin_size + 255), BinFromIndex(b));
    CHECK_EQ(BinForSize(bin_size * 2 - 1), BinFromIndex(b));
    if (b + 1 < kNumBins) {
      CHECK_NE(BinForSize(bin_size * 2), BinFromIndex(b));
    }
  }
}

GPUBFCAllocator::~GPUBFCAllocator() {
  // Return memory back.
  if (base_ptr_) {
    gpu::DeviceMemoryBase gpu_ptr{base_ptr_};
    stream_exec_->Deallocate(&gpu_ptr);
  }

  for (BinNum b = 0; b < kNumBins; b++) {
    BinFromIndex(b)->~Bin();
  }
}

GPUBFCAllocator::Chunk* GPUBFCAllocator::ChunkFromHandle(ChunkHandle h) {
  DCHECK_GE(h, 0);
  DCHECK_LT(h, static_cast<int>(chunks_.size()));
  return &(chunks_[h]);
}

GPUBFCAllocator::ChunkHandle GPUBFCAllocator::AllocateChunk() {
  if (free_chunks_list_ != kInvalidChunkHandle) {
    ChunkHandle h = free_chunks_list_;
    Chunk* c = ChunkFromHandle(h);
    free_chunks_list_ = c->next;
    return h;
  } else {
    ChunkHandle h = chunks_.size();
    chunks_.resize(h + 1);
    return h;
  }
}

void GPUBFCAllocator::DeallocateChunk(ChunkHandle h) {
  Chunk* c = ChunkFromHandle(h);
  c->next = free_chunks_list_;
  free_chunks_list_ = h;
}

void GPUBFCAllocator::MaybeInitialize() {
  if (base_ptr_ != nullptr) {
    return;
  }

  LOG(INFO) << "Allocating " << strings::HumanReadableNumBytes(gpu_memory_size_)
            << " bytes.";
  gpu::DeviceMemory<char> gpu_mem =
      stream_exec_->AllocateArray<char>(gpu_memory_size_);

  QCHECK(gpu_mem != nullptr)
      << " Could not allocate GPU device memory for device " << device_id_
      << ". Tried to allocate "
      << strings::HumanReadableNumBytes(gpu_memory_size_);
  base_ptr_ = gpu_mem.opaque();
  ptr_to_chunk_map_.set_base_ptr(base_ptr_);
  LOG(INFO) << "GPU " << device_id_ << " memory begins at " << base_ptr_
            << " extends to "
            << static_cast<void*>(
                   (static_cast<char*>(base_ptr_) + gpu_memory_size_));

  // Create one large chunk for the whole memory space that will
  // be chunked later.
  ChunkHandle h = AllocateChunk();
  GPUBFCAllocator::Chunk* c = ChunkFromHandle(h);
  c->ptr = gpu_mem.opaque();
  c->size = gpu_memory_size_;
  c->allocation_id = -1;
  c->prev = kInvalidChunkHandle;
  c->next = kInvalidChunkHandle;

  ptr_to_chunk_map_.set_handle(c->ptr, h);

  // Insert the chunk into the right bin.
  InsertFreeChunkIntoBin(h);

  // Invoke visitors on newly allocated region.
  for (auto visitor : region_visitors_) {
    visitor(base_ptr_, gpu_memory_size_);
  }
}

void* GPUBFCAllocator::AllocateRaw(size_t unused_alignment, size_t num_bytes) {
  // Fast path: Try once to allocate without getting the retry_helper_ involved
  void* r = AllocateRawInternal(unused_alignment, num_bytes, false);
  if (r != nullptr) {
    return r;
  } else {
    static const int64 kMaxMillisToWait = 10000;  // 10 seconds
    return retry_helper_.AllocateRaw(
        [this](size_t a, size_t nb, bool v) {
          return AllocateRawInternal(a, nb, v);
        },
        kMaxMillisToWait, unused_alignment, num_bytes);
  }
}

void* GPUBFCAllocator::AllocateRaw(
    size_t unused_alignment, size_t num_bytes,
    const AllocationAttributes& allocation_attr) {
  if (allocation_attr.no_retry_on_failure) {
    // Return immediately upon the first failure if this is for allocating an
    // optional scratch space.
    void* result = AllocateRawInternal(unused_alignment, num_bytes, false);
    if (result == nullptr) {
      // The counter incrementing is not thread-safe. But we don't really care.
      // TODO(zhengxq): we should implement a LOG_FIRST_N and LOG_EVERY_N for
      // more general usage.
      static int log_counter = 0;
      if (log_counter < 10) {
        log_counter++;
        LOG(WARNING)
            << "Ran out of memory trying to allocate "
            << strings::HumanReadableNumBytes(num_bytes)
            << ". The caller indicates that this is not a failure, but"
            << " may mean that there could be performance gains if more"
            << " memory is available.";
      }
    }
    return result;
  } else {
    return AllocateRaw(unused_alignment, num_bytes);
  }
}

void* GPUBFCAllocator::AllocateRawInternal(size_t unused_alignment,
                                           size_t num_bytes,
                                           bool dump_log_on_failure) {
  if (num_bytes == 0) {
    LOG(ERROR) << "tried to allocate 0 bytes";
    return nullptr;
  }
  // First, always allocate memory of at least 256 bytes, and always
  // allocate multiples of 256 bytes so all memory addresses are
  // nicely byte aligned.
  size_t rounded_bytes = (256 * ((num_bytes + 255) / 256));
  DCHECK_EQ(size_t{0}, rounded_bytes % 256);

  // The BFC allocator tries to find the best fit first.
  //
  // First identify the first bin that could satisfy rounded_bytes.
  BinNum bin_num = BinNumForSize(rounded_bytes);

  mutex_lock l(lock_);
  MaybeInitialize();

  for (; bin_num < kNumBins; bin_num++) {
    // Start searching from the first bin for the smallest chunk that fits
    // rounded_bytes.
    Bin* b = BinFromIndex(bin_num);
    for (auto citer = b->free_chunks.begin(); citer != b->free_chunks.end();
         ++citer) {
      const GPUBFCAllocator::ChunkHandle h = (*citer);
      GPUBFCAllocator::Chunk* chunk = ChunkFromHandle(h);
      DCHECK(!chunk->in_use());
      if (chunk->size >= rounded_bytes) {
        // We found an existing chunk that fits us that wasn't in use, so remove
        // it from the free bin structure prior to using.
        RemoveFreeChunkIterFromBin(&b->free_chunks, citer);

        // If we can break the size of the chunk into two reasonably
        // large pieces, do so.
        //
        // TODO(vrv): What should be the criteria when deciding when
        // to split?
        if (chunk->size >= rounded_bytes * 2) {
          SplitChunk(h, rounded_bytes);
          chunk = ChunkFromHandle(h);  // Update chunk pointer in case it moved
        }

        // The requested size of the returned chunk is what the user
        // has allocated.
        chunk->requested_size = num_bytes;
        // Assign a unique id and increment the id counter, marking the
        // chunk as being in use.
        chunk->allocation_id = next_allocation_id_++;

        // Update stats.
        ++stats_.num_allocs;
        stats_.bytes_in_use += chunk->size;
        stats_.max_bytes_in_use =
            std::max(stats_.max_bytes_in_use, stats_.bytes_in_use);
        stats_.max_alloc_size =
            std::max<std::size_t>(stats_.max_alloc_size, chunk->size);

        VLOG(4) << "Returning: " << chunk->ptr;
        return chunk->ptr;
      }
    }
  }

  // We searched all bins for an existing free chunk to use and
  // couldn't find one.  This means we must have run out of memory,
  // Dump the memory log for analysis.
  if (dump_log_on_failure) {
    DumpMemoryLog(rounded_bytes);
    LOG(WARNING) << "Ran out of memory trying to allocate "
                 << strings::HumanReadableNumBytes(num_bytes)
                 << ".  See logs for memory state";
  }
  return nullptr;
}

void GPUBFCAllocator::SplitChunk(GPUBFCAllocator::ChunkHandle h,
                                 size_t num_bytes) {
  // Allocate the new chunk before we do any ChunkFromHandle
  ChunkHandle h_new_chunk = AllocateChunk();

  Chunk* c = ChunkFromHandle(h);
  CHECK(!c->in_use() && (c->bin_num == kInvalidBinNum));

  // Create a new chunk starting num_bytes after c
  GPUBFCAllocator::Chunk* new_chunk = ChunkFromHandle(h_new_chunk);
  new_chunk->ptr = static_cast<void*>(static_cast<char*>(c->ptr) + num_bytes);
  ptr_to_chunk_map_.set_handle(new_chunk->ptr, h_new_chunk);

  // Set the new sizes of the chunks.
  new_chunk->size = c->size - num_bytes;
  c->size = num_bytes;

  // The new chunk is not in use.
  new_chunk->allocation_id = -1;

  // Maintain the pointers.
  // c <-> c_neighbor becomes
  // c <-> new_chunk <-> c_neighbor
  GPUBFCAllocator::ChunkHandle h_neighbor = c->next;
  new_chunk->prev = h;
  new_chunk->next = h_neighbor;
  c->next = h_new_chunk;
  if (h_neighbor != kInvalidChunkHandle) {
    Chunk* c_neighbor = ChunkFromHandle(h_neighbor);
    c_neighbor->prev = h_new_chunk;
  }

  // Add the newly free chunk to the free bin.
  InsertFreeChunkIntoBin(h_new_chunk);
}

void GPUBFCAllocator::DeallocateRaw(void* ptr) {
  DeallocateRawInternal(ptr);
  retry_helper_.NotifyDealloc();
}

void GPUBFCAllocator::DeallocateRawInternal(void* ptr) {
  if (ptr == nullptr) {
    LOG(ERROR) << "tried to deallocate nullptr";
    return;
  }
  mutex_lock l(lock_);

  // Find the chunk from the ptr.
  GPUBFCAllocator::ChunkHandle h = ptr_to_chunk_map_.get_handle(ptr);
  CHECK(h != kInvalidChunkHandle);

  // Consider coalescing it.
  FreeAndMaybeCoalesce(h);
}

// Merges h1 and h2 when Chunk(h1)->next is h2 and Chunk(h2)->prev is c1.
// We merge Chunk(h2) into Chunk(h1).
void GPUBFCAllocator::Merge(GPUBFCAllocator::ChunkHandle h1,
                            GPUBFCAllocator::ChunkHandle h2) {
  Chunk* c1 = ChunkFromHandle(h1);
  Chunk* c2 = ChunkFromHandle(h2);
  // We can only merge chunks that are not in use.
  CHECK(!c1->in_use() && !c2->in_use());

  // c1's prev doesn't change, still points to the same ptr, and is
  // still not in use.

  // Fix up neighbor pointers
  //
  // c1 <-> c2 <-> c3 should become
  // c1 <-> c3

  GPUBFCAllocator::ChunkHandle h3 = c2->next;
  c1->next = h3;
  CHECK(c2->prev == h1);
  if (h3 != kInvalidChunkHandle) {
    GPUBFCAllocator::Chunk* c3 = ChunkFromHandle(h3);
    c3->prev = h1;
  }

  // Set the new size
  c1->size += c2->size;

  DeleteChunk(h2);
}

void GPUBFCAllocator::DeleteChunk(ChunkHandle h) {
  // Delete h and cleanup all state
  Chunk* c = ChunkFromHandle(h);
  //  VLOG(4) << "Removing: " << c->ptr;
  ptr_to_chunk_map_.erase(c->ptr);
  DeallocateChunk(h);
}

void GPUBFCAllocator::InsertFreeChunkIntoBin(GPUBFCAllocator::ChunkHandle h) {
  Chunk* c = ChunkFromHandle(h);
  CHECK(!c->in_use() && (c->bin_num == kInvalidBinNum));
  BinNum bin_num = BinNumForSize(c->size);
  Bin* new_bin = BinFromIndex(bin_num);
  c->bin_num = bin_num;
  new_bin->free_chunks.insert(h);
}

void GPUBFCAllocator::RemoveFreeChunkIterFromBin(
    GPUBFCAllocator::Bin::FreeChunkSet* free_chunks,
    const GPUBFCAllocator::Bin::FreeChunkSet::iterator& citer) {
  ChunkHandle h = *citer;
  Chunk* c = ChunkFromHandle(h);
  CHECK(!c->in_use() && (c->bin_num != kInvalidBinNum));
  free_chunks->erase(citer);
  c->bin_num = kInvalidBinNum;
}

void GPUBFCAllocator::RemoveFreeChunkFromBin(GPUBFCAllocator::ChunkHandle h) {
  Chunk* c = ChunkFromHandle(h);
  CHECK(!c->in_use() && (c->bin_num != kInvalidBinNum));
  int count = BinFromIndex(c->bin_num)->free_chunks.erase(h);
  CHECK(count > 0) << "Could not find chunk in bin";
  c->bin_num = kInvalidBinNum;
}

void GPUBFCAllocator::FreeAndMaybeCoalesce(GPUBFCAllocator::ChunkHandle h) {
  Chunk* c = ChunkFromHandle(h);
  CHECK(c->in_use() && (c->bin_num == kInvalidBinNum));

  // Mark the chunk as no longer in use
  c->allocation_id = -1;

  // Updates the stats.
  stats_.bytes_in_use -= c->size;

  // This chunk is no longer in-use, consider coalescing the chunk
  // with adjacent chunks.
  ChunkHandle chunk_to_reassign = h;

  // If the next chunk is free, coalesce the two
  if (c->next != kInvalidChunkHandle) {
    Chunk* cnext = ChunkFromHandle(c->next);
    if (!cnext->in_use()) {
      //      VLOG(8) << "Chunk at " << cnext->ptr << " merging with c " <<
      //      c->ptr;

      chunk_to_reassign = h;

      // Deletes c->next
      RemoveFreeChunkFromBin(c->next);
      Merge(h, ChunkFromHandle(h)->next);
    }
  }

  // If the previous chunk is free, coalesce the two
  c = ChunkFromHandle(h);
  if (c->prev != kInvalidChunkHandle) {
    Chunk* cprev = ChunkFromHandle(c->prev);
    if (!cprev->in_use()) {
      //      VLOG(8) << "Chunk at " << c->ptr << " merging into c->prev "
      //       << cprev->ptr;

      chunk_to_reassign = c->prev;

      // Deletes c
      RemoveFreeChunkFromBin(c->prev);
      Merge(ChunkFromHandle(h)->prev, h);
      c = ChunkFromHandle(h);
    }
  }

  InsertFreeChunkIntoBin(chunk_to_reassign);
}

void GPUBFCAllocator::AddAllocVisitor(Visitor visitor) {
  VLOG(1) << "AddVisitor";
  mutex_lock l(lock_);
  region_visitors_.push_back(visitor);
  if (base_ptr_ != nullptr) {
    visitor(base_ptr_, gpu_memory_size_);
  }
}

bool GPUBFCAllocator::TracksAllocationSizes() { return true; }

size_t GPUBFCAllocator::RequestedSize(void* ptr) {
  mutex_lock l(lock_);
  GPUBFCAllocator::ChunkHandle h = ptr_to_chunk_map_.get_handle(ptr);
  CHECK(h != kInvalidChunkHandle)
      << "Asked for requested size of pointer we never allocated: " << ptr;
  GPUBFCAllocator::Chunk* c = ChunkFromHandle(h);
  return c->requested_size;
}

size_t GPUBFCAllocator::AllocatedSize(void* ptr) {
  mutex_lock l(lock_);
  GPUBFCAllocator::ChunkHandle h = ptr_to_chunk_map_.get_handle(ptr);
  CHECK(h != kInvalidChunkHandle)
      << "Asked for allocated size of pointer we never allocated: " << ptr;
  GPUBFCAllocator::Chunk* c = ChunkFromHandle(h);
  return c->size;
}

int64 GPUBFCAllocator::AllocationId(void* ptr) {
  mutex_lock l(lock_);
  GPUBFCAllocator::ChunkHandle h = ptr_to_chunk_map_.get_handle(ptr);
  CHECK(h != kInvalidChunkHandle)
      << "Asked for allocation id of pointer we never allocated: " << ptr;
  GPUBFCAllocator::Chunk* c = ChunkFromHandle(h);
  return c->allocation_id;
}

void GPUBFCAllocator::DumpMemoryLog(size_t num_bytes) {
  // For each bin: tally up the total number of chunks and bytes.
  // Note that bins hold only free chunks.
  for (BinNum bin_num = 0; bin_num < kNumBins; bin_num++) {
    Bin* b = BinFromIndex(bin_num);

    size_t total_bytes_in_use = 0;
    size_t total_bytes_in_bin = 0;
    size_t total_requested_bytes_in_use = 0;
    size_t total_requested_bytes_in_bin = 0;
    size_t total_chunks_in_use = 0;
    size_t total_chunks_in_bin = 0;
    for (ChunkHandle h : b->free_chunks) {
      Chunk* c = ChunkFromHandle(h);
      total_bytes_in_bin += c->size;
      total_requested_bytes_in_bin += c->requested_size;
      ++total_chunks_in_bin;
      if (c->in_use()) {
        total_bytes_in_use += c->size;
        total_requested_bytes_in_use += c->requested_size;
        ++total_chunks_in_use;
      }
    }

    LOG(INFO) << "Bin (" << b->bin_size
              << "): \tTotal Chunks: " << total_chunks_in_bin
              << ", Chunks in use: " << total_chunks_in_use << " "
              << strings::HumanReadableNumBytes(total_bytes_in_bin)
              << " allocated for chunks. "
              << strings::HumanReadableNumBytes(total_requested_bytes_in_bin)
              << " client-requested for chunks. "
              << strings::HumanReadableNumBytes(total_bytes_in_use)
              << " in use in bin. "
              << strings::HumanReadableNumBytes(total_requested_bytes_in_use)
              << " client-requested in use in bin.";
  }

  // Find the bin that we would have liked to allocate in, so we
  // can get some further analysis about fragmentation.
  Bin* b = BinForSize(num_bytes);

  LOG(INFO) << "Bin for " << strings::HumanReadableNumBytes(num_bytes)
            << " was " << strings::HumanReadableNumBytes(b->bin_size)
            << ", Chunk State: ";

  for (ChunkHandle h : b->free_chunks) {
    Chunk* c = ChunkFromHandle(h);
    LOG(INFO) << c->DebugString(this, true);
  }

  // Next show the chunks that are in use, and also summarize their
  // number by size.
  std::map<size_t, int> in_use_by_size;
  for (const char* p = reinterpret_cast<const char*>(base_ptr_);
       p < reinterpret_cast<const char*>(base_ptr_) + gpu_memory_size_;
       p += 256) {
    const void* void_p = reinterpret_cast<const void*>(p);
    ChunkHandle h = ptr_to_chunk_map_.get_handle(void_p);
    if (h != kInvalidChunkHandle) {
      const Chunk* c = ChunkFromHandle(h);
      in_use_by_size[c->size]++;
      LOG(INFO) << "Chunk at " << void_p << " of size " << c->size;
    }
  }

  LOG(INFO) << "     Summary of in-use Chunks by size: ";
  size_t total_bytes = 0;
  for (auto& it : in_use_by_size) {
    LOG(INFO) << it.second << " Chunks of size " << it.first << " totalling "
              << strings::HumanReadableNumBytes(it.first * it.second);
    total_bytes += (it.first * it.second);
  }
  LOG(INFO) << "Sum Total of in-use chunks: "
            << strings::HumanReadableNumBytes(total_bytes);
}

void GPUBFCAllocator::GetStats(AllocatorStats* stats) {
  mutex_lock l(lock_);
  *stats = stats_;
}

}  // namespace tensorflow
