/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xrt/xrt_memory_manager.h"

#include <algorithm>
#include <list>
#include <unordered_map>

#include "absl/memory/memory.h"
#include "tensorflow/compiler/xrt/xrt_metrics.h"
#include "tensorflow/core/lib/monitoring/timed.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/profiler/lib/traceme.h"

namespace tensorflow {
namespace {

// We use kDeviceBits to store the device ordinal in the handle. We store the
// device in the upper part of the int64 handle to make sure the random bits are
// in the lower part which is better when storing the handle as a key for
// unordered maps.
const int kDeviceBits = 12;

int64_t MakeDeviceHandle(int64_t device_ordinal, int64_t rnd_value) {
  const int64_t kUidMask = (static_cast<int64_t>(1) << (64 - kDeviceBits)) - 1;
  return (device_ordinal << (64 - kDeviceBits)) | (rnd_value & kUidMask);
}

int GetDeviceFromHandle(int64_t handle) {
  return (handle >> (64 - kDeviceBits)) & ((1 << kDeviceBits) - 1);
}

}  // namespace

class XRTMemoryManager::DeviceContext {
  struct Alloc {
    explicit Alloc(RefPtr<XRTTupleAllocation> tuple)
        : tuple(std::move(tuple)) {}

    RefPtr<XRTTupleAllocation> tuple;
  };

  using AllocList = std::list<Alloc>;

 public:
  int64_t Register(RefPtr<XRTTupleAllocation> tuple) {
    while (true) {
      int64_t handle = MakeDeviceHandle(tuple->device_ordinal(), CreateUid());
      mutex_lock lock(lock_);
      allocs_.emplace_front(tuple);
      if (alloc_map_.emplace(handle, allocs_.begin()).second) {
        return handle;
      }
      // The chances of hitting an existing handle are so remote, it is much
      // more convenient to add to the list before, and eventually removing.
      allocs_.erase(allocs_.begin());
    }
  }

  bool Release(int64_t handle) {
    mutex_lock lock(lock_);
    auto it = alloc_map_.find(handle);
    if (it == alloc_map_.end()) {
      return false;
    }
    allocs_.erase(it->second);
    alloc_map_.erase(it);
    return true;
  }

  RefPtr<XRTTupleAllocation> Lookup(int64_t handle) {
    mutex_lock lock(lock_);
    auto it = alloc_map_.find(handle);
    if (it == alloc_map_.end()) {
      return nullptr;
    }
    // LRU
    allocs_.splice(allocs_.begin(), allocs_, it->second);
    return it->second->tuple;
  }

  void Clear() {
    mutex_lock lock(lock_);
    alloc_map_.clear();
    allocs_.clear();
  }

  Status CompactAllocations(XRTMemoryManager* memory_manager,
                            xla::Backend* backend,
                            se::DeviceMemoryAllocator* allocator) {
    profiler::TraceMe trace_me("XRTMemoryManager::CompactAllocations",
                               /*level=*/2);
    auto timed = monitoring::MakeTimed(xrt_metrics::GetMemoryCompactCell());
    VLOG(4) << "CompactAllocations started";
    mutex_lock lock(lock_);
    Status status;
    std::vector<AllocList::iterator> swapped;
    // We are swapping out from the most recently used allocations. This is
    // desirable since the most recently used will be finding themselves at the
    // bottom of the allocation space. Since these are more likely to be pinned
    // allocations, a further trim done by following TryFreeMemory() call will
    // eventually drop the higher located allocations, with better chance of
    // reducing fragmentation.
    // Also, by swapping out the pinned allocations first, those will also be
    // the first to be restored, and hence if we will ever find OOM on the way
    // out, we would more likely be swapping in not pinned ones.
    for (auto it = allocs_.begin(); it != allocs_.end(); ++it) {
      // We are compacting all the allocations, so we will temporarily swap out
      // even pinned allocations.
      auto swap_result_or = it->tuple->SwapOut(backend, /*swap_pinned=*/true);
      if (!swap_result_or.ok()) {
        status = swap_result_or.status();
        break;
      }
      if (swap_result_or.value()) {
        swapped.push_back(it);
      }
    }
    // At this point we have released all the device memory we could release.
    // Load back the tuple allocations we have swapped out above.
    for (auto& it : swapped) {
      auto swap_result_or =
          it->tuple->SwapIn(memory_manager, backend, allocator);
      if (!swap_result_or.ok()) {
        // If we failed to restored a pinned allocation, better to CHECK here
        // than wondering why XRTTupleAllocation calls fail with errors about
        // missing buffers.
        CHECK(!it->tuple->IsPinned());  // Crash OK
        if (status.ok()) {
          status = swap_result_or.status();
        }
      }
    }
    VLOG(4) << "CompactAllocations finished: " << status;
    return status;
  }

  // Tries to free size bytes by freeing some unpinned device memory. Returns
  // the amount of memory which was able to free.
  xla::StatusOr<size_t> TryFreeMemory(xla::Backend* backend, size_t size) {
    profiler::TraceMe trace_me("XRTMemoryManager::TryFreeMemory", /*level=*/2);
    auto timed = monitoring::MakeTimed(xrt_metrics::GetTryFreeMemoryCell());
    mutex_lock lock(lock_);
    size_t swapped_size = 0;
    for (auto it = allocs_.rbegin(); it != allocs_.rend(); ++it) {
      TF_ASSIGN_OR_RETURN(bool swap_result,
                          it->tuple->SwapOut(backend, /*swap_pinned=*/false));
      if (swap_result) {
        swapped_size += it->tuple->GetDeviceMemorySize();
        if (swapped_size >= size) {
          break;
        }
      }
    }
    VLOG(3) << "Swapped out " << swapped_size << " bytes";
    return swapped_size;
  }

 private:
  static int64_t CreateUid() {
    int64_t uid;
    do {
      uid = random::New64() & INT64_MAX;
    } while (uid == InvalidKey());
    return uid;
  }

  // We store Alloc records inside an std::list<Alloc> so we can LRU it, and
  // store the list iterators within the handle map, as list iterators don't get
  // invalidated by (other elements) removals or position swaps.
  mutex lock_;
  AllocList allocs_;
  std::unordered_map<int64_t, AllocList::iterator> alloc_map_;
};

XRTMemoryManager::WorkingSet::WorkingSet(
    RefPtr<XRTMemoryManager> memory_manager)
    : memory_manager_(std::move(memory_manager)) {}

XRTMemoryManager::WorkingSet::~WorkingSet() {
  for (auto& tuple : pinned_tuples_) {
    tuple->Unpin();
  }
}

Status XRTMemoryManager::WorkingSet::LookupAndPin(
    xla::Backend* backend, int64_t handle,
    se::DeviceMemoryAllocator* allocator) {
  TF_ASSIGN_OR_RETURN(auto tuple, memory_manager_->Lookup(handle));
  TF_RETURN_IF_ERROR(
      tuple->PinAndSwapIn(memory_manager_.get(), backend, allocator).status());
  pinned_tuples_.push_back(std::move(tuple));
  return OkStatus();
}

/* static */ RefPtr<XRTMemoryManager> XRTMemoryManager::Get(ResourceMgr* rm) {
  static string* container = new string("XrtState");
  static string* name = new string("MemoryManager");
  XRTMemoryManager* memory_manager = nullptr;
  TF_CHECK_OK(rm->LookupOrCreate<XRTMemoryManager>(
      *container, *name, &memory_manager, [](XRTMemoryManager** ret) {
        *ret = new XRTMemoryManager();
        return OkStatus();
      }));
  return memory_manager;
}

int64_t XRTMemoryManager::Register(RefPtr<XRTTupleAllocation> tuple) {
  DeviceContext* device_context = GetDeviceContext(tuple->device_ordinal(),
                                                   /*create_if_missing=*/true);
  return device_context->Register(std::move(tuple));
}

xla::StatusOr<RefPtr<XRTTupleAllocation>> XRTMemoryManager::Lookup(
    int64_t handle) {
  int device_ordinal = GetDeviceFromHandle(handle);
  DeviceContext* device_context = GetDeviceContext(device_ordinal,
                                                   /*create_if_missing=*/false);
  if (device_context == nullptr) {
    return errors::NotFound("XRT memory handle not found: ", handle);
  }
  RefPtr<XRTTupleAllocation> tuple = device_context->Lookup(handle);
  if (tuple == nullptr) {
    return errors::NotFound("XRT memory handle not found: ", handle);
  }
  return std::move(tuple);
}

Status XRTMemoryManager::Release(int64_t handle) {
  int device_ordinal = GetDeviceFromHandle(handle);
  DeviceContext* device_context = GetDeviceContext(device_ordinal,
                                                   /*create_if_missing=*/false);
  if (device_context == nullptr || !device_context->Release(handle)) {
    return errors::NotFound("XRT memory handle not found: ", handle);
  }
  return OkStatus();
}

Status XRTMemoryManager::CompactAllocations(
    xla::Backend* backend, int device_ordinal,
    se::DeviceMemoryAllocator* allocator) {
  DeviceContext* device_context = GetDeviceContext(device_ordinal,
                                                   /*create_if_missing=*/false);
  return device_context != nullptr
             ? device_context->CompactAllocations(this, backend, allocator)
             : OkStatus();
}

void XRTMemoryManager::ReleaseAllAllocations() {
  mutex_lock lock(lock_);
  for (auto& device_context : device_contexts_) {
    if (device_context != nullptr) {
      device_context->Clear();
    }
  }
}

xla::StatusOr<se::OwningDeviceMemory> XRTMemoryManager::Allocate(
    xla::Backend* backend, int device_ordinal, size_t size,
    se::DeviceMemoryAllocator* allocator) {
  auto memory_or =
      allocator->Allocate(device_ordinal, size, /*retry_on_failure=*/false);
  if (memory_or.status().code() == error::RESOURCE_EXHAUSTED) {
    VLOG(4) << "Allocate of " << size << " bytes failed on device "
            << device_ordinal;

    DeviceContext* device_context =
        GetDeviceContext(device_ordinal,
                         /*create_if_missing=*/false);
    if (device_context != nullptr) {
      Status status = device_context->TryFreeMemory(backend, size).status();
      if (status.ok()) {
        // As long as there is no error, we still try again the allocation, even
        // if the TryFreeMemory() call ended up freeing less memory than the
        // required size. Fragmentation could make the memory allocation succeed
        // even if the freed memory is indeed lower.
        memory_or = allocator->Allocate(device_ordinal, size,
                                        /*retry_on_failure=*/false);
      } else if (status.code() != error::RESOURCE_EXHAUSTED) {
        VLOG(4) << "Allocate of " << size << " bytes on device "
                << device_ordinal << ": " << status;
        return status;
      }
    }
  }
  return memory_or;
}

string XRTMemoryManager::DebugString() const {
  // We might want to emit more detailed information here, like per device
  // memory allocations.
  return "XRTMemoryManager";
}

XRTMemoryManager::DeviceContext* XRTMemoryManager::GetDeviceContext(
    int device_ordinal, bool create_if_missing) {
  mutex_lock lock(lock_);
  if (device_ordinal >= device_contexts_.size()) {
    if (!create_if_missing) {
      return nullptr;
    }
    device_contexts_.resize(device_ordinal + 1);
  }
  DeviceContext* device_context = device_contexts_[device_ordinal].get();
  if (device_context == nullptr && create_if_missing) {
    device_contexts_[device_ordinal] = absl::make_unique<DeviceContext>();
    device_context = device_contexts_[device_ordinal].get();
  }
  return device_context;
}

Status XRTMemoryManager::TryFreeMemoryStep(MemoryReclaimContext* mrctx,
                                           const Status& status) {
  DeviceContext* device_context = GetDeviceContext(mrctx->device_ordinal,
                                                   /*create_if_missing=*/false);
  if (device_context == nullptr) {
    return status;
  }
  if (!mrctx->done_freeing) {
    // If the caller passed us a zero requested_free_size, we try to free chunks
    // of kMaxFreeSize memory, until either the run function succeeds, or we run
    // out of freeable memory.
    const size_t kMaxFreeSize = 1000000000;
    size_t free_size =
        (mrctx->requested_free_size > 0)
            ? std::min<size_t>(mrctx->requested_free_size - mrctx->free_size,
                               kMaxFreeSize)
            : kMaxFreeSize;
    if (free_size > 0) {
      auto free_size_or =
          device_context->TryFreeMemory(mrctx->backend, free_size);
      if (!free_size_or.ok()) {
        return status;
      }
      size_t size = free_size_or.value();
      mrctx->free_size += size;
      if (size > 0) {
        return OkStatus();
      }
    }
    mrctx->done_freeing = true;
  }
  if (!mrctx->done_compacting) {
    mrctx->done_compacting = true;
    if (device_context
            ->CompactAllocations(this, mrctx->backend, mrctx->allocator)
            .ok()) {
      return OkStatus();
    }
  }
  return status;
}

}  // namespace tensorflow
