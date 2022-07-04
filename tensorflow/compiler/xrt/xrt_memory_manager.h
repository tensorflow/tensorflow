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

#ifndef TENSORFLOW_COMPILER_XRT_XRT_MEMORY_MANAGER_H_
#define TENSORFLOW_COMPILER_XRT_XRT_MEMORY_MANAGER_H_

#include <memory>
#include <vector>

#include "tensorflow/compiler/xla/service/backend.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/compiler/xrt/xrt_refptr.h"
#include "tensorflow/compiler/xrt/xrt_state.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/stream_executor/device_memory_allocator.h"
#include "tensorflow/stream_executor/stream_executor.h"

namespace tensorflow {

// The XRTMemoryManager manages all the XRT allocations. It is a ResourceBase
// object which leaves within the ResourceMgr. This is only one XRT memory
// manager object within the ResourceMgr container.
class XRTMemoryManager : public ResourceBase {
  // The DeviceContext class, defined and implemented locally inside the
  // xrt_memory_manager.cc file, holds, for each device, all the information
  // related to the XRT memory management for such device.
  class DeviceContext;

 public:
  // A working set is a set of tuple allocations which are the input of a given
  // operation, and as such they must be pinned on the device memory. The tuple
  // allocations added to the WorkingSet will be unpinned at object destruction.
  class WorkingSet {
   public:
    explicit WorkingSet(RefPtr<XRTMemoryManager> memory_manager);

    ~WorkingSet();

    // Looks up the tuple handle within the memory manager, and pins it to the
    // device (if not already pinned).
    Status LookupAndPin(xla::Backend* backend, int64_t handle,
                        se::DeviceMemoryAllocator* allocator);

    const std::vector<RefPtr<XRTTupleAllocation>>& PinnedTuples() const {
      return pinned_tuples_;
    }

    const RefPtr<XRTMemoryManager>& MemoryManager() const {
      return memory_manager_;
    }

   private:
    RefPtr<XRTMemoryManager> memory_manager_;
    std::vector<RefPtr<XRTTupleAllocation>> pinned_tuples_;
  };

  // Retrieves the XRTMemoryManager singleton stored within the ResourceMgr.
  static RefPtr<XRTMemoryManager> Get(ResourceMgr* rm);

  // Registers an XRTTupleAllocation and returns the unique handle identifying
  // it.
  int64_t Register(RefPtr<XRTTupleAllocation> tuple);

  // Looks up an handle returned by the Register() API and returns the
  // XRTTupleAllocation behind it.
  xla::StatusOr<RefPtr<XRTTupleAllocation>> Lookup(int64_t handle);

  Status Lookup(int64_t handle, RefPtr<XRTTupleAllocation>* tuple) {
    TF_ASSIGN_OR_RETURN(*tuple, Lookup(handle));
    return OkStatus();
  }

  // Releases an handle by dropping the references count held on the
  // XRTTupleAllocation by the XRTMemoryManager. Existing XRTTupleAllocation
  // references will continue to be valid.
  Status Release(int64_t handle);

  // Tries to compact all the memory allocations on a given device. This is
  // currently done by swapping-out all the existing allocation, and swapping
  // them back in.
  Status CompactAllocations(xla::Backend* backend, int device_ordinal,
                            se::DeviceMemoryAllocator* allocator);

  // Releases all the device memory allocated by XRT within the resource
  // manager.
  void ReleaseAllAllocations();

  // Tries to allocate size bytes of device memory from the device_ordinal
  // device. Might attempt to free some unpinned device memory, if the underline
  // allocator call fails, and try the allocation again.
  xla::StatusOr<se::OwningDeviceMemory> Allocate(
      xla::Backend* backend, int device_ordinal, size_t size,
      se::DeviceMemoryAllocator* allocator);

  // Runs the specified function and handling the error::RESOURCE_EXHAUSTED
  // status code coming out of it. In such cases, we run different memory
  // freeing operations trying to make runfn succeed. The requested_free_size
  // argument represents an hint of the requested memory size which would make
  // runfn succeed.
  template <typename T>
  xla::StatusOr<T> Run(const std::function<xla::StatusOr<T>()>& runfn,
                       xla::Backend* backend, int device_ordinal,
                       size_t requested_free_size,
                       se::DeviceMemoryAllocator* allocator);

  string DebugString() const override;

  // Returns the invalid key value, which will be never generated by the
  // Intern() API.
  static int64_t InvalidKey() { return 0; }

 private:
  // Structure used to track the progress of a try-to-free operation. It is
  // initialized and the passed to the TryFreeMemoryStep() API.
  struct MemoryReclaimContext {
    MemoryReclaimContext(xla::Backend* backend, int device_ordinal,
                         size_t requested_free_size,
                         se::DeviceMemoryAllocator* specific_allocator)
        : backend(backend),
          device_ordinal(device_ordinal),
          requested_free_size(requested_free_size) {
      allocator = specific_allocator;
    }

    xla::Backend* const backend = nullptr;
    se::DeviceMemoryAllocator* allocator = nullptr;
    const int device_ordinal = 0;
    const size_t requested_free_size = 0;
    size_t free_size = 0;
    bool done_freeing = false;
    bool done_compacting = false;
  };

  DeviceContext* GetDeviceContext(int device_ordinal, bool create_if_missing);

  // Called multiple times while trying to make a memory consuming function call
  // to fit. Performs progressively more expensive memory reduction operations,
  // until returning error::RESOURCE_EXHAUSTED when no further reductions are
  // possible.
  Status TryFreeMemoryStep(MemoryReclaimContext* mrctx, const Status& status);

  mutex lock_;
  std::vector<std::unique_ptr<DeviceContext>> device_contexts_;
};

template <typename T>
xla::StatusOr<T> XRTMemoryManager::Run(
    const std::function<xla::StatusOr<T>()>& runfn, xla::Backend* backend,
    int device_ordinal, size_t requested_free_size,
    se::DeviceMemoryAllocator* allocator) {
  MemoryReclaimContext mrctx(backend, device_ordinal, requested_free_size,
                             allocator);
  while (true) {
    // We assume that runfn is a relatively fast-fail function compared to the
    // operations required to free up the required memory. Here we call into the
    // TryFreeMemoryStep() API multiple times, which will run progressively more
    // expensive operations.
    auto result_or = runfn();
    if (result_or.status().code() != error::RESOURCE_EXHAUSTED) {
      return result_or;
    }
    TF_RETURN_IF_ERROR(TryFreeMemoryStep(&mrctx, result_or.status()));
  }
}

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_XRT_XRT_MEMORY_MANAGER_H_
