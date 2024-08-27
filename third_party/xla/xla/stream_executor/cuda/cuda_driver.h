/* Copyright 2015 The OpenXLA Authors.

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

// CUDA userspace driver library wrapper functionality.

#ifndef XLA_STREAM_EXECUTOR_CUDA_CUDA_DRIVER_H_
#define XLA_STREAM_EXECUTOR_CUDA_CUDA_DRIVER_H_

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/node_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/synchronization/mutex.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/cuda/cuda_status.h"
#include "xla/stream_executor/gpu/context.h"
#include "xla/stream_executor/gpu/gpu_driver.h"

namespace stream_executor {
namespace gpu {

// Polls (without blocking) to determine the status of an event - pending or
// complete (or an error status).
// http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html#group__CUDA__EVENT_1g6f0704d755066b0ee705749ae911deef
absl::StatusOr<CUresult> QueryEvent(GpuContext* context, CUevent event);

// CUDAContext wraps a cuda CUcontext handle, and includes a unique id. The
// unique id is positive, and ids are not repeated within the process.
class GpuContext : public Context {
 public:
  GpuContext(CUcontext context, int device_ordinal)
      : context_(context), device_ordinal_(device_ordinal) {}

  void SetActive() override;
  bool IsActive() const override;
  CUcontext context() const { return context_; }
  int device_ordinal() const override { return device_ordinal_; }

  // Disallow copying and moving.
  GpuContext(GpuContext&&) = delete;
  GpuContext(const GpuContext&) = delete;
  GpuContext& operator=(GpuContext&&) = delete;
  GpuContext& operator=(const GpuContext&) = delete;

 private:
  CUcontext const context_;
  const int device_ordinal_;
};

// Manages the singleton map of contexts that we've created, mapping
// from the CUcontext to the GpuContext* that we pass around internally.
// This also manages assignment of unique ids to GpuContexts, to allow
// for fast comparison of a context against the current context.
//
// CUDA-runtime-created contexts are avoided, if triple angle
// brace launches are required, by using the scoped activations in
// gpu/gpu_activation.h.
class CreatedContexts {
 public:
  // Returns whether context is a member of the live set.
  static bool Has(CUcontext context) {
    absl::ReaderMutexLock lock(&mu_);
    return Live()->find(context) != Live()->end();
  }

  // Adds context to the live set, or returns it if it's already present.
  static GpuContext* Add(CUcontext context, int device_ordinal) {
    CHECK(context != nullptr);
    absl::MutexLock lock(&mu_);

    auto insert_result = Live()->insert(std::make_pair(context, nullptr));
    auto it = insert_result.first;
    if (insert_result.second) {
      // context was not present in the map.  Add it.
      it->second = std::make_unique<GpuContext>(context, device_ordinal);
      (*LiveOrdinal())[device_ordinal].push_back(context);
    }
    return it->second.get();
  }

  // Removes context from the live set.
  static void Remove(CUcontext context) {
    CHECK(context != nullptr);
    absl::MutexLock lock(&mu_);
    auto it = Live()->find(context);
    CHECK(it != Live()->end()) << context;
    Live()->erase(it);
    for (auto p : (*LiveOrdinal())) {
      auto it2 = std::find(p.second.begin(), p.second.end(), context);
      if (it2 != p.second.end()) {
        p.second.erase(it2, it2++);
        if (p.second.empty()) {
          LiveOrdinal()->erase(p.first);
        }
        break;
      }
    }
  }

  // Find device id from cuda pointer value.
  static int GetDeviceOrdinal(void* ptr) {
    int device_ordinal;
    absl::Status status = cuda::ToStatus(
        cuPointerGetAttribute(static_cast<void*>(&device_ordinal),
                              CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                              reinterpret_cast<CUdeviceptr>(ptr)));
    if (!status.ok()) {
      LOG(FATAL) << "Not able to get the device_ordinal for ptr: " << ptr
                 << ". Error: " << status;
    }
    return device_ordinal;
  }

  // Return the context associated to that ptr.
  static CUcontext GetAnyContext(void* ptr) {
    absl::ReaderMutexLock lock(&mu_);
    int device_ordinal = GetDeviceOrdinal(ptr);
    CHECK_EQ(LiveOrdinal()->count(device_ordinal), 1);
    CHECK(!LiveOrdinal()->at(device_ordinal).empty())
        << "Need at least one context.";
    return LiveOrdinal()->at(device_ordinal)[0];
  }

 private:
  // Returns the live map singleton.
  static absl::node_hash_map<CUcontext, std::unique_ptr<GpuContext>>* Live() {
    static auto singleton =
        new absl::node_hash_map<CUcontext, std::unique_ptr<GpuContext>>;
    return singleton;
  }
  static absl::node_hash_map<int, std::vector<CUcontext>>* LiveOrdinal() {
    static auto singleton =
        new absl::node_hash_map<int, std::vector<CUcontext>>;
    return singleton;
  }

  // Lock that guards access-to/mutation-of the live set.
  static absl::Mutex mu_;
};
}  // namespace gpu

namespace cuda {

using CUDADriver = gpu::GpuDriver;

using ScopedActivateContext = gpu::ScopedActivateContext;

using CudaContext = gpu::GpuContext;

// Returns the current context set in CUDA. This is done by calling the cuda
// driver (e.g., this value is not our cached view of the current context).
CUcontext CurrentContextOrDie();

}  // namespace cuda
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_CUDA_CUDA_DRIVER_H_
