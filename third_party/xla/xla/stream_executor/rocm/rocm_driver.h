/* Copyright 2023 The OpenXLA Authors.

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

// The ROCM-specific Driver library support, implementing the general Driver
// interface.

#ifndef XLA_STREAM_EXECUTOR_ROCM_ROCM_DRIVER_H_
#define XLA_STREAM_EXECUTOR_ROCM_ROCM_DRIVER_H_

#include "absl/container/node_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/gpu/gpu_driver.h"
#include "tsl/platform/logging.h"

namespace stream_executor {
namespace gpu {
// Formats hipError_t to output prettified values into a log stream.
// Error summaries taken from:
std::string ToString(hipError_t result);

absl::StatusOr<hipError_t> QueryEvent(GpuContext* context, hipEvent_t event);

// GpuContext wraps the device_ordinal and hipCtx_t handle.
class GpuContext {
 public:
  GpuContext(hipCtx_t context, const int v)
      : context_(context), device_ordinal_(v) {}

  hipCtx_t context() const { return context_; }
  int device_ordinal() const { return device_ordinal_; }

  // Disallow copying and moving.
  GpuContext(GpuContext&&) = delete;
  GpuContext(const GpuContext&) = delete;
  GpuContext& operator=(GpuContext&&) = delete;
  GpuContext& operator=(const GpuContext&) = delete;

 private:
  hipCtx_t const context_;
  const int device_ordinal_;
};

// Manages the singleton map of contexts that we've created, mapping
// from the hipCtx_t to the GpuContext* that we pass around internally.
// This also manages assignment of unique ids to GpuContexts, to allow
// for fast comparison of a context against the current context.
//
// HIP-runtime-created contexts are avoided, if triple angle
// brace launches are required, by using the scoped activations in
// gpu/gpu_activation.h.
class CreatedContexts {
 public:
  // Returns whether context is a member of the live set.
  static bool Has(hipCtx_t context) {
    absl::ReaderMutexLock lock(&mu_);
    return Live()->find(context) != Live()->end();
  }

  // Adds context to the live set, or returns it if it's already present.
  static GpuContext* Add(hipCtx_t context, int device_ordinal) {
    CHECK(context != nullptr);
    absl::MutexLock lock(&mu_);

    auto insert_result = Live()->insert(std::make_pair(context, nullptr));
    auto it = insert_result.first;
    if (insert_result.second) {
      // context was not present in the map.  Add it.
      it->second = std::make_unique<GpuContext>(context, next_id_++);
      (*LiveOrdinal())[device_ordinal].push_back(context);
    }
    return it->second.get();
  }

  // Removes context from the live set.
  static void Remove(hipCtx_t context) {
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

  // Return the context associated to that ptr.
  static hipCtx_t GetAnyContext(void* ptr) {
    absl::ReaderMutexLock lock(&mu_);
    int device_ordinal;
    hipError_t result =
        hipPointerGetAttribute(static_cast<void*>(&device_ordinal),
                               HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                               reinterpret_cast<hipDeviceptr_t>(ptr));
    if (result != hipSuccess) {
      LOG(FATAL) << "Not able to get the device_ordinal for ptr: " << ptr
                 << ". Error: " << ToString(result);
    }
    CHECK_EQ(LiveOrdinal()->count(device_ordinal), 1);
    CHECK(!LiveOrdinal()->at(device_ordinal).empty())
        << "Need at least one context.";
    return LiveOrdinal()->at(device_ordinal)[0];
  }

 private:
  // Returns the live map singleton.
  static absl::node_hash_map<hipCtx_t, std::unique_ptr<GpuContext>>* Live() {
    static auto singleton =
        new absl::node_hash_map<hipCtx_t, std::unique_ptr<GpuContext>>;
    return singleton;
  }
  static absl::node_hash_map<int, std::vector<hipCtx_t>>* LiveOrdinal() {
    static auto singleton = new absl::node_hash_map<int, std::vector<hipCtx_t>>;
    return singleton;
  }

  // Lock that guards access-to/mutation-of the live set.
  static absl::Mutex mu_;
  static int64_t next_id_;
};
}  // namespace gpu

namespace rocm {

using MemorySpace = gpu::MemorySpace;
using ScopedActivateContext = gpu::ScopedActivateContext;

// TODO: this function shall be added to the GpuDriver API as well
absl::Status OccupancyGetMaxPotentialBlockSize(int* gridSize, int* blockSize,
                                               hipFunction_t func,
                                               size_t dynSharedMemPerBlk,
                                               int blockSizeLimit);

// Returns the current context set in ROCm. This is done by calling ROCm
// driver (e.g., this value is not our cached view of the current context).
hipCtx_t CurrentContextOrDie();
}  // namespace rocm
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_ROCM_ROCM_DRIVER_H_
