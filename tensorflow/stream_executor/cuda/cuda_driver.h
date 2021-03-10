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

// CUDA userspace driver library wrapper functionality.

#ifndef TENSORFLOW_STREAM_EXECUTOR_CUDA_CUDA_DRIVER_H_
#define TENSORFLOW_STREAM_EXECUTOR_CUDA_CUDA_DRIVER_H_

#include "absl/memory/memory.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/stream_executor/gpu/gpu_driver.h"

namespace stream_executor {
namespace gpu {
// CUDAContext wraps a cuda CUcontext handle, and includes a unique id. The
// unique id is positive, and ids are not repeated within the process.
class GpuContext {
 public:
  GpuContext(CUcontext context, int64 id) : context_(context), id_(id) {}

  CUcontext context() const { return context_; }
  int64 id() const { return id_; }

  // Disallow copying and moving.
  GpuContext(GpuContext&&) = delete;
  GpuContext(const GpuContext&) = delete;
  GpuContext& operator=(GpuContext&&) = delete;
  GpuContext& operator=(const GpuContext&) = delete;

 private:
  CUcontext const context_;
  const int64 id_;
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
      it->second = absl::make_unique<GpuContext>(context, next_id_++);
    }
    CHECK(LiveOrdinal()->count(device_ordinal) == 0);
    auto insert_result_ordinal = LiveOrdinal()->insert(
        std::make_pair(device_ordinal, context));
    return it->second.get();
  }

  // Removes context from the live set.
  static void Remove(CUcontext context) {
    CHECK(context != nullptr);
    absl::MutexLock lock(&mu_);
    auto it = Live()->find(context);
    CHECK(it != Live()->end()) << context;
    Live()->erase(it);
  }

  // Return the context associated to that device id.
  static CUcontext GetContext(int device_ordinal) {
    absl::ReaderMutexLock lock(&mu_);
    CHECK(LiveOrdinal()->count(device_ordinal) == 1); // TODO
    return (*LiveOrdinal())[device_ordinal];
  }

  // Return the context associated to that ptr.
  static CUcontext GetContext(void* ptr) {
    absl::ReaderMutexLock lock(&mu_);
    int device_ordinal;
    CUresult result =
        cuPointerGetAttribute((void*)&device_ordinal,
                              CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                              (CUdeviceptr) ptr);
    if (result != CUDA_SUCCESS) {
      LOG(FATAL) << "Not able to get the device_ordinal for ptr: " << ptr;
    }
    return (*LiveOrdinal())[(int)device_ordinal];
  }

 private:
  // Returns the live map singleton.
  static std::map<CUcontext, std::unique_ptr<GpuContext>>* Live() {
    static auto singleton =
        new std::map<CUcontext, std::unique_ptr<GpuContext>>;
    return singleton;
  }
  static std::map<int, CUcontext>* LiveOrdinal() {
    static auto singleton =
        new std::map<int, CUcontext>;
    return singleton;
  }

  // Lock that guards access-to/mutation-of the live set.
  static absl::Mutex mu_;
  static int64 next_id_;
};
}  // namespace gpu

namespace cuda {

using MemorySpace = gpu::MemorySpace;

using CUDADriver = gpu::GpuDriver;

using ScopedActivateContext = gpu::ScopedActivateContext;

using CudaContext = gpu::GpuContext;

// Returns the current context set in CUDA. This is done by calling the cuda
// driver (e.g., this value is not our cached view of the current context).
CUcontext CurrentContextOrDie();

}  // namespace cuda
}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_CUDA_CUDA_DRIVER_H_
