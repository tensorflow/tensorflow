/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_CUDA_CUDA_USER_OBJECT_H_
#define XLA_STREAM_EXECUTOR_CUDA_CUDA_USER_OBJECT_H_

#include <utility>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/cuda/cuda_status.h"

namespace stream_executor::gpu {

// Wraps a C++ object of type `T` and binds its lifetime to a CUDA graph using
// `CUuserObject` mechanisms. This ensures that the underlying C++ resource
// is retained as long as the CUDA graph exists,
// and gets correctly deleted when the graph is destroyed.
template <typename T>
class CudaUserObject {
 public:
  CudaUserObject(const CudaUserObject&) = delete;
  CudaUserObject& operator=(const CudaUserObject&) = delete;

  // Creates and initializes a `CudaUserObject`.
  // The object `instance` is wrapped, registered as a `CUuserObject` with CUDA,
  // and attached to the supplied `graph` to retain its reference.
  static absl::StatusOr<CudaUserObject*> Create(T&& instance, CUgraph graph) {
    auto obj_ptr = absl::WrapUnique(new CudaUserObject(std::move(instance)));

    ABSL_RETURN_IF_ERROR(cuda::ToStatus(cuUserObjectCreate(
        &obj_ptr->handle_, obj_ptr.get(),
        [](void* user_data) {
          auto* cb = static_cast<CudaUserObject<T>*>(user_data);
          delete cb;
        },
        1, CU_USER_OBJECT_NO_DESTRUCTOR_SYNC)));

    CudaUserObject* res = std::move(obj_ptr).release();
    ABSL_RETURN_IF_ERROR(res->AttachTo(graph));

    return res;
  }

  T* get() { return &instance_; }
  CUuserObject handle() { return handle_; }

 private:
  explicit CudaUserObject(T instance) : instance_(std::move(instance)) {}

  absl::Status AttachTo(CUgraph graph) {
    absl::Status res = cuda::ToStatus(
        cuGraphRetainUserObject(graph, handle_, 1, CU_GRAPH_USER_OBJECT_MOVE));
    if (!res.ok()) {
      cuUserObjectRelease(handle_, 1);
    }
    return res;
  }

  T instance_;
  CUuserObject handle_;
};

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_CUDA_CUDA_USER_OBJECT_H_
