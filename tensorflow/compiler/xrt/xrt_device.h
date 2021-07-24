/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

// Classes for keeping track of on-device state.

#ifndef TENSORFLOW_COMPILER_XRT_XRT_DEVICE_H_
#define TENSORFLOW_COMPILER_XRT_XRT_DEVICE_H_

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xrt/xrt_compilation_cache.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/stream_executor/tf_allocator_adapter.h"

namespace tensorflow {

// This accessor is used for XLA CPU/GPU. It uses the device resource manager,
// so e.g., on multi-GPU setups the compilation cache will not be shared across
// devices.
class XRTGenericDeviceAccessor {
 public:
  static Status GetResourceManager(OpKernelContext* ctx, ResourceMgr** rm);

  static xla::StatusOr<RefPtr<XRTCompilationCache>> GetOrCreateCompilationCache(
      OpKernelContext* ctx, int64_t max_number_of_entries);

  // We use a ScopedRef pattern here even though it's not strictly necessary,
  // just so that templated uses of this and the TPU accessor class will be as
  // similar as possible.
  class ScopedRef {
   public:
    ScopedRef() {}
    ~ScopedRef() {}

    ScopedRef(const ScopedRef&) = delete;
    ScopedRef& operator=(const ScopedRef&) = delete;

    // Returns the XLA device protected by this ScopedRef.
    xla::LocalClient* client() const { return client_; }
    xla::Backend* backend() { return client_->mutable_backend(); }
    int device_ordinal() const { return ordinal_; }
    se::DeviceMemoryAllocator* allocator() { return allocator_; }

   private:
    // XRTGenericDeviceAccessor::InitScopedRef is the only way to initialize
    // ScopedRef.
    friend class XRTGenericDeviceAccessor;

    void Acquire(xla::LocalClient* client, int ordinal,
                 const std::string& platform_name, OpKernelContext* ctx);

    xla::LocalClient* client_ = nullptr;
    int ordinal_ = 0;
    se::DeviceMemoryAllocator* allocator_ = nullptr;
    static tensorflow::mutex cuda_allocator_mutex_;
    static absl::flat_hash_map<stream_executor::Stream*,
                               std::unique_ptr<se::TfAllocatorAdapter>>*
        cuda_allocators_;
  };

  static Status InitScopedRef(OpKernelContext* ctx, int device_ordinal,
                              ScopedRef* scoped_ref);

  static Status InitScopedRef(OpKernelContext* ctx, ScopedRef* scoped_ref);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_XRT_XRT_DEVICE_H_
