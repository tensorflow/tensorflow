/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

// Classes for keeping track of on-device state for TPUs.

#ifndef TENSORFLOW_COMPILER_XRT_XRT_TPU_DEVICE_H_
#define TENSORFLOW_COMPILER_XRT_XRT_TPU_DEVICE_H_

#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/stream_executor/tpu/tpu_node_context.h"

namespace tensorflow {

// This accessor is used for XLA TPU. It uses the distributed TPU compilation
// cache infrastructure which it accesses via the TPU_SYSTEM resource manager.
class XRTTpuDeviceAccessor {
 public:
  static Status GetResourceManager(OpKernelContext* ctx, ResourceMgr** rm);

  class ScopedRef {
   public:
    ScopedRef() {}
    ~ScopedRef() {}

    ScopedRef(const ScopedRef&) = delete;
    ScopedRef& operator=(const ScopedRef&) = delete;

    // Returns the XLA device properties from the TpuNodeContext object
    // protected by this ScopedRef.
    xla::Backend* backend() { return node_context_->backend(); }
    int device_ordinal() { return ordinal_; }

   private:
    // XRTTpuDeviceAccessor::InitScopedRef is the only way to initialize
    // ScopedRef.
    friend class XRTTpuDeviceAccessor;

    Status Acquire(int device_ordinal);

    Status Acquire(OpKernelContext* ctx);

    std::unique_ptr<tpu::TpuNodeContext> node_context_;
    int ordinal_ = 0;
  };

  static Status InitScopedRef(OpKernelContext* ctx, int device_ordinal,
                              ScopedRef* scoped_ref);

  static Status InitScopedRef(OpKernelContext* ctx, ScopedRef* scoped_ref);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_XRT_XRT_TPU_DEVICE_H_
