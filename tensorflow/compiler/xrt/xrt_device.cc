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

// Classes for managing access to XLA resources.

#include "tensorflow/compiler/xrt/xrt_device.h"

#include <map>

#include "tensorflow/compiler/jit/xla_device.h"
#include "tensorflow/core/common_runtime/gpu/gpu_process_state.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace {

class ResourceMgrArena {
 public:
  static ResourceMgrArena* Get() {
    static ResourceMgrArena* arena = new ResourceMgrArena();
    return arena;
  }

  ResourceMgr* GetResourceMgr(const std::string& platform_name) {
    mutex_lock lock(mutex_);
    auto it = resource_managers_.find(platform_name);
    if (it == resource_managers_.end()) {
      it = resource_managers_.emplace(platform_name, new ResourceMgr()).first;
    }
    return it->second;
  }

 private:
  mutex mutex_;
  std::map<std::string, ResourceMgr*> resource_managers_;
};

}  // namespace

/*static*/ Status XRTGenericDeviceAccessor::GetResourceManager(
    OpKernelContext* ctx, ResourceMgr** rm) {
  const XlaDevice::Metadata* metadata;
  TF_RETURN_IF_ERROR(XlaDevice::GetMetadata(ctx, &metadata));
  *rm = ResourceMgrArena::Get()->GetResourceMgr(metadata->platform()->Name());
  return Status::OK();
}

/* static */ xla::StatusOr<RefPtr<XRTCompilationCache>>
XRTGenericDeviceAccessor::GetOrCreateCompilationCache(
    OpKernelContext* ctx, int64 max_number_of_entries) {
  ResourceMgr* rm;
  TF_RETURN_IF_ERROR(GetResourceManager(ctx, &rm));
  return tensorflow::GetOrCreateCompilationCache(rm, max_number_of_entries);
}

/*static*/ Status XRTGenericDeviceAccessor::InitScopedRef(
    OpKernelContext* ctx, int device_ordinal, ScopedRef* scoped_ref) {
  const XlaDevice::Metadata* metadata;
  TF_RETURN_IF_ERROR(XlaDevice::GetMetadata(ctx, &metadata));
  if (device_ordinal != metadata->device_ordinal()) {
    return errors::Internal("XRT device ordinal requested ", device_ordinal,
                            " on device with ordinal ",
                            metadata->device_ordinal());
  }
  scoped_ref->Acquire(metadata->client(), device_ordinal,
                      metadata->platform()->Name());
  return Status::OK();
}

/*static*/ Status XRTGenericDeviceAccessor::InitScopedRef(
    OpKernelContext* ctx, ScopedRef* scoped_ref) {
  const XlaDevice::Metadata* metadata;
  TF_RETURN_IF_ERROR(XlaDevice::GetMetadata(ctx, &metadata));
  scoped_ref->Acquire(metadata->client(), metadata->device_ordinal(),
                      metadata->platform()->Name());
  return Status::OK();
}

mutex XRTGenericDeviceAccessor::ScopedRef::mutex_;
std::map<void*, std::unique_ptr<se::TfAllocatorAdapter>>
    XRTGenericDeviceAccessor::ScopedRef::compile_cuda_allocators_;
std::map<std::string, std::unique_ptr<se::TfAllocatorAdapter>>
    XRTGenericDeviceAccessor::ScopedRef::other_cuda_allocators_;

se::DeviceMemoryAllocator*
XRTGenericDeviceAccessor::ScopedRef::GetMemoryAllocator() {
  if (platform_name_ != "CUDA") {
    return client_->mutable_backend()->memory_allocator();
  }
  if (!other_cuda_allocators_.count(platform_name_)) {
    mutex_lock lock(mutex_);
    if (!other_cuda_allocators_.count(platform_name_)) {
      se::Platform* platform =
          se::MultiPlatformManager::PlatformWithName(platform_name_)
              .ValueOrDie();
      GPUOptions gpu_options;
      Allocator* raw_allocator =
          GPUProcessState::singleton()->GetGPUAllocator(TfDeviceId(ordinal_));
      other_cuda_allocators_[platform_name_] =
          std::make_unique<se::TfAllocatorAdapter>(raw_allocator, platform,
                                                   false);
    }
  }
  return other_cuda_allocators_[platform_name_].get();
}

se::DeviceMemoryAllocator*
XRTGenericDeviceAccessor::ScopedRef::GetMemoryAllocator(OpKernelContext* ctx) {
  if (platform_name_ != "CUDA") {
    return client_->mutable_backend()->memory_allocator();
  }
  auto stream = ctx->op_device_context()->stream();
  if (!compile_cuda_allocators_.count(stream)) {
    mutex_lock lock(mutex_);
    if (!compile_cuda_allocators_.count(stream)) {
      GPUOptions gpu_options;
      Allocator* raw_allocator =
          GPUProcessState::singleton()->GetGPUAllocator(TfDeviceId(ordinal_));
      compile_cuda_allocators_[stream] =
          std::make_unique<se::TfAllocatorAdapter>(raw_allocator, stream,
                                                   false);
    }
  }
  return compile_cuda_allocators_[stream].get();
}
}  // namespace tensorflow
