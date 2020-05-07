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
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/mutex.h"

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
  scoped_ref->Acquire(metadata->client(), device_ordinal);
  return Status::OK();
}

/*static*/ Status XRTGenericDeviceAccessor::InitScopedRef(
    OpKernelContext* ctx, ScopedRef* scoped_ref) {
  const XlaDevice::Metadata* metadata;
  TF_RETURN_IF_ERROR(XlaDevice::GetMetadata(ctx, &metadata));
  scoped_ref->Acquire(metadata->client(), metadata->device_ordinal());
  return Status::OK();
}

}  // namespace tensorflow
