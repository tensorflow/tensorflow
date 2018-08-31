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

#include "tensorflow/compiler/jit/xla_device.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

/*static*/ Status XRTGenericDeviceAccessor::GetResourceManager(
    OpKernelContext* ctx, ResourceMgr** rm) {
  *rm = ctx->resource_manager();
  return Status::OK();
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
  scoped_ref->Acquire(metadata->client());
  return Status::OK();
}

}  // namespace tensorflow
