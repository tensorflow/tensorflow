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

#include "tensorflow/compiler/xrt/xrt_tpu_device.h"

#include "tensorflow/compiler/jit/xla_device.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/tpu/tpu_configuration.h"

namespace tensorflow {

/*static*/ Status XRTTpuDeviceAccessor::GetResourceManager(OpKernelContext* ctx,
                                                           ResourceMgr** rm) {
  // ctx is unused here, but maintained because XRTGenericDeviceAccessor uses
  // it in its GetResourceManager.
  *rm = GetTPUConfigResourceMgr();
  if (*rm == nullptr) {
    return errors::Internal("No Tpu resource manager.");
  }
  return OkStatus();
}

Status XRTTpuDeviceAccessor::ScopedRef::Acquire(int device_ordinal) {
  TF_ASSIGN_OR_RETURN(node_context_,
                      tpu::TpuNodeContext::Create(device_ordinal));
  ordinal_ = device_ordinal;
  return OkStatus();
}

Status XRTTpuDeviceAccessor::ScopedRef::Acquire(OpKernelContext* ctx) {
  const XlaDevice::Metadata* metadata;
  TF_RETURN_IF_ERROR(XlaDevice::GetMetadata(ctx, &metadata));
  return Acquire(metadata->device_ordinal());
}

/*static*/ Status XRTTpuDeviceAccessor::InitScopedRef(
    OpKernelContext* /*unused ctx*/, int device_ordinal,
    ScopedRef* scoped_ref) {
  return scoped_ref->Acquire(device_ordinal);
}

/*static*/ Status XRTTpuDeviceAccessor::InitScopedRef(OpKernelContext* ctx,
                                                      ScopedRef* scoped_ref) {
  return scoped_ref->Acquire(ctx);
}

}  // namespace tensorflow
