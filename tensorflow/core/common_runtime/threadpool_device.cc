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

#include "tensorflow/core/common_runtime/threadpool_device.h"

#include "tensorflow/core/common_runtime/local_device.h"
#include "tensorflow/core/common_runtime/scoped_allocator.h"
#include "tensorflow/core/common_runtime/scoped_allocator_mgr.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/allocator_registry.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.pb_text.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/types.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session_options.h"

#ifdef INTEL_MKL
#include "tensorflow/core/common_runtime/mkl_cpu_allocator.h"
#endif

namespace tensorflow {

ThreadPoolDevice::ThreadPoolDevice(const SessionOptions& options,
                                   const string& name, Bytes memory_limit,
                                   const DeviceLocality& locality,
                                   Allocator* allocator)
    : LocalDevice(options, Device::BuildDeviceAttributes(
                               name, DEVICE_CPU, memory_limit, locality)),
      allocator_(allocator),
      scoped_allocator_mgr_(new ScopedAllocatorMgr(name)) {}

ThreadPoolDevice::~ThreadPoolDevice() {}

void ThreadPoolDevice::Compute(OpKernel* op_kernel, OpKernelContext* context) {
  // When TraceMe profiling is off (which is the default), the
  // following TraceMe constructor is simply a conditional test of
  // false value. Measurements show that its overhead is negligible.
  port::Tracing::TraceMe trace_me(op_kernel->name(), op_kernel->type_string(),
                                  op_kernel->IsExpensive());
  if (port::Tracing::IsActive()) {
    // TODO(pbar) We really need a useful identifier of the graph node.
    const uint64 id = Hash64(op_kernel->name());
    port::Tracing::ScopedActivity region(port::Tracing::EventCategory::kCompute,
                                         id);
    op_kernel->Compute(context);
  } else {
    op_kernel->Compute(context);
  }
}

Allocator* ThreadPoolDevice::GetAllocator(AllocatorAttributes attr) {
  return allocator_;
}

Allocator* ThreadPoolDevice::GetScopedAllocator(AllocatorAttributes attr,
                                                int64 step_id) {
  if (attr.scope_id > 0) {
    return scoped_allocator_mgr_->GetContainer(step_id)->GetInstance(
        attr.scope_id);
  }
  LOG(FATAL) << "Unexpected call to ThreadPoolDevice::GetScopedAllocator "
             << "attr.scope_id = " << attr.scope_id;
  return allocator_;
}

Status ThreadPoolDevice::MakeTensorFromProto(
    const TensorProto& tensor_proto, const AllocatorAttributes alloc_attrs,
    Tensor* tensor) {
  if (tensor_proto.dtype() > 0 && tensor_proto.dtype() <= DataType_MAX) {
    Tensor parsed(tensor_proto.dtype());
    if (parsed.FromProto(cpu_allocator(), tensor_proto)) {
      *tensor = std::move(parsed);
      return Status::OK();
    }
  }
  return errors::InvalidArgument("Cannot parse tensor from proto: ",
                                 ProtoDebugString(tensor_proto));
}

#ifdef INTEL_MKL
REGISTER_MEM_ALLOCATOR("MklCPUAllocator", 200, MklCPUAllocator);
#endif

}  // namespace tensorflow
