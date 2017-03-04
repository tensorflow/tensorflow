/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#if TENSORFLOW_USE_SYCL

#include "tensorflow/core/common_runtime/sycl/sycl_device.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tensorflow/core/framework/tensor.pb_text.h"
#include "tensorflow/core/platform/tracing.h"

namespace tensorflow {

static std::unordered_set<SYCLDevice *> live_devices;
static bool first_time = true;

void ShutdownSycl() {
  for (auto device : live_devices) {
    device->EnterLameDuckMode();
  }
  live_devices.clear();
}

void SYCLDevice::RegisterDevice() {
  if (first_time) {
    first_time = false;
    atexit(ShutdownSycl);
  }
  live_devices.insert(this);
}

SYCLDevice::~SYCLDevice() {
  device_context_->Unref();
  sycl_allocator_->EnterLameDuckMode();
  if (sycl_device_) {
    sycl_device_->synchronize();
    delete sycl_device_;
  }
  if (sycl_queue_) {
    delete sycl_queue_;
  }
  live_devices.erase(this);
}

void SYCLDevice::EnterLameDuckMode() {
  sycl_allocator_->EnterLameDuckMode();
  if (sycl_device_) {
    sycl_device_->synchronize();
    delete sycl_device_;
    sycl_device_ = nullptr;
  }
  if (sycl_queue_) {
    delete sycl_queue_;
    sycl_queue_ = nullptr;
  }
}

void SYCLDevice::Compute(OpKernel *op_kernel, OpKernelContext *context) {
  assert(context);
  if (port::Tracing::IsActive()) {
    // TODO(pbar) We really need a useful identifier of the graph node.
    const uint64 id = Hash64(op_kernel->name());
    port::Tracing::ScopedActivity region(port::Tracing::EventCategory::kCompute,
                                         id);
  }
  op_kernel->Compute(context);
}

Allocator *SYCLDevice::GetAllocator(AllocatorAttributes attr) {
  if (attr.on_host())
    return cpu_allocator_;
  else
    return sycl_allocator_;
}

Status SYCLDevice::MakeTensorFromProto(const TensorProto &tensor_proto,
                                       const AllocatorAttributes alloc_attrs,
                                       Tensor *tensor) {
  Tensor parsed(tensor_proto.dtype());
  if (!parsed.FromProto(cpu_allocator_, tensor_proto)) {
    return errors::InvalidArgument("Cannot parse tensor from proto: ",
                                   tensor_proto.DebugString());
  }
  Status status;
  if (alloc_attrs.on_host()) {
    *tensor = parsed;
  } else {
    Tensor copy(GetAllocator(alloc_attrs), parsed.dtype(), parsed.shape());
    device_context_->CopyCPUTensorToDevice(
        &parsed, this, &copy, [&status](const Status &s) { status = s; });
    *tensor = copy;
  }
  return status;
}

Status SYCLDevice::FillContextMap(const Graph *graph,
                                  DeviceContextMap *device_context_map) {
  // Fill in the context map.  It is OK for this map to contain
  // duplicate DeviceContexts so long as we increment the refcount.
  device_context_map->resize(graph->num_node_ids());
  for (Node *n : graph->nodes()) {
    device_context_->Ref();
    (*device_context_map)[n->id()] = device_context_;
  }

  return Status::OK();
}

Status SYCLDevice::Sync() {
  sycl_device_->synchronize();
  if (sycl_device_->ok()) {
    return Status::OK();
  } else {
    return errors::Internal("Unknown error detected on device ", name());
  }
}

}  // namespace tensorflow

#endif  // TENSORFLOW_USE_SYCL
