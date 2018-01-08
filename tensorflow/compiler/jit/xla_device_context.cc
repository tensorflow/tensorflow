/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/jit/xla_device_context.h"

#include "tensorflow/compiler/tf2xla/literal_util.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/platform/mem.h"

namespace se = ::perftools::gputools;

namespace tensorflow {

// The allocator used for Tensors assigned to the XLA device.
XlaDeviceAllocator::XlaDeviceAllocator(const xla::Backend* backend,
                                       int device_ordinal)
    : backend_(backend), device_ordinal_(device_ordinal) {}

XlaDeviceAllocator::~XlaDeviceAllocator() = default;

string XlaDeviceAllocator::Name() { return "xla"; }

void* XlaDeviceAllocator::AllocateRaw(size_t alignment, size_t num_bytes) {
  se::DeviceMemoryBase dmem =
      backend_->memory_allocator()
          ->Allocate(device_ordinal_, num_bytes, /*retry_on_failure=*/false)
          .ValueOrDie();
  VLOG(2) << "Allocated XLA device tensor " << dmem.opaque() << "(" << num_bytes
          << ")";
  return dmem.opaque();
}

void XlaDeviceAllocator::DeallocateRaw(void* ptr) {
  se::DeviceMemoryBase dmem(ptr);
  TF_CHECK_OK(backend_->memory_allocator()->Deallocate(device_ordinal_, &dmem));
  VLOG(2) << "Deallocated XLA device tensor " << ptr;
}

void XlaDeviceAllocator::GetStats(AllocatorStats* stats) { stats->Clear(); }

XlaTransferManager::XlaTransferManager(se::Stream* stream) : stream_(stream) {}

void XlaTransferManager::CopyCPUTensorToDevice(const Tensor* cpu_tensor,
                                               Device* device,
                                               Tensor* device_tensor,
                                               StatusCallback done) const {
  if (cpu_tensor->NumElements() > 0) {
    VLOG(2) << "CopyCPUTensorToDevice "
            << reinterpret_cast<const void*>(cpu_tensor->tensor_data().data())
            << " "
            << reinterpret_cast<const void*>(
                   device_tensor->tensor_data().data())
            << " " << cpu_tensor->NumElements();

    void* src_ptr = const_cast<void*>(DMAHelper::base(cpu_tensor));
    const int64 total_bytes = cpu_tensor->TotalBytes();
    void* dst_ptr = DMAHelper::base(device_tensor);
    se::DeviceMemoryBase dev_dst_ptr(dst_ptr, total_bytes);

    Status status;
    stream_->ThenMemcpy(&dev_dst_ptr, src_ptr, total_bytes);
    // TODO(hpucha): Make this asynchronous.
    Status block_status = stream_->BlockHostUntilDone();
    if (!block_status.ok()) {
      status = xla::InternalError(
          "Failed to complete data transfer on stream %p: %s", stream_,
          block_status.error_message().c_str());
    }

    done(status);
    return;
  }

  VLOG(2) << "CopyCPUTensorToDevice empty tensor";
  done(Status::OK());
}

void XlaTransferManager::CopyDeviceTensorToCPU(const Tensor* device_tensor,
                                               StringPiece tensor_name,
                                               Device* device,
                                               Tensor* cpu_tensor,
                                               StatusCallback done) {
  if (device_tensor->NumElements() > 0) {
    VLOG(2) << "CopyDeviceTensorToCPU "
            << reinterpret_cast<const void*>(
                   device_tensor->tensor_data().data())
            << " "
            << reinterpret_cast<const void*>(cpu_tensor->tensor_data().data())
            << device_tensor->NumElements();

    const int64 total_bytes = cpu_tensor->TotalBytes();
    void* src_ptr = const_cast<void*>(DMAHelper::base(device_tensor));
    se::DeviceMemoryBase dev_src_ptr(src_ptr, total_bytes);
    void* dst_ptr = DMAHelper::base(cpu_tensor);

    Status status;
    stream_->ThenMemcpy(dst_ptr, dev_src_ptr, total_bytes);
    // TODO(hpucha): Make this asynchronous.
    Status block_status = stream_->BlockHostUntilDone();
    if (!block_status.ok()) {
      status = xla::InternalError(
          "Failed to complete data transfer on stream %p: %s", stream_,
          block_status.error_message().c_str());
    }

    done(status);
    return;
  }

  VLOG(2) << "CopyDeviceTensorToCPU empty tensor";
  done(Status::OK());
}

XlaDeviceContext::XlaDeviceContext(se::Stream* stream) : manager_(stream) {}

void XlaDeviceContext::CopyCPUTensorToDevice(const Tensor* cpu_tensor,
                                             Device* device,
                                             Tensor* device_tensor,
                                             StatusCallback done) const {
  manager_.CopyCPUTensorToDevice(cpu_tensor, device, device_tensor, done);
}

void XlaDeviceContext::CopyDeviceTensorToCPU(const Tensor* device_tensor,
                                             StringPiece tensor_name,
                                             Device* device, Tensor* cpu_tensor,
                                             StatusCallback done) {
  manager_.CopyDeviceTensorToCPU(device_tensor, tensor_name, device, cpu_tensor,
                                 done);
}

}  // namespace tensorflow
