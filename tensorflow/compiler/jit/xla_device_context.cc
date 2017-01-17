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
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/platform/mem.h"

namespace tensorflow {

// The contents of tensors allocated by XlaDeviceAllocator.
struct XlaGlobalData {
  mutable mutex mu;
  // May be nullptr if there is no xla::GlobalData backing this Tensor.
  std::shared_ptr<xla::GlobalData> data GUARDED_BY(mu);
};

// The allocator used for Tensors assigned to the XLA device. The allocator
// doesn't actually back Tensors with storage. Instead, each tensor contains
// a XlaGlobalData that wraps XLA-managed storage.
XlaDeviceAllocator::XlaDeviceAllocator() = default;
XlaDeviceAllocator::~XlaDeviceAllocator() = default;

string XlaDeviceAllocator::Name() { return "xla"; }

void* XlaDeviceAllocator::AllocateRaw(size_t alignment, size_t num_bytes) {
  // Regardless of the size requested, always allocate a XlaGlobalData. Respect
  // the aligment request because there is alignment checking even for Tensors
  // whose data is never accessed.
  void* p = port::AlignedMalloc(sizeof(XlaGlobalData), alignment);
  VLOG(2) << "Allocated XLA device tensor " << p;
  return new (p) XlaGlobalData();
}

void XlaDeviceAllocator::DeallocateRaw(void* ptr) {
  XlaGlobalData* global_data = reinterpret_cast<XlaGlobalData*>(ptr);
  VLOG(2) << "Deallocated XLA device tensor " << ptr;
  global_data->~XlaGlobalData();
  port::AlignedFree(ptr);
}

void XlaDeviceAllocator::GetStats(AllocatorStats* stats) { stats->Clear(); }

// Don't run any constructors or destructors for complex objects,
// since there is no backing store for the tensor to run them
// on. strings are the only complex objects currently stored in
// Tensors. If others are added, this set of overrides must be
// extended to include them.
void XlaDeviceAllocator::RunStringCtor(string* p, size_t n) {}
void XlaDeviceAllocator::RunStringDtor(string* p, size_t n) {}
void XlaDeviceAllocator::RunResourceCtor(ResourceHandle* p, size_t n) {}
void XlaDeviceAllocator::RunResourceDtor(ResourceHandle* p, size_t n) {}

static const XlaGlobalData* CastTensorToXlaGlobalData(const Tensor& tensor) {
  const XlaGlobalData* expression =
      reinterpret_cast<const XlaGlobalData*>(tensor.tensor_data().data());
  return expression;
}

static XlaGlobalData* CastTensorToXlaGlobalData(Tensor* tensor) {
  const XlaGlobalData* expression =
      reinterpret_cast<const XlaGlobalData*>(tensor->tensor_data().data());
  return const_cast<XlaGlobalData*>(expression);
}

XlaTransferManager::XlaTransferManager(xla::Client* client) : client_(client) {}

void XlaTransferManager::CopyCPUTensorToDevice(const Tensor* cpu_tensor,
                                               Device* device,
                                               Tensor* device_tensor,
                                               StatusCallback done) const {
  if (cpu_tensor->NumElements() > 0) {
    VLOG(2) << "CopyCPUTensorToDevice "
            << reinterpret_cast<const void*>(cpu_tensor->tensor_data().data())
            << " " << reinterpret_cast<const void*>(
                          device_tensor->tensor_data().data())
            << cpu_tensor->NumElements();
    xla::Literal literal;
    Status status = HostTensorToLiteral(*cpu_tensor, &literal);
    if (!status.ok()) {
      done(status);
      return;
    }
    auto gd = client_->TransferToServer(literal);
    if (!gd.ok()) {
      done(gd.status());
      return;
    }
    SetTensorGlobalData(
        std::shared_ptr<xla::GlobalData>(std::move(gd.ValueOrDie())),
        device_tensor);
  } else {
    VLOG(2) << "CopyCPUTensorToDevice empty tensor";
  }
  done(Status::OK());
}

void XlaTransferManager::CopyDeviceTensorToCPU(const Tensor* device_tensor,
                                               StringPiece tensor_name,
                                               Device* device,
                                               Tensor* cpu_tensor,
                                               StatusCallback done) {
  if (device_tensor->NumElements() > 0) {
    VLOG(2) << "CopyDeviceTensorToCPU"
            << reinterpret_cast<const void*>(
                   device_tensor->tensor_data().data())
            << " "
            << reinterpret_cast<const void*>(cpu_tensor->tensor_data().data())
            << device_tensor->NumElements();
    std::shared_ptr<xla::GlobalData> global_data =
        GetTensorGlobalData(*device_tensor);

    xla::Shape shape;
    Status status =
        TensorShapeToXLAShape(cpu_tensor->dtype(), cpu_tensor->shape(), &shape);
    if (!status.ok()) {
      done(status);
      return;
    }
    auto result = client_->Transfer(*global_data, &shape);
    if (!result.ok()) {
      done(result.status());
      return;
    }
    const void* src_ptr = xla::LiteralUtil::InternalData(*result.ValueOrDie());
    void* dst_ptr = DMAHelper::base(cpu_tensor);
    size_t total_bytes = cpu_tensor->TotalBytes();
    memcpy(dst_ptr, src_ptr, total_bytes);
  } else {
    VLOG(2) << "CopyDeviceTensorToCPU empty tensor";
  }
  done(Status::OK());
}

std::shared_ptr<xla::GlobalData> XlaTransferManager::GetTensorGlobalData(
    const Tensor& tensor) {
  const XlaGlobalData* data = CastTensorToXlaGlobalData(tensor);
  mutex_lock lock(data->mu);
  CHECK(data->data);
  return data->data;
}

void XlaTransferManager::SetTensorGlobalData(
    std::shared_ptr<xla::GlobalData> global_data, Tensor* tensor) {
  XlaGlobalData* data = CastTensorToXlaGlobalData(tensor);
  mutex_lock lock(data->mu);
  data->data = std::move(global_data);
}

XlaDeviceContext::XlaDeviceContext(xla::Client* client) : manager_(client) {}

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
