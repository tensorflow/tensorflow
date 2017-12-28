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

#ifndef TENSORFLOW_COMPILER_JIT_XLA_DEVICE_CONTEXT_H_
#define TENSORFLOW_COMPILER_JIT_XLA_DEVICE_CONTEXT_H_

#include <memory>

#include "tensorflow/compiler/xla/client/global_data.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

// The allocator used for Tensors assigned to the XLA device. It uses
// XLA backend's allocator.
class XlaDeviceAllocator : public Allocator {
 public:
  XlaDeviceAllocator(const xla::Backend* backend, int device_ordinal);
  ~XlaDeviceAllocator() override;

  string Name() override;

  void* AllocateRaw(size_t alignment, size_t num_bytes) override;
  void DeallocateRaw(void* ptr) override;
  void GetStats(AllocatorStats* stats) override;

 private:
  // Which backend in the client this allocator belongs to.
  const xla::Backend* backend_;
  // Which hardware device in the client's backend this allocator belongs to.
  const int device_ordinal_;
};

// Helper class for managing data transfers between host and XLA devices.
class XlaTransferManager {
 public:
  explicit XlaTransferManager(perftools::gputools::Stream* stream);

  void CopyCPUTensorToDevice(const Tensor* cpu_tensor, Device* device,
                             Tensor* device_tensor, StatusCallback done) const;
  void CopyDeviceTensorToCPU(const Tensor* device_tensor,
                             StringPiece tensor_name, Device* device,
                             Tensor* cpu_tensor, StatusCallback done);
  perftools::gputools::Stream* stream() const { return stream_; }

 private:
  // Stream obtained from a Device, used to transfer tensors between
  // CPU and device.
  perftools::gputools::Stream* stream_;
};

// DeviceContext for operators assigned to XlaDevice devices. The
// implementation must inherit from DeviceContext but otherwise just
// wraps the methods in XlaTransferManager.
class XlaDeviceContext : public DeviceContext {
 public:
  explicit XlaDeviceContext(perftools::gputools::Stream* stream);

  void CopyCPUTensorToDevice(const Tensor* cpu_tensor, Device* device,
                             Tensor* device_tensor,
                             StatusCallback done) const override;
  void CopyDeviceTensorToCPU(const Tensor* device_tensor,
                             StringPiece tensor_name, Device* device,
                             Tensor* cpu_tensor, StatusCallback done) override;
  perftools::gputools::Stream* stream() const override {
    return manager_.stream();
  }

 private:
  XlaTransferManager manager_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_XLA_DEVICE_CONTEXT_H_
