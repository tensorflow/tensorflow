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

#include "absl/synchronization/mutex.h"
#include "tensorflow/compiler/jit/xla_tensor.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/xla/client/global_data.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

// The allocator used for Tensors assigned to the XLA device. The allocator
// ignores the alignment and size of the request and always returns a new,
// empty, XlaTensor.
class XlaDeviceAllocator : public Allocator {
 public:
  XlaDeviceAllocator();
  ~XlaDeviceAllocator() override;

  string Name() override;

  void* AllocateRaw(size_t alignment, size_t num_bytes) override;
  void DeallocateRaw(void* ptr) override;
  void GetStats(AllocatorStats* stats) override;
};

// Helper class for managing data transfers between host and XLA devices.
class XlaDeviceContext : public DeviceContext {
 public:
  explicit XlaDeviceContext(
      std::shared_ptr<se::Stream> compute_stream,
      std::shared_ptr<se::Stream> host_to_device_stream,
      std::shared_ptr<se::Stream> device_to_host_stream,
      std::vector<std::shared_ptr<se::Stream>> device_to_device_streams,
      xla::LocalClient* client,
      XlaCompiler::ShapeRepresentationFn shape_representation_fn,
      thread::ThreadPool* thread_pool);

  void CopyCPUTensorToDevice(const Tensor* cpu_tensor, Device* device,
                             Tensor* device_tensor,
                             StatusCallback done) const override;
  void CopyDeviceTensorToCPU(const Tensor* device_tensor,
                             absl::string_view tensor_name, Device* device,
                             Tensor* cpu_tensor, StatusCallback done) override;
  void CopyTensorInSameDevice(const Tensor* input_tensor, Device* device,
                              Tensor* output_tensor,
                              StatusCallback done) const override;

  xla::LocalClient* client() const { return client_; }
  se::Stream* stream() const { return stream_.get(); }
  se::Stream* host_to_device_stream() const {
    return host_to_device_stream_.get();
  }
  se::Stream* device_to_host_stream() const {
    return device_to_host_stream_.get();
  }
  se::Stream* device_to_device_stream(int index) const {
    return device_to_device_streams_.at(index).get();
  }
  xla::TransferManager* transfer_manager() const { return transfer_manager_; }
  const XlaCompiler::ShapeRepresentationFn& shape_representation_fn() const {
    return shape_representation_fn_;
  }

  // Returns a device-to-device stream, in round-robin fashion.
  se::Stream* GetDeviceToDeviceStream();

 private:
  bool UseMultipleStreams() const { return stream_ != host_to_device_stream_; }

  // The main compute stream of the device, used to synchronize the transfer
  // streams if they are set.
  std::shared_ptr<se::Stream> stream_;
  // The stream to use for transferring data from host to device. Can be
  // idential to stream_, but must not be nullptr.
  std::shared_ptr<se::Stream> host_to_device_stream_;
  // The stream to use for transferring data from device to host. Can be
  // idential to stream_, but must not be nullptr.
  std::shared_ptr<se::Stream> device_to_host_stream_;
  // Streams to use for transferring data directly between different devices,
  // e.g., over NVLINK.
  std::vector<std::shared_ptr<se::Stream>> device_to_device_streams_;

  // For the underlying memory allocator and XLA's TransferManager.
  xla::LocalClient* client_;
  // Transfer manager, for marshalling data to and from the device.
  xla::TransferManager* transfer_manager_;

  XlaCompiler::ShapeRepresentationFn shape_representation_fn_;

  // Thread pool used for running closures
  thread::ThreadPool* thread_pool_;

  absl::Mutex mu_;
  int next_stream_ GUARDED_BY(mu_) = 0;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_XLA_DEVICE_CONTEXT_H_
