/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_COMPILER_JIT_XLA_HOST_SEND_DEVICE_CONTEXT_H_
#define TENSORFLOW_COMPILER_JIT_XLA_HOST_SEND_DEVICE_CONTEXT_H_

#include "xla/shape.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/stream.h"
#include "tensorflow/core/framework/device_base.h"
#include "tfrt/concurrency/async_value_ref.h"  // from @tf_runtime

namespace tensorflow {

// XlaHostSendDeviceContext is a DeviceContext that is intended to be
// used to transfer from host->device using Rendezvous. It transfers the
// content of `device_memory_base` with `shape` using `stream`. Only
// `CopyCPUTensorToDevice` method is implemented. The `done_event` is marked as
// Concrete once transfer is completed.
//
// Example usage:
//
//  Device device;
//  stream_executor::Stream stream(executor);
//  Tensor cpu_tensor(host_allocator, DT_FLOAT, TensorShape({2, 2}));
//  Tensor device_tensor(device_allocator, DT_FLOAT, TensorShape({2, 2}));
//  se::DeviceMemoryBase gpu_dst{device_tensor.data(), 4 * sizeof(float)};
//  xla::Shape shape(xla::F32, {2, 2}, {}, {})
//  tsl::AsyncValueRef<std::unique_ptr<se::Event>> done_event =
//      tsl::MakeConstructedAsyncValueRef<std::unique_ptr<se::Event>>(stream.parent());
//  done_event->Init();
//
//  XlaHostSendDeviceContext device_context(&stream, &gpu_dst,
//    shape, done_event);
//  device_context.CopyCPUTensorToDeviceSync(
//    &cpu_tensor, &device, &device_tensor);

class XlaHostSendDeviceContext : public DeviceContext {
 public:
  XlaHostSendDeviceContext(
      se::Stream* stream, se::DeviceMemoryBase* device_memory_base,
      const xla::Shape& shape,
      tsl::AsyncValueRef<std::unique_ptr<se::Event>>& done_event)
      : stream_(stream),
        device_memory_base_(device_memory_base),
        shape_(shape),
        done_event_(done_event) {}

  // Copies 'cpu_tensor' to `device_memory_base_` with `shape_`.
  // `device_tensor` is unused.
  void CopyCPUTensorToDevice(const Tensor* cpu_tensor, Device* device,
                             Tensor* device_tensor, StatusCallback done,
                             bool sync_dst_compute) const override;

  void CopyDeviceTensorToCPU(const Tensor* device_tensor,
                             absl::string_view tensor_name, Device* device,
                             Tensor* cpu_tensor, StatusCallback done) override {
    done(errors::Internal("host->device copy not implemented."));
  }

  void CopyTensorInSameDevice(const Tensor* input_tensor, Device* device,
                              Tensor* output_tensor,
                              StatusCallback done) const override {
    done(errors::Internal("device->device copy not implemented."));
  }

 private:
  se::Stream* stream_;                        // Not owned.
  se::DeviceMemoryBase* device_memory_base_;  // Not owned.
  const xla::Shape shape_;
  tsl::AsyncValueRef<std::unique_ptr<se::Event>> done_event_;

  XlaHostSendDeviceContext(const XlaHostSendDeviceContext&) = delete;
  void operator=(const XlaHostSendDeviceContext&) = delete;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_XLA_HOST_SEND_DEVICE_CONTEXT_H_
