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

#ifndef TENSORFLOW_COMMON_RUNTIME_GPU_DEVICE_CONTEXT_H_
#define TENSORFLOW_COMMON_RUNTIME_GPU_DEVICE_CONTEXT_H_

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/device_base.h"

namespace stream_executor {
class Stream;
}  // namespace stream_executor

namespace tensorflow {

// TODO(b/77980417): Replace stream_executor:: with se:: once our namespace
// migration is complete and the alias is available.

class GPUDeviceContext : public DeviceContext {
 public:
  // Does not take ownership of streams.
  GPUDeviceContext(int stream_id, stream_executor::Stream* stream,
                   stream_executor::Stream* host_to_device_stream,
                   stream_executor::Stream* device_to_host_stream,
                   stream_executor::Stream* device_to_device_stream)
      : stream_id_(stream_id),
        stream_(stream),
        host_to_device_stream_(host_to_device_stream),
        device_to_host_stream_(device_to_host_stream),
        device_to_device_stream_(device_to_device_stream) {}

  ~GPUDeviceContext() override {}

  stream_executor::Stream* stream() const override { return stream_; }
  stream_executor::Stream* host_to_device_stream() const {
    return host_to_device_stream_;
  }
  stream_executor::Stream* device_to_host_stream() const {
    return device_to_host_stream_;
  }
  stream_executor::Stream* device_to_device_stream() const {
    return device_to_device_stream_;
  }
  int stream_id() const { return stream_id_; }

  void CopyCPUTensorToDevice(const Tensor* cpu_tensor, Device* device,
                             Tensor* device_tensor,
                             StatusCallback done) const override;

  void CopyDeviceTensorToCPU(const Tensor* device_tensor, StringPiece edge_name,
                             Device* device, Tensor* cpu_tensor,
                             StatusCallback done) override;

  void MaintainLifetimeOnStream(
      const Tensor* t, perftools::gputools::Stream* stream) const override {}

 private:
  int stream_id_;
  // The default primary stream to use for this context.
  // All the memory belongs to this stream.
  stream_executor::Stream* stream_;
  // The stream to use for copy data from host into GPU.
  stream_executor::Stream* host_to_device_stream_;
  // The stream to use for copy data from GPU to host.
  stream_executor::Stream* device_to_host_stream_;
  // The stream to use for copy data between GPU.
  stream_executor::Stream* device_to_device_stream_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMMON_RUNTIME_GPU_DEVICE_CONTEXT_H_
