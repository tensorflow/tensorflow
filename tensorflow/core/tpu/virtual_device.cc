/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/tpu/virtual_device.h"

#include "tensorflow/core/framework/tensor.pb.h"

namespace tensorflow {
namespace {

class VirtualDeviceContext : public DeviceContext {
 public:
  VirtualDeviceContext() = default;

  void CopyCPUTensorToDevice(const Tensor* cpu_tensor, Device* device,
                             Tensor* device_tensor, StatusCallback done,
                             bool sync_dst_compute) const override;
  void CopyDeviceTensorToCPU(const Tensor* device_tensor,
                             StringPiece tensor_name, Device* device,
                             Tensor* cpu_tensor, StatusCallback done) override;
  void CopyTensorInSameDevice(const Tensor* input_tensor, Device* device,
                              Tensor* output_tensor,
                              StatusCallback done) const override;
};

void VirtualDeviceContext::CopyCPUTensorToDevice(const Tensor* cpu_tensor,
                                                 Device* device,
                                                 Tensor* device_tensor,
                                                 StatusCallback done,
                                                 bool sync_dst_compute) const {
  *device_tensor = *cpu_tensor;
  done(absl::OkStatus());
}

void VirtualDeviceContext::CopyDeviceTensorToCPU(const Tensor* device_tensor,
                                                 StringPiece tensor_name,
                                                 Device* device,
                                                 Tensor* cpu_tensor,
                                                 StatusCallback done) {
  *cpu_tensor = *device_tensor;
  done(absl::OkStatus());
}

void VirtualDeviceContext::CopyTensorInSameDevice(const Tensor* input_tensor,
                                                  Device* device,
                                                  Tensor* output_tensor,
                                                  StatusCallback done) const {
  *output_tensor = *input_tensor;
  done(absl::OkStatus());
}

}  // namespace

// VirtualDevice

VirtualDevice::VirtualDevice(Env* env,
                             const DeviceAttributes& device_attributes)
    : Device(env, device_attributes) {}

Status VirtualDevice::Sync() { return absl::OkStatus(); }

Allocator* VirtualDevice::GetAllocator(AllocatorAttributes attr) {
  // Tensors always live on the host.
  return cpu_allocator();
}

Status VirtualDevice::MakeTensorFromProto(const TensorProto& tensor_proto,
                                          const AllocatorAttributes alloc_attrs,
                                          Tensor* tensor) {
  Tensor parsed(tensor_proto.dtype());
  Allocator* allocator = cpu_allocator();
  if (!parsed.FromProto(allocator, tensor_proto)) {
    return errors::InvalidArgument("Cannot parse tensor from proto: ",
                                   tensor_proto.DebugString());
  }
  *tensor = parsed;
  return absl::OkStatus();
}

Status VirtualDevice::TryGetDeviceContext(DeviceContext** out_context) {
  *out_context = new VirtualDeviceContext;
  return absl::OkStatus();
}

}  // namespace tensorflow
