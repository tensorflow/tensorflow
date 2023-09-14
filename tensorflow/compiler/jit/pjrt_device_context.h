/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_JIT_PJRT_DEVICE_CONTEXT_H_
#define TENSORFLOW_COMPILER_JIT_PJRT_DEVICE_CONTEXT_H_

#include <utility>

#include "tensorflow/compiler/tf2xla/layout_util.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {

// Helper class for managing data transfers between host and accelerator
// devices using PjRt.
class PjRtDeviceContext : public DeviceContext {
 public:
  explicit PjRtDeviceContext(
      XlaShapeLayoutHelpers::ShapeDeterminationFns shape_determination_fns,
      bool use_pjrt_tensor_buffer = false)
      : shape_determination_fns_(std::move(shape_determination_fns)),
        use_pjrt_tensor_buffer_(use_pjrt_tensor_buffer) {}

  void CopyCPUTensorToDevice(const Tensor* cpu_tensor, Device* device,
                             Tensor* device_tensor, StatusCallback done,
                             bool sync_dst_compute) const override;
  void CopyDeviceTensorToCPU(const Tensor* device_tensor,
                             absl::string_view tensor_name, Device* device,
                             Tensor* cpu_tensor, StatusCallback done) override;
  void CopyTensorInSameDevice(const Tensor* input_tensor, Device* device,
                              Tensor* output_tensor,
                              StatusCallback done) const override;

  bool use_pjrt_tensor_buffer() const { return use_pjrt_tensor_buffer_; }

 private:
  XlaShapeLayoutHelpers::ShapeDeterminationFns shape_determination_fns_;
  // Note: we currently assume the PjRtBuffer is a PjRtStreamExecutorBuffer.
  bool use_pjrt_tensor_buffer_;
};

void PjRtDeviceToDeviceCopy(DeviceContext* send_dev_context,
                            DeviceContext* recv_dev_context, Device* src,
                            Device* dst, AllocatorAttributes src_alloc_attr,
                            AllocatorAttributes dst_alloc_attr,
                            const Tensor* input, Tensor* output,
                            int dev_to_dev_stream_index, StatusCallback done);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_PJRT_DEVICE_CONTEXT_H_
