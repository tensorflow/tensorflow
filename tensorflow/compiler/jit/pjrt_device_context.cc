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

#include "tensorflow/compiler/jit/pjrt_device_context.h"

#include <memory>
#include <utility>

#include "tensorflow/compiler/tf2xla/literal_util.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/core/framework/device.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/tfrt/common/async_value_tensor.h"
#include "tensorflow/core/tfrt/common/create_pjrt_client_util.h"

namespace tensorflow {
namespace {

StatusOr<std::unique_ptr<xla::PjRtBuffer>> HostTensorToPjRtBuffer(
    const tensorflow::Tensor* cpu_tensor, tensorflow::Device* device,
    xla::PjRtClient* pjrt_client) {
  // TODO(b/262472386): Consider layout_preference_fn and
  // shape_representation_fn.
  xla::Shape shape;
  TF_RETURN_IF_ERROR(
      TensorShapeToXLAShape(cpu_tensor->dtype(), cpu_tensor->shape(), &shape));
  TF_ASSIGN_OR_RETURN(
      xla::PjRtDevice * pjrt_device,
      pjrt_client->LookupAddressableDevice(device->parsed_name().id));
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::PjRtBuffer> buffer,
      pjrt_client->BufferFromHostBuffer(
          cpu_tensor->data(), shape.element_type(), shape.dimensions(),
          /*byte_strides=*/std::nullopt,
          xla::PjRtClient::HostBufferSemantics::kZeroCopy,
          /*on_done_with_host_buffer=*/
          [cpu_tensor = *cpu_tensor]() { /* frees tensor */ }, pjrt_device));
  return buffer;
}

}  // namespace

void PjRtDeviceContext::CopyDeviceTensorToCPU(const Tensor* device_tensor,
                                              absl::string_view tensor_name,
                                              Device* device,
                                              Tensor* cpu_tensor,
                                              StatusCallback done) {
  profiler::TraceMe traceme("PjRtDeviceContext::CopyDeviceTensorToCPU");
  if (device_tensor->NumElements() == 0) {
    VLOG(2) << "CopyDeviceTensorToCPU empty tensor";
    done(OkStatus());
    return;
  }
  auto literal = std::make_unique<xla::MutableBorrowingLiteral>();
  auto status = tensorflow::HostTensorToMutableBorrowingLiteral(cpu_tensor,
                                                                literal.get());
  if (!status.ok()) {
    done(status);
  }
  std::shared_ptr<xla::PjRtBuffer> device_buffer =
      tensorflow::AsyncValueTensor::FromTensor(device_tensor)->GetBuffer();
  xla::PjRtFuture<Status> future = device_buffer->ToLiteral(literal.get());
  future.OnReady([literal = std::move(literal), done = std::move(done),
                  device_buffer = std::move(device_buffer)](
                     const tensorflow::Status& status) { done(status); });
}

void PjRtDeviceContext::CopyCPUTensorToDevice(const Tensor* cpu_tensor,
                                              Device* device,
                                              Tensor* device_tensor,
                                              StatusCallback done,
                                              bool sync_dst_compute) const {
  profiler::TraceMe traceme("PjRtDeviceContext::CopyCPUTensorToDevice");
  if (cpu_tensor->NumElements() == 0) {
    VLOG(2) << "CopyCPUTensorToDevice empty tensor";
    done(OkStatus());
    return;
  }
  AsyncValueTensor* result_tensor =
      tensorflow::AsyncValueTensor::FromTensor(device_tensor);
  // The result tensor should be newly allocated, which does not point to a
  // valid buffer yet.
  CHECK(!result_tensor->GetBuffer());  // Crash OK
  // TODO(b/252887149): figure out how to cache PJRT client.
  StatusOr<xla::PjRtClient*> pjrt_client =
      GetOrCreatePjRtClient(DeviceType(device->device_type()));
  if (!pjrt_client.ok()) {
    done(pjrt_client.status());
    return;
  }
  StatusOr<std::unique_ptr<xla::PjRtBuffer>> buffer_or =
      HostTensorToPjRtBuffer(cpu_tensor, device, *pjrt_client);
  if (!buffer_or.ok()) {
    done(buffer_or.status());
    return;
  }
  std::unique_ptr<xla::PjRtBuffer> device_buffer = std::move(buffer_or.value());
  // TODO(b/244666476): evaluate the performance impact of marking ready when
  // the data in device buffer is computed. In `tpu_device_context`, it is
  // marked done when the allocation finished.
  device_buffer->GetReadyFuture().OnReady(std::move(done));
  result_tensor->SetBuffer(std::move(device_buffer));
}

void PjRtDeviceContext::CopyTensorInSameDevice(const Tensor* input_tensor,
                                               Device* device,
                                               Tensor* output_tensor,
                                               StatusCallback done) const {
  done(errors::Unimplemented("Same-device copies not implemented."));
}

}  // namespace tensorflow
