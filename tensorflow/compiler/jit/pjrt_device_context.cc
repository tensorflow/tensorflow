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
#include <optional>
#include <utility>

#include "tensorflow/compiler/tf2xla/literal_util.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/core/framework/device.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/tfrt/common/async_value_tensor.h"
#include "tensorflow/core/tfrt/common/create_pjrt_client_util.h"

namespace tensorflow {
namespace {

StatusOr<std::unique_ptr<xla::PjRtBuffer>> HostTensorToPjRtBuffer(
    const tensorflow::Tensor* cpu_tensor, tensorflow::Device* device,
    xla::PjRtClient* pjrt_client,
    const XlaShapeLayoutHelpers::ShapeDeterminationFns
        shape_determination_fns) {
  XlaLayoutPreference layout_preference =
      shape_determination_fns.layout_preference_fn(
          cpu_tensor->shape(), cpu_tensor->dtype(), std::nullopt);
  TF_ASSIGN_OR_RETURN(xla::Shape shape,
                      shape_determination_fns.shape_representation_fn(
                          cpu_tensor->shape(), cpu_tensor->dtype(),
                          /*fast_mem=*/false, layout_preference));

  const xla::Layout* device_layout = &(shape.layout());
  TF_ASSIGN_OR_RETURN(
      xla::PjRtDevice * pjrt_device,
      pjrt_client->LookupAddressableDevice(device->parsed_name().id));
  auto first_try_buffer = pjrt_client->BufferFromHostBuffer(
      cpu_tensor->data(), shape.element_type(), shape.dimensions(),
      /*byte_strides=*/std::nullopt,
      xla::PjRtClient::HostBufferSemantics::kZeroCopy,
      /*on_done_with_host_buffer=*/
      [cpu_tensor = *cpu_tensor]() { /* frees tensor */ }, pjrt_device,
      device_layout);
  if (first_try_buffer.ok()) {
    return std::move(*first_try_buffer);
  }
  if (first_try_buffer.status().code() == absl::StatusCode::kUnimplemented) {
    LOG_FIRST_N(WARNING, 1)
        << first_try_buffer.status()
        << "; fallback to BufferFromHostBuffer without device layout.";
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<xla::PjRtBuffer> second_try_buffer,
        pjrt_client->BufferFromHostBuffer(
            cpu_tensor->data(), shape.element_type(), shape.dimensions(),
            /*byte_strides=*/std::nullopt,
            xla::PjRtClient::HostBufferSemantics::kZeroCopy,
            /*on_done_with_host_buffer=*/
            [cpu_tensor = *cpu_tensor]() { /* frees tensor */ }, pjrt_device));
    return second_try_buffer;
  } else {
    return first_try_buffer.status();
  }
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

void PjRtDeviceContext::CopyCPUTensorToDevice(
    const Tensor* cpu_tensor, Device* device, Tensor* device_tensor,
    StatusCallback done, bool sync_dst_compute, bool sync_dst_recv) const {
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
  StatusOr<std::unique_ptr<xla::PjRtBuffer>> buffer_or = HostTensorToPjRtBuffer(
      cpu_tensor, device, *pjrt_client, shape_determination_fns_);
  if (!buffer_or.ok()) {
    done(buffer_or.status());
    return;
  }
  result_tensor->SetBuffer(std::move(*buffer_or));
  // TODO(b/244666476): evaluate the performance impact of marking ready when
  // the data in device buffer is computed.
  result_tensor->GetBuffer()->GetReadyFuture().OnReady(std::move(done));
}

void PjRtDeviceContext::CopyTensorInSameDevice(const Tensor* input_tensor,
                                               Device* device,
                                               Tensor* output_tensor,
                                               StatusCallback done) const {
  done(errors::Unimplemented("Same-device copies not implemented."));
}

}  // namespace tensorflow
