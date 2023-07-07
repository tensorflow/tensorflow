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

#include "absl/status/status.h"
#include "tensorflow/c/experimental/next_pluggable_device/tensor_pjrt_buffer_util.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/compiler/jit/pjrt_tensor_buffer_util.h"
#include "tensorflow/compiler/tf2xla/literal_util.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/core/common_runtime/next_pluggable_device/next_pluggable_device_api.h"
#include "tensorflow/core/framework/device.h"
#include "tensorflow/core/framework/device_factory.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/tfrt/common/async_value_tensor.h"
#include "tensorflow/core/tfrt/common/create_pjrt_client_util.h"
#include "tensorflow/tsl/framework/device_id_utils.h"

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
  // The device id should match the local_hardware_id in
  // tensorflow/compiler/xla/pjrt/pjrt_client.h.
  TF_ASSIGN_OR_RETURN(
      const int pjrt_device_id,
      tsl::GetDeviceIdFromDeviceParsedName(device->parsed_name(),
                                           DeviceType(device->device_type())));
  TF_ASSIGN_OR_RETURN(xla::PjRtDevice * pjrt_device,
                      pjrt_client->LookupAddressableDevice(pjrt_device_id));
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
  xla::PjRtBuffer* device_buffer =
      tensorflow::AsyncValueTensor::FromTensor(device_tensor)
          ->GetBuffer()
          .get();
  xla::PjRtFuture<Status> future = device_buffer->ToLiteral(literal.get());
  future.OnReady([literal = std::move(literal), done = std::move(done)](
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

  xla::PjRtBuffer* pjrt_buffer = (*buffer_or).get();
  if (use_pjrt_tensor_buffer_) {
    // Copy the newly created tensor with PjRtTensorBuffer to output device
    // tensor.
    //
    // We currently assume the PjRtBuffer is a PjRtStreamExecutorBuffer.
    *device_tensor = MakeTensorFromPjRtStreamExecutorBuffer(
        device_tensor->dtype(), device_tensor->shape(), std::move(*buffer_or));
  } else {
    AsyncValueTensor* result_tensor =
        tensorflow::AsyncValueTensor::FromTensor(device_tensor);
    // The result tensor should be newly allocated, which does not point to a
    // valid buffer yet.
    CHECK(!result_tensor->GetBuffer());  // Crash OK
    result_tensor->SetBuffer(std::move(*buffer_or));
  }
  // TODO(b/244666476): evaluate the performance impact of marking ready when
  // the data in device buffer is computed.
  pjrt_buffer->GetReadyFuture().OnReady(std::move(done));
}

void PjRtDeviceContext::CopyTensorInSameDevice(const Tensor* input_tensor,
                                               Device* device,
                                               Tensor* output_tensor,
                                               StatusCallback done) const {
  if (!DeviceFactory::IsPluggableDevice(device->device_type())) {
    done(absl::UnimplementedError(
        "Same-device copies in PjRtDeviceContext is only implemented when "
        "is_pluggable_device is true."));
  }
  // TODO(b/288585098): consider whether to support same device copy in PJRT
  // API.
  StatusOr<PJRT_Buffer*> c_src_buffer = GetPjRtCBufferFromTensor(input_tensor);
  if (!c_src_buffer.ok()) {
    done(c_src_buffer.status());
  }
  StatusOr<xla::PjRtCApiClient*> c_api_client = tensorflow::GetPjRtCApiClient(
      tensorflow::DeviceType(device->device_type()));
  if (!c_api_client.ok()) {
    done(c_api_client.status());
  }

  TF_StatusPtr c_status_ptr(TF_NewStatus());
  PJRT_Buffer* dst_buffer = TfnpdApi()->TFNPD_SameDevicePjRtBufferCopy(
      *c_src_buffer, (*c_api_client)->pjrt_c_client(), c_status_ptr.get());
  auto copy_c_buffer_status = StatusFromTF_Status(c_status_ptr.get());
  if (!copy_c_buffer_status.ok()) {
    done(copy_c_buffer_status);
  }

  auto set_c_buffer_status =
      SetPjRtCBufferToTensor(dst_buffer, *c_api_client, output_tensor);
  if (!set_c_buffer_status.ok()) {
    done(set_c_buffer_status);
  }
  AsyncValueTensor* result_tensor =
      tensorflow::AsyncValueTensor::FromTensor(output_tensor);
  result_tensor->GetBuffer()->GetReadyFuture().OnReady(std::move(done));
}

}  // namespace tensorflow
