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
#include <utility>

#include "tensorflow/core/common_runtime/copy_tensor.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_compat_request_state.h"
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_utils.h"
#include "tensorflow/core/runtime_fallback/kernel/tensor_util.h"
#include "tensorflow/core/tfrt/utils/fallback_tensor.h"
#include "tensorflow/core/tfrt/utils/gpu_variables_table.h"
#include "tensorflow/core/tfrt/utils/tensor_util.h"
#include "tfrt/host_context/async_dispatch.h"  // from @tf_runtime
#include "tfrt/host_context/async_value_ref.h"  // from @tf_runtime
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime
#include "tfrt/host_context/kernel_registry.h"  // from @tf_runtime
#include "tfrt/host_context/kernel_utils.h"  // from @tf_runtime
#include "tfrt/support/forward_decls.h"  // from @tf_runtime

namespace tensorflow {
namespace gpu {

namespace {
using tfrt_stub::FallbackTensor;

constexpr char kGpuDeviceName[] = "/device:GPU:0";

// Transfers `tensor` from `src_device` to `dst_device`.
tfrt::AsyncValueRef<FallbackTensor> TransferTensor(
    const tfrt::ExecutionContext& exec_ctx, const FallbackTensor& tensor,
    Device* src_device, Device* dst_device) {
  const tensorflow::Tensor& src = tensor.tensor();
  return tfd::TransferTensorToDevice<FallbackTensor>(exec_ctx, src, src_device,
                                                     dst_device);
}

struct Devices {
  Device* cpu_device = nullptr;
  Device* gpu_device = nullptr;
};

// Gets CPU and GPU devices from the fallback state. Currently, we only consider
// a single GPU device.
Status GetDevices(const tfrt::ExecutionContext& exec_ctx, Devices* devices) {
  tfrt::RequestContext* req_ctx = exec_ctx.request_ctx();
  const auto* fallback_request_state =
      req_ctx->GetDataIfExists<tfd::KernelFallbackCompatRequestState>();
  if (!fallback_request_state) {
    return tensorflow::errors::Internal("Fallback request state is not found.");
  }

  devices->cpu_device = fallback_request_state->device_manager().HostCPU();
  if (!devices->cpu_device) {
    return tensorflow::errors::Internal(
        "Fallback request state must have a valid host cpu device.");
  }
  TF_RETURN_IF_ERROR(fallback_request_state->device_manager().LookupDevice(
      kGpuDeviceName, &devices->gpu_device));
  return OkStatus();
}

// Kernel for transferring `tensor` from host to device.
tfrt::AsyncValueRef<tfrt_stub::FallbackTensor> TransferToDevice(
    const tfrt_stub::FallbackTensor& tensor,
    const tfrt::ExecutionContext& exec_ctx) {
  Devices devices;
  Status status = GetDevices(exec_ctx, &devices);
  if (!status.ok()) {
    return tfrt::MakeErrorAsyncValueRef(absl::InternalError(status.message()));
  }
  return TransferTensor(exec_ctx, tensor, devices.cpu_device,
                        devices.gpu_device);
}

// Kernel for transferring `tensor` from device to host.
tfrt::AsyncValueRef<tfrt_stub::FallbackTensor> TransferFromDevice(
    const tfrt_stub::FallbackTensor& tensor,
    const tfrt::ExecutionContext& exec_ctx) {
  Devices devices;
  Status status = GetDevices(exec_ctx, &devices);
  if (!status.ok()) {
    return tfrt::MakeErrorAsyncValueRef(absl::InternalError(status.message()));
  }
  return TransferTensor(exec_ctx, tensor, devices.gpu_device,
                        devices.cpu_device);
}

// Kernel for transferring `variable` from host to device. If it has been
// transferred, the variable will be returned from the variable cache.
tfrt::AsyncValueRef<tfrt_stub::FallbackTensor> MaybeTransferVariable(
    const tfrt_stub::FallbackTensor& variable,
    const tfrt::ExecutionContext& exec_ctx) {
  // For now, we only consider a single GPU device.
  const int kCopyIndex = 0;
  auto vars_table = exec_ctx.resource_context()
                        ->GetOrCreateResource<tfrt::gpu::GpuVariablesTable>(
                            tfrt::gpu::kGpuVariablesTableResourceName);
  auto cached_device_variable =
      vars_table->GetDeviceVariable(variable, kCopyIndex);
  if (cached_device_variable) {
    return cached_device_variable.CopyRef();
  }

  // The variable has not been transferred, so we transfer the variable and save
  // the device copy in the variable table.
  Devices devices;
  Status status = GetDevices(exec_ctx, &devices);
  if (!status.ok()) {
    return tfrt::MakeErrorAsyncValueRef(absl::InternalError(status.message()));
  }
  auto device_variable = TransferTensor(exec_ctx, variable, devices.cpu_device,
                                        devices.gpu_device);
  if (device_variable.IsError()) return device_variable;

  vars_table->AddOrUpdateDeviceVariable(variable, kCopyIndex,
                                        std::move(device_variable));
  return vars_table->GetDeviceVariable(variable, kCopyIndex).CopyRef();
}

}  // namespace

void RegisterGpurtKernels(tfrt::KernelRegistry* registry) {
  registry->AddKernel("gpurt.transfer_to_device",
                      TFRT_KERNEL(TransferToDevice));
  registry->AddKernel("gpurt.transfer_from_device",
                      TFRT_KERNEL(TransferFromDevice));
  registry->AddKernel("gpurt.maybe_transfer_variable",
                      TFRT_KERNEL(MaybeTransferVariable));
}

TFRT_STATIC_KERNEL_REGISTRATION(RegisterGpurtKernels);

}  // namespace gpu
}  // namespace tensorflow
