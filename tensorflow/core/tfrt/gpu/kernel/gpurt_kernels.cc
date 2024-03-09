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
#include <cstdint>
#include <optional>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "tensorflow/core/common_runtime/copy_tensor.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_compat_request_state.h"
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_utils.h"
#include "tensorflow/core/runtime_fallback/kernel/tensor_util.h"
#include "tensorflow/core/tfrt/gpu/kernel/gpu_runner.h"
#include "tensorflow/core/tfrt/utils/fallback_tensor.h"
#include "tensorflow/core/tfrt/utils/gpu_variables_table.h"
#include "tensorflow/core/tfrt/utils/tensor_util.h"
#include "tfrt/host_context/async_dispatch.h"  // from @tf_runtime
#include "tfrt/host_context/async_value_ref.h"  // from @tf_runtime
#include "tfrt/host_context/attribute_utils.h"  // from @tf_runtime
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime
#include "tfrt/host_context/kernel_registry.h"  // from @tf_runtime
#include "tfrt/host_context/kernel_utils.h"  // from @tf_runtime
#include "tfrt/support/forward_decls.h"  // from @tf_runtime

namespace tensorflow {
namespace gpu {

namespace {
using tfrt_stub::FallbackTensor;

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
  // A map from Device ID to the Device, which should contain IDs [0, num_gpus).
  absl::flat_hash_map<int, Device*> gpu_devices;
};

// Gets CPU and GPU devices from the fallback state. Currently, we only consider
// a single GPU device.
Status GetDevices(const tfrt::ExecutionContext& exec_ctx, Devices* devices) {
  tfrt::RequestContext* req_ctx = exec_ctx.request_ctx();
  const auto* fallback_request_state =
      req_ctx->GetDataIfExists<tfd::KernelFallbackCompatRequestState>();
  if (!fallback_request_state) {
    return absl::InternalError("Fallback request state is not found.");
  }

  devices->cpu_device = fallback_request_state->device_manager().HostCPU();
  if (!devices->cpu_device) {
    return absl::InternalError(
        "Fallback request state must have a valid host cpu device.");
  }
  for (Device* device :
       fallback_request_state->device_manager().ListDevices()) {
    if (device->device_type() == DEVICE_GPU) {
      if (!devices->gpu_devices.try_emplace(device->parsed_name().id, device)
               .second) {
        return absl::InternalError(absl::StrCat(
            "A device with the same device ID already exists when adding ",
            device->name()));
      }
    }
  }
  if (devices->gpu_devices.empty()) {
    return absl::InternalError("No GPU device is found.");
  }
  for (const auto& [id, device] : devices->gpu_devices) {
    if (id >= devices->gpu_devices.size()) {
      return absl::InternalError("Device IDs are not consecutive.");
    }
  }
  return absl::OkStatus();
}

// Kernel for transferring `tensor` from host to device.
// This only supports a single GPU device.
tfrt::AsyncValueRef<tfrt_stub::FallbackTensor> TransferToDevice(
    const tfrt_stub::FallbackTensor& tensor,
    const tfrt::ExecutionContext& exec_ctx) {
  Devices devices;
  Status status = GetDevices(exec_ctx, &devices);
  if (!status.ok()) {
    return tfrt::MakeErrorAsyncValueRef(absl::InternalError(status.message()));
  }
  return TransferTensor(exec_ctx, tensor, devices.cpu_device,
                        devices.gpu_devices.at(0));
}

// Kernel for transferring `tensor` from device to host.
// This only supports a single GPU device.
tfrt::AsyncValueRef<tfrt_stub::FallbackTensor> TransferFromDevice(
    const tfrt_stub::FallbackTensor& tensor,
    const tfrt::ExecutionContext& exec_ctx) {
  Devices devices;
  Status status = GetDevices(exec_ctx, &devices);
  if (!status.ok()) {
    return tfrt::MakeErrorAsyncValueRef(absl::InternalError(status.message()));
  }
  return TransferTensor(exec_ctx, tensor, devices.gpu_devices.at(0),
                        devices.cpu_device);
}

// Kernel for transferring `variable` from host to device. If it has been
// transferred, the variable will be returned from the variable cache.
// This only supports a single GPU device.
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
                                        devices.gpu_devices.at(0));
  if (device_variable.IsError()) return device_variable;

  vars_table->AddOrUpdateDeviceVariable(variable, kCopyIndex,
                                        std::move(device_variable));
  return vars_table->GetDeviceVariable(variable, kCopyIndex).CopyRef();
}

// Kernel for `gpurt.compile_and_execute`.
// - `args` is a list of AsyncValue* tensors.
// - `func_name` is the name of the XLA function in the function library.
// - `resource_indices` are the indices of args that are variables.
// - `used_output_indices` are the indices of outputs that will be used.
void CompileAndExecute(tfrt::RemainingArguments args,
                       tfrt::RemainingResults results,
                       tfrt::StringAttribute func_name,
                       tfrt::ArrayAttribute<int64_t> resource_indices,
                       tfrt::ArrayAttribute<int64_t> used_output_indices,
                       tfrt::KernelErrorHandler error_handler,
                       const tfrt::ExecutionContext& exec_ctx) {
  // Construct the run inputs.
  llvm::SmallVector<tfrt_stub::FallbackTensor> arg_tensors;
  for (int i = 0; i < args.size(); ++i) {
    arg_tensors.push_back(args[i]->get<tfrt_stub::FallbackTensor>());
  }

  GpuRunInputs run_inputs;
  run_inputs.args = &arg_tensors;
  run_inputs.num_outputs = results.size();
  run_inputs.resource_indices = resource_indices.data();
  run_inputs.used_output_indices = used_output_indices.data();
  run_inputs.func_name = func_name.str();

  Devices devices;
  Status device_status = GetDevices(exec_ctx, &devices);
  if (!device_status.ok()) {
    error_handler.ReportError(device_status.message());
    return;
  }
  run_inputs.cpu_device = devices.cpu_device;
  run_inputs.gpu_devices = &devices.gpu_devices;

  tfrt::RequestContext* req_ctx = exec_ctx.request_ctx();
  const auto* fallback_request_state =
      req_ctx->GetDataIfExists<tfd::KernelFallbackCompatRequestState>();
  if (!fallback_request_state) {
    error_handler.ReportError("Fallback request state is not found.");
    return;
  }
  run_inputs.fallback_request_state = fallback_request_state;
  run_inputs.exec_ctx = &exec_ctx;

  // Get GpuRunner from the resource context.
  std::optional<GpuRunner*> gpu_runner =
      exec_ctx.resource_context()->GetResource<GpuRunner>(
          tensorflow::gpu::kGpuRunnerResourceName);
  if (!gpu_runner) {
    error_handler.ReportError("Missing GpuRunner in ResourceContext.");
    return;
  }

  auto fallback_tensor_results = (*gpu_runner)->Run(run_inputs);
  if (!fallback_tensor_results.ok()) {
    error_handler.ReportError(fallback_tensor_results.status().message());
    return;
  }
  for (auto it : llvm::zip(results.values(), *fallback_tensor_results)) {
    std::get<0>(it) = std::move(std::get<1>(it));
  }
}

}  // namespace

void RegisterGpurtKernels(tfrt::KernelRegistry* registry) {
  registry->AddKernel("gpurt.transfer_to_device",
                      TFRT_KERNEL(TransferToDevice));
  registry->AddKernel("gpurt.transfer_from_device",
                      TFRT_KERNEL(TransferFromDevice));
  registry->AddKernel("gpurt.maybe_transfer_variable",
                      TFRT_KERNEL(MaybeTransferVariable));
  registry->AddKernel("gpurt.compile_and_execute",
                      TFRT_KERNEL(CompileAndExecute));
}

TFRT_STATIC_KERNEL_REGISTRATION(RegisterGpurtKernels);

}  // namespace gpu
}  // namespace tensorflow
