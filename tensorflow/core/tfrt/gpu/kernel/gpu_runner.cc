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
#include "tensorflow/core/tfrt/gpu/kernel/gpu_runner.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "llvm/ADT/SmallVector.h"
#include "tensorflow/compiler/jit/pjrt_compile_util.h"
#include "tensorflow/compiler/jit/pjrt_tensor_buffer_util.h"
#include "tensorflow/compiler/jit/xla_compile_util.h"
#include "tensorflow/compiler/jit/xla_launch_util.h"
#include "tensorflow/compiler/jit/xla_platform_info.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/tsl/framework/device_id.h"
#include "xla/tsl/framework/device_id_manager.h"
#include "xla/tsl/framework/serving_device_selector.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/device.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/notification.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_compat_request_state.h"
#include "tensorflow/core/tfrt/common/global_state.h"
#include "tensorflow/core/tfrt/utils/fallback_tensor.h"
#include "tensorflow/core/tfrt/utils/gpu_variables_table.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/fingerprint.h"
#include "tsl/platform/protobuf.h"
#include "tsl/platform/statusor.h"
#include "tfrt/host_context/async_dispatch.h"  // from @tf_runtime
#include "tfrt/host_context/async_value_ref.h"  // from @tf_runtime
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime
#include "tfrt/host_context/kernel_registry.h"  // from @tf_runtime
#include "tfrt/support/forward_decls.h"  // from @tf_runtime

namespace tensorflow {
namespace gpu {

namespace {

// TODO(b/298478068): Consider to integrate this into
// tfd::TransferTensorToDevice().
tfrt::AsyncValueRef<tfrt_stub::FallbackTensor> TransferTensorToDevice(
    const tfrt::ExecutionContext& exec_ctx,
    const tfrt_stub::FallbackTensor& tensor, Device* gpu_device) {
  const tensorflow::Tensor& src = tensor.tensor();
  tensorflow::AllocatorAttributes attr;
  attr.set_use_pjrt_allocator(true);

  tensorflow::Tensor dst(gpu_device->GetAllocator(attr), src.dtype(),
                         src.shape());
  if (src.shape().num_elements() == 0) {
    return tfrt::MakeAvailableAsyncValueRef<tfrt_stub::FallbackTensor>(dst);
  }
  auto result =
      tfrt::MakeUnconstructedAsyncValueRef<tfrt_stub::FallbackTensor>();

  DeviceContext* pjrt_device_context =
      gpu_device->tensorflow_accelerator_device_info()->pjrt_context;
  bool enqueued = tfrt::EnqueueBlockingWork(
      exec_ctx.host(),
      [result = result.CopyRef(), gpu_device, pjrt_device_context, src,
       dst = std::move(dst)]() mutable {
        tensorflow::Notification n;
        tensorflow::Status status;
        pjrt_device_context->CopyCPUTensorToDevice(
            &src, gpu_device, &dst, [&status, &n](Status s) mutable {
              status = s;
              n.Notify();
            });
        n.WaitForNotification();
        if (!status.ok()) {
          result.SetError(absl::InternalError(status.message()));
        } else {
          result.emplace(std::move(dst));
        }
      });

  if (!enqueued) {
    return tfrt::MakeErrorAsyncValueRef(absl::InternalError(
        "Failed to enqueue blocking task to transfer tensor."));
  }
  return result;
}

tfrt::AsyncValueRef<tfrt_stub::FallbackTensor> TransferTensorFromDevice(
    const tfrt::ExecutionContext& exec_ctx,
    const tfrt_stub::FallbackTensor& tensor, Device* cpu_device,
    Device* gpu_device) {
  const tensorflow::Tensor& src = tensor.tensor();

  tensorflow::AllocatorAttributes attr;
  tensorflow::Tensor dst(cpu_device->GetAllocator(attr), src.dtype(),
                         src.shape());
  if (src.shape().num_elements() == 0) {
    return tfrt::MakeAvailableAsyncValueRef<tfrt_stub::FallbackTensor>(dst);
  }
  auto result =
      tfrt::MakeUnconstructedAsyncValueRef<tfrt_stub::FallbackTensor>();

  DeviceContext* pjrt_device_context =
      gpu_device->tensorflow_accelerator_device_info()->pjrt_context;
  bool enqueued = tfrt::EnqueueBlockingWork(
      exec_ctx.host(),
      [result = result.CopyRef(), gpu_device, pjrt_device_context, src,
       dst = std::move(dst)]() mutable {
        tensorflow::Notification n;
        tensorflow::Status status;
        pjrt_device_context->CopyDeviceTensorToCPU(
            &src, "tensor_name", gpu_device, &dst,
            [&status, &n](Status s) mutable {
              status = s;
              n.Notify();
            });
        n.WaitForNotification();
        if (!status.ok()) {
          result.SetError(absl::InternalError(status.message()));
        } else {
          result.emplace(std::move(dst));
        }
      });

  if (!enqueued) {
    return tfrt::MakeErrorAsyncValueRef(absl::InternalError(
        "Failed to enqueue blocking task to transfer tensor."));
  }
  return result;
}

absl::StatusOr<
    llvm::SmallVector<tfrt::AsyncValueRef<tfrt_stub::FallbackTensor>>>
PopulateResultsFromPjRtExecutableOutputs(
    const XlaCompiler::CompilationResult& compilation_result,
    std::vector<std::unique_ptr<xla::PjRtBuffer>>& executable_outputs,
    Device* device, int num_outputs) {
  llvm::SmallVector<tfrt::AsyncValueRef<tfrt_stub::FallbackTensor>>
      fallback_tensor_results;

  for (int i = 0; i < num_outputs; ++i) {
    const DataType& dtype = compilation_result.outputs[i].type;
    CHECK(!compilation_result.outputs[i].is_constant);  // Crash OK
    CHECK(dtype != DT_RESOURCE);                        // Crash OK

    xla::PjRtBuffer* output_buffer = executable_outputs[i].get();
    if (output_buffer->IsTuple()) {
      return absl::InvalidArgumentError(
          "Tuple PJRT buffer output is not supported.");
    }
    absl::Span<const int64_t> dims;
    std::optional<std::vector<int64_t>> logical_dims_storage;
    if (output_buffer->has_dynamic_dimensions()) {
      TF_ASSIGN_OR_RETURN(std::vector<int64_t> logical_dims,
                          output_buffer->logical_dimensions());
      logical_dims_storage.emplace(std::move(logical_dims));
      dims = *logical_dims_storage;
    } else {
      dims = output_buffer->dimensions();
    }
    TensorShape tensor_shape;
    for (int i = 0; i < dims.size(); ++i) {
      TF_RETURN_IF_ERROR(tensor_shape.AddDimWithStatus(dims[i]));
    }

    TF_ASSIGN_OR_RETURN(
        Tensor output_tensor,
        MakeTensorFromPjRtBuffer(dtype, tensor_shape,
                                 std::move(executable_outputs[i])));
    auto result = tfrt::MakeAvailableAsyncValueRef<tfrt_stub::FallbackTensor>(
        output_tensor);
    fallback_tensor_results.emplace_back(std::move(result));
  }
  return fallback_tensor_results;
}

absl::StatusOr<
    llvm::SmallVector<tfrt::AsyncValueRef<tfrt_stub::FallbackTensor>>>
TransferOutputsToHostIfNeeded(
    llvm::SmallVector<tfrt::AsyncValueRef<tfrt_stub::FallbackTensor>> outputs,
    tfrt::ArrayRef<int64_t> used_output_indices, Device* cpu_device,
    Device* gpu_device, const tfrt::ExecutionContext& exec_ctx) {
  llvm::SmallVector<tfrt::AsyncValueRef<tfrt_stub::FallbackTensor>> results;
  for (int i = 0, j = 0; i < outputs.size(); ++i) {
    if (j < used_output_indices.size() && i == used_output_indices[j]) {
      CHECK(outputs[i].IsAvailable());  // Crash OK
      tfrt::AsyncValueRef<tfrt_stub::FallbackTensor> output_on_cpu =
          TransferTensorFromDevice(exec_ctx, outputs[i].get(), cpu_device,
                                   gpu_device);
      results.push_back(std::move(output_on_cpu));
      ++j;
    } else {
      results.push_back(std::move(outputs[i]));
    }
  }
  return results;
}

absl::StatusOr<
    llvm::SmallVector<tfrt::AsyncValueRef<tfrt_stub::FallbackTensor>>>
TransferVariablesAndInputs(
    int device_idx, const llvm::SmallVector<tfrt_stub::FallbackTensor>& args,
    tfrt::ArrayRef<int64_t> resource_indices, Device* cpu_device,
    absl::flat_hash_map<int, Device*> gpu_devices,
    tfrt::gpu::GpuVariablesTable& vars_table,
    const tfrt::ExecutionContext& exec_ctx) {
  llvm::SmallVector<tfrt::AsyncValueRef<tfrt_stub::FallbackTensor>> results;

  // Find all devices that are on the same platform (physical GPU). Variables
  // will be distributed to the memory of virtual devices on the same GPU.
  tsl::PlatformDeviceId platform_device_id;
  DeviceType device_type(DEVICE_GPU);
  TF_RETURN_IF_ERROR(tsl::DeviceIdManager::TfToPlatformDeviceId(
      device_type, tsl::TfDeviceId(device_idx), &platform_device_id));
  TF_ASSIGN_OR_RETURN(const std::vector<tsl::TfDeviceId> devices_on_platform,
                      tsl::DeviceIdManager::GetTfDevicesOnPlatform(
                          device_type, platform_device_id));
  const int platform_idx = platform_device_id.value();
  absl::flat_hash_set<int64_t> resource_indices_set(resource_indices.begin(),
                                                    resource_indices.end());

  for (int i = 0, resource_idx = 0; i < args.size(); ++i) {
    if (resource_indices_set.contains(i)) {
      // Transfer resources.
      tfrt::AsyncValueRef<tfrt_stub::FallbackTensor> device_tensor;
      auto cached_device_variable =
          vars_table.GetDeviceVariable(args[i], platform_idx);
      if (cached_device_variable) {
        VLOG(2) << "Cache hit for resource arg[" << i << "]";
        device_tensor = cached_device_variable.CopyRef();
      } else {
        VLOG(2) << "Cache miss for resource arg[" << i << "]";
        // Distribute variables on virtual devices on the same GPU.
        const int idx = resource_idx % devices_on_platform.size();
        const int gpu_device_idx = devices_on_platform[idx].value();
        device_tensor = TransferTensorToDevice(exec_ctx, args[i],
                                               gpu_devices.at(gpu_device_idx));
        vars_table.AddOrUpdateDeviceVariable(args[i], platform_idx,
                                             std::move(device_tensor));
        device_tensor =
            vars_table.GetDeviceVariable(args[i], platform_idx).CopyRef();
      }
      results.push_back(device_tensor);
      ++resource_idx;
    } else {
      // Transfer inputs.
      tfrt::AsyncValueRef<tfrt_stub::FallbackTensor> device_tensor =
          TransferTensorToDevice(exec_ctx, args[i], gpu_devices.at(device_idx));
      results.push_back(device_tensor);
    }
  }
  return results;
}

absl::StatusOr<uint64_t> GenerateFingerprint(
    const std::string& function_name,
    const tfd::KernelFallbackCompatRequestState* fallback_request_state) {
  const FunctionLibraryDefinition* flib_def =
      fallback_request_state->cpu_function_library_runtime()
          ->GetFunctionLibraryDefinition();
  const FunctionDef* fdef = flib_def->Find(function_name);
  if (!fdef) {
    return absl::InternalError(
        absl::StrCat("Failed to find the function ", function_name));
  }
  return tsl::Fingerprint64(
      absl::StrCat(fallback_request_state->session_metadata().name(),
                   fallback_request_state->session_metadata().version(),
                   tsl::LegacyUnredactedDebugString(fdef->signature())));
}

std::vector<XlaCompiler::Argument> BuildXlaCompilerArguments(
    const llvm::SmallVector<tfrt_stub::FallbackTensor>& inputs) {
  std::vector<XlaCompiler::Argument> out;
  out.resize(inputs.size());

  for (int input_num = 0; input_num < inputs.size(); ++input_num) {
    const tensorflow::Tensor& input = inputs[input_num].tensor();
    CHECK_GT(input.NumElements(), 0);     // Crash OK
    CHECK(input.dtype() != DT_RESOURCE);  // Crash OK

    XlaCompiler::Argument& arg = out[input_num];
    arg.kind = XlaCompiler::Argument::kParameter;
    arg.type = input.dtype();
    arg.shape = input.shape();
  }
  return out;
}

Status CompileProgram(const GpuRunInputs& run_inputs, int device_idx,
                      const XlaCompiler::CompilationResult** compilation_result,
                      xla::PjRtClient** pjrt_client,
                      xla::PjRtLoadedExecutable** pjrt_executable) {
  std::vector<XlaCompiler::Argument> xla_compiler_args =
      BuildXlaCompilerArguments(*run_inputs.args);

  DeviceBase* device = run_inputs.gpu_devices->at(device_idx);
  FunctionLibraryRuntime* flr =
      run_inputs.fallback_request_state->process_function_library_runtime()
          .GetFLR(run_inputs.gpu_devices->at(device_idx)->name());
  XlaPlatformInfo platform_info =
      XlaPlatformInfoFromDevice(run_inputs.gpu_devices->at(device_idx));
  NameAttrList function;
  function.set_name(run_inputs.func_name);

  // We store information about the JIT-compiled XLA computation in the
  // ResourceMgr.
  ResourceMgr* rm = tfrt_global::GetTFGlobalResourceMgr();

  return CompileToPjRtLoadedExecutable(
      device, platform_info, function, xla_compiler_args,
      /*compile_mode=*/DeviceCompileMode::kStrict,
      /*has_ref_vars=*/false,
      /*may_alias_resource_update=*/false, flr, rm, compilation_result,
      pjrt_client, pjrt_executable);
}

}  // namespace

absl::StatusOr<
    llvm::SmallVector<tfrt::AsyncValueRef<tfrt_stub::FallbackTensor>>>
GpuRunner::Run(const GpuRunInputs& run_inputs) {
  // Select a device to run this input.
  TF_ASSIGN_OR_RETURN(uint64_t fingerprint,
                      GenerateFingerprint(run_inputs.func_name,
                                          run_inputs.fallback_request_state));
  tsl::DeviceReservation device_reservation =
      serving_device_selector_->ReserveDevice(absl::StrCat(fingerprint));
  const int device_idx = device_reservation.device_index();

  // Compile the program.
  const XlaCompiler::CompilationResult* compilation_result;
  xla::PjRtClient* pjrt_client;                // Not owned.
  xla::PjRtLoadedExecutable* pjrt_executable;  // Not owned.
  TF_RETURN_IF_ERROR(CompileProgram(run_inputs, device_idx, &compilation_result,
                                    &pjrt_client, &pjrt_executable));

  // Transfer variables and inputs.
  TF_ASSIGN_OR_RETURN(
      llvm::SmallVector<tfrt::AsyncValueRef<tfrt_stub::FallbackTensor>>
          transferred_args,
      TransferVariablesAndInputs(device_idx, *run_inputs.args,
                                 run_inputs.resource_indices,
                                 run_inputs.cpu_device, *run_inputs.gpu_devices,
                                 vars_table_, *run_inputs.exec_ctx));

  llvm::SmallVector<tfrt::RCReference<tfrt::AsyncValue>, 4>
      transferred_args_to_wait;
  for (const auto& arg : transferred_args) {
    if (!arg.IsAvailable()) {
      transferred_args_to_wait.push_back(arg.CopyRCRef());
    }
  }
  run_inputs.exec_ctx->host()->Await(transferred_args_to_wait);

  // Execute the program.
  std::vector<const Tensor*> inputs;
  for (const auto& arg : transferred_args) {
    if (arg.IsError()) {
      return absl::InternalError(
          absl::StrCat("Data transfer failed: ", arg.GetError().message()));
    }
    inputs.push_back(&arg->tensor());
  }

  // TODO(b/297948279): Add support for execution with collectives.
  if (compilation_result->collective_info.has_value()) {
    return absl::UnimplementedError(
        "Execution with collectives is not supported yet.");
  }

  TF_ASSIGN_OR_RETURN(
      xla::PjRtDevice * pjrt_device,
      pjrt_client->LookupAddressableDevice(xla::PjRtLocalDeviceId(device_idx)));
  TF_ASSIGN_OR_RETURN(
      std::vector<std::unique_ptr<xla::PjRtBuffer>> executable_outputs,
      RunPjRtExecutable(/*num_missing_prefix_ctx_inputs=*/0, inputs,
                        /*variable_snapshots=*/{}, /*updated_variables=*/{},
                        DeviceType(DEVICE_GPU),
                        /*use_pjrt_tensor_buffer=*/true, *compilation_result,
                        pjrt_device, pjrt_client, pjrt_executable));

  // Populate the results and transfer the results to host.
  TF_ASSIGN_OR_RETURN(
      llvm::SmallVector<tfrt::AsyncValueRef<tfrt_stub::FallbackTensor>> results,
      PopulateResultsFromPjRtExecutableOutputs(
          *compilation_result, executable_outputs,
          run_inputs.gpu_devices->at(device_idx), run_inputs.num_outputs));

  return TransferOutputsToHostIfNeeded(
      results, run_inputs.used_output_indices, run_inputs.cpu_device,
      run_inputs.gpu_devices->at(device_idx), *run_inputs.exec_ctx);
}

}  // namespace gpu
}  // namespace tensorflow
