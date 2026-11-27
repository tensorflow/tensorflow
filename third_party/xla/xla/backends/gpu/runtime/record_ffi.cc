/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/backends/gpu/runtime/record_ffi.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/functional/overload.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/runtime/ffi_kernel_registry.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/kernel_c_api.h"
#include "xla/ffi/api/record_c_api.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_interop.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/util.h"

namespace xla::gpu {

namespace {

se::CommandBuffer::Command* AsStreamExecutorCommand(
    const XLA_FFI_Command* command) {
  // We encode the se::CommandBuffer::Command* into the XLA_FFI_Command* C
  // object. Here we simply decode it back.
  // NOLINTNEXTLINE
  return reinterpret_cast<se::CommandBuffer::Command*>(
      // FFI API is const we need to cast away constness.
      // NOLINTNEXTLINE
      const_cast<XLA_FFI_Command*>(command));
}

const XLA_FFI_Command* AsXlaFfiCommand(
    const se::CommandBuffer::Command* command) {
  // We encode the se::CommandBuffer::Command* into the opaque XLA_FFI_Command*.
  // NOLINTNEXTLINE
  return reinterpret_cast<XLA_FFI_Command*>(
      // reinterpret_cast does not work with const.
      // NOLINTNEXTLINE
      const_cast<se::CommandBuffer::Command*>(command));
}

// Returns `ctx->dependencies` when `num_dependencies == 0`, or validates and
// converts the caller-specified `dependencies` otherwise.
absl::StatusOr<std::vector<const se::CommandBuffer::Command*>>
ResolveDependencies(const XLA_FFI_RecordContext* ctx,
                    const XLA_FFI_Command* const* dependencies,
                    uint32_t num_dependencies) {
  if (num_dependencies == 0) {
    return ctx->dependencies;
  }
  if (dependencies == nullptr) {
    return absl::InvalidArgumentError(
        "dependencies must not be null when num_dependencies > 0");
  }
  std::vector<const se::CommandBuffer::Command*> deps;
  deps.reserve(num_dependencies);
  for (uint32_t i = 0; i < num_dependencies; ++i) {
    if (dependencies[i] == nullptr) {
      return absl::InvalidArgumentError("dependency command must not be null");
    }
    deps.push_back(AsStreamExecutorCommand(dependencies[i]));
  }
  return deps;
}

struct FfiLaunchParams {
  LaunchDimensions launch_dimensions;

  std::variant<se::Kernel*, se::NativeKernel> kernel;
  uint32_t shared_mem_bytes;

  absl::string_view kernel_name() const {
    return std::visit(absl::Overload{[](se::Kernel* k) { return k->name(); },
                                     [](const se::NativeKernel& nk) {
                                       return absl::string_view(nk.name());
                                     }},
                      kernel);
  }

  bool use_pdl() const {
    return std::visit(
        absl::Overload{[](se::Kernel* k) { return k->use_pdl(); },
                       [](const se::NativeKernel& nk) { return nk.use_pdl(); }},
        kernel);
  }
};

template <typename Sink>
void AbslStringify(Sink& sink, const FfiLaunchParams& params) {
  absl::Format(&sink,
               "FfiLaunchParams(launch_dimensions=%s, kernel=%s, "
               "shared_mem_bytes=%u)",
               params.launch_dimensions.ToString(), params.kernel_name(),
               params.shared_mem_bytes);
}

class FfiKernelCache : public se::CommandBuffer::Resource {
 public:
  void SaveLaunchParams(const se::CommandBuffer::Command* cmd,
                        FfiLaunchParams params) {
    launch_params_[cmd] = std::move(params);
  }

  const FfiLaunchParams* GetLaunchParams(
      const se::CommandBuffer::Command* cmd) const {
    auto it = launch_params_.find(cmd);
    if (it != launch_params_.end()) {
      return &it->second;
    }
    return nullptr;
  }

  FfiLaunchParams* GetMutableLaunchParams(
      const se::CommandBuffer::Command* cmd) {
    auto it = launch_params_.find(cmd);
    if (it != launch_params_.end()) {
      return &it->second;
    }
    return nullptr;
  }

 private:
  absl::flat_hash_map<const se::CommandBuffer::Command*, FfiLaunchParams>
      launch_params_;
};

XLA_FFI_Error* RecordKernelLaunch(
    XLA_FFI_RecordContext* ctx,
    std::variant<se::Kernel*, se::NativeKernel> kernel_target,
    const XLA_FFI_KernelLaunchParams* launch_params,
    const XLA_FFI_Command* const* dependencies, uint32_t num_dependencies,
    const XLA_FFI_Command** out_command) {
  absl::StatusOr<std::vector<const se::CommandBuffer::Command*>> deps =
      ResolveDependencies(ctx, dependencies, num_dependencies);
  if (!deps.ok()) {
    return xla::ffi::CreateError(deps.status());
  }
  const std::vector<const se::CommandBuffer::Command*>& resolved_deps = *deps;
  const uint32_t shared_mem_bytes = launch_params->shared_mem_bytes;
  absl::StatusOr<LaunchDimensions> launch_dimensions =
      ConvertFfiLaunchDims(launch_params->launch_dims);
  if (!launch_dimensions.ok()) {
    return xla::ffi::CreateError(launch_dimensions.status());
  }
  absl::StatusOr<std::unique_ptr<se::KernelArgsPackedArray>> packed_args =
      PackKernelArgs(launch_params->args, shared_mem_bytes);
  if (!packed_args.ok()) {
    return xla::ffi::CreateError(packed_args.status());
  }

  se::CommandBuffer* cmd_buffer = ctx->command_buffer;
  auto* cache = cmd_buffer->GetOrConstructResource<FfiKernelCache>();

  FfiLaunchParams params{
      /*.launch_dimensions=*/*launch_dimensions,
      /*.kernel=*/std::move(kernel_target),
      /*.shared_mem_bytes=*/shared_mem_bytes,
  };
  XLA_VLOG_DEVICE(3, ctx->executor->device_ordinal())
      << "FfiCreateLaunch for kernel: " << params.kernel_name()
      << ", use_pdl: " << params.use_pdl()
      << ", num_dependencies passed: " << num_dependencies
      << ", resolved deps size: " << resolved_deps.size()
      << ", launch_params: " << params;

  const se::KernelArgsPackedArray& kernel_args = **packed_args;
  absl::StatusOr<const se::CommandBuffer::Command*> status_or_cmd = std::visit(
      absl::Overload{[&](se::Kernel* k) {
                       return cmd_buffer->CreateLaunch(
                           params.launch_dimensions.thread_counts_per_block(),
                           params.launch_dimensions.block_counts(),
                           params.launch_dimensions.cluster_dims(), *k,
                           kernel_args, resolved_deps);
                     },
                     [&](const se::NativeKernel& nk) {
                       return cmd_buffer->CreateLaunch(
                           params.launch_dimensions.thread_counts_per_block(),
                           params.launch_dimensions.block_counts(),
                           params.launch_dimensions.cluster_dims(), nk,
                           kernel_args, resolved_deps);
                     }},
      params.kernel);

  if (!status_or_cmd.ok()) {
    return xla::ffi::CreateError(status_or_cmd.status());
  }

  const se::CommandBuffer::Command* cmd = *status_or_cmd;
  XLA_VLOG_DEVICE(3, ctx->executor->device_ordinal())
      << "FfiCreateLaunch: created command ptr: " << cmd;
  cache->SaveLaunchParams(cmd, std::move(params));

  *out_command = AsXlaFfiCommand(cmd);
  return nullptr;
}

// Records a launch of a kernel registered through the KernelRegistry
// extension. PDL was fixed on the kernel when it was registered.
XLA_FFI_Error* CreateRegisteredKernelLaunch(
    XLA_FFI_RecordContext* ctx, const XLA_FFI_Kernel* kernel,
    const XLA_FFI_KernelLaunchParams* launch_params,
    const XLA_FFI_Command* const* dependencies, uint32_t num_dependencies,
    const XLA_FFI_Command** out_command) {
  absl::StatusOr<se::Kernel*> se_kernel =
      FindDeviceKernel(kernel, ctx->executor->device_ordinal());
  if (!se_kernel.ok()) {
    return xla::ffi::CreateError(se_kernel.status());
  }
  return RecordKernelLaunch(ctx, *se_kernel, launch_params, dependencies,
                            num_dependencies, out_command);
}

// Records a launch of a device function handle loaded by the caller.
XLA_FFI_Error* CreateFunctionLaunch(
    XLA_FFI_RecordContext* ctx, const XLA_FFI_RecordKernel* kernel,
    const XLA_FFI_KernelLaunchParams* launch_params,
    const XLA_FFI_Command* const* dependencies, uint32_t num_dependencies,
    const XLA_FFI_Command** out_command) {
  if (kernel->function_name.ptr == nullptr && kernel->function_name.len > 0) {
    return xla::ffi::CreateError(absl::InvalidArgumentError(
        "XLA_FFI_RecordKernel::function_name.ptr is null with non-zero len"));
  }
  const absl::string_view name(kernel->function_name.ptr,
                               kernel->function_name.len);
  const bool pdl = ctx->pdl_enabled && (launch_params->uses_pdl != 0);
  se::NativeKernel native_kernel{kernel->function, std::string(name), pdl};
  return RecordKernelLaunch(ctx, std::move(native_kernel), launch_params,
                            dependencies, num_dependencies, out_command);
}

XLA_FFI_Error* FfiCreateLaunch(XLA_FFI_RecordContext* ctx,
                               const XLA_FFI_RecordKernel* kernel,
                               const XLA_FFI_KernelLaunchParams* launch_params,
                               const XLA_FFI_Command* const* dependencies,
                               uint32_t num_dependencies,
                               const XLA_FFI_Command** out_command) {
  if (ctx == nullptr || out_command == nullptr) {
    return xla::ffi::CreateError(
        absl::InvalidArgumentError("ctx and out_command must not be null"));
  }
  if (kernel == nullptr ||
      kernel->struct_size < XLA_FFI_RecordKernel_STRUCT_SIZE) {
    return xla::ffi::CreateError(
        absl::InvalidArgumentError("Invalid XLA_FFI_RecordKernel"));
  }
  if (launch_params == nullptr ||
      launch_params->struct_size < XLA_FFI_KernelLaunchParams_STRUCT_SIZE) {
    return xla::ffi::CreateError(
        absl::InvalidArgumentError("Invalid XLA_FFI_KernelLaunchParams"));
  }
  if ((kernel->kernel != nullptr) == (kernel->function != nullptr)) {
    return xla::ffi::CreateError(absl::InvalidArgumentError(
        "Exactly one of XLA_FFI_RecordKernel::kernel and ::function must be "
        "set"));
  }
  if (kernel->kernel != nullptr) {
    return CreateRegisteredKernelLaunch(ctx, kernel->kernel, launch_params,
                                        dependencies, num_dependencies,
                                        out_command);
  }
  return CreateFunctionLaunch(ctx, kernel, launch_params, dependencies,
                              num_dependencies, out_command);
}

XLA_FFI_Error* FfiUpdateLaunch(XLA_FFI_RecordContext* ctx,
                               const XLA_FFI_Command* command,
                               const XLA_FFI_KernelArgs* args) {
  se::CommandBuffer* cmd_buffer = ctx->command_buffer;
  se::CommandBuffer::Command* cmd = AsStreamExecutorCommand(command);

  auto* cache = cmd_buffer->GetOrConstructResource<FfiKernelCache>();
  FfiLaunchParams* params = cache->GetMutableLaunchParams(cmd);
  if (params == nullptr) {
    return xla::ffi::CreateError(absl::InternalError(
        "Failed to find cached launch parameters for update"));
  }
  absl::StatusOr<std::unique_ptr<se::KernelArgsPackedArray>> packed_args =
      PackKernelArgs(args, params->shared_mem_bytes);
  if (!packed_args.ok()) {
    return xla::ffi::CreateError(packed_args.status());
  }
  const se::KernelArgsPackedArray& kernel_args = **packed_args;
  absl::Status status = std::visit(
      absl::Overload{
          [&](se::Kernel* k) {
            return cmd_buffer->UpdateLaunch(
                cmd, params->launch_dimensions.thread_counts_per_block(),
                params->launch_dimensions.block_counts(),
                params->launch_dimensions.cluster_dims(), *k, kernel_args);
          },
          [&](const se::NativeKernel& nk) {
            return cmd_buffer->UpdateLaunch(
                cmd, params->launch_dimensions.thread_counts_per_block(),
                params->launch_dimensions.block_counts(),
                params->launch_dimensions.cluster_dims(), nk, kernel_args);
          }},
      params->kernel);
  if (!status.ok()) {
    return xla::ffi::CreateError(status);
  }

  return nullptr;
}

XLA_FFI_Error* FfiCreateMemcpyD2D(XLA_FFI_RecordContext* ctx, void* dst,
                                  void* src, int64_t size,
                                  const XLA_FFI_Command* const* dependencies,
                                  uint32_t num_dependencies,
                                  const XLA_FFI_Command** out_command) {
  if (ctx == nullptr || out_command == nullptr) {
    return xla::ffi::CreateError(
        absl::InvalidArgumentError("ctx and out_command must not be null"));
  }
  absl::StatusOr<std::vector<const se::CommandBuffer::Command*>> deps =
      ResolveDependencies(ctx, dependencies, num_dependencies);
  if (!deps.ok()) {
    return xla::ffi::CreateError(deps.status());
  }

  se::CommandBuffer* cmd_buffer = ctx->command_buffer;
  se::DeviceAddressBase dst_addr(dst, size);
  se::DeviceAddressBase src_addr(src, size);
  absl::StatusOr<const se::CommandBuffer::Command*> status_or_cmd =
      cmd_buffer->CreateMemcpyD2D(&dst_addr, src_addr, size, *deps);

  if (!status_or_cmd.ok()) {
    return xla::ffi::CreateError(status_or_cmd.status());
  }

  *out_command = AsXlaFfiCommand(*status_or_cmd);
  return nullptr;
}

XLA_FFI_Error* FfiUpdateMemcpyD2D(XLA_FFI_RecordContext* ctx,
                                  const XLA_FFI_Command* command, void* dst,
                                  void* src, int64_t size) {
  se::CommandBuffer* cmd_buffer = ctx->command_buffer;
  se::CommandBuffer::Command* cmd = AsStreamExecutorCommand(command);

  se::DeviceAddressBase dst_addr(dst, size);
  se::DeviceAddressBase src_addr(src, size);
  absl::Status status =
      cmd_buffer->UpdateMemcpyD2D(cmd, &dst_addr, src_addr, size);
  if (!status.ok()) {
    return xla::ffi::CreateError(status);
  }
  return nullptr;
}

XLA_FFI_Error* FfiRequestStreamCapture(XLA_FFI_RecordContext* ctx) {
  ctx->stream_capture_requested = true;
  return nullptr;
}

XLA_FFI_Error* FfiCreateEmptyCommand(XLA_FFI_RecordContext* ctx,
                                     const XLA_FFI_Command* const* dependencies,
                                     uint32_t num_dependencies,
                                     const XLA_FFI_Command** out_command) {
  if (ctx == nullptr || out_command == nullptr) {
    return xla::ffi::CreateError(
        absl::InvalidArgumentError("ctx and out_command must not be null"));
  }
  absl::StatusOr<std::vector<const se::CommandBuffer::Command*>> deps =
      ResolveDependencies(ctx, dependencies, num_dependencies);
  if (!deps.ok()) {
    return xla::ffi::CreateError(deps.status());
  }

  se::CommandBuffer* cmd_buffer = ctx->command_buffer;
  absl::StatusOr<const se::CommandBuffer::Command*> status_or_cmd =
      cmd_buffer->CreateEmptyCmd(*deps);
  if (!status_or_cmd.ok()) {
    return xla::ffi::CreateError(status_or_cmd.status());
  }
  *out_command = AsXlaFfiCommand(*status_or_cmd);
  return nullptr;
}

}  // namespace

const XLA_FFI_RecordApi* GetXlaFfiRecordApi() {
  static constexpr XLA_FFI_RecordApi ffi_record_api = {
      /*.create_launch =*/FfiCreateLaunch,
      /*.update_launch =*/FfiUpdateLaunch,
      /*.create_memcpy_d2d =*/FfiCreateMemcpyD2D,
      /*.update_memcpy_d2d =*/FfiUpdateMemcpyD2D,
      /*.request_stream_capture =*/FfiRequestStreamCapture,
      /*.create_empty_command =*/FfiCreateEmptyCommand,
  };
  return &ffi_record_api;
}

}  // namespace xla::gpu
