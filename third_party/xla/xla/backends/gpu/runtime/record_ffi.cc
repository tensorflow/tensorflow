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

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <memory>
#include <new>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/debugging/symbolize.h"
#include "absl/hash/hash.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/record_c_api.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_interop.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/util.h"
#include "tsl/platform/mem.h"

namespace xla::gpu {
namespace {

template <int64_t kAlignment>
class AlignedMemory {
 public:
  explicit AlignedMemory(size_t size_bytes)
      : base_(tsl::port::AlignedMalloc(
            size_bytes, static_cast<std::align_val_t>(kAlignment))),
        size_bytes_(size_bytes) {}
  AlignedMemory(const AlignedMemory&) = delete;
  AlignedMemory& operator=(const AlignedMemory&) = delete;
  AlignedMemory& operator=(AlignedMemory&& other) = delete;

  // Move only type.
  AlignedMemory(AlignedMemory&& other) noexcept
      : base_(std::exchange(other.base_, nullptr)),
        size_bytes_(std::exchange(other.size_bytes_, 0)) {}

  ~AlignedMemory() {
    // Check for nullptr because a moved-from object will have base_ == nullptr.
    if (base_ != nullptr) {
      tsl::port::AlignedSizedFree(base_, size_bytes_,
                                  static_cast<std::align_val_t>(kAlignment));
    }
  }
  void* data() const { return base_; }
  size_t size_bytes() const { return size_bytes_; }

 private:
  void* base_;
  size_t size_bytes_;
};

// Custom implementation of se::KernelArgsPackedArrayBase for XLA:FFI.
class FfiKernelArgsPacked : public se::KernelArgsPackedArrayBase {
  static constexpr int64_t kAlignment = 64;

 public:
  explicit FfiKernelArgsPacked(int64_t num_args) {
    argument_addresses_.reserve(num_args);
  }

  // Arguments passed by pointer to the kernel.
  void AddDeviceAddress(void* opaque) {
    device_ptrs_.push_back(opaque);
    argument_addresses_.push_back(&device_ptrs_.back());
  }

  // Arguments passed by value to the kernel.
  void AddHostAddress(const void* data, int64_t size) {
    CHECK_GT(size, 0) << "Host value size must be positive.";
    AlignedMemory<kAlignment> aligned_memory(size);
    std::memcpy(aligned_memory.data(), data, size);
    allocated_args_.push_back(std::move(aligned_memory));
    argument_addresses_.push_back(allocated_args_.back().data());
  }

  void add_shared_bytes(int64_t bytes) { shared_memory_bytes_ += bytes; }

  absl::Span<const void* const> argument_addresses() const override {
    return argument_addresses_;
  }

  size_t number_of_arguments() const override {
    return argument_addresses_.size() + (shared_memory_bytes_ > 0);
  }

  size_t number_of_shared_bytes() const override {
    return shared_memory_bytes_;
  }

 private:
  // Dequeue to keep addresses stable.
  std::deque<void*> device_ptrs_;
  std::vector<AlignedMemory<kAlignment>> allocated_args_;
  std::vector<const void*> argument_addresses_;
  int64_t shared_memory_bytes_ = 0;
};

std::string GetSymbolName(const void* ptr) {
  char buf[512];
  if (absl::Symbolize(ptr, buf, sizeof(buf))) {
    return std::string(buf);
  }
  return "unknown";
}

absl::Status PackArgs(const XLA_FFI_KernelArgs* args,
                      FfiKernelArgsPacked& packed_args) {
  if (args == nullptr || (args->num_args > 0 && args->args == nullptr)) {
    return absl::InvalidArgumentError("Invalid XLA_FFI_KernelArgs struct");
  }
  for (int64_t i = 0; i < args->num_args; ++i) {
    if (args->args[i].type == XLA_FFI_KernelArgType_HostValue) {
      if (args->args[i].size <= 0) {
        return absl::InvalidArgumentError(
            "Host value args should have size > 0.");
      }
      packed_args.AddHostAddress(args->args[i].arg_address, args->args[i].size);
      continue;
    }
    if (args->args[i].type == XLA_FFI_KernelArgType_DevicePtr) {
      if (args->args[i].size != 0) {
        return absl::InvalidArgumentError(
            "Device pointer args should have size 0.");
      }
      packed_args.AddDeviceAddress(
          // FFI API is const we need to cast away constness.
          // NOLINTNEXTLINE
          const_cast<void*>(args->args[i].arg_address));
      continue;
    }
    return absl::InvalidArgumentError(absl::StrFormat(
        "Unsupported kernel argument type: %d", args->args[i].type));
  }
  return absl::OkStatus();
}

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

absl::string_view AsStringView(const void* kernel_data, int64_t kernel_size) {
  // NOLINTNEXTLINE
  return absl::string_view(reinterpret_cast<const char*>(kernel_data),
                           kernel_size);
}

absl::Span<const uint8_t> AsByteSpan(const void* kernel_data,
                                     int64_t kernel_size) {
  return absl::Span<const uint8_t>(
      // NOLINTNEXTLINE
      reinterpret_cast<const uint8_t*>(kernel_data), kernel_size);
}

struct FfiLaunchParams {
  LaunchDimensions launch_dimensions;
  std::optional<stream_executor::ClusterDim> cluster_dims;

  se::Kernel* kernel;
  uint32_t shared_mem_bytes;
};

template <typename Sink>
void AbslStringify(Sink& sink, const FfiLaunchParams& params) {
  absl::Format(&sink,
               "FfiLaunchParams(launch_dimensions=%s, cluster_dims=%s, "
               "kernel=%s, shared_mem_bytes=%u)",
               params.launch_dimensions.ToString(),
               params.cluster_dims.value_or(se::ClusterDim{0, 0, 0}).ToString(),
               GetSymbolName(params.kernel), params.shared_mem_bytes);
}

class FfiKernelCache : public se::CommandBuffer::Resource {
 private:
  // Lightweight comparator that allows using string_view for lookup.
  struct KernelKey {
    std::string name;
    const void* data;
    struct Hash {
      // https://abseil.io/tips/144
      using is_transparent = void;
      size_t operator()(const KernelKey& k) const {
        return absl::HashOf(k.name, k.data);
      }
      // Lookups can happen using cheap types.
      using Key = std::pair<absl::string_view, const void*>;
      size_t operator()(const Key& k) const {
        return absl::HashOf(k.first, k.second);
      }
    };
    struct Eq {
      using is_transparent = void;
      bool operator()(const KernelKey& a, const KernelKey& b) const {
        return a.name == b.name && a.data == b.data;
      }
      bool operator()(
          const KernelKey& a,
          const std::pair<absl::string_view, const void*>& b) const {
        return a.name == b.first && a.data == b.second;
      }
    };
  };

 public:
  absl::StatusOr<se::Kernel*> GetOrCreateKernel(
      se::StreamExecutor* executor, absl::string_view kernel_name_view,
      const void* kernel_data, int64_t kernel_size, XLA_FFI_SourceFormat format,
      int64_t num_args) {
    auto key = std::make_pair(kernel_name_view, kernel_data);
    auto it = kernels_.find(key);
    const int32_t device_ordinal = executor->device_ordinal();
    if (it != kernels_.end()) {
      XLA_VLOG_DEVICE(3, device_ordinal)
          << "FfiKernelCache: found kernel: " << kernel_name_view
          << ", ptr: " << it->second.get();
      return it->second.get();
    }
    bool is_ptx = (format == XLA_FFI_SourceFormat_PTX);
    std::string kernel_name(kernel_name_view);
    se::KernelLoaderSpec spec =
        is_ptx
            ? se::KernelLoaderSpec::CreateCudaPtxInMemorySpec(
                  AsStringView(kernel_data, kernel_size), kernel_name, num_args)
            : se::KernelLoaderSpec::CreateCudaCubinInMemorySpec(
                  AsByteSpan(kernel_data, kernel_size), kernel_name, num_args);

    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<se::Kernel> kernel,
                     executor->LoadKernel(spec));

    se::Kernel* kernel_ptr = kernel.get();
    kernels_[KernelKey{kernel_name, kernel_data}] = std::move(kernel);
    XLA_VLOG_DEVICE(3, device_ordinal)
        << "FfiKernelCache: created kernel: " << kernel_name
        << ", ptr: " << kernel_ptr;
    return kernel_ptr;
  }

  void SaveLaunchParams(const se::CommandBuffer::Command* cmd,
                        FfiLaunchParams params) {
    launch_params_[cmd] = params;
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
  absl::flat_hash_map<KernelKey, std::unique_ptr<se::Kernel>, KernelKey::Hash,
                      KernelKey::Eq>
      kernels_;
  absl::flat_hash_map<const se::CommandBuffer::Command*, FfiLaunchParams>
      launch_params_;
};

XLA_FFI_Error* FfiCreateLaunch(
    XLA_FFI_RecordContext* ctx, const char* kernel_name,
    const void* kernel_data, int64_t kernel_size, XLA_FFI_SourceFormat format,
    XLA_FFI_LaunchDims launch_dims, uint32_t shared_mem_bytes,
    const XLA_FFI_KernelArgs* args, const XLA_FFI_Command* const* dependencies,
    uint32_t num_dependencies, const XLA_FFI_Command** out_command) {
  se::CommandBuffer* cmd_buffer = ctx->command_buffer;

  auto* cache = cmd_buffer->GetOrConstructResource<FfiKernelCache>();
  auto kernel_or =
      cache->GetOrCreateKernel(ctx->executor, kernel_name, kernel_data,
                               kernel_size, format, args->num_args);
  if (!kernel_or.ok()) {
    return xla::ffi::CreateError(kernel_or.status());
  }
  se::Kernel* kernel = *kernel_or;
  kernel->set_use_pdl(ctx->use_pdl);

  FfiKernelArgsPacked packed_args(args->num_args);
  packed_args.add_shared_bytes(shared_mem_bytes);
  if (absl::Status status = PackArgs(args, packed_args); !status.ok()) {
    return xla::ffi::CreateError(status);
  }
  // Map dependencies
  std::vector<const se::CommandBuffer::Command*> deps;
  deps.reserve(num_dependencies);
  for (uint32_t i = 0; i < num_dependencies; ++i) {
    deps.push_back(AsStreamExecutorCommand(dependencies[i]));
  }

  se::ThreadDim threads(launch_dims.block.x, launch_dims.block.y,
                        launch_dims.block.z);
  se::BlockDim blocks(launch_dims.grid.x, launch_dims.grid.y,
                      launch_dims.grid.z);
  std::optional<se::ClusterDim> cluster_dims = std::nullopt;
  if (launch_dims.cluster.x != 0 || launch_dims.cluster.y != 0 ||
      launch_dims.cluster.z != 0) {
    cluster_dims = se::ClusterDim(launch_dims.cluster.x, launch_dims.cluster.y,
                                  launch_dims.cluster.z);
  }

  FfiLaunchParams params{
      /*.launch_dimensions = */ LaunchDimensions(blocks, threads),
      /*.cluster_dims =*/cluster_dims,
      /*.kernel =*/kernel,
      /*.shared_mem_bytes = */ shared_mem_bytes,
  };
  const int32_t device_ordinal = ctx->executor->device_ordinal();
  XLA_VLOG_DEVICE(3, device_ordinal)
      << "FfiCreateLaunch for kernel: " << kernel_name
      << ", use_pdl from ctx: " << ctx->use_pdl
      << ", num_dependencies passed: " << num_dependencies
      << ", resolved deps size: " << deps.size()
      << ", launch_params: " << params;

  auto status_or_cmd = cmd_buffer->CreateLaunch(threads, blocks, cluster_dims,
                                                *kernel, packed_args, deps);

  if (!status_or_cmd.ok()) {
    return xla::ffi::CreateError(status_or_cmd.status());
  }

  const se::CommandBuffer::Command* cmd = *status_or_cmd;
  XLA_VLOG_DEVICE(3, device_ordinal)
      << "FfiCreateLaunch: created command ptr: " << cmd;
  cache->SaveLaunchParams(cmd, std::move(params));

  *out_command = AsXlaFfiCommand(cmd);
  return nullptr;
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
  FfiKernelArgsPacked packed_args(args->num_args);
  packed_args.add_shared_bytes(params->shared_mem_bytes);
  if (absl::Status status = PackArgs(args, packed_args); !status.ok()) {
    return xla::ffi::CreateError(status);
  }
  if (absl::Status status = cmd_buffer->UpdateLaunch(
          cmd, params->launch_dimensions.thread_counts_per_block(),
          params->launch_dimensions.block_counts(), params->cluster_dims,
          *params->kernel, packed_args);
      !status.ok()) {
    return xla::ffi::CreateError(status);
  }

  return nullptr;
}

XLA_FFI_Error* FfiCreateMemcpyD2D(XLA_FFI_RecordContext* ctx, void* dst,
                                  void* src, int64_t size,
                                  const XLA_FFI_Command* const* dependencies,
                                  uint32_t num_dependencies,
                                  const XLA_FFI_Command** out_command) {
  se::CommandBuffer* cmd_buffer = ctx->command_buffer;

  std::vector<const se::CommandBuffer::Command*> deps;
  deps.reserve(num_dependencies);
  for (uint32_t i = 0; i < num_dependencies; ++i) {
    deps.push_back(
        reinterpret_cast<const se::CommandBuffer::Command*>(dependencies[i]));
  }

  se::DeviceAddressBase dst_addr(dst, size);
  se::DeviceAddressBase src_addr(src, size);
  auto status_or_cmd =
      cmd_buffer->CreateMemcpyD2D(&dst_addr, src_addr, size, deps);

  if (!status_or_cmd.ok()) {
    return xla::ffi::CreateError(status_or_cmd.status());
  }

  *out_command = reinterpret_cast<const XLA_FFI_Command*>(*status_or_cmd);
  return nullptr;
}

XLA_FFI_Error* FfiUpdateMemcpyD2D(XLA_FFI_RecordContext* ctx,
                                  const XLA_FFI_Command* command, void* dst,
                                  void* src, int64_t size) {
  se::CommandBuffer* cmd_buffer = ctx->command_buffer;
  auto* cmd = reinterpret_cast<const se::CommandBuffer::Command*>(command);

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
  se::CommandBuffer* cmd_buffer = ctx->command_buffer;
  std::vector<const se::CommandBuffer::Command*> deps;
  deps.reserve(num_dependencies);
  for (uint32_t i = 0; i < num_dependencies; ++i) {
    deps.push_back(AsStreamExecutorCommand(dependencies[i]));
  }
  auto status_or_cmd = cmd_buffer->CreateEmptyCmd(deps);
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
